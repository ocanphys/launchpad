"""One call, one worker: a logger, a lease, and a heartbeat.

A job receives the wiring finished -- it never builds a `Lease`, never names a
log file, and never calls `commit`:

    def run(self, root, worker):
        worker.log.info("counting")
        with worker.publishing(root / self.artifact.artifact_path / "count.txt") as out:
            out.write(b"3")

`worker.log` stamps every row it writes with this call's id and
`source="worker"`, from any thread, so `start_logging` files it under
`{call_id}:livedict:worker` and publishes that channel whole every
HEARTBEAT_SECONDS. What the container logged around the call and no adapter
stamped -- a library's warning -- is the same call's `ambient` source
(`system.logs`). `persist_logs` files both on the volume, so the worker never
holds its log open on the mount. A bare `print` reaches only Modal's own
capture.

A call puts two messages on the `refreshes` Queue, and the launcher recomputes
its map on each: "started", from the heartbeat's first pass and right after
the beat that pass published, so the row it was granted stops saying it is
starting; and, however the call ends, one naming that ending, after the commit
that published its files and after the last beat, which is marked `exited` so
a reader stops waiting out the flatline for a call that is already gone.

Each message trails what it announces, so a launcher sent to look finds
something new. A call that goes on to fail its lease has already said it
started: the message is a prompt to read the volume, never something the map
is computed from.
"""

import logging
import threading
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import count
from pathlib import Path
from typing import BinaryIO

import modal
from modal.exception import InputCancellation

from config import HEARTBEAT_SECONDS
from system.lease_protocol import Lease, LeaseLost, beats, refreshes
from system.logs import WORKER, call_logger, current_call_id, start_logging, try_publish


@dataclass(frozen=True)
class Worker:
    """What a job is handed.

    `progress` is a plain dict a job mutates; only the heartbeat thread reads
    it, on its own cadence, so reporting costs no network.
    """

    artifact_path: str
    call_id: str
    log: logging.LoggerAdapter
    confirm_lease: Callable[..., None]
    progress: dict

    @contextmanager
    def publishing(self, path: Path) -> Generator[BinaryIO, None, None]:
        """An open binary file to write `path`'s contents into, closed and
        renamed into place on the way out under a confirmed lease.

            with worker.publishing(self.artifact.paths(root)["tokens"]) as out:
                array("H", ids).tofile(out)

        Every file that has to appear whole or not at all is written this way,
        and a job never names a temporary path, opens one or renames one. The
        rename is what publishes a file, and `volume.commit()` is not: every
        mount runs with background commits, so bytes written to it reach the
        volume whether or not the call lives to commit, while a `.tmp` nothing
        renamed satisfies no completion. So the confirm sits between the write
        and the rename, the last instant at which a cancelled or superseded
        call can be stopped from finishing an artifact (LESSONS.md).

        Nothing this call wrote survives it not publishing: a body that raises
        and a lease lost at the confirm both delete the temporary file.

        The name is `{filename}.{call_id}.tmp` -- appended, never substituted,
        and carrying the call. Two owned files that share a stem cannot stage
        through one temporary name, and neither can two calls that overlap on
        one artifact, which is reachable for as long as a cancelled container
        takes to notice (LESSONS.md): each writes its own file and only the one
        holding the lease renames.
        """
        tmp = path.with_name(f"{path.name}.{self.call_id}.tmp")
        try:
            with tmp.open("wb") as out:  # closed before the rename, which is what makes it safe
                yield out
            self.confirm_lease(f"before publishing {path.name}")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)


@contextmanager
def initialize_worker(artifact_path: str, volume: modal.Volume):
    """Set up this call's logging and lease; on the way out, commit and tell
    the launcher.

    The log and the heartbeat exist before anything touches the mount, so
    a call that dies on the reload itself still beats once and publishes the
    error as its log. The commit is unconditional: it is what lands the
    job's files, and a call that raised may have written some -- so the
    message that follows it is unconditional too, and names which way the
    call ended.
    """
    call_id = current_call_id()
    stop_logging = start_logging(call_id)
    logger = call_logger("job", WORKER, call_id)
    lease = Lease(artifact_path, call_id, logger)

    def announce(event: str) -> None:
        """Tells the launcher this call reached `event` and the volume is
        worth reading again. Nothing is computed from it."""
        message = {"artifact_path": artifact_path, "call_id": call_id, "event": event}
        try_publish(event, partial(refreshes.put, message), logger)

    worker = Worker(
        artifact_path=artifact_path,
        call_id=call_id,
        log=logger,
        confirm_lease=lease.confirm,
        progress={},
    )

    # Set on the way out. The container outlives the call, so a thread that
    # only died with the process would keep a finished call's beat alive.
    finished = threading.Event()

    def beat(exited: bool) -> None:
        # This call is the only writer of every key it puts, so there is no
        # read-modify-write.
        try_publish(
            "beat",
            partial(beats.put, call_id, {
                "artifact_path": artifact_path,
                "last_beat_ts": time.time(),
                "progress": dict(worker.progress) or None,
                "exited": exited,
            }),
            logger,
        )

    def heartbeat() -> None:
        # The first pass beats before it waits, so the call is announced with a
        # beat already readable rather than one interval from now; the pass that
        # sees `finished` beats once more, and it is the one that marks the beat
        # `exited`, which is how a reader tells a call that has ended from one
        # whose beats are merely late.
        for passes in count(1):
            stop = finished.is_set()
            beat(exited=stop)
            if passes == 1:
                announce("started")
            if stop:
                return
            finished.wait(HEARTBEAT_SECONDS)

    # A bare thread: the call id every row is filed under rides on the record
    # `worker.log` stamps, not on anything this thread would have to inherit.
    # It beats and nothing else -- the log publishes on `start_logging`'s own
    # thread. A beat is one small put; a publish is the whole channel and grows
    # with the call, and sharing a pass would charge the beat's timeliness to
    # the log's size: a call whose publish stalled would read as flatlined.
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    ending = "failed"
    try:
        volume.reload()
        lease.confirm("boot")
        yield worker
        lease.confirm("commit")
        ending = "done"
    except BaseException as exc:
        # A cancellation is a lease lost: the launcher drops the grant before it
        # asks Modal to stop the call, so the fence usually raises first and the
        # signal Modal delivers later means the same thing.
        if isinstance(exc, (LeaseLost, InputCancellation)):
            ending = "lease lost"
            logger.error(f"{exc} -- stopping")
        else:
            logger.exception("failed under a held lease")
        # The traceback pins every frame it unwound through, and their locals
        # with them: a memmap over the volume in one of those frames keeps the
        # mapping open for as long as the runtime holds the exception, and the
        # next call on this container fails its reload (LESSONS.md).
        traceback.clear_frames(exc.__traceback__)
        raise
    else:
        logger.info("done")
    finally:
        volume.commit()
        logger.info(f"committed ({ending}); telling the launcher")
        finished.set()
        thread.join()  # the pass this waits for is the one that marks the beat
        # By now the files are committed and the beat says the call is over, so
        # the map the launcher builds on this message is the whole truth about
        # the call rather than a call still starting.
        announce(ending)
        stop_logging()  # last: its final publish carries everything above
