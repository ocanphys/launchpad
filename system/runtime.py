"""One call, one worker: a logger, a lease, and a heartbeat that carries the log.

A job receives the wiring finished -- it never builds a `Lease`, never names a
log file, and never calls `commit`:

    def run(self, root, worker):
        worker.log.info("counting")
        worker.confirm_lease("before write")   # raises LeaseLost if superseded
        (root / self.artifact.artifact_path / "count.txt").write_text("3")

The log is the root logger's records for this call, kept by the
`BufferHandler` in `system.logs` and published whole on every heartbeat as
`call_logs["{call_id}:container"]`; `persist_logs` is what files it on the
volume, so the worker never holds its log open on the mount. Anything that goes
through `logging` is in it, and anything that bypasses it (a bare `print`,
tqdm on stderr) reaches only Modal's own capture.

A call puts two messages on the `refreshes` Queue, and the launcher recomputes
its map on each: "started", once it holds the lease and has published a first
beat, so the row it was granted stops saying it is starting; and, however the
call ends, one naming that ending, after the commit that published its files
and after the last beat, which is marked `exited` so a reader stops waiting
out the flatline for a call that is already gone. Each message trails what it
announces; a launcher sent to look any earlier would find what it already had.
"""

import contextvars
import logging
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import modal

from config import HEARTBEAT_SECONDS
from system.lease_protocol import Lease, LeaseLost, beats, call_logs, refreshes
from system.logs import BufferHandler, current_call_id, setup_logging


@dataclass(frozen=True)
class Worker:
    """What a job is handed.

    `progress` is a plain dict a job mutates; only the heartbeat thread reads
    it, on its own cadence, so reporting costs no network.
    """

    artifact_path: str
    call_id: str
    log: logging.Logger
    confirm_lease: Callable[..., None]
    progress: dict


@contextmanager
def initialize_worker(artifact_path: str, volume: modal.Volume):
    """Set up this call's logging and lease; on the way out, commit and tell
    the launcher.

    The buffer and the heartbeat exist before anything touches the mount, so
    a call that dies on the reload itself still beats once and publishes the
    error as its log. The commit is unconditional: it is what lands the
    job's files, and a call that raised may have written some -- so the
    message that follows it is unconditional too, and names which way the
    call ended.
    """
    call_id = current_call_id()

    setup_logging()
    for noisy in ("modal", "grpc", "urllib3"):  # the heartbeat's own RPCs stay out of the log
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root = logging.getLogger()
    buffer = BufferHandler(call_id)
    root.addHandler(buffer)
    logger = logging.getLogger("job")
    logger.setLevel(logging.DEBUG)  # everything; filtering is the reader's job
    lease = Lease(artifact_path, call_id, logger)

    def publish(what: str, put: Callable[[], None]) -> None:
        try:
            put()
        except Exception as exc:
            logger.warning(f"{what} not published ({exc})")

    def announce(event: str) -> None:
        """Tells the launcher this call reached `event` and the volume is
        worth reading again. Nothing is computed from it."""
        publish(event, partial(refreshes.put, {"artifact_path": artifact_path, "call_id": call_id, "event": event}))

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
        publish(
            "beat",
            partial(beats.put, call_id, {
                "artifact_path": artifact_path,
                "last_beat_ts": time.time(),
                "progress": dict(worker.progress) or None,
                "exited": exited,
            }),
        )

    def heartbeat():
        # This call is the only writer of every key it puts, so there is no
        # read-modify-write. The pass that sees `finished` still publishes,
        # so the rows logged on the way out land -- and it is the one that
        # marks the beat `exited`, which is how a reader tells a call that
        # has ended from one whose beats are merely late.
        while True:
            stop = finished.wait(HEARTBEAT_SECONDS)
            beat(exited=stop)
            publish("log", lambda: call_logs.put(f"{call_id}:container", list(buffer.rows)))
            if stop:
                return

    # Under a copy of this call's context: the call id the handlers filter on
    # is a contextvar, which a bare thread would not carry.
    thread = threading.Thread(target=contextvars.copy_context().run, args=(heartbeat,), daemon=True)
    thread.start()
    ending = "failed"
    try:
        volume.reload()
        lease.confirm("boot")
        # The beat before the announcement, and both before the job runs: a
        # launcher sent to look while this call still had no heartbeat would
        # find the starting call it already knew about.
        beat(exited=False)
        announce("started")
        yield worker
        lease.confirm("commit")
        ending = "done"
    except BaseException as exc:
        if isinstance(exc, LeaseLost):
            ending = "lease lost"
            logger.error(f"{exc} -- stopping; the holder's writes will overtake ours")
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
        root.removeHandler(buffer)
        finished.set()  # after the commit: committing is still working
        thread.join()  # the pass this waits for is the one that marks the beat
        # Last of all: by now the files are committed and the beat says the
        # call is over, so the map the launcher builds on this message is the
        # whole truth about the call rather than a call still starting.
        announce(ending)
