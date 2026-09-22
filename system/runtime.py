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
"""

import contextvars
import logging
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import modal

from config import HEARTBEAT_SECONDS
from system.lease_protocol import Lease, LeaseLost, beats, call_logs
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
    """Set up this call's logging and lease; on the way out, commit.

    The buffer and the heartbeat exist before anything touches the mount, so
    a call that dies on the reload itself still beats once and publishes the
    error as its log. The commit is unconditional: it is what lands the
    job's files, and a call that raised may have written some.
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
            logger.warning(f"heartbeat: {what} not published ({exc})")

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

    def heartbeat():
        # This call is the only writer of every key it puts, so there is no
        # read-modify-write. The pass that sees `finished` still publishes,
        # so the rows logged on the way out land.
        while True:
            stop = finished.wait(HEARTBEAT_SECONDS)
            publish(
                "beat",
                lambda: beats.put(
                    call_id,
                    {
                        "artifact_path": artifact_path,
                        "last_beat_ts": time.time(),
                        "progress": dict(worker.progress) or None,
                    },
                ),
            )
            publish("log", lambda: call_logs.put(f"{call_id}:container", list(buffer.rows)))
            if stop:
                return

    # Under a copy of this call's context: the call id the handlers filter on
    # is a contextvar, which a bare thread would not carry.
    thread = threading.Thread(target=contextvars.copy_context().run, args=(heartbeat,), daemon=True)
    thread.start()
    try:
        volume.reload()
        lease.confirm("boot")
        yield worker
        lease.confirm("commit")
    except BaseException as exc:
        if isinstance(exc, LeaseLost):
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
        root.removeHandler(buffer)
        volume.commit()
        finished.set()  # after the commit: committing is still working
        thread.join()
