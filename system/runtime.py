"""One call, one worker: a logger, a lease, and a heartbeat that carries the log.

A job receives the wiring finished -- it never builds a `Lease`, never names a
log file, and never calls `commit`:

    def run(self, root, worker):
        worker.log.info("counting")
        worker.confirm_lease("before write")   # raises LeaseLost if superseded
        (root / self.artifact.artifact_path / "count.txt").write_text("3")

The log is taken from Modal's capture of the call rather than a handler, so it
holds stderr, tqdm and every library that never heard of `worker.log`.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import modal

from config import HEARTBEAT_SECONDS, LOG_FLUSH_SECONDS, STORAGE
from system.lease_protocol import Lease, LeaseLost, beats, call_logs
from system.logs import archive_path, setup_logging


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
    """Set up this call's logging and lease; on the way out, archive the log
    and commit.

    The commit is unconditional: a call that raised is the one whose log is
    worth keeping, and the log only reaches the volume when something commits.
    """
    call_id = modal.current_function_call_id() or "local"
    started = datetime.now(UTC)  # floor for the history fetch below
    volume.reload()

    setup_logging()
    # The heartbeat streams this container's stdout back in; a client that
    # narrates its own RPCs at INFO would multiply every beat.
    for noisy in ("modal", "grpc", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger("job")
    logger.setLevel(logging.DEBUG)  # everything; filtering is the reader's job
    lease = Lease(artifact_path, call_id, logger)
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

    # Every line Modal has captured of this call so far.
    lines: list[str] = []
    # Printed by the `finally`; the feed is ordered, so once the subscription
    # sees it, everything before it is in `lines`. Dropped, not kept.
    sentinel = f"--- end of log {call_id} ---"
    caught_up = threading.Event()

    def heartbeat():
        # `beats` and `call_logs` key on call_id: this is the only writer of
        # both keys, so there is no read-modify-write.
        #
        # The subscription blocks and keeps its cursor internally, so it is
        # entered once, as a task; the beat is the loop body, on its own clock.
        async def main():
            call = modal.FunctionCall.from_id(call_id)
            seen = set()
            try:
                async for entry in call.logs.fetch.aio(since=started):
                    seen.add((entry.timestamp, entry.message))
                    lines.append(entry.message.rstrip("\r\n"))
            except Exception as exc:
                logger.warning(f"heartbeat: history not fetched ({exc})")

            # Ends itself at the sentinel rather than being cancelled: cancelling
            # Modal's stream mid-wait trips its own teardown (aclose on a running
            # generator), which surfaces as unretrieved task exceptions in the
            # next call's log.
            async def follow():
                async for entry in call.logs.stream.aio():
                    line = entry.message.rstrip("\r\n")
                    if line == sentinel:
                        caught_up.set()
                        return
                    if (entry.timestamp, entry.message) not in seen:
                        lines.append(line)

            asyncio.ensure_future(follow())
            while not finished.is_set():
                await asyncio.sleep(HEARTBEAT_SECONDS)
                try:
                    await beats.put.aio(
                        call_id,
                        {
                            "artifact_path": artifact_path,
                            "last_beat_ts": time.time(),
                            "progress": dict(worker.progress) or None,
                        },
                    )
                except Exception as exc:
                    logger.warning(f"heartbeat: not recorded ({exc})")
                try:
                    await call_logs.put.aio(call_id, list(lines))
                except Exception as exc:
                    logger.warning(f"heartbeat: logs not published ({exc})")

        try:
            asyncio.run(main())
        except Exception:
            logger.exception("heartbeat thread died")

    try:
        threading.Thread(target=heartbeat, daemon=True).start()
        lease.confirm("boot")
        yield worker
        lease.confirm("commit")
    except LeaseLost as exc:
        logger.error(f"{exc} -- stopping; the holder's writes will overtake ours")
        raise
    except BaseException:
        logger.exception("failed under a held lease")
        raise
    else:
        logger.info("done")
    finally:
        print(sentinel, flush=True)
        if not caught_up.wait(LOG_FLUSH_SECONDS):
            logger.warning("log flush: the last lines did not come back in time")

        path = archive_path(Path(STORAGE), artifact_path, call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        volume.commit()
        finished.set()  # after the commit: committing is still working
