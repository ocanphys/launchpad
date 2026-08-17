"""One call, one scope: a logger, a lease, and a commit that had to be earned.

`logs.py` gives a call its own logger; `lease_protocol.py` tells a call whether it
still owns the run. Neither is useful to a job on its own, and every job would
otherwise wire them together the same way -- and get the ordering wrong in the
same two places:

    reload -> open the log file      (reload fails while a file on the volume is open)
    close the log file -> commit     (the last lines ship only if they are flushed first)

So the wiring lives here once, as a context manager, and a job receives the
finished pair. A job never constructs a `Lease`, never names a log file, and never
calls `commit` -- it writes, and it calls `scope.confirm(...)` before anything it
would not want a superseded container to have done.

    def count(scope, n):
        scope.log.info(f"counting to {n}")
        scope.confirm("before write")     # raises LeaseLost if we were superseded
        (scope.dir / "count.txt").write_text(str(n))

The commit is the point of the whole arrangement. It happens once, on the way out,
and only after a final `confirm` -- so a container that lost the run mid-job
cannot land its writes, no matter how far it got before anyone noticed.
"""
import time
import threading
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import modal

from config import STORAGE, HEARTBEAT_SECONDS
from lease_protocol import Lease, LeaseLost, beats
from logs import call_logger, release_call_logger


@dataclass(frozen=True)
class Worker:
    """What a job is handed. Everything call-specific, nothing call-specific to build.

    `confirm` is the lease's bound method rather than the lease itself: a job has
    no business releasing, re-granting, or inspecting a lease, and the narrower
    handle is what keeps that true without a rule anyone has to remember.
    """

    run_id: str
    call_id: str
    log: logging.Logger
    confirm_lease: Callable[..., None]
    dir: Path

@contextmanager
def initialize_worker(run_id: str, job_type: str, volume: modal.Volume):
    """Set up one call's logger and lease; commit on the way out, if still owed.

    `owner` names the actor, not the call -- `worker`, `etl` -- and becomes the log
    file's name inside the call's own folder, per the layout `logs.py` describes.

    Three ways out, three policies, because they are three different events:

    - clean: confirm once more, close the log, commit.
    - `LeaseLost`: no commit. Someone else owns this run, and the writes under it
      are theirs now. The log lines still reach Modal's container log through the
      stream handler, which is the only place they can honestly go.
    - any other exception: close the log and commit anyway. The job failed while
      it still owned the run, so its log -- the thing that explains the failure --
      is worth keeping, and a partial output under a held lease is not a
      correctness problem the way a superseded one is.
    """


    call_id = modal.current_function_call_id() or "local"

    # Before the log file is opened, never after: an open file on the volume
    # blocks reload, and this is the only moment nothing is open yet.
    volume.reload()

    run_dir = Path(STORAGE) / "runs" / run_id
    log_dir = run_dir / "logs" / call_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = call_logger(call_id, log_dir / f"{job_type}.log")
    lease = Lease(run_id, call_id, logger)
    worker = Worker(run_id=run_id, call_id=call_id, log=logger, confirm_lease=lease.confirm, dir=run_dir)

    def heartbeat():
        # No fence check first -- see `Lease.confirm`'s note on why. A single
        # key write, not a read-modify-write: `beats` keys on call_id directly,
        # so this call's beat lives at a key nobody else ever writes to. Two
        # containers that have both, at different times, held this run get two
        # different keys -- this beat can never land on top of another call's,
        # and no read is needed first to avoid it. A reader tells a stale beat
        # from a live one by checking whether its call_id is still the run's
        # current holder (`main.read_book` does this), not by racing to write
        # first.
        while True:
            time.sleep(HEARTBEAT_SECONDS)
            try:
                beats.put(call_id, {"run_id": run_id, "last_beat_ts": time.time()})
                logger.debug(f"{job_type}: heartbeat!")
            except Exception as exc:
                logger.warning(f"heartbeat: not recorded ({exc})")

    # A daemon thread, not a task: nothing here is async, so the only way to have
    # this run alongside the job's own code is a second thread. It dies with the
    # process on its own -- nothing to cancel on the way out.
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)

    try:
        heartbeat_thread.start()
        lease.confirm(f"{job_type}: boot")
        yield worker
        lease.confirm(f"{job_type}: commit")
    except LeaseLost as exc:
        logger.warning(f"{job_type}: {exc} -- discarding this call's writes")
        release_call_logger()
        raise
    except BaseException:
        logger.exception(f"{job_type}: failed under a held lease")
        release_call_logger()
        raise
    else:
        logger.info(f"{job_type}: done")
        # Close first, commit second. `release_call_logger` closes the file, so
        # the last lines above are on disk before the commit that ships them.
        release_call_logger()
        volume.commit()
