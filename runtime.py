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

import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import modal

from config import STORAGE
from lease_protocol import Lease, LeaseLost
from logs import call_logger, release_call_logger


@dataclass(frozen=True)
class CallScope:
    """What a job is handed. Everything call-specific, nothing call-specific to build.

    `confirm` is the lease's bound method rather than the lease itself: a job has
    no business releasing, re-granting, or inspecting a lease, and the narrower
    handle is what keeps that true without a rule anyone has to remember.
    """

    run_id: str
    call_id: str
    log: logging.Logger
    confirm: Callable[..., None]
    dir: Path


@contextmanager
def call_scope(run_id: str, owner: str, volume: modal.Volume):
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

    logger = call_logger(call_id, log_dir / f"{owner}.log")
    lease = Lease(run_id, call_id, logger)
    scope = CallScope(run_id=run_id, call_id=call_id, log=logger, confirm=lease.confirm, dir=run_dir)

    try:
        lease.confirm(f"{owner}: boot")
        yield scope
        lease.confirm(f"{owner}: commit")
    except LeaseLost as exc:
        logger.warning(f"{owner}: {exc} -- discarding this call's writes")
        release_call_logger()
        raise
    except BaseException:
        logger.exception(f"{owner}: failed under a held lease")
        release_call_logger()
        raise
    else:
        logger.info(f"{owner}: done")
        # Close first, commit second. `release_call_logger` closes the file, so
        # the last lines above are on disk before the commit that ships them.
        release_call_logger()
        volume.commit()
