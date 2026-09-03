"""One call, one scope: a logger, a lease, and a commit that had to be earned.

`system.logs` gives a call its own logger; `system.lease_protocol` tells a call whether it
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
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import modal

from config import HEARTBEAT_SECONDS, STORAGE
from system.lease_protocol import Lease, LeaseLost, beats
from system.logs import LOG_FILENAME, call_logger, release_call_logger


@dataclass(frozen=True)
class Worker:
    """What a job is handed. Everything call-specific, nothing call-specific to build.

    `confirm` is the lease's bound method rather than the lease itself: a job has
    no business releasing, re-granting, or inspecting a lease, and the narrower
    handle is what keeps that true without a rule anyone has to remember.

    No `dir`: a job resolves its own paths via `artifact.paths(root)` (root
    being the storage root, a constant, not a per-call value), so there is
    nothing left for a per-call directory to do.

    `progress` is a plain mutable dict, not a method: a job reports by
    mutating it directly (`worker.progress.update(...)`), which costs nothing
    but a local write -- no network, so a job can report as often as it likes
    without ever blocking its own loop on one. Nobody else reads it except the
    heartbeat thread below, on its own cadence, which is the only thing that
    ever puts it on the wire.
    """

    artifact_path: str
    call_id: str
    log: logging.Logger
    confirm_lease: Callable[..., None]
    progress: dict


@contextmanager
def initialize_worker(artifact_path: str, volume: modal.Volume):
    """Set up one call's logger and lease; commit on the way out, if still owed.

    The log lands at `{artifact_path}/logs/{call_id}/job.log`, nested under
    the artifact's own folder the same way a run's logs used to be nested
    under the run's -- one job per artifact, so the folder already says what
    ran there.

    Three ways out, three policies, because they are three different events:

    - clean: confirm once more, close the log, commit.
    - `LeaseLost`: no commit. Someone else owns this artifact, and the writes
      under it are theirs now. The log lines still reach Modal's container
      log through the stream handler, which is the only place they can
      honestly go.
    - any other exception: close the log and commit anyway. The job failed
      while it still owned the artifact, so its log -- the thing that
      explains the failure -- is worth keeping, and a partial output under a
      held lease is not a correctness problem the way a superseded one is.
    """

    call_id = modal.current_function_call_id() or "local"

    # Before the log file is opened, never after: an open file on the volume
    # blocks reload, and this is the only moment nothing is open yet.
    volume.reload()

    log_dir = Path(STORAGE) / artifact_path / "logs" / call_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = call_logger(call_id, log_dir / LOG_FILENAME)
    lease = Lease(artifact_path, call_id, logger)
    worker = Worker(
        artifact_path=artifact_path,
        call_id=call_id,
        log=logger,
        confirm_lease=lease.confirm,
        progress={},
    )

    def heartbeat():
        # No fence check first -- see `Lease.confirm`'s note on why. A single
        # key write, not a read-modify-write: `beats` keys on call_id directly,
        # so this call's beat lives at a key nobody else ever writes to. Two
        # containers that have both, at different times, held this artifact get
        # two different keys -- this beat can never land on top of another
        # call's, and no read is needed first to avoid it. A reader tells a
        # stale beat from a live one by checking whether its call_id is still
        # the artifact's current holder (`main.read_state` does this), not by
        # racing to write first.
        #
        # `worker.progress` rides along on the same put, snapshotted with
        # `dict(...)` rather than passed by reference -- the job's own thread
        # can still be mutating it the moment this fires, and a snapshot is
        # what keeps the beat's payload internally consistent even so. Its own
        # cadence is entirely this thread's, not the job's: a job that reports
        # progress ten times between two beats still only ever puts twice, and
        # one that reports nothing at all just rides `None` along instead.
        while True:
            time.sleep(HEARTBEAT_SECONDS)
            try:
                beats.put(call_id, {
                    "artifact_path": artifact_path,
                    "last_beat_ts": time.time(),
                    "progress": dict(worker.progress) if worker.progress else None,
                })
            except Exception as exc:
                logger.warning(f"heartbeat: not recorded ({exc})")

    # A daemon thread, not a task: nothing here is async, so the only way to have
    # this run alongside the job's own code is a second thread. It dies with the
    # process on its own -- nothing to cancel on the way out.
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)

    try:
        heartbeat_thread.start()
        lease.confirm("boot")
        yield worker
        lease.confirm("commit")
    except LeaseLost as exc:
        logger.error(f"{exc} -- discarding this call's writes")
        release_call_logger()
        raise
    except BaseException:
        logger.exception("failed under a held lease")
        release_call_logger()
        raise
    else:
        logger.info("done")
        # Close first, commit second. `release_call_logger` closes the file, so
        # the last lines above are on disk before the commit that ships them.
        release_call_logger()
        volume.commit()
