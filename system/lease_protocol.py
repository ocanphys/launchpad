import time

import modal

from config import APP_NAME, VOLUME_NAME

# Three Dicts, not one shared store with prefixed keys: an artifact_path is
# already a unique key in `leases`, a call_id is already a unique key in `beats`
# and in `call_logs`, and they never need to tell each other's keys apart
# because they are never in the same Dict. Every key any of this ever touches is
# a top-level key -- no blob, no read-modify-write, no chance of one write
# clobbering an unrelated entry. `call_logs` is its own Dict rather than a field
# of the beat so that a log too big to store can fail without taking the
# heartbeat down with it.
leases = modal.Dict.from_name(f"{APP_NAME}-leases", create_if_missing=True)
beats = modal.Dict.from_name(f"{APP_NAME}-beats", create_if_missing=True)
call_logs = modal.Dict.from_name(f"{APP_NAME}-call-logs", create_if_missing=True)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

LEASE_RETRIES = 5  # how many times an indeterminate lease read is worth re-asking
LEASE_BACKOFF = 5.0  # seconds between retries.

MATCH, MISMATCH, UNKNOWN = "match", "mismatch", "unknown"


class LeaseLost(Exception):
    """We can no longer prove we own this artifact, so we must stop writing.

    Raised rather than returned: the check sits inside a context manager the job's
    loop enters, so raising unwinds that loop without the job knowing a lease
    exists.
    """


def new_grant(call_id: str, artifact_type: str) -> dict:
    """
    create the value stored under leases[artifact_path].
    """
    return {
        "call_id": call_id,
        "granted_ts": time.time(),
        "attempt": 1,
        "artifact_type": artifact_type,
    }


def fence(artifact_path: str, my_call_id: str) -> tuple[str, dict | None]:
    """Do we own this artifact? One fresh read, three answers -- never two.

    The grant comes back with the verdict: the read already fetched it, and callers
    want to name *who* holds the artifact, not just whether we do.

    - MATCH / MISMATCH are real answers. Someone else's id means stop at once;
      asking again is just hoping it changes.
    - UNKNOWN (no key, or the read failed) is silence, not a no -- a Dict outage,
      or a call never granted anything. Denying on silence takes down the fleet
      on one blip; granting on it lets a zombie write. So callers wait a bounded
      time, then give up.

    Always the module's own `leases` Dict. Nothing here runs anywhere but inside a
    container or the local entrypoint, both of which have a real Dict to reach.
    """
    try:
        grant = leases.get(artifact_path)
    except Exception:
        return UNKNOWN, None
    if grant is None:
        return UNKNOWN, None
    return (MATCH if grant["call_id"] == my_call_id else MISMATCH), grant


class Lease:
    """
    this is a worker's handle of a lease.
    worker will ask the lease manager for the owner of the lease for
    artifact_path and check if its own call_id matches what is in the global
    record.
    """

    def __init__(
        self,
        artifact_path: str,
        call_id: str,
        logger,
        tries: int = LEASE_RETRIES,
        backoff: float = LEASE_BACKOFF,
    ):
        self.artifact_path = artifact_path
        self.call_id = call_id
        self.logger = logger
        self.tries = tries
        self.backoff = backoff

    def confirm(self, label: str) -> None:
        """check if the grant still names us, or raise LeaseLost.

        Someone else's id raises at once; only UNKNOWN (see `fence`) is worth
        waiting out, and only briefly. Waiting cannot cost us the artifact -- the
        launcher checks Modal for liveness before reassigning, and a worker that
        merely lost the Dict still looks alive there.

        Passes are logged too: afterwards only the log tells a boundary that
        committed from one that was merely allowed to. `label` has no default --
        a caller states what point in its own work this confirm guards (e.g.
        "before run", "before commit"), because the log line is only useful if
        it says what was about to happen, and two confirms sharing an unstated
        default read as one confirm logged twice.
        """
        for i in range(1, self.tries + 1):
            verdict, grant = fence(self.artifact_path, self.call_id)
            if verdict == MATCH:
                self.logger.debug(
                    f"{label}: lease held (attempt {grant['attempt']}, try {i}/{self.tries})"
                )
                return
            if verdict == MISMATCH:
                # Which kind of holder matters to whoever reads this log: a
                # different artifact_type means someone else is now producing
                # this artifact under us, i.e. we were superseded.
                raise LeaseLost(
                    f"{label}: another {grant['artifact_type']} holds this artifact ({grant['call_id']})"
                )
            if i == self.tries:
                raise LeaseLost(
                    f"{label}: indeterminate after {self.tries} tries -- ownership never confirmed"
                )
            self.logger.debug(f"{label}: indeterminate, retry {i}/{self.tries}")
            time.sleep(self.backoff)
