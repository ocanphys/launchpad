"""What the volume and the Dicts say about every declared artifact.

The read side of the launcher: one map keyed by artifact path, and the
liveness a reader can have fresher than that map. Everything here is a
function of a root and the Dicts, so a test points it at a `tmp_path` and a
container points it at the mount.
"""

import threading
import time
from pathlib import Path

import modal

from artifacts.core.artifact import MANIFEST, Artifact
from config import (
    FLATLINE,
    HEARTBEAT_SECONDS,
    STARTUP_GRACE_SECONDS,
    STORAGE,
    VOLUME_NAME,
)
from system.lease_protocol import beats, call_history, leases
from system.logs import LIVE, VOLUME, stream

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# What the manifest at each artifact path decoded to, for this process's
# life. A definition at a path is immutable once declared (spec.md), and the
# map shows nothing else from the manifest, so a path read once is never read
# again here; what launching needs fresh (resources) `attempt_launch` reads
# off the volume itself.
resolved: dict[str, Artifact] = {}

# Held by anything in this process that reloads the mount or holds a file on
# it open: `state`, `attempt_launch`, and the dashboard's `/artifact` route. A
# reload replaces the mount's view of the volume and refuses to run while
# this process has a file under it open, so the listener thread's refresh and
# a request reading a manifest have to take turns (LESSONS.md).
mount_lock = threading.Lock()


def safe_relpath(path: str) -> bool:
    """Is `path` safe to join under STORAGE -- non-empty, not absolute, no
    `..` component that could walk it outside the volume. Every route that
    takes a path from the URL (an artifact_path or a run_id) checks this
    before it ever reaches `STORAGE / path`.

    safe_relpath("runs/toy/pretraining") -> True
    safe_relpath("../../etc/passwd") -> False
    """
    return bool(path) and not Path(path).is_absolute() and ".." not in Path(path).parts


def is_active(grant: dict | None, beat: dict | None, now: float) -> bool:
    """Whether a leased call is still working. A call marks its last beat
    `exited` on the way out, so the answer is no from that moment rather than
    a flatline later."""
    if grant is None or beat is None or beat.get("exited"):
        return False
    return now - beat["last_beat_ts"] < FLATLINE * HEARTBEAT_SECONDS


def is_starting(grant: dict | None, beat: dict | None, now: float) -> bool:
    """Whether a leased call is still within its grace period for a first beat."""
    if grant is None or beat is not None:
        return False
    granted_ts = grant.get("granted_ts")
    return granted_ts is not None and now - granted_ts < STARTUP_GRACE_SECONDS


def liveness(entry: dict, grant: dict | None, beat: dict | None, now: float) -> dict:
    """The half of an entry that is true only right now: which call holds the
    artifact, what it last said, and the one word the row shows.

        liveness(entry, grant, beat, now) -> {
            "call_id": "fc-01JQ8W", "active": True,
            "last_heartbeat": 1757260800.1,
            "live_progress": {"step": 120, "end_step": 500},
            "verdict": "running",
        }

    `entry` supplies what the volume said (`done`, `ready`, `status`); the
    grant and the beat supply what is true now. `verdict` is the first match
    of: "done"; "running" for an active call; "starting" for a lease younger
    than STARTUP_GRACE_SECONDS whose call has not beaten yet; "failed" for a
    manifest that would not load or a lease whose call stopped beating or
    exhausted its grace; "runnable" when ready; "blocked" otherwise. A failed
    artifact stays `ready`, so it can be run again.

    Nothing here touches the mount, so a caller holding a grant and a beat
    has this answer without a reload.
    """
    active = is_active(grant, beat, now)
    return {
        "call_id": grant["call_id"] if grant else None,
        "active": active,
        "last_heartbeat": beat["last_beat_ts"] if beat else None,
        "live_progress": beat.get("progress") if beat and active else None,
        "verdict": (
            "done" if entry["done"]
            else "running" if active
            else "starting" if is_starting(grant, beat, now)
            else "failed" if entry["status"] == "conflict" or grant
            else "runnable" if entry["ready"]
            else "blocked"
        ),
    }


def state(root: Path = STORAGE) -> dict[str, dict]:
    """The current state of every declared artifact under `root`, keyed by
    artifact path: one reload, one glob, one status apiece, one lease and
    heartbeat snapshot for the whole scan, and a manifest read only the
    first time this process sees its path (`resolved`).

        state()["runs/toy/pretraining"] -> {
            "type": "Pretraining", "status": "partial", "error": None,
            "depends_on": ["mappeddatasets/mapped-7f3c1a2b"], "blocked_by": [],
            "parameters": {"run_id": "toy", "config": {...}},
            "done": False, "ready": True, "verdict": "running",
            "call_id": "fc-01JQ8W", "active": True, "last_heartbeat": 1757260800.1,
            "live_progress": {"step": 120, "end_step": 500},
            "durable_progress": {"phase": "step", "done": 100, "total": 500},
        }

    A manifest that cannot be read is an entry with status "conflict" and
    its `error`, and the scan continues. `blocked_by` is every direct
    dependency not `done` on this same map, one with no manifest included;
    `ready` is additionally false for an artifact nothing produces.

    The last five fields are `liveness`, over this scan's lease and beat
    snapshot: every fact is from the one snapshot, so a lease granted after
    it is not on the map until the next scan. They are also the only fields
    a reader can have fresher than the map -- `/state` reads them again per
    request, for the calls this scan found under way.

    Holds `mount_lock` for the reload and the scan, so a caller can be kept
    waiting by whatever else in this process is reading the mount.
    """
    # `now` before the snapshot: the reload and the Dict scans take a
    # measurable time, and taking the clock after them would charge that
    # time to every beat's age. A beat written during the snapshot is
    # newer than `now` and reads as live, which is the true answer.
    now = time.time()
    with mount_lock:
        if root == STORAGE:
            volume.reload()
        beat_records = dict(beats.items())
        grants = dict(leases.items())

        # Subtrees the manifests read this scan share, decoded once: a run's
        # legs each embed every leg before them. Gone with the scan; `resolved`
        # keeps the artifacts.
        memo: dict[str, Artifact] = {}
        entries: dict[str, dict] = {}
        for manifest in sorted(root.rglob(MANIFEST)):
            path = manifest.parent.relative_to(root).as_posix()
            entry = {
                "type": None,
                "status": "conflict",
                "error": None,
                "depends_on": [],
                "parameters": None,
                "durable_progress": None,
            }
            try:
                if path not in resolved:
                    resolved[path] = Artifact.load(path, root, memo)
            except ValueError as error:
                entry["error"] = str(error)
            else:
                artifact = resolved[path]
                present = list(artifact.status(root).completion.values())
                entry.update(
                    type=type(artifact).__name__,
                    status="done" if all(present) else "partial" if any(present) else "declared",
                    depends_on=[dep.artifact_path.as_posix() for dep in artifact.deps()],
                    # This artifact's own fields, never a dependency's manifest. A
                    # dependency is a path in `depends_on` and nothing more: its own
                    # entry on this same map is where anything else about it lives,
                    # so nothing is stored twice and nothing here goes stale when it
                    # changes.
                    parameters=artifact.parameters(),
                    durable_progress=artifact.durable_progress(root),
                )
            entries[path] = entry

    for path, entry in entries.items():
        blocked_by = [
            dep for dep in entry["depends_on"] if entries.get(dep, {}).get("status") != "done"
        ]
        done = entry["status"] == "done"
        entry["blocked_by"] = blocked_by
        entry["done"] = done
        entry["ready"] = path in resolved and resolved[path].producer is not None and not done and not blocked_by
        grant = grants.get(path)
        entry.update(liveness(entry, grant, beat_records.get(grant["call_id"]) if grant else None, now))
    return entries


def artifact_calls(artifact_path: str) -> list[dict]:
    """Every call ever granted for `artifact_path`, oldest grant first, each
    with the last beat it left and its log in both storages -- what the
    containers have published (`livedict`) and what its file holds
    (`volume`) -- read from the Dicts now. Every row says which source wrote
    it, so the page unions the two and counts a row once.

    A call that never ran has `last_heartbeat` None; a grant
    `load_snapshot_from_volume` made up for a log file has `granted_ts` None
    and sorts first. Bounded by this one artifact's history: never a scan of
    either Dict.
    """
    return [
        {
            "call_id": grant["call_id"],
            "granted_ts": grant["granted_ts"],
            "last_heartbeat": (beats.get(grant["call_id"]) or {}).get("last_beat_ts"),
            LIVE: stream(grant["call_id"], LIVE),
            VOLUME: stream(grant["call_id"], VOLUME),
        }
        for grant in call_history.get(artifact_path) or []
    ]
