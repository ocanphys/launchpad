APP_NAME = "launchpad"
VOLUME_NAME = "trainvols"
STORAGE = "/storage"  # this is the container mount name for the volume.
CONTAINER_LIFETIME = 3600  # no container lives beyond this many seconds.
HEARTBEAT_SECONDS = 1
FLATLINE = 5  # if heartbeat age is longer than this many HEARTBEAT_SECONDS, the call is not active.
STARTUP_GRACE_SECONDS = 60  # time after a lease is granted to wait for its first heartbeat.
PERSIST_LOGS_EVERY = 60  # seconds between `persist_logs` passes appending the Dict's log rows to the volume.
# How long leasebook's listener blocks on the `refreshes` Queue before looking
# at whether its container is shutting down. Nothing waits this long for a
# refresh: a message wakes the read at once.
REFRESH_WAIT_SECONDS = 30
# Where the launcher-side containers run. Modal's Dicts and Volumes are served
# from us-east, and every request those containers answer is a handful of
# round trips to them: a Dict get is ~25 ms here and ~250 ms from a far region.
# Not the GPU worker, which waits for whatever region has its GPU.
REGION = "us-east"

# One call's log, `{call_id}.jsonl` inside this folder of the artifact that
# call was producing, written by `persist_logs` alone. The
# `launchpad-call-logs` Dict holds three channels per call:
# `{call_id}:launcher` (the launcher's own rows), `{call_id}:container` (the
# worker's, republished whole on every heartbeat) and `{call_id}:volume`
# (the file's rows, read at leasebook startup and on every persist pass).
# The launcher's own log is `launcher.jsonl` in this folder at the volume
# root, with `launcher` and `launcher:volume` as its two channels.
LOGS = "logs"
# Every call ever granted, per artifact_path: the `launchpad-call-history`
# Dict, written to this file at the volume root on every persist pass and
# merged back into the Dict when leasebook starts and on every pass.
CALL_HISTORY = "call_history.json"
# A leg's per-step record, in its artifact folder: the worker's file, read
# off the volume by the dashboard once the worker has committed it.
TRAIN_LOG = "train.jsonl"

LAB_PORT = 8888
LAB_SECRET = "launchpad-lab"  # supplies JUPYTER_TOKEN to both the lab and the dashboard
LAB_IDLE_SECONDS = 900  # scale the lab container down after this much idle

# The lab container has no `git` binary and no `.git` directory -- only the
# named source files are shipped there, not the repo -- so it can't answer
# `get_git_commit()` itself. main.py calls get_git_commit() once, locally,
# while building lab_image, and bakes the result in under this env var name;
# artifacts.core.artifact._head() reads it instead of shelling out when it's set.
LAB_COMMIT_ENV = "LAUNCHPAD_LAB_COMMIT"

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def get_git_commit() -> str:
    """PROJECT_ROOT's HEAD hash, with "-dirty" appended when the working tree
    has uncommitted changes: `add_local_python_source` ships what is on disk,
    not the last commit, so a clean hash alone would overstate what ran.
    """
    git = lambda *args: subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    commit = git("rev-parse", "HEAD")
    return commit + "-dirty" if git("status", "--porcelain") else commit
