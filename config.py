import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- the app and where it runs -----------------------------------------------

APP_NAME = "launchpad"
VOLUME_NAME = "trainvols"
STORAGE = "/storage"  # the container mount name for the volume
CONTAINER_LIFETIME = 3600  # seconds; no container lives beyond this
# Where the launcher-side containers run, not the GPU worker, which goes
# wherever its GPU is. Modal serves Dicts and Volumes from us-east, and a Dict
# get is ~25 ms here against ~250 ms from a far region.
REGION = "us-east"

# --- liveness: how a call is seen to be running ------------------------------

HEARTBEAT_SECONDS = 1
FLATLINE = 5  # a beat older than this many HEARTBEAT_SECONDS means the call is not active
STARTUP_GRACE_SECONDS = 60  # how long a granted lease may go without a first beat
# How long leasebook's listener blocks on the `refreshes` Queue before checking
# whether its container is shutting down. A message wakes the read at once.
REFRESH_WAIT_SECONDS = 30
# How many messages that read takes at once. One read is one recompute however
# many it took, so this is how far a burst of calls starting or ending together
# collapses; what it leaves behind is taken by the next read.
REFRESH_BATCH = 100

# --- what the volume holds: path names, each joined onto a root ---------------

LOGS = "logs"  # folder: `{artifact_path}/logs/{call_id}.jsonl`, and `logs/launcher.jsonl` at the root
CALL_HISTORY = "call_history.json"  # file at the root: every call ever granted, keyed by artifact_path
TRAIN_LOG = "train.jsonl"  # file in a leg's own folder: one row per training step
PERSIST_LOGS_EVERY = 60  # seconds between the passes appending the Dicts' log rows to those files

# --- the lab -----------------------------------------------------------------

LAB_PORT = 8888
LAB_SECRET = "launchpad-lab"  # supplies JUPYTER_TOKEN to both the lab and the dashboard
LAB_IDLE_SECONDS = 900  # scale the lab container down after this much idle
# The lab ships named source files, not the repo, so it has no `git` to answer
# `get_git_commit()` with. main.py bakes the answer in under this name while
# building lab_image; `artifacts.core.artifact._head()` reads it when it is set.
LAB_COMMIT_ENV = "LAUNCHPAD_LAB_COMMIT"


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
