APP_NAME = "launchpad"
VOLUME_NAME = "trainvols"
STORAGE = "/storage"  # this is the container mount name for the volume.
CONTAINER_LIFETIME = 3600  # no container lives beyond this many seconds.
HEARTBEAT_SECONDS = 1
FLATLINE = 5  # if heartbeat age is longer than this many HEARTBEAT_SECONDS, the call is not active.

# Where a notebook lives on the volume, relative to STORAGE. The Jupyter
# container that used to serve them is removed for now; lab.py still names
# this folder, and the notebooks already on the volume are still there.
LAB_NOTEBOOKS = "notebooks"

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def get_git_commit(dirty_suffix: bool = True) -> str:
    """Git commit hash of PROJECT_ROOT's current HEAD. Meant to be called from
    the driver notebook right before dispatching a run to Modal -- captures
    exactly what code is about to be shipped, since add_local_python_source
    mounts local disk directly rather than doing any git checkout of its own,
    so there's no Modal-side notion of "commit" independent of this.

    Doesn't care which branch HEAD is on -- the hash alone fully identifies
    the commit's content regardless of branch (or even a detached HEAD with
    no branch at all).

    dirty_suffix: when True (default), appends "-dirty" if the working tree
    has uncommitted changes, since Modal ships whatever's actually on disk,
    not just the last commit.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty_suffix:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            commit += "-dirty"
    return commit
