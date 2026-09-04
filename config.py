APP_NAME = "launchpad"
VOLUME_NAME = "trainvols"
STORAGE = "/storage"  # this is the container mount name for the volume.
CONTAINER_LIFETIME = 3600  # no container lives beyond this many seconds.
HEARTBEAT_SECONDS = 1
FLATLINE = 5  # if heartbeat age is longer than this many HEARTBEAT_SECONDS, the call is not active.

# Where a notebook lives on the volume, relative to STORAGE. lab.py names
# this folder; the `jupyter` function (main.py) roots its file browser one
# level up, at STORAGE itself, so this is a landmark inside it, not a mount
# point of its own.
LAB_NOTEBOOKS = "notebooks"

LAB_PORT = 8888
LAB_SECRET = "launchpad-lab"  # supplies JUPYTER_TOKEN to both the lab and the dashboard
LAB_IDLE_SECONDS = 900  # scale the lab container down after this much idle

# The lab container has no `git` binary and no `.git` directory -- only the
# named source files are shipped there, not the repo -- so it can't answer
# `get_git_commit()` itself. main.py calls get_git_commit() once, locally,
# while building lab_image, and bakes the result in under this env var name;
# dag.artifact._head() reads it instead of shelling out when it's set.
LAB_COMMIT_ENV = "LAUNCHPAD_LAB_COMMIT"

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# lab.py's "local" target -- a project-local storage root with the same
# manifest/artifact layout as the volume, for declaring/binding against
# without a Modal deployment. Under .scratch/, already gitignored. Distinct
# from the older, informal `.scratch/demo-volume` the demo notebooks build by
# hand -- those don't go through lab.py and aren't part of this.
LOCAL_STORAGE = PROJECT_ROOT / ".scratch" / "storage"


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
