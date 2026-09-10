"""Where a container's output goes: stdout, which Modal captures per call.

A worker takes that capture and files it in `call_logs[call_id]` (live,
expires) and at `archive_path` (for good). `leasebook` opens neither file: it
reloads the mount on a clock, and a reload cannot run while the same
container has a file open (docs/QUEUES.md §3.3).
"""

import logging
import sys
import time
from pathlib import Path

from config import CALL_LOGS


def setup_logging() -> None:
    """Point this container's logging at stdout, one line per record.

    Called once, at the top of a container's life. There are no handlers to
    manage and no files to close: Modal captures stdout per function call, so
    the call id it is filed under is Modal's business, not ours.
    """
    logging.Formatter.converter = time.gmtime  # timestamps are UTC wherever this runs
    logging.basicConfig(
        level=logging.INFO,  # the job's own logger goes to DEBUG; libraries stay here
        stream=sys.stdout,
        format="%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,  # whatever a library configured first does not get to win
    )


def archive_path(root: Path, artifact_path: str, call_id: str) -> Path:
    """Where one call's log is kept for good -- inside the artifact folder, so
    one `ls` shows every call ever launched for it, superseded attempts included.
    """
    return root / artifact_path / CALL_LOGS / f"{call_id}.log"
