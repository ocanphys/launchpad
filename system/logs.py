"""Where a call's log goes: three channels in the `call_logs` Dict, one
writer each, and the file `persist_logs` appends them to. The launcher's
own log goes the same way: `launcher` is what the leasebook container has
logged (`start_launcher_logging`), republished whole on the same cadence,
`launcher:volume` is its file's rows, `logs/launcher.jsonl` at the volume
root, and the persist pass appends the live list's rows past the file's
count like any channel's. A new container continues the last one's list,
which is what keeps that cursor true across restarts.

`{call_id}:container` is the worker's: `BufferHandler` keeps every row the
call logs, filtered by `CallFilter` on the call id Modal keeps in a
contextvar so a container that runs one call after another never files a
row under the wrong one, and the heartbeat republishes the whole list.
`{call_id}:launcher` is the launcher's, one row per thing it did to the call
(`launcher_log`). `{call_id}:volume` is the file's,
`{artifact_path}/logs/{call_id}.jsonl`: read at leasebook startup and at
the top of every persist pass (`load_snapshot_from_volume`) and extended
each time the pass appends the other two channels' new rows to it
(`save_snapshot_to_volume`). Which calls belong to an artifact is the
`call_history` Dict's to say, written by the launcher at grant time and to
`call_history.json` on the same pass.
"""

import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import modal

from config import CALL_HISTORY, HEARTBEAT_SECONDS, LOGS
from system.lease_protocol import call_history, call_logs

CHANNELS = ("launcher", "container")  # the two the persist pass appends to the file
LAUNCHER_LOG_KEY = "launcher"
LAUNCHER_VOLUME_KEY = f"{LAUNCHER_LOG_KEY}:volume"


def setup_logging() -> None:
    """Point this container's logging at stdout, one line per record.

    Called once, at the top of a container's life. Modal captures stdout per
    function call, which is where `modal app logs` reads it back from.
    """
    logging.Formatter.converter = time.gmtime  # timestamps are UTC wherever this runs
    logging.basicConfig(
        level=logging.INFO,  # the job's own logger goes to DEBUG; libraries stay here
        stream=sys.stdout,
        format="%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,  # whatever a library configured first does not get to win
    )


def current_call_id() -> str:
    """The Modal call this code runs under, or "local" outside a container."""
    return modal.current_function_call_id() or "local"


_formatter = logging.Formatter()  # only for its traceback rendering


def row(record: logging.LogRecord) -> dict:
    """One record as the dict every channel keeps, traceback folded into `msg`."""
    msg = record.getMessage()
    if record.exc_info:
        msg += "\n" + _formatter.formatException(record.exc_info)
    return {"ts": record.created, "level": record.levelname, "logger": record.name, "msg": msg}


class CallFilter(logging.Filter):
    """Passes a record only when it was logged under `call_id`."""

    def __init__(self, call_id: str):
        super().__init__()
        self.call_id = call_id

    def filter(self, record: logging.LogRecord) -> bool:
        return current_call_id() == self.call_id


class BufferHandler(logging.Handler):
    """Keeps every row logged in `rows`, in order, for a heartbeat to
    publish: one call's rows given its `call_id`, the whole container's
    given none. Touches no file."""

    def __init__(self, call_id: str | None = None):
        super().__init__()
        if call_id is not None:
            self.addFilter(CallFilter(call_id))
        self.rows: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.rows.append(row(record))


def start_launcher_logging() -> Callable[[], None]:
    """Points this container's logging at stdout and starts the thread that
    republishes everything it logs, whole, as `call_logs["launcher"]` every
    HEARTBEAT_SECONDS; returns what stops it.

    The list continues the last container's, so it extends the file the
    persist pass appends to like a call's channel does, across restarts:
    the longer of what that container published and what the file holds,
    so a wiped Dict comes back from the file. Stopping publishes once more,
    so the rows logged on the way out land.
    """
    setup_logging()
    for noisy in ("modal", "grpc", "urllib3"):  # the publisher's own RPCs stay out of the log
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("leasebook")
    log.setLevel(logging.DEBUG)  # everything; filtering is the reader's job
    buffer = BufferHandler()
    buffer.rows = list(max((call_logs.get(key) or [] for key in (LAUNCHER_LOG_KEY, LAUNCHER_VOLUME_KEY)), key=len))
    logging.getLogger().addHandler(buffer)
    finished = threading.Event()

    def publish():
        while True:
            stop = finished.wait(HEARTBEAT_SECONDS)
            try:
                call_logs.put(LAUNCHER_LOG_KEY, list(buffer.rows))
            except Exception as exc:
                log.warning(f"launcher log not published ({exc})")
            if stop:
                return

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()

    def stop():
        finished.set()
        thread.join()
        logging.getLogger().removeHandler(buffer)

    return stop


def launcher_log(call_id: str, msg: str, level: str = "INFO") -> None:
    """Appends one row to `call_logs["{call_id}:launcher"]` at `level` and logs
    `msg` there too, so what the launcher did to a call reads on the call's
    artifact page and in the launcher's own log alike.

    `level` is a level name, the same string the row carries and the log views
    filter on. DEBUG is for what was asked of a call -- the page hides those
    rows until a reader switches the level on; what became of it is INFO.

    The append is a read-modify-write on one key, and the one leasebook
    container is its only writer.
    """
    key = f"{call_id}:launcher"
    entry = {"ts": time.time(), "level": level, "logger": "launcher", "msg": msg}
    call_logs.put(key, [*(call_logs.get(key) or []), entry])
    logging.getLogger("leasebook").log(logging.getLevelNamesMapping()[level], msg)


@contextmanager
def open_json(file: Path) -> Generator[dict, None, None]:
    """The object in `file` on the mount, `{}` for a file that does not
    exist, the descriptor closed on the way out."""
    if not file.exists():
        yield {}
        return
    with file.open() as f:
        yield json.load(f)


@contextmanager
def open_jsonl(file: Path) -> Generator[Iterator[dict], None, None]:
    """The rows of `file` on the mount, parsed as they are read, for the
    length of the block; a file that does not exist reads as no rows. The
    descriptor is closed on the way out, so a reload after the block never
    finds it open: consume the rows inside.

        with open_jsonl(root / artifact_path / TRAIN_LOG) as rows:
            train = list(rows)
    """
    if not file.exists():
        yield iter(())
        return
    with file.open() as f:
        yield parse_jsonl(f)


def parse_jsonl(lines: Iterable[str]) -> Iterator[dict]:
    """One row per line, stopping at a last line still landing (no newline
    yet) rather than raising on it."""
    for line in lines:
        try:
            yield json.loads(line)
        except ValueError:
            if line.endswith("\n"):
                raise
            return


def load_snapshot_from_volume(root: Path) -> dict[str, int]:
    """How many rows of each `call_logs` channel (`{call_id}:launcher`,
    `{call_id}:container`, `launcher`) the files already hold, the cursor
    `save_snapshot_to_volume` appends from, after publishing every log file
    as its `:volume` channel and merging `call_history.json` into the
    `call_history` Dict.

    The merge is a union per artifact, so a wiped Dict comes back from the
    file and a file behind the Dict drops nothing; a log file no grant names
    gets a grant with no `granted_ts`, so nothing on the volume goes
    unlisted. Reads the mount: the caller reloads first.
    """
    history_file = root / CALL_HISTORY
    history = json.loads(history_file.read_text()) if history_file.exists() else {}
    launcher_file = root / LOGS / f"{LAUNCHER_LOG_KEY}.jsonl"
    with open_jsonl(launcher_file) as parsed:
        rows = list(parsed)
    call_logs.put(LAUNCHER_VOLUME_KEY, rows)
    persisted = {LAUNCHER_LOG_KEY: len(rows)}
    for file in sorted(root.rglob(f"{LOGS}/*.jsonl")):
        if file == launcher_file:
            continue
        with open_jsonl(file) as parsed:
            rows = list(parsed)
        call_logs.put(f"{file.stem}:volume", rows)
        for source in CHANNELS:
            persisted[f"{file.stem}:{source}"] = sum(r.get("source") == source for r in rows)
        grants = history.setdefault(file.parent.parent.relative_to(root).as_posix(), [])
        if all(grant["call_id"] != file.stem for grant in grants):
            grants.append({"call_id": file.stem, "granted_ts": None, "artifact_type": None})
    for artifact_path, grants in history.items():
        merged = {grant["call_id"]: grant for grant in (*grants, *(call_history.get(artifact_path) or []))}
        call_history.put(artifact_path, sorted(merged.values(), key=lambda grant: grant["granted_ts"] or 0))
    return persisted


def save_snapshot_to_volume(root: Path, volume: modal.Volume, persisted: dict[str, int]) -> None:
    """Appends the rows past `persisted`'s cursor to their file: the
    launcher and container rows of every call in `call_history` to the
    call's, the launcher's own to `logs/launcher.jsonl` at the root. Extends
    each file's `:volume` channel by the same rows, writes `call_history`
    to `call_history.json` and commits.

    A channel is append-only, so the rows past the cursor are the whole
    diff; a channel that came back shorter (a wiped Dict) moves nothing.
    """

    def new_rows(key: str) -> list[dict]:
        return (call_logs.get(key) or [])[persisted.get(key, 0):]

    history = dict(call_history.items())
    files = [
        (
            root / artifact_path / LOGS / f"{grant['call_id']}.jsonl",
            f"{grant['call_id']}:volume",
            [{**r, "source": source} for source in CHANNELS for r in new_rows(f"{grant['call_id']}:{source}")],
        )
        for artifact_path, grants in history.items()
        for grant in grants
    ]
    files.append((root / LOGS / f"{LAUNCHER_LOG_KEY}.jsonl", LAUNCHER_VOLUME_KEY, new_rows(LAUNCHER_LOG_KEY)))
    for file, volume_key, new in files:
        if not new:
            continue
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in new)
        call_logs.put(volume_key, [*(call_logs.get(volume_key) or []), *new])
    tmp = root / f"{CALL_HISTORY}.tmp"
    tmp.write_text(json.dumps(history))
    tmp.replace(root / CALL_HISTORY)
    volume.commit()
