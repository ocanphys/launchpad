"""One shape for every log row, and where a row of that shape goes.

A row is `{"ts", "call_id", "source", "level", "logger", "msg"}`, a traceback
folded into `msg`, and the `call_logs` key it lives under is
`{call_id}:{storage}:{source}` -- `storage` is `livedict` for what a container
has published or `volume` for what its file holds, and `source` is who wrote
it: `worker`, `launcher`, or `ambient` for a record none of our loggers
stamped (a library's warning is the container's own noise, not the call's
account of itself).

A container configures the whole of it with `start_logging(call_id)`, which is
whose its unstamped records are, and logs through
`call_logger(name, source, call_id)`, which stamps the rest. The launcher
stamps what it did to a call it manages with that call's id and its own doings
with `LAUNCHER`; a row about a call is filed under both, so the artifact page
reads the grant and the cancel while the launcher's own log stays the whole
account of what the container did.

`persist_snapshot` is the only thing that writes a log file:
`{artifact_path}/logs/{call_id}.jsonl` for a call, `logs/launcher.jsonl` at
the volume root for the launcher. Which calls belong to an artifact is the
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
from functools import partial
from pathlib import Path

import modal

from config import CALL_HISTORY, HEARTBEAT_SECONDS, LOGS
from system.lease_protocol import call_history, call_logs

LIVE, VOLUME = "livedict", "volume"  # the Dict a container publishes to, and the file
# Who wrote a row. LAUNCHER is also a call id: the one the leasebook container
# files its own rows under, so its log is a call's log in every other way.
WORKER, LAUNCHER, AMBIENT = "worker", "launcher", "ambient"
SOURCES = (WORKER, LAUNCHER, AMBIENT)

_log = logging.getLogger(__name__)  # the publisher's own, ambient wherever it runs


def channel(call_id: str, storage: str, source: str) -> str:
    """The `call_logs` key one source's rows for one call live under."""
    return f"{call_id}:{storage}:{source}"


def stream(call_id: str, storage: str) -> list[dict]:
    """Every row `storage` holds for `call_id`, all sources together: what a
    route hands back for the page to union with the other storage's."""
    return [
        row
        for source in SOURCES
        for row in (call_logs.get(channel(call_id, storage, source)) or [])
    ]


def current_call_id() -> str:
    """The Modal call this code runs under, or "local" outside a container."""
    return modal.current_function_call_id() or "local"


def call_logger(name: str, source: str, call_id: str) -> logging.LoggerAdapter:
    """A logger whose every record carries `source` and `call_id`, from any
    thread, so a handler files it without having to ask which call is
    current."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # everything; filtering is the reader's job
    return logging.LoggerAdapter(logger, {"source": source, "call_id": call_id})


def launcher_logger(call_id: str = LAUNCHER) -> logging.LoggerAdapter:
    """The leasebook's logger, for what it is doing to `call_id` -- or, given
    no call, for the container itself."""
    return call_logger("leasebook", LAUNCHER, call_id)


_formatter = logging.Formatter()  # only for its traceback rendering


def row(record: logging.LogRecord, call_id: str) -> dict:
    """One record as the dict every channel keeps.

    `call_id` is the container's own, which a record no adapter stamped is
    filed under as `ambient`.
    """
    msg = record.getMessage()
    if record.exc_info:
        msg += "\n" + _formatter.formatException(record.exc_info)
    return {
        "ts": record.created,
        "call_id": getattr(record, "call_id", call_id),
        "source": getattr(record, "source", AMBIENT),
        "level": record.levelname,
        "logger": record.name,
        "msg": msg,
    }


def continued(call_id: str, source: str) -> list[dict]:
    """What a container's channel starts from: the longer of what the Dict
    holds and what the file's rows say.

    A channel is one list republished whole, so a container that started its
    own from empty would drop the rows a previous one published and leave the
    count the persist pass appends from pointing past them.
    """
    keys = (channel(call_id, LIVE, source), channel(call_id, VOLUME, source))
    return list(max((call_logs.get(key) or [] for key in keys), key=len))


def try_publish(what: str, put: Callable[[], None], logger: logging.Logger | logging.LoggerAdapter) -> None:
    """Runs `put`, turning a failure into a warning row the next pass carries
    rather than an exception that would take its thread down."""
    try:
        put()
    except Exception as exc:
        logger.warning(f"{what} not published ({exc})")


class BufferHandler(logging.Handler):
    """Keeps every record as a row in the live channel its own `call_id` and
    `source` name -- and in this container's own, when the row is about
    another call -- and publishes each of those channels whole. Touches no
    file."""

    def __init__(self, call_id: str):
        super().__init__()
        self.call_id = call_id
        self.channels: dict[str, list[dict]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        entry = row(record, self.call_id)
        # Under the call it is about, and under this container's own call id as
        # well when they differ: what the launcher did to a call reads on that
        # call's page and in the launcher's own log alike. A worker stamps no
        # call but its own, so there the two are one.
        for call_id in {entry["call_id"], self.call_id}:
            key = channel(call_id, LIVE, entry["source"])
            if key not in self.channels:
                # Claimed before the read, which reaches the Dict and can log
                # on this thread; whatever it logs stays behind what it reads.
                self.channels[key] = []
                self.channels[key][:0] = continued(call_id, entry["source"])
            self.channels[key].append(entry)

    def publish(self) -> None:
        """Puts every channel whole. This container is the only writer of each
        key it holds, so nothing is read back first."""
        with self.lock:
            channels = {key: list(rows) for key, rows in self.channels.items()}
        for key, rows in channels.items():
            try_publish(key, partial(call_logs.put, key, rows), _log)


def start_logging(call_id: str) -> Callable[[], None]:
    """Points this container's logging at stdout, keeps every record in the
    live channel its stamp names, and publishes them all every
    HEARTBEAT_SECONDS; returns what stops it.

    `call_id` is whose the container's unstamped records are -- LAUNCHER for
    the leasebook, its own call id for a worker. Stopping publishes once more,
    so the rows logged on the way out land.
    """
    logging.Formatter.converter = time.gmtime  # timestamps are UTC wherever this runs
    logging.basicConfig(
        level=logging.INFO,  # our own loggers go to DEBUG; libraries stay here
        stream=sys.stdout,  # Modal captures it per call, which `modal app logs` reads back
        format="%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,  # whatever a library configured first does not get to win
    )
    for noisy in ("modal", "grpc", "urllib3", "uvicorn.access"):  # the publisher's own RPCs stay out of the log
        logging.getLogger(noisy).setLevel(logging.WARNING)
    buffer = BufferHandler(call_id)
    logging.getLogger().addHandler(buffer)
    finished = threading.Event()

    def publish() -> None:
        while True:
            stop = finished.wait(HEARTBEAT_SECONDS)
            buffer.publish()
            if stop:
                return

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()

    def stop() -> None:
        logging.getLogger().removeHandler(buffer)
        finished.set()
        thread.join()

    return stop


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


def load_snapshot_from_volume(root: Path) -> dict[str, list[dict]]:
    """The rows of every log file under `root`, keyed by the `volume` channel
    each was published as, after merging `call_history.json` into the
    `call_history` Dict.

    The merge is a union per artifact, so a wiped Dict comes back from the
    file and a file behind the Dict drops nothing; a log file no grant names
    gets a grant with no `granted_ts`, so nothing on the volume goes
    unlisted. Reads the mount: the caller reloads first.
    """
    history_file = root / CALL_HISTORY
    history = json.loads(history_file.read_text()) if history_file.exists() else {}
    launcher_file = root / LOGS / f"{LAUNCHER}.jsonl"
    filed: dict[str, list[dict]] = {}
    for file in sorted(root.rglob(f"{LOGS}/*.jsonl")):
        with open_jsonl(file) as parsed:
            rows = list(parsed)
        for source in SOURCES:
            written = [entry for entry in rows if entry.get("source") == source]
            if written:
                key = channel(file.stem, VOLUME, source)
                call_logs.put(key, written)
                filed[key] = written
        if file == launcher_file:  # at the root, and the only file naming no artifact
            continue
        grants = history.setdefault(file.parent.parent.relative_to(root).as_posix(), [])
        if all(grant["call_id"] != file.stem for grant in grants):
            grants.append({"call_id": file.stem, "granted_ts": None, "artifact_type": None})
    for artifact_path, grants in history.items():
        merged = {grant["call_id"]: grant for grant in (*grants, *(call_history.get(artifact_path) or []))}
        call_history.put(artifact_path, sorted(merged.values(), key=lambda grant: grant["granted_ts"] or 0))
    return filed


def persist_snapshot(root: Path, volume: modal.Volume) -> None:
    """Appends every live row past what the files already hold to the call's
    file -- `{artifact_path}/logs/{call_id}.jsonl`, and `logs/launcher.jsonl`
    at the root for the launcher -- extends each file's `volume` channels by
    the same rows, writes `call_history` to `call_history.json` and commits.

    One `persist_logs` pass, and the only writer of any of those files. It
    reads the files back first, so the count it appends from is the files'
    own and a pass that never ran costs nothing but lag. A channel is
    append-only, so the rows past that count are the whole diff; a channel
    that came back shorter (a wiped Dict) moves nothing.
    """
    filed = load_snapshot_from_volume(root)
    history = dict(call_history.items())
    files = {
        grant["call_id"]: root / artifact_path / LOGS / f"{grant['call_id']}.jsonl"
        for artifact_path, grants in history.items()
        for grant in grants
    }
    files[LAUNCHER] = root / LOGS / f"{LAUNCHER}.jsonl"
    for call_id, file in files.items():
        new = {}
        for source in SOURCES:
            already = filed.get(channel(call_id, VOLUME, source), [])
            live = call_logs.get(channel(call_id, LIVE, source)) or []
            if len(live) > len(already):
                new[source] = live[len(already):]
        if not new:
            continue
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a") as f:
            f.writelines(json.dumps(entry) + "\n" for rows in new.values() for entry in rows)
        for source, rows in new.items():
            key = channel(call_id, VOLUME, source)
            call_logs.put(key, [*filed.get(key, []), *rows])
    tmp = root / f"{CALL_HISTORY}.tmp"
    tmp.write_text(json.dumps(history))
    tmp.replace(root / CALL_HISTORY)
    volume.commit()
