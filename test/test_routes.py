"""Does every route `leasebook` is supposed to serve actually answer?

`import main` proves nothing about the web app: the routes are registered
inside `leasebook()`, which only runs when a container starts, and a handler's
body only runs when a request arrives. So a deleted route, a name that no
longer exists, or a handler that raises are all invisible to an import -- and
one deletion that took `/launch` and `/cancel` with it is why this file
exists.

Nothing here talks to Modal. `leasebook`'s raw function is built against stubs:
`state` returns a fixed map, the volume-reading helpers return fixed
answers, and launching/cancelling record what they were asked rather than
spawning anything. What's left under test is the wiring -- which routes exist,
what shape they answer, and whether the handlers reference anything that isn't
there any more.

Plain functions named test_*, plain assert -- `python test/test_routes.py`,
same shape as the rest of test/. Needs `fastapi` and `httpx` locally (the
deployed image has both; `uv pip install fastapi httpx` if a bare checkout
doesn't).
"""

import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from config import FLATLINE, HEARTBEAT_SECONDS, STARTUP_GRACE_SECONDS
from system import logs

# Every listener thread `leasebook` starts is a daemon that outlives the test
# that made it, and it reads whatever `main.refreshes` names when it wakes.
# Standing in for the real Queue here, before any test saves the module's
# vars, is what keeps one of those threads from ever reaching Modal's.
main.refreshes = None


# `client` swaps the stubs straight into `main`; this hands the module back
# after each test, so test_state gets the real `state` and `volume`.
@pytest.fixture(autouse=True)
def restore_main():
    saved = dict(vars(main))
    yield
    vars(main).update(saved)


# --- the stubs ---------------------------------------------------------------


class Queue(queue.Queue):
    """The `refreshes` Queue as the listener thread uses it: one blocking
    read with a timeout, raising `queue.Empty` when nothing arrives."""

    def get_many(self, n_values: int, timeout: float | None = None) -> list:
        messages = [self.get(timeout=timeout)]
        while len(messages) < n_values:
            try:
                messages.append(self.get_nowait())
            except queue.Empty:
                break
        return messages


class Counting(dict):
    """A Dict that counts the reads made of it, so a test can assert that a
    route answered without asking Modal anything."""

    gets = 0

    def get(self, key, default=None):
        self.gets += 1
        return super().get(key, default)


def until(predicate, timeout: float = 2.0) -> bool:
    """Whether `predicate` came true inside `timeout` -- how a test waits on
    the listener thread without sleeping a fixed amount for it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()

STATE = {
    "sources/tinyshakespeare": {
        "type": "Source", "status": "done", "error": None,
        "depends_on": ["sources/other"], "parameters": {"name": "tinyshakespeare"},
        "blocked_by": [], "done": True, "ready": False, "verdict": "done", "call_id": None,
        "active": False, "last_heartbeat": None, "live_progress": None,
        "durable_progress": {"phase": "files", "done": 1, "total": 1},
    },
    # A manifest `state` couldn't read: no parameters to show, and the reason
    # is what the artifact's page shows in their place.
    "sources/broken": {
        "type": None, "status": "conflict", "error": "not a readable manifest",
        "depends_on": [], "parameters": None,
        "blocked_by": [], "done": False, "ready": False, "verdict": "failed", "call_id": None,
        "active": False, "last_heartbeat": None, "live_progress": None,
        "durable_progress": None,
    },
}


# What `/artifact/<path>` serves of a manifest: everything but the
# dependency manifests nested under `dependencies`.
MANIFEST = {
    "artifact": "artifacts.stages.pretraining.Pretraining",
    "commit": "a299cc8",
    "allocated_resources": {"cpu": None, "gpu_type": "A100", "gpu_count": 1},
    "parameters": {"run_id": "toy"},
}


def client(**overrides):
    """A TestClient over `leasebook`'s real ASGI app, with everything that
    would touch Modal replaced, and the volume a temporary folder holding
    one step log.
    """
    from fastapi.testclient import TestClient

    calls = {"launched": [], "cancelled": []}

    # Counted, so a test can tell a GET that served the snapshot from a
    # refresh that took a new one.
    calls["state"] = 0
    calls["reload"] = 0
    calls["stop_logging"] = overrides.get("stop_logging", Mock())
    main.start_logging = Mock(return_value=calls["stop_logging"])

    def state():
        calls["state"] += 1
        if "state_error" in overrides:
            raise overrides["state_error"]
        return overrides.get("state", STATE)

    main.state = state
    # What a worker puts a message on when its call exits, and the listener
    # thread blocks on. Its wait is long enough that an idle listener stays
    # parked on its own queue for the rest of the session instead of waking
    # up to read whichever one `main.refreshes` names by then.
    calls["refreshes"] = main.refreshes = Queue()
    main.REFRESH_WAIT_SECONDS = 300
    main.attempt_launch = lambda path: (
        calls["launched"].append(path) or (overrides.get("launched", True), f"granted {path}")
    )
    main.cancel_call = lambda path: (
        calls["cancelled"].append(path) or (overrides.get("cancelled", True), f"cancelled {path}")
    )
    main.jupyter = type("Stub", (), {"get_web_url": staticmethod(lambda: "https://lab.test")})()
    # The startup sync: nothing to reload, no log files to read.
    def reload():
        calls["reload"] += 1

    main.volume = type("Stub", (), {"reload": staticmethod(reload)})()
    main.load_snapshot_from_volume = lambda root: {}
    # A plain dict answers `.get()`/`.items()` the same way the Dict does.
    # Every call the launcher ever granted, per artifact, oldest first: fc-9
    # is a grant made up for a log file, fc-0 never beat, fc-1 is running.
    main.call_history = {
        "runs/toy/pretraining": [
            {"call_id": "fc-9", "granted_ts": None, "artifact_type": None},
            {"call_id": "fc-0", "granted_ts": 40.0, "artifact_type": "Pretraining"},
            {"call_id": "fc-1", "granted_ts": 90.0, "artifact_type": "Pretraining"},
        ],
        "sources/tinyshakespeare": [{"call_id": "fc-2", "granted_ts": 70.0, "artifact_type": "Source"}],
    }
    ROW = {"ts": 1757246400.0, "call_id": "fc-1", "source": "worker", "level": "INFO", "logger": "job", "msg": "boot: lease held"}
    GRANTED = {"ts": 90.0, "call_id": "fc-1", "source": "launcher", "level": "INFO", "logger": "leasebook", "msg": "granted"}
    NOISE = {"ts": 1757246401.0, "call_id": "fc-1", "source": "ambient", "level": "WARNING", "logger": "urllib3", "msg": "retrying"}
    # `stream` reads this one, so it is the logs module's the routes reach
    # through, not a name of main's.
    main.call_logs = logs.call_logs = {
        "fc-1:livedict:launcher": [GRANTED],
        "fc-1:livedict:worker": [ROW],
        "fc-1:livedict:ambient": [NOISE],
        "fc-1:volume:worker": [ROW],
        "fc-9:volume:worker": [ROW],
    }
    # The mount: a folder with one leg's manifest and step log on it.
    STEP = {"step": 1, "attempt": 1, "loss": 2.0, "grad_norm": 1.0, "learning_rate": 0.5}
    main.STORAGE = Path(mkdtemp())
    leg = main.STORAGE / "runs/toy/pretraining"
    leg.mkdir(parents=True)
    (leg / "manifest.json").write_text(json.dumps({**MANIFEST, "dependencies": {"dataset": {"artifact": "artifacts.dataset.DataSet"}}}))
    (leg / "train.jsonl").write_text(json.dumps(STEP) + "\n")
    main.beats = Counting({
        "fc-1": {"artifact_path": "runs/toy/pretraining", "last_beat_ts": 100.0},
        "fc-2": {"artifact_path": "sources/tinyshakespeare", "last_beat_ts": 75.0},
    })
    # Only `/state`'s overlay reads this one, and only for a call the map
    # found under way.
    main.leases = Counting()
    # In the container `web/` is mounted at /web; locally it's the repo's own
    # copy, which is the same files and lets the static mount succeed.
    main.WEB_DIR = Path(__file__).resolve().parents[1] / "web"

    return TestClient(main.leasebook.get_raw_f()(), follow_redirects=False), calls


# --- the routes --------------------------------------------------------------


def test_every_route_is_registered():
    api, _ = client()
    paths = {route.path for route in api.app.routes}
    for expected in (
        "/refresh",
        "/state",
        "/launcher-logs",
        "/lab",
        "/launch/{artifact_path:path}",
        "/cancel/{artifact_path:path}",
        "/artifact/{artifact_path:path}",
        "/logs/{artifact_path:path}",
    ):
        assert expected in paths, f"{expected} is not registered: {sorted(paths)}"


def test_state_serves_the_computed_map():
    api, _ = client()
    assert api.get("/state").json() == STATE


# --- the live overlay ---------------------------------------------------------
#
# What the map holds for an artifact the last recompute found under way. Its
# five live fields are the overlay's whole business; the rest is the volume's
# and must come back untouched.
RUNNING = {
    "type": "Pretraining", "status": "partial", "error": None,
    "depends_on": [], "parameters": {"run_id": "toy"},
    "blocked_by": [], "done": False, "ready": True,
    "verdict": "running", "call_id": "fc-1", "active": True,
    "last_heartbeat": 100.0, "live_progress": {"step": 1, "end_step": 500},
    "durable_progress": None,
}
LEG = "runs/toy/pretraining"
DURABLE = ("type", "status", "error", "depends_on", "parameters", "blocked_by", "done", "ready", "durable_progress")


def working(api, beat: dict | None = None, granted_age: float = 5.0) -> dict:
    """`LEG`'s entry off `/state`, with the Dicts holding one grant for it and
    `beat` -- `{"age": seconds since it was written, ...}`, or none at all."""
    main.leases[LEG] = {"call_id": "fc-1", "granted_ts": time.time() - granted_age}
    if beat is None:
        main.beats.pop("fc-1", None)
    else:
        main.beats["fc-1"] = {"artifact_path": LEG, **beat, "last_beat_ts": time.time() - beat.pop("age")}
    return api.get("/state").json()[LEG]


def test_state_reads_a_running_calls_progress_on_every_request():
    """The point of the overlay: the step count climbs on the page while
    nothing recomputes the map or reloads the volume."""
    api, calls = client(state={LEG: dict(RUNNING)})
    first = working(api, {"age": 0.2, "progress": {"step": 120, "end_step": 500}})
    assert first["live_progress"] == {"step": 120, "end_step": 500}
    assert first["verdict"] == "running" and first["active"] is True

    second = working(api, {"age": 0.1, "progress": {"step": 121, "end_step": 500}})
    assert second["live_progress"] == {"step": 121, "end_step": 500}
    assert second["last_heartbeat"] > first["last_heartbeat"]
    # The volume's half of the entry is the map's, untouched by any of it.
    assert {key: second[key] for key in DURABLE} == {key: RUNNING[key] for key in DURABLE}
    assert calls["state"] == 1, "the map was recomputed"
    assert calls["reload"] == 1, "the volume was reloaded after startup"


def test_state_reads_nothing_for_artifacts_no_call_is_working_on():
    """Bounded by the calls under way, not by the size of the map."""
    api, _ = client()  # STATE: one done, one unreadable, neither with a call
    assert api.get("/state").json() == STATE
    assert (main.leases.gets, main.beats.gets) == (0, 0)


def test_state_leaves_a_call_that_said_it_exited_to_the_queue():
    """Its ending is already on the way; reading that beat here would call a
    finished run "failed" for the moment before the recompute lands."""
    api, _ = client(state={LEG: dict(RUNNING)})
    entry = working(api, {"age": 0.1, "progress": {"step": 500}, "exited": True})
    assert entry == RUNNING


def test_state_fails_a_running_call_whose_beats_stopped():
    """A container killed hard announces nothing, so no recompute is coming:
    the flatline is what the row has to notice."""
    api, _ = client(state={LEG: dict(RUNNING)})
    entry = working(api, {"age": FLATLINE * HEARTBEAT_SECONDS + 1, "progress": {"step": 120}})
    assert entry["verdict"] == "failed"
    assert entry["active"] is False and entry["live_progress"] is None
    assert entry["ready"] is True, "a failed artifact can still be run again"


def test_state_expires_a_starting_calls_grace_without_a_recompute():
    api, _ = client(state={LEG: {**RUNNING, "verdict": "starting", "active": False, "last_heartbeat": None, "live_progress": None}})
    assert working(api, granted_age=STARTUP_GRACE_SECONDS - 1)["verdict"] == "starting"
    assert working(api, granted_age=STARTUP_GRACE_SECONDS + 1)["verdict"] == "failed"


def test_launcher_logs_are_live_and_do_not_refresh_state_or_storage():
    api, calls = client()
    assert api.get("/launcher-logs").json() == {"livedict": [], "volume": []}
    row = lambda ts, msg, source="launcher": {
        "ts": ts, "call_id": "launcher", "source": source,
        "level": "INFO", "logger": "leasebook", "msg": msg,
    }
    first, second = row(1.0, "started"), row(2.0, "failed\ntraceback")
    main.call_logs["launcher:livedict:launcher"] = [first, second]
    main.call_logs["launcher:volume:launcher"] = [first]
    # the noise around the leasebook's own rows is the same call's, told apart
    # by `source` rather than by belonging to something else
    noise = row(3.0, "GET /state", source="ambient")
    main.call_logs["launcher:livedict:ambient"] = [noise]
    assert api.get("/launcher-logs").json() == {"livedict": [first, second, noise], "volume": [first]}
    assert calls["state"] == 1
    assert calls["reload"] == 1
    assert "launcher" not in main.call_history


def test_launcher_logging_is_stopped_on_asgi_shutdown():
    api, calls = client()
    main.start_logging.assert_called_once_with(logs.LAUNCHER)
    with api:
        assert api.get("/launcher-logs").status_code == 200
        calls["stop_logging"].assert_not_called()
    calls["stop_logging"].assert_called_once_with()


def test_launcher_logging_is_stopped_after_logging_a_startup_failure():
    stop_logging = Mock()
    with TestCase().assertLogs("leasebook", level=logging.ERROR) as captured:
        try:
            client(stop_logging=stop_logging, state_error=RuntimeError("startup failed"))
        except RuntimeError as exc:
            assert str(exc) == "startup failed"
        else:
            raise AssertionError("startup failure was swallowed")
    assert "leasebook failed to start" in captured.output[0]
    assert "RuntimeError: startup failed" in captured.output[0]
    stop_logging.assert_called_once_with()


def test_request_failures_are_logged_with_their_traceback():
    api, _ = client()
    logs.call_logs = Mock()
    logs.call_logs.get.side_effect = RuntimeError("Dict unavailable")
    with TestCase().assertLogs("leasebook", level=logging.ERROR) as captured:
        try:
            api.get("/launcher-logs")
        except RuntimeError as exc:
            assert str(exc) == "Dict unavailable"
        else:
            raise AssertionError("request failure was swallowed")
    assert "GET /launcher-logs failed" in captured.output[0]
    assert "RuntimeError: Dict unavailable" in captured.output[0]


def test_state_is_computed_when_something_changed_it_and_never_on_a_read():
    """`/state` serves the map the container holds. A new one is computed
    when a call exits, when this container grants or releases a lease, and
    when the page asks -- never by a route that only reads. The logs, by
    contrast, are read from the Dicts on every request: a row that lands in
    `call_logs` shows up without any of that."""
    api, calls = client()
    assert calls["state"] == 1
    api.get("/state")
    api.get("/artifact/runs/toy/pretraining")
    api.get("/logs/runs/toy/pretraining")
    assert calls["state"] == 1
    assert api.post("/refresh").json() == {"artifacts": len(STATE)}
    assert calls["state"] == 2

    main.call_logs["fc-1:livedict:worker"] = [*main.call_logs["fc-1:livedict:worker"],
                                              {"ts": 101.0, "call_id": "fc-1", "source": "worker", "level": "INFO", "logger": "job", "msg": "step 1"}]
    body = api.get("/logs/runs/toy/pretraining").json()
    assert [row["msg"] for row in body["calls"][2]["livedict"]] == ["boot: lease held", "step 1", "granted", "retrying"]


def test_a_call_that_exited_recomputes_the_map_without_anyone_asking():
    """The whole point of the queue: the launcher's map catches up with the
    volume because the worker said it had committed, not because the page
    clicked."""
    api, calls = client()
    with api:
        assert calls["state"] == 1
        calls["refreshes"].put({"artifact_path": "runs/toy/pretraining", "call_id": "fc-1", "event": "done"})
        assert until(lambda: calls["state"] == 2), "the listener never recomputed"
        # A burst of messages is one recompute, not one apiece.
        for call_id in ("fc-3", "fc-4", "fc-5"):
            calls["refreshes"].put({"artifact_path": "sources/tinyshakespeare", "call_id": call_id, "event": "started"})
        assert until(lambda: calls["state"] > 2)
        time.sleep(0.05)
        assert calls["state"] == 3, "each message in one batch recomputed separately"


def test_an_accepted_launch_or_cancel_recomputes_before_it_answers():
    """So the page's next fetch of /state shows the lease it just asked for,
    with nothing to poll for and no refresh of its own to make."""
    api, calls = client()
    api.post("/launch/runs/toy/pretraining")
    assert calls["state"] == 2
    api.post("/cancel/runs/toy/pretraining")
    assert calls["state"] == 3

    # A refused one changed nothing, so there is nothing to recompute.
    api, calls = client(launched=False, cancelled=False)
    api.post("/launch/runs/toy/pretraining")
    api.post("/cancel/runs/toy/pretraining")
    api.post("/cancel/%2e%2e/etc/passwd")
    assert calls["state"] == 1


def test_the_artifact_route_and_a_recompute_never_read_the_mount_at_once():
    """`mount_lock`: a reload replaces the mount under whatever has a file
    open on it, so the route that opens one waits for the refresh holding
    the lock, and only then answers."""
    api, _ = client()
    answered = threading.Event()

    def read():
        api.get("/artifact/runs/toy/pretraining")
        answered.set()

    with main.mount_lock:
        threading.Thread(target=read, daemon=True).start()
        assert not answered.wait(0.2), "the route read the mount while the lock was held"
    assert answered.wait(2), "the route never answered after the lock was released"


def test_launch_reaches_the_launcher():
    api, calls = client()
    body = api.post("/launch/runs/toy/pretraining").json()
    assert body == {"launched": True, "message": "granted runs/toy/pretraining"}
    assert calls["launched"] == ["runs/toy/pretraining"], calls


def test_cancel_reaches_the_canceller():
    api, calls = client()
    body = api.post("/cancel/runs/toy/pretraining").json()
    assert body["cancelled"] is True
    assert calls["cancelled"] == ["runs/toy/pretraining"], calls


def test_a_path_that_walks_out_of_the_volume_is_refused():
    """`safe_relpath`, over HTTP.

    The traversal is percent-encoded because the client collapses `..` in a
    path before it ever sends the request -- a plain "/cancel/../../etc/passwd"
    is a request for "/etc/passwd" and never reaches the handler at all, which
    makes for a test that passes without testing anything. `%2e%2e` survives
    the client, and Starlette decodes it into the path parameter, which is
    what a caller who means it would do.
    """
    api, calls = client()
    cancelled = api.post("/cancel/%2e%2e/%2e%2e/etc/passwd").json()
    assert cancelled["cancelled"] is False and "invalid" in cancelled["message"], cancelled
    assert calls["cancelled"] == [], "a bad path reached the canceller"


def test_the_lab_link_carries_the_token():
    api, _ = client()
    root = api.get("/lab")
    assert root.status_code == 307 and "lab.test" in root.headers["location"]


def test_artifact_logs_aggregate_every_call_oldest_first():
    """`/logs/<path>` -- every call ever granted for this one artifact,
    not just its current holder, in the launcher's order, each with the
    beat it left and its log in both storages, out of the Dicts.
    """
    api, _ = client()
    body = api.get("/logs/runs/toy/pretraining").json()
    assert body["artifact_path"] == "runs/toy/pretraining"
    call_ids = [call["call_id"] for call in body["calls"]]
    assert call_ids == ["fc-9", "fc-0", "fc-1"], body

    by_id = {call["call_id"]: call for call in body["calls"]}
    assert by_id["fc-9"]["last_heartbeat"] is None and by_id["fc-9"]["granted_ts"] is None
    # fc-0 was granted and never ran: listed, with nothing in either storage.
    assert by_id["fc-0"]["last_heartbeat"] is None
    assert (by_id["fc-0"]["livedict"], by_id["fc-0"]["volume"]) == ([], [])
    assert by_id["fc-9"]["livedict"] == [] and len(by_id["fc-9"]["volume"]) == 1
    assert by_id["fc-1"]["last_heartbeat"] == 100.0
    # every source of a call's live log arrives together, each row saying which
    # it is: what the call logged, what the launcher did to it, and what the
    # container logged around it
    assert [(r["source"], r["msg"]) for r in by_id["fc-1"]["livedict"]] == [
        ("worker", "boot: lease held"), ("launcher", "granted"), ("ambient", "retrying"),
    ]
    assert [r["msg"] for r in by_id["fc-1"]["volume"]] == ["boot: lease held"]

    # A call that beat for a *different* artifact (fc-2, sources/tinyshakespeare)
    # never leaks into this one's aggregation.
    assert "fc-2" not in call_ids

    # An artifact no call has ever touched: empty, not an error.
    assert api.get("/logs/sources/never-run").json()["calls"] == []


def test_artifact_page_is_the_manifest_and_step_log_off_the_mount():
    """`/artifact/<path>` -- the leg's manifest (its own keys, the nested
    dependency manifests dropped) and step log as this container's image
    of the mount holds them, and nothing from a Dict."""
    api, _ = client()
    body = api.get("/artifact/runs/toy/pretraining").json()
    assert body == {
        "artifact_path": "runs/toy/pretraining",
        "manifest": MANIFEST,
        "train": [{"step": 1, "attempt": 1, "loss": 2.0, "grad_norm": 1.0, "learning_rate": 0.5}],
    }
    # An artifact with nothing on disk: empty, not an error.
    assert api.get("/artifact/sources/tinyshakespeare").json() == {"artifact_path": "sources/tinyshakespeare", "manifest": {}, "train": []}


def test_artifact_routes_reject_a_path_that_walks_out():
    api, _ = client()
    body = api.get("/artifact/%2e%2e/%2e%2e/etc/passwd").json()
    assert body["manifest"] == {} and body["train"] == [] and "invalid" in body["error"], body
    body = api.get("/logs/%2e%2e/%2e%2e/etc/passwd").json()
    assert body["calls"] == [] and "invalid" in body["error"], body


# --- driver ------------------------------------------------------------------


def run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            if "-v" in sys.argv:
                import traceback

                traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return len(failed)


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
