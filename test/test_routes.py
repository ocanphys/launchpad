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
import sys
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main

# --- the stubs ---------------------------------------------------------------

STATE = {
    "sources/tinyshakespeare": {
        "type": "Source", "status": "done", "error": None,
        "depends_on": ["sources/other"], "parameters": {"name": "tinyshakespeare"},
        "blocked_by": [], "done": True, "ready": False, "call_id": None,
        "active": False, "last_heartbeat": None, "live_progress": None,
        "durable_progress": {"phase": "files", "done": 1, "total": 1},
    },
    # A manifest `state` couldn't read: no parameters to show, and the reason
    # is what the artifact's page shows in their place.
    "sources/broken": {
        "type": None, "status": "conflict", "error": "not a readable manifest",
        "depends_on": [], "parameters": None,
        "blocked_by": [], "done": False, "ready": False, "call_id": None,
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

    def state():
        calls["state"] += 1
        return overrides.get("state", STATE)

    main.state = state
    main.attempt_launch = lambda path: (
        calls["launched"].append(path) or (True, f"granted {path}", None)
    )
    main.cancel_call = lambda path: (
        calls["cancelled"].append(path) or (True, f"cancelled {path}")
    )
    main.jupyter = type("Stub", (), {"get_web_url": staticmethod(lambda: "https://lab.test")})()
    # The startup sync: nothing to reload, no log files to read.
    main.volume = type("Stub", (), {"reload": staticmethod(lambda: None)})()
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
    ROW = {"ts": 1757246400.0, "level": "INFO", "logger": "job", "msg": "boot: lease held"}
    GRANTED = {"ts": 90.0, "level": "INFO", "logger": "launcher", "msg": "granted"}
    main.call_logs = {"fc-1:launcher": [GRANTED], "fc-1:container": [ROW], "fc-1:volume": [ROW], "fc-9:volume": [ROW]}
    # The mount: a folder with one leg's manifest and step log on it.
    STEP = {"step": 1, "attempt": 1, "loss": 2.0, "grad_norm": 1.0, "learning_rate": 0.5}
    main.STORAGE = mkdtemp()
    leg = Path(main.STORAGE) / "runs/toy/pretraining"
    leg.mkdir(parents=True)
    (leg / "manifest.json").write_text(json.dumps({**MANIFEST, "dependencies": {"dataset": {"artifact": "artifacts.dataset.DataSet"}}}))
    (leg / "train.jsonl").write_text(json.dumps(STEP) + "\n")
    main.beats = {
        "fc-1": {"artifact_path": "runs/toy/pretraining", "last_beat_ts": 100.0},
        "fc-2": {"artifact_path": "sources/tinyshakespeare", "last_beat_ts": 75.0},
    }
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


def test_state_is_computed_at_startup_and_on_refresh_only():
    """`/state` serves the map the container holds; only POST /refresh
    computes a new one. The logs, by contrast, are read from the Dicts on
    every request: a row that lands in `call_logs` shows up without one."""
    api, calls = client()
    assert calls["state"] == 1
    api.get("/state")
    api.get("/artifact/runs/toy/pretraining")
    api.get("/logs/runs/toy/pretraining")
    assert calls["state"] == 1
    assert api.post("/refresh").json() == {"artifacts": len(STATE)}
    assert calls["state"] == 2

    main.call_logs["fc-1:container"] = [*main.call_logs["fc-1:container"], {"ts": 101.0, "level": "INFO", "logger": "job", "msg": "step 1"}]
    body = api.get("/logs/runs/toy/pretraining").json()
    assert [row["msg"] for row in body["calls"][2]["container"]] == ["boot: lease held", "step 1"]


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
    beat it left and all three channels of its log, out of the Dicts.
    """
    api, _ = client()
    body = api.get("/logs/runs/toy/pretraining").json()
    assert body["artifact_path"] == "runs/toy/pretraining"
    call_ids = [call["call_id"] for call in body["calls"]]
    assert call_ids == ["fc-9", "fc-0", "fc-1"], body

    by_id = {call["call_id"]: call for call in body["calls"]}
    assert by_id["fc-9"]["last_heartbeat"] is None and by_id["fc-9"]["granted_ts"] is None
    # fc-0 was granted and never ran: listed, with nothing in any channel.
    assert by_id["fc-0"]["last_heartbeat"] is None
    assert (by_id["fc-0"]["launcher"], by_id["fc-0"]["container"], by_id["fc-0"]["volume"]) == ([], [], [])
    assert by_id["fc-9"]["container"] == [] and len(by_id["fc-9"]["volume"]) == 1
    assert by_id["fc-1"]["last_heartbeat"] == 100.0
    assert by_id["fc-1"]["launcher"][0]["msg"] == "granted"
    assert by_id["fc-1"]["container"][0]["msg"] == "boot: lease held"
    assert by_id["fc-1"]["volume"] == by_id["fc-1"]["container"]

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
