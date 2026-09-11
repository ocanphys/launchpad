"""Does every route `leasebook` is supposed to serve actually answer?

`import main` proves nothing about the web app: the routes are registered
inside `leasebook()`, which only runs when a container starts, and a handler's
body only runs when a request arrives. So a deleted route, a name that no
longer exists, or a handler that raises are all invisible to an import -- and
one deletion that took `/launch`, `/cancel` and `/manifest` with it
is why this file exists.

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

import sys
from pathlib import Path

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
    # is what /manifest hands back in their place.
    "sources/broken": {
        "type": None, "status": "conflict", "error": "not a readable manifest",
        "depends_on": [], "parameters": None,
        "blocked_by": [], "done": False, "ready": False, "call_id": None,
        "active": False, "last_heartbeat": None, "live_progress": None,
        "durable_progress": None,
    },
}


def client(**overrides):
    """A TestClient over `leasebook`'s real ASGI app, with everything that
    would touch Modal or the volume replaced.

    The refresh thread is left running: it is a daemon, its first pass happens
    before any request, and `state` is a stub, so it costs a stubbed call
    per interval and proves the thread starts without raising.
    """
    from fastapi.testclient import TestClient

    calls = {"launched": [], "cancelled": []}

    # `**_` because the refresh thread passes the `beats` snapshot it shares
    # with `calls_by_artifact`; this stub answers the same either way. It
    # accepts keywords only: a caller in main.py that passes `beat_records`
    # positionally breaks every test here, not just the one that exercises it.
    main.state = lambda **_: overrides.get("state", STATE)
    main.attempt_launch = lambda path: (
        calls["launched"].append(path) or (True, f"granted {path}", None)
    )
    main.cancel_call = lambda path: (
        calls["cancelled"].append(path) or (True, f"cancelled {path}")
    )
    main.jupyter = type("Stub", (), {"get_web_url": staticmethod(lambda: "https://lab.test")})()
    # A plain dict answers `.keys()`/`.get()`/`.items()` the same way the
    # Dict does.
    main.call_logs = {"fc-1": ["2026-09-07T12:00:00.000Z INFO boot: lease held"]}
    main.beats = {
        "fc-1": {"artifact_path": "runs/toy/pretraining", "last_beat_ts": 100.0},
        "fc-0": {"artifact_path": "runs/toy/pretraining", "last_beat_ts": 50.0},
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
        "/state",
        "/lab",
        "/launch/{artifact_path:path}",
        "/cancel/{artifact_path:path}",
        "/manifest/{artifact_path:path}",
        "/logs",
        "/logs/{call_id}",
        "/logs/artifact/{artifact_path:path}",
    ):
        assert expected in paths, f"{expected} is not registered: {sorted(paths)}"


def test_state_serves_what_the_thread_computed():
    api, _ = client()
    assert api.get("/state").json() == STATE


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
    assert api.get("/manifest/%2e%2e/etc/passwd").json()["error"] == "invalid artifact_path"
    assert calls["cancelled"] == [], "a bad path reached the canceller"


def test_manifest_answers_from_the_computed_state_not_the_mount():
    """`/manifest` is a lookup into what the refresh thread computed, not a
    read of the volume -- the read it used to do raced that thread's own
    `volume.reload()` and made an artifact's page flicker. `Path(STORAGE)`
    isn't mounted here, so a handler that reached for it would raise."""
    api, _ = client()
    body = api.get("/manifest/sources/tinyshakespeare").json()
    entry = STATE["sources/tinyshakespeare"]
    assert body["artifact_path"] == "sources/tinyshakespeare"
    assert body["type"] == entry["type"]
    assert body["parameters"] == entry["parameters"]
    # Dependencies are artifact paths, never nested manifests -- whatever else
    # is true of a dependency lives on its own entry in this same map.
    assert body["depends_on"] == ["sources/other"], body


def test_an_unbuilt_artifact_says_so_rather_than_erroring():
    api, _ = client()
    body = api.get("/manifest/sources/nothing").json()
    assert "not built yet" in body["error"], body


def test_an_unreadable_manifest_answers_with_its_reason():
    api, _ = client()
    body = api.get("/manifest/sources/broken").json()
    assert body["error"] == "not a readable manifest", body


def test_the_lab_link_carries_the_token():
    api, _ = client()
    root = api.get("/lab")
    assert root.status_code == 307 and "lab.test" in root.headers["location"]


def test_logs_come_from_the_dict_not_the_mount():
    """The log routes answer from `call_logs` -- see docs/LOGGING.md. They
    must never open a file on the mount this container reloads on a clock;
    with `call_logs` stubbed to a plain dict here, a route that reached for
    `Path(STORAGE)` would raise, since it isn't mounted."""
    api, _ = client()
    assert api.get("/logs").json() == {"call_ids": ["fc-1"]}
    body = api.get("/logs/fc-1").json()
    assert body["call_id"] == "fc-1" and len(body["lines"]) == 1, body
    assert api.get("/logs/fc-nope").json()["lines"] == []


def test_artifact_logs_aggregate_every_call_oldest_first():
    """`/logs/artifact/<path>` -- every call that has ever beaten for this
    one artifact, not just its current holder, ordered by last heartbeat.
    Reads only `beats`/`call_logs` (both stubbed to plain dicts here, same
    as `test_logs_come_from_the_dict_not_the_mount`), so this never touches
    the mount either, however many calls it aggregates.
    """
    api, _ = client()
    body = api.get("/logs/artifact/runs/toy/pretraining").json()
    assert body["artifact_path"] == "runs/toy/pretraining"
    call_ids = [call["call_id"] for call in body["calls"]]
    assert call_ids == ["fc-0", "fc-1"], body  # fc-0 beat first (ts=50), fc-1 later (ts=100)

    # fc-0 never appears in call_logs -- an empty list, not an error.
    by_id = {call["call_id"]: call for call in body["calls"]}
    assert by_id["fc-0"]["lines"] == []
    assert by_id["fc-1"]["lines"] == ["2026-09-07T12:00:00.000Z INFO boot: lease held"]

    # A call that beat for a *different* artifact (fc-2, sources/tinyshakespeare)
    # never leaks into this one's aggregation.
    assert "fc-2" not in call_ids

    # An artifact no call has ever touched: empty, not an error.
    assert api.get("/logs/artifact/sources/never-run").json()["calls"] == []


def test_artifact_logs_read_the_index_not_the_beats_dict():
    """`/logs/artifact/<path>` looks its calls up in what the refresh pass
    indexed, never scanning `beats` itself -- that scan was on the request
    path and grows with every call ever run (§7). A `beats` that raises on
    `.items()` proves it: the index was already built at startup, so the
    route still answers, while a handler that scanned would raise.
    """
    api, _ = client()
    fixture = main.beats

    class Scanned:
        def items(self):
            raise AssertionError("/logs/artifact scanned the beats Dict per request")

    main.beats = Scanned()  # the refresh thread swallows this and keeps its last pass
    try:
        body = api.get("/logs/artifact/runs/toy/pretraining").json()
        assert [call["call_id"] for call in body["calls"]] == ["fc-0", "fc-1"], body
    finally:
        main.beats = fixture


def test_artifact_logs_rejects_a_path_that_walks_out():
    api, _ = client()
    body = api.get("/logs/artifact/%2e%2e/%2e%2e/etc/passwd").json()
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
