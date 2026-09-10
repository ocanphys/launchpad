"""Does every route `leasebook` is supposed to serve actually answer?

`import main` proves nothing about the web app: the routes are registered
inside `leasebook()`, which only runs when a container starts, and a handler's
body only runs when a request arrives. So a deleted route, a name that no
longer exists, or a handler that raises are all invisible to an import -- and
one deletion that took `/launch`, `/cancel`, `/manifest` and `/lab/run` with it
is why this file exists.

Nothing here talks to Modal. `leasebook`'s raw function is built against stubs:
`read_state` returns a fixed payload, the volume-reading helpers return fixed
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
    "now": 1757260800.0,
    "runs": {"toy": {"artifacts": {}, "notebook": True}},
    "problem_runs": {},
    "leases": {},
    "beats": {},
    "sources": {},
    "datasets": {},
    "metrics": {"read_state_seconds": 0.1},
}

SUMMARY = {
    "type": "artifacts.sources.Source",
    "parameters": {"name": "tinyshakespeare"},
    "depends_on": [{"artifact_path": "sources/other", "type": "Source"}],
}


def client(**overrides):
    """A TestClient over `leasebook`'s real ASGI app, with everything that
    would touch Modal or the volume replaced.

    The refresh thread is left running: it is a daemon, its first pass happens
    before any request, and `read_state` is a stub, so it costs a stubbed call
    per interval and proves the thread starts without raising.
    """
    from fastapi.testclient import TestClient

    calls = {"launched": [], "cancelled": []}

    main.read_state = lambda: STATE
    main.artifact_manifest_summary = lambda path, root: overrides.get("summary", SUMMARY)
    main.attempt_launch = lambda path: (
        calls["launched"].append(path) or (True, f"granted {path}", None)
    )
    main.cancel_call = lambda path: (
        calls["cancelled"].append(path) or (True, f"cancelled {path}")
    )
    main.jupyter = type("Stub", (), {"get_web_url": staticmethod(lambda: "https://lab.test")})()
    # A plain dict answers `.keys()` and `.get()` the same way the Dict does.
    main.call_logs = {"fc-1": ["2026-09-07T12:00:00.000Z INFO boot: lease held"]}
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
        "/lab/run/{run_id:path}",
        "/launch/{artifact_path:path}",
        "/cancel/{artifact_path:path}",
        "/manifest/{artifact_path:path}",
        "/logs",
        "/logs/{call_id}",
    ):
        assert expected in paths, f"{expected} is not registered: {sorted(paths)}"


def test_state_serves_what_the_thread_computed():
    api, _ = client()
    body = api.get("/state").json()
    assert body["runs"] == STATE["runs"]
    assert body["metrics"]["read_state_seconds"] == 0.1


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


def test_manifest_answers_with_the_summary():
    api, _ = client()
    body = api.get("/manifest/sources/tinyshakespeare").json()
    assert body["artifact_path"] == "sources/tinyshakespeare"
    assert body["type"] == SUMMARY["type"]
    assert body["depends_on"] == SUMMARY["depends_on"]


def test_an_unbuilt_artifact_says_so_rather_than_erroring():
    api, _ = client(summary=None)
    body = api.get("/manifest/sources/nothing").json()
    assert "not built yet" in body["error"], body


def test_the_lab_links_carry_the_token():
    api, _ = client()
    root = api.get("/lab")
    assert root.status_code == 307 and "lab.test" in root.headers["location"]
    run = api.get("/lab/run/toy")
    assert "runs/toy/notebook.ipynb" in run.headers["location"], run.headers
    # Encoded, for the reason the traversal test above spells out.
    bad = api.get("/lab/run/%2e%2e/secrets")
    assert bad.headers["location"] == "/lab", "an unsafe run_id should fall back"


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
