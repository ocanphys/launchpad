import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

from artifacts.core.artifact import MANIFEST, Artifact, Resources
from config import (
    APP_NAME,
    CONTAINER_LIFETIME,
    FLATLINE,
    HEARTBEAT_SECONDS,
    LAB_COMMIT_ENV,
    LAB_IDLE_SECONDS,
    LAB_PORT,
    LAB_SECRET,
    STATE_REFRESH_SECONDS,
    STORAGE,
    VOLUME_NAME,
    get_git_commit,
)
from system.lease_protocol import beats, call_logs, leases, new_grant
from system.logs import setup_logging
from system.runtime import initialize_worker

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.12").pip_install("regex", "tqdm")

# groups of python packages from this repo - these are added to images which specify
# the python environments of the containers.
CALL_SOURCE = ("config", "system")
ARTIFACTS_SOURCE = ("artifacts",)


def safe_relpath(path: str) -> bool:
    """Is `path` safe to join under STORAGE -- non-empty, not absolute, no
    `..` component that could walk it outside the volume. Every route that
    takes a path from the URL (an artifact_path or a run_id) checks this
    before it ever reaches `Path(STORAGE) / path`.

    safe_relpath("runs/toy/pretraining") -> True
    safe_relpath("../../etc/passwd") -> False
    """
    return bool(path) and not Path(path).is_absolute() and ".." not in Path(path).parts


def is_active(grant: dict | None, beat: dict | None, now: float) -> bool:
    if grant is None or beat is None:
        return False
    return now - beat["last_beat_ts"] < FLATLINE * HEARTBEAT_SECONDS

WEB_DIR = Path("/web")  # where web_image mounts the web/ folder
web_image = (
    base_image.pip_install("fastapi[standard]")
    .add_local_dir(local_path="web", remote_path=WEB_DIR.as_posix())
    .add_local_python_source(*CALL_SOURCE, *ARTIFACTS_SOURCE)
)

worker_image = base_image.pip_install("numpy").add_local_python_source(
    *CALL_SOURCE, *ARTIFACTS_SOURCE,
)

lab_image = (
    base_image.pip_install("numpy", "torch", "einops", "matplotlib", "jupyterlab")
    .env({LAB_COMMIT_ENV: get_git_commit() if modal.is_local() else "unknown"})
    .add_local_python_source(*CALL_SOURCE, *ARTIFACTS_SOURCE, "lab")
)


def state(root: Path = Path(STORAGE)) -> dict[str, dict]:
    """The current state of every declared artifact under `root`, keyed by
    artifact path: one glob, one manifest read and one status apiece, and one
    lease and heartbeat snapshot for the whole scan.

        state()["runs/toy/pretraining"] -> {
            "type": "Pretraining", "status": "partial", "error": None,
            "depends_on": ["mappeddatasets/mapped-7f3c1a2b"], "blocked_by": [],
            "done": False, "ready": True,
            "call_id": "fc-01JQ8W", "active": True, "last_heartbeat": 1757260800.1,
            "live_progress": {"step": 120, "total_steps": 500},
            "durable_progress": {"step": 100, "total_steps": 500},
        }

    A manifest that cannot be read is an entry with status "conflict" and
    its `error`, and the scan continues. `blocked_by` is every direct
    dependency not `done` on this same map, one with no manifest included;
    `ready` is additionally false for an artifact nothing produces.
    """
    if root == Path(STORAGE):
        volume.reload()
    grants, beat_records, now = dict(leases.items()), dict(beats.items()), time.time()

    loaded: dict[str, Artifact] = {}
    entries: dict[str, dict] = {}
    for manifest in sorted(root.rglob(MANIFEST)):
        path = manifest.parent.relative_to(root).as_posix()
        grant = grants.get(path)
        beat = beat_records.get(grant["call_id"]) if grant else None
        active = is_active(grant, beat, now)
        entry = {
            "type": None,
            "status": "conflict",
            "error": None,
            "depends_on": [],
            "call_id": grant["call_id"] if grant else None,
            "active": active,
            "last_heartbeat": beat["last_beat_ts"] if beat else None,
            "live_progress": beat.get("progress") if beat and active else None,
            "durable_progress": None,
        }
        try:
            artifact = Artifact.load(path, root)
        except ValueError as error:
            entry["error"] = str(error)
        else:
            present = list(artifact.status(root).completion.values())
            entry.update(
                type=type(artifact).__name__,
                status="done" if all(present) else "partial" if any(present) else "declared",
                depends_on=[dep.artifact_path.as_posix() for dep in artifact.deps()],
                durable_progress=artifact.durable_progress(root),
            )
            loaded[path] = artifact
        entries[path] = entry

    for path, entry in entries.items():
        blocked_by = [
            dep for dep in entry["depends_on"] if entries.get(dep, {}).get("status") != "done"
        ]
        done = entry["status"] == "done"
        entry["blocked_by"] = blocked_by
        entry["done"] = done
        entry["ready"] = (
            path in loaded and loaded[path].producer is not None and not done and not blocked_by
        )
    return entries


def artifact_manifest_summary(artifact_path: str, root: Path) -> dict | None:
    """The artifact declared at `artifact_path`, reduced to what a drill-down
    page wants: its own type and parameters (`to_manifest()` already keeps
    these separate from its dependencies) plus one line per direct dependency
    naming where it lives. Each dependency is named, not inlined -- `deps()`
    hands back the live Artifact objects, not their nested manifests, and a
    link to that artifact's own page is all the summary needs.

    None if there's no manifest yet -- declared but not built is a normal
    state, not an error.
    """
    manifest_path = root / artifact_path / MANIFEST
    if not manifest_path.exists():
        return None
    artifact = Artifact.load(artifact_path, root)
    manifest = artifact.to_manifest()
    return {
        "type": manifest["artifact"],
        "parameters": manifest["parameters"],
        "depends_on": [
            {"artifact_path": dep.artifact_path.as_posix(), "type": type(dep).__name__}
            for dep in artifact.deps()
        ],
    }


@app.function(
    image=web_image,
    volumes={STORAGE: volume},
    max_containers=1,
    # Only for the lab's token, so `/lab` can hand it straight to Jupyter --
    # nothing else in here reads a secret.
    secrets=[modal.Secret.from_name(LAB_SECRET)],
)
@modal.asgi_app()
def leasebook():
    """The page and the data it polls, under one URL.

    An ASGI app rather than two `fastapi_endpoint`s because two endpoints are two
    URLs on two subdomains: the page would have to be told where its data lives,
    and the browser would treat the answer as cross-origin and refuse to read it.
    Served together, `state` is just a relative path, and no CORS question arises.
    """
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    # Everything below is this container's whole setup and its whole life --
    # one try around all of it, so anything that goes wrong anywhere in here
    # (building the app, the background thread, any request any route
    # handles) reaches this same log instead of vanishing into a container
    # crash or a bare 500 with nothing on record.
    try:
        api = FastAPI()

        setup_logging()  # stdout, where `modal app logs leasebook` picks it up
        log = logging.getLogger("leasebook")
        log.info("leasebook container started")

        # One thread, one job: keep this container's picture of the volume
        # fresh. `state()` is all of it -- one `volume.reload()`, one glob of
        # the manifests, one lease/beat snapshot, the whole map computed here
        # rather than in whatever request happens to arrive next. It writes
        # nothing and syncs nothing, so no request ever waits on it and it
        # never has to be told what someone is looking at.
        latest_state = {"value": state()}

        def refresh_state() -> None:
            while True:
                time.sleep(STATE_REFRESH_SECONDS)
                try:
                    computed = state()
                except Exception:
                    # This thread is the dashboard's only clock; letting it die
                    # would freeze /state with nothing saying why. The last
                    # good state stays up and the next pass tries again.
                    log.exception("state refresh failed, keeping last good state")
                    continue
                # One assignment of one key, which the GIL makes atomic: a
                # reader gets the whole previous state or the whole new one,
                # never a mix, so there is nothing here for a lock to protect.
                latest_state["value"] = computed

        threading.Thread(target=refresh_state, daemon=True).start()

        # The one state route for all three table views: one flat map of
        # every artifact on the volume, which the frontend slices per view
        # locally instead of round-tripping on every nav click. Touches
        # nothing itself: it hands back whatever the thread above last
        # computed, so it is never slower than a dict lookup and never
        # fresher than the last pass.
        @api.get("/state")
        def state_route() -> dict:
            return latest_state["value"]

        # The header's "lab" link. A redirect rather than a URL the page fetches:
        # the lab lives on its own subdomain, and the token that gets it past the
        # login screen is this container's to hold, not something to hand to the
        # browser as data and then hope it isn't logged. app.js never learns either
        # -- the anchor in index.html is a plain relative href.
        @api.get("/lab")
        def lab_redirect() -> RedirectResponse:
            url = jupyter.get_web_url()
            token = os.environ.get("JUPYTER_TOKEN")
            return RedirectResponse(f"{url}/lab?token={token}" if token else url)

        # :path, not a plain path segment -- an artifact_path contains its own
        # /s (runs/my-run/pretraining), which a plain segment can't match.
        @api.post("/launch/{artifact_path:path}")
        def launch_endpoint(artifact_path: str) -> dict:
            launched, message, _call = attempt_launch(artifact_path)
            return {"launched": launched, "message": message}

        # Stop whatever call is working on this artifact -- what the row's
        # button offers while it runs, and the only way to end a job from the
        # dashboard.
        @api.post("/cancel/{artifact_path:path}")
        def cancel_endpoint(artifact_path: str) -> dict:
            if not safe_relpath(artifact_path):
                return {"cancelled": False, "message": f"invalid artifact_path {artifact_path!r}"}
            cancelled, message = cancel_call(artifact_path)
            log.info(f"cancel {artifact_path}: {message}")
            return {"cancelled": cancelled, "message": message}

        # The `call_logs` Dict, straight through: which calls have a log, and
        # one call's lines. A Dict read, not a file -- this container reloads
        # the mount on a clock and opens no log file on it (docs/LOGGING.md).
        @api.get("/logs")
        def logs_index() -> dict:
            return {"call_ids": sorted(call_logs.keys())}

        @api.get("/logs/{call_id}")
        def logs_endpoint(call_id: str) -> dict:
            return {"call_id": call_id, "lines": call_logs.get(call_id) or []}

        # What an artifact's own page shows: type, own parameters, and
        # dependency links -- see artifact_manifest_summary. The one request
        # here that reads the mount, and one small file is all it reads.
        @api.get("/manifest/{artifact_path:path}")
        def manifest_endpoint(artifact_path: str) -> dict:
            if not safe_relpath(artifact_path):
                return {"artifact_path": artifact_path, "error": "invalid artifact_path"}
            try:
                summary = artifact_manifest_summary(artifact_path, Path(STORAGE))
            except Exception as exc:
                return {"artifact_path": artifact_path, "error": str(exc)}
            if summary is None:
                return {"artifact_path": artifact_path, "error": "not built yet -- no manifest"}
            return {"artifact_path": artifact_path, **summary}

        # Everything else -- index.html at "/" and its same-origin JS modules
        # (app.js, el.js, render.js) -- is a static file under WEB_DIR. Mounted
        # last: routes are matched in registration order, so /state and /launch
        # are claimed above before this catch-all sees them.
        api.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

        return api
    except Exception:
        # getLogger rather than `log`: this also catches a failure that happened
        # before that line ever ran.
        logging.getLogger("leasebook").exception("leasebook failed to start")
        raise


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def run_job(artifact_path: str) -> None:
    """Load the artifact at `artifact_path`, run whatever produces it, under
    this call's own logger and lease.

    The producing job is resolved here, and only here -- `artifact.job()`
    (the same lookup dependency resolution already uses), called right
    before it's needed rather than anywhere upstream of this. Nothing about
    launching (`attempt_launch`) has to know it either.
    """
    with initialize_worker(artifact_path, volume) as worker:
        worker.confirm_lease("pre run")
        artifact = Artifact.load(artifact_path, Path(STORAGE))
        job = artifact.job()
        job.run(Path(STORAGE), worker)
        worker.confirm_lease("pre vol commit")


@app.function(image=worker_image, volumes={STORAGE: volume})
def declared_artifact(artifact_path: str) -> tuple[Artifact, dict] | None:
    """The artifact declared at `artifact_path` and its current status, or
    None if nothing's declared there.

    Needs the volume actually mounted (`volume.reload()`, then plain `Path`
    reads against it) -- pulled out of `attempt_launch` and into its own
    function so that work always happens inside a container, regardless of
    whether `attempt_launch` itself was called from the web route (already
    inside one) or `launch_job` (a local entrypoint, which is never inside
    one -- `Path(STORAGE)` isn't a real mount on the machine running `modal
    run`, and volume.reload() outright refuses to run there).
    """
    entry = state().get(artifact_path)  # reloads the volume itself
    if entry is None or entry["error"]:
        return None
    return Artifact.load(artifact_path, Path(STORAGE)), entry


@app.function(
    image=lab_image,
    volumes={STORAGE: volume},
    secrets=[modal.Secret.from_name(LAB_SECRET)],
    # One lab, so one filesystem view: two containers would each hold their own
    # uncommitted copy of the same notebook and the later commit would win.
    max_containers=1,
    timeout=CONTAINER_LIFETIME,
    scaledown_window=LAB_IDLE_SECONDS,
)
# A notebook UI is a browser holding a websocket open and firing many requests
# in parallel -- without this, each one is a separate input and max_containers=1
# serializes the whole session into a queue.
@modal.concurrent(max_inputs=100)
@modal.web_server(LAB_PORT, startup_timeout=120)
def jupyter():
    """JupyterLab, rooted at the volume.

    Named `jupyter`, not `lab`: `lab` is the module a notebook imports (lab.py)
    and `/lab` is the dashboard route that redirects here, and a Modal function
    object called `lab` in this module's namespace would shadow the first and
    read as the second.

    `root_dir` is the volume itself, not a notebooks folder -- pointing at an
    artifact path is the whole point, and the file browser is where you find
    out what the paths are (`runs/`, `sources/`, `tokenizers/`).

    No `--IdentityProvider.token`: jupyter_server reads JUPYTER_TOKEN out of
    the environment, which LAB_SECRET puts there, and keeping it out of the
    command line keeps it out of every `ps` and traceback.

    No explicit `volume.commit()` here or on a background clock: every Volume
    mount already sets `allow_background_commits=True`, so the platform
    flushes writes on its own, and JupyterLab's own autosave writes a
    notebook to disk on its own clock too. Unlike `run_job`, nothing is
    waiting on a precise moment to see the lab's writes, so background
    commits are enough -- no reason to force one.
    """
    # Default-on autocomplete and editor niceties: ONE overrides.json, in
    # JupyterLab's application settings directory -- not the per-user
    # settings tree (~/.jupyter/lab/user-settings/), which holds one
    # <plugin-id>.jupyterlab-settings file per plugin (what the Settings
    # Editor writes when a person changes something by hand), and isn't
    # where overrides.json is read from. Keyed by full plugin id, verified
    # against JupyterLab's own schemas rather than guessed:
    #   - completer-extension:manager's `autoCompletion` (default false) is
    #     what actually turns "press Tab to see suggestions" into
    #     suggestions appearing as you type.
    #   - codemirror-extension:plugin's `defaultConfig` is an open object of
    #     CodeMirror editor options (autoClosingBrackets, lineNumbers, ...),
    #     applied to every editor -- notebook cells included, so there's no
    #     separate notebook-extension setting needed for these.
    # Deliberately not attempting "open the contextual-help/inspector panel
    # by default" here: inspector-extension's own schema declares zero
    # properties (`additionalProperties: false`, nothing in between) --
    # there is no settings key for it. That needs a pre-built default
    # *workspace* (layout state, a different mechanism from settings
    # overrides entirely), not attempted here.
    from jupyterlab.commands import get_app_dir

    settings_dir = Path(get_app_dir()) / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "overrides.json").write_text(
        json.dumps(
            {
                "@jupyterlab/completer-extension:manager": {"autoCompletion": True},
                "@jupyterlab/codemirror-extension:plugin": {
                    "defaultConfig": {
                        "autoClosingBrackets": True,
                        "lineNumbers": True,
                        "codeFolding": True,
                    }
                },
            },
            indent=2,
        )
    )

    subprocess.Popen(
        [
            "jupyter",
            "lab",
            "--ip=0.0.0.0",
            f"--port={LAB_PORT}",
            "--no-browser",
            "--allow-root",
            f"--ServerApp.root_dir={STORAGE}",
            # Modal terminates TLS and forwards to this container under a
            # different host than the browser typed, which jupyter_server reads
            # as a remote/cross-origin access and blocks by default.
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.allow_origin=*",
        ]
    )


def resource_options(resources: Resources) -> dict:
    """An artifact's `allocated_resources` -> `Function.with_options()` kwargs.

        resource_options(Resources()) == {}
        resource_options(Resources(cpu=2)) == {"cpu": 2}
        resource_options(Resources(gpu_type="A100")) == {"gpu": "A100"}
        resource_options(Resources(gpu_type="A100", gpu_count=2)) == {"gpu": "A100:2"}

    Unvalidated by design -- `Resources` is only ever constructed by trusted
    code (a notebook declaring an artifact), and Modal itself rejects a bad
    cpu/gpu value server-side at spawn time. Three interpretations worth
    being explicit about instead of leaving to accident:

    - `cpu=None` counts as not declared, same as the field being absent
      entirely. Calling `with_options()` at all, even with every argument at
      its own default, still moves the call into its own dynamically
      configured container pool (see `attempt_launch`), and a no-op override
      is exactly the case not worth paying that for.
    - `gpu_count` with no `gpu_type` is dropped -- nothing to attach a
      count to, so the job runs GPU-less rather than erroring.
    - `gpu_count: 0` reads the same as omitted -- Modal's own "TYPE:COUNT"
      string has no way to ask for zero of a GPU.
    """
    options = {}
    if resources.cpu is not None:
        options["cpu"] = resources.cpu
    if resources.gpu_type:
        options["gpu"] = (
            f"{resources.gpu_type}:{resources.gpu_count}"
            if resources.gpu_count
            else resources.gpu_type
        )
    return options


def attempt_launch(
    artifact_path: str,
) -> tuple[bool, str, modal.functions.FunctionCall | None]:
    """Grant `artifact_path` to one call of `run_job`, or refuse and say why.

    Shared by `launch_job` (a local entrypoint, which waits on the call) and
    the web `/launch` route (which cannot wait -- a request has to return) so
    the checks and the grant/spawn itself are decided in one place, not
    twice: `artifact_path` is untrusted here in a way it wasn't for a
    CLI-only launcher, since a POST route is reachable by anyone with the
    URL, so it's validated before touching a lease or a path built from it.

    Readiness comes from `declared_artifact` (a remote call -- see its own
    docstring for why this can't just read `Path(STORAGE)` here directly),
    one lookup in the same `state()` map the dashboard shows, so "is this
    safe to launch" and "what does the dashboard show" never have two
    different answers.

    Resources come from the artifact's own `allocated_resources`, turned into
    `Function.with_options()` kwargs by `resource_options`. That call is
    skipped entirely when an artifact declares none (`options` empty): calling
    it unconditionally would move every launch into its own dynamically
    configured container pool, separate even from another call with the same
    empty options, so an artifact that asks for nothing special stays pooled
    on `run_job`'s own base configuration.

    The producing job is never looked up here -- readiness only needs the
    artifact and its declared dependencies' statuses, not what runs it.
    `run_job` resolves the job itself, right before running it.
    """
    if not safe_relpath(artifact_path):
        return False, f"invalid artifact_path {artifact_path!r}", None

    grant = leases.get(artifact_path)
    beat = beats.get(grant["call_id"]) if grant else None
    if is_active(grant, beat, time.time()):
        return False, f"{artifact_path}: already has an active call -- not launching", None

    declared = declared_artifact.remote(artifact_path)
    if declared is None:
        return False, f"{artifact_path}: not declared -- nothing to launch", None
    artifact, state = declared
    if not state["ready"]:
        return False, f"{artifact_path}: blocked on {state['blocked_by']}", None

    leases.pop(artifact_path, None)
    options = resource_options(artifact.allocated_resources)
    fn = run_job.with_options(**options) if options else run_job
    call = fn.spawn(artifact_path)
    leases.put(artifact_path, new_grant(call.object_id, type(artifact).__name__))

    return True, f"granted lease for {artifact_path} -> {call.object_id}", call


def cancel_call(artifact_path: str) -> tuple[bool, str]:
    """Stop the call currently holding `artifact_path`, or say there wasn't one.

    The lease is released first: releasing it is what a worker between
    checkpoints notices (`Lease.confirm` raises on the next one) even if the
    cancel itself never lands, and a lease left behind on a killed call reads
    as one that went stale -- red on the dashboard, for something that was
    stopped on purpose.

    """
    grant = leases.pop(artifact_path, None)
    if grant is None:
        return False, f"{artifact_path}: no active call to cancel"
    modal.FunctionCall.from_id(grant["call_id"]).cancel()
    return True, f"cancelled {artifact_path} -> {grant['call_id']}"


@app.local_entrypoint()
def launch_job(artifact_path: str):
    """Grant one artifact to one job call, then wait for it -- same shape as
    `launch`. The decision itself is `attempt_launch`'s.
    """
    launched, message, call = attempt_launch(artifact_path)
    print(message)
    if launched:
        print("waiting for it to finish")
        call.get()
