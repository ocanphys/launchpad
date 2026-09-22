import json
import logging
import os
import queue
import subprocess
import threading
import time
from contextlib import asynccontextmanager
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
    PERSIST_LOGS_EVERY,
    REFRESH_WAIT_SECONDS,
    REGION,
    STARTUP_GRACE_SECONDS,
    STORAGE,
    TRAIN_LOG,
    VOLUME_NAME,
    get_git_commit,
)
from system.lease_protocol import (
    beats,
    call_history,
    call_logs,
    leases,
    new_grant,
    refreshes,
)
from system.logs import (
    CHANNELS,
    LAUNCHER_LOG_KEY,
    LAUNCHER_VOLUME_KEY,
    launcher_log,
    load_snapshot_from_volume,
    open_json,
    open_jsonl,
    save_snapshot_to_volume,
    start_launcher_logging,
)
from system.runtime import initialize_worker

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# The launcher's own log: what the leasebook container does, published as
# `call_logs["launcher"]` for the dashboard's panel once
# `start_launcher_logging` has run (system/logs.py).
log = logging.getLogger("leasebook")

base_image = modal.Image.debian_slim(python_version="3.12").pip_install("regex", "tqdm")

# groups of python packages from this repo - these are added to images which specify
# the python environments of the containers.
CALL_SOURCE = ("config", "system")
# models/ rides along wherever manifests are decoded: a leg's model_parameters
# are rebuilt from the model package it names, torch-free at import
ARTIFACTS_SOURCE = ("artifacts", "models")


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
    """Whether a leased call is still working. A call marks its last beat
    `exited` on the way out, so the answer is no from that moment rather than
    a flatline later."""
    if grant is None or beat is None or beat.get("exited"):
        return False
    return now - beat["last_beat_ts"] < FLATLINE * HEARTBEAT_SECONDS


def is_starting(grant: dict | None, beat: dict | None, now: float) -> bool:
    """Whether a leased call is still within its grace period for a first beat."""
    if grant is None or beat is not None:
        return False
    granted_ts = grant.get("granted_ts")
    return granted_ts is not None and now - granted_ts < STARTUP_GRACE_SECONDS


WEB_DIR = Path("/web")  # where web_image mounts the web/ folder
web_image = (
    base_image.pip_install("fastapi[standard]")
    .add_local_dir(local_path="web", remote_path=WEB_DIR.as_posix())
    .add_local_python_source(*CALL_SOURCE, *ARTIFACTS_SOURCE)
)

worker_image = base_image.pip_install(
    "numpy", "torch", "einops"
).add_local_python_source(*CALL_SOURCE, *ARTIFACTS_SOURCE)

lab_image = (
    base_image.pip_install("numpy", "torch", "einops", "matplotlib", "jupyterlab")
    .env({LAB_COMMIT_ENV: get_git_commit() if modal.is_local() else "unknown"})
    .add_local_python_source(*CALL_SOURCE, *ARTIFACTS_SOURCE, "lab")
)


# What the manifest at each artifact path decoded to, for this process's
# life. A definition at a path is immutable once declared (spec.md), and the
# map shows nothing else from the manifest, so a path read once is never read
# again here; what launching needs fresh (resources) `attempt_launch` reads
# off the volume itself.
resolved: dict[str, Artifact] = {}

# Held by anything in this process that reloads the mount or holds a file on
# it open: `state`, `attempt_launch`, and leasebook's `/artifact` route. A
# reload replaces the mount's view of the volume and refuses to run while
# this process has a file under it open, so the listener thread's refresh and
# a request reading a manifest have to take turns (LESSONS.md).
mount_lock = threading.Lock()


def state(root: Path = Path(STORAGE)) -> dict[str, dict]:
    """The current state of every declared artifact under `root`, keyed by
    artifact path: one reload, one glob, one status apiece, one lease and
    heartbeat snapshot for the whole scan, and a manifest read only the
    first time this process sees its path (`resolved`).

        state()["runs/toy/pretraining"] -> {
            "type": "Pretraining", "status": "partial", "error": None,
            "depends_on": ["mappeddatasets/mapped-7f3c1a2b"], "blocked_by": [],
            "parameters": {"run_id": "toy", "config": {...}},
            "done": False, "ready": True, "verdict": "running",
            "call_id": "fc-01JQ8W", "active": True, "last_heartbeat": 1757260800.1,
            "live_progress": {"step": 120, "end_step": 500},
            "durable_progress": {"phase": "step", "done": 100, "total": 500},
        }

    A manifest that cannot be read is an entry with status "conflict" and
    its `error`, and the scan continues. `blocked_by` is every direct
    dependency not `done` on this same map, one with no manifest included;
    `ready` is additionally false for an artifact nothing produces.

    `verdict` is the one word the row shows, first match wins: "done";
    "running" for an active call; "starting" for a lease younger than
    STARTUP_GRACE_SECONDS whose call has not beaten yet; "failed" for a
    manifest that would not load or a lease whose call stopped beating or
    exhausted its startup grace period; "runnable" when ready;
    "blocked" otherwise. A failed artifact stays `ready`, so it can be run
    again. Every fact is from the one snapshot, so a lease granted after it
    is not on the map until the next refresh.

    Holds `mount_lock` for the reload and the scan, so a caller can be kept
    waiting by whatever else in this process is reading the mount.
    """
    # `now` before the snapshot: the reload and the Dict scans take a
    # measurable time, and taking the clock after them would charge that
    # time to every beat's age. A beat written during the snapshot is
    # newer than `now` and reads as live, which is the true answer.
    now = time.time()
    with mount_lock:
        if root == Path(STORAGE):
            volume.reload()
        beat_records = dict(beats.items())
        grants = dict(leases.items())

        # Subtrees the manifests read this scan share, decoded once: a run's
        # legs each embed every leg before them. Gone with the scan; `resolved`
        # keeps the artifacts.
        memo: dict[str, Artifact] = {}
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
                "parameters": None,
                "call_id": grant["call_id"] if grant else None,
                "active": active,
                "last_heartbeat": beat["last_beat_ts"] if beat else None,
                "live_progress": beat.get("progress") if beat and active else None,
                "durable_progress": None,
            }
            try:
                if path not in resolved:
                    resolved[path] = Artifact.load(path, root, memo)
            except ValueError as error:
                entry["error"] = str(error)
            else:
                artifact = resolved[path]
                present = list(artifact.status(root).completion.values())
                entry.update(
                    type=type(artifact).__name__,
                    status="done" if all(present) else "partial" if any(present) else "declared",
                    depends_on=[dep.artifact_path.as_posix() for dep in artifact.deps()],
                    # This artifact's own fields, never a dependency's manifest. A
                    # dependency is a path in `depends_on` and nothing more: its own
                    # entry on this same map is where anything else about it lives,
                    # so nothing is stored twice and nothing here goes stale when it
                    # changes.
                    parameters=artifact.parameters(),
                    durable_progress=artifact.durable_progress(root),
                )
            entries[path] = entry

    for path, entry in entries.items():
        starting = is_starting(grants.get(path), beat_records.get(entry["call_id"]), now)
        blocked_by = [
            dep for dep in entry["depends_on"] if entries.get(dep, {}).get("status") != "done"
        ]
        done = entry["status"] == "done"
        ready = path in resolved and resolved[path].producer is not None and not done and not blocked_by
        entry["blocked_by"] = blocked_by
        entry["done"] = done
        entry["ready"] = ready
        entry["verdict"] = (
            "done" if done
            else "running" if entry["active"]
            else "starting" if starting
            else "failed" if entry["status"] == "conflict" or entry["call_id"]
            else "runnable" if ready
            else "blocked"
        )
    return entries


def artifact_calls(artifact_path: str) -> list[dict]:
    """Every call ever granted for `artifact_path`, oldest grant first, each
    with the last beat it left and its three log channels (`launcher`,
    `container`, `volume`), read from the Dicts now. A call that never ran
    has `last_heartbeat` None; a grant `load_snapshot_from_volume` made up
    for a log file has `granted_ts` None and sorts first.

    Bounded by this one artifact's history: one `beats.get` and three
    `call_logs.get`s per call, never a scan of either Dict.
    """
    return [
        {
            "call_id": grant["call_id"],
            "granted_ts": grant["granted_ts"],
            "last_heartbeat": (beats.get(grant["call_id"]) or {}).get("last_beat_ts"),
            **{
                channel: call_logs.get(f"{grant['call_id']}:{channel}") or []
                for channel in (*CHANNELS, "volume")
            },
        }
        for grant in call_history.get(artifact_path) or []
    ]


@app.function(
    image=web_image,
    volumes={STORAGE: volume},
    max_containers=1,
    region=REGION,
    # Only for the lab's token, so `/lab` can hand it straight to Jupyter --
    # nothing else in here reads a secret.
    secrets=[modal.Secret.from_name(LAB_SECRET)],
)
@modal.asgi_app()
def leasebook():
    """The page and the data it fetches, under one URL.

    An ASGI app rather than two `fastapi_endpoint`s because two endpoints are two
    URLs on two subdomains: the page would have to be told where its data lives,
    and the browser would treat the answer as cross-origin and refuse to read it.
    Served together, `state` is just a relative path, and no CORS question arises.
    """
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    # Everything below is this container's whole setup and its whole life --
    # the log is publishing before anything touches the volume, and one try
    # around all of it, so anything that goes wrong anywhere in here (building
    # the app, any request any route handles) reaches that log instead of
    # vanishing into a container crash or a bare 500 with nothing on record.
    stop_logging = None
    try:
        stop_logging = start_launcher_logging()
        log.info("leasebook container started")

        # Set on the way out, for the listener thread below.
        finished = threading.Event()

        @asynccontextmanager
        async def lifespan(_api):
            yield
            log.info("leasebook container stopped")
            finished.set()
            stop_logging()

        api = FastAPI(lifespan=lifespan)

        @api.middleware("http")
        async def log_request_errors(request, call_next):
            try:
                return await call_next(request)
            except Exception:
                log.exception(f"{request.method} {request.url.path} failed")
                raise

        # The files on the volume and the Dicts reconciled once, so a
        # dashboard that comes back after any gap lists what was filed.
        root = Path(STORAGE)
        volume.reload()
        persisted = load_snapshot_from_volume(root)
        log.info(f"synced {sum(key.endswith(':container') for key in persisted)} call logs off the volume")

        # This container's picture of the volume: the `state` map, and the
        # authority on it -- lagged, but complete, since an artifact's files
        # never leave the volume once they land. Recomputed when a worker's
        # message says its call has started or committed, when this container
        # itself grants or releases a lease, and when the page asks. No
        # clock: a volume nothing wrote to is a volume worth no reload.
        # Everything a request reads off the mount in between is the last
        # recompute's image of it.
        latest: dict[str, dict] = {}

        def recompute(reason: str) -> None:
            nonlocal latest
            latest = state()
            log.debug(f"refreshed ({reason}): {len(latest)} artifacts on the volume")

        recompute("startup")

        # Blocks on the queue until a call starts or exits, then recomputes
        # once for everything that landed together -- the whole reason the
        # dashboard needs no clock. The wait is bounded only so the thread
        # notices its container going away. Nothing gets to end this loop: a
        # read or a reload that failed costs one refresh, and the dashboard
        # would stop keeping up for the rest of the container's life if the
        # thread died of it.
        def listen() -> None:
            while not finished.is_set():
                try:
                    messages = refreshes.get_many(100, timeout=REFRESH_WAIT_SECONDS)
                    for message in messages:
                        log.info(f"{message['artifact_path']}: {message['call_id']} {message['event']}")
                    recompute("worker")
                except queue.Empty:
                    continue
                except Exception:
                    log.exception("refresh listener: this refresh is lost; still listening")
                    finished.wait(REFRESH_WAIT_SECONDS)

        threading.Thread(target=listen, daemon=True).start()

        # The manual reload: a manifest declared from the lab lands on the
        # volume without a call, so nothing tells this container about it.
        @api.post("/refresh")
        def refresh_route() -> dict:
            recompute("page")
            return {"artifacts": len(latest)}

        # The one state route for all three table views: one flat map of
        # every artifact on the volume, which the frontend slices per view
        # locally instead of round-tripping on every nav click.
        @api.get("/state")
        def state_route() -> dict:
            return latest

        # The launcher's own log, both channels as the Dict holds them now
        # (what this container has published, and the file's rows), which
        # the page unions like a call's. Dict reads only, so the dashboard
        # polls it.
        @api.get("/launcher-logs")
        def launcher_logs_endpoint() -> dict:
            return {
                "launcher": call_logs.get(LAUNCHER_LOG_KEY) or [],
                "volume": call_logs.get(LAUNCHER_VOLUME_KEY) or [],
            }

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
        # Both of these change a lease, which the map is computed from, so an
        # accepted one recomputes before answering: the page's next fetch of
        # /state shows what it just asked for, with nothing to poll for.
        @api.post("/launch/{artifact_path:path}")
        def launch_endpoint(artifact_path: str) -> dict:
            launched, message = attempt_launch(artifact_path)
            if launched:
                recompute("launch")
            return {"launched": launched, "message": message}

        # Stop whatever call is working on this artifact -- what the row's
        # button offers while it runs, and the only way to end a job from the
        # dashboard.
        @api.post("/cancel/{artifact_path:path}")
        def cancel_endpoint(artifact_path: str) -> dict:
            if not safe_relpath(artifact_path):
                return {"cancelled": False, "message": f"invalid artifact_path {artifact_path!r}"}
            cancelled, message = cancel_call(artifact_path)
            if cancelled:
                recompute("cancel")
            return {"cancelled": cancelled, "message": message}

        # One artifact as this container's disk holds it, off the image the
        # last recompute reloaded: its manifest minus the dependency manifests
        # nested in it (a dependency is a path in the state map's
        # `depends_on`, and its own page is where it lives), and its step
        # log if its job writes one. Only ever what a worker committed under
        # its lease; nothing streams an attempt's rows before that. Under
        # `mount_lock`, as every open file on the mount is: a recompute
        # reloading underneath these two descriptors is what the lock exists
        # to prevent. A sync def, so the lock is held on the thread that
        # takes it -- do not make this one async.
        @api.get("/artifact/{artifact_path:path}")
        def artifact_endpoint(artifact_path: str) -> dict:
            if not safe_relpath(artifact_path):
                return {"artifact_path": artifact_path, "manifest": {}, "train": [], "error": "invalid artifact_path"}
            folder = root / artifact_path
            with mount_lock, open_json(folder / MANIFEST) as manifest, open_jsonl(folder / TRAIN_LOG) as rows:
                return {
                    "artifact_path": artifact_path,
                    "manifest": {key: value for key, value in manifest.items() if key != "dependencies"},
                    "train": list(rows),
                }

        # Every call ever granted for this one artifact, each with all three
        # channels of its log as the Dicts hold them now. Dict reads only,
        # never the mount, so the page polls this one while it is open.
        @api.get("/logs/{artifact_path:path}")
        def logs_endpoint(artifact_path: str) -> dict:
            if not safe_relpath(artifact_path):
                return {"artifact_path": artifact_path, "calls": [], "error": "invalid artifact_path"}
            return {"artifact_path": artifact_path, "calls": artifact_calls(artifact_path)}

        # TEMPORARY: any of the Dicts whole, for inspection. Remove once
        # the logging rework has been checked on a real deployment.
        dicts = {"call_logs": call_logs, "call_history": call_history, "beats": beats, "leases": leases}

        @api.get("/debug/{name}")
        def dict_dump(name: str) -> dict:
            if name not in dicts:
                return {"error": f"no Dict {name!r}; one of {sorted(dicts)}"}
            return dict(dicts[name].items())

        # Everything else -- index.html at "/" and its same-origin JS modules
        # (app.js, el.js, render.js) -- is a static file under WEB_DIR. Mounted
        # last: routes are matched in registration order, so /state and /launch
        # are claimed above before this catch-all sees them.
        api.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

        return api
    except Exception:
        log.exception("leasebook failed to start")
        if stop_logging:
            stop_logging()
        raise


@app.function(image=web_image, volumes={STORAGE: volume}, region=REGION, schedule=modal.Period(seconds=PERSIST_LOGS_EVERY))
def persist_logs() -> None:
    """Appends every call's new log rows to its file on the volume, writes
    `call_history.json` and commits: the only writer of either.

    Stateless, in its own container: each pass rebuilds the row-count cursor
    from the files themselves, so a pass that never ran costs nothing but
    lag. Fires only on a deployed app (`modal deploy`), not under `modal serve`.
    """
    root = Path(STORAGE)
    volume.reload()
    save_snapshot_to_volume(root, volume, load_snapshot_from_volume(root))


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def run_job(artifact_path: str) -> None:
    """Load the artifact at `artifact_path`, run whatever produces it, under
    this call's own logger and lease.

    The producing job is resolved here, and only here -- `artifact.job()`,
    called right before it's needed rather than anywhere upstream of this.
    Nothing about launching (`attempt_launch`) has to know it either.
    """
    with initialize_worker(artifact_path, volume) as worker:
        worker.confirm_lease("pre run")
        artifact = Artifact.load(artifact_path, Path(STORAGE))
        job = artifact.job()
        job.run(Path(STORAGE), worker)
        worker.confirm_lease("pre vol commit")


@app.function(
    image=lab_image,
    volumes={STORAGE: volume},
    secrets=[modal.Secret.from_name(LAB_SECRET)],
    # One lab, so one filesystem view: two containers would each hold their own
    # uncommitted copy of the same notebook and the later commit would win.
    max_containers=1,
    region=REGION,
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
    commits are enough -- no reason to force one. A person who wants one now
    calls `lab.save()` from a cell; `lab.refresh()` is the reload.
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


def attempt_launch(artifact_path: str, root: Path = Path(STORAGE)) -> tuple[bool, str]:
    """Grant `artifact_path` to one call of `run_job`, or refuse and say why.

    What the web `/launch` route does, which cannot wait on the call -- a
    request has to return. `artifact_path` is untrusted, since a POST route
    is reachable by anyone with the URL, so it's validated before touching a
    lease or a path built from it.

    Readiness is read off the volume as it is now, after a reload, under
    `mount_lock`: the manifest at the path, and the manifest and completion
    files of each direct dependency -- the same facts `state()` derives
    `ready` from, for this one artifact instead of every one on the volume.
    The dashboard's map is as old as its last recompute, so a launch judged on
    it could refuse an artifact whose dependency has since finished.

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

    Every step lands in the launcher's log, so a launch that refuses says
    where it got to.
    """

    def refuse(message: str) -> tuple[bool, str]:
        log.info(f"launch {artifact_path}: refused -- {message}")
        return False, message

    log.debug(f"launch {artifact_path}: requested")
    if not safe_relpath(artifact_path):
        return refuse(f"invalid artifact_path {artifact_path!r}")

    grant = leases.get(artifact_path)
    beat = beats.get(grant["call_id"]) if grant else None
    now = time.time()
    if is_active(grant, beat, now) or is_starting(grant, beat, now):
        return refuse(f"{artifact_path}: already has a call under way ({grant['call_id']}) -- not launching")

    log.debug(f"launch {artifact_path}: no call under way; reading the volume")
    with mount_lock:
        if root == Path(STORAGE):
            volume.reload()
        try:
            artifact = Artifact.load(artifact_path, root)
        except (FileNotFoundError, ValueError) as error:
            return refuse(str(error))
        if artifact.producer is None:
            return refuse(f"{artifact_path}: nothing produces it -- done when its dependencies are")
        if all(artifact.status(root).completion.values()):
            return refuse(f"{artifact_path}: already done")
        blocked_by = [
            dep.artifact_path.as_posix()
            for dep in artifact.deps()
            if not (footprint := dep.status(root)).manifest or not all(footprint.completion.values())
        ]
    if blocked_by:
        return refuse(f"{artifact_path}: blocked on {blocked_by}")

    if grant:
        log.info(f"launch {artifact_path}: dropping the stale lease of {grant['call_id']}")
    leases.pop(artifact_path, None)
    options = resource_options(artifact.allocated_resources)
    log.debug(f"launch {artifact_path}: asking run_job to spawn" + (f" with {options}" if options else ""))
    fn = run_job.with_options(**options) if options else run_job
    call = fn.spawn(artifact_path)
    grant = new_grant(call.object_id, type(artifact).__name__)
    leases.put(artifact_path, grant)
    # The launcher's own record of the call, before the container has said
    # anything: what lists it under the artifact and what its log starts
    # with, so a call whose container never runs is still a call that was made.
    call_history.put(artifact_path, [*(call_history.get(artifact_path) or []), grant])
    launcher_log(call.object_id, f"launch requested for {artifact_path}" + (f" with {options}" if options else ""), "DEBUG")
    message = f"spawned {artifact_path} -> {call.object_id}, lease granted"
    launcher_log(call.object_id, message)

    return True, message


def cancel_call(artifact_path: str) -> tuple[bool, str]:
    """Stop the call currently holding `artifact_path`, or say there wasn't one.

    The lease is released first: releasing it is what a worker between
    checkpoints notices (`Lease.confirm` raises on the next one) even if the
    cancel itself never lands, and a lease left behind on a killed call reads
    as one that went stale -- red on the dashboard, for something that was
    stopped on purpose.

    A cancel request can arrive after the call is over. What the call's own
    output says once the request has been sent is what the logged outcome
    says, so a request that came too late never reads as a stopped job.
    """
    grant = leases.pop(artifact_path, None)
    if grant is None:
        log.info(f"cancel {artifact_path}: refused -- no active call to cancel")
        return False, f"{artifact_path}: no active call to cancel"
    log.debug(f"cancel {artifact_path}: lease released; requesting cancel of {grant['call_id']}")
    launcher_log(grant["call_id"], f"cancel requested for {artifact_path}; lease released", "DEBUG")
    call = modal.FunctionCall.from_id(grant["call_id"])
    call.cancel()
    try:
        call.get(timeout=0)
        outcome = "the request came too late -- the call had already finished and its result stands"
    except TimeoutError:  # no output yet: the call was live, so the request is what ends it
        outcome = "cancelled -- the call had no result when the request landed"
    except Exception as error:
        outcome = f"the request came after the call ended: {type(error).__name__}: {error}"
    message = f"{artifact_path} -> {grant['call_id']}: {outcome}"
    launcher_log(grant["call_id"], message)
    return True, message

