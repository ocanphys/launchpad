"""The launcher and its dashboard: one container, one module.

`attempt_launch` and `cancel_call` are the write side of the lease protocol,
and the two routes that call them are the only callers there are. What one of
them changes is a lease, and so the map `launcher.state` computes from it.

The web app is an ASGI app rather than two `fastapi_endpoint`s because two
endpoints are two URLs on two subdomains: the page would have to be told where
its data lives, and the browser would treat the answer as cross-origin and
refuse to read it. Served together, `state` is just a relative path, and no
CORS question arises.
"""

import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import modal
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from artifacts.core.artifact import MANIFEST, Artifact, Resources
from config import REFRESH_BATCH, REFRESH_WAIT_SECONDS, STORAGE, TRAIN_LOG
from launcher.gate import gate
from launcher.state import (
    artifact_calls,
    is_active,
    is_starting,
    liveness,
    mount_lock,
    safe_relpath,
    state,
    volume,
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
    LAUNCHER,
    LIVE,
    VOLUME,
    launcher_logger,
    load_snapshot_from_volume,
    open_json,
    open_jsonl,
    start_logging,
    stream,
)

log_launcher = launcher_logger()  # this container's doings
log_call = launcher_logger  # a call this container manages


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


def attempt_launch(artifact_path: str, run_job, root: Path = STORAGE) -> tuple[bool, str]:
    """Grant `artifact_path` to one call of `run_job`, or refuse and say why.

    What the web `/launch` route does, which cannot wait on the call -- a
    request has to return. `artifact_path` is untrusted, since a POST route
    is reachable by anyone with the URL, so it's validated before touching a
    lease or a path built from it.

    `run_job` is the Modal function to spawn, handed in because its decorator
    is in main.py, which imports this module.

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

    Every step is logged, and the ones naming a call are stamped with it: the
    call under way for a refusal that names one, the new call from the spawn
    on, so its page opens on the grant that made it. All of it is in the
    launcher's own log either way.
    """

    def refuse(message: str, call_id: str = LAUNCHER) -> tuple[bool, str]:
        log_call(call_id).info(f"launch {artifact_path}: refused -- {message}")
        return False, message

    log_launcher.debug(f"launch {artifact_path}: requested")
    if not safe_relpath(artifact_path):
        return refuse(f"invalid artifact_path {artifact_path!r}")

    grant = leases.get(artifact_path)
    beat = beats.get(grant["call_id"]) if grant else None
    now = time.time()
    if is_active(grant, beat, now) or is_starting(grant, beat, now):
        # The call under way is the one this concerns: a relaunch attempted
        # while it works reads on its page, beside what it was doing.
        return refuse(f"{artifact_path}: already a call running - not launching", grant["call_id"])

    log_launcher.debug(f"launch {artifact_path}: no current call for artifact; preflight check from volume")
    with mount_lock:
        if root == STORAGE:
            volume.reload()
        try:
            artifact = Artifact.load(artifact_path, root)
        except (FileNotFoundError, ValueError) as error:
            return refuse(str(error))
        if artifact.producer is None:
            return refuse(f"{artifact_path}: nothing produces it -- done when its dependencies are")
        if artifact.status(root).complete:
            return refuse(f"{artifact_path}: already done")
        blocked_by = [
            dep.artifact_path.as_posix()
            for dep in artifact.deps()
            if not (footprint := dep.status(root)).manifest or not footprint.complete
        ]
    if blocked_by:
        return refuse(f"{artifact_path}: blocked on {blocked_by}")

    if grant:
        log_call(grant["call_id"]).info(f"{artifact_path}: stale lease dropped by a new launch")
    leases.pop(artifact_path, None)
    options = resource_options(artifact.allocated_resources)
    fn = run_job.with_options(**options) if options else run_job
    call = fn.spawn(artifact_path)
    grant = new_grant(call.object_id, type(artifact).__name__)
    leases.put(artifact_path, grant)
    # The launcher's own record of the call, before the container has said
    # anything: what lists it under the artifact and what its log starts
    # with, so a call whose container never runs is still a call that was made.
    call_history.put(artifact_path, [*(call_history.get(artifact_path) or []), grant])
    log_call(call.object_id).debug(f"spawn requested for {artifact_path}" + (f" with {options}" if options else ""))
    message = f"spawned {artifact_path} -> {call.object_id}, lease granted"
    log_call(call.object_id).info(message)

    return True, message


def cancel_call(artifact_path: str) -> tuple[bool, str]:
    """Stop the call currently holding `artifact_path`, or say there wasn't one.

    The lease goes first, then the request to Modal. Dropping the lease is the
    stop that arrives immediately: the call's next `Lease.confirm` finds no
    grant and raises, and every job takes one inside `worker.publishing`,
    between writing a file and renaming it into place. So a cancel lands before
    the artifact can complete even though Modal delivers its own cancellation
    on the container's heartbeat, seconds later.

    Until that container notices, two calls can be writing this artifact: the
    lease is gone, so a launch made in the meantime is granted. Both write into
    `worker.publishing`'s own temporary file, which carries the call id, and
    only the one holding the lease renames.

    A cancel request can arrive after the call is over. What the call's own
    output says once the request has been sent is what the logged outcome
    says, so a request that came too late never reads as a stopped job.
    """
    grant = leases.pop(artifact_path, None)
    if grant is None:
        log_launcher.info(f"cancel {artifact_path}: refused - no active call to cancel")
        return False, f"{artifact_path}: no active call to cancel"
    log_call(grant["call_id"]).debug(f"cancel requested for {artifact_path}; lease released")
    call = modal.FunctionCall.from_id(grant["call_id"])
    call.cancel()
    try:
        call.get(timeout=0)
        outcome = "late request - call had already finished"
    except TimeoutError:  # no output yet: the call was live, so the request is what ends it
        outcome = "cancelled"
    except Exception as error:
        outcome = f"the request came after the call ended: {type(error).__name__}: {error}"
    message = f"{artifact_path} - {grant['call_id']}: {outcome}"
    log_call(grant["call_id"]).info(message)
    return True, message


def build_api(run_job, jupyter, web_dir: Path) -> FastAPI:
    """The whole web app: every route, and the listener thread behind them.

    `run_job` is the Modal function `/launch` spawns and `jupyter` the one
    `/lab` redirects to, both handed in because their decorators are in
    main.py. `web_dir` is where that image mounted the `web/` folder.
    """
    # Everything below is this container's whole setup and its whole life --
    # the log is publishing before anything touches the volume, and one try
    # around all of it, so anything that goes wrong anywhere in here (building
    # the app, any request any route handles) reaches that log instead of
    # vanishing into a container crash or a bare 500 with nothing on record.
    stop_logging = None
    try:
        stop_logging = start_logging(LAUNCHER)
        log_launcher.info("leasebook container started")

        # Set on the way out, for the listener thread below.
        finished = threading.Event()

        @asynccontextmanager
        async def lifespan(_api):
            yield
            log_launcher.info("leasebook container stopped")
            finished.set()
            stop_logging()

        api = FastAPI(lifespan=lifespan)

        @api.middleware("http")
        async def log_request_errors(request, call_next):
            try:
                return await call_next(request)
            except Exception:
                log_launcher.exception(f"{request.method} {request.url.path} failed")
                raise

        # Everything past here is behind the password, `/login` included in
        # the sense that it is the way past it. Registered after the error
        # logger, which puts the gate outside it: Starlette wraps in reverse
        # registration order.
        gate(api, web_dir / "login.html")

        # The token `/lab` carries, read once. Not the dashboard's password:
        # it travels in a URL (see launcher/gate.py).
        token = os.environ["JUPYTER_TOKEN"]

        # The files on the volume and the Dicts reconciled once, so a
        # dashboard that comes back after any gap lists what was filed.
        root = STORAGE
        volume.reload()
        log_launcher.info(f"synced {len(load_snapshot_from_volume(root))} log channels off the volume")

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
            log_launcher.debug(f"refreshed ({reason}): {len(latest)} artifacts on the volume")

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
                    messages = refreshes.get_many(REFRESH_BATCH, timeout=REFRESH_WAIT_SECONDS)
                    for message in messages:
                        # The call's row, not the container's: what a call did
                        # reads on that call's page.
                        log_call(message["call_id"]).info(f"{message['artifact_path']}: {message['event']}")
                    recompute("worker")
                except queue.Empty:
                    continue
                except Exception:
                    log_launcher.exception("refresh listener: this refresh is lost; still listening")
                    finished.wait(REFRESH_WAIT_SECONDS)

        threading.Thread(target=listen, daemon=True).start()

        # The manual reload: a manifest declared from the lab lands on the
        # volume without a call, so nothing tells this container about it.
        @api.post("/refresh")
        def refresh_route() -> dict:
            recompute("page")
            return {"artifacts": len(latest)}

        # One flat map of every artifact on the volume, as the last recompute
        # left it, with the `liveness` of every call that recompute found
        # under way read off the Dicts again here. Durable facts are as old as
        # that recompute; live ones are as new as this request, which is what
        # moves a running job's progress and heartbeat on the page without
        # anything reloading the volume.
        #
        # Bounded by the calls under way, never a scan of either Dict: an
        # artifact nothing is working on cannot acquire a lease except
        # through this container, which recomputes when it grants one.
        @api.get("/state")
        def state_route() -> dict:
            now = time.time()
            entries = {}
            for path, entry in latest.items():
                if entry["verdict"] in ("running", "starting"):
                    grant = leases.get(path)
                    beat = beats.get(grant["call_id"]) if grant else None
                    # A call that has said it exited already put its ending on
                    # the queue, so the recompute that reads its files is on
                    # the way. Reading that beat here would call a finished
                    # run "failed" for the moment in between.
                    if not (beat or {}).get("exited"):
                        entry = {**entry, **liveness(entry, grant, beat, now)}
                entries[path] = entry
            return entries

        # The launcher's own log: the call id `launcher` in both storages, what
        # this container has published and what its file holds, which the page
        # unions like a call's. Its own rows and the noise around them share
        # the storage, told apart by `source`. Dict reads only, so the
        # dashboard polls it.
        @api.get("/launcher-logs")
        def launcher_logs_endpoint() -> dict:
            return {LIVE: stream(LAUNCHER, LIVE), VOLUME: stream(LAUNCHER, VOLUME)}

        # The header's "lab" link. A redirect rather than a URL the page fetches:
        # the lab lives on its own subdomain, and the token that gets it past the
        # login screen is this container's to hold, not something to hand to the
        # browser as data and then hope it isn't logged. app.js never learns either
        # -- the anchor in index.html is a plain relative href.
        @api.get("/lab")
        def lab_redirect() -> RedirectResponse:
            return RedirectResponse(f"{jupyter.get_web_url()}/lab?token={token}")

        # :path, not a plain path segment -- an artifact_path contains its own
        # /s (runs/my-run/pretraining), which a plain segment can't match.
        # Both of these change a lease, which the map is computed from, so an
        # accepted one recomputes before answering: the page's next fetch of
        # /state shows what it just asked for, with nothing to poll for.
        @api.post("/launch/{artifact_path:path}")
        def launch_endpoint(artifact_path: str) -> dict:
            launched, message = attempt_launch(artifact_path, run_job)
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

        # Every call ever granted for this one artifact, each with its log in
        # both storages as the Dicts hold them now. Dict reads only, never the
        # mount, so the page polls this one while it is open.
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
        # (app.js, el.js, render.js) -- is a static file under web_dir. Mounted
        # last: routes are matched in registration order, so /state and /launch
        # are claimed above before this catch-all sees them.
        api.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

        return api
    except Exception:
        log_launcher.exception("leasebook failed to start")
        if stop_logging:
            stop_logging()
        raise
