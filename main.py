import time
from collections.abc import Callable
from pathlib import Path

import modal

import jobs
from config import (
    APP_NAME,
    VOLUME_NAME,
    STORAGE,
    CONTAINER_LIFETIME,
    HEARTBEAT_SECONDS,
    FLATLINE,
)
from lease_protocol import beats, leases, new_grant
from runtime import initialize_worker

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.12")

# The modules every call needs whatever it runs: config, its logger, its lease,
# the scope that wires those two together, the config.json schema, and the
# jobs themselves.
CALL_SOURCE = ("config", "logs", "lease_protocol", "runtime", "run_config", "jobs")

# Two images, because a job's dependencies are not every job's dependencies: a
# job that never runs snakemake should not wait for a container carrying it.
# Both descend from base_image rather than worker_image from etl_image --
# add_local_* must come last in a chain, so nothing can be pip-installed on top
# of an image that already has local files added.
#
# Both also need pydantic explicitly: it's not one of CALL_SOURCE's own local
# files, it's a dependency of one (run_config.py, imported by jobs.py) --
# add_local_python_source copies .py files into the image, it doesn't install
# what they import.
worker_image = base_image.pip_install("pydantic>=2.13.4").add_local_python_source(
    *CALL_SOURCE
)

etl_image = (
    base_image.pip_install("snakemake~=9.25", "pydantic>=2.13.4")
    .add_local_dir(
        local_path="etl", remote_path="/etl"
    )  # copy Snakefile to the container.
    # keep it lightweight: four .py files mounted at runtime, not the package or its deps.
    .add_local_python_source(*CALL_SOURCE)
)


def is_active(grant: dict | None, beat: dict | None, now: float) -> bool:
    """Is `grant`'s holder currently beating fresh enough to trust?

    Pure -- takes the grant and its holder's beat rather than fetching
    either, so `read_state` (a batch snapshot, one call_id's beat already
    looked up per run) and `launch_job` (a single-run lookup, no mount) both
    decide "active" the same way instead of each having its own rule.
    """
    if grant is None or beat is None:
        return False
    return now - beat["last_beat_ts"] < FLATLINE * HEARTBEAT_SECONDS


# A third image, for the same reason there are two: reading the leases and
# beats needs fastapi, and none of what a job needs -- no snakemake, no volume.
#
# It carries all of CALL_SOURCE even so, and not just the two modules the endpoint
# actually touches. A container starts by importing the module its function is
# defined in, and this module imports `runtime`, which imports `logs` -- so the
# web container needs every module main.py names at import time, whether or not
# serving a stream ever calls into it. Trimming this to what the endpoint reads
# gets a crash loop, not a smaller image.
WEB_DIR = Path("/web")  # where web_image mounts the web/ folder

web_image = (
    base_image.pip_install("fastapi[standard]")
    # The page is HTML/JS, so it stays HTML/JS -- mounted like the Snakefile
    # dir, not pasted into this module as strings.
    .add_local_dir(local_path="web", remote_path=WEB_DIR.as_posix())
    .add_local_python_source(*CALL_SOURCE)
)


def read_state() -> dict:
    """One read of the whole book, starting from the runs that exist rather
    than the leases that were granted.

    `runs` comes from the volume's `runs/` directory, not from `leases`: a
    folder with no lease -- never started, or superseded and never reclaimed --
    is exactly the gap worth being able to see, and starting from `leases`
    instead would hide it.

    `leases` and `beats` go back close to untouched: `leases` verbatim, `beats`
    with one field added per entry, `lease`, naming which run that call_id is
    the *current* holder for (None if it is not the current holder of anything
    -- a superseded container still beating, or one that never held a lease at
    all). Both reads come from the one snapshot taken here, so a beat's `lease`
    always agrees with what `runs` says that call holds.

    `jobs.preflight_check` raises rather than reporting "not ready" the same
    way as "broken" (see its own docstring) -- so this is the one place that
    decides what a raise means for a whole run: not `runs`, but
    `problem_runs`, keyed the same way, holding the error instead of a
    lease/heartbeat/jobs snapshot. One run_id's raise doesn't cost the rest
    of the book -- the loop below catches it per run_id, not around the
    whole loop.
    """
    volume.reload()
    runs_root = Path(STORAGE) / "runs"
    run_ids = (
        sorted(p.name for p in runs_root.iterdir() if p.is_dir())
        if runs_root.exists()
        else []
    )

    grants = dict(leases.items())
    beat_records = dict(beats.items())
    now = time.time()

    held_by = {grant["call_id"]: run_id for run_id, grant in grants.items()}

    runs = {}
    problem_runs = {}
    for run_id in run_ids:
        try:
            job_states = jobs.preflight_check(run_id)  # local mount
        except Exception as exc:
            problem_runs[run_id] = {"error": str(exc)}
            continue

        grant = grants.get(run_id)
        call_id = grant["call_id"] if grant else None
        beat = beat_records.get(call_id) if call_id else None
        runs[run_id] = {
            "lease": grant,
            "call_id": call_id,
            "job_type": grant["job_type"] if grant else None,
            "last_heartbeat": beat["last_beat_ts"] if beat else None,
            "active": is_active(grant, beat, now),
            "jobs": job_states,
        }

    beats_out = {
        call_id: {**beat, "lease": held_by.get(call_id)}
        for call_id, beat in beat_records.items()
    }

    return {
        "now": now,
        "runs": runs,
        "problem_runs": problem_runs,
        "beats": beats_out,
        "leases": grants,
    }


@app.function(image=web_image, volumes={STORAGE: volume}, max_containers=1)
@modal.asgi_app()
def leasebook():
    """The page and the data it polls, under one URL.

    An ASGI app rather than two `fastapi_endpoint`s because two endpoints are two
    URLs on two subdomains: the page would have to be told where its data lives,
    and the browser would treat the answer as cross-origin and refuse to read it.
    Served together, `state` is just a relative path, and no CORS question arises.
    """
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    @api.get("/state")
    def state() -> dict:
        return read_state()

    @api.post("/launch/{run_id}/{job}")
    def launch_endpoint(run_id: str, job: str) -> dict:
        launched, message, _call = attempt_launch(run_id, job)
        return {"launched": launched, "message": message}

    # Everything else -- index.html at "/" and its same-origin JS modules
    # (app.js, el.js, render.js) -- is a static file under WEB_DIR. Mounted
    # last: routes are matched in registration order, so /state and /launch
    # are claimed above before this catch-all sees them.
    api.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return api


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def run_job(run_id: str, job_uid: str) -> None:
    """Run any jobs.py job_uid under this call's own logger and lease. Same
    shape as `etl`: confirm, do the work, confirm, commit.

    job_uid is passed straight through as `job_type`, not read off this
    function's own `.info.function_name` the way `job0`/`job1` used to:
    that name is `"run_job"` for every call now, where it used to equal the
    job_uid only because the Modal function and the job class happened to
    share a name. The log file's name, the lease-mismatch message, and the
    dashboard's job-type column all key off `job_type`, so it's threaded
    through explicitly here to keep all three unchanged.

    job_cls is resolved here, from job_uid itself (`jobs.get_job_class`) --
    no registry in this file to keep in sync with jobs.py. job_uid IS the
    class name (`RunConfig` already validated that this run's config.json
    only ever declares job_uids that resolve, when it was written), so
    this is a re-check against jobs.py as currently deployed, not a first
    one.
    """
    job_cls = jobs.get_job_class(job_uid)
    if job_cls is None:
        raise jobs.JobError(f"{job_uid}: no such class in jobs.py")
    with initialize_worker(run_id, job_type=job_uid, volume=volume) as worker:
        worker.confirm_lease()
        job = job_cls(run_id, worker.log, worker)
        job.run()
        worker.confirm_lease()
        volume.commit()


def resource_options(resources: dict) -> dict:
    """A job's config `resources` -> `Function.with_options()` kwargs.

        resource_options({}) == {}
        resource_options({"cpu": 2}) == {"cpu": 2}
        resource_options({"gpu_type": "A100"}) == {"gpu": "A100"}
        resource_options({"gpu_type": "A100", "gpu_count": 2}) == {"gpu": "A100:2"}

    Unvalidated by design -- config.json is only ever written by the
    trusted `push_config.py` (or a human editing it by hand), and Modal
    itself rejects a bad cpu/gpu value server-side at spawn time. Three
    interpretations worth being explicit about instead of leaving to
    accident:

    - `"cpu": null` counts as not declared, same as the key being absent.
      Calling `with_options()` at all, even with every argument at its own
      default, still moves the call into its own dynamically configured
      container pool (see `attempt_launch`), and a no-op override is
      exactly the case not worth paying that for.
    - `gpu_count` with no `gpu_type` is dropped -- nothing to attach a
      count to, so the job runs GPU-less rather than erroring.
    - `gpu_count: 0` reads the same as omitted -- Modal's own "TYPE:COUNT"
      string has no way to ask for zero of a GPU.
    """
    options = {}
    if resources.get("cpu") is not None:
        options["cpu"] = resources["cpu"]
    gpu_type = resources.get("gpu_type")
    if gpu_type:
        gpu_count = resources.get("gpu_count")
        options["gpu"] = f"{gpu_type}:{gpu_count}" if gpu_count else gpu_type
    return options


def attempt_launch(
    run_id: str, job: str
) -> tuple[bool, str, modal.functions.FunctionCall | None]:
    """Grant `run_id` to one call of `job`, or refuse and say why.

    Shared by `launch_job` (a local entrypoint, which waits on the call) and
    the web `/launch` route (which cannot wait -- a request has to return) so
    the checks and the grant/spawn itself are decided in one place, not
    twice: `run_id`/`job` are untrusted here in a way they weren't for a
    CLI-only launcher, since a POST route is reachable by anyone with the
    URL, so both are validated before touching a lease or a path built from
    either.

    Resources are never a parameter -- not here, not in `launch_job`, not in
    the web route. They come only from `run_id`'s own config.json, read the
    same way `state["ready"]` already is a few lines below, and turned into
    `Function.with_options()` kwargs by `resource_options`. That call is
    skipped entirely when a job_uid declares none (`options` empty): calling
    it unconditionally would move every launch into its own dynamically
    configured container pool, separate even from another call with the
    same empty options, so a job_uid that asks for nothing special stays
    pooled on `run_job`'s own base configuration.

    `job` is a job_uid, not looked up against a registry here -- `RunConfig`
    already validated, when this run's config.json was written, that every
    job_uid it declares resolves to a real class in jobs.py. So the only
    question left is whether `run_id`'s config declares *this* job_uid at
    all, which `preflight_check` (below) answers by returning None for one
    it's never heard of.
    """
    if not run_id or "/" in run_id or run_id in (".", ".."):
        return False, f"invalid run_id {run_id!r}", None

    grant = leases.get(run_id)
    beat = beats.get(grant["call_id"]) if grant else None
    if is_active(grant, beat, time.time()):
        return (
            False,
            f"{job}: run {run_id} already has an active call -- not launching",
            None,
        )

    state = jobs.preflight_check(run_id, volume=volume).get(job)
    if state is None:
        return (
            False,
            f"{job}: no config.json (or not declared) for run {run_id} -- nothing to launch",
            None,
        )
    if not state["ready"]:
        return (
            False,
            f"{job}: blocked on {state['missing_dependencies']} for run {run_id}",
            None,
        )

    leases.pop(run_id, None)
    options = resource_options(state["resources"])
    fn = run_job.with_options(**options) if options else run_job
    call = fn.spawn(run_id, job)
    leases.put(run_id, new_grant(call.object_id, job))
    return True, f"granted lease for {run_id} -> {call.object_id}", call


@app.local_entrypoint()
def launch_job(run_id: str, job: str):
    """Grant one run to one job call, then wait for it -- same shape as
    `launch`. The decision itself is `attempt_launch`'s.
    """
    launched, message, call = attempt_launch(run_id, job)
    print(message)
    if launched:
        print("waiting for it to finish")
        call.get()
