
import time
from collections.abc import Callable
from pathlib import Path

import modal

import jobs
from config import APP_NAME, VOLUME_NAME, STORAGE, CONTAINER_LIFETIME, HEARTBEAT_SECONDS
from lease_protocol import beats, leases, new_grant
from runtime import initialize_worker

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.12")

# The modules every call needs whatever it runs: config, its logger, its lease,
# the scope that wires those two together, and the jobs themselves.
CALL_SOURCE = ("config", "logs", "lease_protocol", "runtime", "jobs")

# Two images, because a job's dependencies are not every job's dependencies: a
# job that never runs snakemake should not wait for a container carrying it.
# Both descend from base_image rather than worker_image from etl_image --
# add_local_* must come last in a chain, so nothing can be pip-installed on top
# of an image that already has local files added.
worker_image = base_image.add_local_python_source(*CALL_SOURCE)

etl_image = (
    base_image.pip_install("snakemake~=9.25")
    .add_local_dir(local_path="etl", remote_path="/etl") #copy Snakefile to the container.
    #keep it lightweight: four .py files mounted at runtime, not the package or its deps.
    .add_local_python_source(*CALL_SOURCE)
)

WEB_DIR = Path("/web")  # where web_image mounts the web/ folder

FLATLINE = 5  # if heartbeat age is longer than this many HEARTBEAT_SECONDS, the call is not active.


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
    """
    volume.reload()
    runs_root = Path(STORAGE) / "runs"
    run_ids = sorted(p.name for p in runs_root.iterdir() if p.is_dir()) if runs_root.exists() else []

    grants = dict(leases.items())
    beat_records = dict(beats.items())
    now = time.time()

    held_by = {grant["call_id"]: run_id for run_id, grant in grants.items()}

    runs = {}
    for run_id in run_ids:
        grant = grants.get(run_id)
        call_id = grant["call_id"] if grant else None
        beat = beat_records.get(call_id) if call_id else None
        runs[run_id] = {
            "lease": grant,
            "call_id": call_id,
            "job_type": grant["job_type"] if grant else None,
            "last_heartbeat": beat["last_beat_ts"] if beat else None,
            "active": is_active(grant, beat, now),
            "jobs": jobs.preflight_check(run_id), #get this from local mount
        }

    beats_out = {call_id: {**beat, "lease": held_by.get(call_id)} for call_id, beat in beat_records.items()}

    return {
        "now": now,
        "runs": runs,
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



RUN_DIR = Path(STORAGE) / "data"



@app.function(image=etl_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def etl(run_id: str, cores:int) -> None:
    """Run any job under this call's own logger and lease.

    The job is the only thing passed in. The logger and the `confirm` it gets are
    built here, from the call id this container was actually given -- which is why
    they cannot be handed down from the launcher: the launcher does not know the
    call id until the call exists.
    """
    # The same string the grant carries, read off the same object: inside the
    # container `etl` is the Function, not this def.
    with initialize_worker(run_id, job_type=etl.info.function_name, volume=volume) as worker:
        import subprocess
        worker.confirm_lease()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "snakemake",
                "--snakefile",
                "/etl/Snakefile", #point to Snakefile in the container.
                "--directory",
                str(RUN_DIR),
                "--cores",
                str(cores),
            ],
            check=True,
        )
        time.sleep(20)
        worker.confirm_lease()
        volume.commit()


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def job0(run_id: str) -> None:
    """Run jobs.job0 under this call's own logger and lease. Same shape as
    `etl`: confirm, do the work, confirm, commit."""
    with initialize_worker(run_id, job_type=job0.info.function_name, volume=volume) as worker:
        worker.confirm_lease()
        job = jobs.job0(run_id, worker.log, worker.confirm_lease)
        job.run()
        worker.confirm_lease()
        volume.commit()


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def job1(run_id: str) -> None:
    """Run jobs.job1 under this call's own logger and lease. Same shape as
    `etl`: confirm, do the work, confirm, commit."""
    with initialize_worker(run_id, job_type=job1.info.function_name, volume=volume) as worker:
        worker.confirm_lease()
        job = jobs.job1(run_id, worker.log, worker.confirm_lease)
        job.run()
        worker.confirm_lease()
        volume.commit()


@app.local_entrypoint()
def launch(run_id: str):
    """Grant one run to one call, then stay up until that call is finished.

    `run_id` is required. It defaulted to None, and nothing rejected that: the
    grant went to the key "lease:None" and the container built a path under a
    directory named None.

    Two orderings here are load-bearing.

    The stale key goes *before* the spawn. The call id does not exist until the
    spawn returns, so the grant cannot be written first -- but a container that
    boots fast enough to read the previous run's grant under this same run_id
    finds a call id that is not its own, and a mismatch is fatal on sight. Clearing
    the key first turns that into the silence `fence` already knows how to wait
    out, which is the difference between a run that starts late and one that dies.

    The wait at the end is not politeness. `modal run` builds an ephemeral app and
    tears it down the moment this function returns -- spawned calls included -- so
    a launcher that spawns and exits kills the container it just started, usually
    before it has finished booting. For a launcher that genuinely should not wait,
    deploy the app and spawn through `modal.Function.from_name`, where the call
    outlives whoever started it.
    """
    leases.pop(run_id, None)
    call = etl.spawn(run_id, cores=1)
    leases.put(run_id, new_grant(call.object_id, etl.info.function_name))
    print(f"granted lease for {run_id} -> {call.object_id}, waiting for it to finish")
    call.get()


JOBS = {"job0": (job0, jobs.job0), "job1": (job1, jobs.job1)}


def attempt_launch(run_id: str, job: str) -> tuple[bool, str, modal.functions.FunctionCall | None]:
    """Grant `run_id` to one call of `job`, or refuse and say why.

    Shared by `launch_job` (a local entrypoint, which waits on the call) and
    the web `/launch` route (which cannot wait -- a request has to return) so
    the checks and the grant/spawn itself are decided in one place, not
    twice: `run_id`/`job` are untrusted here in a way they weren't for a
    CLI-only launcher, since a POST route is reachable by anyone with the
    URL, so both are validated before touching a lease or a path built from
    either.
    """
    if not run_id or "/" in run_id or run_id in (".", ".."):
        return False, f"invalid run_id {run_id!r}", None
    if job not in JOBS:
        return False, f"unknown job {job!r}", None

    fn, job_cls = JOBS[job]
    job_uid = job_cls.job_uid

    grant = leases.get(run_id)
    beat = beats.get(grant["call_id"]) if grant else None
    if is_active(grant, beat, time.time()):
        return False, f"{job_uid}: run {run_id} already has an active call -- not launching", None

    state = jobs.preflight_check(run_id, volume=volume).get(job_uid)
    if state is None:
        return False, f"{job_uid}: no config.json (or not declared) for run {run_id} -- nothing to launch", None
    if not state["ready"]:
        return False, f"{job_uid}: blocked on {state['missing_dependencies']} for run {run_id}", None

    leases.pop(run_id, None)
    call = fn.spawn(run_id)
    leases.put(run_id, new_grant(call.object_id, fn.info.function_name))
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