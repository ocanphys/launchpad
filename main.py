
import time
from collections.abc import Callable
from pathlib import Path

import modal

from config import APP_NAME, VOLUME_NAME, STORAGE, CONTAINER_LIFETIME
from lease_protocol import beats, leases, new_grant
from runtime import initialize_worker

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.12")

# The modules every call needs whatever it runs: config, its logger, its lease,
# and the scope that wires those two together.
CALL_SOURCE = ("config", "logs", "lease_protocol", "runtime")

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

PAGE = Path("/page.html")  # where web_image mounts it, read at request time

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
    # The page is HTML, so it stays HTML -- mounted like the Snakefile, not
    # pasted into this module as a string.
    .add_local_file("page.html", PAGE.as_posix())
    .add_local_python_source(*CALL_SOURCE)
)

def read_book() -> dict:
    """One read of the whole book: every lease, each with its latest heartbeat.

    `leases` and `beats` are two separate Dicts, written by different actors --
    the launcher grants, the worker beats -- so they are joined back together
    here, per run, which is the only shape anyone reads them in. The join is a
    direct lookup, `beat_records[grant["call_id"]]`: a beat is keyed by the call
    that wrote it, so whatever comes back under the *current* holder's call_id is
    unambiguously the current holder's, never a superseded call's leftover.

    `age` is computed here rather than left to the reader, because it is the
    answer everyone actually wants and the clock it needs is this one. A reader's
    clock is its own, and a laptop a few seconds off would make a healthy run look
    dead.

    The beats also go back untouched, beside the joined view. The join answers
    "is this run alive"; the raw beats answer "what is actually written down",
    which is the question worth asking when the join says something surprising.
    """
    grants = dict(leases.items())
    beat_records = dict(beats.items())

    now = time.time()
    runs = {}
    for run_id, grant in sorted(grants.items()):
        beat = beat_records.get(grant["call_id"])
        runs[run_id] = {
            **grant,
            "last_heartbeat": beat["last_beat_ts"] if beat else None,
            "heartbeat_age": round(now - beat["last_beat_ts"], 1) if beat else None,
        }

    # call_ids beating for a run they are not (or no longer) the granted holder
    # of: a superseded container still going, or a beat outliving its lease.
    held_call_ids = {grant["call_id"] for grant in grants.values()}
    orphans = sorted(set(beat_records) - held_call_ids)

    return {
        "now": now,
        "runs": runs,
        "beats": beat_records,
        "orphan_beats": orphans,
    }


@app.function(image=web_image)
@modal.asgi_app()
def leasebook():
    """The page and the data it polls, under one URL.

    An ASGI app rather than two `fastapi_endpoint`s because two endpoints are two
    URLs on two subdomains: the page would have to be told where its data lives,
    and the browser would treat the answer as cross-origin and refuse to read it.
    Served together, `data` is just a relative path, and no CORS question arises.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.read_text()

    @api.get("/data")
    def data() -> dict:
        return read_book()

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


FLATLINE = 5 # if heartbeat age is longer than this then lease is considered expired.

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