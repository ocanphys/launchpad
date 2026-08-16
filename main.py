
import time
from collections.abc import Callable
from pathlib import Path

import modal

from config import APP_NAME, VOLUME_NAME, STORAGE, CONTAINER_LIFETIME
from lease_protocol import lease_key, leases, new_grant, function_status
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

# A third image, for the same reason there are two: reading the leasebook needs
# fastapi and nothing else -- no snakemake, no volume, nothing a job needs.
web_image = base_image.pip_install("fastapi[standard]").add_local_python_source(*CALL_SOURCE)

RUN_DIR = Path(STORAGE) / "data"



@app.function(image=etl_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def etl(run_id: str, cores:int) -> object:
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



@app.local_entrypoint()
def launch(run_id: str | None = None):
    call = etl.spawn(run_id, cores=1)
    response = leases.put(lease_key(run_id),new_grant(call.object_id, etl.info.function_name))
    print (call, response)