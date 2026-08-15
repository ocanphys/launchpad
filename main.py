
from collections.abc import Callable
from pathlib import Path

import modal
from config import APP_NAME, VOLUME_NAME, STORAGE, CONTAINER_LIFETIME
from lease_protocol import lease_key, leases, new_grant
from runtime import call_scope

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

RUN_DIR = Path(STORAGE) / "data"

@app.function(image=etl_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def run_etl(cores: int = 4) -> str:
    import subprocess

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
    volume.commit()
    return (RUN_DIR / "results" / "summary.txt").read_text()


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def worker(job: Callable[..., object], run_id: str, **kwargs) -> object:
    """Run any job under this call's own logger and lease.

    The job is the only thing passed in. The logger and the `confirm` it gets are
    built here, from the call id this container was actually given -- which is why
    they cannot be handed down from the launcher: the launcher does not know the
    call id until the call exists.
    """
    with call_scope(run_id, owner=job.__name__, volume=volume) as scope:
        return job(scope, **kwargs)


@app.function(image=etl_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def etl_worker(job: Callable[..., object], run_id: str, **kwargs) -> object:
    """`worker`, on the image that carries snakemake.

    The body is duplicated rather than shared because the image is the one thing
    `.with_options()` cannot retune at call time -- it takes cpu, gpu, memory and
    the rest, but no `image` -- so an image means a Function, and a second
    Function means a second definition. A factory closing over the image would
    also work, but each generated function would then need `serialized=True`,
    which is a heavier thing to take on than these two lines. The part actually
    worth sharing is `call_scope`, and both of these share it.
    """
    with call_scope(run_id, owner=job.__name__, volume=volume) as scope:
        return job(scope, **kwargs)


def count(scope, n: int = 3) -> str:
    """A job, for shape: take a scope, confirm before writing, write."""
    scope.log.info(f"counting to {n}")
    scope.confirm("before write")
    out = scope.dir / "count.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(i) for i in range(n)))
    return str(out)


def etl(scope, cores: int = 4) -> str:
    """`run_etl` as a job: the same snakemake call, now leased and logged.

    Three things the original could not do, each one a consequence of having a
    scope rather than a bare function:

    - Output goes to this run's own folder, not the single shared `data/` that
      every call of `run_etl` overwrites in turn.
    - The lease is confirmed on both sides of the subprocess. Snakemake is the
      long part of the call, so it is the part most likely to still be running
      when the launcher decides this container is gone -- and the confirm after
      it is what stops a superseded run from returning a summary as if it won.
    - Snakemake's own output is teed into the call's log file. `getChild` keeps
      it in the same file under a sub-name, so the run folder holds the reason a
      rule failed, not just the fact that one did.

    No `volume.commit()` here, deliberately. The scope owns that, and it only
    does it after one last confirm.
    """
    import subprocess

    data_dir = scope.dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scope.confirm("before snakemake")
    scope.log.info(f"snakemake: {cores} cores -> {data_dir}")

    snakemake = scope.log.getChild("snakemake")
    proc = subprocess.Popen(
        [
            "snakemake",
            "--snakefile",
            "/etl/Snakefile", #point to Snakefile in the container.
            "--directory",
            str(data_dir),
            "--cores",
            str(cores),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # one stream: interleaved as snakemake wrote it.
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        snakemake.info(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"snakemake exited {proc.returncode}")

    scope.confirm("after snakemake")
    return (data_dir / "results" / "summary.txt").read_text()


@app.local_entrypoint()
def main(run_id: str = "demo", cores: int = 4, original: bool = False):
    """Launch the etl job under a lease.

    Spawn first, grant second -- not the other way round. A grant names a call
    id, and the call id does not exist until the call does. The container may
    well boot before the grant lands and read an empty Dict; that is precisely
    the UNKNOWN `fence` treats as silence rather than as a no, and `confirm`
    waits it out. Nothing here would work without that retry.
    """
    if original:
        print(run_etl.remote(cores))  # the untouched original: no lease, no scope
        return

    call = etl_worker.spawn(etl, run_id, cores=cores)
    leases[lease_key(run_id)] = new_grant(call.object_id, "etl")
    print(call.get())
