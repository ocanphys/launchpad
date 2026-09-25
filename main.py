"""The deployment: one app, three images, four containers.

What each container runs is `launcher/`; nothing here but the resources they
get and the Modal names they answer to.
"""

from pathlib import Path

import modal

from artifacts.core.artifact import Artifact
from config import (
    APP_NAME,
    CONTAINER_LIFETIME,
    LAB_COMMIT_ENV,
    LAB_IDLE_SECONDS,
    LAB_PORT,
    LAB_SECRET,
    PERSIST_LOGS_EVERY,
    REGION,
    STORAGE,
    get_git_commit,
)
from launcher.lab_server import start_jupyterlab
from launcher.state import volume
from system.logs import persist_snapshot
from system.runtime import initialize_worker

app = modal.App(APP_NAME)

base_image = modal.Image.debian_slim(python_version="3.12").pip_install("regex", "tqdm")

# groups of python packages from this repo - these are added to images which specify
# the python environments of the containers.
CALL_SOURCE = ("config", "system", "launcher")
# models/ rides along wherever manifests are decoded: a leg's model_parameters
# are rebuilt from the model package it names, torch-free at import
ARTIFACTS_SOURCE = ("artifacts", "models")


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


@app.function(
    image=web_image,
    volumes={STORAGE: volume},
    max_containers=1,
    region=REGION,
    # The password every route here is behind, and the lab's token, which
    # `/lab` hands straight to Jupyter.
    secrets=[modal.Secret.from_name(LAB_SECRET)],
)
@modal.asgi_app()
def leasebook():
    """The dashboard and the API it serves, in one container.

    Every route but `/login` is behind DASHBOARD_PASSWORD, out of LAB_SECRET.
    A missing key is a container that refuses to start rather than one serving
    in the open. What it serves is `launcher.leasebook`.
    """
    # Imported here, not at module scope: every image ships `launcher`, but
    # only web_image carries fastapi, and every container imports this module
    # to reach its own function.
    from launcher.leasebook import build_api

    return build_api(run_job, jupyter, WEB_DIR)


@app.function(image=web_image, volumes={STORAGE: volume}, region=REGION, schedule=modal.Period(seconds=PERSIST_LOGS_EVERY))
def persist_logs() -> None:
    """Appends every call's new log rows to its file on the volume, writes
    `call_history.json` and commits: the only writer of either.

    Stateless, in its own container: each pass rebuilds the row-count cursor
    from the files themselves, so a pass that never ran costs nothing but
    lag. Fires only on a deployed app (`modal deploy`), not under `modal serve`.
    """
    volume.reload()
    persist_snapshot(STORAGE, volume)


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
        artifact = Artifact.load(artifact_path, STORAGE)
        job = artifact.job()
        job.run(STORAGE, worker)
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

    No explicit `volume.commit()` here or on a background clock: every Volume
    mount already sets `allow_background_commits=True`, so the platform
    flushes writes on its own, and JupyterLab's own autosave writes a
    notebook to disk on its own clock too. Unlike `run_job`, nothing is
    waiting on a precise moment to see the lab's writes, so background
    commits are enough -- no reason to force one. A person who wants one now
    calls `lab.save()` from a cell; `lab.refresh()` is the reload.

    Its settings and its process are `launcher.lab_server`.
    """
    start_jupyterlab()
