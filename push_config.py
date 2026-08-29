"""Push a template config.json to a run folder on the volume.

No mount needed -- like `main.py`'s `launch_job`, this writes over the
Volume API (`batch_upload`/`put_file`), so it can seed a run folder before
any container has ever touched it.

The schema is `run_config.RunConfig` -- see that module and
config-schema.md for the shape and why it's validated on both ends.
"""

import io
import time

import modal

from config import APP_NAME, VOLUME_NAME
from jobs_LEGACY import ETL, artifact_name
from run_config import JobEntry, ResourcesSpec, RunConfig

app = modal.App(f"{APP_NAME}-push-config")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def template_config(run_id: str) -> RunConfig:
    """The config a fresh run folder starts from: ETL with an explicit cpu
    request (it needs one -- `ETL.run()` passes it straight to snakemake's
    `--cores`), Count waiting on ETL's artifact and declaring no
    `resources` at all -- so this one template exercises both the declared
    and the default-fallback path. This is the whole template -- edit it
    here for a different starting point, there is nothing else that
    generates one.

    Job_uid here must name a real class in jobs.py ("ETL", "Count") --
    `RunConfig` checks that at construction, right below. Count's
    dependency is a path to ETL's actual artifact file, not ETL's name --
    built from `ETL.job_uid` (what `ETL.start()` itself names the file),
    not the "ETL" key above (which only has to match the class name).
    """
    return RunConfig(
        metadata={"run_id": run_id, "pushed_ts": time.time()},
        jobs={
            "ETL": JobEntry(
                resources=ResourcesSpec(cpu=1),
            ),
            "Count": JobEntry(
                dependencies=[f"runs/{run_id}/{artifact_name(ETL.job_uid)}"],
            ),
        },
    )


@app.local_entrypoint()
def push_config(run_id: str, force: bool = False):
    """Write runs/{run_id}/config.json from the template above.

    Refuses to clobber an existing config.json unless `--force`: the volume
    API itself enforces this (`batch_upload(force=...)`), so this just lets
    that refusal surface as a clear message instead of a raw exception.
    """
    remote_path = f"runs/{run_id}/config.json"
    payload = template_config(run_id).model_dump_json(indent=2).encode()

    try:
        with volume.batch_upload(force=force) as batch:
            batch.put_file(io.BytesIO(payload), remote_path)
    except FileExistsError:
        print(f"{remote_path} already exists -- pass --force to overwrite")
        return

    print(f"pushed template config to {remote_path}")
