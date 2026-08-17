"""Push a template config.json to a run folder on the volume.

No mount needed -- like `main.py`'s `launch_job`, this writes over the
Volume API (`batch_upload`/`put_file`), so it can seed a run folder before
any container has ever touched it.

The schema matches what `jobs.py` and `main.preflight_check` expect:

    {
      "metadata": {...},
      "jobs": {
        "<job_uid>": {
          "parameters": {...},
          "dependencies": ["<job_uid>", ...],
          "resources": {"cpu": 1, "gpu_type": "A100", "gpu_count": 1}
        },
        ...
      }
    }

`resources` is optional -- see `jobs.py`'s module docstring for what an
absent key (or an absent sub-key within it) falls back to.
"""

import io
import json
import time

import modal

from config import APP_NAME, VOLUME_NAME

app = modal.App(f"{APP_NAME}-push-config")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def template_config(run_id: str) -> dict:
    """The config a fresh run folder starts from: job0 with no dependencies
    and an explicit cpu request, job1 waiting on job0's artifact and
    declaring no "resources" at all -- so this one template exercises both
    the declared and the default-fallback path. This dict is the whole
    template -- edit it here for a different starting point, there is
    nothing else that generates one.
    """
    return {
        "metadata": {
            "run_id": run_id,
            "pushed_ts": time.time(),
        },
        "jobs": {
            "job0": {
                "parameters": {"max": 20},
                "dependencies": [],
                "resources": {"cpu": 1},
            },
            "job1": {
                "parameters": {"max": 30},
                "dependencies": ["job0"],
            },
        },
    }


@app.local_entrypoint()
def push_config(run_id: str, force: bool = False):
    """Write runs/{run_id}/config.json from the template above.

    Refuses to clobber an existing config.json unless `--force`: the volume
    API itself enforces this (`batch_upload(force=...)`), so this just lets
    that refusal surface as a clear message instead of a raw exception.
    """
    remote_path = f"runs/{run_id}/config.json"
    payload = json.dumps(template_config(run_id), indent=2).encode()

    try:
        with volume.batch_upload(force=force) as batch:
            batch.put_file(io.BytesIO(payload), remote_path)
    except FileExistsError:
        print(f"{remote_path} already exists -- pass --force to overwrite")
        return

    print(f"pushed template config to {remote_path}")
