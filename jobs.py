"""The toy jobs a run folder can host, and the config that says which ones.

Unlike `etl`, a run folder is not one job -- it can host several, declared in
the run's own config.json, some depending on others having finished first:

    {
      "metadata": {...},
      "jobs": {
        "<job_uid>": {"parameters": {...}, "dependencies": ["<job_uid>", ...]},
        ...
      }
    }

A job_uid names both the config entry and the class below that runs it --
`main.py` looks the class up by that exact name, so a new job is one class
here plus one entry in some run's config, nothing else to wire up.

"Finished", for whatever a job_uid's dependents are waiting on, means it left
its artifact file behind (`artifact_name`) -- nothing fancier than that yet.
"""

import json
import time
from pathlib import Path

import modal

from config import STORAGE

RUNS = Path(STORAGE) / "runs"


class JobError(Exception):
    """This job cannot run in this folder -- config missing, or a dependency unmet."""


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def artifact_name(job_uid: str) -> str:
    return f"{job_uid}_artifact.txt"


def load_config(run_id: str) -> dict | None:
    """Read config.json off the mounted volume. Only valid where STORAGE is
    actually mounted -- inside a job's own container, which is the only
    place this is called from."""
    try:
        return json.loads((run_dir(run_id) / "config.json").read_text())
    except (OSError, ValueError):
        return None


def check_dependencies(config: dict, present: set[str], job_uid: str) -> tuple[bool, list[str]]:
    """True + [] once every dependency job_uid has left its artifact behind;
    else False + the ones still missing. Pure -- takes a snapshot rather than
    reading anything itself, so the same decision runs whether that snapshot
    came from `preflight_check`'s local mounted read or its Volume API one.
    """
    deps = config.get("jobs", {}).get(job_uid, {}).get("dependencies", [])
    missing = [d for d in deps if artifact_name(d) not in present]
    return not missing, missing


def preflight_check(run_id: str, volume: modal.Volume | None = None) -> dict[str, dict]:
    """Per-run, per-job_uid state: done (artifact present), ready (every
    dependency done), and what's still missing if not.

    Two ways to get the (config, present-filenames) snapshot this is built
    from, picked by whether `volume` is passed:

    `volume` given -- one `listdir` and one `read_file` over the Volume API.
    For callers with no local mount at all, like `launch_job` / `attempt_launch`
    (the latter runs both inside `leasebook`, which does have STORAGE mounted,
    and from the unmounted `launch_job` local_entrypoint -- since it must work
    for both, it always passes `volume` and takes the API path).

    `volume` omitted -- read straight off the local mount instead. For
    `read_state`: its container (`leasebook`) already has `STORAGE` mounted,
    so a network round-trip per run for data already sitting on local disk is
    pure waste -- and under load it's worse than waste, since `leasebook` is
    pinned to one container (`max_containers=1`): a slow Volume API call
    there queues every other request behind it instead of spreading across
    containers.

    Example -- a run whose config.json declares:

        {"jobs": {"job0": {"dependencies": []},
                   "job1": {"dependencies": ["job0"]}}}

    and whose folder holds only "config.json" so far (job0 hasn't written
    its artifact yet, so job1 is stuck behind it):

        preflight_check(run_id) == {
            "job0": {"done": False, "ready": True,  "missing_dependencies": []},
            "job1": {"done": False, "ready": False, "missing_dependencies": ["job0"]},
        }
    """
    if volume is not None:
        try:
            config = json.loads(b"".join(volume.read_file(f"runs/{run_id}/config.json")))
        except FileNotFoundError:
            return {}
        present = {Path(entry.path).name for entry in volume.listdir(f"runs/{run_id}")}
    else:
        config = load_config(run_id)
        if config is None:
            return {}
        rdir = run_dir(run_id)
        present = {p.name for p in rdir.iterdir()} if rdir.exists() else set()

    # Per-job_uid state for every job_uid this run's config declares: `done`
    # if its artifact file is already on disk, `ready` if every dependency's
    # artifact is too, `missing_dependencies` naming what's blocking it when
    # it isn't.
    states = {}
    for job_uid in config.get("jobs", {}):
        ready, missing = check_dependencies(config, present, job_uid)
        states[job_uid] = {
            "done": artifact_name(job_uid) in present,
            "ready": ready,
            "missing_dependencies": missing,
        }
    return states


class job0:
    """Counts to config["jobs"]["job0"]["parameters"]["max"] (default 20),
    logging every step, then leaves job0_artifact.txt behind for anything
    that depends on it.
    """

    job_uid = "job0"

    def __init__(self, run_id: str, logger, confirm_lease):
        self.run_id = run_id
        self.log = logger
        self.confirm = confirm_lease
        self.rdir = run_dir(run_id)

        self.config = load_config(run_id)
        if self.config is None:
            raise JobError(f"{self.job_uid}: config.json missing or unparseable for run {run_id}")

        params = self.config.get("jobs", {}).get(self.job_uid, {}).get("parameters", {})
        self.max = params.get("max", 20)
        self.count = 0

    def check_dependencies(self) -> tuple[bool, list[str]]:
        present = {p.name for p in self.rdir.iterdir()} if self.rdir.exists() else set()
        return check_dependencies(self.config, present, self.job_uid)

    def progress(self) -> float:
        return self.count / self.max

    def status(self) -> str:
        return "finished" if self.count >= self.max else "in progress"

    def run(self) -> None:
        ok, missing = self.check_dependencies()
        if not ok:
            raise JobError(f"{self.job_uid}: blocked on dependencies {missing}")

        while self.count < self.max:
            self.count += 1
            self.log.info(f"{self.job_uid}: count {self.count}/{self.max}")
            time.sleep(1)

        self.confirm(f"{self.job_uid}: before artifact")
        (self.rdir / artifact_name(self.job_uid)).write_text("done")


class job1:
    """Counts to config["jobs"]["job1"]["parameters"]["max"] (default 30),
    logging every step, then leaves job1_artifact.txt behind. Depends on
    job0 by convention in a run's config, not by anything here -- see
    `check_dependencies`.
    """

    job_uid = "job1"

    def __init__(self, run_id: str, logger, confirm_lease):
        self.run_id = run_id
        self.log = logger
        self.confirm = confirm_lease
        self.rdir = run_dir(run_id)

        self.config = load_config(run_id)
        if self.config is None:
            raise JobError(f"{self.job_uid}: config.json missing or unparseable for run {run_id}")

        params = self.config.get("jobs", {}).get(self.job_uid, {}).get("parameters", {})
        self.max = params.get("max", 30)
        self.count = 0

    def check_dependencies(self) -> tuple[bool, list[str]]:
        present = {p.name for p in self.rdir.iterdir()} if self.rdir.exists() else set()
        return check_dependencies(self.config, present, self.job_uid)

    def progress(self) -> float:
        return self.count / self.max

    def status(self) -> str:
        return "finished" if self.count >= self.max else "in progress"

    def run(self) -> None:
        ok, missing = self.check_dependencies()
        if not ok:
            raise JobError(f"{self.job_uid}: blocked on dependencies {missing}")

        while self.count < self.max:
            self.count += 1
            self.log.info(f"{self.job_uid}: count {self.count}/{self.max}")
            time.sleep(1)

        self.confirm(f"{self.job_uid}: before artifact")
        (self.rdir / artifact_name(self.job_uid)).write_text("done")
