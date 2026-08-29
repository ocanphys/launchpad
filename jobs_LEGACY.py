"""The toy jobs a run folder can host, and the config that says which ones.

Unlike `etl`, a run folder is not one job -- it can host several, declared in
the run's own config.json, some depending on others having finished first:

    {
      "metadata": {...},
      "jobs": {
        "<job_uid>": {
          "parameters": {...},
          "dependencies": ["<path relative to the volume root>", ...],
          "resources": {"cpu": 1, "gpu_type": "A100", "gpu_count": 1}
        },
        ...
      }
    }

`resources` is optional, and so is every key inside it -- an absent
sub-key (or an absent `resources` entirely) means "use Modal's platform
default for that dimension," which is every job_uid's behavior today.
`main.py` is the only reader; nothing in this module inspects it.

The schema above is `run_config.RunConfig` -- `load_config` parses and
validates through it, so a malformed config.json fails loudly (a
pydantic `ValidationError`) instead of surfacing as a `KeyError` deep
inside a job's `__init__`. See config-schema.md.

A job_uid names both the config entry and the class below that runs it --
`main.py` looks the class up by that exact name, so a new job is one class
here plus one entry in some run's config, nothing else to wire up.

`dependencies` are files, not job_uids: paths relative to the volume root
that must exist before a job's own `.run()` may start. A job that needs
another job's artifact spells that out as a path to it (typically
`f"runs/{run_id}/{artifact_name(other.job_uid)}"`) rather than naming the
job_uid -- a dependency can live anywhere on the volume this way, not only
in this run's own folder (a job upstream of every run, writing to `/data`
once, is exactly why).
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path

import modal
from pydantic import ValidationError

from config import STORAGE
from run_config import RunConfig, resolve_import_ref

RUNS = Path(STORAGE) / "runs"


class JobError(Exception):
    """This job cannot run in this folder -- config missing, or a dependency unmet."""


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def artifact_name(job_uid: str) -> str:
    return f"{job_uid}_artifact.txt"


def load_config(run_id: str) -> RunConfig | None:
    """Read and validate config.json off the mounted volume. Only valid
    where STORAGE is actually mounted -- inside a job's own container,
    which is the only place this is called from."""
    try:
        return RunConfig.model_validate_json((run_dir(run_id) / "config.json").read_text())
    except (OSError, ValidationError):
        return None


def _path_exists(path: str, volume: modal.Volume | None) -> bool:
    """Does `path` (relative to the volume root) exist? Same volume-or-
    local-mount split as `preflight_check` -- a dependency can live in any
    directory on the volume, not just the current run's, so this checks
    one path directly rather than matching against a pre-listed set.

    _path_exists("runs/r1/etl_artifact.txt", None) -> True/False
    """
    if volume is not None:
        try:
            names = {Path(e.path).name for e in volume.listdir(str(Path(path).parent))}
        except FileNotFoundError:
            return False
        return Path(path).name in names
    return (Path(STORAGE) / path).exists()


def _check_dependencies(deps: list[str], volume: modal.Volume | None) -> tuple[bool, list[str]]:
    """True + [] once every dependency path exists on the volume; else
    False + the ones still missing.

    _check_dependencies(["runs/r1/etl_artifact.txt"], None) -> (True, [])
    """
    missing = [d for d in deps if not _path_exists(d, volume)]
    return not missing, missing


def preflight_check(run_id: str, volume: modal.Volume | None = None) -> dict[str, dict]:
    """Per-job_uid state from a run's config.json -- done, ready, what's
    missing, and (read by `attempt_launch`, not by anything here) each
    job_uid's declared dependencies and resources.

    preflight_check(run_id) -> {
        "etl": {"done": False, "ready": True, "missing_dependencies": [],
                "dependencies": [], "resources": {}},
        "count10": {"done": False, "ready": False,
                    "missing_dependencies": ["runs/r1/etl_artifact.txt"],
                    "dependencies": ["runs/r1/etl_artifact.txt"], "resources": {}},
    }
    # count10 depends on a path etl hasn't written yet

    Reads over the Volume API if `volume` is given, else off the local mount.

    The local-mount branch does not swallow errors: no config.json for
    `run_id` (`FileNotFoundError`) or one that fails `RunConfig` validation
    both raise. This is the branch `read_state` uses to check every run on
    the volume, one by one -- it decides per run_id whether a raise here
    means "not ready yet" or "broken," this function just tells the truth
    about which run_id it was.
    """
    if volume is not None:
        try:
            raw = b"".join(volume.read_file(f"runs/{run_id}/config.json"))
        except FileNotFoundError:
            return {}
        try:
            config = RunConfig.model_validate_json(raw)
        except ValidationError:
            return {}
        present = {Path(entry.path).name for entry in volume.listdir(f"runs/{run_id}")}
    else:
        rdir = run_dir(run_id)
        config = RunConfig.model_validate_json((rdir / "config.json").read_text())
        present = {p.name for p in rdir.iterdir()} if rdir.exists() else set()

    # Per-job_uid state for every job_uid this run's config declares: `done`
    # if its artifact file is already on disk, `ready` if every dependency's
    # artifact is too, `missing_dependencies` naming what's blocking it when
    # it isn't.
    states = {}
    for job_uid, entry in config.jobs.items():
        ready, missing = _check_dependencies(entry.dependencies, volume)
        states[job_uid] = {
            "done": artifact_name(job_uid) in present,
            "ready": ready,
            "missing_dependencies": missing,
            "dependencies": entry.dependencies,
            "resources": entry.resources.model_dump(exclude_none=True),
        }
    return states


class Job(ABC):
    """The contract every job class in this module honors -- previously
    just a convention (see job0/job1 below, which predate it and don't
    inherit from it), now enforced: instantiating a subclass that's
    missing a piece raises immediately, not the first time something
    calls it.

    __init__ is concrete, not abstract: every job reads run_id/logger/
    confirm_lease and its own config.json entry the same way, so that part
    lives here once instead of once per subclass. A subclass with its own
    setup calls super().__init__(...) first, then does the rest -- see
    toy_job below.
    """

    @property
    @abstractmethod
    def job_uid(self) -> str:
        """Names both this class and its entry in a run's config.json."""

    def __init__(self, run_id: str, logger, worker):
        """Read this job_uid's own config.json entry into self.config;
        raise JobError if this run isn't one it can run."""
        self.run_id = run_id
        self.logger = logger
        self.confirm_lease = worker.confirm_lease
        self.rdir = run_dir(run_id)
        self.config = load_config(run_id)
        if self.config is None:
            raise JobError(f"{self.job_uid}: config.json missing or unparseable for run {run_id}")
        entry = self.config.jobs.get(self.job_uid)
        if entry is None:
            raise JobError(f"{self.job_uid}: no entry for this job in config.json for run {run_id}")
        self.resources = entry.resources
        self.dependencies = entry.dependencies
        self.parameters = entry.parameters
        self.metadata = self.config.metadata

    def check_dependencies(self) -> tuple[bool, list[str]]:
        """True + [] once every dependency path this job declared exists;
        else False + the ones still missing."""
        return _check_dependencies(self.dependencies, volume=None)

    def start(self) -> None:
        """Check deps, run the job and leave artifact_name(self.job_uid) behind"""
        ok, missing = self.check_dependencies()
        if not ok:
            raise JobError(f"{self.job_uid}: blocked on dependencies {missing}")

        try:
            self.run() # run the job!

            self.confirm(f"{self.job_uid}: before artifact")
            (self.rdir / artifact_name(self.job_uid)).write_text("done")
        except Exception as e:
            raise JobError(f"{self.job_uid}: failed -- {e}")

    @abstractmethod
    def run(self) -> float:
        """Main content of the job - pytorch loop, """

    @abstractmethod
    def progress(self) -> float:
        """How far through its work this job is, 0.0 to 1.0."""

    @abstractmethod
    def status(self) -> str:
        """Returns one of the following: 'new', 'in progress', 'finished' """


def get_job_class(job_uid: str) -> type[Job] | None:
    """Resolve job_uid to its Job subclass in this module -- job_uid IS
    the class name, so "ETL" resolves to jobs.ETL directly via
    resolve_import_ref, no separate registry to keep in sync with whatever
    classes are actually defined here. `RunConfig` calls this too, to
    validate every job_uid at config-parse time rather than launch time.

    get_job_class("ETL") -> jobs.ETL
    get_job_class("nope") -> None
    """
    try:
        cls = resolve_import_ref(f"jobs.{job_uid}")
    except (ImportError, AttributeError):
        return None
    if isinstance(cls, type) and issubclass(cls, Job) and cls is not Job:
        return cls
    return None


class Count(Job):
    """Smallest possible Job: no config parameters, no dependencies -- just
    sleeps, then leaves its artifact behind. Demonstrates the contract
    above; not registered in main.py's JOBS, so it isn't launchable as-is.
    """

    job_uid = "count10"

    def __init__(self, run_id: str, logger, worker):
        super().__init__(run_id, logger, worker)

    def progress(self) -> float:
        return 1.0 if self.done else 0.0

    def status(self) -> str:
        match self.progress:
            case 0.0:
                return "new"
            case 1.0:
                return "finished" 
            case _:
                return "in progress"

    def run(self) -> None:

        self.logger.info(f"{self.job_uid}: sleeping")
        time.sleep(1)


class ETL(Job):
    job_uid = "etl"

    def __init__(self, run_id: str, logger, worker):
        super().__init__(run_id, logger, worker)

    def progress(self) -> float:
        return 1.0 if self.done else 0.0

    def status(self) -> str:
        match self.progress:
            case 0.0:
                return "new"
            case 1.0:
                return "finished" 
            case _:
                return "in progress"
            
    def run(self) -> None:
        import subprocess

        self.confirm_lease()
        self.rdir.mkdir(parents=True, exist_ok=True)
        cores = self.resources.cpu
        subprocess.run(
            [
                "snakemake",
                "--snakefile",
                "/etl/Snakefile",  # point to Snakefile in the container.
                "--directory",
                str(self.rdir),
                "--cores",
                str(cores),
            ],
            check=True,
        )
        time.sleep(20)
        self.confirm_lease()
        self.volume.commit()


class Download(Job):
    def __init__(self, run_id: str, logger, worker):
        super().__init__(run_id, logger, worker)


class Source(Job):
    """Downloads every source in self.parameters["sources"] to
    data/sources/{name}/content.txt -- shared, not run-scoped, so a later
    run naming an already-downloaded source reuses it instead of
    refetching. Not implemented -- placeholder for the download logic.
    """

    job_uid = "Source"

    def progress(self) -> float:
        return 0.0

    def status(self) -> str:
        return "not implemented"

    def run(self) -> None:
        raise NotImplementedError(
            "download each of self.parameters['sources'] to data/sources/{name}/content.txt"
        )


class Tokenize(Job):
    """Fits a tokenizer on self.parameters["fit_source"], writing it to
    data/tokenizers/{hash of self.parameters}/tokenizer.json -- shared and
    hash-addressed, so the same configuration is never fit twice. Not
    implemented -- placeholder for the fit logic.
    """

    job_uid = "Tokenize"

    def progress(self) -> float:
        return 0.0

    def status(self) -> str:
        return "not implemented"

    def run(self) -> None:
        raise NotImplementedError(
            "fit a tokenizer per self.parameters, write it to data/tokenizers/{hash}/tokenizer.json"
        )


class Train(Job):
    """Trains against the tokenizer named by
    self.parameters["tokenizer_hash"]. Not implemented -- placeholder for
    the training loop.
    """

    job_uid = "Train"

    def progress(self) -> float:
        return 0.0

    def status(self) -> str:
        return "not implemented"

    def run(self) -> None:
        raise NotImplementedError(
            "train against data/tokenizers/{self.parameters['tokenizer_hash']}/tokenizer.json"
        )