"""The schema config.json follows -- one definition, used on both sides:
`push_config.py` (and any notebook) constructs and validates a `RunConfig`
before writing it; `jobs.py` parses one back with `model_validate_json`
instead of walking a plain dict by hand. See config-schema.md for why.
"""

import importlib
from collections.abc import Callable

from pydantic import BaseModel, Field, model_validator


def import_ref(obj: type | Callable) -> str:
    """Encode a class or top-level function as an importable
    "module.qualname" string, so it round-trips through config.json.

    import_ref(RunConfig) -> "run_config.RunConfig"
    """
    return f"{obj.__module__}.{obj.__qualname__}"


def resolve_import_ref(ref: str | type | Callable):
    """Inverse of import_ref. Passes through anything that isn't a string
    -- already a live class/function, as when a config is built fresh in
    Python (a notebook) rather than parsed back from JSON.

    resolve_import_ref("run_config.RunConfig") -> RunConfig
    resolve_import_ref(RunConfig) -> RunConfig
    """
    if not isinstance(ref, str):
        return ref
    module_name, _, qualname = ref.rpartition(".")
    return getattr(importlib.import_module(module_name), qualname)


class ResourcesSpec(BaseModel):
    """A job's optional resource request. Every field absent (None) means
    "use Modal's platform default for that dimension."

    ResourcesSpec(cpu=8, gpu_type="A100", gpu_count=4)
    """

    cpu: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


class JobEntry(BaseModel):
    """One job_uid's declaration inside a run's config.json. job_uid (the
    dict key in RunConfig.jobs, not a field here) doubles as the name of
    the `jobs.py` class that implements it -- `jobs.get_job_class` resolves
    and validates that directly, so there's no separate field to keep in
    sync with it.

    `parameters` stays a generic dict here -- its real shape is job-specific
    (a training job's hyperparameters look nothing like a tokenizer job's),
    so each Job subclass validates its own corner of it. See
    config-schema.md.

    JobEntry(
        parameters={
            "seed": 42,
            "model_params": {"vocab_size": 50257, "context_length": 1024,
                              "n_layers": 12, "n_heads": 12, "d_model": 768},
            "optimizer_params": {"lr": 3e-4, "weight_decay": 0.01, "betas": [0.9, 0.95]},
            "training": {"total_steps": 100_000, "save_every": 1000, "batch_size": 32},
        },
        dependencies=["Tokenize"],
        resources=ResourcesSpec(gpu_type="A100", gpu_count=4),
    )
    """

    parameters: dict = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)


class RunConfig(BaseModel):
    """The whole parsed config.json for one run. Every key in `jobs` must
    name a real Job subclass in jobs.py -- job_uid IS that class's name,
    resolved via resolve_import_ref(f"jobs.{job_uid}") -- checked here, at
    construction time, so a typo'd or renamed class fails loudly in the
    notebook cell that built this config, not later when main.py tries to
    launch it.

    RunConfig(
        metadata={"run_id": "gpt2-124m-run3"},
        jobs={
            "Tokenize": JobEntry(parameters={"vocab_size": 50257}),
            "Train": JobEntry(
                parameters={"seed": 42, "training": {"total_steps": 100_000}},
                dependencies=["Tokenize"],
                resources=ResourcesSpec(gpu_type="A100", gpu_count=4),
            ),
        },
    )
    """

    metadata: dict = Field(default_factory=dict)
    jobs: dict[str, JobEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _job_uids_must_resolve(self) -> "RunConfig":
        import jobs_LEGACY  # deferred: jobs.py imports this module, at the top level

        for job_uid in self.jobs:
            if jobs_LEGACY.get_job_class(job_uid) is None:
                raise ValueError(f"{job_uid!r} does not name a Job class in jobs.py")
        return self
