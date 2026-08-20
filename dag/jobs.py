"""Job types for the DAG described in job-spec.md: a job's dependencies are
always named lists (of live Job objects, or manifest paths that get loaded
immediately as frozen leaves -- never anything else), and every job has a
deterministic id derived from its parameters and its dependencies' ids.

See dag.py for DAG-level operations (topo_sort/status/scheduled) over the
Job objects built here.
"""

import json
from abc import ABC
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

JOB_TYPES: dict[str, type["Job"]] = {}


class JobError(Exception):
    """A job could not be constructed, or a manifest failed to load."""


def canonical_hash(obj) -> str:
    """Stable short hash of a JSON-serializable object.

    canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return sha256(blob.encode()).hexdigest()[:12]


class Job(ABC):
    """Base class every job type subclasses. A subclass declares four class
    attributes -- Parameters (a pydantic model), dependencies (its slots and
    the job class each expects), outputs (path templates), and inputs (which
    slot+output each named input consumes) -- and gets construction,
    identity, and manifest read/write for free.

    Parameters: type[BaseModel]
    dependencies: slot name -> the Job class that slot expects. Documents
        each slot's type right on the class; not enforced at construction
        (duck typing beyond that is deferred -- job-spec.md §8). A slot
        holds a list at construction time, of any length -- no min/max to
        satisfy, just the fixed, deterministic set of names declared here.
    outputs: output name -> path template, interpolated against parameters,
        run_id, and this job's own id; a "*" in the template is a glob family
    inputs: input name -> (slot name, that slot's output name to concatenate)
    """

    Parameters: type[BaseModel]
    dependencies: ClassVar[dict[str, type["Job"]]] = {}
    outputs: ClassVar[dict[str, str]] = {}
    inputs: ClassVar[dict[str, tuple[str, str]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        JOB_TYPES[cls.__name__] = cls

    def __init__(self, parameters: dict, run_id: str, root: str | Path = ".", **deps: list):
        self.parameters = self.Parameters.model_validate(parameters)
        self.run_id = run_id
        self.root = Path(root)
        self.frozen = False
        self._raw_manifest: dict | None = None

        unknown = set(deps) - set(self.dependencies)
        if unknown:
            raise JobError(f"{type(self).__name__}: unknown dependency slot(s) {sorted(unknown)}")

        self.deps: dict[str, list[Job]] = {}
        for name in self.dependencies:
            values = deps.get(name, [])
            if not isinstance(values, list):
                raise JobError(f"{type(self).__name__}: dependency {name!r} must be a list")
            self.deps[name] = [self._resolve_dep(v) for v in values]

        self.id = self._derive_id()

    def _resolve_dep(self, value: "Job | str") -> "Job":
        if isinstance(value, Job):
            return value
        if isinstance(value, str):
            return Job.from_manifest(value, root=self.root)
        raise JobError(
            f"{type(self).__name__}: a dependency must be a Job or a manifest path, got {value!r}"
        )

    def _derive_id(self) -> str:
        return canonical_hash({
            "job_type": type(self).__name__,
            "parameters": self.parameters.model_dump(mode="json"),
            "dependencies": {name: [j.id for j in jobs] for name, jobs in self.deps.items()},
        })

    @classmethod
    def from_manifest(cls, path: str | Path, root: str | Path = ".", *, verify: bool = True) -> "Job":
        """Load a manifest as a frozen leaf: parameters are re-validated
        against the job type's Parameters model (never just trusted), and --
        unless verify=False -- every declared output path is checked to
        exist on disk. Its own dependencies are not rebuilt; DAG traversal
        stops here.
        """
        root = Path(root)
        full_path = root / path
        try:
            raw = json.loads(full_path.read_text())
        except FileNotFoundError:
            raise JobError(f"manifest not found: {full_path}") from None

        job_type = raw.get("job_type")
        job_cls = JOB_TYPES.get(job_type)
        if job_cls is None:
            raise JobError(f"{full_path}: unknown job_type {job_type!r}")

        try:
            parameters = job_cls.Parameters.model_validate(raw["parameters"])
        except ValidationError as e:
            raise JobError(f"{full_path}: parameters no longer validate -- {e}") from e

        if verify:
            missing = [p for paths in raw["outputs"].values() for p in paths if not (root / p).exists()]
            if missing:
                raise JobError(f"{full_path}: declared output(s) missing on disk -- {missing}")

        job = object.__new__(job_cls)
        job.parameters = parameters
        job.run_id = raw.get("run_id")
        job.root = root
        job.frozen = True
        job.deps = {}
        job.id = raw["id"]
        job._raw_manifest = raw
        return job

    def dep(self, name: str) -> list["Job"]:
        return self.deps[name]

    def output_paths(self) -> dict[str, list[Path]]:
        """Real, checkable filesystem paths (root already joined in) -- for
        a frozen job these come from the manifest's root-relative paths
        joined against the root it was loaded with."""
        if self._raw_manifest is not None:
            return {
                name: [self.root / p for p in paths] for name, paths in self._raw_manifest["outputs"].items()
            }
        fields = {**self.parameters.model_dump(mode="json"), "run_id": self.run_id, "id": self.id}
        resolved = {}
        for name, template in self.outputs.items():
            pattern = template.format(**fields)
            resolved[name] = sorted(self.root.glob(pattern)) if "*" in pattern else [self.root / pattern]
        return resolved

    def input_paths(self) -> dict[str, list[Path]]:
        if self._raw_manifest is not None:
            return {
                name: [self.root / p for p in paths] for name, paths in self._raw_manifest["inputs"].items()
            }
        resolved = {}
        for name, (slot, output_name) in self.inputs.items():
            paths = []
            for j in self.deps[slot]:
                paths.extend(j.output_paths()[output_name])
            resolved[name] = paths
        return resolved

    def to_manifest(self) -> dict:
        """The §5 manifest schema: a recursive recipe, not a graph -- each
        dependency is its own nested to_manifest(), not a reference. Input/
        output paths are stored root-relative, like every other manifest
        path -- portable to a different root than the one this job happened
        to be built against."""
        if self._raw_manifest is not None:
            return dict(self._raw_manifest)
        return {
            "id": self.id,
            "job_type": type(self).__name__,
            "parameters": self.parameters.model_dump(mode="json"),
            "dependencies": {
                name: [j.to_manifest() for j in jobs] for name, jobs in self.deps.items()
            },
            "inputs": {
                name: [str(p.relative_to(self.root)) for p in paths]
                for name, paths in self.input_paths().items()
            },
            "outputs": {
                name: [str(p.relative_to(self.root)) for p in paths]
                for name, paths in self.output_paths().items()
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write this job's manifest at root/path; returns path itself
        (root-relative), ready to hand straight to Job.from_manifest."""
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(self.to_manifest(), indent=2))
        return Path(path)


class DownloadSource(Job):
    class Parameters(BaseModel):
        source_uid: str
        url: str
        type: str
        config: dict = Field(default_factory=dict)

    outputs: ClassVar = {"text": "sources/{source_uid}/text/*"}


class TrainTokenizer(Job):
    class Parameters(BaseModel):
        tokenizer_uid: str
        vocab_size: int
        kind: str
        special_tokens: list[str] = Field(default_factory=list)
        token_source_uids: list[str]

    dependencies: ClassVar = {"sources": DownloadSource}
    inputs: ClassVar = {"texts": ("sources", "text")}
    outputs: ClassVar = {
        "tokenizer": "tokenizers/{tokenizer_uid}/tokenizer.json",
        "config": "tokenizers/{tokenizer_uid}/config.json",
    }


class TokenizeSource(Job):
    class Parameters(BaseModel):
        tokenizer_uid: str
        source_uid: str

    dependencies: ClassVar = {"source": DownloadSource, "tokenizer": TrainTokenizer}
    inputs: ClassVar = {
        "text": ("source", "text"),
        "tokenizer": ("tokenizer", "tokenizer"),
        "config": ("tokenizer", "config"),
    }
    outputs: ClassVar = {"bin": "tokenizers/{tokenizer_uid}/bin/{source_uid}/*"}


class BuildSplit(Job):
    class Parameters(BaseModel):
        train_source_uids: list[str]
        valid_source_uids: list[str]
        tokenizer_uid: str

    dependencies: ClassVar = {"tokenizer": TrainTokenizer, "sources": TokenizeSource}
    inputs: ClassVar = {"bins": ("sources", "bin"), "tokenizer": ("tokenizer", "tokenizer")}
    outputs: ClassVar = {
        "train": "datasets/{id}/train.bin",
        "valid": "datasets/{id}/val.bin",
    }


class Pretrain(Job):
    class Parameters(BaseModel):
        model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

        run_id: str
        model_cfg: dict = Field(alias="model_config")
        training_config: dict

    dependencies: ClassVar = {"dataset": BuildSplit}
    inputs: ClassVar = {"train": ("dataset", "train"), "valid": ("dataset", "valid")}
    outputs: ClassVar = {
        "checkpoints": "runs/{run_id}/pretrain/checkpoints/*",
        "logs": "runs/{run_id}/pretrain/logs/*",
        "progress": "runs/{run_id}/pretrain/progress.json",
    }


class SFT(Job):
    class Parameters(BaseModel):
        run_id: str
        base_run_id: str
        sft_training_config: dict

    dependencies: ClassVar = {"base": Pretrain}
    inputs: ClassVar = {"checkpoints": ("base", "checkpoints")}
    outputs: ClassVar = {
        "checkpoints": "runs/{run_id}/sft/checkpoints/*",
        "logs": "runs/{run_id}/sft/logs/*",
        "progress": "runs/{run_id}/sft/progress.json",
    }
