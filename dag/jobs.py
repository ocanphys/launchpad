"""Job types for the DAG described in job-spec.md: a job's dependencies are
always named lists (of live Job objects, or manifest paths -- never anything
else), and every job has a deterministic id derived from its parameters and
its dependencies' ids.

Loading a manifest (Job.from_manifest) defaults to a *deep* load: every
nested dependency in its recursive recipe is itself reconstructed as a live
Job, so parameters are re-validated and every level's id is re-derived and
checked against what was recorded -- all the way up the ancestry, not just
at the one job named by the path. Pass frozen=True for the cheaper
alternative: stop at that one job, treat it as a leaf, and don't rebuild (or
re-check) anything upstream of it.

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


def _check_outputs_exist(raw: dict, root: Path, label: str) -> None:
    """Raise if any path in raw's declared outputs is missing under root --
    shared by from_manifest's frozen and deep paths, one manifest at a
    time (deep verification calls this once per level, not just at top)."""
    missing = [p for paths in raw["outputs"].values() for p in paths if not (root / p).exists()]
    if missing:
        raise JobError(f"{label}: declared output(s) missing on disk -- {missing}")


def _load_manifest(raw: dict, root: Path, *, verify: bool, frozen: bool, label: str) -> "Job":
    """The recursive engine behind Job.from_manifest -- job_type always
    resolves from raw itself (from_manifest already checked it against the
    calling class, if any, before getting here), so this needs no class of
    its own to dispatch through."""
    job_cls = JOB_TYPES.get(raw.get("job_type"))
    if job_cls is None:
        raise JobError(f"{label}: unknown job_type {raw.get('job_type')!r}")

    if frozen:
        try:
            parameters = job_cls.Parameters.model_validate(raw["parameters"])
        except ValidationError as e:
            raise JobError(f"{label}: parameters no longer validate -- {e}") from e
        if verify:
            _check_outputs_exist(raw, root, label)
        job = object.__new__(job_cls)
        job.parameters = parameters
        job.run_id = raw.get("run_id")
        job.root = root
        job.frozen = True
        job.deps = {}
        job.id = raw["id"]
        job._raw_manifest = raw
        return job

    deps = {
        name: [
            _load_manifest(entry, root, verify=verify, frozen=False, label=f"{label} > {name}")
            for entry in entries
        ]
        for name, entries in raw["dependencies"].items()
    }
    try:
        job = job_cls(raw["parameters"], raw.get("run_id"), root=root, **deps)
    except ValidationError as e:
        raise JobError(f"{label}: parameters no longer validate -- {e}") from e
    if verify:
        _check_outputs_exist(raw, root, label)
        if job.id != raw["id"]:
            raise JobError(
                f"{label}: id drift -- recorded {raw['id']!r}, reconstructed {job.id!r} "
                "(parameters or an upstream dependency no longer match what produced "
                "this manifest)"
            )
    return job


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
    dependencies: slot name -> the Job class that slot expects. A manifest
        path given for a slot is loaded through that slot's declared class
        (TrainTokenizer.from_manifest(...), not the generic Job.from_manifest),
        so a manifest naming the wrong job_type is rejected right there. A
        Job object passed in directly isn't checked against it, though --
        duck typing beyond that is deferred, job-spec.md §8. A slot holds a
        list at construction time, of any length -- no min/max to satisfy,
        just the fixed, deterministic set of names declared here.
    outputs: output name -> path template, interpolated against parameters,
        run_id, and this job's own id; a "*" in the template is a glob family
    inputs: input name -> (slot name, that slot's output name to concatenate)

    Every subclass below also overrides __init__ with an explicit, fully
    typed signature (parameters typed as that subclass's own Parameters
    model, a named keyword per dependency slot) that does nothing but call
    this one -- construction behavior is defined here, once; the overrides
    exist purely so an editor shows each job type's real shape. A plain
    dict for parameters still works at runtime (model_validate accepts
    either), matching every example in job-spec.md's appendix -- it just
    doesn't get the same field-by-field autocomplete a Parameters(...) call
    does, since a bare dict isn't shape-typed.
    """

    Parameters: type[BaseModel]
    dependencies: ClassVar[dict[str, type["Job"]]] = {}
    outputs: ClassVar[dict[str, str]] = {}
    inputs: ClassVar[dict[str, tuple[str, str]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        JOB_TYPES[cls.__name__] = cls

    def __init__(
        self, parameters: dict, run_id: str, root: str | Path = ".", **deps: list
    ):
        self.parameters = self.Parameters.model_validate(parameters)
        self.run_id = run_id
        self.root = Path(root)
        self.frozen = False
        self._raw_manifest: dict | None = None

        unknown = set(deps) - set(self.dependencies)
        if unknown:
            raise JobError(
                f"{type(self).__name__}: unknown dependency slot(s) {sorted(unknown)}"
            )

        self.deps: dict[str, list[Job]] = {}
        for name in self.dependencies:
            values = deps.get(name, [])
            if not isinstance(values, list):
                raise JobError(
                    f"{type(self).__name__}: dependency {name!r} must be a list"
                )
            self.deps[name] = [self._resolve_dep(name, v) for v in values]

        self.id = self._derive_id()

    def _resolve_dep(self, name: str, value: "Job | str | Path") -> "Job":
        if isinstance(value, Job):
            return value
        if isinstance(value, (str, Path)):
            # Load through the slot's declared type, not the generic Job --
            # from_manifest then checks the manifest actually names that
            # type, so a manifest path pointed at the wrong slot fails here
            # rather than quietly wiring in whatever job_type it happens to be.
            return self.dependencies[name].from_manifest(value, root=self.root)
        raise JobError(
            f"{type(self).__name__}: a dependency must be a Job or a manifest path, got {value!r}"
        )

    def _derive_id(self) -> str:
        return canonical_hash(
            {
                "job_type": type(self).__name__,
                "parameters": self.parameters.model_dump(mode="json"),
                "dependencies": {
                    name: [j.id for j in jobs] for name, jobs in self.deps.items()
                },
            }
        )

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        root: str | Path = ".",
        *,
        verify: bool = True,
        frozen: bool = False,
    ) -> "Job":
        """Load a manifest. By default this is a *deep* load (see the
        module docstring): every nested dependency is reconstructed as a
        live Job, recursively, through each type's normal constructor.
        Pass frozen=True to stop at this one job instead -- a manifest-path
        dependency inside another job's own constructor resolves through
        this same default, so it deep-loads too unless told otherwise.

        Call this on a specific job type -- TrainTokenizer.from_manifest(path)
        -- to also assert the manifest actually names that type; a mismatch
        raises immediately, before anything is validated or reconstructed.
        Job.from_manifest(path), on the base class, skips that check and
        resolves purely from what the manifest itself declares.
        """
        root = Path(root)
        full_path = root / path
        try:
            raw = json.loads(full_path.read_text())
        except FileNotFoundError:
            raise JobError(f"manifest not found: {full_path}") from None
        if cls is not Job and raw.get("job_type") != cls.__name__:
            raise JobError(
                f"{full_path}: expected job_type {cls.__name__!r}, "
                f"manifest declares {raw.get('job_type')!r}"
            )
        return _load_manifest(raw, root, verify=verify, frozen=frozen, label=str(full_path))

    def dep(self, name: str) -> list["Job"]:
        return self.deps[name]

    def output_paths(self) -> dict[str, list[Path]]:
        """Real, checkable filesystem paths (root already joined in) -- for
        a frozen job these come from the manifest's root-relative paths
        joined against the root it was loaded with."""
        if self._raw_manifest is not None:
            return {
                name: [self.root / p for p in paths]
                for name, paths in self._raw_manifest["outputs"].items()
            }
        fields = {
            **self.parameters.model_dump(mode="json"),
            "run_id": self.run_id,
            "id": self.id,
        }
        resolved = {}
        for name, template in self.outputs.items():
            pattern = template.format(**fields)
            resolved[name] = (
                sorted(self.root.glob(pattern))
                if "*" in pattern
                else [self.root / pattern]
            )
        return resolved

    def input_paths(self) -> dict[str, list[Path]]:
        if self._raw_manifest is not None:
            return {
                name: [self.root / p for p in paths]
                for name, paths in self._raw_manifest["inputs"].items()
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
        to be built against. run_id is stored too (beyond the §5 schema)
        so a deep from_manifest load can reconstruct this job through its
        normal constructor and get run_id-scoped output paths right."""
        if self._raw_manifest is not None:
            return dict(self._raw_manifest)
        return {
            "id": self.id,
            "job_type": type(self).__name__,
            "run_id": self.run_id,
            "parameters": self.parameters.model_dump(mode="json"),
            "dependencies": {
                name: [j.to_manifest() for j in jobs]
                for name, jobs in self.deps.items()
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

    def __init__(
        self, parameters: Parameters | dict, run_id: str, root: str | Path = "."
    ) -> None:
        super().__init__(parameters, run_id, root=root)


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

    def __init__(
        self,
        parameters: Parameters | dict,
        run_id: str,
        root: str | Path = ".",
        *,
        sources: list[DownloadSource | str | Path] = (),
    ) -> None:
        super().__init__(parameters, run_id, root=root, sources=list(sources))


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

    def __init__(
        self,
        parameters: Parameters | dict,
        run_id: str,
        root: str | Path = ".",
        *,
        source: list[DownloadSource | str | Path] = (),
        tokenizer: list[TrainTokenizer | str | Path] = (),
    ) -> None:
        super().__init__(
            parameters, run_id, root=root, source=list(source), tokenizer=list(tokenizer)
        )


class BuildSplit(Job):
    class Parameters(BaseModel):
        train_source_uids: list[str]
        valid_source_uids: list[str]
        tokenizer_uid: str

    dependencies: ClassVar = {"tokenizer": TrainTokenizer, "sources": TokenizeSource}
    inputs: ClassVar = {
        "bins": ("sources", "bin"),
        "tokenizer": ("tokenizer", "tokenizer"),
    }
    outputs: ClassVar = {
        "train": "datasets/{id}/train.bin",
        "valid": "datasets/{id}/val.bin",
    }

    def __init__(
        self,
        parameters: Parameters | dict,
        run_id: str,
        root: str | Path = ".",
        *,
        tokenizer: list[TrainTokenizer | str | Path] = (),
        sources: list[TokenizeSource | str | Path] = (),
    ) -> None:
        super().__init__(
            parameters, run_id, root=root, tokenizer=list(tokenizer), sources=list(sources)
        )


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

    def __init__(
        self,
        parameters: Parameters | dict,
        run_id: str,
        root: str | Path = ".",
        *,
        dataset: list[BuildSplit | str | Path] = (),
    ) -> None:
        super().__init__(parameters, run_id, root=root, dataset=list(dataset))


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

    def __init__(
        self,
        parameters: Parameters | dict,
        run_id: str,
        root: str | Path = ".",
        *,
        base: list[Pretrain | str | Path] = (),
    ) -> None:
        super().__init__(parameters, run_id, root=root, base=list(base))
