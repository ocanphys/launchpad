"""The base artifact model: parameters plugged together with the artifacts
they're built from, a folder each artifact owns, and a manifest that rebuilds
it. Concrete types live in sources/, tokenizers/, tokenized/, dataset/,
mappeddataset/, stages/*.

An artifact has three lives, and this class carries all three:

    recipe    Artifact(params)         parameters, a folder, a manifest
    job       Job(artifact).run(root)  writes the files into that folder
    bound     artifact.bind(root)      the stored declaration, files loaded

`load`, `status` and `bind` take a root, defaulting to STORAGE. That default
is for the top of a notebook only. Anything already holding a root (a job, a
`_load` hook) passes it on, or it silently reads the volume while everything
around it reads a test folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, is_dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, get_args

from artifacts.core import manifest, schema
from artifacts.core.locate import locate

if TYPE_CHECKING:
    # Type-only: job.py imports Artifact from here, so a real import would be
    # circular. job() needs only `producer` (a string) at runtime.
    from artifacts.core.job import Job

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)  # config.py is at the repo root; artifacts/core/artifact.py is two levels down
from config import LAB_COMMIT_ENV, STORAGE, get_git_commit

MANIFEST = "manifest.json"


@cache
def _head() -> str:
    """Returns the commit stamped on every artifact built in this process.

    Cached, so one session cannot stamp two commits onto one tree; restart the
    kernel to pick up new work. The lab container has no git binary and no
    .git, so there the commit comes from LAB_COMMIT_ENV, which main.py bakes in.
    """
    override = os.environ.get(LAB_COMMIT_ENV)
    return override if override else get_git_commit()


def _digest(*parts: object) -> str:
    """Returns a 10-character hash of `parts`, which subclasses build their
    `uid` and `artifact_path` from.

    Hashes key-sorted JSON, so equal parameters give an equal id in every
    process. Every part must be JSON-serializable.

    _digest(("<pad>",), ["a", "b"]) -> "9f2b1c40a7"
    """
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _root(root: Path | None) -> Path:
    """Returns `root`, or STORAGE when it is None, checked to be an absolute Path.

    Every public entry point goes through this so a bad root fails at the
    call. A string is rejected rather than converted, so one type reaches every
    join, glob and relative_to downstream; a relative root would be discarded
    by `root / artifact_path` and surface as a missing file somewhere else.
    """
    resolved = root if root is not None else STORAGE
    if not isinstance(resolved, Path):
        raise TypeError(f"root must be a Path, got {type(resolved).__name__} {resolved!r}")
    if not resolved.is_absolute():
        raise ValueError(f"root must be an absolute path, got {resolved}")
    return resolved


@dataclass(frozen=True)
class Resources:
    """What a job should be given to run this artifact; None means Modal's
    platform default for that dimension. Never identity.

    Resources(gpu_type="A100", gpu_count=2)
    """

    cpu: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


@dataclass(frozen=True)
class Footprint:
    """What one root holds for one artifact, as files: its manifest, its own
    outputs by description, and the files that make it complete by path.

    For most artifacts the last two name the same files. A virtual artifact
    owns no outputs and borrows completion from its dependencies, which is why
    "is anything of mine written" and "am I done" are separate questions.
    """

    manifest: bool
    outputs: dict[str, bool]
    completion: dict[Path, bool]

    @property
    def complete(self) -> bool:
        """Returns whether every file that makes this artifact done is there.

        Ask this, not `all(outputs.values())`: the two agree for everything
        that owns its bytes and differ exactly for virtual artifacts, where
        asking the wrong one is silently wrong.
        """
        return all(self.completion.values())


# frozen: identity is its fields, so it must be hashable/immutable
@dataclass(frozen=True)
class Artifact(ABC):
    # Neither of these says which artifact this is: compare=False keeps both
    # out of ==/hash, so an artifact rebuilt from an old manifest is still the
    # same artifact. kw_only, or a defaulted base field would force defaults
    # on every subclass.
    commit: str = field(default_factory=_head, compare=False, repr=False, kw_only=True)
    allocated_resources: Resources = field(
        default_factory=Resources, compare=False, repr=False, kw_only=True
    )

    # The dotted path of the Job class that produces this artifact
    # ("artifacts.sources.jobs.SourceURLJob"), or None for an artifact nothing
    # produces, whose completion is its dependencies' files.
    #
    # A string, never the class: an artifact knowing its producer must not
    # mean importing it, since a family's jobs.py pulls in torch or numpy and
    # resolving, inspecting or drawing a graph must stay free of that. `job()`
    # is the one place the string becomes a class.
    producer: ClassVar[str | None]

    def __init_subclass__(cls, **kwargs):
        """Rejects a subclass that sets `producer` to anything but a string or None.

        Catches a wrong value when the class is defined, not when its job is
        first located. It cannot catch a missing one: bases like sources.Source
        leave `producer` for their subclasses to set, so __post_init__ checks
        presence once a class is actually instantiated.
        """
        super().__init_subclass__(**kwargs)
        producer = cls.__dict__.get("producer")
        if producer is not None and not isinstance(producer, str):
            raise TypeError(
                f"{cls.__name__} must set `producer` to the dotted path of the Job "
                f'class that produces it -- producer: ClassVar[str] = "artifacts.'
                f'sources.jobs.SourceURLJob" -- or to None if nothing does'
            )

    def __post_init__(self) -> None:
        """Checks the class sets `producer`, holds every field to the field
        schema, and replaces each parameter with what its manifest decodes to.

        Instantiation is the first point where a missing `producer` is an
        error. A set-valued dependency accepts any iterable, and two members
        surviving it with one artifact_path are conflicting definitions of one
        folder. A subclass with its own __post_init__ must call super(), or all
        of this is skipped.
        """
        cls = type(self)
        if not hasattr(cls, "producer"):
            raise TypeError(
                f"{cls.__name__} must set `producer` to the dotted path of "
                f"the Job class that produces it, or to None if nothing does"
            )
        deps, tagged = schema.dependencies(cls), schema.polymorphic(cls)
        for name, annotation in schema.annotations(cls).items():
            value, path = getattr(self, name), f"{cls.__qualname__}.{name}"
            if name not in deps:
                # a polymorphic field holds any subclass of its record, checked as its own class
                if name in tagged and isinstance(value, tuple(filter(is_dataclass, schema.arms(annotation)))):
                    annotation = type(value)
                object.__setattr__(self, name, manifest.normalize(value, annotation, path))
                continue
            container = deps[name]
            if container is frozenset:
                value = frozenset(value)
                object.__setattr__(self, name, value)
            elif container is tuple and not isinstance(value, tuple):
                raise TypeError(f"{path}: expected {annotation}, got {type(value).__name__}")
            arms = schema.arms(get_args(annotation)[0] if container else annotation)
            for item in value if container else (value,):
                if not isinstance(item, arms):
                    raise TypeError(f"{path}: expected {annotation}, got {type(item).__name__}")
            if container is frozenset:
                paths = Counter(item.artifact_path for item in value)
                repeated = sorted(str(p) for p, n in paths.items() if n > 1)
                if repeated:
                    raise ValueError(f"{cls.__name__}.{name} holds more than one definition at {', '.join(repeated)}")

    def deps(self) -> list[Artifact]:
        """Returns the artifacts this one is built from, one level deep:
        dependency fields in name order, set members sorted by artifact path.

        resolve walks this to build the graph, so the order must not depend
        on set iteration; sorting is what makes it deterministic.
        """
        return [
            dep
            for name, container in schema.dependencies(type(self)).items()
            for dep in manifest.members(getattr(self, name), container)
        ]

    def to_manifest(self) -> dict:
        """Returns this artifact as a manifest dict: type, commit, resources,
        own parameters, and a nested manifest per dependency.

        It is what manifest.json at artifact_path holds, and from_manifest
        rebuilds the artifact from it alone (see artifacts.core.manifest).
        """
        return manifest.to_manifest(self)

    def parameters(self) -> dict:
        """Returns this artifact's own fields as JSON: the `"parameters"` half
        of `to_manifest`, without the dependencies' manifests.

        For display (launcher/state.py), where every dependency's nested
        manifest would bury the fields that tell one artifact from another.
        """
        return manifest.parameters(self)

    @staticmethod
    def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
        """Returns the artifact a manifest describes and the tree under it, unbound.

        `memo`, one dict shared across calls, hands back the same instance for
        a manifest decoded before, so a subtree many artifacts share is decoded
        once (see artifacts.core.manifest.from_manifest).
        """
        return manifest.from_manifest(data, memo)

    @staticmethod
    def load(
        artifact_path: Path | str, root: Path | None = None, memo: dict[str, Artifact] | None = None
    ) -> Artifact:
        """Returns the artifact declared at `root / artifact_path`, always
        unbound, even when every one of its files is there; bind it to use them.

        Raises unless the manifest rebuilds an artifact whose artifact_path is
        the folder it was read from. A path is a pure function of parameters,
        so a disagreeing manifest was copied or renamed, and nothing beside it
        is what it claims. `memo` is `from_manifest`'s.
        """
        wanted = Path(artifact_path)
        if wanted.is_absolute() or ".." in wanted.parts:
            raise ValueError(f"artifact_path must stay under root, got {wanted}")
        path = _root(root) / wanted / MANIFEST
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"nothing declared at {path}") from None
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not readable JSON: {error}") from error
        try:
            artifact = Artifact.from_manifest(data, memo)
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} is not a readable manifest: {error!r}") from error
        if artifact.artifact_path != wanted:
            raise ValueError(f"{path} describes {artifact.artifact_path}, not {wanted}")
        return artifact

    def status(self, root: Path | None = None) -> Footprint:
        """Returns a Footprint of what `root` holds for this artifact, as files.

        A filesystem observation and nothing else: no manifest parsed, no
        definition compared, no lease read. What a footprint means (declared,
        partial, done, conflicting) is lab.declare's to say, against a request
        this method never sees.
        """
        at = _root(root)
        return Footprint(
            manifest=(at / self.artifact_path / MANIFEST).is_file(),
            outputs={desc: path.is_file() for desc, path in self.paths(at).items()},
            completion={path: path.is_file() for path in self.completion_paths(at)},
        )

    def bind(self, root: Path | None = None) -> Self:
        """Returns this artifact rebuilt from the manifest stored at `root`,
        with its files loaded onto it, ready to be used rather than just named.

        The stored manifest is the authority on what lives at a path: a
        definition disagreeing with it raises rather than being handed someone
        else's outputs, and the result carries the stored commit and resources.
        The object bind is called on is never modified.
        """
        at = _root(root)
        stored = Artifact.load(self.artifact_path, at)
        if stored != self:
            raise ValueError(
                f"{self.artifact_path} holds a different definition -- bind what is "
                f"declared there, or declare this one at its own path"
            )
        completion = stored.status(at).completion
        missing = [str(path) for path, there in completion.items() if not there]
        if missing:
            raise FileNotFoundError(f"{self.uid} is not built -- missing {missing}")
        stored._load(at)
        return stored

    def _load(self, root: Path) -> None:
        """Populates the in-memory state this artifact's own methods need from
        its files; a subclass hook that does nothing by default.

        Only bind calls it, after confirming every completion file is there,
        so an override may assume its files exist.
        """

    @property
    @abstractmethod
    def uid(self) -> str:
        """Returns a readable id derived from this artifact's own parameters,
        for naming it to people: logs, errors, the dashboard."""

    @property
    @abstractmethod
    def artifact_path(self) -> Path:
        """Returns the folder this artifact owns, relative to root.

        Must be a pure function of the identifying fields: equal artifacts
        share a folder, and load rejects a manifest found at any other path.
        """

    @property
    @abstractmethod
    def files(self) -> dict[str, str]:
        """Returns the filenames inside artifact_path, keyed by a short
        description of each (e.g. "weights", "config").

        Names, not paths, so paths() can join them to any root: the volume or
        a test folder.
        """

    def paths(self, root: Path) -> dict[str, Path]:
        """Returns each of `files` as a full path under `root`, keyed the same way.

        These are where this artifact's job may write; completion_paths, not
        this, says when it is done.
        """
        return {
            desc: root / self.artifact_path / name for desc, name in self.files.items()
        }

    def job(self) -> Job:
        """Returns the Job that produces this artifact, imported from `producer`.

        The one place that dotted path becomes a class. Its only caller is
        main.run_job, inside the container about to run it, so nothing that
        resolves, inspects or draws a graph imports a family's jobs.py.
        """
        if self.producer is None:
            raise ValueError(
                f"{type(self).__name__} declares no producer -- nothing runs it, its "
                f"completion follows the files it depends on"
            )
        return locate(self.producer)(self)

    def completion_paths(self, root: Path) -> list[Path]:
        """Returns the paths whose presence means this artifact is done.

        Its own files, for almost everything. A virtual artifact owning no
        bytes (mappeddataset.MappedDataSet) overrides this to point at its
        dependencies' files, which is why it is separate from `paths()`, the
        question of where this artifact may write.
        """
        return list(self.paths(root).values())

    def durable_progress(self, root: Path) -> dict | None:
        """Returns how far along the volume says this artifact is, as how many
        of its completion paths exist, or None when there is nothing to count.

        The dashboard shows this when no call is running, and the call's own
        self-report (system.runtime.Worker.progress) when one is. A subclass
        whose job writes intermediate state overrides it: SGD.Training reports
        its furthest checkpoint, which says more than "0 of 1 files".
        """
        completion = self.status(root).completion
        if not completion:
            return None
        return {
            "phase": "files",  # says what is being counted, as a job's own does
            "done": sum(completion.values()),
            "total": len(completion),
        }
