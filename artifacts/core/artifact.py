"""The base artifact model: parameters plugged together with the artifacts
they're built from, a folder each artifact owns, and a manifest that rebuilds
it. Concrete types live in sources/, tokenizers/, tokenized/, dataset/,
mappeddataset/, stages/*.

An artifact has three lives, and this class carries all three:

    recipe    Artifact(params)         parameters, a folder, a manifest
    job       Job(artifact).run(root)  writes the files into that folder
    bound     artifact.bind(root)      the stored declaration, files loaded

`load`, `status` and `bind` take a root, defaulting to STORAGE. That default
is for the top of a notebook only: anything already holding a root -- a job, a
`_load` hook -- passes it on, or it silently reads the volume while everything
around it reads a test folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

from artifacts.core import manifest
from artifacts.core.locate import locate

if TYPE_CHECKING:
    # Type-only: job.py imports Artifact from here, so a real import would be
    # circular. job() needs only `producer` (a string) at runtime.
    from artifacts.core.job import Job

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)  # config.py is at the repo root -- artifacts/core/artifact.py is two levels down
from config import LAB_COMMIT_ENV, STORAGE, get_git_commit

MANIFEST = "manifest.json"


@cache
def _head() -> str:
    """The commit every artifact built in this process is stamped with.

    Read once, so a session can't stamp two commits onto one tree -- restart
    the kernel to pick up new work. The lab container has no git binary and no
    .git; main.py bakes the commit in under LAB_COMMIT_ENV instead.
    """
    override = os.environ.get(LAB_COMMIT_ENV)
    return override if override else get_git_commit()


def _digest(*parts: object) -> str:
    """Short stable id for a parameter set.

    _digest(("<pad>",), ["a", "b"]) -> "9f2b1c40a7"
    """
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _root(root: Path | None) -> Path:
    """The root a call reads, STORAGE when none is given.

    A Path, and absolute. A string is rejected rather than converted, so one
    type reaches every join, glob and relative_to downstream; a relative root
    is discarded by `root / artifact_path` rather than failing, so the mistake
    would surface as a missing file somewhere else entirely.
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
        """Whether every file that makes this artifact done is there.

        Ask this, not `all(outputs.values())`: the two name the same files for
        everything that owns its bytes, and differ for exactly the artifacts
        where getting it wrong is silent.
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

    # The dotted path of the Job class that produces this artifact --
    # "artifacts.sources.jobs.SourceURLJob" -- or None for an artifact nothing
    # produces, whose completion is its dependencies' files.
    #
    # A string, and never anything more, is the point: an artifact knowing its
    # producer must not mean an artifact importing it, since a family's jobs.py
    # pulls in torch or numpy and resolving, inspecting or drawing a graph must
    # stay free of that. `job()` is the one place the string becomes a class.
    producer: ClassVar[str | None]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # a base other classes fill in (sources.Source) sets nothing; a class
        # that does set it must set a string or None
        producer = cls.__dict__.get("producer")
        if producer is not None and not isinstance(producer, str):
            raise TypeError(
                f"{cls.__name__} must set `producer` to the dotted path of the Job "
                f'class that produces it -- producer: ClassVar[str] = "artifacts.'
                f'sources.jobs.SourceURLJob" -- or to None if nothing does'
            )

    def __post_init__(self) -> None:
        """Accepts any iterable for a set-valued dependency, and rejects two
        definitions landing on one path inside one.

        A set collapses members that agree, so what survives a collision is
        members that disagree while sharing a folder. A subclass with its own
        __post_init__ must call super().
        """
        if not hasattr(type(self), "producer"):
            raise TypeError(
                f"{type(self).__name__} must set `producer` to the dotted path of "
                f"the Job class that produces it, or to None if nothing does"
            )
        for name, container in manifest.dependencies(type(self)).items():
            if container is not frozenset:
                continue
            members = frozenset(getattr(self, name))
            object.__setattr__(self, name, members)
            paths = [item.artifact_path for item in members]
            repeated = sorted({str(p) for p in paths if paths.count(p) > 1})
            if repeated:
                raise ValueError(
                    f"{type(self).__name__}.{name} holds more than one definition "
                    f"at {', '.join(repeated)}"
                )

    def deps(self) -> list[Artifact]:
        """The artifacts this one is built from, one level deep, fields by
        name and sets by artifact path."""
        found: list[Artifact] = []
        for name, container in manifest.dependencies(type(self)).items():
            value = getattr(self, name)
            if container is None:
                if value is not None:
                    found.append(value)
            elif container is frozenset:
                found.extend(sorted(value, key=lambda a: a.artifact_path.as_posix()))
            else:
                found.extend(value)
        return found

    def to_manifest(self) -> dict:
        """This artifact as a dictionary, enough to rebuild it from nothing
        else -- see artifacts.core.manifest."""
        return manifest.to_manifest(self)

    def parameters(self) -> dict:
        """This artifact's own fields as JSON: the `"parameters"` half of
        `to_manifest`, without the dependencies' manifests."""
        return manifest.parameters(self)

    @staticmethod
    def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
        """The artifact a manifest describes and the tree under it, unbound.

        `memo`, one dict across many calls, hands back the same instance for a
        manifest seen before instead of decoding it again (see
        artifacts.core.manifest.from_manifest).
        """
        return manifest.from_manifest(data, memo)

    @staticmethod
    def load(
        artifact_path: Path | str, root: Path | None = None, memo: dict[str, Artifact] | None = None
    ) -> Artifact:
        """The artifact declared at `root / artifact_path`, always unbound --
        even when every one of its files is there. Bind it to use them.

        Raises unless the manifest rebuilds an artifact that belongs at the
        folder it was read from: a path is a pure function of parameters, so a
        manifest disagreeing with its own folder was copied or renamed, and
        nothing beside it is about what it claims. `memo` is `from_manifest`'s.
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
        """What `root` holds for this artifact, as files.

        A filesystem observation and nothing else: no manifest parsed, no
        definition compared, no lease read. What a footprint means -- declared,
        partial, done, conflicting -- is declaration's to say, against a
        request this has no notion of.
        """
        at = _root(root)
        return Footprint(
            manifest=(at / self.artifact_path / MANIFEST).is_file(),
            outputs={desc: path.is_file() for desc, path in self.paths(at).items()},
            completion={path: path.is_file() for path in self.completion_paths(at)},
        )

    def bind(self, root: Path | None = None) -> Self:
        """This artifact with what its files hold loaded onto it, ready to be
        used rather than just named. The object bind was called on is left
        untouched, and never becomes usable because something else bound it.

        What comes back is built from the declaration stored at `root`, so it
        carries that declaration's commit and resources: the manifest at an
        artifact's own path is the authority on what lives there, and a
        definition disagreeing with it raises rather than being handed
        somebody else's outputs.
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
        """Subclass hook: populate whatever in-memory state this artifact's own
        methods need, once bind has confirmed the files are there. Most
        artifacts have nothing to load."""

    @property
    @abstractmethod
    def uid(self) -> str:
        """Readable id derived from this artifact's own parameters."""

    @property
    @abstractmethod
    def artifact_path(self) -> Path:
        """The folder this artifact owns, relative to root."""

    @property
    @abstractmethod
    def files(self) -> dict[str, str]:
        """Filenames inside artifact_path, keyed by a short description of
        what each file is (e.g. "weights", "config"). Names, not paths."""

    def paths(self, root: Path) -> dict[str, Path]:
        return {
            desc: root / self.artifact_path / name for desc, name in self.files.items()
        }

    def job(self) -> Job:
        """The Job that produces this artifact, imported from `producer`.

        The one place that dotted path becomes a class, and main.run_job --
        inside the container about to run it -- is its only caller, so nothing
        that resolves, inspects or draws imports a family's jobs.py.
        """
        if self.producer is None:
            raise ValueError(
                f"{type(self).__name__} declares no producer -- nothing runs it, its "
                f"completion follows the files it depends on"
            )
        return locate(self.producer)(self)

    def completion_paths(self, root: Path) -> list[Path]:
        """The paths whose presence means this artifact is done.

        Its own files, for almost everything. A virtual artifact owning no
        bytes -- mappeddataset.MappedDataSet -- points at its dependencies'
        files instead. Separate from `paths()`, which stays the question of
        where this artifact may write.
        """
        return list(self.paths(root).values())

    def durable_progress(self, root: Path) -> dict | None:
        """How far along the volume says this artifact is: how many of the
        paths that make it done are there. None when there is nothing to count.

        The dashboard shows this when no call is running and the call's own
        self-report (system.runtime.Worker.progress) when one is. A subclass
        whose job writes intermediate state overrides it -- SGD.Training
        reports its furthest checkpoint, which says more than "0 of 1 files".
        """
        completion = self.status(root).completion
        if not completion:
            return None
        return {
            "phase": "files",  # says what is being counted, as a job's own does
            "done": sum(completion.values()),
            "total": len(completion),
        }
