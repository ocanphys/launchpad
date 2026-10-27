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
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifacts.core import manifest, storage
from artifacts.core.locate import locate
from config import LAB_COMMIT_ENV, get_git_commit

if TYPE_CHECKING:
    # Type-only: job.py imports Artifact from here, so a real import would be
    # circular. job() needs only `producer` (a string) at runtime.
    from artifacts.core.job import Job


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

    Hashes canonical JSON, so equal parameters give an equal id in every
    process, -0.0 and 0.0 included. Every part must be JSON-serializable.

    _digest(("<pad>",), ["a", "b"]) -> "54cacb94a0"
    """
    blob = json.dumps(manifest.canonical(parts), separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


# Strict: no str as int, bool as int or list as tuple, though an int in a float
# field becomes a float. Each base sets it, since a config does not reach the
# models nested in a field.
STRICT = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class ParamConfig(BaseModel):
    """A validated, frozen parameter of an artifact: never a dependency, and
    never a folder or manifest of its own. Subclasses declare their fields and
    the constraints on them; construction refuses a value that breaks one.

    LoopConfig(val_every=0, gpu_check_every=200) -> ValidationError (val_every: greater than 0)
    """

    model_config = STRICT


class Resources(ParamConfig):
    """What a job should be given to run this artifact; None means Modal's
    platform default for that dimension. Never identity.

    Resources(gpu_type="A100", gpu_count=2, cpu=4)
      -> Resources(cpu=4.0, gpu_type='A100', gpu_count=2)
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


def _check_parameter(annotation: object, where: str) -> None:
    """Raises unless a parameter's `annotation` holds only scalars, Literals,
    tuples and param configs: decoding turns every JSON array into a tuple,
    and hashing needs every value hashable."""
    if annotation in (Any, object, list, dict, set, frozenset) or get_origin(annotation) in (list, dict, set, frozenset):
        raise TypeError(f"{where}: annotated {annotation}; a parameter holds scalars, Literals, tuples and param configs")
    if isinstance(annotation, type) and issubclass(annotation, Artifact):
        raise TypeError(
            f"{where}: artifact hidden in a parameter; a dependency is annotated "
            f"A, A | B | None, tuple[A, ...] or frozenset[A]"
        )
    if isinstance(annotation, type) and issubclass(annotation, ParamConfig):
        for name, info in annotation.model_fields.items():
            _check_parameter(info.annotation, f"{where}.{name}")
    elif get_origin(annotation) is not Literal:
        for arg in get_args(annotation):  # a union's arms, a tuple's items
            if arg is not Ellipsis:
                _check_parameter(arg, where)


class Artifact(BaseModel):
    """An artifact definition: its fields annotated with artifact types are
    the artifacts it is built from, and every other field is a parameter.

    `==` and `hash` compare the definition: class, parameters and
    dependencies, recursively, never `commit` or `allocated_resources`, so an
    artifact rebuilt from an older manifest is still the same artifact.

    SourceURL(name="odyssey", url="https://example.org/odyssey.txt").artifact_path -> Path("sources/odyssey")
    """

    model_config = STRICT

    # Neither of these says which artifact this is (manifest.RECORDED).
    commit: str = Field(default_factory=_head, repr=False)
    allocated_resources: Resources = Field(default_factory=Resources, repr=False)

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
        leave `producer` for their subclasses to set, so a validator checks
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

    @classmethod
    @cache
    def dependencies(cls) -> dict[str, type | None]:
        """Returns the fields that hold artifacts, sorted by name, each mapped
        to its container: tuple, frozenset, or None for a single artifact.

        Read off annotations, never values, so an empty `sources=frozenset()`
        is still a dependency. The first call also checks every other field is
        a parameter the codec can rebuild, and raises TypeError naming one
        that is not; it runs at first use, once forward references resolve.

        Tokenizer.dependencies() -> {"sources": frozenset}
        """
        found = {}
        for name in sorted(cls.model_fields):
            annotation = cls.model_fields[name].annotation
            container = get_origin(annotation) if get_origin(annotation) in (tuple, frozenset) else None
            held = [arm for arm in manifest.arms(get_args(annotation)[0] if container else annotation) if arm is not type(None)]
            if held and all(isinstance(arm, type) and issubclass(arm, Artifact) for arm in held):
                found[name] = container
            else:
                _check_parameter(annotation, f"{cls.__qualname__}.{name}")
        return found

    @model_validator(mode="before")
    @classmethod
    def _sets_from_iterables(cls, data: Any) -> Any:
        """Returns `data` with every set-valued dependency made a frozenset,
        so `sources=(a, b)` is accepted as strict validation alone would not."""
        if not isinstance(data, dict):
            return data
        sets = {name: frozenset(data[name]) for name, c in cls.dependencies().items() if c is frozenset and name in data}
        return {**data, **sets}

    @model_validator(mode="after")
    def _producer_is_set(self) -> Self:
        if not hasattr(type(self), "producer"):
            raise TypeError(
                f"{type(self).__name__} must set `producer` to the dotted path of "
                f"the Job class that produces it, or to None if nothing does"
            )
        return self

    @model_validator(mode="after")
    def _one_definition_per_path(self) -> Self:
        """Refuses a set holding two members at one path: unequal, so the set
        keeps both, and two conflicting definitions of one folder."""
        for name, container in type(self).dependencies().items():
            if container is frozenset:
                paths = Counter(member.artifact_path for member in getattr(self, name))
                repeated = sorted(str(p) for p, n in paths.items() if n > 1)
                if repeated:
                    raise ValueError(f"{type(self).__name__}.{name} holds more than one definition at {', '.join(repeated)}")
        return self

    def _definition(self) -> tuple:
        names = sorted(n for n in type(self).model_fields if n not in manifest.RECORDED)
        return type(self), tuple((n, getattr(self, n)) for n in names)

    def __eq__(self, other: object) -> bool:
        """Returns whether `other` is the same definition: class, parameters and dependencies.

        SourceURL(name="a", url=u, commit="x") == SourceURL(name="a", url=u, commit="y") -> True
        """
        return isinstance(other, Artifact) and self._definition() == other._definition()

    def __hash__(self) -> int:
        """Returns a hash that agrees with `==`."""
        return hash(self._definition())

    def deps(self) -> list[Artifact]:
        """Returns the artifacts this one is built from, one level deep:
        dependency fields in name order, set members sorted by artifact path.

        resolve walks this to build the graph, so the order must not depend
        on set iteration; sorting is what makes it deterministic.

        TokenizedSource(tokenizer=bpe, source=odyssey).deps() -> [odyssey, bpe]
        """
        return [
            dep
            for name, container in type(self).dependencies().items()
            for dep in manifest.members(getattr(self, name), container)
        ]

    def to_manifest(self) -> dict:
        """Returns this artifact as a manifest dict: class, commit, resources,
        own parameters, and a nested manifest per dependency.

        It is what manifest.json at artifact_path holds, and from_manifest
        rebuilds the artifact from it alone (see artifacts.core.manifest).

        odyssey.to_manifest()["parameters"] -> {"name": "odyssey", "url": "https://example.org/odyssey.txt"}
        """
        return manifest.to_manifest(self)

    def parameters(self) -> dict:
        """Returns this artifact's own fields as JSON: the `"parameters"` half
        of `to_manifest`, without the dependencies' manifests.

        For display (launcher/state.py), where every dependency's nested
        manifest would bury the fields that tell one artifact from another.

        bpe.parameters() -> {"special_tokens": ["<pad>"], "vocab_size": 1000}
        """
        return manifest.parameters(self)

    @staticmethod
    def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
        """Returns the artifact a manifest describes and the tree under it, unbound.

        `memo`, one dict shared across calls, hands back the same instance for
        a manifest decoded before, so a subtree many artifacts share is decoded
        once (see artifacts.core.manifest.from_manifest).

        Artifact.from_manifest(odyssey.to_manifest()) == odyssey -> True
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

        Artifact.load("sources/odyssey") -> SourceURL(name='odyssey', url='https://example.org/odyssey.txt')
        """
        wanted = Path(artifact_path)
        data = storage.read_manifest(root, wanted)
        path = storage.root_of(root) / wanted / storage.MANIFEST
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

        odyssey.status(root) -> Footprint(manifest=True, outputs={"raw text": True}, completion={...: True})
        """
        at = storage.root_of(root)
        return Footprint(
            manifest=(at / self.artifact_path / storage.MANIFEST).is_file(),
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

        bound = bpe.bind(root); bound.decode(bound.encode("hello")) -> "hello"
        """
        at = storage.root_of(root)
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

        odyssey.paths(Path("/storage")) -> {"raw text": Path("/storage/sources/odyssey/body.txt")}
        """
        return {
            desc: root / self.artifact_path / name for desc, name in self.files.items()
        }

    def job(self) -> Job:
        """Returns the Job that produces this artifact, imported from `producer`.

        The one place that dotted path becomes a class. Its only caller is
        main.run_job, inside the container about to run it, so nothing that
        resolves, inspects or draws a graph imports a family's jobs.py.

        odyssey.job() -> SourceURLJob(odyssey)
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

        odyssey.completion_paths(Path("/storage")) -> [Path("/storage/sources/odyssey/body.txt")]
        """
        return list(self.paths(root).values())

    def durable_progress(self, root: Path) -> dict | None:
        """Returns how far along the volume says this artifact is, as how many
        of its completion paths exist, or None when there is nothing to count.

        The dashboard shows this when no call is running, and the call's own
        self-report (system.runtime.Worker.progress) when one is. A subclass
        whose job writes intermediate state overrides it: SGD.Training reports
        its furthest checkpoint, which says more than "0 of 1 files".

        odyssey.durable_progress(root) -> {"phase": "files", "done": 1, "total": 1}
        """
        completion = self.status(root).completion
        if not completion:
            return None
        return {
            "phase": "files",  # says what is being counted, as a job's own does
            "done": sum(completion.values()),
            "total": len(completion),
        }
