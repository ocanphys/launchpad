"""Artifacts: immutable definitions of what should exist, how they are identified, and how one
is used once built.

An artifact is a class and its typed fields. Fields annotated with artifact types are the
artifacts it is built from, its dependencies, and the rest are its parameters. Constructing one
reads no storage. The examples on this page use these three:

    odyssey = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
    bpe = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(odyssey,))
    tokens = TokenizedSource(tokenizer=bpe, source=odyssey)

**Families.** A family is a package under `artifacts/` with its artifact class in `__init__.py`
and its job in `jobs.py`. A concrete type declares its fields, derives `uid`, `artifact_path`
and `files` from them, and sets `producer`. Its module imports no torch or numpy.

**Validation.** Every artifact and param config is frozen, refuses unknown fields, NaN and
infinity, and is strict about types. An int in a float field is stored as that float, so
`max_norm=1` and `max_norm=1.0` are one artifact at one path; every other conversion is refused
with a ValidationError naming the field: `True` in an int field, `"0.5"` in a float field, a
list where a tuple is declared. `Artifact` and `ParamConfig` each set this config, since a
pydantic config does not reach the models nested in a field.

**Roots.** `load`, `status` and `bind` take a root, defaulting to STORAGE, which is for the top
of a notebook only: anything already holding a root passes it on, or it reads the volume while
everything around it reads a test folder.
"""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.pretty import pprint as rich_pprint

from artifacts.core import manifest, storage
from artifacts.core.locate import locate

if TYPE_CHECKING:
    # Type-only: job.py imports Artifact from here, so a real import would be
    # circular. job() needs only `producer` (a string) at runtime.
    from artifacts.core.job import Job


class ParamConfig(BaseModel):
    """A structured parameter of an artifact, such as a leg's training parameters or its model:
    validated, frozen and part of its artifact's identity, but never a dependency, and never a
    folder, manifest or job of its own.

        LoopConfig(evaluations=(), gpu_check_every=0)
        -> ValidationError: 1 validation error for LoopConfig, gpu_check_every: Input should be greater than 0
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class Resources(ParamConfig):
    """What a job should be given to run an artifact; None means Modal's platform default for that
    dimension. Never identity.

        Resources(gpu_type="A100", gpu_count=2, cpu=4)
        -> Resources(cpu=4.0, gpu_type='A100', gpu_count=2)
    """

    cpu: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


@dataclass(frozen=True)
class Footprint:
    """What one root holds for one artifact, as files: its manifest, its own outputs by description,
    and the files that make it complete by path. The last two differ only for a virtual artifact,
    which owns no outputs and is complete when its dependencies' files are.
    """

    manifest: bool
    outputs: dict[str, bool]
    completion: dict[Path, bool]

    @property
    def complete(self) -> bool:
        """Returns whether every file that makes this artifact done is there.

        Ask this, not `all(outputs.values())`, which is silently wrong for a virtual artifact.

        Returns:
            True when every completion path exists.
        """
        return all(self.completion.values())


def _check_parameter(annotation: object, where: str) -> None:
    """Raises unless a parameter's `annotation` holds only scalars, Literals,
    tuples and param configs: decoding turns every JSON array into a tuple,
    and hashing needs every value hashable."""
    containers = (list, dict, set, frozenset)
    if annotation in (Any, object, *containers) or get_origin(annotation) in containers:
        raise TypeError(
            f"{where}: annotated {annotation}; a parameter holds scalars, Literals, tuples and param configs"
        )
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
    """An artifact definition. Two artifacts agree in one of three ways, kept apart:

    - **identity**: equal `artifact_path`, what deduplication, leases, manifest locations and
      every map of artifacts key on;
    - **definition**: `==` and `hash`, comparing class, parameters and dependencies recursively,
      never the RECORDED fields or bound state, so an artifact rebuilt from an older manifest is
      still the same artifact. `resolve`, `bind` and `lab.declare` all ask this;
    - **provenance**: the RECORDED fields agree.

    Sharing a path is never agreement, so agreement is checked with `==`, never by comparing
    manifest text:

        a = SourceURL(name="odyssey", url="https://a.org/odyssey.txt")
        b = SourceURL(name="odyssey", url="https://b.org/odyssey.txt")
        a.artifact_path == b.artifact_path -> True
        a == b -> False
        a == SourceURL(name="odyssey", url="https://a.org/odyssey.txt", declaring_commit="abc1234") -> True

    A changed definition is a new construction, `type(a)(**dict(a) | changes)`, never
    `model_construct`, `model_copy(update=...)` or `object.__setattr__`: an artifact held in a
    field is never validated again, and a copy keeps cached properties such as a leg's
    `lineage_hash`, and with it the old path. `model_copy(update=...)` replaces a RECORDED field
    and nothing else.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    RECORDED: ClassVar[tuple[str, ...]] = ("declaring_commit", "allocated_resources", "tags")
    """The fields a manifest records but that never identify the artifact. Each is recorded once,
    at the root of the artifact's own manifest: a parent embeds its dependencies without them, so
    a dependency decoded from a parent has all three None. Once written, the commit and resources
    never change, and a matching redeclaration only appends the tags the stored ones lack."""

    declaring_commit: str | None = Field(default=None, repr=False)
    """The commit that declared this artifact's manifest, stamped by `lab.declare` whatever the
    object holds: HEAD, with `-dirty` when the tree has uncommitted changes; None until declared.
    It records the declaration, never the code a job runs, which is whatever is deployed."""
    allocated_resources: Resources | None = Field(default_factory=Resources, repr=False)
    """What its job is given to run; None on a dependency decoded from a parent's manifest,
    which records it at its own path."""
    tags: dict[str, list[str]] | None = Field(default_factory=dict, repr=False)
    """Labels to find it by, which the dashboard filters on, such as `{"experiment": ["lr_study"]}`;
    None on a dependency decoded from a parent's manifest.
    """

    producer: ClassVar[str | None]
    """The dotted path of the Job class that produces this artifact, or None for a virtual
    artifact: one that writes no files, is complete when its `completion_paths` exist, and is
    never launched."""

    @staticmethod
    def digest(*parts: object) -> str:
        """Returns a 10-character hash of `parts`, which subclasses build their `uid` and `artifact_path`
        from.

        Hashes canonical JSON, so equal parameters give an equal id in every process, -0.0 and 0.0
        included. Every part must be JSON-serializable.

            Artifact.digest(("<pad>",), ["a", "b"]) -> "54cacb94a0"

        Args:
            *parts: The values the id is derived from.

        Returns:
            The first 10 hex digits of the sha256 of their canonical JSON.
        """
        blob = json.dumps(manifest.canonical(parts), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @classmethod
    @cache
    def dependencies(cls) -> dict[str, type | None]:
        """Returns the fields that hold artifacts, sorted by name, each mapped to its container: tuple,
        frozenset, or None for a single artifact.

        A dependency is annotated `A`, `A | B | None`, `tuple[A, ...]` or `frozenset[A]`, with only
        artifact types inside: a tuple when order is part of what the artifact is, a frozenset when
        it is not. Construction makes any iterable a frozenset, so `sources=(a, b)` and
        `sources=(b, a)` are one definition, and refuses a set holding two members at one path. A
        field is classified by its annotation, never its value, so `sources=frozenset()` and
        `starting_checkpoint=None` are still dependencies. The first call raises TypeError naming any
        other field that holds a list, dict, set, Any, object, or an artifact: a parameter holds
        scalars, Literals, tuples and param configs, which the codec can rebuild.

            Tokenizer.dependencies() -> {"sources": frozenset}
            TokenizedSource.dependencies() -> {"source": None, "tokenizer": None}

        Returns:
            Each dependency field's name and its container.
        """
        found = {}
        for name in sorted(set(cls.model_fields) - set(cls.RECORDED)):
            annotation = cls.model_fields[name].annotation
            container = get_origin(annotation) if get_origin(annotation) in (tuple, frozenset) else None
            held = [
                arm
                for arm in manifest.arms(get_args(annotation)[0] if container else annotation)
                if arm is not type(None)
            ]
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
        sets = {
            name: frozenset(data[name])
            for name, c in cls.dependencies().items()
            if c is frozenset and name in data
        }
        return {**data, **sets}

    @model_validator(mode="after")
    def _producer_is_set(self) -> Self:
        if not isinstance(getattr(type(self), "producer", 0), str | None):
            raise TypeError(
                f"{type(self).__name__} must set `producer` to the dotted path of "
                f'the Job class that produces it -- producer: ClassVar[str] = "artifacts.'
                f'sources.jobs.SourceURLJob" -- or to None if nothing does'
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
                    raise ValueError(
                        f"{type(self).__name__}.{name} holds more than one definition at {', '.join(repeated)}"
                    )
        return self

    def _definition(self) -> tuple:
        names = sorted(n for n in type(self).model_fields if n not in type(self).RECORDED)
        return type(self), tuple((n, getattr(self, n)) for n in names)

    def __eq__(self, other: object) -> bool:
        """Returns whether `other` is the same definition: class, parameters and dependencies.

            SourceURL(name="a", url=u, declaring_commit="x") == SourceURL(name="a", url=u, declaring_commit="y") -> True

        Args:
            other: The object to compare with.

        Returns:
            True when other is an artifact of the same definition.
        """
        return isinstance(other, Artifact) and self._definition() == other._definition()

    def __hash__(self) -> int:
        """Returns a hash that agrees with `==`.

        Returns:
            The hash of the class and every field but the RECORDED ones.
        """
        return hash(self._definition())

    def deps(self) -> list[Artifact]:
        """Returns the artifacts this one is built from, one level deep: dependency fields in name
        order, set members sorted by artifact path, so the order never depends on set iteration.

            tokens.deps() -> [odyssey, bpe]

        Returns:
            The dependencies, in that order.
        """
        return [
            dep
            for name, container in type(self).dependencies().items()
            for dep in manifest.members(getattr(self, name), container)
        ]

    def to_manifest(self) -> dict:
        """Returns this artifact as the manifest at its `artifact_path`: class, recorded fields, own
        parameters, and a nested definition per dependency, which `from_manifest` rebuilds it from.

            odyssey.to_manifest()
            -> {"artifact": "artifacts.sources.SourceURL", "declaring_commit": None,
                "allocated_resources": {"cpu": None, "gpu_count": None, "gpu_type": None}, "tags": {},
                "parameters": {"name": "odyssey", "url": "https://example.org/odyssey.txt"},
                "dependencies": {}}

        Returns:
            The manifest, as a JSON-ready dict.
        """
        return manifest.to_manifest(self)

    def parameters(self) -> dict:
        """Returns this artifact's own fields as JSON, the `"parameters"` of `to_manifest`, for display
        where the dependencies' nested manifests would bury them.

            bpe.parameters() -> {"special_tokens": ["<pad>"], "vocab_size": 1000}

        Returns:
            Each own field's name and its JSON value.
        """
        return manifest.parameters(self)

    @staticmethod
    def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
        """Returns the artifact a manifest describes and the tree under it, unbound.

            Artifact.from_manifest(odyssey.to_manifest()) == odyssey -> True

        Args:
            data: A manifest, as to_manifest returns it.
            memo: Decoded artifacts keyed by manifest, shared across calls, so a subtree many
                artifacts share is decoded once; None for none.

        Returns:
            The artifact, unbound.
        """
        return manifest.from_manifest(data, memo)

    @staticmethod
    def load(
        artifact_path: Path | str, root: Path | None = None, memo: dict[str, Artifact] | None = None
    ) -> Artifact:
        """Returns the artifact declared at `root / artifact_path`, unbound even when its files are
        there.

            Artifact.load("sources/odyssey") -> SourceURL(name='odyssey', url='https://example.org/odyssey.txt')

        Args:
            artifact_path: The artifact's folder, relative to root; absolute or climbing out of root
                is refused.
            root: The storage root; None for STORAGE.
            memo: Passed on to from_manifest.

        Returns:
            The artifact the manifest describes, unbound.

        Raises:
            FileNotFoundError: Nothing is declared at the path.
            ValueError: Anything else keeps the manifest from rebuilding this artifact (unreadable
                JSON, an unknown class, a field that does not fit, or a path that is not the folder
                it was read from, so it was copied or renamed), naming the file, so a scan turns one
                bad manifest into one bad row.
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
        """Returns what `root` holds for this artifact, as files: no manifest parsed, no definition
        compared, no lease read.

            odyssey.status(Path("/storage"))
            -> Footprint(manifest=True, outputs={"raw text": True},
                         completion={Path("/storage/sources/odyssey/body.txt"): True})

        Args:
            root: The storage root to look in; None for STORAGE.

        Returns:
            Which of its manifest, outputs and completion files exist.
        """
        at = storage.root_of(root)
        return Footprint(
            manifest=(at / self.artifact_path / storage.MANIFEST).is_file(),
            outputs={desc: path.is_file() for desc, path in self.paths(at).items()},
            completion={path: path.is_file() for path in self.completion_paths(at)},
        )

    def pprint(self, expand_all: bool = True, indent_guides: bool = True) -> None:
        """Prints this artifact and everything it holds, by default fully expanded with indent guides.

            odyssey.pprint()

        Args:
            expand_all: Whether every nested value is expanded onto its own lines.
            indent_guides: Whether nesting levels are drawn with guide lines.
        """
        rich_pprint(self, expand_all=expand_all, indent_guides=indent_guides)

    def bind(self, root: Path | None = None) -> Self:
        """Returns the artifact stored at this one's path, its files loaded, ready to be used rather
        than just named.

        It loads the stored artifact, then runs the family's `_load` hook on that fresh copy, which
        reads only the files its artifact or its direct dependencies name and may bind those
        dependencies with the same root. What it attaches is runtime state, built fresh on every
        bind, and never enters the fields, identity or manifest. The result carries the stored
        recorded fields; the object bind is called on is never modified.

            odyssey.bind() -> SourceURL(name='odyssey', url='https://example.org/odyssey.txt')

        Args:
            root: The storage root to read from; None for STORAGE.

        Returns:
            The stored artifact, its files loaded.

        Raises:
            ValueError: The stored definition differs from this one.
            FileNotFoundError: A completion file is missing.
        """
        at = storage.root_of(root)
        stored = Artifact.load(self.artifact_path, at)
        if not isinstance(stored, type(self)) or stored != self:
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
        """Returns a readable id derived from this artifact's own parameters, for naming it to people:
        logs, errors, the dashboard.

            bpe.uid -> "bpe-1000-d44bf6191c"

        Returns:
            The id.
        """

    @property
    @abstractmethod
    def artifact_path(self) -> Path:
        """Returns the folder this artifact owns, relative to root: nonempty, normalized, and a pure
        function of the identifying fields.

        A folder under another's prefix is still its own artifact, and a lease covers one exact path.
        The encoding is a storage contract, so changing it needs a migration; readable names and
        short digests may collide, which the conflict checks on a shared path catch.

            tokens.artifact_path -> Path("tokenized/bpe-1000-d44bf6191c/odyssey")

        Returns:
            The folder, relative to root.
        """

    @property
    @abstractmethod
    def files(self) -> dict[str, str]:
        """Returns the filenames inside `artifact_path`, keyed by a short description of each: unique
        bare names, never `manifest.json`, so `paths` can join them to any root.

            odyssey.files -> {"raw text": "body.txt"}

        Returns:
            Each output's description and its filename.
        """

    def paths(self, root: Path) -> dict[str, Path]:
        """Returns each of `files` as a full path under `root`: where this artifact's job may write.

            odyssey.paths(Path("/storage")) -> {"raw text": Path("/storage/sources/odyssey/body.txt")}

        Args:
            root: The storage root to join under.

        Returns:
            Each output's description and its full path.
        """
        return {desc: root / self.artifact_path / name for desc, name in self.files.items()}

    def job(self) -> Job:
        """Returns the Job that produces this artifact, imported from `producer`.

        Its only caller is `main.run_job`, inside the container about to run it, so nothing that
        resolves, inspects or draws a graph imports a family's `jobs.py`.

            type(odyssey.job()) -> SourceURLJob

        Returns:
            The producing job, constructed on this artifact.
        """
        if self.producer is None:
            raise ValueError(
                f"{type(self).__name__} declares no producer -- nothing runs it, its "
                f"completion follows the files it depends on"
            )
        return locate(self.producer)(self)

    def completion_paths(self, root: Path) -> list[Path]:
        """Returns the paths whose presence means this artifact is done: its own files, except for a
        virtual artifact (`MappedDataSet`), which points at its dependencies' files.

            odyssey.completion_paths(Path("/storage")) -> [Path("/storage/sources/odyssey/body.txt")]

        Args:
            root: The storage root to join under.

        Returns:
            The full paths that must exist.
        """
        return list(self.paths(root).values())

    def durable_progress(self, root: Path) -> dict | None:
        """Returns how far along the volume says this artifact is, as how many of its completion paths
        exist, which the dashboard shows when no call is running.

        A job that writes intermediate state overrides it: a training leg reports its furthest
        checkpoint.

            odyssey.durable_progress(Path("/storage")) -> {"phase": "files", "done": 1, "total": 1}
            tokens.durable_progress(Path("/storage")) -> {"phase": "files", "done": 0, "total": 1}

        Args:
            root: The storage root to look in.

        Returns:
            The phase, how many are done and the total, or None with no completion paths.
        """
        completion = self.status(root).completion
        if not completion:
            return None
        return {
            "phase": "files",  # says what is being counted, as a job's own does
            "done": sum(completion.values()),
            "total": len(completion),
        }


def resolve(*roots: Artifact) -> list[Artifact]:
    """Returns every artifact the roots are built from and then each root, in one graph:
    dependencies before dependents, each path once. It reads no storage and decides nothing
    about running, so an artifact shared by several requests is one path, declared once and
    built once.

    It walks `deps()` depth-first. A path reached twice, from one root or from two, must hold an
    `==` definition, which compares the whole subtree, so a shared path cannot hide a conflicting
    leaf; the first object reaching a path is kept, recorded fields and all.

        resolve(tokens) -> [odyssey, bpe, tokens]
        resolve(tokens, bpe) -> [odyssey, bpe, tokens]

    Args:
        *roots: The requested artifacts.

    Returns:
        The graph's artifacts, each after everything it is built from.

    Raises:
        ValueError: Two different definitions share a path, or the graph has a cycle.
    """
    visited: dict[Path, Artifact] = {}
    active: list[Path] = []  # the current descent, for cycle detection

    def visit(node: Artifact) -> None:
        path = node.artifact_path
        if path in visited:
            if visited[path] != node:
                raise ValueError(f"different definitions at {path}")
            return
        if path in active:
            chain = " -> ".join(str(p) for p in (*active, path))
            raise ValueError(f"cycle: {chain}")
        active.append(path)
        for dep in node.deps():
            visit(dep)
        active.pop()
        visited[path] = node

    for root in roots:
        visit(root)
    return list(visited.values())
