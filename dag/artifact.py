"""The base artifact model -- what an artifact IS, in general: its own
parameters plugged together with the artifacts it's built from, a folder
it owns, and a manifest that can rebuild it. No concrete artifact type lives
here; see sources/, tokenizers/, datasets/, models/* for those (spec.md,
section 2).

Every artifact owns one folder, `artifact_path`, and every file it comprises
lives directly in that folder -- its manifest.json included, written there by
the resolver when the job starts. `files` therefore holds bare filenames, not
paths, which makes writing outside the folder impossible.

An artifact is its own parameters with the artifacts it's built from plugged
in, all the way down to sources. `manifest()` writes exactly that tree out,
and `load()` reads it back: a manifest.json on disk and the object spelled out
in a notebook cell are interchangeable ways of naming the same artifact. That
makes the file the source of truth -- everything else (which job produces it,
whether it's done, whether its commit still matches) is derived on top.

An artifact has three lives, and this class carries all three:

    recipe    Artifact(params)         parameters, a folder, a manifest
    job       Job(artifact).run(root)  writes the files into that folder
    bound     artifact.bind(root)      a copy, with its files loaded onto it

`bind` is the seam between the first and the third. It checks the job's files
are there, loads whatever the artifact's own methods need onto a *copy*, and
returns that copy -- so a Tokenizer you bind is equal to the one you declared
(same parameters, same `==`) and is the thing you call encode() on, but it is
never the same object. The recipe you called `bind` on is left exactly as it
was, unbound, forever -- binding is not a mutation anyone can observe on the
value they already hold a reference to.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Self, Union, get_args, get_origin, get_type_hints

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)  # config.py is at the repo root
from config import get_git_commit

MANIFEST = "manifest.json"

ARTIFACTS: dict[str, type[Artifact]] = {}  # class name -> class, to read manifests


@cache
def _head() -> str:
    """The commit every artifact built in this process is stamped with. Read
    once, so a session can't stamp two commits onto one tree -- restart the
    kernel to pick up new work."""
    return get_git_commit()


def _digest(*parts: object) -> str:
    """Short stable id for a parameter set.

    _digest(("<pad>",), ["a", "b"]) -> "9f2b1c40a7"
    """
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _artifacts(value: object) -> list[Artifact]:
    """The artifacts held in one field's value -- one, several, or none.

    _artifacts((source_a, source_b)) -> [source_a, source_b]
    """
    if isinstance(value, Artifact):
        return [value]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, Artifact)]
    return []


def _encode(value: object) -> object:
    """A field value as JSON. Artifacts nest as whole manifests; tuples and
    plain dataclasses keep their shape and are rebuilt from the field's
    declared type on the way back.

    Every dict comes out sorted by key -- see Artifact.manifest.
    """
    if isinstance(value, Artifact):
        return value.manifest()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if is_dataclass(value):
        return {
            f.name: _encode(getattr(value, f.name))
            for f in sorted(fields(value), key=lambda f: f.name)
        }
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in sorted(value.items())}
    return value


def _decode(annotation: object, value: object) -> object:
    """Rebuild one field value from JSON, using the declared type to say what
    shape it comes back as -- JSON can't tell a tuple from a list, or a nested
    config from any other dict.

    _decode(tuple[str, ...], ["<pad>", "<unk>"]) -> ("<pad>", "<unk>")
    """
    if get_origin(annotation) in (UnionType, Union):
        if value is None:
            return None
        arms = [arm for arm in get_args(annotation) if arm is not type(None)]
        if all(isinstance(arm, type) and issubclass(arm, Artifact) for arm in arms):
            # a field that accepts more than one artifact type (e.g.
            # DataSet | MappedDataSet) isn't actually ambiguous: every
            # manifest already names its own concrete type, so which arm
            # applies is read off the value, not guessed from the
            # annotation -- same call this makes for a single-artifact field.
            return Artifact.from_manifest(value)
        # anything else -- `X | None`, or a union of non-artifact types --
        # is genuinely ambiguous: JSON can't say which arm on its own.
        if len(arms) != 1:
            raise TypeError(f"can't decode into {annotation}: more than one arm")
        return _decode(arms[0], value)
    if get_origin(annotation) is tuple:
        return tuple(_decode(get_args(annotation)[0], item) for item in value)
    if isinstance(annotation, type) and issubclass(annotation, Artifact):
        return Artifact.from_manifest(value)
    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(**{k: _decode(hints[k], v) for k, v in value.items()})
    return value


@dataclass(frozen=True)
class Resources:
    """What a job should be given to run this artifact -- absent (None)
    means "use Modal's platform default for that dimension." Never part of
    an artifact's identity (see Artifact.allocated_resources).

    Resources(gpu_type="A100", gpu_count=2)
    """

    cpu: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


# frozen: identity is its fields, so it must be hashable/immutable
@dataclass(frozen=True)
class Artifact(ABC):
    # The code the producing job runs, not part of what this artifact is:
    # compare=False keeps it out of ==/hash, so an artifact rebuilt from an old
    # manifest is still the same artifact, and a recommit doesn't rename
    # anything. repr=False keeps a nested tree's repr readable.
    commit: str = field(
        default_factory=_head, compare=False, repr=False, kw_only=True
    )  # kw_only: a defaulted base field would force defaults on every subclass

    # What to run this artifact's job with -- never identity (same reasoning
    # as commit, same compare=False mechanism) and never defaulted from
    # anything but Modal's own platform default: a non-default Resources
    # only ever comes from being passed explicitly at construction.
    allocated_resources: Resources = field(
        default_factory=Resources, compare=False, repr=False, kw_only=True
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in ARTIFACTS:
            raise TypeError(f"{cls.__name__} already registered as an artifact type")
        ARTIFACTS[cls.__name__] = cls  # registration at definition time, not lookup

    def deps(self) -> list[Artifact]:
        """The artifact-valued parameters, one level deep -- read off the
        fields, so no subclass keeps a dependency list in sync by hand."""
        return [a for f in fields(self) for a in _artifacts(getattr(self, f.name))]

    def manifest(self) -> dict:
        """This artifact as JSON, complete enough to rebuild it from nothing
        but the file: the type, the commit its jobs run from, its own
        parameters, and a manifest apiece for the artifacts it's built from.

        Source(name="odyssey", url="https://...") -> {
            "artifact": "Source", "commit": "d9479be",
            "parameters": {"name": "odyssey", "url": "https://..."},
            "dependencies": {},
        }

        The parameters/dependencies split is for reading; from_manifest plugs
        both back in as the one set of arguments they were.

        Every dict in the tree is key-ordered: the four keys here by the order
        they're written, everything below by sorting. Dict equality ignores key
        order, so nothing in Python would ever notice it drifting -- but the
        bytes on disk would, and reordering a dataclass's fields would rewrite
        every manifest that mentions it. Sorted, the file is a pure function of
        the artifact, which is what makes it diffable and hashable.
        """
        parameters: dict[str, object] = {}
        dependencies: dict[str, object] = {}
        for f in fields(self):
            if f.name in ("commit", "allocated_resources"):
                continue  # their own top-level keys, being about running and not state
            value = getattr(self, f.name)
            target = dependencies if _artifacts(value) else parameters
            target[f.name] = _encode(value)
        return {
            "artifact": type(self).__name__,
            "commit": self.commit,
            "allocated_resources": _encode(self.allocated_resources),
            "parameters": dict(sorted(parameters.items())),
            "dependencies": dict(sorted(dependencies.items())),
        }

    @staticmethod
    def from_manifest(manifest: dict) -> Artifact:
        """Rebuild the artifact a manifest describes, and the whole tree under
        it. Each node keeps the commit recorded for it, so a subtree produced
        by older code stays visibly older."""
        cls = ARTIFACTS[manifest["artifact"]]
        hints = get_type_hints(cls)
        plugged = {**manifest["parameters"], **manifest["dependencies"]}
        return cls(
            commit=manifest["commit"],
            allocated_resources=_decode(Resources, manifest["allocated_resources"]),
            **{name: _decode(hints[name], value) for name, value in plugged.items()},
        )

    @staticmethod
    def load(path: Path) -> Artifact:
        """Read an artifact back out of a manifest.json written by an earlier
        run, in place of spelling its parameters out again."""
        return Artifact.from_manifest(json.loads(Path(path).read_text()))

    @staticmethod
    def at(path: Path | str) -> Artifact:
        """The artifact whose folder is `path`, ready to use.

        The two ways to get hold of a built artifact are this and writing its
        parameters out then binding; they land on an equal artifact either
        way, though never the same object:

            Artifact.at(root / "tokenizers/bpe-1000-feeeeefa90")
            Tokenizer(vocab_size=1000, ...).bind(root)

        The manifest in the folder says which artifact this is; the folder
        itself says where its root is, since an artifact's path under a root is
        a pure function of its parameters -- peel those components back off and
        what's left is the root. Nothing about the path has to be told to it,
        so `/storage/...` inside a container and a copy pulled down from the
        volume read exactly the same.

        Comes back bound when its files are all there, plain when they aren't
        (declared but not built yet, or built halfway).
        """
        path = Path(path).resolve()
        if path.name == MANIFEST:
            path = path.parent
        artifact = Artifact.load(path / MANIFEST)

        relpath = artifact.artifact_path.parts
        if path.parts[-len(relpath) :] != relpath:
            # identity is location, so a folder that doesn't match the artifact
            # its manifest describes was renamed or copied out from under it --
            # and any root derived from it would be a guess
            raise ValueError(
                f"{path} is not where {artifact.uid} lives ({artifact.artifact_path})"
            )
        root = Path(*path.parts[: -len(relpath)])
        return artifact.bind(root) if artifact.exists(root) else artifact

    def bind(self, root: Path) -> Self:
        """A copy of this artifact with whatever its files hold loaded onto
        it, ready to be used rather than just named. This object -- the one
        `bind` was called on -- is left untouched; nothing about it changes,
        and it never becomes usable just because something else bound it.

        Every artifact gets the same check here: an artifact whose files
        aren't all there yet -- declared, maybe, but not built -- has nothing
        to bind to. What happens once that check passes is per-subclass, via
        `_load`: for most artifacts the files *are* the thing, so the default
        `_load` does nothing further. A subclass that stands for an object --
        tokenizers.bpe.Tokenizer, whose vocab and merges are what encode()
        runs on -- overrides `_load` to read its file and set up the state its
        own bound methods need, on the copy `_load` receives as `self`, never
        on the original. Either way this returns the copy, so
        `Tokenizer(...).bind(root).encode(text)` is one thought -- and so is
        `bound = tokenizer.bind(root)` followed by using `bound`, not
        `tokenizer`, from then on.
        """
        root = Path(root)
        missing = [
            str(path) for path in self.completion_paths(root) if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"{self.uid} is not built -- missing {missing}")
        bound = copy.copy(self)
        bound._load(root)
        return bound

    def _load(self, root: Path) -> None:
        """Subclass hook: populate whatever in-memory state this artifact's
        own methods need, once `bind` has confirmed the files are there. The
        default is a no-op -- most artifacts have nothing to load."""

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

    def completion_paths(self, root: Path) -> list[Path]:
        """The paths whose presence means this artifact is done -- what
        `bind`, `exists`, and `dag.resolve.inspect`'s done/partial check all
        ask about. Defaults to this artifact's own files, which is right for
        almost everything: an artifact is done when the files it declared
        are there. A virtual artifact that owns no bytes of its own --
        datasets.artifact.MappedDataSet, whose completeness is really "are
        the things I depend on done" -- overrides this to point at its
        dependencies' files instead. `paths()` itself is untouched by this:
        a file only ever lives inside its own artifact's folder; this is a
        separate question about what "done" means, not about where writing
        is allowed to happen.
        """
        return list(self.paths(root).values())

    def exists(self, root: Path) -> bool:
        return all(
            p.exists() for p in self.completion_paths(root)
        )  # a partial file set doesn't count as existing
