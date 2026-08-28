"""Artifact definitions -- what each thing IS and where it lives. No job,
registry or resolver knowledge belongs here (see spec.md, section 2).

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

Only one identity hash survives, on Tokenizer: which sources trained it
can't be spelled out readably. Everything else names its folder in full.
"""

from __future__ import annotations

import hashlib
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

sys.path.append(str(Path(__file__).resolve().parents[1]))  # config.py is at the root
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
        # `X | None` -- an optional field. Anything richer is ambiguous: JSON
        # can't say which arm.
        if value is None:
            return None
        arms = [arm for arm in get_args(annotation) if arm is not type(None)]
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


@dataclass(
    frozen=True
)  # frozen: identity is its fields, so it must be hashable/immutable
class Artifact(ABC):
    # The code the producing job runs, not part of what this artifact is:
    # compare=False keeps it out of ==/hash, so an artifact rebuilt from an old
    # manifest is still the same artifact, and a recommit doesn't rename
    # anything. repr=False keeps a nested tree's repr readable.
    commit: str = field(
        default_factory=_head, compare=False, repr=False, kw_only=True
    )  # kw_only: a defaulted base field would force defaults on every subclass

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
            if f.name == "commit":
                continue  # its own top-level key, being about code and not state
            value = getattr(self, f.name)
            target = dependencies if _artifacts(value) else parameters
            target[f.name] = _encode(value)
        return {
            "artifact": type(self).__name__,
            "commit": self.commit,
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
            **{name: _decode(hints[name], value) for name, value in plugged.items()},
        )

    @staticmethod
    def load(path: Path) -> Artifact:
        """Read an artifact back out of a manifest.json written by an earlier
        run, in place of spelling its parameters out again."""
        return Artifact.from_manifest(json.loads(Path(path).read_text()))

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

    def exists(self, root: Path) -> bool:
        return all(
            p.exists() for p in self.paths(root).values()
        )  # a partial file set doesn't count as existing


@dataclass(frozen=True)
class Source(Artifact):
    name: str
    url: str

    @property
    def uid(self) -> str:
        return self.name

    @property
    def artifact_path(self) -> Path:
        return Path("sources") / self.name

    @property
    def files(self) -> dict[str, str]:
        return {"raw text": "body.txt"}


@dataclass(frozen=True)
class Tokenizer(Artifact):
    vocab_size: int
    special_tokens: tuple[str, ...]
    sources: tuple[Source, ...]  # trained on these; order doesn't identify it
    kind: str

    @property
    def uid(self) -> str:
        # The one surviving hash. kind and vocab_size stay readable; the digest
        # covers what can't be (special_tokens, sources), so two tokenizers
        # differing only there don't share a folder. Sources are sorted --
        # *which* sources trained it, not the order they were listed in.
        vocab_label = (
            f"{self.vocab_size / 1000:.1f}k"
            if self.vocab_size > 1000
            else str(self.vocab_size)
        )
        digest = _digest(self.special_tokens, sorted(s.uid for s in self.sources))
        return f"{self.kind}-{vocab_label}-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("tokenizers") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {"tokenizer": "tokenizer.json"}


@dataclass(frozen=True)
class TokenizedSource(Artifact):
    tokenizer: Tokenizer
    source: Source

    @property
    def uid(self) -> str:
        return f"{self.tokenizer.uid}-{self.source.uid}"

    @property
    def artifact_path(self) -> Path:
        # nested under the tokenizer that produced it -- one `ls` shows every
        # source a given tokenizer has been run over
        return self.tokenizer.artifact_path / "bin" / self.source.uid

    @property
    def files(self) -> dict[str, str]:
        return {"tokens": "tokens.bin"}


@dataclass(frozen=True)
class DataSet(Artifact):
    # a memmap reserved for one run: its folder is fixed by run_id alone, so it
    # can be initialized once and reused as the sources behind it change.
    # Eventually this may be replaced by memmaps referring straight to the
    # sources' bin files.
    run_id: str
    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    @classmethod
    def from_sources(
        cls,
        run_id: str,
        tokenizer: Tokenizer,
        train_sources: list[Source],
        valid_sources: list[Source],
    ) -> DataSet:
        return cls(
            run_id=run_id,
            train_set=tuple(TokenizedSource(tokenizer, s) for s in train_sources),
            valid_set=tuple(TokenizedSource(tokenizer, s) for s in valid_sources),
        )

    @property
    def uid(self) -> str:
        return f"{self.run_id}-dataset"

    @property
    def artifact_path(self) -> Path:
        return Path("runs") / self.run_id / "dataset"

    @property
    def files(self) -> dict[str, str]:
        return {"training set": "train.bin", "validation set": "valid.bin"}


@dataclass(frozen=True)
class PretrainingConfig:
    hidden_size: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    seed: int = 0
    checkpoint_every: int = 100


@dataclass(frozen=True)
class Pretraining(Artifact):
    """One leg of a run's pretraining: training carried up to `step`, from
    `model` if given, or from scratch if not -- initializing a fresh model is
    the job's business, not a separate artifact of its own.

    A run is a chain of legs, each continuing the model the one before it
    produced. The chain exists because a container doesn't live forever, not
    because the result depends on where it's cut -- with seeds carried
    deterministically, 0->2000 and 0->1000->2000 land on the same weights.
    Where you cut is operational; that it was cut there is history, and the
    manifest records it.

    Two different legs may take the same `model` as their starting point --
    that's a fork, and it costs nothing beyond constructing both.
    """

    run_id: str
    dataset: DataSet
    tokenizer: Tokenizer
    config: PretrainingConfig
    step: int  # train up to (and checkpoint at) this step
    model: Pretraining | None = None  # the leg this one continues; None = fresh

    @property
    def uid(self) -> str:
        return f"{self.run_id}-pretraining-{self.step}"

    @property
    def artifact_path(self) -> Path:
        return Path("runs") / self.run_id / "pretraining" / f"step-{self.step}"

    @property
    def files(self) -> dict[str, str]:
        # Only the final checkpoint and a completion marker are declared. This
        # also writes intermediate checkpoints along the way, at
        # config.checkpoint_every -- undeclared, since which of them exists
        # after a preemption is the job's business, not something an artifact
        # can predict.
        #
        # progress is written last, after the checkpoint is durable: a
        # checkpoint can exist because the process died midway through writing
        # it.
        return {"checkpoint": "checkpoint.txt", "progress": "progress.json"}
