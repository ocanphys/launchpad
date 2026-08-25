"""Artifact definitions -- what each thing IS and where it lives. No job,
registry or resolver knowledge belongs here (see spec.md, section 2).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


def _digest(*parts: object) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[
        :10
    ]  # short, stable id for a parameter set


@dataclass(
    frozen=True
)  # frozen: identity is its fields, so it must be hashable/immutable
class Artifact(ABC):
    def deps(self) -> list["Artifact"]:
        return []  # artifact-valued parameters, one level deep; none by default

    @property
    @abstractmethod
    def uid(self) -> str:
        """Stable id derived from this artifact's own parameters -- every
        subclass must define this itself; there's no generic default."""

    @property
    def files(self) -> dict[str, Path]:
        """Every file this artifact comprises, relative to root, keyed by a
        short description of what that file is (e.g. "weights", "config")."""
        raise NotImplementedError

    def paths(self, root: Path) -> dict[str, Path]:
        return {desc: root / p for desc, p in self.files.items()}

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
        return (
            self.name
        )  # not digested -- guardrails against name collisions come later

    @property
    def files(self) -> dict[str, Path]:
        return {"body": Path(self.name) / "body.txt"}


@dataclass(frozen=True)
class Tokenizer(Artifact):
    vocab_size: int
    special_tokens: tuple[str, ...]
    sources: tuple[Source, ...]  # trained on these, in order
    kind: str = "bpe"

    def deps(self) -> list[Artifact]:
        return list(self.sources)

    @property
    def uid(self) -> str:
        # kind and vocab_size stay readable; the hash covers the rest (special_tokens,
        # sources) so two tokenizers that only differ there don't collide.
        vocab_label = (
            f"{self.vocab_size / 1000:.1f}k"
            if self.vocab_size > 1000
            else str(self.vocab_size)
        )
        digest = _digest(self.special_tokens, tuple(s.uid for s in self.sources))
        return f"{self.kind}-{vocab_label}-{digest}"

    @property
    def files(self) -> dict[str, Path]:
        return {"tokenizer": Path("tokenizers") / self.uid / "tokenizer.json"}


@dataclass(frozen=True)
class TokenizedSource(Artifact):
    tokenizer: Tokenizer
    source: Source

    def deps(self) -> list[Artifact]:
        return [self.tokenizer, self.source]

    @property
    def uid(self) -> str:
        return f"{self.tokenizer.uid}-{self.source.uid}"  # not digested -- same convention as Source

    @property
    def files(self) -> dict[str, Path]:
        tokenizer_dir = self.tokenizer.files[
            "tokenizer"
        ].parent  # tokenizers/{tokenizer_uid}
        return {"tokens": tokenizer_dir / "bin" / f"{self.source.uid}.bin"}


@dataclass(frozen=True)
class DataSet(Artifact):
    name: str
    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    @classmethod
    def from_sources(
        cls,
        name: str,
        tokenizer: Tokenizer,
        train_sources: list[Source],
        valid_sources: list[Source],
    ) -> DataSet:
        return cls(
            name=name,
            train_set=tuple(TokenizedSource(tokenizer, s) for s in train_sources),
            valid_set=tuple(TokenizedSource(tokenizer, s) for s in valid_sources),
        )

    def deps(self) -> list[Artifact]:
        return [*self.train_set, *self.valid_set]

    @property
    def uid(self) -> str:
        digest = _digest(self.train_set, self.valid_set)
        return f"{self.name}-{digest}"

    @property
    def files(self) -> dict[str, Path]:
        return {
            "training set": Path("datasets") / self.uid / "train.bin",
            "validation set": Path("datasets") / self.uid / "valid.bin",
        }
