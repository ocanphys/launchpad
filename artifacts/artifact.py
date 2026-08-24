"""Artifact definitions -- what each thing IS and where it lives. No job,
registry or resolver knowledge belongs here (see spec.md, section 2).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


def _digest(*parts: object) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:10]  # short, stable id for a parameter set


@dataclass(frozen=True)  # frozen: identity is its fields, so it must be hashable/immutable
class Artifact(ABC):
    def refs(self) -> list["Artifact"]:
        return []  # artifact-valued parameters, one level deep; none by default

    @property
    @abstractmethod
    def uid(self) -> str:
        """Stable id derived from this artifact's own parameters -- every
        subclass must define this itself; there's no generic default."""

    def relpath(self) -> Path:
        raise NotImplementedError

    def path(self, root: Path) -> Path:
        return root / self.relpath()

    def exists(self, root: Path) -> bool:
        return self.path(root).exists()


@dataclass(frozen=True)
class Source(Artifact):
    name: str
    url: str

    @property
    def uid(self) -> str:
        return self.name  # not digested -- guardrails against name collisions come later

    def relpath(self) -> Path:
        return Path(self.name) / "body.txt"


@dataclass(frozen=True)
class Tokenizer(Artifact):
    vocab_size: int
    special_tokens: tuple[str, ...]
    sources: tuple[Source, ...]  # trained on these, in order
    kind: str = "bpe"

    def refs(self) -> list[Artifact]:
        return list(self.sources)

    @property
    def uid(self) -> str:
        # kind and vocab_size stay readable; the hash covers the rest (special_tokens,
        # sources) so two tokenizers that only differ there don't collide.
        vocab_label = f"{self.vocab_size / 1000:.1f}k" if self.vocab_size > 1000 else str(self.vocab_size)
        digest = _digest(self.special_tokens, tuple(s.uid for s in self.sources))
        return f"{self.kind}-{vocab_label}-{digest}"

    def relpath(self) -> Path:
        return Path("tokenizers") / self.uid / "tokenizer.json"


@dataclass(frozen=True)
class TokenizedSource(Artifact):
    tokenizer: Tokenizer
    source: Source

    def refs(self) -> list[Artifact]:
        return [self.tokenizer, self.source]

    @property
    def uid(self) -> str:
        return f"{self.tokenizer.uid}-{self.source.uid}"  # not digested -- same convention as Source

    def relpath(self) -> Path:
        tokenizer_dir = self.tokenizer.relpath().parent  # tokenizers/{tokenizer_uid}
        return tokenizer_dir / "bin" / f"{self.source.uid}.bin"
