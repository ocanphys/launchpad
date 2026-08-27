"""Artifact definitions -- what each thing IS and where it lives. No job,
registry or resolver knowledge belongs here (see spec.md, section 2).

Every artifact owns one folder, `artifact_path`, and every file it comprises
lives directly in that folder -- its resolve() manifest included, written
there by the resolver when the job starts. `files` therefore holds bare
filenames, not paths, which makes writing outside the folder impossible.

Only one identity hash survives, on Tokenizer: which sources trained it
can't be spelled out readably. Everything else names its folder in full.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "manifest.json"


def _digest(*parts: object) -> str:
    """Short stable id for a parameter set.

    _digest(("<pad>",), ["a", "b"]) -> "9f2b1c40a7"
    """
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


@dataclass(
    frozen=True
)  # frozen: identity is its fields, so it must be hashable/immutable
class Artifact(ABC):
    def deps(self) -> list[Artifact]:
        return []  # the artifact-valued fields, one level deep; none by default

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

    def deps(self) -> list[Artifact]:
        return list(self.sources)

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

    def deps(self) -> list[Artifact]:
        return [self.tokenizer, self.source]

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

    def deps(self) -> list[Artifact]:
        return [*self.train_set, *self.valid_set]

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
    """One run's pretraining output -- every checkpoint it writes, in one
    folder. Not one checkpoint: `step` says how far to train, and each
    checkpoint along the way is another file in the same artifact."""

    run_id: str
    dataset: DataSet
    tokenizer: Tokenizer
    config: PretrainingConfig
    step: int  # train up to (and checkpoint at) this step

    def deps(self) -> list[Artifact]:
        return [self.dataset, self.tokenizer]

    @property
    def uid(self) -> str:
        return f"{self.run_id}-pretraining"

    @property
    def artifact_path(self) -> Path:
        return Path("runs") / self.run_id / "pretraining"

    @property
    def checkpoint_steps(self) -> list[int]:
        """Every step this run checkpoints at, the final one always included.
        Both `files` and PretrainJob read it, so the rule lives in one place.

        step=250, checkpoint_every=100 -> [100, 200, 250]
        """
        every = self.config.checkpoint_every
        steps = list(range(every, self.step + 1, every))
        if not steps or steps[-1] != self.step:
            steps.append(self.step)
        return steps

    @property
    def files(self) -> dict[str, str]:
        return {f"step {s}": f"checkpoint_{s}.txt" for s in self.checkpoint_steps}
