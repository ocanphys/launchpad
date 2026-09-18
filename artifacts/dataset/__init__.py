from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact, _digest
from artifacts.sources import Source
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers import Tokenizer

END_OF_TEXT = "<|endoftext|>"


def require_separator(name: str, sources: tuple[TokenizedSource, ...]) -> None:
    """Raises unless every tokenizer in a multi-source split can produce the
    separator that goes between its sources."""
    if len(sources) > 1 and any(
        END_OF_TEXT not in source.tokenizer.special_tokens for source in sources
    ):
        raise ValueError(
            f"DataSet.{name} with multiple sources requires {END_OF_TEXT!r} "
            "in each tokenizer's special_tokens"
        )


@dataclass(frozen=True)
class DataSet(Artifact):
    """A training set built by physically concatenating every source's
    tokens into train.bin/valid.bin, with <|endoftext|> between sources
    (artifacts/dataset/jobs.py's DataSetJob). `from_sources` builds one from
    a tokenizer and raw sources.
    Shared, not run-scoped: its bytes are a pure function of
    train_set/valid_set, so two runs asking for the same tokenizer and
    sources reuse the same folder instead of each paying for their own copy.

    artifacts.mappeddataset.MappedDataSet is a different kind of artifact,
    not a variant of this one -- it reads straight out of the sources' own
    bin files instead of copying them here at all, and lives in its own
    sibling package accordingly.
    """

    producer: ClassVar[str] = "artifacts.dataset.jobs.DataSetJob"

    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        require_separator("train_set", self.train_set)
        require_separator("valid_set", self.valid_set)

    @classmethod
    def from_sources(
        cls,
        tokenizer: Tokenizer,
        train_sources: tuple[Source, ...],
        valid_sources: tuple[Source, ...],
    ) -> DataSet:
        return cls(
            train_set=tuple(TokenizedSource(tokenizer, s) for s in train_sources),
            valid_set=tuple(TokenizedSource(tokenizer, s) for s in valid_sources),
        )

    @property
    def uid(self) -> str:
        # order is identifying here, same reasoning as MappedDataSet's uid:
        # concatenation order changes train.bin/valid.bin's actual contents,
        # so this must not sort train_set/valid_set away.
        digest = _digest(
            [ts.uid for ts in self.train_set], [ts.uid for ts in self.valid_set]
        )
        if len(self.train_set) > 1 or len(self.valid_set) > 1:
            # The separator changes the merged bytes and therefore their identity.
            digest = _digest(digest, END_OF_TEXT)
        return f"dataset-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("datasets") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {"training set": "train.bin", "validation set": "valid.bin"}

    # -- the bound view ------------------------------------------------------

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: one memmap per split over the bin its job
        wrote. object.__setattr__ because the dataclass is frozen -- `self`
        here is the object Artifact.bind read from the stored manifest, never
        the one bind() was called on."""
        import numpy as np  # local: only a process that binds pays for numpy

        for name, path in (
            ("_train_tokens", self.paths(root)["training set"]),
            ("_valid_tokens", self.paths(root)["validation set"]),
        ):
            # an empty file cannot be mapped, and an empty split is a legal one
            tokens = (
                np.memmap(path, dtype=np.uint16, mode="r")
                if path.stat().st_size
                else np.empty(0, dtype=np.uint16)
            )
            object.__setattr__(self, name, tokens)

    @property
    def bound(self) -> bool:
        return hasattr(self, "_train_tokens")

    def _require(self, doing: str) -> None:
        if not self.bound:
            raise RuntimeError(
                f"{self.uid} has no token arrays yet -- bind(root) to map the "
                f"train.bin/valid.bin its job wrote before {doing}"
            )

    @property
    def train_tokens(self):
        self._require("reading train_tokens")
        return self._train_tokens

    @property
    def valid_tokens(self):
        self._require("reading valid_tokens")
        return self._valid_tokens
