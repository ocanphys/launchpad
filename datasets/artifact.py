from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dag.artifact import Artifact, _digest
from sources.artifact import Source
from tokenizers.bpe import TokenizedSource, Tokenizer


@dataclass(frozen=True)
class DataSet(Artifact):
    """A training set built by physically concatenating every source's
    tokens into train.bin/valid.bin (datasets/job.py's DataSetJob). Shared,
    not run-scoped: its bytes are a pure function of train_set/valid_set, so
    two runs asking for the same tokenizer and sources reuse the same
    folder instead of each paying for their own copy.

    mappeddatasets.artifact.MappedDataSet is a different kind of artifact,
    not a variant of this one -- it reads straight out of the sources' own
    bin files instead of copying them here at all, and lives in its own
    top-level package accordingly.
    """

    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    @classmethod
    def from_sources(
        cls,
        tokenizer: Tokenizer,
        train_sources: list[Source],
        valid_sources: list[Source],
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
        return f"dataset-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("datasets") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {"training set": "train.bin", "validation set": "valid.bin"}
