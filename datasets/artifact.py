from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dag.artifact import Artifact
from sources.artifact import Source
from tokenizers.bpe import TokenizedSource, Tokenizer


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
