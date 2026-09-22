"""The pretraining stage: __init__.py for the Pretraining leg, jobs.py for
its batches, loss and evaluation on top of the shared leg and loop in
artifacts.core.SGD. Which model it trains is the leg's `model` parameter, a
package under models/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import _digest
from artifacts.core.SGD.training import Training
from artifacts.dataset import DataSet
from artifacts.mappeddataset import MappedDataSet
from artifacts.tokenizers import Tokenizer


@dataclass(frozen=True, kw_only=True)
class Pretraining(Training):
    """A pretraining leg: next-token prediction over the dataset's token
    stream."""

    producer: ClassVar[str] = "artifacts.stages.pretraining.jobs.PretrainJob"

    dataset: DataSet | MappedDataSet
    tokenizer: Tokenizer
    starting_checkpoint: Pretraining | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # a model speaks one tokenizer for life: the one its dataset was
        # tokenized with, and the one every earlier leg trained with
        if self.model_parameters.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"model_parameters.vocab_size ({self.model_parameters.vocab_size}) does not "
                f"match tokenizer {self.tokenizer.uid}'s vocab_size ({self.tokenizer.vocab_size})"
            )
        sources = (*self.dataset.train_set, *self.dataset.valid_set)
        if any(ts.tokenizer.uid != self.tokenizer.uid for ts in sources):
            raise ValueError(f"dataset {self.dataset.uid} is not tokenized with {self.tokenizer.uid}")
        if self.starting_checkpoint and self.starting_checkpoint.tokenizer.uid != self.tokenizer.uid:
            raise ValueError(
                f"tokenizer {self.tokenizer.uid} differs from starting_checkpoint's "
                f"{self.starting_checkpoint.tokenizer.uid}"
            )

    @cached_property
    def lineage_hash(self) -> str:
        """A readable short id for this leg's trajectory and ancestry.
        The hash includes anything that can change the training trajectory.
        """
        return _digest(
            self.starting_checkpoint.uid if self.starting_checkpoint else None,
            self.dataset.uid,
            self.tokenizer.uid,
            self.model,
            asdict(self.model_parameters),
            asdict(self.training_parameters),
        )

    @property
    def uid(self) -> str:
        return (
            f"{self.run_id}-pretraining-"
            f"{self.start_step}-{self.end_step}-{self.lineage_hash}"
        )

    @property
    def artifact_path(self) -> Path:
        return (
            Path("runs")
            / self.run_id
            / "pretraining"
            / f"{self.start_step}-{self.end_step}-{self.lineage_hash}"
        )
