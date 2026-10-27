"""The pretraining stage: __init__.py for the Pretraining leg, jobs.py for
its batches, loss and evaluation on top of the shared leg and loop in
artifacts.core.SGD. Which model it trains is the leg's `model` parameter, a
package under models/.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import ClassVar, Self

from pydantic import model_validator

from artifacts.core.artifact import _digest
from artifacts.core.SGD.training import Training
from artifacts.dataset import DataSet
from artifacts.mappeddataset import MappedDataSet
from artifacts.tokenizers import Tokenizer


class Pretraining(Training):
    """A pretraining leg: next-token prediction over the dataset's token
    stream."""

    producer: ClassVar[str] = "artifacts.stages.pretraining.jobs.PretrainJob"

    dataset: DataSet | MappedDataSet
    starting_checkpoint: Pretraining | None = None

    @model_validator(mode="after")
    def _one_tokenizer_for_life(self) -> Self:
        """Refuses a model that does not speak its dataset's tokenizer, or a
        tokenizer that differs from the starting checkpoint's: a model speaks
        one tokenizer for life."""
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"model.vocab_size ({self.model.vocab_size}) does not "
                f"match tokenizer {self.tokenizer.uid}'s vocab_size ({self.tokenizer.vocab_size})"
            )
        if self.starting_checkpoint and self.starting_checkpoint.tokenizer.uid != self.tokenizer.uid:
            raise ValueError(
                f"tokenizer {self.tokenizer.uid} differs from starting_checkpoint's "
                f"{self.starting_checkpoint.tokenizer.uid}"
            )
        return self

    @property
    def tokenizer(self) -> Tokenizer:
        """Returns the tokenizer this leg's dataset was tokenized with: the one
        its model reads and writes."""
        return self.dataset.tokenizer

    @cached_property
    def lineage_hash(self) -> str:
        """A readable short id for this leg's trajectory and ancestry.
        The hash includes anything that can change the training trajectory.
        """
        return _digest(
            self.starting_checkpoint.uid if self.starting_checkpoint else None,
            self.dataset.uid,
            self.tokenizer.uid,
            type(self.model).__module__,
            self.model.model_dump(),
            self.training_parameters.model_dump(),
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
