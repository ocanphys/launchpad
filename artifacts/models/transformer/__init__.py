"""The 'transformer' model family: __init__.py for its ModelParameters/
Pretraining, jobs.py for its actual training loop -- same shape as its
sibling folder, models/mamba/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact, _digest
from artifacts.core.SGD.training import LoopConfig, TrainingParameters
from artifacts.dataset import DataSet
from artifacts.mappeddataset import MappedDataSet
from artifacts.tokenizers.bpe import Tokenizer


@dataclass(frozen=True)
class ModelParameters:
    vocab_size : int
    sequence_length: int
    num_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    rope_theta: int
    device: str = "cuda"
    dtype: str = "torch.float32"  # resolved worker-side via artifacts.core.locate.locate


@dataclass(frozen=True)
class Pretraining(Artifact):
    """One training leg from ``start_step`` to ``end_step``.

    ``training_parameters.total_steps`` is the number of steps this leg runs. A leg
    with a starting checkpoint begins at that checkpoint's endpoint.
    """

    producer: ClassVar[str] = "artifacts.models.transformer.jobs.PretrainJob"

    run_id: str
    dataset: DataSet | MappedDataSet
    tokenizer: Tokenizer
    training_parameters: TrainingParameters
    loop_config: LoopConfig
    model_parameters: ModelParameters | None = None
    starting_checkpoint: Pretraining | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_parameters.total_steps <= 0:
            raise ValueError("training_parameters.total_steps must be positive")
        if self.starting_checkpoint is None:
            if self.model_parameters is None:
                raise ValueError(
                    "model_parameters are required without a starting_checkpoint"
                )
            return
        if self.model_parameters is None:
            object.__setattr__(
                self, "model_parameters", self.starting_checkpoint.model_parameters
            )
        elif self.model_parameters != self.starting_checkpoint.model_parameters:
            raise ValueError(
                "model_parameters must match starting_checkpoint.model_parameters"
            )

    @property
    def start_step(self) -> int:
        return self.starting_checkpoint.end_step if self.starting_checkpoint else 0

    @property
    def end_step(self) -> int:
        return self.start_step + self.training_parameters.total_steps

    @property
    def lineage_hash(self) -> str:
        """A readable short id for this leg's trajectory and ancestry.
        The hash includes anything that can change the training trajectory.
        """
        return _digest(
            self.starting_checkpoint.uid if self.starting_checkpoint else None,
            self.dataset.uid,
            self.tokenizer.uid,
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

    @property
    def files(self) -> dict[str, str]:
        # The final model and optimizer are this leg's durable artifact
        # outputs. Failsafe checkpoints live under checkpoints/{step}/ and
        # are intentionally undeclared: recovery state is job-owned and may
        # differ after an interrupted run.
        #
        # progress is written last, after both final outputs are durable.
        return {
            "model": "model.obj",
            "optimizer": "optimizer.obj",
        }
