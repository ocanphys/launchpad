"""The 'mock' model family: a stand-in architecture whose job fakes a
training loop (see job.py) instead of running real torch code. Kept around
as a cheap path for exercising the resolver/lease/dashboard machinery
without paying for a real model -- a real architecture (e.g. a transformer)
gets its own sibling package here with this same shape: artifact.py for its
ModelParameters/Config/Pretraining, job.py for its actual training loop.
"""

from dataclasses import dataclass
from pathlib import Path

from dag.artifact import Artifact
from datasets.artifact import DataSet
from tokenizers.bpe import Tokenizer


@dataclass(frozen=True)
class ModelParameters:
    hidden_size: int = 64
    num_layers: int = 2


@dataclass(frozen=True)
class PretrainingConfig:
    total_steps: int
    batch_size: int
    lr: float = 1e-3
    seed: int = 0
    checkpoint_every: int = 100


@dataclass(frozen=True)
class Pretraining(Artifact):
    """One pretraining run, trained from scratch to `config.total_steps`.

    Single-leg for now -- no mid-run resumption. Legs come back once an
    artifact can nest a prior leg of itself as a parameter, the same way a
    checkpoint nests the dataset and tokenizer it was trained from; nothing
    here forecloses that.
    """

    run_id: str
    dataset: DataSet
    tokenizer: Tokenizer
    model_parameters: ModelParameters
    config: PretrainingConfig

    @property
    def uid(self) -> str:
        return f"{self.run_id}-pretraining"

    @property
    def artifact_path(self) -> Path:
        return Path("runs") / self.run_id / "pretraining"

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
