"""The 'mamba' model family: __init__.py for its ModelParameters/
Pretraining, jobs.py for its (faked) training loop -- same shape as
models/transformer/, but for a selective state-space architecture (Gu &
Dao, 2023) instead of attention. jobs.py fakes the loop rather than
running real mamba-ssm/torch code, the same way models/mock/ does.

This family exists to check that artifacts.core.training
(TrainingParameters/LoopConfig/OptimizerParameters/LRSchedule) and
artifacts.core.locate actually generalize to a second architecture, not
just the transformer they were extracted from -- ModelParameters below is
the one part that's genuinely architecture-specific and doesn't come from
core.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact
from artifacts.core.SGD.training import LoopConfig, TrainingParameters
from artifacts.dataset import DataSet
from artifacts.mappeddataset import MappedDataSet
from artifacts.tokenizers import Tokenizer


@dataclass(frozen=True)
class ModelParameters:
    """Names and defaults follow the official Mamba implementation
    (state-spaces/mamba, mamba_ssm.modules.mamba_simple.Mamba) -- d_state/
    d_conv/expand are the SSM block's own shape, dt_* control the block's
    discretization step. None of these have a transformer equivalent (no
    attention heads, no rope_theta) the way vocab_size/num_layers/d_model
    do -- that's the actual dividing line between "every sequence model
    needs this" and "this architecture's own knobs".
    """

    vocab_size: int
    sequence_length: int
    num_layers: int
    d_model: int
    d_state: int = 16  # SSM state expansion factor (N)
    d_conv: int = 4  # local causal conv width
    expand: int = 2  # block expansion factor (E); inner dim = expand * d_model
    dt_rank: str = "auto"  # "auto" -> ceil(d_model / 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    device: str = "cuda"
    dtype: str = "torch.float32"  # resolved worker-side via artifacts.core.locate.locate


@dataclass(frozen=True)
class MambaPretraining(Artifact):
    """One pretraining run, trained from scratch for
    `training_parameters.total_steps` steps.
    Same single-leg/no-resumption shape as models/transformer's Pretraining.
    """

    producer: ClassVar[str] = "artifacts.models.mamba.jobs.MambaPretrainJob"

    run_id: str
    dataset: DataSet | MappedDataSet
    tokenizer: Tokenizer
    model_parameters: ModelParameters
    training_parameters: TrainingParameters
    loop_config: LoopConfig

    @property
    def uid(self) -> str:
        return f"{self.run_id}-pretraining"

    @property
    def artifact_path(self) -> Path:
        return Path("runs") / self.run_id / "pretraining"

    @property
    def files(self) -> dict[str, str]:
        return {"checkpoint": "checkpoint.txt", "progress": "progress.json"}

    def durable_progress(self, root: Path) -> dict | None:
        if self.paths(root)["progress"].exists():
            return {"step": self.training_parameters.total_steps, "total_steps": self.training_parameters.total_steps}
        folder = root / self.artifact_path
        steps = [
            int(m.group(1))
            for p in folder.glob("checkpoint_*.txt")
            if (m := re.match(r"checkpoint_(\d+)\.txt$", p.name))
        ]
        return {"step": max(steps), "total_steps": self.training_parameters.total_steps} if steps else None
