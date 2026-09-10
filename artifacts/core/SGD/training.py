"""Optimization and training-loop parameters shared by every model family
that trains with an SGD-variant optimizer -- none of this references any
architecture's shape (that's each family's own ModelParameters, e.g.
artifacts/models/transformer/__init__.py), so a future family (Mamba, other
state space models, ...) reuses these as-is instead of redefining them.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OptimizerParameters:
    lr: float  # fallback: what the optimizer runs with when no lr_schedule overrides it
    betas: tuple[float]
    weight_decay: float
    eps: float


@dataclass(frozen=True)
class LRSchedule:
    """Independent of OptimizerParameters -- any schedule can drive any
    optimizer, so this isn't nested under it (mirrors how torch.optim's own
    lr_scheduler wraps an optimizer from outside rather than living in it).
    """

    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    fn: str = "artifacts.core.lr_schedule.lr_cosine_schedule"  # resolved worker-side via artifacts.core.locate.locate


@dataclass(frozen=True)
class TrainingParameters:
    """What changes the optimization trajectory, including its seed and
    duration -- as opposed to LoopConfig, which controls observation and
    recovery cadence.
    """

    total_steps: int
    batch_size: int
    max_norm: int
    lr_schedule: LRSchedule
    optimizer: str  # dotted path to a torch.optim.Optimizer subclass; resolved worker-side via artifacts.core.locate.locate
    optimizer_parameters: OptimizerParameters
    seed: int = 0


@dataclass(frozen=True)
class LoopConfig:
    """Operational settings for validation, failsafe checkpoints, and health checks.

    Every ``checkpoint_every`` steps, the training job writes a failsafe model
    state. ``optimizer_checkpoint_policy="all"`` writes the matching optimizer
    state at every failsafe; ``"latest"`` retains it only for the newest one.
    """

    val_every: int
    gpu_check_every: int
    checkpoint_every: int = 100
    optimizer_checkpoint_policy: Literal["all", "latest"] = "latest"

    def __post_init__(self) -> None:
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.optimizer_checkpoint_policy not in ("all", "latest"):
            raise ValueError(
                "optimizer_checkpoint_policy must be 'all' or 'latest'"
            )
