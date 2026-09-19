"""The training leg (Training) and the parameters it is declared with, shared
by every stage (artifacts/stages/*) that trains a model with an SGD-variant
optimizer. The model is a parameter, not a family: a leg names a package under
models/ by dotted path and holds that package's ModelParameters, so no stage
is tied to an architecture. The loop that runs a leg is job.py's TrainingJob;
this module stays free of torch so declaring an artifact never imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from artifacts.core.artifact import Artifact
from artifacts.core.locate import locate


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


@dataclass(frozen=True)
class TrainingParameters:
    """What changes the optimization trajectory, including its seed and
    duration -- as opposed to LoopConfig, which controls observation and
    recovery cadence.

    ``optimizer`` and ``lr_schedule_fn`` are dotted paths, resolved worker-side
    via artifacts.core.locate.locate; ``optimizer_parameters`` and
    ``lr_schedule`` are exactly the keyword arguments each is called with.
    """

    total_steps: int
    batch_size: int
    max_norm: int
    lr_schedule: LRSchedule
    optimizer: str  # a torch.optim.Optimizer subclass
    optimizer_parameters: OptimizerParameters
    lr_schedule_fn: str = "artifacts.core.SGD.lr_schedule.lr_cosine_schedule"
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


# kw_only here and on every subclass, so their fields never have to be
# ordered around these defaults
@dataclass(frozen=True, kw_only=True)
class Training(Artifact):
    """One training leg from ``start_step`` to ``end_step``.

    ``model`` is the dotted path of a package under models/ (e.g.
    ``"models.transformer"``) exporting ``ModelParameters`` and
    ``build(parameters)``; ``model_parameters`` is that package's
    ``ModelParameters``. ``training_parameters.total_steps`` is the number of
    steps this leg runs. A leg with a starting checkpoint begins at that
    checkpoint's endpoint and inherits its ``model`` and ``model_parameters``;
    a leg without one has to state them. A subclass declares its ``producer``,
    what it trains on, and its ``uid`` and ``artifact_path``.
    """

    run_id: str
    training_parameters: TrainingParameters
    loop_config: LoopConfig
    model: str | None = None
    model_parameters: Any | None = None
    starting_checkpoint: Training | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_parameters.total_steps <= 0:
            raise ValueError("training_parameters.total_steps must be positive")
        # the codec cannot type this field (its type depends on `model`), so a
        # manifest hands it back as a dict; rebuild the package's dataclass
        if isinstance(self.model_parameters, dict):
            object.__setattr__(
                self,
                "model_parameters",
                locate(self.model).ModelParameters(**self.model_parameters),
            )
        if self.starting_checkpoint is None:
            if self.model is None or self.model_parameters is None:
                raise ValueError(
                    "model and model_parameters are required without a starting_checkpoint"
                )
            return
        for name in ("model", "model_parameters"):
            inherited = getattr(self.starting_checkpoint, name)
            if getattr(self, name) is None:
                object.__setattr__(self, name, inherited)
            elif getattr(self, name) != inherited:
                raise ValueError(f"{name} must match starting_checkpoint.{name}")

    @property
    def start_step(self) -> int:
        return self.starting_checkpoint.end_step if self.starting_checkpoint else 0

    @property
    def end_step(self) -> int:
        return self.start_step + self.training_parameters.total_steps

    @property
    def files(self) -> dict[str, str]:
        # Model and optimizer state_dicts at end_step, in one file, are this
        # leg's durable artifact output. Failsafe checkpoints live under
        # checkpoints/{step}.pt and are intentionally undeclared: recovery
        # state is job-owned and may differ after an interrupted run.
        return {"model": "model.pt"}
