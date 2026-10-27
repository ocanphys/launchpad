"""The training leg (Training) and the parameters it is declared with, shared
by every stage (artifacts/stages/*) that trains a model with an SGD-variant
optimizer. The model is a parameter, not a family: a leg holds a Model from
a package under models/, such as models.transformer.Transformer, so no stage is
tied to an architecture. The loop that runs a leg is job.py's TrainingJob; this module
stays free of torch so declaring an artifact never imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from artifacts.core.artifact import Artifact, ParamConfig


class Model(ParamConfig):
    """The base of every architecture under models/: a subclass's fields are
    what a leg declares, and `build` makes the network they describe. The
    fields declared here are the ones stages and jobs read off any model.

    A field annotated `Model` holds any subclass, and its manifest names that
    subclass under "class" (`polymorphic`).

    Transformer(vocab_size=1000, sequence_length=32, num_layers=4, d_model=128, d_ff=256, num_heads=16,
                rope_theta=10000).sequence_length -> 32
    """

    polymorphic: ClassVar[bool] = True

    vocab_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    dtype: str = "torch.float32"  # resolved worker-side via artifacts.core.locate.locate
    device: str = "cuda"

    def build(self):
        """Returns a freshly initialized torch module on this model's device.

        Transformer(...).build() -> TransformerLM(...) on torch.device("cuda")
        """
        raise NotImplementedError

    def step_flops(self, batch_size: int) -> int:
        """Returns the FLOPs of one forward and backward pass over batch_size sequences: an analytic
        estimate from this architecture's shapes, not a measurement.

        Transformer(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2,
                    rope_theta=10000).step_flops(4) -> 3612672
        """
        raise NotImplementedError


class OptimizerParameters(ParamConfig):
    """Exactly the keyword arguments the optimizer is built with, after its
    parameters; torch checks their values when it builds it.

    OptimizerParameters(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8).model_dump()
      -> {"lr": 0.0003, "betas": (0.9, 0.95), "weight_decay": 0.1, "eps": 1e-08}
    """

    lr: float  # fallback: what the optimizer runs with when no lr_schedule overrides it
    betas: tuple[float, ...]
    weight_decay: float
    eps: float


class LRSchedule(ParamConfig):
    """The keyword arguments of the learning-rate schedule function. Independent
    of OptimizerParameters: any schedule can drive any optimizer, so this is
    not nested under it, as torch.optim's own lr_scheduler wraps an optimizer
    from outside.

    LRSchedule(max_learning_rate=3e-4, min_learning_rate=3e-5, warmup_iters=200, cosine_cycle_iters=5000)
    """

    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int


class TrainingParameters(ParamConfig):
    """What changes the optimization trajectory, including its seed and
    duration, as opposed to LoopConfig, which controls observation and
    recovery cadence.

    ``optimizer`` and ``lr_schedule_fn`` are dotted paths, resolved worker-side
    via artifacts.core.locate.locate; ``optimizer_parameters`` and
    ``lr_schedule`` are exactly the keyword arguments each is called with.

    TrainingParameters(total_steps=1000, batch_size=64, max_norm=1, lr_schedule=LRSchedule(...),
                       optimizer="torch.optim.AdamW", optimizer_parameters=OptimizerParameters(...)).max_norm
      -> 1.0
    """

    total_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_norm: float
    lr_schedule: LRSchedule
    optimizer: str  # a torch.optim.Optimizer subclass
    optimizer_parameters: OptimizerParameters
    lr_schedule_fn: str = "artifacts.core.SGD.lr_schedule.lr_cosine_schedule"
    seed: int = 0


class LoopConfig(ParamConfig):
    """Operational settings for validation, failsafe checkpoints, and health checks.

    Every ``checkpoint_every`` steps, the training job writes a failsafe model
    state. ``optimizer_checkpoint_policy="all"`` writes the matching optimizer
    state at every failsafe; ``"latest"`` retains it only for the newest one.
    On CUDA, it reports the MFU and the peak allocated and reserved memory
    over each window of ``gpu_check_every`` steps, validation and checkpoints
    included, a few steps after the window closes.

    LoopConfig(val_every=200, gpu_check_every=200)
      -> LoopConfig(val_every=200, gpu_check_every=200, checkpoint_every=100, optimizer_checkpoint_policy='latest')
    """

    val_every: int = Field(gt=0)
    gpu_check_every: int = Field(gt=0)
    checkpoint_every: int = Field(default=100, gt=0)
    optimizer_checkpoint_policy: Literal["all", "latest"] = "latest"


# kw_only here and on every subclass, so their fields never have to be
# ordered around these defaults
@dataclass(frozen=True, kw_only=True)
class Training(Artifact):
    """One training leg from ``start_step`` to ``end_step``.

    ``model`` is the architecture this leg trains, an instance of a ``Model``
    subclass such as ``models.transformer.Transformer``.
    ``training_parameters.total_steps`` is the number of steps this leg runs. A
    leg with a starting checkpoint begins at that checkpoint's endpoint and
    inherits its ``model``; a leg without one has to state it. A subclass
    declares its ``producer``, what it trains on, and its ``uid`` and
    ``artifact_path``.
    """

    run_id: str
    training_parameters: TrainingParameters
    loop_config: LoopConfig
    model: Model | None = None
    starting_checkpoint: Training | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.starting_checkpoint is None:
            if self.model is None:
                raise ValueError("model is required without a starting_checkpoint")
            return
        inherited = self.starting_checkpoint.model
        if self.model is None:
            object.__setattr__(self, "model", inherited)
        elif self.model != inherited:
            raise ValueError("model must match starting_checkpoint.model")

    @cached_property
    def start_step(self) -> int:
        # Once per instance: it walks the chain of starting checkpoints.
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

    def durable_progress(self, root: Path) -> dict:
        """The step the volume holds this leg at, out of end_step: end_step
        once model.pt is there, else the furthest failsafe, else start_step."""
        if self.paths(root)["model"].exists():
            step = self.end_step
        else:
            failsafes = (root / self.artifact_path / "checkpoints").glob("*.pt")
            step = max((int(p.stem) for p in failsafes), default=self.start_step)
        return {"phase": "step", "done": step, "total": self.end_step}
