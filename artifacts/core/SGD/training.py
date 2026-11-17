"""The training leg (Training) and the parameters it is declared with, shared
by every stage (artifacts/stages/*) that trains a model with an SGD-variant
optimizer. The model is a parameter, not a family: a leg holds a Model from
a package under models/, such as models.transformer.Transformer, so no stage is
tied to an architecture. The loop that runs a leg is job.py's TrainingJob; this module
stays free of torch so declaring an artifact never imports it.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, model_validator

from artifacts.core.artifact import Artifact, ParamConfig
from artifacts.core.SGD.lr_schedule import LRSchedule
from artifacts.core.SGD.optimizers import Optimizer
from models import Model

if TYPE_CHECKING:
    import torch

    from artifacts.core.SGD.job import TrainingJob
    from system.runtime import Worker


class TrainingParameters(ParamConfig):
    """What changes the optimization trajectory, including its seed and
    duration, as opposed to LoopConfig, which controls observation and
    recovery cadence. `lr_schedule` is the only place the learning rate is
    stated.

        TrainingParameters(total_steps=1000, batch_size=64, max_norm=1, lr_schedule=Constant(lr=3e-4),
                           optimizer=AdamW()).max_norm
          -> 1.0
    """

    total_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_norm: float
    lr_schedule: LRSchedule
    optimizer: Optimizer
    seed: int = 0


class Evaluation(ParamConfig):
    """The base of every measurement the loop takes of its model, every
    `eval_every` steps of its own and at end_step: a subclass's other fields
    are its parameters, and `measure` takes it. A stage defines the subclasses;
    a field annotated `Evaluation` holds any of them, named under "class" in
    the manifest.

    What it measures goes into the step's row of train.jsonl beside the
    training metrics, under names starting `val_` when it is of the validation
    split, which is how a reader tells the two apart. One that reads the step's
    gradients, still on the parameters, reads the training batch's, clipped.
    Never identity: measuring changes no weight.
    """

    eval_every: int = Field(gt=0)

    def measure(
        self, job: TrainingJob, model: torch.nn.Module, dataset: Any, worker: Worker
    ) -> dict[str, torch.Tensor]:
        """Returns this measurement of `model`, as named scalar tensors on its device.

        The loop calls it with the model in eval mode and gradients off.

        Args:
            job: The job running the leg, for its device and loss.
            model: The model at the step being measured.
            dataset: The leg's dataset, bound; an evaluation that reads data reads dataset.valid.
            worker: The worker running the job, for live progress.

        Returns:
            Each metric's name, starting `val_` when it is of the validation split, and its value.
        """
        raise NotImplementedError


class LoopConfig(ParamConfig):
    """Operational settings for evaluation, failsafe checkpoints, health checks and compilation.

    The loop takes each of ``evaluations`` every ``eval_every`` steps of its
    own, and all of them at end_step. Every ``checkpoint_every`` steps, the training job writes
    a failsafe model state. ``optimizer_checkpoint_policy="all"`` writes the
    matching optimizer state at every failsafe; ``"latest"`` retains it only
    for the newest one. On CUDA, it reports the MFU and the peak allocated and
    reserved memory over each window of ``gpu_check_every`` steps, evaluation
    and checkpoints included, a few steps after the window closes. A ``compile``
    mode runs the model, in training and evaluation, through ``torch.compile``
    in that mode; None runs it eagerly.

        LoopConfig(evaluations=(TokenLoss(eval_every=200, batch_size=64),), gpu_check_every=200)
          -> LoopConfig(evaluations=(TokenLoss(eval_every=200, batch_size=64),), gpu_check_every=200, checkpoint_every=100, optimizer_checkpoint_policy='latest', compile=None)
    """

    evaluations: tuple[Evaluation, ...]
    gpu_check_every: int = Field(gt=0)
    checkpoint_every: int = Field(default=100, gt=0)
    optimizer_checkpoint_policy: Literal["all", "latest"] = "latest"
    compile: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] | None = None


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
    model: Model  # filled in from starting_checkpoint when omitted
    starting_checkpoint: Training | None = None

    @model_validator(mode="before")
    @classmethod
    def _inherit_model(cls, data: Any) -> Any:
        """Returns `data` with the starting checkpoint's model when it names none,
        so a leg continuing another states only what changes."""
        if isinstance(data, dict) and data.get("model") is None and data.get("starting_checkpoint") is not None:
            return {**data, "model": data["starting_checkpoint"].model}
        return data

    @model_validator(mode="after")
    def _model_matches_checkpoint(self) -> Self:
        if self.starting_checkpoint is not None and self.model != self.starting_checkpoint.model:
            raise ValueError("model must match starting_checkpoint.model")
        return self

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
