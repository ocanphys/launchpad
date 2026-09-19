"""What every stage's job shares: building the model and optimizer a leg
names, resuming from a failsafe or a starting checkpoint, and writing
failsafes. A stage's jobs.py subclasses TrainingJob and writes its own run()
around them.
"""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from artifacts.core.job import Job
from artifacts.core.locate import locate
from artifacts.core.SGD.training import Training

if TYPE_CHECKING:
    from system.runtime import Worker


class TrainingJob(Job):
    """Resume and failsafe for one leg, from the artifact's start_step to its
    end_step.

    A subclass writes a ``run`` that starts from ``resume`` and calls
    ``failsafe`` every ``loop_config.checkpoint_every`` steps.
    """

    artifact: Training

    def __init__(self, artifact: Training):
        super().__init__(artifact)
        self.starting_checkpoint = artifact.starting_checkpoint
        self.training_parameters = artifact.training_parameters
        self.loop_config = artifact.loop_config
        self.device = torch.device(artifact.model_parameters.device)
        # saved as dotted paths in the config - resolved to the python objects
        self.schedule_fn = locate(self.training_parameters.lr_schedule_fn)
        self.schedule_parameters = asdict(self.training_parameters.lr_schedule)
        # seeds every RNG that model initialization draws from, CPU and CUDA
        torch.manual_seed(self.training_parameters.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training_parameters.seed)

    def set_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        """The learning rate of step number `step`, after applying it to
        every one of optimizer's param groups.

        The optimizer holds the rate of the current step: the loop sets it
        just before using it, and a checkpoint saves it along with the model
        and optimizer, so the three are always in sync. Resume sets it after
        loading for the same reason.
        """
        learning_rate = self.schedule_fn(step, **self.schedule_parameters)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def resume(self, root: Path, worker: "Worker"):
        """The (model, optimizer, step) this attempt trains from.

        A failsafe of this leg's own wins, but only one that saved its
        optimizer; then the starting checkpoint's final state; then a fresh
        model. A loaded model gets a fresh optimizer if none was saved
        beside it. Optimizer parameters are this leg's either way.
        """
        # build model and optimizer from this leg's parameters, then load
        # state into them. The optimizer is constructed over the live model's
        # parameters, so the two share tensors; loading state_dicts afterwards
        # keeps that link, which pickled whole objects would not.
        model = locate(self.artifact.model).build(self.artifact.model_parameters)
        optimizer = locate(self.training_parameters.optimizer)(
            model.parameters(), **asdict(self.training_parameters.optimizer_parameters)
        )
        device = self.device

        # pick the state to resume from
        # state = {"model": ..., "step": ..., "optimizer"?: ...} or None for a
        # fresh start. Every checkpoint records the step it is the result of,
        # and step 0 is the initialized model before any training.
        checkpoints = root / self.artifact.artifact_path / "checkpoints"
        # failsafes are named {step}.pt, so the stem sorts them by step
        failsafes = sorted(checkpoints.glob("*.pt"), key=lambda p: int(p.stem))
        state = None
        # newest first: under the "latest" policy only the newest failsafe
        # still holds optimizer state, and one without it cannot be resumed
        # from, so keep looking further back until one has it
        for path in reversed(failsafes):
            candidate = torch.load(path, map_location=device)
            if "optimizer" in candidate:
                state = candidate
                worker.log.info(
                    f"found failsafe {path.name}, starting from step {state['step']}"
                )
                break
        # if no usable failsafe, look for previous leg's final state
        if state is None and self.starting_checkpoint is not None:
            state = torch.load(
                self.starting_checkpoint.paths(root)["model"], map_location=device
            )
            # double check if the state has the correct step saved
            if state["step"] != self.artifact.start_step:
                raise ValueError(
                    f"{self.starting_checkpoint.uid} is at step {state['step']}, "
                    f"this leg starts at {self.artifact.start_step}"
                )
            worker.log.info(
                f"loaded {self.starting_checkpoint.uid}, starting from step {state['step']}"
            )
        if state is None:
            worker.log.info("no checkpoint, initializing model, starting from step 0")
            step = 0
        else:
            step = state["step"]
            model.load_state_dict(state["model"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
                # load_state_dict also restores the saved leg's betas/
                # weight_decay; this leg's are the ones in its lineage hash
                for group in optimizer.param_groups:
                    group.update(asdict(self.training_parameters.optimizer_parameters))
            else:
                # only reachable from a starting checkpoint: failsafes without
                # an optimizer were skipped above
                worker.log.warning(
                    f"{self.starting_checkpoint.uid} saved no optimizer, using a fresh one"
                )
        # the optimizer always holds the rate for `step`, however it got here
        self.set_learning_rate(optimizer, step)
        return model, optimizer, step

    def write_atomic(self, path: Path, state: dict) -> None:
        """Writes state to path atomically: a reader never sees a half-written file."""
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def failsafe(self, checkpoints: Path, model, optimizer, step: int) -> None:
        """Writes checkpoints/{step}.pt with model and optimizer state and, under
        the "latest" policy, drops the optimizer from every earlier failsafe."""
        previous = list(checkpoints.glob("*.pt"))
        self.write_atomic(
            checkpoints / f"{step}.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
        )
        if self.loop_config.optimizer_checkpoint_policy == "latest":
            for path in previous:
                state = torch.load(path, map_location="cpu", mmap=True)
                if state.pop("optimizer", None) is not None:
                    self.write_atomic(path, state)
