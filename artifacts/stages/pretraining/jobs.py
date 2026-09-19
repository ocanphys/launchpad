from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from artifacts.core.SGD.job import TrainingJob
from artifacts.core.SGD.steplog import StepLog
from artifacts.stages.pretraining import Pretraining

if TYPE_CHECKING:
    from system.runtime import Worker


def get_batch(data, batch_size: int, sequence_length: int, seed: int, step: int, device):
    """(inputs, targets) for one step, as int64 tensors of shape
    (batch_size, sequence_length) on device.

    The batch is a function of (seed, step) alone, so a resumed run draws
    exactly the batches the interrupted one would have; no RNG state needs
    checkpointing for this.
    """
    rng = np.random.default_rng((seed, step))
    starts = rng.integers(0, len(data) - sequence_length, size=batch_size)
    # cast on the numpy side - Embedding needs long
    inputs = np.stack([data[i : i + sequence_length] for i in starts]).astype(np.int64)
    targets = np.stack([data[i + 1 : i + sequence_length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


class PretrainJob(TrainingJob):
    artifact: Pretraining

    def __init__(self, artifact: Pretraining):
        super().__init__(artifact)
        self.dataset = artifact.dataset
        self.sequence_length = artifact.model_parameters.sequence_length
        self.loss_function = torch.nn.CrossEntropyLoss()

    def batch_loss(self, model: torch.nn.Module, data, step: int) -> torch.Tensor:
        inputs, targets = get_batch(
            data,
            self.training_parameters.batch_size,
            self.sequence_length,
            self.training_parameters.seed,
            step,
            self.device,
        )
        return self.loss_function(model(inputs).transpose(1, 2), targets)

    def evaluate(self, model, dataset, step, loss, learning_rate, grad_norm, worker: "Worker") -> None:
        """Reports the state of training after `step` completed steps to
        worker.progress and the log: the last training step's loss, learning rate and
        gradient norm, and the validation loss."""
        model.eval()
        with torch.no_grad():
            val_loss = self.batch_loss(model, dataset.valid_tokens, step).item()
        model.train()
        # the only .item() calls in the loop: each one waits for the GPU
        metrics = {"loss": loss.item(), "val_loss": val_loss, "learning_rate": learning_rate, "grad_norm": grad_norm.item()}
        end_step = self.artifact.end_step
        worker.progress.update({"end_step": end_step, **metrics})
        worker.log.info(
            f"step {step}/{end_step} "
            + " ".join(f"{name}={value:.4g}" for name, value in metrics.items())
        )

    def run(self, root: Path, worker: "Worker") -> None:
        end_step = self.artifact.end_step
        if self.artifact.paths(root)["model"].exists(): # return if already done.
            worker.log.info(f"model.pt exists, leg already complete at step {end_step}")
            return
        folder = root / self.artifact.artifact_path
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        # resume from initial model or most recent failsafe checkpoint.
        model, optimizer, step = self.resume(root, worker)
        log = StepLog(folder / "train.jsonl")
        # the artifact run_job hands a job is the declaration only; bind maps
        # the train/valid token files its own job wrote onto it
        dataset = self.dataset.bind(root)
        max_norm = self.training_parameters.max_norm

        # `step` counts completed steps; the increment makes it the number of
        # the step being taken, whose batch and rate are drawn from it. After
        # optimizer.step() the model is the result of `step`, which is what
        # the failsafe records, with the rate it was taken at
        model.train()
        while step < end_step:
            step += 1
            learning_rate = self.set_learning_rate(optimizer, step)
            loss = self.batch_loss(model, dataset.train_tokens, step)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
            log.record(step, loss, grad_norm, learning_rate)
            worker.progress["step"] = step

            if step % self.loop_config.val_every == 0:
                self.evaluate(model, dataset, step, loss, learning_rate, grad_norm, worker)
                log.flush()

            if step % self.loop_config.checkpoint_every == 0:
                self.failsafe(folder / "checkpoints", model, optimizer, step)
                worker.log.info(f"failsafe at step {step}/{end_step}")

        log.flush()
        self.write_atomic(
            self.artifact.paths(root)["model"],
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": end_step},
        )
        worker.log.info(f"training complete at step {end_step}")
