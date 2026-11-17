from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from artifacts.core.locate import locate
from artifacts.core.SGD.job import TrainingJob
from artifacts.core.SGD.training import Evaluation
from artifacts.core.SGD.steplog import StepLog
from artifacts.stages.pretraining import Pretraining
from gpus.monitor import GpuMonitor

if TYPE_CHECKING:
    import numpy as np

    from system.runtime import Worker


class PretrainJob(TrainingJob):
    artifact: Pretraining

    def __init__(self, artifact: Pretraining):
        super().__init__(artifact)
        self.dataset = artifact.dataset
        self.loss_function = torch.nn.CrossEntropyLoss()

    def to_device(self, batch: tuple["np.ndarray", ...]) -> tuple[torch.Tensor, ...]:
        """Returns batch's arrays as tensors on device, copied without waiting for the GPU.

        A copy from pageable memory waits for the GPU to drain; from pinned memory
        it queues behind the work already sent. Pinning needs CUDA, so on any other
        device the arrays are copied as they are."""
        pin = self.device.type == "cuda"
        return tuple(
            (torch.from_numpy(array).pin_memory() if pin else torch.from_numpy(array)).to(self.device, non_blocking=True)
            for array in batch
        )

    def batch_loss(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_function(model(inputs).transpose(1, 2), targets)

    def evaluate(
        self, model: torch.nn.Module, dataset, worker: "Worker", evaluations: Sequence[Evaluation]
    ) -> dict[str, torch.Tensor]:
        """Returns the metrics of every one of evaluations, taken with the model
        in eval mode and gradients off, as scalar tensors on device.

        Two evaluations that name the same metric raise, rather than one
        silently overwriting the other in the step's row."""
        metrics: dict[str, torch.Tensor] = {}
        model.eval()
        with torch.no_grad():
            for evaluation in evaluations:
                measured = evaluation.measure(self, model, dataset, worker)
                if clash := metrics.keys() & measured.keys():
                    raise ValueError(f"{evaluation!r} measures {sorted(clash)}, already measured at this step")
                metrics |= measured
        model.train()
        return metrics

    def run(self, root: Path, worker: "Worker") -> None:
        if self.artifact.status(root).complete:  # a finished leg is never rerun (LESSONS.md)
            worker.log.info(f"model.pt exists, leg already complete at step {self.artifact.end_step}")
            return
        end_step = self.artifact.end_step
        batch_size = self.training_parameters.batch_size
        seed = self.training_parameters.seed
        max_norm = self.training_parameters.max_norm
        evaluations = self.loop_config.evaluations
        checkpoint_every = self.loop_config.checkpoint_every
        gpu_check_every = self.loop_config.gpu_check_every
        folder = root / self.artifact.artifact_path
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        # resume from initial model or most recent failsafe checkpoint.
        model, optimizer, step = self.resume(root, worker)
        steplog = StepLog(root, self.artifact.artifact_path.as_posix())
        # forward and loss, in training and evaluation, run in the leg's precision; backward follows it
        autocast = torch.autocast(
            self.device.type,
            dtype=locate(self.artifact.autocast) if self.artifact.autocast else None,
            enabled=self.artifact.autocast is not None,
        )
        gpu_monitor = GpuMonitor(self.artifact.model, self.artifact.precision, self.device, batch_size, gpu_check_every, step, worker)
        dataset = self.dataset.bind(root)
        get_train_batch = self.artifact.sampler.batches(
            dataset.train, batch_size, self.artifact.model.sequence_length, seed, end_step
        )
        # optimization notes for data pipeline: drawing step + 1's batch
        # before this step's forward would hide the draw too, and sending the
        # uint16 tokens and casting to int64 on device would cut the bytes
        # copied by 4x
        model.train()
        while step < end_step:
            worker.progress["step"] = step
            step += 1
            learning_rate = self.set_learning_rate(optimizer, step)
            inputs, targets = self.to_device(get_train_batch(step))
            with autocast:
                loss = self.batch_loss(model, inputs, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            metrics = {"loss": loss, "grad_norm": grad_norm, "learning_rate": learning_rate} | gpu_monitor.record(step)
            if due := [e for e in evaluations if step % e.eval_every == 0 or step == end_step]:
                with autocast:
                    metrics |= self.evaluate(model, dataset, worker, due)
                steplog.record(step, metrics)
                # one GPU sync: every recorded tensor read back at once
                steplog.flush()
            else:
                steplog.record(step, metrics)

            if step % checkpoint_every == 0:
                # rows first: a crash between the two then repeats steps rather than losing them
                steplog.flush()
                self.failsafe(folder / "checkpoints", model, optimizer, step, worker)
                worker.log.info(f"checkpoint saved at step {step}/{end_step}")

        steplog.flush()
        with worker.publishing(self.artifact.paths(root)["model"]) as out:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": end_step},
                out,
            )
        worker.log.info(f"training complete at step {end_step}")
