from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from artifacts.core.job import Job
from artifacts.core.locate import locate
from artifacts.models.transformer import Pretraining
from artifacts.models.transformer.src import TransformerLM
from artifacts.models.transformer.src.util import seed_everything

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


class PretrainJob(Job):
    artifact: Pretraining

    def __init__(self, artifact: Pretraining):
        super().__init__(artifact)
        self.dataset = artifact.dataset
        self.tokenizer = artifact.tokenizer
        self.starting_checkpoint = artifact.starting_checkpoint
        self.training_parameters = artifact.training_parameters
        self.loop_config = artifact.loop_config

        self.device = torch.device(artifact.model_parameters.device)
        # since these are saved as strings in the config - we resolve them with python modules
        self.dtype = locate(artifact.model_parameters.dtype)
        self.optimizer_cls = locate(self.training_parameters.optimizer)
        self.schedule_fn = locate(self.training_parameters.lr_schedule_fn)
        self.schedule_parameters = asdict(self.training_parameters.lr_schedule)
        seed_everything(self.training_parameters.seed)

    def resume(self, root: Path, worker: "Worker"):
        """The (model, optimizer, step) this attempt trains from.

        A failsafe of this leg's own wins, but only one that saved its
        optimizer; then the starting checkpoint's final state; then a fresh
        model. A loaded model gets a fresh optimizer if none was saved
        beside it. Optimizer parameters are this leg's either way.
        """
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
            candidate = torch.load(path, map_location=self.device)
            if "optimizer" in candidate:
                state = candidate
                worker.log.info(
                    f"found failsafe {path.name}, starting from step {state['step']}"
                )
                break
        # if no usable failsafe, look for previous leg's final state
        if state is None and self.starting_checkpoint is not None:
            state = torch.load(
                self.starting_checkpoint.paths(root)["model"], map_location=self.device
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
        step = state["step"] if state is not None else 0

        # build model and optimizer from this leg's parameters, then
        # load state into them. The optimizer is constructed over the live
        # model's parameters, so the two share tensors; loading state_dicts
        # afterwards keeps that link, which pickled whole objects would not.
        model_parameters = self.artifact.model_parameters
        model = TransformerLM(
            vocab_size=model_parameters.vocab_size,
            context_length=model_parameters.sequence_length,
            num_layers=model_parameters.num_layers,
            d_model=model_parameters.d_model,
            d_ff=model_parameters.d_ff,
            num_heads=model_parameters.num_heads,
            rope_theta=model_parameters.rope_theta,
            device=self.device,
            dtype=self.dtype,
        )
        optimizer_parameters = asdict(self.training_parameters.optimizer_parameters)
        optimizer = self.optimizer_cls(model.parameters(), **optimizer_parameters)
        if state is not None:
            model.load_state_dict(state["model"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
                # load_state_dict also restores the saved leg's lr/betas/
                # weight_decay; this leg's are the ones in its lineage hash
                for group in optimizer.param_groups:
                    group.update(optimizer_parameters)
            else:
                # only reachable from a starting checkpoint: failsafes without
                # an optimizer were skipped above
                worker.log.warning(
                    f"{self.starting_checkpoint.uid} saved no optimizer, using a fresh one"
                )
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

    def run(self, root: Path, worker: "Worker") -> None:
        folder = root / self.artifact.artifact_path
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        model, optimizer, step = self.resume(root, worker)
        loss_function = torch.nn.CrossEntropyLoss()
        # the artifact run_job hands a job is the declaration only; bind maps
        # the train/valid token files its own job wrote onto it
        dataset = self.dataset.bind(root)
        train_data = dataset.train_tokens
        valid_data = dataset.valid_tokens
        batch_size = self.training_parameters.batch_size
        sequence_length = self.artifact.model_parameters.sequence_length
        seed = self.training_parameters.seed
        max_norm = self.training_parameters.max_norm
        end_step = self.artifact.end_step

        # `step` counts completed steps: the batch and lr below are for step
        # number `step`, and after optimizer.step() the model is the result of
        # `step + 1`, which is what the failsafe records
        model.train()
        while step < end_step:
            inputs, targets = get_batch(train_data, batch_size, sequence_length, seed, step, self.device)
            lr = self.schedule_fn(step, **self.schedule_parameters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss = loss_function(model(inputs).transpose(1, 2), targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            worker.progress["step"] = step

            if step % self.loop_config.val_every == 0:
                model.eval()
                with torch.no_grad():
                    vi, vt = get_batch(valid_data, batch_size, sequence_length, seed, step, self.device)
                    val_loss = loss_function(model(vi).transpose(1, 2), vt).item()
                model.train()
                # the only .item() calls in the loop: each one waits for the GPU
                worker.progress.update(
                    {"end_step": end_step, "loss": loss.item(), "val_loss": val_loss, "lr": lr, "grad_norm": grad_norm.item()}
                )
                worker.log.info(
                    f"step {step}/{end_step} loss={loss.item():.4f} val_loss={val_loss:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm.item():.3f}"
                )

            if step % self.loop_config.checkpoint_every == 0:
                self.failsafe(folder / "checkpoints", model, optimizer, step)
                worker.log.info(f"failsafe at step {step}/{end_step}")

        self.write_atomic(
            self.artifact.paths(root)["model"],
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": end_step},
        )
        worker.log.info(f"training complete at step {end_step}")
