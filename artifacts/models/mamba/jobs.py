"""jobs.py for the 'mamba' family: fakes the training loop instead of
running real mamba-ssm/torch code, the same way models/mock/'s job did --
see __init__.py's module docstring for why this family exists.
dtype/optimizer/schedule are still resolved for real via
artifacts.core.locate.locate, to prove that machinery works against a
second family's dotted paths and not just the transformer's own.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.core.locate import locate
from artifacts.models.mamba import MambaPretraining

if TYPE_CHECKING:
    from system.runtime import Worker


class MambaPretrainJob(Job):
    artifact: MambaPretraining

    def __init__(self, artifact: MambaPretraining):
        super().__init__(artifact)
        self.dataset = artifact.dataset
        self.tokenizer = artifact.tokenizer
        self.training_parameters = artifact.training_parameters
        self.loop_config = artifact.loop_config

    def run(self, root: Path, worker: "Worker") -> None:
        folder = root / self.artifact.artifact_path
        folder.mkdir(parents=True, exist_ok=True)

        model_parameters = self.artifact.model_parameters
        schedule = self.training_parameters.lr_schedule
        # not run for real -- resolved here anyway, same as
        # models/transformer/jobs.py, so this actually exercises locate()
        # against this family's own dotted paths instead of only the
        # transformer's.
        dtype = locate(model_parameters.dtype)
        optimizer_cls = locate(self.training_parameters.optimizer)
        schedule_fn = locate(schedule.fn)

        every = self.loop_config.checkpoint_every
        total_steps = self.training_parameters.total_steps
        steps = list(range(every, total_steps + 1, every))
        if not steps or steps[-1] != total_steps:
            steps.append(total_steps)

        worker.log.info(
            f"{total_steps} steps, d_model={model_parameters.d_model}, "
            f"num_layers={model_parameters.num_layers}, d_state={model_parameters.d_state}, "
            f"optimizer={optimizer_cls.__name__}, dtype={dtype}, "
            f"trained_on={self.dataset.uid}"
        )

        loss, done = 10.0, 0
        for step in steps:
            lr = schedule_fn(
                step,
                max_learning_rate=schedule.max_learning_rate,
                min_learning_rate=schedule.min_learning_rate,
                warmup_iters=schedule.warmup_iters,
                cosine_cycle_iters=schedule.cosine_cycle_iters,
            )
            for _ in range(step - done):
                loss *= 0.99  # mock decay, not a real training loop
            done = step
            body = json.dumps(
                {
                    "step": step,
                    "lr": lr,
                    "loss": loss,
                    "model_parameters": {
                        "d_model": model_parameters.d_model,
                        "num_layers": model_parameters.num_layers,
                        "d_state": model_parameters.d_state,
                    },
                    "optimizer": self.training_parameters.optimizer,
                    "trained_on": self.dataset.uid,
                },
                indent=2,
            )
            # only the final step is declared; intermediates are undeclared --
            # written into the folder, absent from `files`, so nothing waits
            # on one that recovery skipped past
            if step == total_steps:
                self.artifact.paths(root)["checkpoint"].write_text(body)
            else:
                (folder / f"checkpoint_{step}.txt").write_text(body)
            worker.progress.update({"step": step, "total_steps": total_steps, "loss": loss})
            worker.log.info(f"step {step}/{total_steps} loss={loss:.4f} lr={lr:.2e}")
        self.artifact.paths(root)["progress"].write_text(
            json.dumps({"step": total_steps, "complete": True}, indent=2)
        )  # last, so `done` can't be observed before the checkpoint is durable
        worker.log.info("training complete")
