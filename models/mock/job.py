import json
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from dag.job import Job
from models.mock.artifact import Pretraining

if TYPE_CHECKING:
    from system.runtime import Worker


class PretrainJob(Job):
    artifact: Pretraining

    def __init__(self, artifact: Pretraining):
        super().__init__(artifact)
        self.dataset = artifact.dataset
        self.tokenizer = artifact.tokenizer

    def run(self, root: Path, worker: "Worker") -> None:
        train_ids = array("H")
        train_ids.frombytes(self.dataset.paths(root)["training set"].read_bytes())
        vocab_size = len(
            json.loads(self.tokenizer.paths(root)["tokenizer"].read_text())["vocab"]
        )

        model_parameters = self.artifact.model_parameters
        config = self.artifact.config
        folder = root / self.artifact.artifact_path
        every = config.checkpoint_every
        steps = list(range(every, config.total_steps + 1, every))
        if not steps or steps[-1] != config.total_steps:
            steps.append(config.total_steps)

        worker.log.info(
            f"{config.total_steps} steps, "
            f"hidden_size={model_parameters.hidden_size}, num_layers={model_parameters.num_layers}, "
            f"{len(train_ids)} train tokens"
        )

        loss, done = 10.0, 0
        for step in steps:
            for _ in range(step - done):
                loss *= 0.99  # mock decay, not a real training loop
            done = step
            body = json.dumps(
                {
                    "step": step,
                    "model_parameters": {
                        "hidden_size": model_parameters.hidden_size,
                        "num_layers": model_parameters.num_layers,
                    },
                    "config": {
                        "batch_size": config.batch_size,
                        "lr": config.lr,
                        "seed": config.seed,
                    },
                    "vocab_size": vocab_size,
                    "trained_on": self.dataset.uid,
                    "num_train_tokens": len(train_ids),
                    "loss": loss,
                },
                indent=2,
            )
            # only the final step is declared; intermediates are undeclared --
            # written into the folder, absent from `files`, so nothing waits
            # on one that recovery skipped past
            if step == config.total_steps:
                self.artifact.paths(root)["checkpoint"].write_text(body)
            else:
                (folder / f"checkpoint_{step}.txt").write_text(body)
            worker.log.info(f"step {step}/{config.total_steps} loss={loss:.4f}")
        self.artifact.paths(root)["progress"].write_text(
            json.dumps({"step": config.total_steps, "complete": True}, indent=2)
        )  # last, so `done` can't be observed before the checkpoint is durable
        worker.log.info("training complete")
