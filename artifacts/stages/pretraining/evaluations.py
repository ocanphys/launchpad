"""The evaluations a pretraining leg's loop can take of its model, each an
Evaluation (artifacts.core.SGD.training) listed in the leg's
loop_config.evaluations. Torch loads inside `measure`, so declaring one does
not import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from artifacts.core.SGD.training import Evaluation
from artifacts.stages.pretraining.samplers import Sweep

if TYPE_CHECKING:
    import torch

    from artifacts.stages.pretraining.jobs import PretrainJob
    from system.runtime import Worker


class TokenLoss(Evaluation):
    """The next-token loss over the validation split, read by a Sweep
    (artifacts.stages.pretraining.samplers) with seed 0: every sequence of the model's
    sequence_length once, in order, batch_size of them to a batch.

    `val_loss` is the mean of the batch losses and `val_loss_var` their
    variance, so 0 for a single batch. A split shorter than one batch is one
    batch of the sequences it holds; the tail too short to fill a batch is in
    no batch. While it runs, worker.progress["val_batch"] reads "done/total".

        TokenLoss(eval_every=200, batch_size=64) -> TokenLoss(eval_every=200, batch_size=64)
    """

    batch_size: int = Field(gt=0)

    def measure(
        self, job: PretrainJob, model: torch.nn.Module, dataset: Any, worker: Worker
    ) -> dict[str, torch.Tensor]:
        import torch

        split = dataset.valid
        sequence_length = job.artifact.model.sequence_length
        sequences = (len(split.tokens) - 1) // sequence_length
        if sequences < 1:
            raise ValueError(
                f"the validation split holds {len(split.tokens)} tokens, fewer than one "
                f"sequence of {sequence_length} and its next token"
            )
        batch_size = min(self.batch_size, sequences)
        total = sequences // batch_size
        batch = Sweep().batches(split, batch_size, sequence_length, seed=0, end_step=total)
        losses = []
        for step in range(1, total + 1):
            worker.progress["val_batch"] = f"{step}/{total}"
            losses.append(job.batch_loss(model, *job.to_device(batch(step))))
        del worker.progress["val_batch"]
        losses = torch.stack(losses)
        return {"val_loss": losses.mean(), "val_loss_var": losses.var(correction=0)}
