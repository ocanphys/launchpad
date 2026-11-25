"""The evaluations a pretraining leg's loop can take of its model, each an
Evaluation (training.leg) listed in the leg's
loop_config.evaluations. Torch loads inside `measure`, so declaring one does
not import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from artifacts.stages.pretraining.samplers import Sweep
from training.leg import Evaluation

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


class BlockGradNorm(Evaluation):
    """The L2 norm of each block's gradient at the step, for each block of every
    ModuleList directly on the network, as `grad_norm_<list>_<i>`
    (`grad_norm_tblocks_0` for a Transformer's first block). It reads the step's
    own gradients, so it costs no forward or backward pass.

    They are read after clipping: a step whose `grad_norm` exceeded max_norm
    has every block scaled down by one factor, so how the norm is spread over
    the blocks holds and its size does not.

        BlockGradNorm(eval_every=10) -> BlockGradNorm(eval_every=10)
    """

    def measure(
        self, job: PretrainJob, model: torch.nn.Module, dataset: Any, worker: Worker
    ) -> dict[str, torch.Tensor]:
        import torch

        return {
            f"grad_norm_{name}_{i}": torch.nn.utils.get_total_norm([p.grad for p in block.parameters()])
            for name, blocks in model.named_children()
            if isinstance(blocks, torch.nn.ModuleList)
            for i, block in enumerate(blocks)
        }
