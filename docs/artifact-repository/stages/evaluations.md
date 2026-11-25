# Evaluations

`artifacts.stages.pretraining.evaluations` holds the evaluations a pretraining
leg can list in `loop_config.evaluations`. `TokenLoss(eval_every=200,
batch_size=64)` reads the validation split every 200 steps with a
[`Sweep`](samplers.md#sweep), every sequence of the model's `sequence_length`
once and in order, and returns `val_loss` and `val_loss_var`: the mean and
variance of the batch losses. Its `batch_size` is its own, not the
training batch size: evaluating needs no gradients, so it can take larger
batches, and the variance depends on the batch size it was measured with, so a
sweep over training batch sizes keeps one evaluation batch size to compare
across.

`BlockGradNorm(eval_every=10)` shows how the gradient is spread over the
blocks: every 10 steps it records the L2 norm of each block's gradient, as
`grad_norm_tblocks_0`, `grad_norm_tblocks_1`, ... for a Transformer (one per
block of every `ModuleList` directly on the network, so `grad_norm_blocks_<i>`
for Mamba). It reads the gradients the step's own backward pass left, so it
adds no forward or backward pass. They are read after clipping, which scales
every block by one factor on a step whose `grad_norm` exceeds `max_norm`: the
spread across blocks holds, the size is capped. Each norm gets its own chart
on the dashboard.

::: artifacts.stages.pretraining.evaluations
