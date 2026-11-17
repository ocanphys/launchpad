# Stages

`artifacts.stages`: artifacts that train a model, built on the shared leg and
loop in `artifacts.core.SGD` ([Training a Model](../training.md)). Which model a
leg trains is its `model` parameter, a package under `models/`
([Model Repository](../model-repository.md)).

## Pretraining

`artifacts.stages.pretraining.Pretraining` is a leg of next-token prediction,
at `runs/<run_id>/pretraining/<start>-<end>-<lineage>/`, trained by
`PretrainJob`.

| Field | Meaning |
| --- | --- |
| `dataset` | A [`DataSet`](dataset.md) or [`MappedDataSet`](mappeddataset.md). |
| `sampler` | Which positions each step reads, a [`Sampler`](samplers.md); `RandomWindows()` unless stated. |
| `autocast` | `"torch.bfloat16"` runs forward and loss, in training and in every evaluation, under `torch.autocast` in bf16, with the parameters and optimizer in the model's `dtype`; `None`, the default, runs them in the model's `dtype`. MFU is measured against the GPU's peak at that precision. |
| `starting_checkpoint` | The `Pretraining` leg this continues from, or `None`. |

The rest (model, training parameters, step range) come from `Training`.

Mixed precision is `autocast="torch.bfloat16"` over an fp32 model, not a model
with `dtype="torch.bfloat16"`: bf16 parameters lose AdamW's small updates to
rounding, while under autocast the parameters and optimizer state stay fp32
and only the matmuls run in bf16; the loss is computed in fp32 either way.

A model speaks one tokenizer for life: construction refuses a model whose
`vocab_size` differs from the dataset tokenizer's, and a tokenizer that differs
from the starting checkpoint's.

`lineage_hash` digests the legs before it, its data, and its model, sampler,
autocast and training parameters, so the uid `<run_id>-pretraining-<start>-<end>-<hash>` identifies
a trajectory and not only a step range.

## The job

`PretrainJob` supplies what is specific to pretraining: cross-entropy over the
shifted targets, batches moved to the device from pinned memory, and
`evaluate`, which takes the leg's `loop_config.evaluations` due at a step, each
on its own `eval_every` ([Training a Model](../training.md)). A finished leg (its `model.pt` exists) is
never rerun; an interrupted one resumes from its latest checkpoint.

## Evaluations

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
    options:
      heading_level: 3
