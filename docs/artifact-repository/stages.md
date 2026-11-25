# Stages

`artifacts.stages`: artifacts that train a model, built on the shared leg and
loop in `training` ([Training a Model](../training.md)). Which model a
leg trains is its `model` parameter, a package under `models/`
([Models](../models.md#implementations)).

## Pretraining

`artifacts.stages.pretraining.Pretraining` is a leg of next-token prediction,
at `runs/<run_id>/pretraining/<start>-<end>-<lineage>/`, trained by
`PretrainJob`.

| Field | Meaning |
| --- | --- |
| `dataset` | A [`DataSet`](dataset.md) or [`MappedDataSet`](mappeddataset.md). |
| `sampler` | Which positions each step reads, a [`Sampler`](stages/samplers.md); `RandomWindows()` unless stated. |
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
`evaluate`, which takes the leg's `loop_config.evaluations` ([Evaluations](stages/evaluations.md)) due at a step, each
on its own `eval_every` ([Training a Model](../training.md)). A finished leg (its `model.pt` exists) is
never rerun; an interrupted one resumes from its latest checkpoint.
