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
| `starting_checkpoint` | The `Pretraining` leg this continues from, or `None`. |

The rest (model, training parameters, step range) come from `Training`.

A model speaks one tokenizer for life: construction refuses a model whose
`vocab_size` differs from the dataset tokenizer's, and a tokenizer that differs
from the starting checkpoint's.

`lineage_hash` digests the legs before it, its data, and its model and training
parameters, so the uid `<run_id>-pretraining-<start>-<end>-<hash>` identifies
a trajectory and not only a step range.

## The job

`PretrainJob` supplies what is specific to pretraining: cross-entropy over the
shifted targets, batches moved to the device from pinned memory, and
`evaluate`, which returns `val_loss` and `val_loss_var`, the mean and variance
over every validation batch. A finished leg (its `model.pt` exists) is never
rerun; an interrupted one resumes from its latest checkpoint.
