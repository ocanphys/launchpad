# Samplers

`artifacts.stages.pretraining.samplers`: which token positions each step reads
from a split, whether a training step of a [pretraining leg](../stages.md) or a
batch of an evaluation. A leg's `sampler` field holds one
`Sampler` subclass, and its manifest names that class under `"class"` beside its
fields, the way it names its `lr_schedule`.

```python
Pretraining(..., sampler=Epochs(max_epochs=3))
```

The leg passes the shape it wants filled: its `batch_size`, its model's
`sequence_length`, its `seed` and its `end_step`. A sampler holds none of these,
only the recipe.

## Starts and batches

A sampler is two functions, each built once per leg:

- `starts(split, batch_size, sequence_length, seed, end_step)` returns the
  function from a step to the `batch_size` positions of the split its windows
  start at. This is the recipe; each subclass writes only this.
- `batches(...)`, with the same arguments, returns the function from a step to
  `(inputs, targets)`: the windows at `starts(step)`, as int64 arrays. It is
  written once, on the base class, and it is what `PretrainJob` calls.

Every step's positions are recomputable from the leg's manifest alone, so the
tokens a step trained on, and their tags (`split.tags[s : s + sequence_length + 1]`),
can be read back afterwards with nothing stored during training.

```python
split = dataset.bind(root).train
leg.sampler.starts(split, 64, 256, leg.training_parameters.seed, leg.end_step)(1200)
```

## The contract

Every sampler keeps four rules:

- **Pure.** `starts(step)` depends only on the sampler's fields, the split,
  the arguments and the step. Any step can be asked for in any order, so a
  resumed leg reads exactly what the interrupted one would have, and no
  sampler state is checkpointed.
- **The whole leg or nothing.** Every step up to `end_step` gets exactly
  `batch_size` starts, each with a full window and its next token after it. A
  split that cannot serve that raises from `starts`, which the job calls before
  its first step.
- **Frozen once used.** A sampler's behaviour and its fields stay as they were
  when a leg first used it. A different behaviour or a new field is a new
  class.
- **No fields on the base.** A field on `Sampler` would appear in every
  sampler's manifest.

## RandomWindows

The default. Each step draws `batch_size` uniformly random starts from
`default_rng((seed, step))`, so windows overlap and repeat, and no step depends
on another.

## Epochs

Every sequence once per epoch, shuffled afresh each epoch. For a split of `n`
tokens:

| | |
| --- | --- |
| Sequences | `S = (n - 1) // sequence_length`, placed `sequence_length` apart, so each has its next token |
| Steps per epoch | `S // batch_size`; the partial batch at the end of an epoch is dropped |
| Step `s` | epoch `(s - 1) // steps_per_epoch`, batch `(s - 1) % steps_per_epoch` of it |
| Epoch `e` | `default_rng((seed, e))` gives a permutation of the `S` sequences and an offset in `[0, (n - 1) % sequence_length]` |

The offset moves every sequence along by up to the tokens no sequence covers,
so a different tail goes unread each epoch, and since the permutation changes
too, so does the partial batch dropped. Epochs count from the run's step 1, so
a leg continuing another continues its epoch.

`max_epochs`, when set, refuses a leg whose `end_step` lies past that many
epochs; the error names the last step they serve. A split too short for one
batch is refused.

`Epochs().steps_per_epoch(leg)` is the steps-per-epoch row for a leg: it binds
the leg's dataset, which must be built, and reads its training split's length.
`leg.end_step / Epochs().steps_per_epoch(leg)` is how many epochs the leg ends at.

## Sweep

Every sequence once, in order, from position 0: step `s` reads sequences
`(s - 1) * batch_size` to `s * batch_size - 1`, each `sequence_length` apart.
Nothing is random, so the seed is unused, and a leg ending past the last full
batch is refused, so a leg declaring `sampler=Sweep()` trains on one pass of
its split in a fixed order. `TokenLoss` reads the validation split with it, passing seed
0 ([Evaluations](evaluations.md)), so the positions behind every
validation loss, and their tags, are as recomputable as a training step's.

## Why

**The sampler belongs to the leg, not the dataset.** How batches are drawn
changes the trajectory, as the seed and batch size do, so it is in the leg's
`lineage_hash`. On a dataset it would change the dataset's identity, and a
`DataSet` would copy the same bytes again under a new name.

**One polymorphic field, so identity only grows.** Adding a sampler class
leaves every existing manifest, and so every existing hash, as it was. A flag
per mode would add a field to every leg's manifest each time a mode is added.

**Positions, not tokens.** A sampler returns starts and `batches` gathers them.
That keeps each recipe to the one thing it decides, and it makes every batch's
provenance a computation over the manifest: which positions, and so which
sources and tags.

::: artifacts.stages.pretraining.samplers
    options:
      heading_level: 3
