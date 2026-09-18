# Training legs and checkpoints

A pretraining run is a chain of legs. Each leg is one `Pretraining` artifact
(`__init__.py`), produced by one `PretrainJob` (`jobs.py`), and covers the steps
from `start_step` to `end_step`. Two kinds of checkpoint exist, and they mean
different things.

## Checkpoint files

Every checkpoint, final or failsafe, is one `.pt` file holding a dict
`{"model": ..., "optimizer": ..., "step": ...}`: the two `state_dict`s, with
`"optimizer"` present or not depending on the policy below, and the step the
state is the result of. Step 0 is the initialized model before it has seen
a training step. The file, not its name or the artifact, is the authority on
the step; a starting checkpoint whose recorded step disagrees with the leg's
`start_step` is rejected. Loading always means constructing the
model and optimizer from this leg's parameters first, then `load_state_dict`
into each. The optimizer is built over the live model's parameters, so the
two stay linked; pickling whole objects into separate files would give the
optimizer its own copy of the weights and `step()` would update the wrong
tensors.

Files are written to a `.tmp` sibling and renamed into place, so a checkpoint
either exists whole or not at all.

## Artifact checkpoint: the leg's `model.pt`

The durable output of a leg is the model and optimizer state at `end_step`.
An artifact can comprise several files, all inside its own folder; a
`Pretraining` comprises one, which `Pretraining.files` declares as
`model.pt`. The leg is complete when it exists; the resolver waits on it, and
other artifacts depend on the leg as a whole.

A leg continues another by naming it as `starting_checkpoint`. The job then
loads that leg's `model.pt` instead of initializing a model, and
the new leg's `start_step` is the previous leg's `end_step`. `model_parameters`
are inherited from the starting checkpoint and must not differ from it; a leg
without a starting checkpoint has to state them. Optimizer hyperparameters
are always this leg's own: `load_state_dict` restores the saved leg's, so the
job overwrites them with `training_parameters.optimizer_parameters` after
loading. If the starting checkpoint saved no optimizer state, the job warns
and continues with a fresh optimizer.

Because `starting_checkpoint` is itself a `Pretraining`, a leg carries its
whole ancestry: every earlier leg, its dataset, tokenizer, and training
parameters. Resolving a leg resolves the chain behind it.

## Failsafe checkpoint: the leg's `checkpoints/`

While a leg trains, the job writes `checkpoints/{step}.pt` every
`loop_config.checkpoint_every` steps. These are not artifacts. They are not
in `files`, nothing waits on them, and no other artifact may reference them.
They exist so that a job interrupted mid-leg can pick up where it stopped: on
start, the job takes the highest-step failsafe that still holds optimizer
state, and only falls back to the starting checkpoint (or a fresh model) when
there is none.

They double as a record of the trajectory inside the leg for anyone who wants
to look at intermediate weights, but that is a side effect, not a contract.

### Optimizer state at failsafes

`loop_config.optimizer_checkpoint_policy` decides how much optimizer state the
failsafes keep:

- `"all"`: every `checkpoints/{step}.pt` holds both model and optimizer.
- `"latest"` (default): after writing a new failsafe, the job rewrites every
  earlier one without its `"optimizer"` entry. Recovery only ever needs the
  newest, so this is the cheaper choice unless the intermediate optimizer
  states are themselves of interest. The new file is written before the old
  ones are stripped, so at every instant at least one failsafe is resumable.

## The loop

`PretrainJob.resume` returns the `(model, optimizer, step)` the attempt trains
from, following the priority above. `step` counts completed steps: the batch
and learning rate for an iteration are those of step number `step`, and after
`optimizer.step()` the model is the result of `step + 1`, which is what a
failsafe written at that point records. The loop runs until `step` reaches
`end_step`, then writes `model.pt` with `"step": end_step`. It writes it even
when it ran zero iterations, which is what happens when the last failsafe of
an interrupted attempt landed on `end_step` itself.

`get_batch` draws the batch for a step from `default_rng((seed, step))`, so
the sequence of batches is a function of the seed alone and a resumed attempt
sees exactly the batches the interrupted one would have. That is why no RNG
state is checkpointed: the model has no dropout, and batch selection is the
only randomness after initialization.

Loss, validation loss, learning rate and gradient norm are read off the GPU
only every `val_every` steps; those `.item()` calls are the only points where
the CPU waits for the GPU, so the rest of the loop pipelines freely.

## TBD

Open recommendations, none applied yet. Each is a decision, and the first
changes what `model_parameters.dtype` means, so decide it before the first
real run (it is in the lineage hash).

- **Mixed precision.** `dtype` currently sets the parameter dtype. Setting it
  to bf16 gives pure-bf16 training, where AdamW updates on bf16 weights lose
  small updates to rounding. Standard practice is fp32 parameters with
  `torch.autocast(device_type, dtype=torch.bfloat16)` around forward and loss.
  `dtype` would then name the autocast dtype, not the parameter dtype.
- **Weight decay groups.** The optimizer is built over `model.parameters()`
  in one group, so embeddings, RMSNorm gains and any biases are decayed. The
  usual recipe is two groups: decay for `p.ndim >= 2`, none for the rest.
  `resume` re-applies `optimizer_parameters` to every group after loading;
  with two groups it must leave the no-decay group's `weight_decay` at 0.
- **Validation set.** Validation is one random batch per `val_every`, so the
  curve is noisy. A fixed set of K batches drawn once from `default_rng(seed)`
  and averaged under `torch.inference_mode()` gives a readable curve for a
  few extra forwards.
- **Failsafes and the volume.** Nothing commits the volume between
  `failsafe()` and the container dying, and the container dying is the only
  reason failsafes exist. `system.runtime` commits once, on the way out of
  `initialize_worker`, which a hard kill (OOM, preemption, timeout) skips.
  Either confirm Modal background-commits this volume type, or give `Worker`
  a commit hook that `failsafe` calls.
- **`mmap=True` on the volume mount.** Stripping optimizers maps each earlier
  failsafe and then renames over it. Fine on a local POSIX filesystem;
  whether Modal's FUSE mount supports the map, and the rename under it, needs
  a check on the real mount. Fallback is a plain `torch.load` to CPU.
- **Peak memory on resume.** State is loaded with `map_location=self.device`
  while the fresh model and optimizer already sit there, so resume briefly
  holds two copies of both. `load_state_dict` copies across devices itself,
  so `map_location="cpu"` removes the transient at the cost of a slower copy.
- **Host-to-device copies.** `torch.from_numpy(...).to(device)` blocks.
  `.pin_memory().to(device, non_blocking=True)` overlaps the copy with the
  previous step's backward.
- **`gpu_check_every`** is declared in `LoopConfig` and used nowhere. Either
  log `torch.cuda` memory stats on that cadence or delete the field.
- **`torch.compile(model)`** is usually a free 1.3 to 2x on this model shape
  once on CUDA.
- **Optimizer flags outside `OptimizerParameters`** (`amsgrad`, `fused`,
  `foreach`, `capturable`) come from the saved leg after `load_state_dict`,
  since the re-apply only covers `lr`, `betas`, `eps`, `weight_decay`. A
  torch upgrade between legs that changes a default is the case this bites.
- **Stale `.tmp` files.** A crash mid-write leaves `{step}.tmp` beside the
  failsafes. Ignored by the `*.pt` glob and overwritten if that step is
  written again; otherwise permanent litter.

## Naming and identity

A step range is not an identity. Two legs can both cover `10000-20000` and be
different models, because they descend from different histories or were
trained with different parameters. So a leg's `uid` and folder name carry a
`lineage_hash` alongside the steps:

```
{run_id}-pretraining-{start_step}-{end_step}-{lineage_hash}
```

The hash digests everything that changes the trajectory: the starting
checkpoint's uid (and so, recursively, the whole chain), the dataset and
tokenizer uids, `model_parameters`, and `training_parameters`. The steps stay
in the name only so a human can read the folder listing.

## Layout

```
runs/
└── run_001/
    └── pretraining/
        └── 10000-20000-a83f2c/
            ├── manifest.json
            ├── model.pt                declared: the artifact
            └── checkpoints/            undeclared: recovery only
                ├── 12000.pt            model only
                ├── 14000.pt            model only
                └── 16000.pt            model + optimizer ("latest" policy)
```
