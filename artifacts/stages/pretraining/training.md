# Training legs and checkpoints

A pretraining run is a chain of legs. A leg is one `Pretraining` artifact
(`__init__.py`), produced by one `PretrainJob` (`jobs.py`), covering steps
`start_step` to `end_step`. The leg is `artifacts/core/SGD/training.py`'s
`Training`; the checkpoint mechanics are `TrainingJob` in
`artifacts/core/SGD/job.py`, which also builds the model and optimizer a leg
names. `Pretraining` adds the dataset and tokenizer, names its job, and
decides its own `uid`, `artifact_path` and `lineage_hash`; `PretrainJob`
draws the batches and owns the loop. Two kinds of checkpoint exist with
different meanings.

## The model

Models are not artifacts. A leg names one by `model`, the dotted path of a
package under `models/` (`"models.transformer"`), and holds that package's
`ModelParameters` as `model_parameters`. The package is torch-free at import
and exports two names: `ModelParameters`, and `build(parameters)`, which
imports the `nn.Module` and constructs it on `parameters.device`. `resume`
calls `locate(model).build(model_parameters)`; nothing in `artifacts/` knows
an architecture. The manifest codec cannot type `model_parameters` (its type
depends on `model`), so `Training.__post_init__` rebuilds it from the dict a
manifest hands back, through the same `locate`.

## Checkpoint files

Every checkpoint is one `.pt` file holding
`{"model": ..., "optimizer": ..., "step": ...}`: the two `state_dict`s
(`"optimizer"` present or not per the policy below) and the step the state is
the result of. Step 0 is the initialized model. The file, not its name or the
artifact, is the authority on the step; a starting checkpoint whose recorded
step disagrees with the leg's `start_step` is rejected.

Loading constructs the model and optimizer from this leg's parameters, then
`load_state_dict` into each. The optimizer is built over the live model's
parameters so the two stay linked; pickling whole objects would give the
optimizer its own copy of the weights and `step()` would update the wrong
tensors.

Files are written to a `.tmp` sibling and renamed into place: a checkpoint
exists whole or not at all.

## Artifact checkpoint: the leg's `model.pt`

The leg's durable output is the model and optimizer state at `end_step`,
declared by `Pretraining.files` as `model.pt`. The leg is complete when it
exists; the resolver waits on it, and other artifacts depend on the leg as a
whole.

A leg continues another by naming it as `starting_checkpoint`. The job loads
that leg's `model.pt` instead of initializing, and `start_step` is the previous
leg's `end_step`. `model` and `model_parameters` are inherited from the starting checkpoint
and must not differ; a leg without one must state them. A model speaks one
tokenizer for life: `Pretraining` refuses, at declaration, a `tokenizer`
whose `vocab_size` differs from `model_parameters.vocab_size`, a dataset not
tokenized with it, or a starting checkpoint trained with another. Optimizer
hyperparameters are always this leg's own: `load_state_dict` restores the
saved leg's, so the job overwrites them with
`training_parameters.optimizer_parameters` after loading. If the starting
checkpoint saved no optimizer state, the job warns and uses a fresh optimizer.

`starting_checkpoint` is itself a `Training`, so a leg carries its whole
ancestry (every earlier leg, dataset, tokenizer, training parameters).
Resolving a leg resolves the chain behind it.

## Failsafe checkpoint: the leg's `checkpoints/`

While training, the job writes `checkpoints/{step}.pt` every
`loop_config.checkpoint_every` steps. These are not artifacts: not in `files`,
nothing waits on them, no other artifact may reference them. On start, the job
resumes from the highest-step failsafe that holds optimizer state, falling
back to the starting checkpoint (or a fresh model) when there is none. They
also record the trajectory inside the leg, but that is a side effect, not a
contract.

### Optimizer state at failsafes

`loop_config.optimizer_checkpoint_policy`:

- `"all"`: every failsafe holds model and optimizer.
- `"latest"` (default): after writing a new failsafe, every earlier one is
  rewritten without `"optimizer"`. Recovery only needs the newest. The new
  file is written before the old ones are stripped, so at every instant at
  least one failsafe is resumable.

## The loop

`PretrainJob.run` is the loop, built on two things from `TrainingJob`:
`resume`, which builds the model and optimizer the leg names and loads state
into them, and `failsafe`.

`resume` returns `(model, optimizer, step)` following the priority above.
`step` counts completed steps; each iteration increments it first, so the
step being taken is step number `step`, its batch and learning rate are drawn
from that number, and after `optimizer.step()` the model is the result of
`step`, which a failsafe written there records. The optimizer holds the rate
of the current step: the loop sets it just before taking the step and the
checkpoint saves it with the model and optimizer, so a checkpoint at step N
carries the rate step N was taken with, whichever attempt wrote it. The loop
runs until `step` reaches `end_step`, then writes `model.pt` with
`"step": end_step`, even after zero iterations (the last failsafe of an
interrupted attempt landed on `end_step`). Once `model.pt` exists, `run`
returns without touching anything: later legs may have trained from it, and
on CUDA a retrain would not reproduce it bit for bit.

`get_batch` draws the batch for a step from `default_rng((seed, step))`, so
batches are a function of the seed alone and a resumed attempt sees exactly
the batches the interrupted one would have. No RNG state is checkpointed: the
model has no dropout, and batch selection is the only randomness after
initialization.

Every step's loss, gradient norm and learning rate go to `train.jsonl` in the
leg's folder through `StepLog` (`artifacts/core/SGD/steplog.py`), one JSON
row per step tagged with the attempt that took it. Rows are kept as device
tensors and read back in one batch when the loop flushes, after each
`evaluate` and once after the loop, so the only CPU/GPU sync points are those
flushes and `evaluate`'s `.item()` calls. Every
attempt appends: after a crash, the steps redone from the last failsafe
appear twice, under different attempts, which is what lets the file be split
back into continuous executions.
Each flush also publishes this attempt's rows so far to the `train` Dict,
for the dashboard's curves (see docs/LOGGING.md).

## TBD

Open recommendations, none applied. The first changes what
`model_parameters.dtype` means and is in the lineage hash, so decide it before
the first real run.

- **Mixed precision.** `dtype` sets the parameter dtype; bf16 gives pure-bf16
  training, where AdamW updates lose small updates to rounding. Standard
  practice is fp32 parameters with
  `torch.autocast(device_type, dtype=torch.bfloat16)` around forward and loss;
  `dtype` would then name the autocast dtype.
- **Weight decay groups.** One param group, so embeddings, RMSNorm gains and
  biases are decayed. Usual recipe: decay for `p.ndim >= 2`, none otherwise.
  `resume` re-applies `optimizer_parameters` to every group; with two groups
  it must leave the no-decay group's `weight_decay` at 0.
- **Validation set.** One random batch per `val_every`, so the curve is noisy.
  A fixed set of K batches from `default_rng(seed)` averaged under
  `torch.inference_mode()` gives a readable curve.
- **Failsafes and the volume.** Nothing commits the volume between
  `failsafe()` and the container dying. `system.runtime` commits once, on the
  way out of `initialize_worker`, which a hard kill (OOM, preemption, timeout)
  skips. Either confirm Modal background-commits this volume type, or give
  `Worker` a commit hook that `failsafe` calls.
- **`mmap=True` on the volume mount.** Stripping optimizers maps each earlier
  failsafe and renames over it. Whether Modal's FUSE mount supports the map
  and the rename under it needs a check on the real mount. Fallback: plain
  `torch.load` to CPU.
- **Peak memory on resume.** State is loaded with `map_location` set to the
  model's device while the fresh model and optimizer already sit there, so
  resume briefly holds two copies of both. `map_location="cpu"` removes the
  transient at the cost of a slower copy.
- **Host-to-device copies.** `torch.from_numpy(...).to(device)` blocks.
  `.pin_memory().to(device, non_blocking=True)` overlaps with the previous
  step's backward.
- **`gpu_check_every`** is declared in `LoopConfig` and unused. Log
  `torch.cuda` memory stats on that cadence or delete it.
- **`torch.compile(model)`** is usually a free 1.3 to 2x on this shape on CUDA.
- **Optimizer flags outside `OptimizerParameters`** (`amsgrad`, `fused`,
  `foreach`, `capturable`) come from the saved leg, since the re-apply covers
  only `lr`, `betas`, `eps`, `weight_decay`. Bites when a torch upgrade
  between legs changes a default.
- **Stale `.tmp` files.** A crash mid-write leaves `{step}.tmp` beside the
  failsafes. Ignored by the `*.pt` glob, overwritten if that step is written
  again, otherwise permanent litter.

## Naming and identity

A step range is not an identity: two legs can cover `10000-20000` and be
different models, from different histories or parameters. A leg's `uid` and
folder name carry a `lineage_hash` alongside the steps:

```
{run_id}-pretraining-{start_step}-{end_step}-{lineage_hash}
```

The hash digests everything that changes the trajectory: the starting
checkpoint's uid (recursively, the whole chain), dataset and tokenizer uids,
`model`, `model_parameters`, `training_parameters`. The steps are in the name only for
human readability.

## Layout

```
runs/
└── run_001/
    └── pretraining/
        └── 10000-20000-a83f2c/
            ├── manifest.json
            ├── model.pt                declared: the artifact
            ├── train.jsonl             undeclared: one row per step, per attempt
            └── checkpoints/            undeclared: recovery only
                ├── 12000.pt            model only
                ├── 14000.pt            model only
                └── 16000.pt            model + optimizer ("latest" policy)
```
