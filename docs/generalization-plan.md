# Extending the training stages to SFT

## Context

Training is organized as stages of a model: `Training`/`TrainingJob` in
[artifacts/core/SGD/](../artifacts/core/SGD/) hold what every SGD-trained leg shares,
including building the model and optimizer a leg names; `Pretraining`/`PretrainJob` in
[artifacts/stages/pretraining/](../artifacts/stages/pretraining/) add the token stream and
next-token batches. Models are not artifacts: a leg names a package under
[models/](../models/) by dotted path and holds its `ModelParameters`. The question is how
well that split holds up when a second stage, SFT, arrives with a different target
construction (loss only on response tokens) and padded, variable-length examples instead
of a flat token stream.

Decisions taken:
- SFT is a **new stage**: base weights only, fresh optimizer, step counter and LR schedule
  restart at 0. `starting_checkpoint` keeps meaning "continue this stage".
- Batches are **one example per row, right-padded, loss mask only**. No attention mask
  plumbing; nothing under `models/` changes.
- SFT record format is **undecided**; the dataset is designed as a seam so the record
  parser is a detail of its job.

### What generalizes as is

- Checkpoint file format, `write_atomic`, `failsafe`, the optimizer-stripping policy,
  `model.pt` as the completion file, the resolver waiting on it.
- Model and optimizer construction: `resume` builds both from `model`,
  `model_parameters` and `training_parameters`, so an SFT job inherits them.
- `starting_checkpoint: Training | None` on the base class: `from_manifest` picks the
  concrete class from the manifest, so an SFT leg can continue an SFT leg with no change to
  the codec.
- `LoopConfig`, `TrainingParameters`, `LRSchedule`, `set_learning_rate`. With a stage
  starting at step 0 the schedule is naturally stage-relative.
- The loss. `torch.nn.CrossEntropyLoss(ignore_index=-100)` is the pretraining loss too:
  "different loss function" is really "different targets", so the seam is batch
  construction, not the loss.
- The model. Causal `tril` plus right padding plus `ignore_index` is sufficient; `SPDA`
  already accepts a mask if packing is wanted later.

### What blocks or duplicates

1. **`starting_checkpoint` conflates two relations.** `resume` treats any loaded state as
   "continue": it checks `state["step"] == start_step`, reloads the optimizer, and warns
   when none is saved. SFT needs "initialize from": weights only, step 0, fresh optimizer,
   no step check, no warning. Needs a second artifact-valued field.
2. **`PretrainJob.run` owns the loop.** An `SFTJob` would copy all of it and change one
   call. With two callers the loop belongs in `TrainingJob`; `evaluate` too, since it
   reads `dataset.valid_tokens`, a pretraining-only attribute.
3. **`Training.__post_init__` requires `model`/`model_parameters` when there is no
   `starting_checkpoint`.** An SFT leg gets them from its base model instead.
4. **The dataset seam is a flat `train_tokens`/`valid_tokens` array**, sampled by
   `get_batch` with random windows. SFT samples whole examples with a response offset.
   `DataSet`/`MappedDataSet` cannot express that; a new family is needed.

## Changes

### 1. `artifacts/core/SGD/training.py`

- Add `base_model: Training | None = None` to `Training`: the leg whose `model.pt`
  weights this stage starts from at step 0. `deps()` picks it up from the annotation.
- `__post_init__`: at most one of `starting_checkpoint`, `base_model`. `model` and
  `model_parameters` are inherited from whichever is given and must agree with it if
  stated; required when neither is.
- `start_step` unchanged (`starting_checkpoint.end_step` else 0), so a leg with only
  `base_model` starts at 0.

### 2. `artifacts/core/SGD/job.py`

- `resume`: after failsafes and `starting_checkpoint` both miss, if `base_model` is set,
  load `base_model.paths(root)["model"]`, `model.load_state_dict(state["model"])` only,
  `step = 0`, log `"loaded base {uid} at step {saved}, starting stage at step 0"`. No
  step check, no optimizer warning. Priority: failsafe > starting_checkpoint >
  base_model > fresh init.
- Move the loop into `TrainingJob.run(root, worker)` and `evaluate` with it, verbatim
  from `PretrainJob.run` except for the data calls. Two new abstract methods:
  - `splits(root) -> tuple[Any, Any]`: the bound `(train, valid)` data this leg trains on.
    One call, at the top of `run`.
  - `batch_loss(model, split, step) -> torch.Tensor`: the loss of step `step`'s batch
    drawn from `split`.
  `evaluate` calls `self.batch_loss(model, valid, step)`. Module-level
  `LOSS = torch.nn.CrossEntropyLoss(ignore_index=-100)` for both stages.

### 3. `artifacts/stages/pretraining/`

- `Pretraining` unchanged. `PretrainJob` keeps `get_batch`, implements `splits`
  (`ds.train_tokens, ds.valid_tokens` from `self.dataset.bind(root)`) and `batch_loss`;
  `run` and `evaluate` are deleted (moved to core).

### 4. New stage `artifacts/stages/sft/`

- `__init__.py`: `SFT(Training)`:
  ```python
  producer = "artifacts.stages.sft.jobs.SFTJob"
  dataset: SFTDataSet
  tokenizer: Tokenizer
  ```
  `__post_init__`: raise unless `starting_checkpoint or base_model`; the tokenizer must
  equal the base chain's (`(base_model or starting_checkpoint).tokenizer`). `lineage_hash`:
  `_digest(starting_checkpoint.uid, base_model.uid, dataset.uid, tokenizer.uid, model,
  asdict(model_parameters), asdict(training_parameters))`. `uid`
  `{run_id}-sft-{start}-{end}-{hash}`, `artifact_path` `runs/{run_id}/sft/{start}-{end}-{hash}`.
- `jobs.py`: `SFTJob(TrainingJob)`: `splits` returns `(ds.train_examples,
  ds.valid_examples)`; `batch_loss` calls `get_sft_batch(examples, batch_size,
  sequence_length, seed, step, device)`: `default_rng((seed, step))` picks `batch_size`
  example indices; each example is `(tokens, response_start)`; rows truncated to
  `sequence_length + 1`, right-padded to the longest in the batch; `inputs = row[:-1]`,
  `targets = row[1:]` with `-100` at every position `< response_start - 1` and every pad
  position. Pad input id is the dataset's `<|endoftext|>` id.

### 5. New family `artifacts/sftdataset/`

Shape only; the record format is still open, so this family is the seam that hides it.

- `SFTDataSet(Artifact)`: `tokenizer: Tokenizer`, `train_set: tuple[Source, ...]`,
  `valid_set: tuple[Source, ...]`, producer `artifacts.sftdataset.jobs.SFTDataSetJob`.
  `uid` `sft-{digest(tokenizer.uid, ordered source uids)}`, path `sftdatasets/{uid}`.
  Shared, not run-scoped. `__post_init__` requires `<|endoftext|>` in
  `tokenizer.special_tokens` (own `END_OF_TEXT` constant).
- `files`: per split `{split}.bin` (uint16 tokens of every example back to back, each
  ending in `<|endoftext|>`) and `{split}.idx` (int64 rows of
  `start, response_start, end`).
- `_load`: memmap both per split; bound view exposes `train_examples` /
  `valid_examples`, each a sequence of `(tokens, response_start)`, plus `pad_id`.
- `jobs.py` `SFTDataSetJob`: binds each source, parses records, encodes prompt and
  response with the bound tokenizer, writes the two files atomically per split. The
  parser `records(text) -> Iterator[tuple[str, str]]` stays `raise NotImplementedError`
  until the format is decided.

### 6. `artifacts/stages/pretraining/training.md`

- "The loop": it is `TrainingJob.run`; a stage's job supplies `splits` and `batch_loss`.
- New section "Stages": `base_model` vs `starting_checkpoint`, resume priority, step 0
  and fresh optimizer at a stage boundary, the loss mask, padding as the batching choice
  with packing as a later option inside `SFTJob`.
- Layout tree gains `runs/run_001/sft/0-1000-<hash>/`.

### 7. Tests

- `test/test_pretrain.py`: unchanged in intent.
- New `test/test_sft.py` mirroring it: a tiny `Pretraining` built in a temp root, an
  `SFTDataSet` with hand-written examples written directly to the bin/idx files, then:
  - resume from base loads weights, step 0, optimizer has no state, no warning logged;
  - `get_sft_batch` targets are `-100` on prompt and pad positions and real ids on
    response positions; same `(seed, step)` gives the same batch;
  - crash after first failsafe, resume continues from it;
  - a second SFT leg via `starting_checkpoint` starts at the first's `end_step`;
  - `SFT(...)` with neither `base_model` nor `starting_checkpoint` raises; with a
    tokenizer disagreeing with the base raises.

## Verification

- `.venv/bin/python -m unittest discover -s test -p 'test_*.py'` on CPU.
- Manifest round trip of an `SFT` leg equals the original and `deps()` lists
  `base_model`, `dataset`, `tokenizer` in field order.
- Grep for `dataset.valid_tokens` and `PretrainJob.run` after the move: no callers left.
