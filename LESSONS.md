# Lessons

Subtleties that cost time to find and would be easy to reintroduce. Each
entry says what is true, how we found out, and what to do about it. Add one
whenever a bug or a review turns up something the code cannot say for itself.

## `torch.save` is not byte-stable

Two files holding identical tensors and param groups can differ in bytes:
pickle memoizes objects by identity, so a state dict that went through
`load_state_dict` pickles in a different layout from a freshly built one.
Found while asserting `model.pt` was recovered "bit for bit" after a crash.

Compare checkpoints by loading them and comparing fields (`torch.equal` per
tensor, `==` on `param_groups`). Never hash a `.pt` file to answer "did this
change".

## The optimizer holds the learning rate of the current step

`TrainingJob.resume` does `group.update(asdict(optimizer_parameters))` after
`load_state_dict`, so a new leg trains with its own betas and weight decay.
That dict also carries `lr`, the fallback, which overwrote the scheduled
rate the failsafe had saved. With the rate set at the top of each iteration
this was invisible, except for an attempt that resumed from a failsafe
already at `end_step`: zero iterations, and the fallback landed in
`model.pt`, so the file depended on which attempt wrote it.

The fix is an invariant, not a patch at the write site: the optimizer holds
the rate of the step it is on. The loop sets it just before taking a step
and the checkpoint saves it with the model and optimizer of that step, so
the three are in sync. `resume` sets it after loading (for a fresh model, a
starting checkpoint or a failsafe alike) for the same reason. A checkpoint
written at step N therefore carries the rate step N was taken with,
whichever path produced it. Any loop built on `resume` must keep the first
half of that invariant.

## A finished leg is never rerun

`run` returns as soon as its own `model.pt` exists. Without that, a rerun
with the failsafes gone retrains the whole leg from scratch; on CUDA the
result is not bit-identical, and a later leg that already trained from the
old `model.pt` now has a starting checkpoint that changed under it. The step
check in `resume` cannot catch this, since the step is still `end_step`.

## Interruption testing: patch a method, compare against a straight run

The idempotence of the loop was checked by patching `set_learning_rate` to
raise before a chosen step (any crash point, not only after a failsafe) and
`write_atomic` to die between the `.tmp` save and the rename, or during a
strip rewrite, then comparing the final checkpoint against an uninterrupted
run in a separate root. Scenarios worth keeping: crash after every failsafe,
crash mid-interval, crash before the first failsafe, crash during strip,
stale `.tmp`, crash on the final write, rerun after completion, a multi-leg
chain with crashes in every leg, and random crash points. The runnable
version lives in the session scratchpad, not the repo; `test_pretrain.py`
keeps the cases that matter as unit tests.

## Dataclass inheritance with defaults needs `kw_only`

A subclass that re-declares a base field (to narrow its type, as
`Pretraining.starting_checkpoint` does) makes that field positional again, in
the base's slot. With a defaulted base field followed by required subclass
fields this raises "non-default argument follows default argument" at class
creation. `Training` and every subclass are `kw_only=True`, so field order is
never a constraint. Every artifact is constructed with keywords anyway.

## Artifact annotations must be real imports

`manifest.py` rebuilds artifacts with `get_type_hints`, which resolves each
annotation in the defining module's globals at call time. An import under
`TYPE_CHECKING` makes the annotation a string nothing can resolve, and
`from_manifest` fails. This is why a base artifact cannot hold a `dataset`
field typed with a family's class without importing that family, and why
`dataset` and `tokenizer` live on `Pretraining`, not on `Training`.

## Failsafe stripping writes the new file first

Under `optimizer_checkpoint_policy="latest"`, `failsafe` writes
`{step}.pt` and only then strips the optimizer from earlier ones. A crash in
between leaves two resumable failsafes, and `resume` takes the newest. The
other order would leave an instant with no resumable failsafe at all.

## The call id is a contextvar, and a thread does not inherit it

`modal.current_function_call_id()` reads a contextvar Modal sets for the
call's own thread. A `threading.Thread` the worker starts sees `None` there,
so a log filter keyed on the call id silently drops everything that thread
logs; the heartbeat's own warnings were the lines that went missing.

Start any such thread under `contextvars.copy_context().run(...)`, as
`initialize_worker` does, and the filter sees the same id on both threads.

## A traceback keeps the volume's files mapped after the call

A memmap over a tokens.bin is a local in `PretrainJob.run` and in every
frame under it (`batch_loss`, `get_batch`). When one of those raises, the
exception's `__traceback__` holds those frames and their locals, so the
mapping stays open for as long as anything holds the exception, and Modal's
runtime holds it past the call. The container is reused, the next call's
`volume.reload()` refuses to run over the open mapping, and it dies before
its own job starts. Closing the mmap by hand does not work either: numpy
exports the mmap's buffer for the array's lifetime, and `mmap.close()`
raises while an export exists. The only way to release the file is to drop
every reference to the array.

`initialize_worker` does that with `traceback.clear_frames` on the failure
path, after logging the traceback: it clears the locals of every finished
frame the exception unwound through (line numbers survive, so Modal's own
rendering of the error is intact). Any new place that opens a file or a
mapping on the mount inside a job is covered by the same clear, as long as
the reference lives in a frame and not on an object that outlives the call
(a module global, a cache, the job instance).

## A call the container never spoke for was invisible

Everything the dashboard knew about a call came from what the call itself
published: its beats, its rows, the file it streamed. A failure before any
of that existed, which is where the reload above landed, left a lease
pointing at a call id with no beat, no rows and no file: the launcher's
"granted" was the last word about it anywhere but Modal's own output, and
`calls_by_artifact`, which inverted the beats, never listed it.

Two things fix it and both are structural. The launcher records the call
itself, in `call_history` and the `:launcher` channel, at grant time, so a
call is listed by the thing that made it, not by whether it ever ran. And
`initialize_worker` attaches the buffer and starts the heartbeat before it
touches the mount, so the reload runs inside the same `try` as the job and
its failure is the call's first container row. The worker holds no log
file at all; leasebook appends the Dict's rows to the file on its own
clock, between two of its reloads.
