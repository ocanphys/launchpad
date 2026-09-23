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

## Ask the record which call it is, never the thread

`modal.current_function_call_id()` reads a contextvar Modal sets for the
call's own thread. A `threading.Thread` the worker starts sees `None` there,
so a log filter that asked the thread silently dropped everything that
thread logged; the heartbeat's own warnings were the lines that went
missing. `contextvars.copy_context().run(...)` fixed that one thread and
nothing else: a library's thread, and any thread a job starts for itself,
still carried no id.

So the id travels on the record instead. `call_logger` is a `LoggerAdapter`
holding `extra={"call_id": ...}`, which stamps every record it makes from
any thread, and `BufferHandler` reads the stamp rather than the ambient
state. Two consequences worth knowing:

- Only a filter on a **handler** sees records from other loggers. A filter
  on the root *logger* never runs for a record logged on `torch.foo`:
  `Logger.handle` runs only the originating logger's filters, and
  `callHandlers` then walks ancestors collecting handlers, consulting no
  further logger filters. Levels are the opposite and do inherit, which is
  why `setLevel("modal")` silences `modal.client` for free.
- A record is one object shared by every handler, so a handler that
  *writes* a stamp is visible to every handler after it. Read with a
  default; do not mutate.

A library never stamps anything, so an unstamped record is the container's,
not a mistake: it goes to that container's `ambient` source rather than being
dropped.

## A library that keeps its own handler never reaches ours

`import torch` gives `torch` and some fifty `torch.*` loggers a
`StreamHandler` of their own on **stderr** and `propagate = False`, one per
module rather than one for the family. Propagation is what carries a record
to the root logger's handlers, so none of it reaches `BufferHandler`, and
putting only the `torch` logger back would not help: `torch._dynamo` sets
`propagate = False` on *itself*. Found by a test asserting on `ambient` that
passed alone and failed in the suite, where an earlier test had imported
torch.

So `ambient` holds what propagates, not everything the container printed:
those lines reach Modal's own capture and stop there. Before assuming any
library is captured, check the whole family, not the top name:

    [(n, l.propagate, bool(l.handlers)) for n, l in
     logging.root.manager.loggerDict.items() if isinstance(l, logging.Logger)]

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

## A background thread makes the mount's reload race its readers

`volume.reload()` replaces the mount's view of the volume and refuses to run
while the process holds a file under it open. `leasebook` was safe from that
without trying: Modal hands it one input at a time (no `@modal.concurrent`)
and its routes are sync `def`, so a reload could never overlap a read. That
argument stopped holding the day a thread started reloading on its own -- the
listener that recomputes the state map when a worker's call exits. Both
directions bite: the thread's reload while `/artifact/<path>` has
`train.jsonl` open raises and loses the refresh, and a read that starts
mid-reload sees files being swapped under it (a half-written manifest reads
as a conflict that is not real).

`main.mount_lock` is what holds them apart, and everything in that process
which reloads the mount or opens a file on it takes it: `state()`,
`attempt_launch`'s readiness read, and the `/artifact` route. Two rules keep
it from becoming its own problem: never hold it across the listener's
blocking queue read (that wait is `REFRESH_WAIT_SECONDS` long, and a request
would wait it out), and keep those route handlers sync `def` -- an `async
def` runs on the event loop rather than the threadpool, and blocking on a
`threading.Lock` there stalls every other request in the container.

## A call the container never spoke for was invisible

Everything the dashboard knew about a call came from what the call itself
published: its beats, its rows, the file it streamed. A failure before any
of that existed, which is where the reload above landed, left a lease
pointing at a call id with no beat, no rows and no file: the launcher's
"granted" was the last word about it anywhere but Modal's own output, and
the call index, which inverted the beats, never listed it.

Two things fix it and both are structural. The launcher records the call
itself, in `call_history` and the `:launcher` channel, at grant time, so a
call is listed by the thing that made it, not by whether it ever ran. And
`initialize_worker` attaches the buffer and starts the heartbeat before it
touches the mount, so the reload runs inside the same `try` as the job and
its failure is the call's first container row. The worker holds no log
file at all; `persist_logs` appends the Dict's rows to the file on its
schedule.

## Encoding line by line is not the same token sequence as encoding whole

`TokenizeSourceJob` feeds `encode_iterable` one line at a time, so no
pretoken can span a newline. `PAT` ends in `\s+(?!\S)|\s+`, which means a
whitespace run that crosses a line boundary pretokenizes differently than it
would in one string:

    'a  \nb'    whole ['a', '  ', '\n', 'b']   by line ['a', '  \n', 'b']
    'a\n\n'     whole ['a', '\n\n']            by line ['a', '\n', '\n']

Both encodings decode back to the identical source, so nothing is lost; the
ids just differ at those boundaries. Found while checking the streaming
rewrite against the old whole-file `encode()`: 2 MB of TinyStories came out
byte-identical (717 blank lines and all, because a run *followed* by text
already splits the same way), and only trailing spaces, whitespace-only
lines and a file ending in a blank line diverge.

What this costs: a `tokens.bin` built before the rewrite is not reproducible
by rebuilding it. Don't diff one against a fresh one to decide whether a
tokenizer changed -- compare `tokenizer.json`, or decode both and compare
the text.
