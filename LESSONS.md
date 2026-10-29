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

`TrainingJob.resume` does `group.update(optimizer_parameters.model_dump())` after
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

## One seed gives different initial weights on different GPU models

`TrainingJob` seeds the CPU and CUDA generators, then builds the model on
`self.device`, so `trunc_normal_` and `normal_` draw from the CUDA generator.
Its output for a seed is not the same across GPU models: PyTorch promises no
reproducibility across platforms, and how a random fill is spread over
thread blocks follows the device. Found when one leg (`seed=0`, fp32, no
dropout) ran on a T4 and then an A10 and the losses differed. The batches
were identical (NumPy on the CPU, seeded by `(seed, step)`), and the gap was
already 4.5e-3 at step 1 (7.01200 against 7.01645), where kernel-level
float differences would be about 1e-6. So the two attempts started from
different weights.

Read the step-1 loss to tell the two causes apart. A gap there means the
initial weights differ. A step 1 that matches and then drifts is cuBLAS
choosing different kernels per architecture, with a different summation
order; that is small in fp32 with TF32 off, and it never goes to zero.
Building on the CPU and then moving the model,
`model.build(torch.device("cpu")).to(device)`, makes the start the same on
every machine. It also changes the start of every fresh leg, compared with
a leg built on the GPU before.

## Interruption testing: patch a method, compare against a straight run

The idempotence of the loop was checked by patching `set_learning_rate` to
raise before a chosen step (any crash point, not only after a failsafe) and
`torch.save` to die while writing the file `worker.publishing` handed it, or
during a strip rewrite, then comparing the final checkpoint against an
uninterrupted run in a separate root. Scenarios worth keeping: crash after
every failsafe,
crash mid-interval, crash before the first failsafe, crash during strip,
stale `.tmp`, crash on the final write, rerun after completion, a multi-leg
chain with crashes in every leg, and random crash points. The runnable
version lives in the session scratchpad, not the repo; `test_pretrain.py`
keeps the cases that matter as unit tests.

## Artifact annotations must be real imports

Pydantic resolves every field's annotation in the defining module's globals:
a self-reference when the class is created, a class defined later in the
module when the class is first used. An import under `TYPE_CHECKING` leaves
the annotation a string nothing can resolve, so the class stays incompletely
defined and raises at its first instantiation. This is why a base artifact
cannot hold a `dataset` field typed with a family's class without importing
that family, and why `dataset` and `tokenizer` live on `Pretraining`, not on
`Training`. It is also why `Artifact.dependencies()`, which reads the
annotations, runs at first use rather than at class creation: in
`__pydantic_init_subclass__` a forward reference is still a `ForwardRef`.

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

## A log handler that calls Modal breaks Modal's own heartbeat

A handler's `emit` runs on the thread that logged, and the Modal client logs
from its own event loop: "Received a cancellation signal while processing
input" is a `logger.warning` inside the heartbeat coroutine. `BufferHandler`
once read `call_logs` the first time a channel appeared, and on a worker that
warning was the first `ambient` row. The blocking `Dict.get` from Modal's loop
raised `Deadlock detected: calling a sync function from the synchronizer
loop`, `emit` let it out, and it surfaced in the coroutine that called
`logger.warning` -- the log read "Heartbeat attempt failed" with that error,
in the middle of the heartbeat delivering a cancel.

So `emit` touches memory and nothing else, and every Modal call a handler
needs happens on the publisher's own thread. The same goes for any handler,
filter or formatter added later: a record from `modal.*` can arrive on
Modal's loop at any time, and a cancellation is when it does.

## A traceback keeps the volume's files mapped after the call

A memmap over a tokens.bin is a local in `PretrainJob.run` and in every
frame under it (`evaluate`, `get_train_batch`, `get_valid_batch`), and the
two batch functions also hold it in their closures. When one of those raises, the
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

`launcher.state.mount_lock` is what holds them apart, and everything in that
process which reloads the mount or opens a file on it takes it: `state()`,
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

## `modal.is_local()` is True in a notebook kernel on Modal

`modal.is_local()` is False only in the process running a Modal Function.
A JupyterLab kernel is a child process of the `jupyter` Function, so in a
lab notebook it says True, on Modal hardware with the volume mounted.
`lab.declare`, `create_experiment` and `load_experiment` used it to decide
whether `root` was the mounted volume, so from the lab they never reloaded
before inspecting, never committed after writing, and never put their
`refreshes` message. The commit went unnoticed because every mount has
background commits; the missing message did not, because the dashboard
never showed a created experiment.

`lab._on_volume` reads `MODAL_IS_REMOTE` instead: Modal sets it in the
container's environment, and every process there inherits it, kernels
included (`lab_server` starts JupyterLab with a plain `Popen`). Anything
else that asks "am I on Modal" from code a notebook runs needs the same
answer.

## `modal.is_local()` is True in a notebook kernel on Modal

It is False only in the process running a Modal Function, and a JupyterLab
kernel is a child process of the `jupyter` Function. `lab.declare` and
`create_experiment` once gated their reload, commit and `refreshes` message
on it, so from the lab none of the three happened; background commits hid
the missing commit, and a created experiment that never reached the
dashboard showed the missing message. Both only ever run in a Modal
container, so they ask nothing about where they run: `root == STORAGE` is
the whole test, and a test's temporary root is what keeps it off the volume
and the queue.

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

## A cancel is a request, and it arrives late

`FunctionCall.cancel()` returns at once, but the container only learns about
it on its own `ContainerHeartbeat` to Modal, which is also where an
`InputCancellation` is raised into the call. How late that is has no bound
worth relying on -- a slow or failed heartbeat postpones it, and anything the
container does on Modal's own event loop can cause one -- so nothing here is
built on a cancel being timely. Cancelling
`runs/RUN2/pretraining/1000-2000-aa9fdbfce0` took 13.3 seconds to land: the
job trained steps 1900 and 2000, wrote `model.pt` and logged "training
complete" in the meantime, and only then was interrupted. The leg was
complete on the volume and its call was recorded as `failed`.

Three things were wrong and each is fixed where it belongs:

- A grant that was simply gone read as `UNKNOWN` -- silence -- so the worker
  slept through five backoffs and reported "ownership never confirmed". A read
  that succeeded and found nothing is an answer: no one holds this artifact,
  and we are no one. This is what makes dropping the lease a usable stop
  signal, and a cancel is nothing more than that: `cancel_call` removes the
  grant, then asks Modal to stop the call.
- The one confirm that could have stopped it ran *after* `model.pt` was
  written. Every file a job writes now goes through `worker.publishing`, which
  opens the temporary file, confirms the lease and renames -- so the confirm is
  between the bytes and the publication of every file, and no job names a
  `.tmp` or calls `replace` itself.
- Nothing stopped a second call being granted the path while the cancelled
  container was still writing, and both would have written the same
  `tokens.bin.tmp`. The temporary file carries the call id, so overlapping
  calls never mix bytes and only the lease holder renames. This is the cost of
  releasing ownership at the moment of the request rather than holding it
  until the call is known to be over: two containers can run, but only one can
  publish.

The thing to know before moving any of those confirms: **`volume.commit()` is
not what publishes a file.** `_volume_to_mount_proto` hard-codes
`allow_background_commits=True` on every mount (modal 1.5.4, no way to turn it
off), and the container's own exit path commits every such volume on the way
out. So bytes on the mount reach the volume whether or not the call lives to
call `commit`, and "write the file, check the lease, then commit only if we
still hold it" would publish exactly the same artifact, just later and with
nothing in the record. What *is* ours to withhold is the rename: a `.tmp`
satisfies no completion, so the confirm goes between writing the bytes and
renaming them into place, and that pairing lives in one context manager on
the worker rather than in each job.

Even that cannot be airtight: a lease dropped between the confirm and the
rename still lets that rename through, so a cancel can still lose by the width
of one rename and leave a complete artifact behind. That is accepted. A
cancellation is recorded as `lease lost`, which is what it is here -- the
launcher took the artifact away.

## Profiler overhead reuses the operator's correlation id

A `torch.profiler` overhead row -- "Activity Buffer Request" -- carries the
same correlation id (`FunctionEvent.id`) as the operator it interrupted and
sits *after* it in `prof.events()`, so a map from correlation id to row built
without a filter keeps the overhead row, and every kernel that operator
launched is attributed to profiler overhead instead. Correlation ids also
come from two counters: an op's from PyTorch's own, a launch call's and a GPU
activity's from CUPTI, so the same number can name an op and an unrelated
launch. The device side mixes them too: a `gpu_user_annotation` (the GPU
span of a `record_function` scope) carries its scope's PyTorch id, and in one
capture `Optimizer.step`'s was 1912, which was also a launch call's. Every
`correlation_id` lookup in `benchmarking/profile.ipynb` is therefore
restricted by `activity_type` (launch calls only, or `kernel`/`gpu_memcpy`/
`gpu_memset` only, never "any CUDA row"); dropping that filter silently
reparents kernels.

Found from a stale `benchmarking/events.json`, where kernel 62 named
`Activity Buffer Request` as its launcher while no kernel named the
`aten::addmm` one row above it. The ancestry still reached `addmm`,
because the overhead row's parent *is* the operator -- which is what makes
this quiet: the tree looks plausible and only the launch edge is wrong.

So: never test graph code against a capture checked in next to it. A
saved capture freezes whatever processing produced it the day it was written
and will disagree with the current code without saying so. Regenerate it
by rerunning the capture in `benchmarking/profile.ipynb` and treat any
on-disk capture as scratch.

## `prof.events()` deletes ops, and their kernels still name them

`EventList._build_tree` runs `_remove_dup_nodes`, which deletes an op that
is the only child of an op with the same name and moves its children to
that parent. Scalar overloads do exactly this: `x * 0.5` records `aten::mul`
inside `aten::mul`. The kernels launched there keep the deleted op's id in
`linked_correlation_id`, so looking that id up finds nothing. Found as 33,
then 36, "unlinked" kernels per step: every norm's `PowBackward0` and
`MeanBackward1` kernels and three from the embedding backward, about 2% of
GPU time but always the same modules.

Link a kernel through its launch call instead (same CUPTI correlation id), whose
`cpu_parent` is the surviving op; `linked_correlation_id` is then a check
that must match unless the op it names is gone. Never treat a missing
linked op as "not from PyTorch".

## pydantic is strict only where it is told to be, and silent where it is not

Artifacts and param configs are pydantic models under one strict config, and
each of these passed a test until a probe hit it:

- **A config does not reach nested models.** Each model validates with its
  own config, so `Artifact` and `ParamConfig` both set `STRICT`; a base that
  forgot it would validate its subclasses laxly, coercing `"1"` to `1`.
- **pydantic serializes a value by its field's annotation, not its class.**
  A `Transformer` in a field annotated `Model` dumps as `Model`'s fields only,
  with no warning. `parameters()` dumps with `serialize_as_any=True`, and
  dependencies are never dumped by pydantic at all.
- **Under `strict=True`, any `before` or `wrap` validator breaks JSON-mode
  decoding of tuples.** After such a validator, pydantic continues with Python
  objects and refuses a JSON array for a `tuple` field, even inside
  `model_validate_json`. `Artifact` has one (the set conversion), so decoding
  runs in Python mode, turning arrays into tuples first.
- **Strict mode accepts only a frozenset for a `frozenset` field.** A tuple or
  a list is refused, which is why `Artifact` converts set-valued dependencies
  before validation.
- **`-0.0` survives validation**, and `-0.0 == 0.0` while JSON writes them
  differently, so identity digests would differ; `canonical()` parses floats
  with `float(s) + 0.0`, for digests and manifests alike.
- **`model_copy(update=...)` neither validates nor clears cached properties.**
  It accepts `True` into an `int` field, and a copied leg keeps its old cached
  `lineage_hash`, so its old path. A definition change is a new construction;
  `model_copy` replaces recorded fields only.
- **pydantic has no per-field `compare=False`**, so `Artifact` overrides
  `__eq__` and `__hash__` to leave the recorded fields out; subclasses inherit
  both, since pydantic installs its own `__hash__` only where no base defines
  one.

A test that the codec round-trips is not enough; test that it refuses, per
case (test/test_codec.py), and that a subclass comes back as its own class at
every depth.
