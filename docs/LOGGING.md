# Logging conventions

## Mechanism

Stdlib `logging`. A job receives its logger as `worker.log` (see
`runtime.Worker`) and should use that, not `print()`: only what goes through
`logging` is filed under the call.

`setup_logging` ([system/logs.py](../system/logs.py)) points the root logger
at stdout once per container, which is what `modal app logs <app>` reads
back. A call's own log is **three channels in the `call_logs` Dict, one
writer each**, and one file leasebook writes them to:

- `{call_id}:container`, the worker's. `initialize_worker` adds a
  `BufferHandler` to the root logger for the length of the call, behind a
  `CallFilter` that passes a record only when
  `modal.current_function_call_id()` is this call's id, so a container that
  runs one call after another never files a row under the wrong one. On
  every heartbeat the worker publishes the whole list, beside the beat it
  puts in `beats[call_id]`; the heartbeat's last pass runs after the handler
  is removed, so the rows logged on the way out land too. The buffer and the
  heartbeat exist before the call touches the mount, so a call that dies on
  `volume.reload()` itself still beats and its error is its log. The worker
  never opens its log on the mount.
- `{call_id}:launcher`, the launcher's: one row per thing it did to the call
  (`launcher_log`: the grant, a cancel). Written at grant time, so a call
  whose container never runs still has a log saying it was made.
- `{call_id}:volume`, the file's: `{artifact_path}/logs/{call_id}.jsonl`,
  read once at leasebook startup and extended on every persist pass.

A row is `{"ts": <epoch seconds>, "level": "INFO", "logger": "job", "msg":
"..."}`, a traceback folded into `msg`; a row in the file also carries
`"source": "launcher" | "container"`.

**Which calls belong to an artifact is the `call_history` Dict's to say**:
`attempt_launch` appends the grant to `call_history[artifact_path]` right
after `spawn`, and `calls_by_artifact` joins that list with `beats` on the
refresh pass. Nothing is inferred from a beat or a filename.

**`leasebook` is the only thing that reads or writes a log file, on its one
thread.** At startup, before the clock exists, `load_snapshot_from_volume`
walks `**/logs/*.jsonl` into `call_logs["{call_id}:volume"]`, merges
`call_history.json` at the volume root into the `call_history` Dict (a union
per artifact, so a wiped Dict comes back from the file; a log file no grant
names gets a grant with no `granted_ts`), and hands back how many rows of
each channel every file already holds. Then every `PERSIST_LOGS_EVERY`
seconds the refresh thread, between two of its reloads, calls
`save_snapshot_to_volume`: for every call that beat since the previous pass
or that the launcher marked, the rows of the launcher and container channels
past that count are appended to the file and to the `:volume` channel, then
`call_history` is written to `call_history.json` and the volume committed.
Every file is closed before the pass returns, so the next reload finds none
open ([QUEUES.md](QUEUES.md) §3.3).

`/logs/artifact/<path>` hands back all three channels per call straight
from the Dict; the page unions them, a row counted once, into one stream
across calls, each line showing its call id and which side wrote it
(`launcher`, `container`, or the file's own `source` for a row read off
the volume), filtered by level (`artifactview.js`). The volume copy trails
the other two by a pass; the union is what is complete.

## The step log

A leg's `train.jsonl` (one row per training step, tagged with the attempt
that took it, written by `artifacts/core/SGD/steplog.py`) is the worker's
own file, with a two-copy path of its own through the `train` Dict, keyed
by artifact_path rather than call_id. `StepLog.flush` puts `train["{artifact_path}:live"]` itself, right
after appending to the file: every row this attempt has written, so the
Dict holds the running attempt and nothing is read back off the disk. A
put that fails is a warning in the call's log, not a dead training loop. Earlier attempts reach the Dict as
`train["{artifact_path}:volume"]` when `sync_volume_train` reads the files at
leasebook startup, so an attempt that ran and crashed since the last
redeploy is on the volume but not yet in the Dict. The page unions the two,
one row per (attempt, step), and buckets each attempt down to a point budget
before drawing. Nothing on the dashboard's clock ever opens the file.

## Levels

**DEBUG** -- internal/protocol chatter only useful when actively debugging:
lease confirms/retries, per-step or per-batch training detail, retry
attempts, cache hits, anything high-frequency.
Example: `lease_protocol.py`'s `Lease.confirm` logs every successful confirm
and every indeterminate retry at DEBUG.

**INFO** -- one line per meaningful lifecycle event, not per iteration: job
started, job milestones (e.g. per dataset subset), checkpoint progress (e.g.
every Nth training step, not every step), job completed.
Example: `artifacts/*/jobs.py`'s jobs narrate their own start/progress/done at
INFO; `runtime.py` logs `"done"` once a call's writes are safely committed.

**WARNING** -- recoverable/retryable problems that don't lose work:
heartbeat write failure, a lease indeterminate-retry that eventually
succeeds.
Example: `runtime.py`'s heartbeat thread logs a failed `beats.put` or
`call_logs.put` at WARNING and keeps beating -- the next attempt may well
succeed.

**ERROR** -- the call's work is lost or it crashed: lease lost, unhandled
exception under a held lease.
Example: `runtime.py` logs a `LeaseLost` at ERROR (the call's writes are
about to be discarded) and any other exception via `logger.exception` at
ERROR (with traceback) before re-raising.

**CRITICAL** -- reserved for infra-level failures that abort more than one
call (e.g. the artifact store itself unreachable). Unused today; don't
manufacture a distinction that doesn't exist yet just to exercise the level.

## Categories

There's no separate "training" vs "system" logger -- every call has exactly
one logger (`job`), and any log line from anything that call does (lease,
job, heartbeat) lands in that one file. If a category split is ever worth
adding, use `worker.log.getChild("name")` (e.g. `.getChild("lease")`) rather
than a new naming scheme: a child's records carry their own `logger` name in
the row and land in the same file for free.

## Not done

- Output that bypasses `logging` (a bare `print`, tqdm on stderr) reaches
  only Modal's own capture and expires there.
- A service container's output (`leasebook`, the `jupyter` lab) has no call
  id to be filed under, so it reaches `modal app logs` and expires there.
- A container killed hard (out of memory, evicted) or one that never starts
  (image build failure, no GPU to be had) is listed with what it published
  up to its last beat, if any, after the launcher's rows; Modal's own line
  saying why stays in Modal.
- The launcher's channel is a read-modify-write on one key. Its writers
  are the one leasebook container and `launch_job` at a keyboard; the two
  are not guarded against writing the same call at once.
