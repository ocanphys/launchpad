# Logging conventions

## Mechanism

Stdlib `logging`. A job receives its logger as `worker.log` (see
`runtime.Worker`) and should use that, not `print()`: only what goes through
`logging` is filed under the call.

`setup_logging` ([system/logs.py](../system/logs.py)) points the root logger
at stdout once per container, which is what `modal app logs <app>` reads
back. A call's own log is **three channels in the `call_logs` Dict, one
writer each**, and one file `persist_logs` writes them to:

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
  read at leasebook startup and on every persist pass.

The same Dict also has a **`launcher`** key for the leasebook container's
own log, kept the way a worker keeps its `:container` channel:
`start_launcher_logging` attaches a `BufferHandler` to the root logger,
with no call-id filter so startup, every request and every thread share
one stream, and a thread republishes the whole list every
`HEARTBEAT_SECONDS`, once more when the ASGI app shuts down (or when
building it raised), so the rows logged on the way out land. A publish
that fails is a warning row the next one carries. What lands there:
the container starting and stopping, the startup sync, every refresh,
every step of a launch (`attempt_launch`: requested, refused and why,
reading the volume, spawning with which resources, the grant) and
of a cancel, and any request that raised, with its traceback (the
middleware in `leasebook`). `launcher_log` writes a call's `:launcher` row
and logs it here too, so the grant and the cancel read in both places.

The launcher's log is filed the same way as a call's: `persist_logs`
appends the `launcher` rows past its cursor to `logs/launcher.jsonl` at
the volume root (under `logs/` like a call's file, one level up) and
`launcher:volume` is that file's rows, read at leasebook startup and on
every pass. For the cursor to hold across restarts, the live list has to
keep extending the file, so a new container starts its list from the
longer of what the last one published and what the file holds: nothing
a container logged in its last minute is lost to its replacement, and a
wiped Dict comes back from the file at the next restart. It has no call
id, grant or heartbeat.

A row is `{"ts": <epoch seconds>, "level": "INFO", "logger": "job", "msg":
"..."}`, a traceback folded into `msg`; a row in the file also carries
`"source": "launcher" | "container"`.

**Which calls belong to an artifact is the `call_history` Dict's to say**:
`attempt_launch` appends the grant to `call_history[artifact_path]` right
after `spawn`, and `artifact_calls` joins that list with `beats` on every
request for the artifact's page. Nothing is inferred from a beat or a
filename.

**`persist_logs` is the only thing that writes a log file**, a scheduled
function in its own container, every `PERSIST_LOGS_EVERY` seconds. Each
pass is stateless: `load_snapshot_from_volume` walks `**/logs/*.jsonl` into
`call_logs["{call_id}:volume"]` (the root's `logs/launcher.jsonl` into
`launcher:volume`), merges `call_history.json` at the volume root into the
`call_history` Dict (a union per artifact, so a wiped Dict comes back
from the file; a log file no grant names gets a grant with no
`granted_ts`), and hands back how many rows of each channel every file
already holds. Then `save_snapshot_to_volume`, for every call in
`call_history` and for the launcher, appends the rows of the channels
past that count to the file and to the `:volume` channel, writes
`call_history` to `call_history.json` and commits. `leasebook` runs the
same `load_snapshot_from_volume` once at startup, so a dashboard that comes
back after any gap agrees with the files.

`/logs/<path>` hands back all three channels per call straight from the
Dict, and an open artifact page polls it; the page unions them, a row
counted once, into one stream
across calls, each line showing its call id and which side wrote it
(`launcher`, `container`, or the file's own `source` for a row read off
the volume), filtered by level (`artifactview.js`). The volume copy trails
the other two by a pass; the union is what is complete.

`/launcher-logs` hands back the `launcher` and `launcher:volume` channels
straight from the Dict, and the dashboard's launcher panel polls it every
two seconds, unioning the two like the artifact page does its three: the
same renderer (`logview.js`), newest first,
filtered by level, one line per row (time, level, message; the logger's
name on hover), a traceback keeping its line breaks. Neither poll reloads
the volume or recomputes the map.

## The step log

A leg's `train.jsonl` (one row per training step, tagged with the attempt
that took it, written by `artifacts/core/SGD/steplog.py`) is the worker's
own file and nothing else: it reaches the volume when the worker commits
under its lease, and `/artifact/<path>` reads it off leasebook's mount as
the last refresh reloaded it. No attempt's rows are streamed before that
commit. The page draws every row, one uPlot line per attempt.

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
- The `jupyter` lab's output reaches only `modal app logs`.
- The launcher's live list is the whole of `logs/launcher.jsonl` plus what
  came after, republished every second; nothing trims it.
- A container killed hard (out of memory, evicted) or one that never starts
  (image build failure, no GPU to be had) is listed with what it published
  up to its last beat, if any, after the launcher's rows; Modal's own line
  saying why stays in Modal.
- The launcher's channel is a read-modify-write on one key, and the one
  leasebook container is its only writer.
