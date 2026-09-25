# Logging conventions

## One shape

Every row any container files looks the same:

```json
{"ts": 1757260800.1, "call_id": "fc-01JQ8W", "source": "worker",
 "level": "INFO", "logger": "job", "msg": "..."}
```

a traceback folded into `msg`. The `call_logs` key it lives under is built
from two of its own fields plus where it is kept:

    {call_id}:{storage}:{source}

- **`storage`** is `livedict` for what a container has published to the Dict,
  `volume` for the rows its file holds. Same rows, one trailing the other by
  a persist pass; the page unions them and counts a row once.
- **`source`** is who wrote it: `worker` (the call's own account of itself),
  `launcher` (what the leasebook did to that call), `ambient` (what the
  container logged around it and no logger of ours stamped -- a library's
  warning).
- **`call_id`** is the call the row is about. The leasebook's own rows use the
  call id `launcher`, so its log is a call's log in every other way: no
  `worker` source, no grant, no heartbeat, and its file is `logs/launcher.jsonl`
  at the volume root rather than in an artifact's folder.

## One way in

Two names, in [system/logs.py](../system/logs.py), and nothing else writes a
row:

- `start_logging(call_id)`, once per container: points logging at stdout
  (which is what `modal app logs <app>` reads back), attaches the
  `BufferHandler` that keeps every record in the channel its stamp names, and
  starts the thread that publishes each channel whole every
  `HEARTBEAT_SECONDS`. `call_id` is whose the container's *unstamped* records
  are -- `LAUNCHER` for the leasebook, its own call id for a worker -- and
  those are its `ambient` source.
- `call_logger(name, source, call_id)`, which stamps the rest. A job gets one
  as `worker.log` (`runtime.initialize_worker`, source `worker`); the
  leasebook binds the two it needs once, in main.py: `log_launcher` for its
  own doings and `log_call(call_id)` for a call it manages.

So which channel a row lands in is the logger it was written with, never a
second function next to the first. What the launcher does to a call it manages
-- the grant, the cancel, a stale lease dropped, a refusal naming the call
under way, a worker's start and end messages off the `refreshes` Queue -- is
stamped with that call's id, and **the handler files it under the container's
own call id as well**, so it reads on the artifact's page and in the
dashboard's panel alike. What the container does for itself -- starting,
stopping, the startup sync, every recompute and what prompted it, a refusal
with no call to name, any request that raised -- is stamped `launcher` and
reads in the panel only. Either way the launcher's own log is the whole
account of what that container did.

A worker stamps no call but its own, so the two channels collapse to one
there; the fan-out is the launcher's alone.

The stamp rides on the record, so any thread the call or the container starts
files under the right call without inheriting anything (LESSONS.md).

## The two storages

`{call_id}:livedict:{source}` is what a container has published: the whole
list, republished on each pass, so a page polling the Dict sees a call's log
grow while it runs. A container starting a channel continues the longer of
what the Dict holds and what the file's rows say, so a leasebook that restarts
mid-call extends the list rather than truncating it.

`{call_id}:volume:{source}` is the file's rows:
`{artifact_path}/logs/{call_id}.jsonl`, `logs/launcher.jsonl` for the
launcher. **`persist_snapshot` is the only thing that writes one**, on the
schedule `persist_logs` runs it (`PERSIST_LOGS_EVERY` seconds, its own
container). Each pass is stateless: `load_snapshot_from_volume` walks
`**/logs/*.jsonl`, publishes each file's rows as that call's `volume`
channels, and merges `call_history.json` at the volume root into the
`call_history` Dict (a union per artifact, so a wiped Dict comes back from the
file; a log file no grant names gets a grant with no `granted_ts`). Then the
pass appends every live row past its file's count to that file and to the
`volume` channel, writes `call_history` and commits. `leasebook` runs the same
`load_snapshot_from_volume` once at startup, so a dashboard that comes back
after any gap agrees with the files.

**Which calls belong to an artifact is the `call_history` Dict's to say**:
`attempt_launch` appends the grant right after `spawn`, and `artifact_calls`
joins that list with `beats` on every request for the artifact's page. Nothing
is inferred from a beat or a filename.

## What the page does with them

`/logs/<path>` hands back, per call, `livedict` and `volume` straight from the
Dict, and an open artifact page polls it; the page unions them, a row counted
once, into one stream across calls, newest first, each line showing its call id
and its `source`, filtered by level and by an **ambient** box
(`artifactview.js`). `/launcher-logs` is the same two for the call id
`launcher`, polled by the dashboard's panel and rendered by the same
`logview.js`, call id and source included -- the panel's rows are not all
about the container itself. Neither poll reloads the volume or recomputes the
map.

## The step log

A leg's `train.jsonl` (one row per training step, tagged with the attempt that
took it, written by `artifacts/core/SGD/steplog.py`) is the worker's own file
and nothing else: it reaches the volume when the worker commits under its
lease, and `/artifact/<path>` reads it off leasebook's mount as the last
reload left it. No attempt's rows are streamed before that commit. The page
draws every row, one uPlot line per metric per attempt, with the validation
loss -- null on the steps no validation ran on -- over the training loss on
one chart.

## Levels

**DEBUG** -- internal/protocol chatter only useful when actively debugging:
lease confirms/retries, per-step or per-batch training detail, retry
attempts, cache hits, anything high-frequency. What was *asked* of a call (the
spawn request and its resources, a cancel request) is DEBUG too; what became
of it is INFO, and the page hides DEBUG until a reader switches it on, so a
call reads as its outcomes first.
Example: `lease_protocol.py`'s `Lease.confirm` logs every successful confirm
and every indeterminate retry at DEBUG.

**INFO** -- one line per meaningful lifecycle event, not per iteration: job
started, job milestones (e.g. per dataset subset), checkpoint progress (e.g.
every Nth training step, not every step), job completed.
Example: `artifacts/*/jobs.py`'s jobs narrate their own start/progress/done at
INFO; `runtime.py` logs `"done"` once a call's writes are safely committed,
and a line naming the ending it is about to tell the launcher about.

**WARNING** -- recoverable/retryable problems that don't lose work:
heartbeat write failure, a lease indeterminate-retry that eventually
succeeds.
Example: `try_publish` logs a failed `beats.put` or `call_logs.put` at WARNING
and keeps going -- the next attempt may well succeed.

**ERROR** -- the call's work is lost or it crashed: lease lost, unhandled
exception under a held lease.
Example: `runtime.py` logs a `LeaseLost` at ERROR (the call's writes are
about to be discarded), in one line and without a traceback -- a cancel
arrives this way too, since `cancel_call` drops the lease -- and any other
exception via `logger.exception` at ERROR (with traceback) before re-raising.

**CRITICAL** -- reserved for infra-level failures that abort more than one
call (e.g. the artifact store itself unreachable). Unused today; don't
manufacture a distinction that doesn't exist yet just to exercise the level.

## Categories

There's no separate "training" vs "system" logger -- a call has one logger
(`job`), the leasebook has one (`leasebook`), and every line from anything
they do (lease, job, heartbeat, a route) lands in the one channel. If a
category split is ever worth adding, take another `call_logger("job.lease",
...)` rather than a new naming scheme: its records carry their own `logger`
name in the row and land in the same channel for free.

## Not done

- Output that bypasses `logging` (a bare `print`, tqdm on stderr) reaches only
  Modal's own capture and expires there. Teeing the streams into `ambient`
  would take a `dup2` onto a pipe and a reader thread; not attempted.
- A library that gives itself a handler and sets `propagate = False` never
  reaches ours, so its lines are Modal's only (LESSONS.md).
- The `jupyter` lab's output reaches only `modal app logs`.
- A live channel is the whole of its file plus what came after, republished
  every `HEARTBEAT_SECONDS`; nothing trims it.
- A container killed hard (out of memory, evicted) or one that never starts
  (image build failure, no GPU to be had) is listed with what it published up
  to its last beat, if any, beside the launcher's own rows about it; Modal's
  line saying why stays in Modal.
