# Logging conventions

## Mechanism

Stdlib `logging`, pointed at stdout by [system/logs.py](../system/logs.py)'s
`setup_logging` and nowhere else. Modal captures a container's stdout per
function call, so a worker's narration is filed under the call that produced
it and read back with `modal app logs <app>` or the Modal dashboard. A job
receives its logger as `worker.log` (see `runtime.Worker`) and should use that,
not `print()` -- the format below is what makes a line legible next to the
lease protocol's own.

Each line is `<iso-timestamp>.<ms>Z <LEVEL> <message>`, UTC wherever the
container happens to run.

**The worker copies its own output out of Modal, into two places.** Its
heartbeat thread (`runtime.initialize_worker`) subscribes to the call's own
log feed -- `FunctionCall.logs.fetch()` for whatever came before, then
`.stream()` for the rest -- and on every beat publishes the lines so far, the
whole list, as `call_logs[call_id]`, beside the beat it puts in
`beats[call_id]`. On the way out, the worker writes the same lines to
`{artifact_path}/call_functions/{call_id}.log` (`logs.archive_path`) and
commits. Asking Modal rather than installing a handler is what makes the copy
complete: stderr, tqdm and every library that never heard of `worker.log` are
in what Modal collected.

The Dict entry is live and expires with the Dict's own retention; the file is
for good. **The dashboard reads neither.** `leasebook` is the container that
reloads the volume on a clock, and a reload cannot run while the same
container has a file on it open (see [QUEUES.md](QUEUES.md) §3.3) -- that is
what took out the previous attempt, which had the dashboard doing the copying.
A worker has no such clock and may open files on the volume freely.

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
one logger (`call.{call_id}`), and any log line from anything that call
does (lease, job, heartbeat) lands in that one file. If a category split is
ever worth adding, use `worker.log.getChild("name")` (e.g.
`.getChild("lease")`) rather than a new naming scheme: children of a call's
logger land in the same file for free, so this costs nothing to read back
and nothing to add later.

## Not done

- `GET /logs` lists the call ids in `call_logs` and `GET /logs/{call_id}`
  returns one call's lines, straight from the Dict. No page renders them
  yet, and nothing reads the file.
- A service container's output (`leasebook`, the `jupyter` lab) has no call
  id to be filed under, so it reaches `modal app logs` and expires there.
- A container killed hard (out of memory, evicted) never runs its last beat,
  so its final `HEARTBEAT_SECONDS` of lines, and Modal's own line saying why
  it died, reach neither copy. They stay in Modal for a day.
