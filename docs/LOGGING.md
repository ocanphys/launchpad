# Logging conventions

## Mechanism

Stdlib `logging`, wired up in [logs.py](logs.py). Every Modal call gets its
own logger, named `call.{call_id}` (`logs.py`'s `logger_name`), so a stale
handler left behind by an earlier call in the same container can never write
into the next call's file. A job receives its logger as `worker.log` (see
`runtime.Worker`) and should use that, not `print()` -- `print()` and any
third-party library logger (Modal, wandb, etc.) are never captured; see
`logs.py`'s module docstring for why.

Each line lands in `{artifact_path}/logs/{call_id}/job.log` as
`<iso-timestamp> <level> <message>`, UTC, one line per record
(`logs.py`'s `CallFormatter`). The dashboard reads these back with
`read_call_logs`/`read_log` and renders them in [web/logview.js](web/logview.js).

Handlers are attached at `DEBUG`, so nothing is dropped at write time --
filtering by level is a read-time/UI concern, not a write-time one. The log
viewer's level checkboxes reflect this: every level present in a view gets a
checkbox, and DEBUG starts unchecked (see below) but is never unavailable.

## Levels

**DEBUG** -- internal/protocol chatter only useful when actively debugging:
lease confirms/retries, per-step or per-batch training detail, retry
attempts, cache hits, anything high-frequency.
Example: `lease_protocol.py`'s `Lease.confirm` logs every successful confirm
and every indeterminate retry at DEBUG.

**INFO** -- one line per meaningful lifecycle event, not per iteration: job
started, job milestones (e.g. per dataset subset), checkpoint progress (e.g.
every Nth training step, not every step), job completed.
Example: `artifacts/job.py`'s jobs narrate their own start/progress/done at
INFO; `runtime.py` logs `"done"` once a call's writes are safely committed.

**WARNING** -- recoverable/retryable problems that don't lose work:
heartbeat write failure, a lease indeterminate-retry that eventually
succeeds.
Example: `runtime.py`'s heartbeat thread logs a failed `beats.put` at
WARNING and keeps beating -- the next attempt may well succeed.

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

## Dashboard defaults

The log viewer ([web/logview.js](web/logview.js), [web/app.js](web/app.js))
hides DEBUG by default on a fresh view -- it's protocol noise, not what
you open a log to look at -- but its checkbox still renders whenever DEBUG
entries exist, so turning it back on is one click. Toggling a level persists
across polls of the same view and resets only on a genuine scope/id change.
