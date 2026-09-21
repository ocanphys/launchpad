# launchpad

Artifacts are folders on a Modal Volume, each owning a `manifest.json` that
names what it is and what it's built from; state and orchestration are four
Modal Dicts, one web container and one scheduled function. The full model -- what an artifact and a
job *are*, how declaration/resolution/launching fit together -- is
[artifacts/core/spec.md](artifacts/core/spec.md); this file is the map of
where things live and how traffic flows between them.

## Where things live

- **Volume (`trainvols`)**: one folder per artifact, holding its own
  `manifest.json`, whatever files that artifact type declares, and a
  `logs/` folder with one `{call_id}.jsonl` per call ever launched for
  it -- appended by `persist_logs` out of the Dicts on its schedule, one
  JSON row per log record (see [docs/LOGGING.md](docs/LOGGING.md)). At
  the root, `call_history.json`: every grant ever made, per artifact_path,
  written on the same pass. A leg's `train.jsonl` is the worker's own
  file, one row per step, visible once the worker commits under its lease.
  - **Shared roots** -- `sources/`, `tokenizers/`, `tokenized/`, `datasets/`,
    `mappeddatasets/` -- hold artifacts with no `run_id` in their
    parameters: reused across runs rather than rebuilt per run.
    `tokenized/{tokenizer-uid}/{source-uid}/` holds a `TokenizedSource`
    (`artifacts/tokenized/`): one source run through one tokenizer, as
    `tokens.bin`, shared by every dataset that names that pair. `datasets/`
    (`DataSet`, which copies those bins into its own `train.bin`/`valid.bin`)
    and `mappeddatasets/` (`MappedDataSet`, which owns no bytes of its own and
    reads straight out of the `tokens.bin` files) are separate sibling
    packages (`artifacts/dataset/`, `artifacts/mappeddataset/`) and folders --
    different kinds of artifact, not variants of one "dataset" concept.
  - **`runs/{run_id}/`** holds that run's own run-scoped artifacts --
    today, just `pretraining/`. Everything a run depends on that *isn't*
    run-scoped (its tokenizer, its dataset, the sources behind them) lives
    under a shared root instead, found by walking the dependency tree
    embedded in the run's own manifest, not nested under the run's folder.
- **Dict `launchpad-leases`**: one entry per **artifact_path** (not per
  run -- see artifacts/core/spec.md §7), naming the call_id currently
  holding that artifact.
- **Dict `launchpad-beats`**: one entry per call_id, the timestamp of its
  last heartbeat -- how a reader tells a live call from a dead one.
- **Dict `launchpad-call-history`**: one entry per **artifact_path**, the
  list of every grant the launcher ever made for it -- which calls belong
  to an artifact, said once, by the thing that made them.
- **Dict `launchpad-call-logs`**: three entries per call_id, one writer
  each: `{call_id}:launcher` (the launcher's own rows: the grant, a
  cancel), `{call_id}:container` (every log row the call has produced so
  far, republished whole on each beat) and `{call_id}:volume` (the file's
  rows, read when the dashboard starts and on every persist pass).

## Dashboard

The web app (`leasebook`) serves a small dashboard: one table, one row per
artifact on the volume, plus a drill-down page per artifact. No build step,
no framework. Full writeup: [docs/UI.md](docs/UI.md).

## The lab

A `jupyter` function serves JupyterLab with the volume mounted, in its own
container (own image, own idle-scaledown) separate from `leasebook`. The
dashboard's header links to `/lab`, which redirects there with the auth
token already attached. `lab_image` carries every dependency
`pyproject.toml` declares, not just what today's registered jobs need, so a
notebook running there can import and run anything in the repo (`torch`
included), not only what `worker_image` is trimmed to.

Requires a one-time secret, per Modal workspace/environment:
```
modal secret create launchpad-lab JUPYTER_TOKEN=$(openssl rand -hex 24)
```

The API those notebooks import is [lab.py](lab.py), and it has one entry
point, declaration. Everything else stays on the artifact classes, reading
the volume's mount (`config.STORAGE`) unless handed another root:

```python
import lab
report = lab.declare(pretraining)                 # resolve + check, no writes
report = lab.declare(pretraining, commit=True)    # ...and publish missing manifests

Artifact.load("tokenizers/bpe-3.0k-e4649eb4ff")   # what's declared there, unbound
Tokenizer(vocab_size=3000, ...).bind()            # the built object, ready to use
```

There is nothing to initialize and no environment detection. The mount
exists only inside a container, so a laptop works against a folder of its
own by passing `root=` -- the same mechanism a test uses with a `tmp_path`.

Nothing here forces a `volume.commit()`: every Volume mount sets
`allow_background_commits=True`, so the platform flushes the lab's writes on
its own, and JupyterLab's own autosave does the rest.

## Read/write traffic

- **`leasebook`** (the web app) is pinned to a single container
  (`max_containers=1`) -- it's both the dashboard and the launcher, so
  there's one copy of the traffic pattern below, not several racing each
  other.
  - **One `state` map in memory, computed at startup and on `POST
    /refresh`.** `state()` is one `volume.reload()`, one `leases`/`beats`
    Dict snapshot and one glob of the manifests off the **local mount**:
    what launching and the table need, and nothing more. That reload is
    the only one this container ever does, so a file read off the mount
    afterwards is that reload's image of it. No clock. Modal hands
    `leasebook` one input at a time (no `@modal.concurrent`), so a refresh
    never reloads while another request holds a file open. It writes
    nothing to the volume; at startup it runs `load_snapshot_from_volume`
    once so the Dicts and the files agree after any gap.
  - `/refresh`: what the page's refresh button POSTs. Reloads and
    recomputes the map.
  - `/state`: that map, one flat map of every artifact on the volume. An
    artifact's own page reads its type, parameters and dependencies out of
    this same map.
  - `/artifact/{artifact_path:path}`: the artifact as this container's
    disk holds it, as the last refresh left it -- its `manifest.json`
    minus the dependency manifests nested in it, and its `train.jsonl` --
    read through `open_json`/`open_jsonl` so every descriptor is closed
    before the request returns.
  - `/logs/{artifact_path:path}`: live -- every call ever granted for the
    artifact (`call_history`), each with all three channels of its log
    read from the `call_logs` Dict now (`{call_id}:launcher`, `:container`,
    `:volume`; the page dedupes). Dict reads only, never the mount, which
    is what lets an open artifact page poll it.
  - `/launch/{artifact_path:path}`: checks the lease, makes a `.remote()`
    call to `declared_artifact` (below) to confirm the artifact is
    declared and ready, then spawns `run_job` and writes the new grant --
    returns immediately, it doesn't wait for the job to finish.
  - `/cancel/{artifact_path:path}`: releases the artifact's lease and cancels
    the call holding it -- the lease first, so a worker between checkpoints
    discovers it lost the artifact even if the cancel itself never lands.
- **`persist_logs`** is a scheduled function (`modal.Period`, every
  `PERSIST_LOGS_EVERY` seconds) in its own container with its own mount:
  the only writer of log files and `call_history.json`. Each pass is
  stateless -- reload, rebuild the row-count cursor from the files
  (`load_snapshot_from_volume`), append every call's new Dict rows
  (`save_snapshot_to_volume`), commit. Schedules only fire on a deployed
  app (`modal deploy`), not under `modal serve`.
- **`declared_artifact`** and **`run_job`** are separate Modal functions,
  each with their own container and their own mount of the volume --
  `declared_artifact` computes `state()` and picks one entry; `run_job` writes
  locally and only publishes those writes with an explicit
  `volume.commit()` right before exiting. Nothing either writes is visible
  to any other reader, mounted or not, until that commit lands.
- **The local CLI** (`launch_job`, a `modal.local_entrypoint`) has no mount
  at all and never touches the volume directly -- it reads/writes the
  `leases`/`beats` Dicts directly (reachable from anywhere), and reaches
  everything volume-shaped through a `.remote()`/`.spawn()` call into a
  container that has one (`declared_artifact`, `run_job`). `lab.py`
  (below) is the exception: it's meant to run *inside* a container that
  already has the volume mounted (a notebook server), so its functions
  touch `Path(STORAGE)` directly.

So the volume is always the actual source of truth; every reader --
mounted or not -- is working from some snapshot of it: a local mount
refreshed on `reload()`, or a `.remote()` call into a container that just
took one.

## Jobs

The full model is [artifacts/core/spec.md](artifacts/core/spec.md); this is
the shape of it. Each artifact family (`artifacts/sources/`,
`artifacts/tokenizers/bpe/`, `artifacts/tokenized/`, `artifacts/dataset/`,
`artifacts/mappeddataset/`, `artifacts/stages/pretraining/`, ...) pairs an
`Artifact` subclass (parameters, where it lives, what files it comprises)
with exactly one `Job` subclass that produces it -- no `job_uid`, no per-run
config file. A `Job` subclass declares what it produces with one class
annotation (`artifact: Tokenizer`), and defining the class registers it as
that type's producer automatically, at import time
(`Job.__init_subclass__` populating `artifacts.core.job.REGISTRY`). Its inputs are derived,
never declared separately: `artifact.deps()` -- the artifact-valued
parameters of the one thing it produces -- so a job's dependency list can
never drift out of sync with what its own artifact actually names.

Declaring (writing `manifest.json` files ahead of the work, for a whole
dependency tree at once) and launching (granting a lease and spawning the
job that fills one manifest in) are separate steps -- see
`artifacts.core.resolve` (`resolve`/`declare`) and `main.attempt_launch`.
There's no per-job `resources`
block in a config file; a job's resource ask lives on its own artifact
(`allocated_resources: Resources`, e.g. `Resources(gpu_type="A100")`),
turned into `Function.with_options(...)` kwargs by `grant_and_spawn` right
before spawning -- an artifact that declares none runs on `run_job`'s own
default pool.

## Scheduling, deferred

There is no queue in the tree. Launching is per artifact and by hand:
`/launch` starts one job, `/cancel` stops one. A scheduler that takes a whole
plan and works through it was built, run against real containers, and taken
back out -- the launcher underneath it wants simplifying first.

What it looked like, every way it broke, and what to do differently:
[docs/QUEUES.md](docs/QUEUES.md). The code itself is in `stash@{0}`.

## One job, traced

One artifact's job, spawned, run to completion, and exited -- against the
Dicts and the volume it reads and writes along the way. Alongside it,
`leasebook` answering each `/state` request with a fresh picture of the very
same book through its own separate local mount, and `persist_logs` filing
the Dicts' rows on its schedule.

```mermaid
sequenceDiagram
    participant L as Launcher (cli / web)
    participant Le as Leases (Dict)
    participant B as Beats + call_logs + call_history (Dicts)
    participant D as declared_artifact (container)
    participant V as Volume (source of truth)
    participant J as Job container (run_job)

    L->>Le: GET lease -- already active?
    L->>D: remote(): load manifest, check status
    D->>V: reload(); read manifest.json + files
    D-->>L: (artifact, ready?)
    L->>Le: DELETE stale grant (if any)
    L->>J: spawn(artifact_path)
    L->>Le: PUT new grant
    L->>B: APPEND grant to call_history[artifact_path]; APPEND "granted" to {call_id}:launcher

    activate J
    J->>V: reload()
    J->>Le: confirm ("boot")
    J->>Le: confirm ("pre run")
    J->>J: run() -- resolve producing Job, write files; every log record lands in the buffer
    J-->>B: PUT heartbeat; PUT {call_id}:container (daemon thread, every beat, all rows so far)
    J->>Le: confirm ("pre vol commit")
    J->>Le: confirm ("commit")
    J->>V: commit()
    deactivate J

    Note over L,J: meanwhile, independently
    loop every POST /refresh
        L->>V: reload() (leasebook's own local mount)
        L-->>Le: GET (batch)
        L-->>B: GET beats (batch)
    end
    loop persist_logs, every PERSIST_LOGS_EVERY
        Note over B,V: reload(); append new launcher + container rows to logs/{call_id}.jsonl, write call_history.json, commit()
    end
```

Four confirms, not two: `initialize_worker` (`runtime.py`) brackets the
whole call with "boot" (before anything runs) and "commit" (right after,
before `volume.commit()`); `run_job` itself adds "pre run" (before
resolving and calling the job) and "pre vol commit" (right after) -- each
one a fresh re-read of the grant, so whichever of two racing launches lost
discovers it as early as the next checkpoint, not only at the very end.

A job's writes are private until `commit()` publishes them to the volume --
its output files live only on that one container's own disk until then; its
log never touches that disk at all, and reaches the volume through the
Dict, on `persist_logs`'s schedule. `leasebook` keeps a separate copy of
its own, refreshed by its own `reload()` when the page asks -- so the
dashboard's view is exactly as old as its last refresh, and never less
than one commit behind whatever the job is actually doing. Neither
container's disk is the other's cache; the volume is the only thing both
of them agree on.
