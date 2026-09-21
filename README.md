# launchpad

Artifacts are folders on a Modal Volume, each owning a `manifest.json` that
names what it is and what it's built from; state and orchestration are two
Modal Dicts and one web container. The full model -- what an artifact and a
job *are*, how declaration/resolution/launching fit together -- is
[artifacts/core/spec.md](artifacts/core/spec.md); this file is the map of
where things live and how traffic flows between them.

## Where things live

- **Volume (`trainvols`)**: one folder per artifact, holding its own
  `manifest.json`, whatever files that artifact type declares, and a
  `logs/` folder with one `{call_id}.jsonl` per call ever launched for
  it -- appended by leasebook out of the Dicts on a clock, one JSON row per
  log record (see [docs/LOGGING.md](docs/LOGGING.md)). At the root,
  `call_history.json`: every grant ever made, per artifact_path, written on
  the same clock.
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
  rows, read when the dashboard starts and extended as it appends).
- **Dict `launchpad-train`**: two entries per **artifact_path** for a leg's
  `train.jsonl`, `:live` (the worker's) and `:volume` (read at startup), one
  row per training step per attempt.

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
  - **One thread keeps this container's picture of the volume fresh**, on a
    `STATE_REFRESH_SECONDS` clock: `state()`, and nothing else -- one
    `volume.reload()`, one `leases`/`beats` Dict snapshot (no mount needed),
    one glob of the manifests off `leasebook`'s own **local mount**, one flat
    map of every artifact on the volume. The whole state is computed there,
    in that thread. It writes nothing, syncs
    nothing, and no request waits on it.
  - `/state`: hands back what that thread last computed -- a dict lookup, not
    a reload. Never slower than that, never fresher than the last pass.
    Published by a single key assignment, which the GIL makes atomic, so
    there is nothing for a lock to protect.
  - `/manifest/{artifact_path:path}`: reads one manifest, in the request. The
    last thing here that touches the mount outside the refresh thread, and
    the only one left small enough not to matter -- a reload replaces the
    tree rather than refreshing it, so anything that *walks* the volume in a
    request is asking to watch a file it just listed disappear
    ([docs/QUEUES.md](docs/QUEUES.md) §3.3). Log files are why that rule
    exists; the dashboard reads them, and `train.jsonl`, once at startup
    (`load_snapshot_from_volume`, `sync_volume_train`), and appends to
    them only from the refresh thread itself, between two of its reloads
    (`save_snapshot_to_volume`, every `PERSIST_LOGS_EVERY` seconds).
  - `/logs/artifact/{artifact_path:path}`: every call ever granted for the
    artifact (`call_history`), each with all three channels of its log out
    of the `call_logs` Dict (`{call_id}:launcher`, `:container`, `:volume`),
    and both copies of the artifact's step log out of the `train` Dict.
    Dict reads, never a file on the mount; the page dedupes.
  - `/launch/{artifact_path:path}`: checks the lease, makes a `.remote()`
    call to `declared_artifact` (below) to confirm the artifact is
    declared and ready, then spawns `run_job` and writes the new grant --
    returns immediately, it doesn't wait for the job to finish.
  - `/cancel/{artifact_path:path}`: releases the artifact's lease and cancels
    the call holding it -- the lease first, so a worker between checkpoints
    discovers it lost the artifact even if the cancel itself never lands.
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
two Dicts and the volume it reads and writes along the way. Running the
whole time alongside it, on its own clock: `leasebook`, polling `/state*`
for a picture of the very same book through its own separate local mount.

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

    Note over L,J: meanwhile, independently -- leasebook's one thread
    loop every STATE_REFRESH_SECONDS
        L-->>Le: GET (batch)
        L-->>B: GET beats, call_history (batch)
        L->>V: reload() (leasebook's own local mount)
        Note over L,V: every PERSIST_LOGS_EVERY: append new launcher + container rows to logs/{call_id}.jsonl, write call_history.json, commit()
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
Dict, on leasebook's clock. `leasebook` keeps a separate copy of its
own, refreshed by its own `reload()` whenever something polls `/state*` --
so the dashboard's view is never more than one poll behind whatever's been
committed, and never less than one commit behind whatever the job is
actually doing. Neither container's disk is the other's cache; the volume
is the only thing both of them agree on.
