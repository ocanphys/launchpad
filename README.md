# launchpad

Artifacts are folders on a Modal Volume, each owning a `manifest.json` that
names what it is and what it's built from; state and orchestration are four
Modal Dicts, one Queue, one web container and one scheduled function. The full model -- what an artifact and a
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
- **Dict `launchpad-call-logs`**: one entry per call_id, storage and source,
  keyed `{call_id}:{livedict|volume}:{worker|launcher|ambient}` and one writer
  each. `livedict` is what a container has published, republished whole every
  `HEARTBEAT_SECONDS`; `volume` is the rows its file holds, read when the
  dashboard starts and on every persist pass. `worker` is the call's own
  account of itself, `launcher` what the leasebook did to it (the grant, a
  cancel, the start and end it was told about), `ambient` what a container
  logged around it and no logger of ours stamped. Every row carries its own
  `call_id` and `source`, so a channel is only where it is kept. The
  leasebook's own log is the same keys under the call id `launcher`, its file
  `logs/launcher.jsonl` at the volume root; a row it writes about a call is
  filed under both, so the call's page has it and the launcher's log stays the
  whole account of what that container did.
- **Queue `launchpad-refreshes`**: two messages per call,
  `{artifact_path, call_id, event}` -- `started`, from the heartbeat's first
  pass and right after the beat it published, and one naming how it ended
  (`done`, `failed`, `lease lost`), after its `volume.commit()` and its last
  beat. A call that goes on to fail its lease has already said it started;
  nothing is computed from a message, so it costs one refresh and no more.
  Put by the worker and taken by the one `leasebook` container, whose listener thread
  recomputes its state map when it finds one. The worker is the only writer,
  `leasebook` the only reader. Nothing is computed *from* a message: it says
  only that the volume and the Dicts are worth looking at again.

## Dashboard

The web app (`leasebook`) serves a small dashboard: one table, one row per
artifact on the volume, a live launcher-log side panel, and a drill-down page per artifact. No build step,
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
  - **One `state` map in memory, and the authority on the volume** --
    lagged, but complete, since an artifact's files never leave the volume
    once they land. `state()` is one `volume.reload()`, one `leases`/`beats`
    Dict snapshot and one glob of the manifests off the **local mount**:
    what the table needs, and nothing more. A file read off the mount
    afterwards is that reload's image of it; the only other reload is a
    `/launch`, which reads one artifact fresh. It writes
    nothing to the volume; at startup it runs `load_snapshot_from_volume`
    once so the Dicts and the files agree after any gap. Runs pinned to
    `REGION`, next to Modal's Dicts, so a Dict read is a short round trip.
  - **An entry has two halves.** The durable half (type, parameters, status,
    `done`, `ready`, `blocked_by`, `durable_progress`) can only change when
    the volume does, so it changes only on a recompute. The live half
    (`call_id`, `active`, `last_heartbeat`, `live_progress`, `verdict`) is
    `main.liveness` over a grant and a beat -- Dict reads, no mount -- so
    `/state` reads it again on every request for the calls the map found
    under way. That is how a running job's progress and heartbeat move on
    the page between recomputes.
  - **It is recomputed when something changed the volume or a lease**, never
    on a clock: a message on the `launchpad-refreshes` Queue (a worker's
    call started beating, or committed and exited, picked up by a listener
    thread blocked on that queue), an accepted `/launch` or `/cancel` in
    this same container, or a `POST /refresh`. Between the three, a row
    goes `runnable` -> `starting` -> `running` -> `done` on its own.
  - **Everything in this container that reloads the mount or holds a file
    on it open takes `mount_lock` first** -- `state()`, `/launch`'s read and
    `/artifact`. A reload replaces the mount's view of the volume and
    refuses to run while this process has a file under it open, so the
    listener thread's refresh and a request reading a manifest take turns
    (LESSONS.md).
  - `/refresh`: the manual reload, what the page's refresh button POSTs.
    For what nothing announces -- a manifest declared from the lab.
  - `/state`: that map, one flat map of every artifact on the volume,
    polled by the page every two seconds. An artifact's own page reads its
    type, parameters and dependencies out of this same map. Served from
    memory, plus two Dict reads for each call the map found under way (its
    grant and its beat, for the live half above) -- so polling it costs the
    volume nothing and an idle table costs nothing at all. A call whose beat
    says it has exited is left alone: its ending is already on the queue,
    and the recompute that reads its files is on the way.
  - `/artifact/{artifact_path:path}`: the artifact as this container's
    disk holds it, as the last recompute left it -- its `manifest.json`
    minus the dependency manifests nested in it, and its `train.jsonl` --
    read under `mount_lock` through `open_json`/`open_jsonl`, so every
    descriptor is closed before the request returns. The page fetches it
    only when the artifact's entry on the state map has changed.
  - `/logs/{artifact_path:path}`: live -- every call ever granted for the
    artifact (`call_history`), each with its log in both storages read from
    the `call_logs` Dict now (`livedict`, every source together, and
    `volume`; the page unions them and dedupes). Dict reads only, never the
    mount, which is what lets an open artifact page poll it.
  - `/launcher-logs`: the same two for the call id `launcher`, polled by the
    dashboard's panel every two seconds. Dict reads only.
  - `/launch/{artifact_path:path}`: checks the lease, reloads the volume
    and reads the artifact's manifest and its direct dependencies' files to
    confirm it is declared and ready, then spawns `run_job` and writes the
    new grant -- returns immediately, it doesn't wait for the job to finish.
    An accepted launch recomputes the map before answering.
  - `/cancel/{artifact_path:path}`: releases the artifact's lease and cancels
    the call holding it -- the lease first, so a worker between checkpoints
    discovers it lost the artifact even if the cancel itself never lands.
    What the call's output says once the request is sent is what the row
    logged under it says, so a request that arrived after the job finished
    doesn't read as one that stopped it. Recomputes the map, like a launch.
- **`persist_logs`** is a scheduled function (`modal.Period`, every
  `PERSIST_LOGS_EVERY` seconds) in its own container with its own mount:
  the only writer of log files and `call_history.json`. Each pass is
  stateless -- reload, then `persist_snapshot`: read the files back
  (`load_snapshot_from_volume`), append every call's rows past what they
  hold, commit. Schedules only fire on a deployed app (`modal deploy`), not
  under `modal serve`.
- **`run_job`** is its own Modal function, with its own container and its
  own mount of the volume: it writes locally and only publishes those
  writes with an explicit `volume.commit()` right before exiting. Nothing
  it writes is visible to any other reader, mounted or not, until that
  commit lands. The call puts one message on the `launchpad-refreshes` Queue
  on its heartbeat's first pass, once that pass has beaten, and one after
  that commit and its last beat, which it marks `exited` -- so every refresh
  the launcher makes for a call is one where what it is about to read is
  already true, and a finished call stops reading as live at once rather
  than a flatline later.
- **`lab.py`** runs *inside* a container that already has the volume
  mounted (the notebook server), so its functions touch `STORAGE`
  directly; `local.py` is how a laptop gets there, one `.remote()` call
  into a container that has the mount.

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
annotation (`artifact: Tokenizer`); the artifact names its job the other way
round, as a dotted string (`producer: ClassVar[str] =
"artifacts.tokenizers.bpe.jobs.TokenizerJob"`) that only `Artifact.job()`
ever imports, so resolving and inspecting never load a family's `jobs.py`.
Its inputs are derived, never declared separately: `artifact.deps()` -- the
artifact-valued parameters of the one thing it produces -- so a job's
dependency list can never drift out of sync with what its own artifact
actually names.

Declaring (writing `manifest.json` files ahead of the work, for a whole
dependency tree at once) and launching (granting a lease and spawning the
job that fills one manifest in) are separate steps -- see `lab.declare`
(over `artifacts.core.resolve`) and `main.attempt_launch`. There's no
per-job `resources` block in a config file; a job's resource ask lives on
its own artifact (`allocated_resources: Resources`, e.g.
`Resources(gpu_type="A100")`), turned into `Function.with_options(...)`
kwargs by `resource_options` right before `attempt_launch` spawns -- an
artifact that declares none runs on `run_job`'s own default pool.

## Scheduling, deferred

There is no queue in the tree. Launching is per artifact and by hand:
`/launch` starts one job, `/cancel` stops one. A scheduler that takes a whole
plan and works through it was built, run against real containers, and taken
back out -- the launcher underneath it wants simplifying first.

The code is in `git stash` ("queue but its got complicated").

## One job, traced

One artifact's job, spawned, run to completion, and exited -- against the
Dicts, the Queue and the volume it reads and writes along the way. Alongside
it, `leasebook` recomputing its picture of the very same book through its own
separate local mount when the job's exit tells it to, and `persist_logs`
filing the Dicts' rows on its schedule.

```mermaid
sequenceDiagram
    participant L as Launcher (leasebook)
    participant Le as Leases (Dict)
    participant B as Beats + call_logs + call_history (Dicts)
    participant Q as Refreshes (Queue)
    participant V as Volume (source of truth)
    participant J as Job container (run_job)

    L->>Le: GET lease -- already active?
    L->>V: reload(); read manifest.json + dependencies' files -- ready?
    L->>Le: DELETE stale grant (if any)
    L->>J: spawn(artifact_path)
    L->>Le: PUT new grant
    L->>B: APPEND grant to call_history[artifact_path]; APPEND "granted" to {call_id}:launcher
    L->>V: reload(); recompute the state map (the new grant is on it)

    activate J
    J->>V: reload()
    J->>Le: confirm ("boot")
    J->>B: PUT first heartbeat
    J->>Q: PUT {artifact_path, call_id, event: started}
    J->>Le: confirm ("pre run")
    J->>J: run() -- resolve producing Job, write files; every log record lands in the buffer
    J-->>B: PUT heartbeat (daemon thread); PUT {call_id}:livedict:{worker,ambient} (its own thread, all rows so far)
    J->>Le: confirm ("pre vol commit")
    J->>Le: confirm ("commit")
    J->>V: commit()
    J-->>B: PUT last heartbeat, marked exited
    J->>Q: PUT {artifact_path, call_id, event: done | failed | lease lost}
    deactivate J

    Note over Q,L: on each message, the listener thread's blocking read returns
    L->>V: reload(); recompute the state map (the beat, then the files, are on it)

    Note over L,J: meanwhile, independently
    loop persist_logs, every PERSIST_LOGS_EVERY
        Note over B,V: reload(); append every source's new rows to logs/{call_id}.jsonl, write call_history.json, commit()
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
its own, refreshed by its own `reload()` -- so the dashboard's view is
exactly as old as its last recompute, and never less than one commit behind
whatever the job is actually doing. Which is why each message trails the
thing it announces rather than leading it -- the start message goes after
the call's own first beat, the ending one after the commit and the final
beat. A launcher that reloaded any earlier would find exactly what it
already had. Neither container's disk is the
other's cache; the volume is the only thing both of them agree on.
