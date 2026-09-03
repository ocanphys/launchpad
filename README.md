# launchpad

Artifacts are folders on a Modal Volume, each owning a `manifest.json` that
names what it is and what it's built from; state and orchestration are two
Modal Dicts and one web container. The full model -- what an artifact and a
job *are*, how declaration/resolution/launching fit together -- is
[dag/spec.md](dag/spec.md); this file is the map of where things live and
how traffic flows between them.

## Where things live

- **Volume (`trainvols`)**: one folder per artifact, holding its own
  `manifest.json`, whatever files that artifact type declares, and its
  `logs/{call_id}/job.log` -- a lease (and so a log) is granted per
  artifact, not per run (see `runtime.initialize_worker`). Level and
  category conventions for what goes in these files are in
  [docs/LOGGING.md](docs/LOGGING.md).
  - **Shared roots** -- `sources/`, `tokenizers/`, `datasets/`,
    `mappeddatasets/` -- hold artifacts with no `run_id` in their
    parameters: reused across runs rather than rebuilt per run. `datasets/`
    (`DataSet`, which copies bytes into its own `train.bin`/`valid.bin`) and
    `mappeddatasets/` (`MappedDataSet`, which owns no bytes of its own and
    reads straight out of its sources' `tokens.bin`) are separate top-level
    packages and folders -- different kinds of artifact, not variants of one
    "dataset" concept.
  - **`runs/{run_id}/`** holds that run's own run-scoped artifacts --
    today, just `pretraining/`. Everything a run depends on that *isn't*
    run-scoped (its tokenizer, its dataset, the sources behind them) lives
    under a shared root instead, found by walking the dependency tree
    embedded in the run's own manifest, not nested under the run's folder.
- **Dict `launchpad-leases`**: one entry per **artifact_path** (not per
  run -- see dag/spec.md §8), naming the call_id currently holding that
  artifact.
- **Dict `launchpad-beats`**: one entry per call_id, the timestamp of its
  last heartbeat -- how a reader tells a live call from a dead one.

## Dashboard

The web app (`leasebook`) serves a small dashboard with three views --
`runs`, `datasets`, `sources` -- sharing one set of UI components; a run's
own artifacts and a dataset's dependency closure are both grouped by
artifact type the same way. No build step, no framework. Full writeup:
[docs/UI.md](docs/UI.md).

## The lab (removed for now)

There was a `jupyter` function serving JupyterLab with the volume mounted,
linked from the dashboard's header. It is gone: no `lab_image`, no `/lab`
route, no `launchpad-lab` secret, and nothing seeds or commits
`/storage/notebooks` any more. Notebooks already saved there are untouched.

What survives is [lab.py](lab.py), the API those notebooks import -- a thin
wrapper over `dag.resolve` that works anywhere `lab.ROOT` is a real mount:

```python
import lab
lab.ls("tokenizers")                              # what's declared
lab.load("tokenizers/bpe-3.0k-e4649eb4ff")        # by path, bound if built
Tokenizer(vocab_size=3000, ...).bind(lab.ROOT)    # by parameters, equal artifact
lab.check(pretraining); lab.declare(pretraining)  # local, not a round trip
```

Bringing the lab back means re-adding the image and the `web_server`
function, plus the secret; nothing in `lab.py` has to change.

## Read/write traffic

- **`leasebook`** (the web app) is pinned to a single container
  (`max_containers=1`) -- it's both the dashboard and the launcher, so
  there's one copy of the traffic pattern below, not several racing each
  other.
  - `/state`: reads the whole volume once per request -- one `leases`/
    `beats` Dict snapshot (no mount needed), one `volume.reload()`, one walk
    of the manifests off `leasebook`'s own **local mount** of the volume --
    and slices the result into the runs/sources/datasets shapes the
    dashboard's three views each want, so a shared artifact (a source
    behind several tokenizers, say) is only inspected once per request no
    matter how many views reference it. Fast, but only as fresh as that
    container's last `volume.reload()`.
  - `/launch/{artifact_path:path}`: checks the lease, makes a `.remote()`
    call to `declared_artifact` (below) to confirm the artifact is
    declared and ready, then spawns `run_job` and writes the new grant --
    returns immediately, it doesn't wait for the job to finish.
- **`declared_artifact`** and **`run_job`** are separate Modal functions,
  each with their own container and their own mount of the volume --
  `declared_artifact` reads one manifest and its status; `run_job` writes
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

The full model is [dag/spec.md](dag/spec.md); this is the shape of it.
Each artifact family (`sources/`, `tokenizers/`, `datasets/`,
`mappeddatasets/`, `models/mock/`, ...) pairs an `Artifact` subclass
(parameters, where it lives, what files it comprises) with exactly one
`Job` subclass that produces it -- no `job_uid`, no per-run config file.
A `Job` subclass declares what it produces with one class annotation
(`artifact: Tokenizer`), and defining the class registers it as that
type's producer automatically, at import time (`Job.__init_subclass__`
populating `dag.job.REGISTRY`). Its inputs are derived,
never declared separately: `artifact.deps()` -- the artifact-valued
parameters of the one thing it produces -- so a job's dependency list can
never drift out of sync with what its own artifact actually names.

Declaring (writing `manifest.json` files ahead of the work, for a whole
dependency tree at once) and launching (granting a lease and spawning the
job that fills one manifest in) are separate steps -- see `dag.resolve.
Declaration` and `main.attempt_launch`. There's no per-job `resources`
block in a config file; a job's resource ask lives on its own artifact
(`allocated_resources: Resources`, e.g. `Resources(gpu_type="A100")`),
turned into `Function.with_options(...)` kwargs by `attempt_launch` right
before spawning -- an artifact that declares none runs on `run_job`'s own
default pool.

## One job, traced

One artifact's job, spawned, run to completion, and exited -- against the
two Dicts and the volume it reads and writes along the way. Running the
whole time alongside it, on its own clock: `leasebook`, polling `/state*`
for a picture of the very same book through its own separate local mount.

```mermaid
sequenceDiagram
    participant L as Launcher (cli / web)
    participant Le as Leases (Dict)
    participant B as Beats (Dict)
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

    activate J
    J->>V: reload()
    J->>Le: confirm ("boot")
    J->>Le: confirm ("pre run")
    J->>J: run() -- resolve producing Job, write files
    J-->>B: PUT heartbeat (daemon thread, throughout)
    J->>Le: confirm ("pre vol commit")
    J->>Le: confirm ("commit")
    J->>V: commit()
    deactivate J

    Note over L,J: meanwhile, independently -- leasebook polls /state* every ~2s
    loop every ~2s
        L-->>Le: GET (batch)
        L-->>B: GET (batch)
        L->>V: reload() (leasebook's own local mount)
    end
```

Four confirms, not two: `initialize_worker` (`runtime.py`) brackets the
whole call with "boot" (before anything runs) and "commit" (right after,
before `volume.commit()`); `run_job` itself adds "pre run" (before
resolving and calling the job) and "pre vol commit" (right after) -- each
one a fresh re-read of the grant, so whichever of two racing launches lost
discovers it as early as the next checkpoint, not only at the very end.

A job's writes are private until `commit()` publishes them to the volume --
its log, its manifest check, its output files all live only on that one
container's own disk until then. `leasebook` keeps a separate copy of its
own, refreshed by its own `reload()` whenever something polls `/state*` --
so the dashboard's view is never more than one poll behind whatever's been
committed, and never less than one commit behind whatever the job is
actually doing. Neither container's disk is the other's cache; the volume
is the only thing both of them agree on.
