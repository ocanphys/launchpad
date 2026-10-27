# Workflows

Every workflow in the repo as a flowchart. Each box names an abstract step on
its first line and the functions that carry it out below. Python and JS only;
tests and notebooks are left out.

1. [System map](#1-system-map)
2. [Artifact lifecycle](#2-artifact-lifecycle)
3. [Constructing an artifact](#3-constructing-an-artifact)
4. [Declaring](#4-declaring)
5. [Manifest codec, load and bind](#5-manifest-codec-load-and-bind)
6. [Launching](#6-launching)
7. [Cancelling](#7-cancelling)
8. [Running a job](#8-running-a-job)
9. [Lease confirm](#9-lease-confirm)
10. [State recompute and the read routes](#10-state-recompute-and-the-read-routes)
11. [Logging](#11-logging)
12. [Artifact families](#12-artifact-families)
13. [Training a leg](#13-training-a-leg)
14. [Dashboard front end](#14-dashboard-front-end)
15. [Auth gate and the lab](#15-auth-gate-and-the-lab)
16. [One-off scripts](#16-one-off-scripts)

## 1. System map

Who talks to what. Containers are the four Modal functions in `main.py`.

```mermaid
flowchart LR
  subgraph clients ["Clients"]
    NB["Notebook in JupyterLab<br/>lab.declare(), lab.refresh(), lab.save()"]
    LP["Laptop<br/>local.declare_on_volume()"]
    BR["Browser<br/>web/app.js"]
  end

  subgraph app ["Modal app: main.py"]
    LB["leasebook (web, max_containers=1)<br/>build_api()"]
    RJ["run_job (worker_image, GPU optional)<br/>initialize_worker(), Job.run()"]
    PL["persist_logs (scheduled)<br/>persist_snapshot()"]
    JP["jupyter (lab_image)<br/>start_jupyterlab()"]
  end

  subgraph stores ["State"]
    V[("Volume trainvols<br/>manifest.json, outputs, logs/, call_history.json")]
    D[("Dicts<br/>leases, beats, call_logs, call_history")]
    Q[("Queue<br/>refreshes")]
  end

  BR -->|"HTTP: state, launch, cancel, logs"| LB
  NB -->|"writes manifests"| V
  LP -->|"_declare.remote()"| V
  JP --- NB
  LB -->|"spawn()"| RJ
  LB -->|"/lab redirect"| JP
  LB <-->|"grant, read beats and logs"| D
  LB -->|"reload, read"| V
  Q -->|"listen()"| LB
  RJ -->|"confirm lease, beat, publish logs"| D
  RJ -->|"announce()"| Q
  RJ -->|"write outputs, commit()"| V
  PL -->|"read live logs"| D
  PL -->|"append log files, commit()"| V
```

## 2. Artifact lifecycle

Moved to [docs/artifacts.md](../docs/artifacts.md#lifecycle).

## 3. Constructing an artifact

Moved to [docs/artifacts.md](../docs/artifacts.md#construction).

## 4. Declaring

`lab.declare` in a notebook, or `local.declare_on_volume` from a laptop, which
runs the same function in a container that mounts the volume.

```mermaid
flowchart TD
  LAP["Laptop<br/>local.declare_on_volume()<br/>current_notebook() shipped along"] -->|"_declare.remote() in an ephemeral app"| S
  S["lab.declare(artifact, commit, strict_commit)"] --> R["Resolve the graph, dependencies first<br/>resolve.resolve(): visit() over Artifact.deps()<br/>one node per artifact_path, raise on conflicting definitions or cycles"]
  R --> NBK["Read the declaring notebook (commit only)<br/>current_notebook()"]
  NBK --> RF["Reload the mount (inside a container only)<br/>refresh() -> _outside_volume()"]
  RF --> I["Compare each node with what the disk holds<br/>_inspect(): Artifact.status(), Artifact.load(), _leaves()<br/>state: new, undeclared, conflict, declared, partial, done"]
  I --> REP["Build the report<br/>DeclarationReport: blockers, render()"]
  REP --> C{"commit?"}
  C -->|no| OUT["Print and return the report"]
  C -->|yes| BL{"Any blockers?<br/>conflict, undeclared, or drift under strict_commit"}
  BL -->|yes| ERR["raise DeclarationError(report)"]
  BL -->|no| W["Write missing manifests, update changed resources<br/>to_manifest() -> manifest_json() -> manifest.json<br/>Artifact.load() + replace(allocated_resources)"]
  W --> RI["Re-inspect written rows<br/>_inspect()"]
  RI --> CP["Copy the notebook if it created a Training<br/>runs/run_id/declare-uid.ipynb"]
  CP --> SV["Commit the mount (inside a container only)<br/>save() -> _outside_volume()"]
  SV --> OUT
```

## 5. Manifest codec, load and bind

The codec, load and bind moved to [docs/manifests.md](../docs/manifests.md).
Status stays here until the repository page takes it. It is a pure filesystem
read, used by declaring, launching and the state map:

```mermaid
flowchart LR
  ST["Artifact.status(root)"] --> FP["Footprint<br/>manifest: bool<br/>outputs: paths() exist<br/>completion: completion_paths() exist"]
  FP --> CO["Footprint.complete"]
  FP --> DP["Artifact.durable_progress()<br/>Training.durable_progress(): furthest checkpoint"]
```

## 6. Launching

The dashboard's run button, through to a spawned `run_job` call
(`launcher/leasebook.py`, `launcher/state.py`).

```mermaid
flowchart TD
  U["Click run<br/>render.actionButton() -> ctx.onLaunch -> app.act('launch')"] -->|"POST /launch/artifact_path"| E["launch_endpoint()"]
  E --> AL["attempt_launch(artifact_path, run_job)"]
  AL --> V1{"Path safe?<br/>safe_relpath()"}
  V1 -->|no| RF["Refuse and log<br/>refuse() -> launcher_logger()"]
  V1 -->|yes| G["Read the current grant and its beat<br/>leases.get(), beats.get()"]
  G --> A1{"Call already under way?<br/>is_active(), is_starting()"}
  A1 -->|yes| RF
  A1 -->|no| PF["Read the artifact fresh under mount_lock<br/>volume.reload(), Artifact.load()"]
  PF --> C1{"Launchable?<br/>producer set, not status().complete,<br/>every deps() has manifest and is complete"}
  C1 -->|no| RF
  C1 -->|yes| SP["Drop the stale lease, size the call, spawn<br/>leases.pop(), resource_options(allocated_resources)<br/>run_job.with_options().spawn()"]
  SP --> GR["Grant the lease and record the call<br/>new_grant(), leases.put(), call_history.put()"]
  GR --> RC["Recompute the map<br/>recompute('launch') -> state()"]
  RF --> RESP["Answer launched, message"]
  RC --> RESP
  RESP --> UI["Page refetches<br/>app.load(view)"]
```

## 7. Cancelling

```mermaid
flowchart TD
  U["Click stop<br/>render.actionButton() -> ctx.onCancel -> app.act('cancel')"] -->|"POST /cancel/artifact_path"| E["cancel_endpoint(): safe_relpath()"]
  E --> CC["cancel_call(artifact_path)"]
  CC --> P{"Was there a grant?<br/>leases.pop()"}
  P -->|no| N["Refuse: no active call"]
  P -->|yes| K["Ask Modal to stop the call<br/>modal.FunctionCall.from_id().cancel()"]
  K --> O["Classify the outcome<br/>call.get(timeout=0): cancelled, late request, already ended"]
  O --> RC["Recompute the map<br/>recompute('cancel')"]
  P -.->|"lease gone: worker's next confirm fails"| LL["LeaseLost raised in the worker<br/>Lease.confirm() inside worker.publishing()"]
```

## 8. Running a job

Inside the `run_job` container (`main.py`, `system/runtime.py`).

```mermaid
flowchart TD
  S["run_job(artifact_path)"] --> IW["Set up the call<br/>initialize_worker()"]
  IW --> L1["Start logging before touching the mount<br/>current_call_id(), start_logging(call_id), call_logger('job', WORKER)"]
  L1 --> LS["Make the lease handle and the Worker<br/>Lease(artifact_path, call_id), Worker(log, confirm_lease, progress)"]
  LS --> HB["Start the heartbeat thread<br/>heartbeat() -> beat() -> beats.put()<br/>first pass: announce('started') -> refreshes.put()"]
  HB --> RL["Reload and confirm<br/>volume.reload(), lease.confirm('boot')"]
  RL --> C2["worker.confirm_lease('pre run')"]
  C2 --> LD["Read the declaration<br/>Artifact.load(artifact_path, STORAGE)"]
  LD --> JB["Find the producer<br/>artifact.job() -> locate(producer)(artifact)"]
  JB --> RUN["Do the work<br/>job.run(STORAGE, worker), see section 12"]
  RUN --> PUB["Every output file<br/>worker.publishing(path): write path.call_id.tmp -> confirm_lease() -> tmp.replace(path)"]
  PUB --> C3["Confirm before commit<br/>confirm_lease('pre vol commit'), lease.confirm('commit')"]
  C3 --> OK["ending = done"]
  RUN -.->|raises| EX{"LeaseLost or InputCancellation?"}
  EX -->|yes| ELL["ending = lease lost"]
  EX -->|no| EF["ending = failed<br/>logger.exception(), traceback.clear_frames()"]
  OK --> FIN["Land the files<br/>volume.commit()"]
  ELL --> FIN
  EF --> FIN
  FIN --> FB["Last beat, marked exited<br/>finished.set(), thread.join()"]
  FB --> AN["Tell the launcher how it ended<br/>announce(ending) -> refreshes.put()"]
  AN --> SL["Final log publish<br/>stop_logging()"]
```

## 9. Lease confirm

Every fence a worker takes (`system/lease_protocol.py`).

```mermaid
flowchart TD
  C["Lease.confirm(label)"] --> F["One fresh read<br/>fence(): leases.get(artifact_path)"]
  F --> V{"Verdict"}
  V -->|MATCH| OK["Log lease held, return"]
  V -->|"MISMATCH, no grant"| L1["raise LeaseLost: cancelled or taken"]
  V -->|"MISMATCH, another call_id"| L2["raise LeaseLost: superseded"]
  V -->|"UNKNOWN, Dict read failed"| T{"Tries left?<br/>LEASE_RETRIES"}
  T -->|yes| SL["Wait LEASE_BACKOFF"] --> F
  T -->|no| L3["raise LeaseLost: indeterminate"]
```

## 10. State recompute and the read routes

The leasebook's in-memory map and the routes that read it
(`launcher/leasebook.py`, `launcher/state.py`).

```mermaid
flowchart TD
  T4["Container start<br/>build_api(): start_logging(LAUNCHER), load_snapshot_from_volume()"] --> RC
  T1["A worker started or ended<br/>listen(): refreshes.get_many()"] --> RC
  T2["Accepted launch or cancel<br/>launch_endpoint(), cancel_endpoint()"] --> RC
  T3["Refresh button<br/>POST /refresh: refresh_route()"] --> RC
  RC["recompute(reason)"] --> ST["Snapshot under mount_lock<br/>state(): volume.reload(), beats.items(), leases.items()"]
  ST --> SC["One entry per manifest.json found<br/>Artifact.load(path, memo), cached in resolved<br/>status().completion, deps(), parameters(), durable_progress()<br/>unreadable manifest: status conflict"]
  SC --> DR["Derive blocked_by, done, ready"]
  DR --> LV["Live half of each entry<br/>liveness(): is_active(), is_starting()<br/>verdict: done, running, starting, failed, runnable, blocked"]
  LV --> LAT[("latest map in memory")]

  LAT --> SR["GET /state: state_route()<br/>re-read lease and beat for running and starting rows -> liveness()"]
  AR["GET /artifact/path: artifact_endpoint()<br/>mount_lock, open_json(manifest.json), open_jsonl(train.jsonl)"]
  LG["GET /logs/path: logs_endpoint()<br/>artifact_calls() -> stream(call_id, livedict | volume)"]
  LL["GET /launcher-logs<br/>stream(LAUNCHER, livedict | volume)"]
  DBG["GET /debug/name: dict_dump()"]
```

## 11. Logging

From a log call to a row on the page (`system/logs.py`).

```mermaid
flowchart TD
  subgraph writers ["Who writes a row"]
    WL["Worker<br/>worker.log = call_logger('job', WORKER, call_id)"]
    LL["Launcher<br/>launcher_logger(call_id)"]
    AL["Anything unstamped<br/>source = ambient"]
  end
  WL --> BH
  LL --> BH
  AL --> BH
  BH["Buffer the row per channel<br/>BufferHandler.emit(): row(), channel()<br/>first row of a channel: continued()"] --> PT["Publish every HEARTBEAT_SECONDS<br/>start_logging() thread -> BufferHandler.publish() -> try_publish()"]
  PT --> LIVE[("call_logs<br/>call_id:livedict:source")]

  subgraph persist ["persist_logs, every PERSIST_LOGS_EVERY"]
    PS["persist_logs(): volume.reload()<br/>persist_snapshot()"] --> LS["Read what the files hold<br/>load_snapshot_from_volume(): parse_jsonl(), merge call_history.json"]
    LS --> AP["Append live rows past the file length<br/>artifact_path/logs/call_id.jsonl, logs/launcher.jsonl"]
    AP --> CH["Write call_history.json, volume.commit()"]
  end
  LIVE --> PS
  AP --> VOL[("call_logs<br/>call_id:volume:source")]

  LIVE --> RD["Read both storages<br/>stream(), artifact_calls()"]
  VOL --> RD
  RD --> UI["Dedupe and draw<br/>logview.union() -> logStream() -> logLine()"]
```

## 12. Artifact families

Moved to [docs/artifacts.md](../docs/artifacts.md#families).

## 13. Training a leg

Moved to [docs/training.md](../docs/training.md#what-the-job-does).

## 14. Dashboard front end

`web/`: `app.js` owns state and the network; `render.js`, `artifactview.js`,
`logview.js` and `filter.js` build DOM through `el.js`.

```mermaid
flowchart TD
  NAV["Page load or hashchange<br/>route() -> parseRoute(), routeToken++"] --> V{"Which view"}
  V -->|table| FT["fetchTable()<br/>GET state, GET launcher-logs"]
  V -->|"artifact/path"| FA["fetchArtifactView(path)<br/>GET state, GET logs/path<br/>GET artifact/path only when the entry changed"]
  NAV --> POLL["Refetch every POLL_MS until the route changes<br/>poll() -> view()"]
  POLL --> FT
  POLL --> FA

  FT --> DRAW["Draw the table if anything changed<br/>draw(): filterBar(), visible()<br/>artifactRow(): dot(), progressCell() -> formatProgress(), actionButton()<br/>problemRow()"]
  FT --> LOGS["Draw the launcher log<br/>drawLauncherLogs() -> union() -> logStream()"]
  FA --> AV["Draw the artifact page<br/>renderArtifactView(): summary(), curvesSection() -> columns(), chart()<br/>logs() -> logRows() -> logStream()"]

  BTN["Row button<br/>act(route, path, label)"] --> PEND["Mark pending and redraw<br/>pending.set(), redraw()"]
  PEND -->|"POST launch or cancel"| SRV["leasebook, sections 6 and 7"]
  SRV --> LD["load(view)"]
  REF["Refresh button<br/>load(refresh) -> POST refresh, pending.clear()"] --> LD
  FLT["Filter change<br/>filter.js: level(), choices(), range(), passes()"] --> RD["redraw()"]
  SPLIT["Drag the log split<br/>pointerdown handler sets --log-height"]
```

## 15. Auth gate and the lab

Every leasebook request passes the gate first (`launcher/gate.py`,
`launcher/lab_server.py`, `lab.py`).

```mermaid
flowchart TD
  REQ["Any request"] --> G["require_password middleware, from gate()"]
  G --> C{"Path is /login,<br/>or cookie equals sha256(DASHBOARD_PASSWORD)?"}
  C -->|"no, GET"| RED["303 to /login<br/>login_page()"]
  C -->|"no, other method"| U401["401"]
  RED --> POST["POST /login: login()<br/>hmac.compare_digest(), set_cookie()"]
  POST --> REQ
  C -->|yes| MW["log_request_errors middleware"]
  MW --> ROUTE["Route handler"]
  ROUTE -->|"GET /lab"| LR["lab_redirect()<br/>jupyter.get_web_url() + JUPYTER_TOKEN"]
  LR --> JUP["jupyter container<br/>start_jupyterlab(): overrides.json, jupyter lab rooted at STORAGE"]
  JUP --> NBK["Notebook<br/>lab.declare(), lab.refresh(), lab.save()<br/>Artifact.load(), artifact.bind()"]
  NBK --> HAND["Run a job by hand<br/>job.run(root, lab.worker): no lease, console log"]
```

## 16. One-off scripts

```mermaid
flowchart TD
  subgraph migrate ["migrate_model_parameters.py"]
    M0["__main__: app.run()"] --> M1["_on_volume.remote(write)<br/>volume.reload()"]
    M1 --> M2["migrate(root, write)<br/>rglob manifest.json, converted(), nested()"]
    M2 --> M3["Check every rewrite decodes to its folder<br/>Artifact.from_manifest().artifact_path"]
    M3 --> M4["Write if --write<br/>manifest_json(), volume.commit()"]
  end

  subgraph profile ["profiling/"]
    P0["main.profile on worker_image runs util.profile<br/>torch.profiler captures returned as prof.events()"]
  end
```
