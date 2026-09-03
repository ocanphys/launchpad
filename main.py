import time
from pathlib import Path

import modal

import dag.resolve as dag_resolve
from config import (
    APP_NAME,
    CONTAINER_LIFETIME,
    FLATLINE,
    HEARTBEAT_SECONDS,
    STORAGE,
    VOLUME_NAME,
)
from dag.artifact import MANIFEST, Artifact, Resources
from lease_protocol import beats, leases, new_grant
from logs import LOG_FILENAME, read_call_logs, read_log
from runtime import initialize_worker

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.12").pip_install("regex", "tqdm")

# The modules every call needs whatever it runs: config, its logger, its
# lease, the scope that wires those two together. No third-party
# dependencies of their own -- add_local_python_source copies these .py
# files into an image, it doesn't install what they import, and none of
# them import anything outside the stdlib.
CALL_SOURCE = ("config", "logs", "lease_protocol", "runtime")

# The artifact/job model, one package per family (see dag/spec.md): "dag"
# is the framework (Artifact, Job, resolve, visualizer), the rest are its
# concrete artifact+job pairs. Each is its own top-level package rather than
# nested under one "artifacts" package, so a family can import another
# (e.g. datasets importing sources.artifact) without a shared parent import
# pulling in every family at once. datasets and mappeddatasets are separate
# packages, not one -- DataSet and MappedDataSet are different kinds of
# artifact (one copies bytes, one never writes any), not variants of a
# shared "dataset" concept.
DAG_SOURCE = ("dag", "sources", "tokenizers", "datasets", "mappeddatasets", "models")


def safe_relpath(path: str) -> bool:
    """Is `path` safe to join under STORAGE -- non-empty, not absolute, no
    `..` component that could walk it outside the volume. Every route that
    takes a path from the URL (an artifact_path or a run_id) checks this
    before it ever reaches `Path(STORAGE) / path`.

    safe_relpath("runs/toy/pretraining") -> True
    safe_relpath("../../etc/passwd") -> False
    """
    return bool(path) and not Path(path).is_absolute() and ".." not in Path(path).parts


def is_active(grant: dict | None, beat: dict | None, now: float) -> bool:
    """Is `grant`'s holder currently beating fresh enough to trust?

    Pure -- takes the grant and its holder's beat rather than fetching
    either, so `read_state` (a batch snapshot, one call_id's beat already
    looked up per run) and `launch_job` (a single-run lookup, no mount) both
    decide "active" the same way instead of each having its own rule.
    """
    if grant is None or beat is None:
        return False
    return now - beat["last_beat_ts"] < FLATLINE * HEARTBEAT_SECONDS


# Two images: the web dashboard needs fastapi and none of what a job needs;
# a job needs neither fastapi nor the web/ folder.
#
# Both carry all of CALL_SOURCE even so, and not just the modules each one's
# own function reads. A container starts by importing the module its function
# is defined in -- this one -- and this module imports `runtime`, which
# imports `logs`, whether or not the function actually being called touches
# them. Trimming an image to only what its own function reads gets a crash
# loop, not a smaller image (see git history if that stops being obvious).
#
# Both also carry DAG_SOURCE alongside CALL_SOURCE, not folded into it: each
# entry there is its own package (dag/__init__.py, sources/__init__.py, ...),
# not a single module -- add_local_python_source stages each as a sibling of
# main.py, which is what keeps `import dag...`/`import sources...`/etc valid
# remotely, the same as they are locally.
WEB_DIR = Path("/web")  # where web_image mounts the web/ folder

web_image = (
    base_image.pip_install("fastapi[standard]")
    # The page is HTML/JS, so it stays HTML/JS -- mounted like a data file,
    # not pasted into this module as strings.
    .add_local_dir(local_path="web", remote_path=WEB_DIR.as_posix())
    .add_local_python_source(*CALL_SOURCE, *DAG_SOURCE)
)

# What `declare` and `run_job` run under: no fastapi, no volume-unrelated
# deps -- dag/sources/datasets/mappeddatasets/*.py are stdlib-only (plus
# config.get_git_commit, which neither function calls: every artifact either
# arrives fully constructed from a client that already stamped its commit,
# or is loaded straight off a manifest that already has one). tokenizers/bpe.py
# needs regex+tqdm, hence base_image's pip_install above. models/mock is
# stdlib-only too; a real model family under models/ would need its own
# torch-installing image, not this one.
#
# The one exception is numpy, needed by mappeddatasets/tokenstream.py's memmap
# windowing -- but only at the moment something actually binds a
# MappedDataSet, via a local import inside MappedDataSet._load, not at
# mappeddatasets/artifact.py's own top level. So it's added here, not to
# base_image: web_image never calls .bind() on anything (leasebook only
# resolves and inspects), and has no reason to carry it.
worker_image = base_image.pip_install("numpy").add_local_python_source(
    *CALL_SOURCE, *DAG_SOURCE
)


def artifact_state(
    artifact: Artifact, root: Path, cache: dag_resolve.InspectCache | None = None
) -> dict:
    """One declared artifact's status plus what it's waiting on.

    `done`/`ready` are both derived from `status`, not stored -- there's
    only ever one on-disk answer to agree with.

    `cache` -- see dag_resolve.inspect -- is what keeps a whole-book read
    (many artifacts, sharing dependencies) from re-inspecting the same
    shared tokenizer or source once per artifact that happens to depend on
    it. Every state-building function below passes the one cache it built
    for its own request through every call it makes here.
    """
    status, drift = dag_resolve.inspect(artifact, root, cache)
    depends_on = artifact.deps()
    blocked_by = [
        dep.artifact_path.as_posix()
        for dep in depends_on
        if dag_resolve.status(dep, root, cache) != "done"
    ]
    done = status == "done"
    return {
        "type": type(artifact).__name__,
        "status": status,
        "drift": drift,
        "depends_on": [dep.artifact_path.as_posix() for dep in depends_on],
        "blocked_by": blocked_by,
        "done": done,
        "ready": not done and not blocked_by,
    }


def _lease_snapshot() -> tuple[dict, dict, float]:
    """One read of `leases`/`beats` plus the clock to judge them by -- shared
    by every state-building function below so a beat's freshness is judged
    against the same `now` its lease was read alongside, regardless of which
    view is asking."""
    return dict(leases.items()), dict(beats.items()), time.time()


def _with_lease(info: dict, path: str, grants: dict, beat_records: dict, now: float) -> dict:
    """Stamp one artifact's state dict with its lease/heartbeat, read off a
    `_lease_snapshot()`. A lease is granted per artifact_path (see
    `attempt_launch`), so this one stamp is correct regardless of whether
    `path` belongs to a run, or to a shared kind's own flat/grouped view.
    """
    grant = grants.get(path)
    beat = beat_records.get(grant["call_id"]) if grant else None
    info["call_id"] = grant["call_id"] if grant else None
    info["active"] = is_active(grant, beat, now)
    info["last_heartbeat"] = beat["last_beat_ts"] if beat else None
    return info


def read_state() -> dict:
    """One read of the whole volume, sliced into the three shapes the
    dashboard's views (runs/sources/datasets) each want.

    Every view used to be its own function, each doing its own
    `volume.reload()`, its own lease snapshot, and its own `InspectCache` --
    and since a source or tokenizer is routinely reachable from a run *and*
    from `sources`/`datasets`' own shared-folder listing, the same artifact
    got independently re-inspected once per view that happened to reach it.
    One read fixes that at the root: one `volume.reload()`, one lease
    snapshot, one cache, and one `states` map below that computes each
    distinct artifact_path's state at most once no matter how many of
    `runs`/`sources`/`datasets` reference it -- dependency graphs are cheap
    to walk once the state they're built from is already in hand.

    `runs` comes from the volume's `runs/` directory, not from `leases`: a
    folder with no lease -- never started, or superseded and never reclaimed --
    is exactly the gap worth being able to see, and starting from `leases`
    instead would hide it. A run's artifacts come from
    `dag_resolve.declared_under`, which can raise on a bad manifest --
    "broken", not "not ready". This is the one place that decides what a
    raise means for a whole run: not `runs`, but `problem_runs`, keyed the
    same way, holding the error instead of an artifacts snapshot. One
    run_id's raise doesn't cost the rest of the book -- the loop below
    catches it per run_id, not around the whole loop.

    `sources` is flat -- every Source ever declared under root/sources,
    attached to a run or not (see dag_resolve.declared_of_kind), unlike
    `runs` there's no grouping. `datasets` is every DataSet and
    MappedDataSet ever declared (root/datasets and root/mappeddatasets),
    each shown the way a run shows its artifacts -- its own row, plus its
    full transitive dependency closure (dag_resolve.dependency_closure),
    which for a dataset reaches every TokenizedSource its train/valid mix
    uses and, through those, the Tokenizer and Source(s) behind them. Each
    dataset's own entry also carries `mapped`: True for a MappedDataSet, so
    the frontend can tag it inline without parsing `type` -- the two kinds
    share this one view by design, but a MappedDataSet owns no bytes of its
    own (mappeddatasets/artifact.py), worth flagging at a glance. That flag
    is stamped on a copy of the shared state dict, not the cached original
    -- `states` is shared with `runs`/`sources`, and a dataset reachable
    from a run too should not show a run artifact tagged `mapped`.

    `leases` and `beats` go back close to untouched: `leases` verbatim, `beats`
    with one field added per entry, `lease`, naming which artifact_path that
    call_id is the *current* holder for (None if it is not the current holder
    of anything -- a superseded container still beating, or one that never
    held a lease at all). Both come from the one snapshot taken here, so a
    beat's `lease` always agrees with what a run's own artifacts say that
    call holds -- a lease is granted per artifact_path (see `attempt_launch`),
    so that agreement is checked per artifact, not per run.
    """
    volume.reload()
    storage_root = Path(STORAGE)
    runs_root = storage_root / "runs"
    run_ids = (
        sorted(p.name for p in runs_root.iterdir() if p.is_dir())
        if runs_root.exists()
        else []
    )

    grants, beat_records, now = _lease_snapshot()

    # `grants` is keyed by artifact_path (a lease is granted per
    # artifact_path -- see attempt_launch), so this maps each call_id to
    # the artifact_path it currently holds the lease for.
    held_by = {grant["call_id"]: artifact_path for artifact_path, grant in grants.items()}

    # One cache, and one memoized state dict, for the whole read -- not one
    # per run or per view. A tokenizer or source shared across several runs,
    # or reachable from both a run and the sources/datasets views, then gets
    # inspected and stamped with its lease exactly once, regardless of how
    # many places reference it below.
    cache: dag_resolve.InspectCache = {}
    states: dict[str, dict] = {}

    def state_for(a: Artifact) -> dict:
        path = a.artifact_path.as_posix()
        if path not in states:
            states[path] = _with_lease(
                artifact_state(a, storage_root, cache), path, grants, beat_records, now
            )
        return states[path]

    runs = {}
    problem_runs = {}
    for run_id in run_ids:
        try:
            declared = dag_resolve.declared_under(run_id, storage_root)
        except Exception as exc:
            problem_runs[run_id] = {"error": str(exc)}
            continue
        runs[run_id] = {"artifacts": {a.artifact_path.as_posix(): state_for(a) for a in declared}}

    sources = {
        a.artifact_path.as_posix(): state_for(a)
        for a in dag_resolve.declared_of_kind("sources", storage_root)
    }

    datasets = {}
    for a in [
        *dag_resolve.declared_of_kind("datasets", storage_root),
        *dag_resolve.declared_of_kind("mappeddatasets", storage_root),
    ]:
        info = {**state_for(a), "mapped": type(a).__name__ == "MappedDataSet"}
        deps_state = {dep.artifact_path.as_posix(): state_for(dep) for dep in dag_resolve.dependency_closure(a)}
        datasets[a.artifact_path.as_posix()] = {"state": info, "artifacts": deps_state}

    beats_out = {
        call_id: {**beat, "lease": held_by.get(call_id)}
        for call_id, beat in beat_records.items()
    }

    return {
        "now": now,
        "runs": runs,
        "problem_runs": problem_runs,
        "beats": beats_out,
        "leases": grants,
        "sources": sources,
        "datasets": datasets,
    }


def artifact_job_name(artifact_path: str, root: Path) -> str | None:
    """The class name of the job that produces the artifact at
    `artifact_path` (e.g. "SourceJob"), or None if there's no manifest to
    resolve it from yet -- a call's log directory can exist before (or
    outlive) the manifest that names its job, so this is best-effort, not
    load-bearing for anything but display.

    Same lookup `run_job` already does to find a job to run -- load the
    manifest, resolve its producer -- just read for its class name instead
    of run.
    """
    manifest_path = root / artifact_path / MANIFEST
    if not manifest_path.exists():
        return None
    try:
        artifact = Artifact.load(manifest_path)
        return type(dag_resolve.producer_for(artifact)).__name__
    except Exception:
        return None


def artifact_manifest_summary(artifact_path: str, root: Path) -> dict | None:
    """The artifact declared at `artifact_path`, reduced to what a drill-down
    page wants: its own type and parameters (`manifest()` already keeps these
    separate from its dependencies) plus one line per direct dependency
    naming where it lives. Each dependency is named, not inlined -- `deps()`
    hands back the live Artifact objects, not their nested manifests, and a
    link to that artifact's own page is all the summary needs.

    None if there's no manifest yet (declared but not built) -- same
    "best-effort" contract as `artifact_job_name`.
    """
    manifest_path = root / artifact_path / MANIFEST
    if not manifest_path.exists():
        return None
    artifact = Artifact.load(manifest_path)
    manifest = artifact.manifest()
    return {
        "type": manifest["artifact"],
        "parameters": manifest["parameters"],
        "depends_on": [
            {"artifact_path": dep.artifact_path.as_posix(), "type": type(dep).__name__}
            for dep in artifact.deps()
        ],
    }


def _stamp(entries: list[dict], artifact_path: str, job_name: str | None) -> list[dict]:
    """Tag every entry with the artifact_path and job name it belongs to --
    `read_call_logs`/`read_log` have no notion of either, they only know
    call_ids and log files (see logs.py)."""
    for entry in entries:
        entry["artifact_path"] = artifact_path
        entry["job"] = job_name
    return entries


def artifact_log_entries(artifact_path: str, root: Path) -> list[dict]:
    """One artifact's whole log history -- every call that has ever held its
    lease, merged into one timeline and tagged with the artifact_path and
    job that produced it.
    """
    entries = read_call_logs(root / artifact_path / "logs")
    return _stamp(entries, artifact_path, artifact_job_name(artifact_path, root))


def run_log_entries(run_id: str, root: Path) -> list[dict]:
    """Every worker that has ever touched anything declared under `run_id`,
    merged into one timeline -- the run-scoped counterpart to
    `artifact_log_entries`.

    Walks the same artifact set `read_state` shows for this run
    (`declared_under`: the run's own manifests plus every shared dependency
    reachable from them), so "what happened in this run" never disagrees
    with "what this run's dashboard rows are".
    """
    declared = dag_resolve.declared_under(run_id, root)
    entries: list[dict] = []
    for artifact in declared:
        entries.extend(artifact_log_entries(artifact.artifact_path.as_posix(), root))
    entries.sort(key=lambda e: e["ts"])
    return entries


@app.function(
    image=web_image,
    volumes={STORAGE: volume},
    max_containers=1,
)
@modal.asgi_app()
def leasebook():
    """The page and the data it polls, under one URL.

    An ASGI app rather than two `fastapi_endpoint`s because two endpoints are two
    URLs on two subdomains: the page would have to be told where its data lives,
    and the browser would treat the answer as cross-origin and refuse to read it.
    Served together, `state` is just a relative path, and no CORS question arises.
    """
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    # The one state route for all three table views (runs/sources/datasets)
    # -- read_state() reads the whole volume once and slices it three ways,
    # so the frontend fetches this once per poll and switches views locally
    # instead of round-tripping again on every nav click.
    @api.get("/state")
    def state() -> dict:
        return read_state()

    # :path, not a plain path segment -- an artifact_path contains its own
    # /s (runs/my-run/pretraining), which a plain segment can't match.
    @api.post("/launch/{artifact_path:path}")
    def launch_endpoint(artifact_path: str) -> dict:
        launched, message, _call = attempt_launch(artifact_path)
        return {"launched": launched, "message": message}

    # call_id is a query param, not a path segment: "logs of this one call"
    # is a narrowing of the artifact-scoped view, not a route of its own --
    # the dashboard links to it as #/artifact/<path>?call=<call_id>.
    @api.get("/logs/artifact/{artifact_path:path}")
    def artifact_logs_endpoint(artifact_path: str, call_id: str | None = None) -> dict:
        if not safe_relpath(artifact_path):
            return {"artifact_path": artifact_path, "entries": [], "error": "invalid artifact_path"}
        if call_id is not None and not safe_relpath(call_id):
            return {"artifact_path": artifact_path, "call_id": call_id, "entries": [], "error": "invalid call_id"}

        volume.reload()
        root = Path(STORAGE)

        if call_id is not None:
            log_path = root / artifact_path / "logs" / call_id / LOG_FILENAME
            entries = read_log(log_path, call_id) if log_path.exists() else []
            entries = _stamp(entries, artifact_path, artifact_job_name(artifact_path, root))
            return {"artifact_path": artifact_path, "call_id": call_id, "entries": entries}

        return {"artifact_path": artifact_path, "entries": artifact_log_entries(artifact_path, root)}

    # What the artifact drill-down page shows above its log table: type,
    # own parameters, and dependency links -- see artifact_manifest_summary.
    @api.get("/manifest/{artifact_path:path}")
    def manifest_endpoint(artifact_path: str) -> dict:
        if not safe_relpath(artifact_path):
            return {"artifact_path": artifact_path, "error": "invalid artifact_path"}
        volume.reload()
        try:
            summary = artifact_manifest_summary(artifact_path, Path(STORAGE))
        except Exception as exc:
            return {"artifact_path": artifact_path, "error": str(exc)}
        if summary is None:
            return {"artifact_path": artifact_path, "error": "not built yet -- no manifest"}
        return {"artifact_path": artifact_path, **summary}

    # :path even though a run_id has no /s of its own -- kept consistent
    # with the artifact route above rather than a plain segment, on the same
    # reasoning safe_relpath already exists for: never trust the shape of a
    # URL segment to match the shape of the thing it names.
    @api.get("/logs/run/{run_id:path}")
    def run_logs_endpoint(run_id: str) -> dict:
        if not safe_relpath(run_id):
            return {"run_id": run_id, "entries": [], "error": "invalid run_id"}
        volume.reload()
        try:
            entries = run_log_entries(run_id, Path(STORAGE))
        except Exception as exc:
            return {"run_id": run_id, "entries": [], "error": str(exc)}
        return {"run_id": run_id, "entries": entries}

    # Everything else -- index.html at "/" and its same-origin JS modules
    # (app.js, el.js, render.js) -- is a static file under WEB_DIR. Mounted
    # last: routes are matched in registration order, so /state and /launch
    # are claimed above before this catch-all sees them.
    api.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return api


@app.function(image=worker_image, volumes={STORAGE: volume})
def declare(artifact: Artifact, *, write: bool = False, strict_commit: bool = False) -> str:
    """Check (or, with write=True, declare) `artifact` and its full
    dependency tree against the volume -- the notebook equivalent of
    `Declaration(artifact, root).check()` / `.write()`, run where `root` can
    be the volume itself rather than a local mirror of it. Returns the
    human-readable report either way.
    """
    volume.reload()
    declaration = dag_resolve.Declaration(artifact, Path(STORAGE), strict_commit)
    if write:
        declaration.write()
        volume.commit()
    else:
        declaration.check()
    return str(declaration)


@app.function(image=worker_image, volumes={STORAGE: volume}, timeout=CONTAINER_LIFETIME)
def run_job(artifact_path: str) -> None:
    """Load the artifact at `artifact_path`, run whatever produces it, under
    this call's own logger and lease.

    The producing job is resolved from the registry here, and only here --
    `resolve.producer_for` (the same lookup dependency resolution already
    uses), called right before it's needed rather than anywhere upstream of
    this. Nothing about launching (`attempt_launch`) has to know it either.
    """
    with initialize_worker(artifact_path, volume) as worker:
        worker.confirm_lease("pre run")
        artifact = Artifact.load(Path(STORAGE) / artifact_path / MANIFEST)
        job = dag_resolve.producer_for(artifact)
        job.run(Path(STORAGE), worker)
        worker.confirm_lease("pre vol commit")


@app.function(image=worker_image, volumes={STORAGE: volume})
def declared_artifact(artifact_path: str) -> tuple[Artifact, dict] | None:
    """The artifact declared at `artifact_path` and its current status, or
    None if nothing's declared there.

    Needs the volume actually mounted (`volume.reload()`, then plain `Path`
    reads against it) -- pulled out of `attempt_launch` and into its own
    function so that work always happens inside a container, regardless of
    whether `attempt_launch` itself was called from the web route (already
    inside one) or `launch_job` (a local entrypoint, which is never inside
    one -- `Path(STORAGE)` isn't a real mount on the machine running `modal
    run`, and volume.reload() outright refuses to run there).
    """
    volume.reload()
    manifest_path = Path(STORAGE) / artifact_path / MANIFEST
    if not manifest_path.exists():
        return None
    artifact = Artifact.load(manifest_path)
    return artifact, artifact_state(artifact, Path(STORAGE))


def resource_options(resources: Resources) -> dict:
    """An artifact's `allocated_resources` -> `Function.with_options()` kwargs.

        resource_options(Resources()) == {}
        resource_options(Resources(cpu=2)) == {"cpu": 2}
        resource_options(Resources(gpu_type="A100")) == {"gpu": "A100"}
        resource_options(Resources(gpu_type="A100", gpu_count=2)) == {"gpu": "A100:2"}

    Unvalidated by design -- `Resources` is only ever constructed by trusted
    code (a notebook declaring an artifact), and Modal itself rejects a bad
    cpu/gpu value server-side at spawn time. Three interpretations worth
    being explicit about instead of leaving to accident:

    - `cpu=None` counts as not declared, same as the field being absent
      entirely. Calling `with_options()` at all, even with every argument at
      its own default, still moves the call into its own dynamically
      configured container pool (see `attempt_launch`), and a no-op override
      is exactly the case not worth paying that for.
    - `gpu_count` with no `gpu_type` is dropped -- nothing to attach a
      count to, so the job runs GPU-less rather than erroring.
    - `gpu_count: 0` reads the same as omitted -- Modal's own "TYPE:COUNT"
      string has no way to ask for zero of a GPU.
    """
    options = {}
    if resources.cpu is not None:
        options["cpu"] = resources.cpu
    if resources.gpu_type:
        options["gpu"] = (
            f"{resources.gpu_type}:{resources.gpu_count}"
            if resources.gpu_count
            else resources.gpu_type
        )
    return options


def attempt_launch(
    artifact_path: str,
) -> tuple[bool, str, modal.functions.FunctionCall | None]:
    """Grant `artifact_path` to one call of `run_job`, or refuse and say why.

    Shared by `launch_job` (a local entrypoint, which waits on the call) and
    the web `/launch` route (which cannot wait -- a request has to return) so
    the checks and the grant/spawn itself are decided in one place, not
    twice: `artifact_path` is untrusted here in a way it wasn't for a
    CLI-only launcher, since a POST route is reachable by anyone with the
    URL, so it's validated before touching a lease or a path built from it.

    Readiness comes from `declared_artifact` (a remote call -- see its own
    docstring for why this can't just read `Path(STORAGE)` here directly),
    computed the same way `read_state` already does for the dashboard, so
    "is this safe to launch" and "what does the dashboard show" never have
    two different answers. Resources come from the artifact's own
    `allocated_resources`, turned into `Function.with_options()` kwargs by
    `resource_options`. That call is skipped entirely when an artifact
    declares none (`options` empty): calling it unconditionally would move
    every launch into its own dynamically configured container pool,
    separate even from another call with the same empty options, so an
    artifact that asks for nothing special stays pooled on `run_job`'s own
    base configuration.

    The producing job is never looked up here -- readiness only needs the
    artifact and its declared dependencies' statuses, not what runs it.
    `run_job` resolves the job itself, right before running it.
    """
    if not safe_relpath(artifact_path):
        return False, f"invalid artifact_path {artifact_path!r}", None

    grant = leases.get(artifact_path)
    beat = beats.get(grant["call_id"]) if grant else None
    if is_active(grant, beat, time.time()):
        return False, f"{artifact_path}: already has an active call -- not launching", None

    declared = declared_artifact.remote(artifact_path)
    if declared is None:
        return False, f"{artifact_path}: not declared -- nothing to launch", None
    artifact, state = declared
    if not state["ready"]:
        return False, f"{artifact_path}: blocked on {state['blocked_by']}", None

    leases.pop(artifact_path, None)
    options = resource_options(artifact.allocated_resources)
    fn = run_job.with_options(**options) if options else run_job
    call = fn.spawn(artifact_path)
    leases.put(artifact_path, new_grant(call.object_id, type(artifact).__name__))
    return True, f"granted lease for {artifact_path} -> {call.object_id}", call


@app.local_entrypoint()
def launch_job(artifact_path: str):
    """Grant one artifact to one job call, then wait for it -- same shape as
    `launch`. The decision itself is `attempt_launch`'s.
    """
    launched, message, call = attempt_launch(artifact_path)
    print(message)
    if launched:
        print("waiting for it to finish")
        call.get()
