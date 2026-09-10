"""Lab API: environment (where this runs) vs. target (what storage an
operation touches) -- two different questions, kept separate on purpose.

`environment` is a fact about this process, auto-detected, never chosen:
`"modal"` inside a container with the volume mounted, `"local"` everywhere
else (a laptop, VS Code, a local Jupyter). `target` is a choice, made once at
the top of a notebook via `lab.init(target=...)`, naming which storage
backend `declare`/`bind` should actually touch: `"local"` (a project-local
directory, `config.LOCAL_STORAGE`) or `"modal"` (the volume,
`config.STORAGE`). `target` defaults to `environment` -- reaching the real
volume from a laptop, or touching local-only storage from a container, are
both things you have to ask for, never something that happens silently:

    import lab
    lab.init(target="modal")   # only needed to reach the real volume from
                                # a local environment -- otherwise skip this,
                                # target already defaults to environment

Not every (environment, target) pair is allowed:

    environment  target   declare   bind
    local        local    yes       yes
    local        modal    yes       no
    modal        local    no        no
    modal        modal    yes       yes

`declare`/`bind` call `_require` first and raise `PermissionError` on a
disallowed pair rather than doing something surprising -- an artifact never
has to know any of this itself, it's `lab`'s rule to enforce, not the
artifact's to ask about. The two `no`-for-bind cells collapse to one fact:
`bind` is only ever allowed when `target == environment`, so it never needs a
remote call -- it's always a plain local read against whichever root this
process already has. `declare` has the one real cross-environment case:
local environment, modal target -- write a manifest to the real volume from a
laptop, via `Artifact.declare()` (a thin wrapper over the deployed `declare`
function). A modal environment can never touch local-target storage at all
-- that would be ephemeral, untracked state living only on one container's
own disk, gone the moment it recycles.

    lab.declare(pretraining)                 # resolve + check, no writes
    lab.declare(pretraining, commit=True)     # ...and write/publish
    lab.plan(pretraining)                     # the artifacts to build, in order
    lab.declare(pretraining, visualize=True)  # ...drawn, colored by status
    lab.bind(tokenizer)                       # bind an object you have
    lab.bind("tokenizers/bpe-3.0k-e4649eb4ff")  # or by path, if you don't

`bind` never mutates what you hand it -- same contract `Artifact.bind` always
had: the artifact you pass in stays exactly as unbound as it was, and what
comes back is a new object, same class and parameters, with `_load` run and
its methods usable.

Distributed execution is not this API's job -- that's the Launcher (main.py's
attempt_launch/run_job, leases, concurrency). What this API gives a notebook
for running something itself is `lab.worker` plus `lab.root()`, for driving a
job's `run(root, worker)` by hand, sequentially, against local-target storage
-- the same shape the demo notebooks already use by hand against their own
throwaway root.
"""

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import artifacts.core.resolve as core_resolve
from artifacts.core.artifact import MANIFEST, Artifact
from artifacts.core.visualizer import SVG
from artifacts.core.visualizer import visualize as draw_graph
from config import LOCAL_STORAGE, STORAGE, VOLUME_NAME
from system.runtime import Worker


def _detect_environment() -> str:
    """Where this process is actually running: "modal" if the volume is
    mounted here, "local" otherwise. A fact about the process, not a
    setting -- computed once, at import, and never re-checked: nothing about
    a running process's own mounts changes later."""
    return "modal" if Path(STORAGE).exists() and Path(STORAGE).is_dir() else "local"


environment = _detect_environment()
target = environment  # default: touch whatever this environment natively has


def init(target: str | None = None) -> None:
    """Configure this session's target -- call once, at the top of a
    notebook, right after `import lab`:

        import lab
        lab.init(target="modal")   # reach the real volume from anywhere

    Omit `target` (or pass None) to reset it back to the default,
    `environment` -- touch whatever this environment natively has. `target`
    is a plain module attribute underneath (`lab.target = ...` still works;
    `init` doesn't do anything `declare`/`bind` couldn't already see by
    reading `lab.target` directly) -- this is just the one obvious place to
    look for "how do I point this at something else."
    """
    globals()["target"] = target if target is not None else environment


# Which (environment, target) pairs may do what -- see this module's own
# docstring for the reasoning behind each cell.
_PERMISSIONS = {
    ("local", "local"): {"declare", "bind"},
    ("local", "modal"): {"declare"},
    ("modal", "local"): set(),
    ("modal", "modal"): {"declare", "bind"},
}


def _require(op: str) -> None:
    if op not in _PERMISSIONS[(environment, target)]:
        raise PermissionError(
            f"{op}() not allowed: environment={environment!r} cannot target "
            f"{target!r} storage -- see lab.py's module docstring for the "
            f"full permission matrix"
        )


def _root_for(t: str) -> Path:
    if t == "local":
        LOCAL_STORAGE.mkdir(parents=True, exist_ok=True)
        return LOCAL_STORAGE
    return Path(STORAGE)


def root() -> Path:
    """The current target's storage root -- what to pass as `root` to
    job.run(root, worker) for a manual local run, or anywhere else a plain
    root is needed. Not permission-checked itself (execution is out of this
    API's scope, see the module docstring) -- by convention this is only
    ever meaningful with target == environment, the same restriction bind()
    itself enforces.
    """
    return _root_for(target)


# A fake Worker for running a job by hand from a lab cell --
# job.run(root, worker) -- instead of through main.run_job, which is the only
# other thing that ever builds a real one. No lease, no heartbeat:
# confirm_lease is a no-op, since nothing else is racing to hold an
# artifact_path from inside a notebook. No job.run() implementation actually
# reads artifact_path/call_id today (only main.run_job does, around the
# call), so these are placeholders, not identity.
#
# log goes to the console only. A job run through the launcher has its log
# kept by Modal under the call id that ran it, for a day (docs/LOGGING.md); a
# job run by hand from a cell is not a Modal call at all, so there is nothing
# to fetch and nowhere for it to be filed.
_log = logging.getLogger("lab")
_log.setLevel(logging.INFO)
if not _log.handlers:  # module-level, so this only ever runs once per process
    _log.addHandler(logging.StreamHandler())

worker = Worker(
    artifact_path="lab",
    call_id="lab",
    log=_log,
    confirm_lease=lambda *_, **__: None,
    progress={},
)


def _volume():
    """The volume handle, built on demand rather than at import.

    Module scope would mean a network call the moment anything imports `lab`
    -- including Modal itself, whenever an image stages this module as source.
    """
    import modal

    return modal.Volume.from_name(VOLUME_NAME)


def _outside_volume(op):
    """Run a Modal volume operation with cwd stepped outside the mount first.

    Modal counts an open cwd under the volume the same as an open file, and
    every notebook here starts rooted under root() (`jupyter`'s `root_dir`)
    -- so reload (and commit, which reloads internally to pick up what it
    just committed) would otherwise fail from essentially every real cell
    with "there are open files preventing the operation: cwd is inside
    volume".
    """
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        return op()
    finally:
        os.chdir(cwd)


def refresh() -> None:
    """Pull in whatever other containers have committed since this one booted.

    Keyed on `environment` alone, not `target`: this is about the volume
    *this container* has mounted, a fact of where the process is running,
    not a caller's storage choice -- no-op outside a container, since
    there's no local mount to reload.

    Not automatic even in a container. A notebook server rooted at the
    volume routinely holds files open under it, and reload can't run while
    it does -- better an explicit call in a cell than a background thread
    that fails half the time and yanks the filesystem the other half. Real
    failures (an actually open file, most commonly) reach the caller
    unchanged -- see `_outside_volume`'s own docstring.
    """
    if environment == "modal":
        _outside_volume(_volume().reload)


def publish() -> None:
    """Make this container's writes visible to everything else.

    Same `environment`-only gating as `refresh`, same reasoning. `declare`
    already calls this on its own when it commits against a modal target.
    This is for anything else a cell writes under root() by hand -- until it
    lands, the writes exist only on this container's own disk, same as a
    job's do before `initialize_worker` commits.
    """
    if environment == "modal":
        _outside_volume(_volume().commit)


def ls(prefix: str = "") -> list[str]:
    """Every artifact declared under `prefix`, on the current target, as
    artifact_paths.

    The listing that makes `bind` usable without guessing -- an artifact's
    folder is its identity, so "what's declared" is exactly "where are the
    manifests". Same access pattern as `bind` (a direct local read against
    whichever root the target resolves to, no remote equivalent exists for
    either), so it shares `bind`'s permission cell rather than getting a row
    of its own in the matrix.

        lab.ls()             -> every artifact on the current target
        lab.ls("tokenizers") -> just the shared tokenizers
        lab.ls("runs/RUN_TOKENS")
    """
    _require("bind")
    r = _root_for(target)
    base = r / prefix
    if not base.exists():
        return []
    return sorted(path.parent.relative_to(r).as_posix() for path in base.rglob(MANIFEST))


def bind(artifact_or_path: Artifact | str | Path) -> Artifact:
    """The bound version of `artifact_or_path`, on the current target.

    Two ways in, same old structure either way -- this never reimplements
    binding, it only ever supplies the root:

    - Given an artifact object -- one you already have, with its own
      parameters and therefore its own artifact_path -- calls *that
      object's own* `.bind(root)` directly. `lab.bind(tokenizer)` is exactly
      `tokenizer.bind(lab.root())`, just letting `lab` name the root instead
      of you spelling it out. The result is a new object, same class and
      parameters as what you passed in, just now with `_load` run and its
      methods usable (`.encode(...)`, etc.) -- `Artifact.bind`'s own
      contract, untouched. What you passed in is never mutated.
    - Given a path -- resolves whatever's declared there first
      (`Artifact.at`), for when you don't have the object in hand, only its
      folder.

    Only ever allowed when `target == environment` (see the permission
    matrix) -- so this is always a plain local read, never a remote call.
    Comes back bound when every declared file is there, plain when it
    isn't -- declared but not run yet, or run halfway.
    """
    _require("bind")
    r = _root_for(target)
    if isinstance(artifact_or_path, Artifact):
        return artifact_or_path.bind(r)
    return Artifact.at(r / artifact_or_path)


def _write_run_notebook(run_id: str, cell: str, root: Path) -> None:
    """Starter notebook at `root`/runs/{run_id}/notebook.ipynb, the first
    time this run declares successfully -- lab imports first, then `cell`,
    so opening it in the lab picks up right where the declaring notebook
    left off. Exclusive create: left alone on every later call for the same
    run_id, since by then it may already be the thing someone's editing.

    Same shape `main.py`'s own `_write_run_notebook` writes for the
    cross-environment case (routed through the deployed `declare` function)
    -- duplicated here on purpose rather than imported: `declare`'s direct
    path (target == environment) runs in every worker container, and none
    of them stage `lab.py` except the lab's own image, so `main.py` can't
    import this module without breaking `run_job`/`declare` elsewhere. A
    few lines of duplication is cheaper than that.
    """
    path = root / "runs" / run_id / "notebook.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = 'import lab\nfrom lab import worker\nlab.init(target="modal")\nlab.refresh()\n\n' + cell
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    try:
        with path.open("x") as handle:
            handle.write(json.dumps(notebook, indent=1))
    except FileExistsError:
        pass  # already there -- left alone, not resynced


def declare(
    artifact: Artifact,
    *,
    commit: bool = False,
    strict_commit: bool = False,
    cell: str | None = None,
    visualize: bool = False,
    verbose: bool = False,
) -> core_resolve.Dag | None:
    """Resolve `artifact` and its whole dependency tree against the current
    target and check it -- creating the plan, nothing more -- unless
    `commit` is True, in which case it also writes a manifest for everything
    still `new` and publishes. An artifact never has to know any of this
    permission logic itself; it just needs to know the target it's being
    asked to declare against, which this supplies.

    If `artifact` has its own `run_id` (a run-scoped artifact, e.g. a
    Pretraining) and `commit` is True, this also sets up that run's starter
    notebook the first time -- no separate call needed, `run_id` is read
    straight off the artifact rather than asked for again. `cell` -- the
    defining cell's own source, typically `In[-1]` -- seeds it beyond the
    bare `import lab` boilerplate; omit it and the notebook still gets
    created, just without that seed.

    When `target == environment`, this is a direct local operation: resolves
    the graph against whatever root the target resolves to, refuses outright
    if it is inconsistent when `commit` is True (nothing is written in that
    case, so a refusal leaves the target exactly as it was), and returns the
    resolved `Dag` -- iterate it, or read `.problems`/`.ok`, when the printed
    report isn't enough.

    After a successful commit the graph is resolved a second time, so the
    object handed back reports what's on disk *now* rather than the
    pre-write snapshot in which everything just written still reads `new`.
    That second pass is paid here, at the one place a human reads the
    result, rather than inside every write.

    `verbose=True` makes anything that blocks say *why*, in the report this
    returns and in a refusal to commit alike: for a `conflict`, the manifest
    on disk against the one being requested, leaf by leaf; for an
    `undeclared`, the files found in a folder nothing declared; for a
    `drift` under `strict_commit`, both commits. It applies to a plain check
    too -- a check is where you usually find out something is wrong, and
    "BLOCKED" without a reason is the moment you wanted the reason. The
    returned graph also answers on demand, verbose or not, via
    `print("\\n".join(dag.explain()))`.

    `visualize=True` additionally renders the dependency graph inline, each
    node outlined by its status on disk, reusing the graph this call already
    resolved rather than resolving again. It is an argument here rather than a
    function of its own because a drawing of statuses is only ever as true as
    the root it was resolved against, and this is the one place that knows how
    to reach the real one from either environment.

    The one other allowed case (local environment, modal target) routes
    through `artifact.declare(write=commit, strict_commit=strict_commit,
    run_id=..., cell=cell)` instead -- the deployed `declare` function, the
    only thing a local process can reach on the volume, which sets up the
    same starter notebook server-side -- and returns `None`: its report is
    printed there, same as `main.declare`'s always was, but there's no local
    `Dag` to hand back when there's no local root to have resolved one
    against, and nothing local for `visualize` to draw either.
    """
    _require("declare")
    run_id = getattr(artifact, "run_id", None)
    if target == environment:
        r = _root_for(target)
        def resolved() -> core_resolve.Dag:
            return core_resolve.resolve(
                artifact, target=r, strict_commit=strict_commit, verbose=verbose
            )

        dag = resolved()
        if commit:
            core_resolve.declare(dag)  # raises rather than writing over a mess
            if run_id:
                _write_run_notebook(run_id, cell or "", r)
            if environment == "modal":
                publish()
            dag = resolved()
        if visualize:
            _show(draw_graph(dag))
        return dag

    svg = artifact.declare(
        write=commit,
        strict_commit=strict_commit,
        run_id=run_id,
        cell=cell,
        verbose=verbose,
        visualize=visualize,
    )
    if svg:
        _show(SVG(svg))
    return None


def _show(svg: SVG) -> None:
    """Render a drawing in the notebook that asked for it. Imported here rather
    than at module scope: `lab` is staged as source into images that have no
    IPython, and importing it there would break every one of them."""
    from IPython.display import display

    display(svg)


def plan(artifact: Artifact) -> list[Artifact]:
    """The artifacts that have to be built to get `artifact`, in the order
    they have to be built, including `artifact` itself.

    One job produces one artifact, so this list *is* the work: the nth entry
    is the nth thing to run. It keeps the whole graph rather than only what's
    pending, which is what you want to read in a cell -- and that makes it a
    question about the artifact's own structure, not about any storage root.
    So it resolves without a target, touches no filesystem, and reads the same
    from anywhere. Use `declare` for what's actually on disk.
    """
    return core_resolve.plan(core_resolve.resolve(artifact))
