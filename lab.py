"""Lab API: the same functions work from JupyterLab (volume mounted) or a
local notebook (VS Code, no volume) -- the same names, the same shapes back,
routed differently underneath:

    lab.load("tokenizers/bpe-3.0k-e4649eb4ff")     # by path
    lab.check(pretraining)      # preview the graph, no writes
    lab.declare(pretraining)    # declare, then commit

In JupyterLab, `ROOT` is real (the volume is mounted), so these read and
write it directly -- `load` comes back bound when the artifact is built,
`check`/`declare` return the full `Declaration` (`.rows`, `.problems`, `.ok`).
Locally, `ROOT` doesn't exist, so the same calls route through the deployed
Modal functions instead (`declared_artifact`, `declare`) -- `load` comes back
*unbound* (the manifest, not the bytes: bind() needs files that are only ever
on the volume), and `check`/`declare` return `None` -- their report is
printed, same as `main.declare`'s always was, because there's no local
Declaration to hand back when there's no local filesystem to build one from.

That's the one real difference, and it's `_in_container()` below deciding it,
not the caller: a notebook that only calls `load`/`check`/`declare` and reads
their *printed* output doesn't need to know or care which side it's on. It
only becomes visible the moment code tries to use what `load` gave back as if
it were bound (`.encode(...)`, `.train_tokens`, ...) -- which is the "loading
something too big to have locally" case, and there's no way around that one
except being in a container.
"""

import logging
import os
import tempfile
from pathlib import Path

import dag.resolve as dag_resolve

# Importing dag.resolve is what fills ARTIFACTS/REGISTRY -- every concrete
# family is imported for its side effect there, which is what makes
# `Artifact.at` able to name the class a manifest refers to. main.py leans on
# the same import for the same reason.
from config import APP_NAME, LAB_NOTEBOOKS, STORAGE, VOLUME_NAME
from dag.artifact import MANIFEST, Artifact
from dag.visualizer import SVG, visualize
from system.runtime import Worker

ROOT = Path(STORAGE)
NOTEBOOKS = ROOT / LAB_NOTEBOOKS


def _in_container() -> bool:
    """Is the volume actually mounted here? The one thing every function
    below branches on -- cheap and deterministic (a stat, not a network
    call), so it's fine to call it on every entry rather than caching it
    once at import: nothing about a running process's mounts changes later,
    but recomputing costs nothing and needs no invalidation story either.
    """
    return ROOT.exists() and ROOT.is_dir()

# A fake Worker for running a job by hand from a lab cell --
# job.run(root, worker) -- instead of through main.run_job, which is the only
# other thing that ever builds a real one. No lease, no heartbeat:
# confirm_lease is a no-op, since nothing else is racing to hold an
# artifact_path from inside a notebook. No job.run() implementation actually
# reads artifact_path/call_id today (only main.run_job does, around the
# call), so these are placeholders, not identity.
#
# log goes to the console only, not to a file under ROOT -- TODO: wire this
# up to system.logs.call_logger once something needs to read a lab-run job's
# log back. That opens a real file on the volume, which would need routing
# through _outside_volume the same way refresh/publish are, since an open
# log file blocks reload exactly like an open cwd does.
_log = logging.getLogger("lab")
_log.setLevel(logging.INFO)
if not _log.handlers:  # module-level, so this only ever runs once per process
    _log.addHandler(logging.StreamHandler())

worker = Worker(
    artifact_path="lab",
    call_id="lab",
    log=_log,
    confirm_lease=lambda *_, **__: None,
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
    every notebook here starts rooted under ROOT (`jupyter`'s `root_dir`) --
    so reload (and commit, which reloads internally to pick up what it just
    committed) would otherwise fail from essentially every real cell with
    "there are open files preventing the operation: cwd is inside volume".
    """
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        return op()
    finally:
        os.chdir(cwd)


def refresh() -> None:
    """Pull in whatever other containers have committed since this one booted.

    No-op outside a container -- there's no local mount to reload. Inside
    one, this does real work and can raise for real reasons (an actually
    open file, most commonly), which reach the caller unchanged: swallowing
    them here would hide exactly the failure a notebook needs to see and act
    on (close whatever's open, or read the error to find out what is -- see
    `_outside_volume`'s own docstring for the one case already handled).

    Not automatic even in a container. A notebook server rooted at the
    volume routinely holds files open under it, and reload can't run while
    it does -- better an explicit call in a cell than a background thread
    that fails half the time and yanks the filesystem the other half.
    """
    if _in_container():
        _outside_volume(_volume().reload)


def publish() -> None:
    """Make this container's writes visible to everything else.

    No-op outside a container -- there's nothing local to commit; `declare`
    already routes to the volume on its own when called locally (see
    `declare`). Inside a container, this is for anything else a cell writes
    under ROOT by hand -- until it lands, the writes exist only on this
    container's own disk, same as a job's do before `initialize_worker`
    commits. Real failures propagate, same reasoning as `refresh`.
    """
    if _in_container():
        _outside_volume(_volume().commit)


def ls(prefix: str = "") -> list[str]:
    """Every artifact declared under `prefix`, as artifact_paths.

    The listing that makes `load` usable without guessing -- an artifact's
    folder is its identity, so "what's on the volume" is exactly "where are the
    manifests".

        lab.ls()             -> every artifact
        lab.ls("tokenizers") -> just the shared tokenizers
        lab.ls("runs/RUN_TOKENS")
    """
    base = ROOT / prefix
    if not base.exists():
        return []
    return sorted(
        path.parent.relative_to(ROOT).as_posix() for path in base.rglob(MANIFEST)
    )


def load(artifact_path: str | Path) -> Artifact:
    """The artifact declared at `artifact_path`.

    In a container, this is `Artifact.at(ROOT / artifact_path)` -- a local
    read, bound when every declared file is there, plain when it isn't
    (declared but not run yet, or run halfway). Locally, there's no ROOT to
    read, so this calls the deployed `declared_artifact` function instead
    and gets back the same kind of object, minus the binding: the manifest
    and its parameters, never the bytes `bind()` would have read off disk,
    because those bytes only ever exist on the volume. Calling a method that
    needs the bound state (`.encode(...)`, `.train_tokens`, ...) on what
    this returns locally raises the same `RuntimeError` an unbuilt artifact's
    would in a container -- "not bound yet" -- for the same reason: nothing
    to read it from here.
    """
    if _in_container():
        return Artifact.at(ROOT / artifact_path)

    import modal
    declared_artifact_fn = modal.Function.from_name(APP_NAME, "declared_artifact")
    result = declared_artifact_fn.remote(str(artifact_path))
    if result is None:
        raise FileNotFoundError(f"{artifact_path}: not declared on the volume")
    artifact, _state = result
    return artifact


def check(artifact: Artifact, strict_commit: bool = False) -> dag_resolve.Declaration | None:
    """Reconcile `artifact` and its whole dependency tree against the volume,
    writing nothing.

    In a container, returns the `Declaration` itself rather than its report:
    in a cell it prints the report anyway (`Declaration.__repr__` is
    `__str__`), and having the object means `.rows`, `.problems` and `.ok`
    are there when the report isn't enough. Locally there's no ROOT to build
    a `Declaration` from, so this routes through `artifact.declare(write=
    False)` (the deployed `declare` function) instead and returns `None` --
    its report is printed, same as `main.declare`'s always was, but there's
    no local object behind it to hand back.
    """
    if _in_container():
        return dag_resolve.Declaration(artifact, ROOT, strict_commit).check()
    artifact.declare(write=False, strict_commit=strict_commit)
    return None


def declare(artifact: Artifact, strict_commit: bool = False) -> dag_resolve.Declaration | None:
    """`check`, then write a manifest for everything still `new`, then commit.

    In a container: writes locally, commits, and returns the `Declaration`
    (same reasoning as `check`). Refuses outright if the tree is
    inconsistent -- `Declaration.write` raises rather than declaring half of
    a disputed graph, and nothing is committed in that case, so a refusal
    leaves the volume exactly as it was. Locally: routes through
    `artifact.declare(write=True)`, which does the same check-then-write on
    the volume via the deployed `declare` function, and returns `None` for
    the same reason `check` does.
    """
    if _in_container():
        declaration = dag_resolve.Declaration(artifact, ROOT, strict_commit)
        declaration.write()
        publish()
        return declaration
    artifact.declare(write=True, strict_commit=strict_commit)
    return None


def plan(artifact: Artifact) -> SVG:
    """The dependency graph, drawn, with each node outlined by its status on
    the volume. Renders inline as a cell's last line."""
    return visualize(dag_resolve.resolve(artifact), ROOT)
