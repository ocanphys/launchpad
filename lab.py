"""What a notebook with the volume actually mounted imports.

The Jupyter container that served these notebooks is removed for now (it was
`main.jupyter`), so nothing currently deploys this module -- it is kept
because it is the whole notebook-facing API, and bringing the lab back is
re-adding the image and the web_server function, not rewriting this.

Everything here needs the volume mounted. A notebook on your laptop can only
reach it through `modal.Function.from_name(APP_NAME, "declare").remote(...)`
-- one function, returning one string, and nothing else about the volume is
reachable at all: `Path(STORAGE)` is not a mount out there, and
`volume.reload()` refuses to run off a container (see
`main.declared_artifact`'s docstring). Where `ROOT` *is* the volume, the two
ways of getting hold of an artifact both work directly:

    lab.load("tokenizers/bpe-3.0k-e4649eb4ff")     # by path
    BPETokenizer(vocab_size=3000, ...).bind(lab.ROOT)   # by parameters

and declaring is a local call rather than a round trip:

    lab.check(pretraining)      # preview, writes nothing
    lab.declare(pretraining)    # writes the manifests, then commits

Every function below is a thin wrapper over something in `dag/` that already
does the work. Nothing here re-implements resolution, inspection, or manifest
writing; if a behaviour is surprising, it belongs to `dag.resolve` or
`dag.artifact`, not to this module.
"""

from pathlib import Path

import dag.resolve as dag_resolve

# Importing dag.resolve is what fills ARTIFACTS/REGISTRY -- every concrete
# family is imported for its side effect there, which is what makes
# `Artifact.at` able to name the class a manifest refers to. main.py leans on
# the same import for the same reason.
from config import LAB_NOTEBOOKS, STORAGE, VOLUME_NAME
from dag.artifact import MANIFEST, Artifact
from dag.visualizer import SVG, visualize

ROOT = Path(STORAGE)
NOTEBOOKS = ROOT / LAB_NOTEBOOKS


def _volume():
    """The volume handle, built on demand rather than at import.

    Module scope would mean a network call the moment anything imports `lab`
    -- including Modal itself, whenever an image stages this module as source.
    """
    import modal

    return modal.Volume.from_name(VOLUME_NAME)


def refresh() -> None:
    """Pull in whatever other containers have committed since this one booted.

    Not automatic. A notebook server rooted at the volume routinely holds
    files open under it, and reload can't run while it does -- better an
    explicit call in a cell than a background thread that fails half the time
    and yanks the filesystem the other half.
    """
    _volume().reload()


def publish() -> None:
    """Make this container's writes visible to everything else.

    `declare` already calls it. This is for anything else a cell writes under
    ROOT by hand -- until it lands, the writes exist only on this container's
    own disk, same as a job's do before `initialize_worker` commits.
    """
    _volume().commit()


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
    """The artifact declared at `artifact_path`, bound if it's built.

    `Artifact.at` does all of it, including refusing a folder whose manifest
    describes an artifact that belongs somewhere else. Comes back bound when
    every declared file is there, plain when it isn't -- declared but not run
    yet, or run halfway.
    """
    return Artifact.at(ROOT / artifact_path)


def check(artifact: Artifact, strict_commit: bool = False) -> dag_resolve.Declaration:
    """Reconcile `artifact` and its whole dependency tree against the volume,
    writing nothing.

    Returns the `Declaration` itself rather than its report: in a cell it
    prints the report anyway (`Declaration.__repr__` is `__str__`), and having
    the object means `.rows`, `.problems` and `.ok` are there when the report
    isn't enough. That is the one difference from `main.declare`, which has to
    flatten to a string to cross the wire.
    """
    return dag_resolve.Declaration(artifact, ROOT, strict_commit).check()


def declare(artifact: Artifact, strict_commit: bool = False) -> dag_resolve.Declaration:
    """`check`, then write a manifest for everything still `new`, then commit.

    Refuses outright if the tree is inconsistent -- `Declaration.write` raises
    rather than declaring half of a disputed graph. Nothing is committed in
    that case, so a refusal leaves the volume exactly as it was.
    """
    declaration = dag_resolve.Declaration(artifact, ROOT, strict_commit)
    declaration.write()
    publish()
    return declaration


def plan(artifact: Artifact) -> SVG:
    """The dependency graph, drawn, with each node outlined by its status on
    the volume. Renders inline as a cell's last line."""
    return visualize(dag_resolve.resolve(artifact), ROOT)
