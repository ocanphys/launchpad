"""The notebook API. One entry point, declaration:

    import lab
    report = lab.declare(tokenizer)               # preview, writes nothing
    report = lab.declare(tokenizer, commit=True)  # publish missing manifests

A committed declaration of a run also copies the notebook it ran from to
`runs/<run_id>/declare-<uid>.ipynb`, so the volume keeps what declared it.

Construction, loading and binding stay on the artifact classes --
`Tokenizer(...)`, `Artifact.load(path)`, `tokenizer.bind()` -- and every one
of them reads STORAGE unless handed another root. Nothing to initialize.

Two volume verbs for the lab container: `refresh()` reloads the mount so
files written elsewhere (a job's outputs, a notebook saved in another lab)
appear; `save()` commits it so a notebook saved in JupyterLab outlives the
container. `worker` is a stand-in execution context for running one job by
hand from a cell, `job.run(root, lab.worker)`: no lease, no heartbeat, log to
the console.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import modal

from artifacts.core.artifact import Artifact, Resources
from artifacts.core.resolve import resolve
from artifacts.core.SGD.training import Training
from artifacts.core.storage import write_manifest
from config import LAB_COMMIT_ENV, STORAGE, VOLUME_NAME, get_git_commit
from system.runtime import Worker

# States a person has to look at before anything is written.
BLOCKING = ("conflict", "undeclared")


@cache
def _head() -> str:
    """Returns the commit of this process, stamped on every manifest it declares.

    Cached, so one session cannot stamp two commits; restart the kernel to pick
    up new work. The lab container has no git binary and no .git, so there the
    commit comes from LAB_COMMIT_ENV, which main.py bakes in.
    """
    override = os.environ.get(LAB_COMMIT_ENV)
    return override if override else get_git_commit()


class DeclarationError(ValueError):
    """A commit refused over blocking rows; `report` holds them.

    `args` is (report, verbose) rather than the rendered text so the error
    pickles: it crosses back from a container when declaring remotely.
    """

    def __init__(self, report: DeclarationReport, verbose: bool = False):
        super().__init__(report, verbose)
        self.report = report

    def __str__(self) -> str:
        return self.report.render(self.args[1])


@dataclass
class DeclarationReport:
    """One row per resolved path: path, state, drift, differences, created,
    updated, the stored declaring commit beside the one declaring now, and the
    stored and requested resources and tags. Counts and the verdict are derived."""

    rows: list[dict]
    strict_commit: bool = False

    @property
    def blockers(self) -> list[dict]:
        return [
            row
            for row in self.rows
            if row["state"] in BLOCKING or (self.strict_commit and row["drift"])
        ]

    def render(self, verbose: bool = False) -> str:
        """The report as printed; `verbose` adds each blocker's differences."""
        width = max(len(row["path"]) for row in self.rows) + 2
        state_width = max(len(row["state"]) for row in self.rows) + 2
        blocked = {row["path"] for row in self.blockers}
        lines = []
        for row in self.rows:
            notes = []
            if row["drift"]:
                stored, declaring = row["declaring_commit"]
                notes.append(f"drift: {stored[:7]} -> {declaring[:7]}")
            stored, requested = row["resources"]
            if stored is not None and stored != requested:
                verb = "updated" if row["updated"] else "differ, requested apply on commit"
                notes.append(f"resources {verb}: {_set(stored)} -> {_set(requested)}")
            stored, requested = row["tags"]
            if stored is not None and stored != requested:
                verb = "updated" if row["updated"] else "differ, requested apply on commit"
                notes.append(f"tags {verb}: {json.dumps(stored)} -> {json.dumps(requested)}")
            line = f"{row['path']:<{width}}{row['state']:<{state_width}}"
            lines.append(line.rstrip() + (f"  ({'; '.join(notes)})" if notes else ""))
            if verbose and row["path"] in blocked:
                for key, (on_disk, requested) in sorted(row["differences"].items()):
                    lines.append(
                        f"    {key}: on disk {json.dumps(on_disk)}, "
                        f"requested {json.dumps(requested)}"
                    )
        counts = Counter(row["state"] for row in self.rows)
        summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
        written = [row["path"] for row in self.rows if row["created"] or row["updated"]]
        if self.blockers:
            verdict = f"BLOCKED -- {len(self.blockers)} to resolve"
            if not verbose:
                verdict += " (verbose=True to see why)"
        elif written:
            plural = "s" if len(written) != 1 else ""
            verdict = f"{len(written)} manifest{plural} written: {', '.join(written)}"
        elif counts["new"]:
            verdict = f"ok -- {counts['new']} to declare"
        else:
            verdict = "ok -- everything already declared"
        return "\n".join([*lines, "", summary, verdict])

    __str__ = render


def _set(resources: dict) -> str:
    """The dimensions a Resources dict sets, as JSON: `{"gpu_type": "A100"}`."""
    return json.dumps({k: v for k, v in resources.items() if v is not None})


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """A manifest flattened to dotted leaf paths, the recorded fields left
    out: neither is definition."""
    if isinstance(value, dict) and value:
        out: dict[str, object] = {}
        for key, item in value.items():
            if key in Artifact.RECORDED:
                continue
            out.update(_leaves(item, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, item in enumerate(value):
            out.update(_leaves(item, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def _inspect(artifact: Artifact, root: Path, memo: dict[str, Artifact]) -> dict:
    """One report row: how what `root` holds at this artifact's path compares
    with the artifact requested. Every difference is `key: [on disk, requested]`.
    `memo` is `Artifact.load`'s, shared across one declaration's rows so a
    subtree many manifests embed decodes once.

    Drift is the declaring commit stored at this path against `_head()`. A
    request with `allocated_resources=None` or `tags=None` (a dependency of a
    loaded artifact) requests nothing: its row reports the stored ones as
    requested, so nothing is compared or written for them."""
    footprint = artifact.status(root)
    requested, requested_tags = artifact.allocated_resources, artifact.tags
    row = {
        "path": artifact.artifact_path.as_posix(),
        "state": "new",
        "drift": False,
        "differences": {},
        "created": False,
        "updated": False,
        "declaring_commit": [None, _head()],
        "resources": [None, None if requested is None else requested.model_dump()],
        "tags": [None, requested_tags],
    }
    if not footprint.manifest:
        present = [desc for desc, there in footprint.outputs.items() if there]
        if present:
            row["state"] = "undeclared"
            row["differences"] = {f"outputs.{desc}": ["present", None] for desc in present}
        return row
    try:
        stored = Artifact.load(artifact.artifact_path, root, memo)
    except ValueError as error:
        row["state"] = "conflict"
        row["differences"] = {"manifest": [str(error), None]}
        return row
    row["declaring_commit"] = [stored.declaring_commit, _head()]
    row["drift"] = stored.declaring_commit != _head()
    assert stored.allocated_resources is not None  # a manifest at its own path records them
    on_disk = stored.allocated_resources.model_dump()
    row["resources"] = [on_disk, on_disk if requested is None else requested.model_dump()]
    stored_tags = stored.tags or {}  # a manifest written before tags existed records none
    row["tags"] = [stored_tags, stored_tags if requested_tags is None else requested_tags]
    if stored != artifact:
        row["state"] = "conflict"
        on_disk, wanted = _leaves(stored.to_manifest()), _leaves(artifact.to_manifest())
        row["differences"] = {
            key: [on_disk.get(key), wanted.get(key)]
            for key in sorted(on_disk.keys() | wanted.keys())
            if on_disk.get(key) != wanted.get(key)
        }
        return row
    present = list(footprint.completion.values())
    row["state"] = "done" if all(present) else "partial" if any(present) else "already declared"
    return row


def _volume():
    """The volume handle, built on demand: at module scope it would be a
    network call the moment anything imports `lab`, image staging included."""
    return modal.Volume.from_name(VOLUME_NAME)


def _outside_volume(op):
    """Run a volume operation with cwd stepped outside the mount first.

    Modal counts a cwd under the volume as an open file, and a notebook rooted
    there would otherwise fail every reload and commit.
    """
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        return op()
    finally:
        os.chdir(cwd)


def refresh() -> None:
    """Reloads the volume, so files written outside this container appear.
    Only a container with the volume mounted can do this; Modal refuses
    anywhere else."""
    _outside_volume(_volume().reload)


def save() -> None:
    """Commits the volume, so what is on disk here outlives this container:
    save the notebook in JupyterLab first, then `lab.save()`. Only a
    container with the volume mounted can do this; Modal refuses anywhere
    else."""
    _outside_volume(_volume().commit)


def current_notebook() -> bytes | None:
    """The notebook file this kernel runs, or None outside a notebook.

    jupyter_server starts every kernel with JPY_SESSION_NAME set to its
    notebook's path; VS Code leaves `__vsc_ipynb_file__` in the namespace
    instead. A notebook renamed since its kernel started is a missing file
    here, which raises rather than being skipped.
    """
    path = os.environ.get("JPY_SESSION_NAME")
    if path is None:
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:  # a worker image: no notebook, by construction
            return None
        kernel = get_ipython()
        path = kernel.user_ns.get("__vsc_ipynb_file__") if kernel else None
    if path is None:
        return None
    return (STORAGE / path).read_bytes()  # absolute `path` wins the join


def declare(
    artifact: Artifact,
    *,
    root: Path = STORAGE,
    commit: bool = False,
    strict_commit: bool = False,
    verbose: bool = False,
    notebook: bytes | None = None,
) -> DeclarationReport:
    """The declaration report for `artifact` and everything it is built from,
    printed and returned; with `commit`, missing manifests are published first,
    stamped with this process's `_head()` as their declaring commit, and
    existing ones whose resources or tags differ from those requested take the
    requested ones, everything else in them kept.

    Preview writes nothing and returns blockers as rows. Commit refuses over
    any blocker -- a conflict, an undeclared folder, or drift under
    `strict_commit` -- and raises DeclarationError with the report attached
    before touching disk. `root` is
    for tests and scripts; a notebook gets STORAGE. The volume is reloaded
    before inspection and committed after writes only when this runs inside
    a container and `root` is its mount: anywhere else `/storage` is a plain
    folder with nothing behind it to reload.

    A commit that creates a run's artifact (one with a `run_id`) also writes
    the declaring notebook to `runs/<run_id>/declare-<uid>.ipynb`: the
    kernel's own by default, or the bytes handed in as `notebook` by a
    caller that has them and no kernel (`local.declare_on_volume`).
    """
    on_volume = root == STORAGE and not modal.is_local()
    graph = resolve(artifact)
    if commit and notebook is None:
        notebook = current_notebook()
    if on_volume:
        refresh()
    memo: dict[str, Artifact] = {}
    rows = [_inspect(node, root, memo) for node in graph]
    report = DeclarationReport(rows, strict_commit)
    if commit:
        if report.blockers:
            raise DeclarationError(report, verbose)
        for node, row in zip(graph, rows):
            (stored, requested), (stored_tags, requested_tags) = row["resources"], row["tags"]
            if row["state"] == "new":
                stamped = node.model_copy(
                    update={
                        "declaring_commit": _head(),
                        "allocated_resources": Resources() if node.allocated_resources is None else node.allocated_resources,
                        "tags": node.tags or {},
                    }
                )
                manifest = stamped.to_manifest()
            elif stored != requested or stored_tags != requested_tags:
                on_disk = Artifact.load(node.artifact_path, root, memo)
                update = {"allocated_resources": node.allocated_resources, "tags": node.tags}
                manifest = on_disk.model_copy(update=update).to_manifest()
            else:
                continue
            write_manifest(root, node.artifact_path, manifest)
            if row["state"] == "new":
                row.update(_inspect(node, root, memo), created=True)
                row["state"] = row["state"].removeprefix("already ")  # this commit declared it
            else:  # keep the pair, so the report shows what changed to what
                row.update(
                    _inspect(node, root, memo), updated=True, resources=[stored, requested], tags=[stored_tags, requested_tags]
                )
        if rows[-1]["created"] and notebook is not None and isinstance(artifact, Training):
            copy = root / "runs" / artifact.run_id / f"declare-{artifact.uid}.ipynb"
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(notebook)
        if on_volume:
            save()
    print(report.render(verbose))
    return report


# A job run by hand from a cell is not a Modal call: nothing races it for a
# lease, and its log has nowhere to be filed but the console.
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
