"""The notebook API. Two entry points, declaration of one artifact and of
an experiment's, and the experiment's artifacts read back:

    import lab
    report = lab.declare(tokenizer)               # preview, writes nothing
    report = lab.declare(tokenizer, commit=True)  # publish missing manifests
    report = lab.create_experiment("batch-size-study", runs, commit=True)
    runs = lab.load_experiment("batch-size-study")

A committed declaration that creates an artifact it was asked for also copies
the notebook it ran from to `<artifact_path>/declaring.ipynb`, so the volume
keeps what declared it; an experiment keeps its own copy too, in
`experiments/<experiment_uid>/declaration.ipynb`.

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
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import modal

from artifacts.core.artifact import Artifact, Resources
from artifacts.core.resolve import resolve
from artifacts.core.storage import MANIFEST, read_manifest, write_manifest
from config import LAB_COMMIT_ENV, LAB_WORKSPACE, STORAGE, VOLUME_NAME, get_git_commit
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
    updated, the stored declaring commit beside the one declaring now, the
    stored and requested resources, and the stored tags beside what a commit
    leaves. `roots` are the paths that were asked for, the rest of the rows
    what they are built from. Counts and the verdict are derived."""

    rows: list[dict]
    roots: list[str]
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
                notes.append(f"resources differ, stored kept: {_set(stored)} -> {_set(requested)}")
            stored, merged = row["tags"]
            if stored is not None and stored != merged:
                verb = "added" if row["updated"] else "to add on commit"
                notes.append(f"tags {verb}: {json.dumps(stored)} -> {json.dumps(merged)}")
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
        created = {row["path"] for row in self.rows if row["created"]}
        declared = [path for path in self.roots if path in created]
        heading = ["", f"declared on artifact path{'s' if len(declared) > 1 else ''}:"] if declared else []
        return "\n".join([*lines, "", summary, verdict, *heading, *declared])

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

    Drift is the declaring commit stored at this path against `_head()`.
    Tags are `[stored, what a commit leaves]`: the stored ones with every
    requested value they lack appended, so a commit adds tags and never
    removes one. A request with `allocated_resources=None` or `tags=None` (a
    dependency of a loaded artifact) requests nothing: its row reports the
    stored ones as requested."""
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
    merged = {key: list(values) for key, values in stored_tags.items()}
    for key, values in (requested_tags or {}).items():
        added = [value for value in dict.fromkeys(values) if value not in merged.get(key, [])]
        if added:
            merged[key] = merged.get(key, []) + added
    row["tags"] = [stored_tags, merged]
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
    return (LAB_WORKSPACE / path).read_bytes()  # absolute `path` wins the join


def _headed(notebook: bytes, title: str) -> str:
    """Returns the notebook as JSON text under a markdown heading naming
    `title`, which replaces a first cell of markdown and goes above one of code."""
    copy = json.loads(notebook)
    cells = copy["cells"]
    replaced = 1 if cells and cells[0]["cell_type"] == "markdown" else 0
    cells[:replaced] = [{"cell_type": "markdown", "id": "declared", "metadata": {}, "source": f"# {title}"}]
    return json.dumps(copy, indent=1)


def _declare(
    graph: list[Artifact],
    roots: list[str],
    root: Path,
    *,
    commit: bool,
    strict_commit: bool,
    verbose: bool,
    notebook: bytes | None,
) -> DeclarationReport:
    """Returns the report for a resolved graph whose asked-for paths are
    `roots`; with `commit`, writes it first, or raises DeclarationError
    before writing anything if a row blocks. Touches no volume."""
    memo: dict[str, Artifact] = {}
    rows = [_inspect(node, root, memo) for node in graph]
    report = DeclarationReport(rows, roots, strict_commit)
    if not commit:
        return report
    if report.blockers:
        raise DeclarationError(report, verbose)
    for node, row in zip(graph, rows):
        if row["state"] == "new":
            stamped = node.model_copy(
                update={
                    "declaring_commit": _head(),
                    "allocated_resources": Resources() if node.allocated_resources is None else node.allocated_resources,
                    "tags": node.tags or {},
                }
            )
            write_manifest(root, node.artifact_path, stamped.to_manifest())
            row.update(_inspect(node, root, memo), created=True)
            row["state"] = row["state"].removeprefix("already ")  # this commit declared it
        elif row["tags"][0] != row["tags"][1]:
            stored_tags, merged = row["tags"]
            on_disk = Artifact.load(node.artifact_path, root, memo)
            write_manifest(root, node.artifact_path, on_disk.model_copy(update={"tags": merged}).to_manifest())
            row.update(_inspect(node, root, memo), updated=True, tags=[stored_tags, merged])  # keep the pair
    if notebook is not None:
        for row in rows:
            if row["created"] and row["path"] in roots:
                (root / row["path"] / "declaring.ipynb").write_text(_headed(notebook, row["path"]))
    return report


def declare(
    artifact: Artifact,
    *,
    root: Path = STORAGE,
    commit: bool = False,
    strict_commit: bool = False,
    verbose: bool = False,
    notebook: bytes | None = None,
) -> DeclarationReport:
    """Returns the declaration report for `artifact` and everything it is
    built from, printed; with `commit`, missing manifests are published first,
    stamped with this process's `_head()` as their declaring commit, and an
    existing one gains the requested tags it lacks, nothing else in it touched.

    Preview writes nothing and returns blockers as rows. Commit refuses over
    any blocker -- a conflict, an undeclared folder, or drift under
    `strict_commit` -- and raises DeclarationError with the report attached
    before touching disk. `root` is
    for tests and scripts; a notebook gets STORAGE. The volume is reloaded
    before inspection and committed after writes only when this runs inside
    a container and `root` is its mount: anywhere else `/storage` is a plain
    folder with nothing behind it to reload.

    A commit that creates `artifact` also writes the declaring notebook to
    `<artifact_path>/declaring.ipynb` under a heading naming that path, which
    replaces a first cell of markdown and goes above one of code: the
    kernel's own as last saved, or the bytes handed in as `notebook` by a
    caller that has them and no kernel (`local.declare_on_volume`).

        lab.declare(Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(odyssey,)), commit=True)
        -> ... declared on artifact path:
           tokenizers/bpe-1000-d44bf6191c
    """
    on_volume = root == STORAGE and not modal.is_local()
    graph = resolve(artifact)
    if commit and notebook is None:
        notebook = current_notebook()
    if on_volume:
        refresh()
    report = _declare(
        graph, [artifact.artifact_path.as_posix()], root,
        commit=commit, strict_commit=strict_commit, verbose=verbose, notebook=notebook,
    )
    if commit and on_volume:
        save()
    print(report.render(verbose))
    return report


def create_experiment(
    experiment_uid: str,
    artifacts: list[Artifact],
    *,
    root: Path = STORAGE,
    commit: bool = False,
    strict_commit: bool = False,
    verbose: bool = False,
    notebook: bytes | None = None,
) -> DeclarationReport:
    """Returns the declaration report for an experiment's artifacts and
    everything they are built from, resolved together as one graph and
    printed; with `commit`, declares the graph as `declare` does, each path
    once, and creates `experiments/<experiment_uid>/` holding
    `declaration.ipynb`, the notebook declaring it, and `notebook.ipynb`,
    which calls `load_experiment`.

    Each artifact asked for is tagged `{"experiment": [experiment_uid]}`, on
    top of its own tags and, when it is already declared, the stored ones;
    what they are built from is not. A taken `experiment_uid` raises before
    anything else. A commit keeps the notebook it runs from, as last
    saved, and refuses one whose saved file lacks the running cell: save it
    first. The experiment's folder appears only once everything is written,
    so a failed commit leaves no experiment, and running it again completes
    it.

        lab.create_experiment("batch-size-study", runs, commit=True)  # declarations/experiment_template.ipynb's runs
        -> ... declared on artifact paths:
           runs/batch-size-study-bs32/pretraining/0-20000-960dd4bc7f
           runs/batch-size-study-bs64/pretraining/0-20000-2956000f6f
           runs/batch-size-study-bs128/pretraining/0-20000-08147cf8f3
           experiment created: experiments/batch-size-study
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_uid):
        raise ValueError(f"experiment_uid must be letters, digits, '.', '_' and '-', got {experiment_uid!r}")
    if not artifacts:
        raise ValueError("an experiment needs at least one artifact")
    on_volume = root == STORAGE and not modal.is_local()
    if on_volume:
        refresh()
    experiments = root / "experiments"
    final = experiments / experiment_uid
    if final.exists():
        raise FileExistsError(f"experiments/{experiment_uid} exists: pick another experiment_uid")
    roots = list(dict.fromkeys(artifact.artifact_path.as_posix() for artifact in artifacts))
    graph = []
    for node in resolve(*artifacts):
        if node.artifact_path.as_posix() in roots:
            tags = node.tags or {}
            members = list(dict.fromkeys([*tags.get("experiment", []), experiment_uid]))
            node = node.model_copy(update={"tags": {**tags, "experiment": members}})
        graph.append(node)
    if commit and notebook is None:
        notebook = current_notebook()
        if notebook is not None:  # a script a notebook started inherits its name, but has no kernel
            from IPython.core.getipython import get_ipython

            kernel = get_ipython()
        if notebook is None or kernel is None:
            raise ValueError("a committed experiment keeps the notebook declaring it: run this in one")
        running = kernel.history_manager.input_hist_raw[-1].strip()
        if running not in ["".join(cell["source"]).strip() for cell in json.loads(notebook)["cells"]]:
            raise ValueError("the notebook on disk is not the one running: save it, then run this cell again")
    options = {"commit": commit, "strict_commit": strict_commit, "verbose": verbose, "notebook": notebook}
    if commit:
        # Staged under a name the existence check above never matches, and
        # renamed into place last: the experiment exists only once complete.
        staging = experiments / f".staging-{experiment_uid}"
        shutil.rmtree(staging, ignore_errors=True)  # one writer, so a leftover is a dead attempt's
        staging.mkdir(parents=True)
        try:
            (staging / "declaration.ipynb").write_text(_headed(notebook, f"experiments/{experiment_uid}"))
            source = f"import lab\n\nartifacts = lab.load_experiment({experiment_uid!r})\nartifacts"
            tagged = {
                "cells": [{"cell_type": "code", "execution_count": None, "id": "tagged", "metadata": {}, "outputs": [], "source": source}],
                "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (staging / "notebook.ipynb").write_text(_headed(json.dumps(tagged).encode(), f"experiments/{experiment_uid}"))
            report = _declare(graph, roots, root, **options)
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if on_volume:
            save()
    else:
        report = _declare(graph, roots, root, **options)
    print(report.render(verbose))
    if commit:
        print(f"experiment created: experiments/{experiment_uid}")
    return report


def load_experiment(experiment_uid: str, root: Path = STORAGE) -> list[Artifact]:
    """Returns every artifact under `root` whose manifest is tagged with the
    experiment, by artifact path, unbound.

    Read off the manifests each call, the mounted volume reloaded first, so
    an artifact tagged after the experiment was created is listed too.

        lab.load_experiment("batch-size-study")  # declarations/experiment_template.ipynb's, created
        -> [Pretraining(run_id='batch-size-study-bs128', ...),
            Pretraining(run_id='batch-size-study-bs32', ...),
            Pretraining(run_id='batch-size-study-bs64', ...)]
    """
    if root == STORAGE and not modal.is_local():
        refresh()
    memo: dict[str, Artifact] = {}
    paths = sorted(manifest.parent.relative_to(root) for manifest in root.rglob(MANIFEST))
    return [
        Artifact.load(path, root, memo)
        for path in paths
        if experiment_uid in (read_manifest(root, path).get("tags") or {}).get("experiment", [])
    ]


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
