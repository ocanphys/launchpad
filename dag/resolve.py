"""Resolution, checking and declaration -- turning a requested artifact into a
plan, reconciling that plan against what's already on disk, and writing the
manifests for whatever isn't there yet (see spec.md, sections 6-7).

A plan is what this layer adds on top of an artifact: the same tree, with the
job that produces each node attached. It is never written. What lands on disk
is each artifact's own manifest, which names no job at all -- the job is derived
from the artifact's type, and the commit in the manifest pins the code it is.

Manifests are declared ahead of the work, for the whole graph, and are immutable
once written. So a manifest on disk that disagrees with the one being requested
is never a re-declaration; it means the history recorded there was edited, and
`Declaration.check` refuses. Changing parameters means a new run, and
`Declaration.write` only ever adds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dag.artifact import MANIFEST, Artifact
from dag.job import REGISTRY, Job
from datasets import job as _datasets_job  # noqa: F401
from mappeddatasets import job as _mappeddatasets_job  # noqa: F401
from models.mock import job as _mock_job  # noqa: F401

# Importing every concrete family below is what populates REGISTRY/ARTIFACTS
# (via Artifact.__init_subclass__/Job.__init_subclass__) -- this module is
# the one thing everything else (main.py, visualizer.py, notebooks) already
# imports to get a plan, so it's the natural place for that side effect
# rather than requiring every caller to import each family module by hand.
from sources import job as _sources_job  # noqa: F401
from tokenizers import bpe as _tokenizers_bpe  # noqa: F401

Status = Literal["new", "declared", "partial", "done", "conflict", "undeclared"]

_BLOCKING = ("conflict", "undeclared")


def producer_for(artifact: Artifact) -> Job:
    return REGISTRY[type(artifact)](artifact)


def resolve(artifact: Artifact, stack: tuple[Artifact, ...] = ()) -> dict:
    """Build the plan -- the dependency tree with each node's producing job
    attached -- needed to produce `artifact`.

    `stack` is the chain of artifacts encountered while recursing down the
    tree, used only to detect cycles: if we hit an artifact already on the
    stack, we're stuck in a loop.
    """
    if artifact in stack:
        raise ValueError(f"cycle: {artifact} already on the resolution stack")
    job = producer_for(artifact)
    return {
        "artifact": artifact,
        "job": job,
        "dependencies": [resolve(dep, stack + (artifact,)) for dep in artifact.deps()],
    }


def job_list(plan: dict) -> list[Job]:
    order: list[Job] = []
    seen: set[Path] = set()  # dedup key: the folder, since that IS the identity

    def visit(node: dict) -> None:
        for child in node["dependencies"]:
            visit(child)  # dependencies before dependents -- post-order DFS
        key = node["artifact"].artifact_path
        if key not in seen:
            seen.add(key)
            order.append(node["job"])

    visit(plan)
    return order


class _Unreadable:
    """A manifest that's there but can't be loaded -- an artifact type this code
    no longer defines, or a file that isn't the manifest it claims to be. Still
    something somebody declared, so it isn't absence, and it reads as conflict."""


_UNREADABLE = _Unreadable()


def recorded(artifact: Artifact, root: Path) -> Artifact | _Unreadable | None:
    """Whatever the manifest in this artifact's folder describes, or None when
    there isn't one."""
    path = root / artifact.artifact_path / MANIFEST
    if not path.exists():
        return None
    try:
        return Artifact.load(path)
    except (KeyError, TypeError, ValueError):
        # an artifact type this code no longer defines, a field it no longer
        # takes, or not valid JSON at all -- all of it "something is declared
        # here and we can't tell what", which is the caller's problem, not ours
        return _UNREADABLE


InspectCache = dict[Path, tuple[Status, bool]]


def inspect(
    artifact: Artifact, root: Path, cache: InspectCache | None = None
) -> tuple[Status, bool]:
    """What the disk says about one artifact, and whether what's declared there
    was declared under a different commit than the one being asked for.

    new         nothing there; `write` will declare it
    declared    manifest agrees, no outputs yet
    partial     manifest agrees, some outputs
    done        manifest agrees, every output present
    conflict    a manifest is there and describes something else
    undeclared  outputs are there with no manifest saying who asked for them

    "outputs" before a manifest exists means this artifact's *own* files --
    that check is "did something write into my folder before I was asked
    for", which only makes sense about a folder that's actually mine. Once a
    manifest agrees, done/partial switches to `completion_paths(root)`,
    which for almost every type is the same files -- except a virtual
    artifact like MappedDataSet, whose completion is its dependencies'
    files. Checking those pre-manifest too would make a fresh MappedDataSet
    over already-tokenized sources read `undeclared` the instant it's
    constructed, which is exactly the reuse case it exists for, not an
    anomaly.

    `cache`, when given, remembers each artifact_path's result for the
    caller's whole traversal rather than just this one call. A shared
    dependency (one tokenizer behind a dozen datasets, one source behind a
    dozen tokenizers) gets asked about once per *occurrence* as someone's
    dependency, not once per distinct artifact -- without a cache, every
    one of those repeats re-reads and re-parses that artifact's
    manifest.json (which, being a full nested tree per `Artifact.manifest`,
    can itself be sizeable) from scratch. Keyed by artifact_path rather
    than the Artifact object: two instances describing the same on-disk
    thing already compare equal, but keying on the object would still hash
    and __eq__ down through the whole nested tree on every lookup, which is
    exactly the redundant work being avoided. Absent (the default) for
    call sites that only ever inspect each artifact once, like
    `Declaration.check`.
    """
    if cache is not None and artifact.artifact_path in cache:
        return cache[artifact.artifact_path]

    on_disk = recorded(artifact, root)
    own_present = [path for path in artifact.paths(root).values() if path.exists()]
    if on_disk is None:
        result = ("undeclared" if own_present else "new"), False
    elif isinstance(on_disk, _Unreadable) or on_disk != artifact:
        result = "conflict", False  # == ignores commit: a parameter disagreement
    else:
        drift = on_disk.commit != artifact.commit
        complete = artifact.completion_paths(root)
        present = [path for path in complete if path.exists()]
        done_or_partial = "done" if len(present) == len(complete) else ("partial" if present else "declared")
        result = done_or_partial, drift

    if cache is not None:
        cache[artifact.artifact_path] = result
    return result


def status(artifact: Artifact, root: Path, cache: InspectCache | None = None) -> Status:
    return inspect(artifact, root, cache)[0]


def _walk_deps(artifacts: list[Artifact], seen: dict[Path, Artifact]) -> None:
    """Depth-first through every dependency reachable from `artifacts`,
    adding each newly-seen one to `seen` (keyed by artifact_path) and
    recursing into it. `seen` starting non-empty is what lets a caller
    decide whether the roots themselves count as "seen" -- `declared_under`
    pre-seeds it with its own manifests, `dependency_closure` doesn't.
    """
    for artifact in artifacts:
        for dep in artifact.deps():
            if dep.artifact_path in seen:
                continue  # also what keeps this cycle-safe: a repeat is just skipped
            seen[dep.artifact_path] = dep
            _walk_deps([dep], seen)


def declared_under(run_id: str, root: Path) -> list[Artifact]:
    """Every artifact declared for `run_id`: its own manifests under
    root/runs/run_id, plus every dependency reachable from them (which may
    live under a shared root, e.g. tokenizers/ or sources/), deduplicated by
    identity (artifact_path).

    Unlike `resolve()`, this doesn't start from one requested artifact and
    walk down a registry-derived plan -- it starts from whatever manifests
    actually exist and reads their embedded dependency trees back out. A
    run's declared state is however many manifests were written, not one
    tree top-down.
    """
    manifests = sorted((root / "runs" / run_id).rglob(MANIFEST))
    roots = [Artifact.load(path) for path in manifests]
    seen: dict[Path, Artifact] = {a.artifact_path: a for a in roots}
    _walk_deps(roots, seen)
    return list(seen.values())


def dependency_closure(artifact: Artifact) -> list[Artifact]:
    """Every artifact `artifact` depends on, directly or transitively,
    deduplicated by artifact_path -- the same walk `declared_under` does
    starting from a run's manifests, but starting from one already-loaded
    artifact instead, and not including `artifact` itself (unlike
    `declared_under`, whose roots are part of what it returns).

    Needs no `root` -- `deps()` reads straight off each artifact's own
    fields (nested manifests already loaded, for anything reached via
    `Artifact.load`), never the filesystem.
    """
    seen: dict[Path, Artifact] = {}
    _walk_deps([artifact], seen)
    return list(seen.values())


def declared_of_kind(folder: str, root: Path) -> list[Artifact]:
    """Every artifact declared directly under `root/folder` -- e.g.
    root/sources or root/datasets -- one level deep, straight off each
    manifest.json there.

    The shared-folder counterpart to `declared_under`: a run's folder holds
    only that run's own top nodes and needs a dependency walk to find the
    rest of what it uses, but a shared kind's folder (sources/, datasets/,
    mappeddatasets/, tokenizers/) already *is* the complete membership list
    for that kind -- every instance that has ever been declared gets a
    subfolder here, attached to a run or not. No recursion into
    dependencies, and no dedup-by-walk needed either: one subfolder is one
    artifact, by construction.
    """
    folder_root = root / folder
    if not folder_root.exists():
        return []
    return [Artifact.load(path) for path in sorted(folder_root.glob(f"*/{MANIFEST}"))]


@dataclass(frozen=True)
class Row:
    artifact: Artifact
    status: Status
    drift: bool  # declared under a different commit than the one being requested

    def __str__(self) -> str:
        mark = " (drift)" if self.drift else ""
        return f"{self.status:11}{self.artifact.artifact_path}{mark}"


def _run_id_of(artifact: Artifact) -> str | None:
    """The run an artifact belongs to, or None if it's shared across runs --
    read off the fields, so nothing declares its own scope."""
    return getattr(artifact, "run_id", None)


@dataclass(repr=False)  # repr=False: the report reads better, and a generated
class Declaration:  # one would dump the whole plan into a notebook
    """A request against a particular root -- what `check()` and `write()` act on.

    check()   resolve `artifact` and reconcile the whole plan against `root`;
              show what's consistent and what isn't. Touches disk, writes
              nothing, safe to call again any time disk state may have changed.
    write()   check(), then declare every `new` artifact, dependencies first.
              Refuses outright if anything is inconsistent -- a partial
              declaration over a disputed tree is worse than none.

    strict_commit turns commit drift from a note into a refusal. Off by
    default: a tree built by more than one version of the code is normal, and
    usually fine.
    """

    artifact: Artifact
    root: Path
    strict_commit: bool = False
    plan: dict | None = field(init=False, default=None, repr=False)
    rows: tuple[Row, ...] = field(init=False, default=(), repr=False)

    @property
    def run_id(self) -> str | None:
        return _run_id_of(self.artifact)

    @property
    def problems(self) -> list[Row]:
        return [
            row
            for row in self.rows
            if row.status in _BLOCKING or (row.drift and self.strict_commit)
        ]

    @property
    def ok(self) -> bool:
        return not self.problems

    def check(self) -> Declaration:
        self.plan = resolve(self.artifact)
        artifacts = [job.artifact for job in job_list(self.plan)]  # deduped
        self.rows = tuple(Row(a, *inspect(a, self.root)) for a in artifacts)
        return self

    def write(self) -> list[Path]:
        self.check()
        if not self.ok:
            raise ValueError(
                "refusing to declare over an inconsistent tree:\n"
                + "\n".join(str(r) for r in self.problems)
            )
        written = []
        for row in self.rows:  # dependency order: an interrupted write still
            if row.status != "new":  # leaves every manifest's ancestry declared
                continue
            path = self.root / row.artifact.artifact_path / MANIFEST
            path.parent.mkdir(parents=True, exist_ok=True)
            body = json.dumps(row.artifact.manifest(), indent=2)
            try:
                with path.open("x") as handle:  # exclusive: whoever loses, notices
                    handle.write(body)
            except FileExistsError:
                # someone declared this between the check and now. Identical
                # bytes mean they declared the same thing, which is no problem
                # at all -- manifests are canonical, so equality is byte equality.
                if path.read_text() != body:
                    raise ValueError(f"{path} was declared as something else")
                continue
            written.append(path)
        # self.rows above is a pre-write snapshot -- every row just written
        # still reads "new" on it. Re-check so the report (and .rows/.ok for
        # anything reading the object directly afterward) reflects what's
        # actually on disk now, not what was true before this call started.
        self.check()
        return written

    def __str__(self) -> str:
        """The report: every artifact the request needs, in the order work would
        happen, with what the disk says about each."""
        if self.plan is None:
            return f"run {self.run_id or '-'} under {self.root} (not checked)"
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        drifted = sum(row.drift for row in self.rows)
        summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        if drifted:
            summary += f", {drifted} at another commit"
        verdict = (
            f"ok -- {counts.get('new', 0)} to declare"
            if self.ok
            else f"BLOCKED -- {len(self.problems)} to resolve"
        )
        head = f"run {self.run_id or '-'} under {self.root}"
        return "\n".join(
            [
                head,
                *(f"  {row}" for row in self.rows),
                "",
                summary,
                verdict,
            ]
        )

    __repr__ = __str__  # so a cell ending in .check() shows the report
