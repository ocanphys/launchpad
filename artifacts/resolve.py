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
`check` refuses. Changing parameters means a new run, and `write` only ever adds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from artifact import MANIFEST, Artifact
from job import REGISTRY, Job

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


def inspect(artifact: Artifact, root: Path) -> tuple[Status, bool]:
    """What the disk says about one artifact, and whether what's declared there
    was declared under a different commit than the one being asked for.

    new         nothing there; `write` will declare it
    declared    manifest agrees, no outputs yet
    partial     manifest agrees, some outputs
    done        manifest agrees, every output present
    conflict    a manifest is there and describes something else
    undeclared  outputs are there with no manifest saying who asked for them
    """
    on_disk = recorded(artifact, root)
    present = [path for path in artifact.paths(root).values() if path.exists()]
    if on_disk is None:
        return ("undeclared" if present else "new"), False
    if isinstance(on_disk, _Unreadable) or on_disk != artifact:
        return "conflict", False  # == ignores commit: a parameter disagreement
    drift = on_disk.commit != artifact.commit
    if len(present) == len(artifact.files):
        return "done", drift
    return ("partial" if present else "declared"), drift


def status(artifact: Artifact, root: Path) -> Status:
    return inspect(artifact, root)[0]


@dataclass(frozen=True)
class Row:
    artifact: Artifact
    status: Status
    drift: bool  # declared under a different commit than the one being requested

    def __str__(self) -> str:
        mark = " (drift)" if self.drift else ""
        return f"{self.status:11}{self.artifact.artifact_path}{mark}"


@dataclass(frozen=True, repr=False)  # repr=False: the report reads better, and
class Declaration:  # a generated one would dump the whole plan into a notebook
    """What a request looks like against a particular root, and the means to
    write down whatever of it isn't there yet."""

    plan: dict
    rows: tuple[Row, ...]  # dependency order -- the order work would happen in
    root: Path
    run_id: str | None
    strict_commit: bool

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

    def write(self) -> list[Path]:
        """Declare every `new` artifact, dependencies first, and return the
        manifests written. Refuses outright if anything is inconsistent -- a
        partial declaration over a disputed tree is worse than none."""
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
        return written

    def __str__(self) -> str:
        """The report: every artifact the request needs, in the order work would
        happen, with what the disk says about each."""
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

    __repr__ = __str__  # so a cell ending in check(...) shows the report


def _run_id_of(artifact: Artifact) -> str | None:
    """The run an artifact belongs to, or None if it's shared across runs --
    read off the fields, so nothing declares its own scope."""
    return getattr(artifact, "run_id", None)


def check(
    artifact: Artifact,
    root: Path,
    strict_commit: bool = False,
) -> Declaration:
    """Resolve `artifact` and reconcile the whole plan against `root`.

    Reports every artifact the request needs and what the disk says about each.
    Nothing is written; the returned Declaration does that, and only if nothing
    is inconsistent.

    strict_commit turns commit drift from a note into a refusal. Off by default:
    a tree built by more than one version of the code is normal, and usually
    fine.
    """
    plan = resolve(artifact)
    artifacts = [job.artifact for job in job_list(plan)]  # deduped, dependency order

    rows = []
    for a in artifacts:
        state, drift = inspect(a, root)
        rows.append(Row(artifact=a, status=state, drift=drift))

    return Declaration(
        plan=plan,
        rows=tuple(rows),
        root=root,
        run_id=_run_id_of(artifact),
        strict_commit=strict_commit,
    )


def declare(
    artifact: Artifact,
    root: Path,
    strict_commit: bool = False,
) -> Declaration:
    """check, then write. The Declaration comes back either way -- print it to
    see what was already there."""
    declaration = check(artifact, root, strict_commit=strict_commit)
    declaration.write()
    return declaration


def run_all(artifact: Artifact, root: Path) -> Declaration:
    """Declare the graph, then run everything in it that isn't done. The local
    executor: the real one reads manifests off disk and launches jobs whose
    dependencies are satisfied, but it decides what to run the same way."""
    declaration = declare(artifact, root)
    for job in job_list(declaration.plan):
        state = status(job.artifact, root)
        if state == "done":
            print(f"skip  {job.artifact.artifact_path}  (already done)")
            continue
        job.run(root)
        note = "  (was partial -- redone)" if state == "partial" else ""
        print(f"run   {job.artifact.artifact_path}{note}")
    return declaration
