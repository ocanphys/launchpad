"""Resolve artifact dependencies, inspect disk state, and declare manifests."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from artifacts.core.artifact import MANIFEST, Artifact

Status = Literal["new", "declared", "partial", "done", "conflict", "undeclared"]

# Statuses that mean a human has to look before anything is written or run.
BLOCKING: tuple[Status, ...] = ("conflict", "undeclared")


@dataclass(frozen=True)
class Node:
    """An artifact and its dependency paths, with optional disk status."""

    artifact: Artifact
    deps: tuple[Path, ...]
    status: Status | None = None  # None when resolved without a target
    drift: bool = False  # declared under a different commit than requested
    recorded: Artifact | None = None  # retained only for parseable conflicts
    recorded_commit: str | None = None

    @property
    def path(self) -> Path:
        return self.artifact.artifact_path

    def __str__(self) -> str:
        mark = " (drift)" if self.drift else ""
        return f"{self.status or 'unchecked':11}{self.path}{mark}"


@dataclass
class Dag:
    """Nodes deduplicated by path and ordered dependencies first.

    Disk statuses are a snapshot taken by `resolve`."""

    nodes: dict[Path, Node]
    roots: tuple[Path, ...]
    target: Path | None = None
    strict_commit: bool = False  # treat drift as a blocker
    verbose: bool = False  # include blocker reasons in reports and refusals

    def __getitem__(self, path: Path | str) -> Node:
        return self.nodes[Path(path)]

    def __contains__(self, path: object) -> bool:
        return Path(path) in self.nodes if isinstance(path, (Path, str)) else False

    def __iter__(self) -> Iterator[Node]:
        """Iterate over nodes, dependencies first."""
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    def blocked_by(self, path: Path | str) -> list[Path]:
        """Direct dependencies that are not done."""
        return [dep for dep in self[path].deps if self[dep].status != "done"]

    def closure(self, path: Path | str) -> list[Path]:
        """Transitive dependency paths in graph order, excluding the artifact itself."""
        seen: set[Path] = set()

        def walk(at: Path) -> None:
            for dep in self[at].deps:
                if dep not in seen:
                    seen.add(dep)
                    walk(dep)

        walk(Path(path))
        return [node.path for node in self if node.path in seen]

    @property
    def run_ids(self) -> list[str]:
        """Sorted run IDs from the roots; shared artifacts contribute no run ID."""
        found = {
            run_id
            for path in self.roots
            if (run_id := getattr(self[path].artifact, "run_id", None))
        }
        return sorted(found)

    @property
    def problems(self) -> list[Node]:
        return [
            node
            for node in self
            if node.status in BLOCKING or (node.drift and self.strict_commit)
        ]

    @property
    def ok(self) -> bool:
        return not self.problems

    def reasons(self, node: Node) -> list[str]:
        """Explain a blocking status or strict commit drift."""
        if node.status == "conflict":
            return conflict_diff(node)
        if node.status == "undeclared" and self.target is not None:
            return [
                f"undeclared file present: {path}"
                for path in node.artifact.paths(self.target).values()
                if path.exists()
            ]
        if node.drift and self.strict_commit:
            return [
                (
                    f"declared under commit {node.recorded_commit}, "
                    f"requested {node.artifact.commit}"
                )
            ]
        return []

    def explain(self) -> list[str]:
        """Describe every blocking node, regardless of `verbose`."""
        lines: list[str] = []
        for node in self.problems:
            lines.append(str(node))
            lines.extend(f"    {line}" for line in self.reasons(node))
        return lines

    def __str__(self) -> str:
        """Report statuses, counts, and blockers; include reasons when verbose."""
        head = f"run {', '.join(self.run_ids) or '-'} under {self.target or '-'}"
        if self.target is None:
            return f"{head} ({len(self)} artifacts, not checked)"

        rows: list[str] = []
        blocking = {node.path for node in self.problems}
        for node in self:
            rows.append(f"  {node}")
            if self.verbose and node.path in blocking:
                rows.extend(f"      {line}" for line in self.reasons(node))

        counts = Counter(node.status for node in self)
        summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        drifted = sum(node.drift for node in self)
        if drifted:
            summary += f", {drifted} at another commit"
        if blocking:
            verdict = f"BLOCKED -- {len(blocking)} to resolve"
            if not self.verbose:
                verdict += " (resolve(..., verbose=True) to see why)"
        else:
            verdict = f"ok -- {counts['new']} to declare"
        return "\n".join([head, *rows, "", summary, verdict])

    __repr__ = __str__  # so a cell ending in resolve() shows the report


def _check(
    artifact: Artifact, target: Path
) -> tuple[Status, bool, Artifact | None, str | None]:
    """Return status, commit drift, conflicting artifact, and recorded commit.

    Without a manifest, own outputs mean undeclared; otherwise the artifact is
    new. A mismatched or invalid manifest means conflict. Matching manifests
    are declared, partial, or done according to completion paths."""
    manifest = target / artifact.artifact_path / MANIFEST
    if not manifest.exists():
        own = any(path.exists() for path in artifact.paths(target).values())
        return ("undeclared" if own else "new"), False, None, None

    try:
        recorded = Artifact.load(manifest)
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        # Invalid JSON or an artifact schema this code cannot reconstruct.
        return "conflict", False, None, None

    if recorded != artifact:
        # == ignores commit: this is a parameter disagreement
        return "conflict", recorded.commit != artifact.commit, recorded, recorded.commit

    complete = artifact.completion_paths(target)
    present = sum(path.exists() for path in complete)
    status: Status = (
        "done" if present == len(complete) else ("partial" if present else "declared")
    )
    return status, recorded.commit != artifact.commit, None, recorded.commit


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten a manifest to dotted paths and list indices, retaining empty containers."""
    if isinstance(value, dict) and value:
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, item in enumerate(value):
            out.update(_leaves(item, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def conflict_diff(node: Node) -> list[str]:
    """Describe manifest differences, excluding commits; empty unless conflicting."""
    if node.status != "conflict":
        return []
    if node.recorded is None:
        return ["manifest on disk could not be read as an artifact at all"]

    on_disk = _leaves(node.recorded.manifest())
    wanted = _leaves(node.artifact.manifest())
    lines = []
    for key in sorted(set(on_disk) | set(wanted)):
        if key == "commit" or key.endswith(".commit"):
            continue  # never identity, and drift is already reported on its own
        was, now = on_disk.get(key, _MISSING), wanted.get(key, _MISSING)
        if was != now:
            lines.append(f"{key}: on disk {_show(was)}, requested {_show(now)}")
    return lines


_MISSING = object()


def _show(value: object) -> str:
    return "(absent)" if value is _MISSING else json.dumps(value)


def resolve(
    *artifacts: Artifact,
    target: Path | None = None,
    strict_commit: bool = False,
    verbose: bool = False,
) -> Dag:
    """Resolve roots in dependency order, deduplicating by artifact path.

    Cycles raise ValueError. With a target, inspect each node on disk.
    Otherwise statuses are None. Strict commit drift blocks declaration;
    verbose includes blocker details in reports and declaration errors."""
    nodes: dict[Path, Node] = {}
    resolving: list[Path] = []  # the current descent, for cycle detection

    def visit(artifact: Artifact) -> Path:
        path = artifact.artifact_path
        if path in nodes:  # already fully resolved by another route
            if nodes[path].artifact != artifact:
                raise ValueError(f"different artifacts requested at {path}")
            return path
        if path in resolving:
            chain = " -> ".join(str(p) for p in (*resolving, path))
            raise ValueError(f"cycle: {chain}")
        resolving.append(path)
        deps = tuple(visit(dep) for dep in artifact.deps())
        resolving.pop()
        checked = (
            _check(artifact, target)
            if target is not None
            else (None, False, None, None)
        )
        nodes[path] = Node(artifact, deps, *checked)
        return path

    roots = tuple(visit(artifact) for artifact in artifacts)
    return Dag(nodes, roots, target, strict_commit, verbose)


def plan(dag: Dag, *, pending_only: bool = False) -> list[Artifact]:
    """Return artifacts in dependency order, optionally excluding completed nodes."""
    return [node.artifact for node in dag if not pending_only or node.status != "done"]


def declared(target: Path, prefix: str = "", *, deep: bool = False) -> list[Artifact]:
    """Load manifests one directory below target/prefix, or recursively when deep."""
    base = target / prefix
    if not base.exists():
        return []
    pattern = f"**/{MANIFEST}" if deep else f"*/{MANIFEST}"
    return [Artifact.load(path) for path in sorted(base.glob(pattern))]


def declare(dag: Dag) -> list[Path]:
    """Write manifests for new nodes in dependency order; return written paths.

    Requires a target and no known blockers. Exclusive creation compares bytes
    if a manifest already exists. A race or I/O error can leave earlier writes
    in place. The graph remains unchanged; resolve again to refresh statuses."""
    if dag.target is None:
        raise ValueError(
            "nothing to declare against -- resolve(artifact, target=root) first"
        )
    if not dag.ok:
        raise ValueError(
            "refusing to declare over an inconsistent tree:\n"
            + "\n".join(
                dag.explain() if dag.verbose else [str(n) for n in dag.problems]
            )
        )

    written: list[Path] = []
    for node in dag:
        if node.status != "new":
            continue
        path = dag.target / node.path / MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(node.artifact.manifest(), indent=2)
        try:
            with path.open("x") as handle:  # exclusive: whoever loses, notices
                handle.write(body)
        except FileExistsError:
            if path.read_text() != body:
                raise ValueError(f"{path} was declared as something else")
            continue
        written.append(path)
    return written
