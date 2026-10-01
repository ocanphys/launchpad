"""Pure dependency resolution: the artifacts a request is built from, in the
order they have to exist. Reads no storage and decides nothing about running.
"""

from __future__ import annotations

from pathlib import Path

from artifacts.core.artifact import Artifact
from artifacts.core.sweep import Declaration, expand


def resolve(declaration: Declaration) -> list[Artifact]:
    """Returns the combined dependency graph of every artifact `declaration`
    expands to: dependencies before dependents, each path once.

    A path reached twice must hold an agreeing definition, and `==` compares
    the whole subtree, so a shared path cannot hide a conflicting leaf. The
    first object reaching a path is the one kept, recorded fields and all.
    """
    visited: dict[Path, tuple[Artifact, str]] = {}  # each path's kept node and where it came from
    active: list[Path] = []  # the current descent, for cycle detection

    def visit(node: Artifact, location: str) -> None:
        path = node.artifact_path
        if path in visited:
            kept, origin = visited[path]
            if kept != node:
                raise ValueError(f"different definitions at {path}: {origin} and {location}")
            return
        if path in active:
            chain = " -> ".join(str(p) for p in (*active, path))
            raise ValueError(f"cycle: {chain}")
        active.append(path)
        for dep in node.deps():
            visit(dep, f"{location} -> {dep.artifact_path}")
        active.pop()
        visited[path] = (node, location)

    for i, root in enumerate(expand(declaration)):
        visit(root, f"roots[{i}]")
    return [node for node, _ in visited.values()]
