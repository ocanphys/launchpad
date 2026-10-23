"""Pure dependency resolution: the artifacts a request is built from, in the
order they have to exist. Reads no storage and decides nothing about running.
"""

from __future__ import annotations

from pathlib import Path

from artifacts.core.artifact import Artifact


def resolve(artifact: Artifact) -> list[Artifact]:
    """Every artifact `artifact` is built from and then itself, dependencies
    before dependents, each path once.

    A path reached twice must hold an agreeing definition, and `==` compares
    the whole subtree, so a shared path cannot hide a conflicting leaf. The
    first object reaching a path is the one kept, recorded fields and all.
    """
    visited: dict[Path, Artifact] = {}
    active: list[Path] = []  # the current descent, for cycle detection

    def visit(node: Artifact) -> None:
        path = node.artifact_path
        if path in visited:
            if visited[path] != node:
                raise ValueError(f"different definitions at {path}")
            return
        if path in active:
            chain = " -> ".join(str(p) for p in (*active, path))
            raise ValueError(f"cycle: {chain}")
        active.append(path)
        for dep in node.deps():
            visit(dep)
        active.pop()
        visited[path] = node

    visit(artifact)
    return list(visited.values())
