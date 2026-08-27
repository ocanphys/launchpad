"""Resolution and scheduling logic -- turning a requested artifact into a
manifest, a manifest into an ordered job list, and running whatever in that
list isn't done yet (see spec.md, sections 6-7).

The manifest is also what lands on disk: before a job runs, its artifact's
own manifest is written into that artifact's folder, so every folder in the
tree carries the recipe that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from artifact import MANIFEST, Artifact
from job import REGISTRY, Job

Status = Literal["done", "started", "runnable", "blocked"]


def producer_for(artifact: Artifact) -> Job:
    return REGISTRY[type(artifact)](artifact)


def resolve(artifact: Artifact, stack: tuple[Artifact, ...] = ()) -> dict:
    """Build the manifest -- a dict recipe describing the complete dependency
    graph -- needed to produce `artifact`.

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


def manifest_json(node: dict) -> dict:
    """The same manifest with objects replaced by what identifies them -- an
    artifact by the folder it owns, a job by its class name -- so it can be
    written into that folder and read back without importing anything."""
    artifact = node["artifact"]
    return {
        "artifact": type(artifact).__name__,
        "uid": artifact.uid,
        "artifact_path": str(artifact.artifact_path),
        "files": artifact.files,
        "job": type(node["job"]).__name__,
        "dependencies": [manifest_json(child) for child in node["dependencies"]],
    }


def job_list(manifest: dict) -> list[Job]:
    order: list[Job] = []
    seen: set[Path] = set()  # dedup key: the folder, since that IS the identity

    def visit(node: dict) -> None:
        for child in node["dependencies"]:
            visit(child)  # dependencies before dependents -- post-order DFS
        key = node["artifact"].artifact_path
        if key not in seen:
            seen.add(key)
            order.append(node["job"])

    visit(manifest)
    return order


def status(job: Job, root: Path) -> Status:
    if job.artifact.exists(root):
        return "done"
    if (root / job.artifact.artifact_path / MANIFEST).exists():
        return "started"  # manifest written but files incomplete -- interrupted
    if all(dep.exists(root) for dep in job.artifact.deps()):
        return "runnable"
    return "blocked"


def run_all(artifact: Artifact, root: Path) -> None:
    for job in job_list(resolve(artifact)):
        folder = root / job.artifact.artifact_path
        state = status(job, root)
        if state == "done":
            print(f"skip  {job.artifact.artifact_path}  (already done)")
            continue
        folder.mkdir(parents=True, exist_ok=True)
        (folder / MANIFEST).write_text(
            json.dumps(manifest_json(resolve(job.artifact)), indent=2)
        )  # the manifest lands first, so an interrupted job leaves a record
        job.run(root)
        note = "  (was interrupted -- redone)" if state == "started" else ""
        print(f"run   {job.artifact.artifact_path}{note}")
