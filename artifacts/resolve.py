"""Resolution and scheduling logic -- turning a requested artifact into a
manifest, a manifest into an ordered job list, and running whatever in that
list isn't done yet (see spec.md, sections 6-7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from artifact import Artifact
from job import REGISTRY, Job

Status = Literal["done", "runnable", "blocked"]


def producer_for(artifact: Artifact) -> Job:
    return REGISTRY[type(artifact)].for_artifact(artifact)


def resolve(artifact: Artifact, stack: tuple[Artifact, ...] = ()) -> dict:
    if artifact in stack:
        raise ValueError(f"cycle: {artifact} already on the resolution stack")
    job = producer_for(artifact)
    return {
        "artifact": artifact,
        "job": job,
        "outputs": job.outputs,
        "inputs": [resolve(a, stack + (artifact,)) for a in job.inputs],
    }


def job_list(manifest: dict) -> list[Job]:
    order: list[Job] = []
    seen: set[tuple] = set()  # dedup key: the job's output paths, since outputs define a job's identity

    def visit(node: dict) -> None:
        for child in node["inputs"]:
            visit(child)  # dependencies before dependents -- post-order DFS
        key = tuple(out.relpath() for out in node["outputs"])
        if key not in seen:
            seen.add(key)
            order.append(node["job"])

    visit(manifest)
    return order


def status(job: Job, root: Path) -> Status:
    if all(out.exists(root) for out in job.outputs):
        return "done"
    if all(inp.exists(root) for inp in job.inputs):
        return "runnable"
    return "blocked"


def run_all(artifact: Artifact, root: Path) -> None:
    manifest = resolve(artifact)
    for job in job_list(manifest):
        if status(job, root) == "done":
            print(f"skip  {job.outputs[0].relpath()}  (already done)")
            continue
        job.run(root)
        print(f"run   {job.outputs[0].relpath()}")
