"""Job definitions -- each job imports the artifact type(s) it produces and
registers itself as their producer (see spec.md, sections 4-5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from artifact import Artifact, CombinedSource, Source

REGISTRY: dict[type[Artifact], type["Job"]] = {}  # artifact type -> the job that produces it


class Job(ABC):
    produces: type[Artifact]  # declared by each subclass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.produces in REGISTRY:
            raise TypeError(f"{cls.produces} already registered to {REGISTRY[cls.produces]}")
        REGISTRY[cls.produces] = cls  # registration happens at class-definition time, not lookup time

    @classmethod
    @abstractmethod
    def for_output(cls, artifact: Artifact) -> "Job":
        """Reconstruct the job that would produce this one requested artifact."""

    @property
    @abstractmethod
    def outputs(self) -> list[Artifact]:
        """The full output set this job produces (usually just one artifact)."""

    @property
    def inputs(self) -> list[Artifact]:
        return [ref for out in self.outputs for ref in out.refs()]  # derived, not declared

    @abstractmethod
    def run(self, root: Path) -> None:
        """Do the work, writing self.outputs under root."""


class SourceJob(Job):
    produces = Source

    def __init__(self, artifact: Source):
        self.artifact = artifact

    @classmethod
    def for_output(cls, artifact: Source) -> "SourceJob":
        return cls(artifact)

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        path = self.artifact.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"[{self.artifact.name}] fetched from {self.artifact.url}\n")  # stand-in for a real download


class CombineJob(Job):
    produces = CombinedSource

    def __init__(self, artifact: CombinedSource):
        self.artifact = artifact

    @classmethod
    def for_output(cls, artifact: CombinedSource) -> "CombineJob":
        return cls(artifact)

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        bodies = [source.path(root).read_text() for source in self.artifact.sources]
        path = self.artifact.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(bodies))
