"""Artifact definitions -- what each thing IS and where it lives. No job,
registry or resolver knowledge belongs here (see spec.md, section 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)  # frozen: identity is its fields, so it must be hashable/immutable
class Artifact:
    def refs(self) -> list["Artifact"]:
        return []  # artifact-valued parameters, one level deep; none by default

    def relpath(self) -> Path:
        raise NotImplementedError

    def path(self, root: Path) -> Path:
        return root / self.relpath()

    def exists(self, root: Path) -> bool:
        return self.path(root).exists()


@dataclass(frozen=True)
class Source(Artifact):
    name: str
    url: str

    def relpath(self) -> Path:
        return Path(self.name) / "body.txt"


@dataclass(frozen=True)
class CombinedSource(Artifact):
    sources: tuple[Source, Source]  # exactly two, by this toy's contract

    def refs(self) -> list[Artifact]:
        return list(self.sources)

    def relpath(self) -> Path:
        uid = "-".join(s.name for s in self.sources)  # readable only, no digest -- toy simplification
        return Path(uid) / "combined.txt"
