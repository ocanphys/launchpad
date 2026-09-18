from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact


@dataclass(frozen=True)
class Source(Artifact):
    """A named body of raw text at sources/<name>/body.txt. Subclasses say
    where the text comes from; everything downstream reads only the file."""

    name: str

    @property
    def uid(self) -> str:
        return self.name

    @property
    def artifact_path(self) -> Path:
        return Path("sources") / self.name

    @property
    def files(self) -> dict[str, str]:
        return {"raw text": "body.txt"}


@dataclass(frozen=True)
class SourceURL(Source):
    producer: ClassVar[str] = "artifacts.sources.jobs.SourceURLJob"

    url: str

