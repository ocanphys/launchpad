from dataclasses import dataclass
from pathlib import Path

from dag.artifact import Artifact


@dataclass(frozen=True)
class Source(Artifact):
    name: str
    url: str

    @property
    def uid(self) -> str:
        return self.name

    @property
    def artifact_path(self) -> Path:
        return Path("sources") / self.name

    @property
    def files(self) -> dict[str, str]:
        return {"raw text": "body.txt"}
