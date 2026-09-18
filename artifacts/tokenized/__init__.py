"""One source run through one tokenizer: tokens.bin, raw uint16 ids. Its own
family because it belongs to neither side -- a tokenizer family does not own
what its tokenizers are run over, and a dataset only consumes these.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact
from artifacts.sources import Source
from artifacts.tokenizers import Tokenizer


@dataclass(frozen=True)
class TokenizedSource(Artifact):
    producer: ClassVar[str] = "artifacts.tokenized.jobs.TokenizeSourceJob"

    tokenizer: Tokenizer
    source: Source

    @property
    def uid(self) -> str:
        return f"{self.tokenizer.uid}-{self.source.uid}"

    @property
    def artifact_path(self) -> Path:
        # its own root, grouped by tokenizer -- one `ls` shows every source a
        # given tokenizer has been run over
        return Path("tokenized") / self.tokenizer.uid / self.source.uid

    @property
    def files(self) -> dict[str, str]:
        return {"tokens": "tokens.bin"}
