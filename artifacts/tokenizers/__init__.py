"""The tokenizer contract, and all that anything outside artifacts/tokenizers/
depends on. A family is a package here (bpe/, ...) whose tokenizer subclasses
this and adds its own parameters, files and training job; a consumer
annotates `tokenizer: Tokenizer` and never names a family.
"""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from artifacts.core.artifact import Artifact


@dataclass(frozen=True)
class Tokenizer(Artifact):
    vocab_size: int
    special_tokens: tuple[str, ...]

    @property
    def artifact_path(self) -> Path:
        return Path("tokenizers") / self.uid

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids for `text`; raises unless bound."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """The text `ids` spell; raises unless bound."""
