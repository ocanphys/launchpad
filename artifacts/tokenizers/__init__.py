"""The tokenizer contract, and all that anything outside artifacts/tokenizers/
depends on. A family is a package here (bpe/, ...) whose tokenizer subclasses
this and adds its own parameters, files and training job; a consumer
annotates `tokenizer: Tokenizer` and never names a family.
"""

from abc import abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from artifacts.core.artifact import Artifact


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

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """Returns the ids of each text in turn, encoding one at a time, so a
        source of any size is never held whole.

            list(bound.encode_iterable(["hello", " world"])) -> bound.encode("hello") + bound.encode(" world")
        """
        for text in texts:
            yield from self.encode(text)
