"""Pure resolution: order, deduplication, agreement, cycles. No storage."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact
from artifacts.core.resolve import resolve
from artifacts.sources import Source, SourceURL
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
ILIAD = SourceURL(name="iliad", url="https://example.org/iliad.txt")


def tokenizer(*sources: Source) -> Tokenizer:
    return Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=sources)


@dataclass(frozen=True)
class Link(Artifact):
    """One optional dependency, so a cycle can be closed by hand."""

    producer: ClassVar[str | None] = None

    name: str
    to: Link | None = None

    @property
    def uid(self) -> str:
        return self.name

    @property
    def artifact_path(self) -> Path:
        return Path("links") / self.name

    @property
    def files(self) -> dict[str, str]:
        return {}


class ResolveTests(unittest.TestCase):
    def test_dependencies_come_first_and_the_request_last(self):
        tokens = TokenizedSource(tokenizer=tokenizer(ODYSSEY, ILIAD), source=ODYSSEY)
        # fields by name: `source` is walked before `tokenizer`, whose own
        # sources then come by path, with odyssey already seen
        self.assertEqual(
            [str(a.artifact_path) for a in resolve(tokens)],
            [
                "sources/odyssey",
                "sources/iliad",
                f"tokenizers/{tokens.tokenizer.uid}",
                str(tokens.artifact_path),
            ],
        )

    def test_a_shared_dependency_resolves_once_keeping_the_first_object(self):
        direct = SourceURL(name="odyssey", url=ODYSSEY.url, commit="direct")
        nested = SourceURL(name="odyssey", url=ODYSSEY.url, commit="nested")
        tokens = TokenizedSource(tokenizer=tokenizer(nested), source=direct)
        sources = [a for a in resolve(tokens) if isinstance(a, Source)]
        self.assertEqual(len(sources), 1)
        self.assertIs(sources[0], direct)  # `source` is walked before `tokenizer`

    def test_a_conflicting_leaf_behind_a_shared_path_is_found(self):
        elsewhere = SourceURL(name="odyssey", url="https://example.org/other.txt")
        # tokenizer(elsewhere) computes the same path as tokenizer(ODYSSEY):
        # the uid digests source names, not URLs. Only the leaf disagrees.
        tokens = TokenizedSource(tokenizer=tokenizer(elsewhere), source=ODYSSEY)
        with self.assertRaisesRegex(ValueError, "different definitions at sources/odyssey"):
            resolve(tokens)

    def test_a_cycle_is_reported_as_a_chain(self):
        a, b = Link("a"), Link("b")
        object.__setattr__(a, "to", b)
        object.__setattr__(b, "to", a)
        with self.assertRaisesRegex(ValueError, "cycle: links/a -> links/b -> links/a"):
            resolve(a)


if __name__ == "__main__":
    unittest.main()
