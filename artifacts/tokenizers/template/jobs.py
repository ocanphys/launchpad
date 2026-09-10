"""Template for a new tokenizer family's jobs -- see __init__.py's own
docstring for how to copy this whole family and what has to stay unique.
"""

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.tokenizers.template import (
    UNKNOWN,
    TemplateTokenizedSource,
    TemplateTokenizer,
)

if TYPE_CHECKING:
    from system.runtime import Worker


class TemplateTokenizerJob(Job):
    artifact: TemplateTokenizer

    def __init__(self, artifact: TemplateTokenizer):
        super().__init__(artifact)
        # bind the artifact's parameters under their own names, once
        self.sources = artifact.sources
        self.special_tokens = list(artifact.special_tokens)
        self.vocab_size = artifact.vocab_size

    def run(self, root: Path, worker: "Worker") -> None:
        """Write self.artifact's files under root, and nothing else.

        worker.log is this call's own log file -- narrate here rather than
        printing, and don't repeat the job's name into the message: it's
        resolved from the manifest when the log is read back.
        """
        worker.log.info(
            f"training tokenizer (vocab_size={self.vocab_size}) "
            f"on {len(self.sources)} source(s)"
        )

        # sorted by uid, matching uid's own sort: if source order doesn't change
        # which tokenizer this is, it mustn't change what gets trained either
        texts = [
            source.paths(root)["raw text"].read_text()
            for source in sorted(self.sources, key=lambda s: s.uid)
        ]

        # -- ALGORITHM (1 of 2): whatever training means for this family -----
        # Here: count words, keep the most frequent. Yours goes in this block,
        # and everything below it stays as it is.
        counts = Counter(word for text in texts for word in text.split())
        reserved = [UNKNOWN, *self.special_tokens]
        learned = [
            word for word, _ in counts.most_common(self.vocab_size - len(reserved))
        ]
        vocab = {idx: token for idx, token in enumerate(reserved + learned)}
        # --------------------------------------------------------------------

        self.save(root, vocab)
        worker.log.info(f"trained, vocab has {len(vocab)} entries")

    def save(self, root: Path, vocab: dict[int, str]) -> None:
        """Write vocab into the folder self.artifact owns -- the last thing
        training does, and the inverse of TemplateTokenizer._load."""
        self.artifact.paths(root)["tokenizer"].write_text(
            json.dumps(
                {
                    "vocab_size": len(vocab),
                    "special_tokens": list(self.special_tokens),
                    "vocab": {str(idx): tok for idx, tok in vocab.items()},
                },
                indent=2,
            )
        )


class TemplateTokenizeSourceJob(Job):
    artifact: TemplateTokenizedSource

    def __init__(self, artifact: TemplateTokenizedSource):
        super().__init__(artifact)
        self.tokenizer = artifact.tokenizer
        self.source = artifact.source

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info(f"tokenizing {self.source.name}")

        tokenizer = self.tokenizer.bind(root)  # reads the tokenizer.json its job wrote
        text = self.source.paths(root)["raw text"].read_text()
        token_ids = tokenizer.encode(text)
        self.artifact.paths(root)["tokens"].write_text(
            " ".join(map(str, token_ids))
        )  # mock binary encoding as whitespace-joined ids

        worker.log.info(f"wrote {len(token_ids)} tokens for {self.source.name}")
