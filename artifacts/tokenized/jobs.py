from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.tokenized import TokenizedSource

if TYPE_CHECKING:
    from system.runtime import Worker


class TokenizeSourceJob(Job):
    artifact: TokenizedSource

    def __init__(self, artifact: TokenizedSource):
        super().__init__(artifact)
        self.tokenizer = artifact.tokenizer
        self.source = artifact.source

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info(f"tokenizing {self.source.name}")

        tokenizer = self.tokenizer.bind(root)  # reads the tokenizer.json its job wrote
        text = self.source.paths(root)["raw text"].read_text()
        token_ids = tokenizer.encode(text)
        # uint16: every id is below vocab_size, and vocab_size is expected to
        # stay under 2**16 -- two bytes, no header, so a dataset can copy or
        # map these with uint16 separators between sources.
        self.artifact.paths(root)["tokens"].write_bytes(array("H", token_ids).tobytes())

        worker.log.info(f"wrote {len(token_ids)} tokens for {self.source.name}")
