from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.tokenized import TokenizedSource

if TYPE_CHECKING:
    from system.runtime import Worker

# ids held between writes: 2 MB of uint16, so a source costs this much to
# tokenize whatever its size
FLUSH_TOKENS = 1 << 20


class TokenizeSourceJob(Job):
    artifact: TokenizedSource

    def __init__(self, artifact: TokenizedSource):
        super().__init__(artifact)
        self.tokenizer = artifact.tokenizer
        self.source = artifact.source

    def run(self, root: Path, worker: "Worker") -> None:
        tokenizer = self.tokenizer.bind(root)  # reads the tokenizer.json its job wrote
        body = self.source.paths(root)["raw text"]
        total_bytes = body.stat().st_size
        worker.log.info(f"tokenizing {self.source.name}, {total_bytes} bytes")

        # Published by rename: a tokens.bin at its final path is what makes this
        # artifact done, so a half-written one must never sit there.
        tokens = self.artifact.paths(root)["tokens"]
        tmp = tokens.with_suffix(".tmp")

        # uint16: every id is below vocab_size, and vocab_size is expected to
        # stay under 2**16 -- two bytes, no header, so a dataset can copy or
        # map these with uint16 separators between sources.
        buffered = array("H")
        written = logged_bytes = 0
        with open(body, "rb") as handle, open(tmp, "wb") as out:
            # A line at a time, so neither the text nor its ids are ever held
            # whole -- the source runs to gigabytes, and a list of ids for one
            # costs about ten times what its tokens.bin does.
            ids = tokenizer.encode_iterable(line.decode("utf-8") for line in handle)
            for token_id in ids:
                buffered.append(token_id)
                if len(buffered) == FLUSH_TOKENS:
                    buffered.tofile(out)
                    written += len(buffered)
                    del buffered[:]
                    read_bytes = handle.tell()
                    worker.progress.update(
                        {"phase": "tokenizing", "done": read_bytes, "total": total_bytes}
                    )
                    if read_bytes - logged_bytes >= total_bytes / 10:
                        logged_bytes = read_bytes
                        worker.log.info(
                            f"tokenizing {read_bytes}/{total_bytes} bytes, "
                            f"{written} tokens so far"
                        )
            buffered.tofile(out)
            written += len(buffered)

        tmp.replace(tokens)
        worker.log.info(
            f"wrote {written} tokens for {self.source.name}, "
            f"{tokens.stat().st_size} bytes"
        )
