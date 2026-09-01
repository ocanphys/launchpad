"""Template for a new tokenizer family -- copy this file, keep the shape,
replace the algorithm.

A tokenizer family is one module under tokenizers/ (see tokenizers/bpe.py for
the real one) holding four classes: the tokenizer artifact, the tokenized-source
artifact, and a job apiece. The algorithm below is deliberately the dumbest one
that works -- a word-level vocabulary of the most frequent words -- so that what
remains is exactly the contract a family has to satisfy. Swap the two marked
ALGORITHM blocks and you have a different tokenizer.

To use it:

1. Copy to tokenizers/<yourname>.py.
2. Rename every `Template*` class. The names must be unique across the whole
   repo: artifacts register by class name (dag/artifact.py's ARTIFACTS) and
   jobs by the artifact class they produce (dag/job.py's REGISTRY), so a second
   `Tokenizer` would collide with bpe's at import time.
3. Replace the ALGORITHM blocks -- train (in the job) and encode/decode (on the
   artifact). Everything else is bookkeeping that stays as it is.
4. Add `from tokenizers import <yourname> as _tokenizers_<yourname>  # noqa: F401`
   to dag/resolve.py's imports, which is what puts the family in the registries.
5. Downstream artifacts name their tokenizer type concretely
   (datasets.artifact.DataSet, models/*/artifact.py), so widen those annotations
   to accept the new family before declaring a run that uses it.

Two things that will bite if changed:

- No `from __future__ import annotations` in this module. Job registration reads
  `cls.__annotations__["artifact"]` as a real class; postponed evaluation would
  hand it the string "TemplateTokenizer" and every lookup would KeyError.
- `uid` must be injective over the parameters. Identity is the folder, so two
  different parameter sets rendering to one uid is one artifact silently
  overwriting another -- hence the digest over whatever can't stay readable.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dag.artifact import Artifact, _digest
from dag.job import Job
from sources.artifact import Source

if TYPE_CHECKING:
    from runtime import Worker

UNKNOWN = "<unk>"  # this template's own choice; your algorithm may not need one


@dataclass(frozen=True)
class TemplateTokenizer(Artifact):
    """The parameters of a training run, and the tokenizer it produces.

    Parameters (the fields) say which tokenizer this is and where it lives.
    The trained state -- whatever your algorithm learns -- is not a parameter:
    it's what the job wrote into this artifact's folder, so it arrives later
    through `bind`, and stays off the fields so it can't touch ==, hash, or
    the manifest.
    """

    vocab_size: int
    special_tokens: tuple[str, ...]
    sources: tuple[Source, ...]  # trained on these; order doesn't identify it

    @property
    def uid(self) -> str:
        # readable half first, then a digest of what can't stay readable
        digest = _digest(self.special_tokens, sorted(s.uid for s in self.sources))
        return f"template-{self.vocab_size}-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("tokenizers") / self.uid

    @property
    def files(self) -> dict[str, str]:
        # bare filenames, not paths: the resolver creates this folder (and
        # writes manifest.json into it) before the job runs
        return {"tokenizer": "tokenizer.json"}

    # -- the trained state -------------------------------------------------

    def bind(
        self, root: Path | None = None, *, vocab: dict[int, str] | None = None
    ) -> "TemplateTokenizer":
        """Give this artifact the state to tokenize with -- read from the file
        its job wrote (`bind(root)`), or handed straight over by that job
        (`bind(vocab=...)`). Returns self, so it chains.

        This overrides Artifact.bind, which by itself only checks the files are
        there; override it whenever "using" your artifact means holding
        something in memory. `Artifact.at(folder)` calls it for you.

        object.__setattr__ because the dataclass is frozen, which is the point:
        binding state must not change which artifact this is.
        """
        if root is not None:
            if vocab is not None:
                raise TypeError("bind reads the file or takes vocab, not both")
            vocab = self._read(Path(root))
        elif vocab is None:
            raise TypeError("bind needs a root to read from, or a vocab")
        object.__setattr__(self, "_vocab", dict(vocab))
        object.__setattr__(self, "_ids", {tok: idx for idx, tok in vocab.items()})
        return self

    def _read(self, root: Path) -> dict[int, str]:
        data = json.loads(self.paths(root)["tokenizer"].read_text())
        if tuple(data["special_tokens"]) != self.special_tokens:
            # the folder is keyed by a digest over special_tokens, so a file
            # that disagrees was written by something other than this job
            raise ValueError(
                f"{self.paths(root)['tokenizer']} has special tokens "
                f"{tuple(data['special_tokens'])}, but {self.uid} declares "
                f"{self.special_tokens}"
            )
        return {int(idx): token for idx, token in data["vocab"].items()}

    def save(self, root: Path) -> None:
        """Write the bound state into the folder this artifact owns -- the last
        thing the training job does, and the inverse of `bind(root)`."""
        self._require("saving")
        self.paths(Path(root))["tokenizer"].write_text(
            json.dumps(
                {
                    "vocab_size": len(self.vocab),
                    "special_tokens": list(self.special_tokens),
                    "vocab": {str(idx): tok for idx, tok in self.vocab.items()},
                },
                indent=2,
            )
        )

    @property
    def bound(self) -> bool:
        return hasattr(self, "_vocab")

    def _require(self, doing: str) -> None:
        if not self.bound:
            raise RuntimeError(
                f"{self.uid} has no vocab yet -- bind(root) to read the "
                f"tokenizer.json its job wrote, or bind(vocab=...), before {doing}"
            )

    @property
    def vocab(self) -> dict[int, str]:
        self._require("reading the vocab")
        return self._vocab

    # -- ALGORITHM (2 of 2): how text becomes ids and back ------------------
    #
    # Keep these names. Downstream jobs (datasets/, models/*) call encode /
    # encode_iterable / decode on whatever tokenizer they were given, so the
    # names are the whole interface between a family and everything above it.

    def encode(self, text: str) -> list[int]:
        self._require("encoding")
        return [self._ids.get(word, self._ids[UNKNOWN]) for word in text.split()]

    def encode_iterable(self, iterable):
        for item in iterable:
            yield from self.encode(item)

    def decode(self, ids: list[int]) -> str:
        self._require("decoding")
        return " ".join(self._vocab[id] for id in ids)


@dataclass(frozen=True)
class TemplateTokenizedSource(Artifact):
    """One source encoded by one tokenizer -- an artifact, because it's a file
    someone has to produce and everything downstream depends on."""

    tokenizer: TemplateTokenizer
    source: Source

    @property
    def uid(self) -> str:
        return f"{self.tokenizer.uid}-{self.source.uid}"

    @property
    def artifact_path(self) -> Path:
        # nested under the tokenizer that produced it -- one `ls` shows every
        # source a given tokenizer has been run over
        return self.tokenizer.artifact_path / "bin" / self.source.uid

    @property
    def files(self) -> dict[str, str]:
        return {"tokens": "tokens.bin"}


class TemplateTokenizerJob(Job):
    # This annotation is the registration: it says the job produces
    # TemplateTokenizer, and types self.artifact for everything below.
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

        # hand the trained state to the artifact that declared this run, and
        # let it write itself into the folder it owns
        self.artifact.bind(vocab=vocab).save(root)
        worker.log.info(f"trained, vocab has {len(vocab)} entries")


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
