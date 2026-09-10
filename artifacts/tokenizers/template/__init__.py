"""Template for a new tokenizer family's artifacts -- copy this file, keep
the shape, replace the algorithm. See jobs.py's own docstring for the
training side of this same family.

A tokenizer family is a folder under artifacts/tokenizers/ (see
artifacts/tokenizers/bpe/ for the real one) holding an __init__.py (the
tokenizer artifact and the tokenized-source artifact) and a jobs.py (a job
apiece). The algorithm below is deliberately the dumbest one that works --
a word-level vocabulary of the most frequent words -- so that what remains
is exactly the contract a family has to satisfy. Swap the ALGORITHM block
here (encode/decode) and the one in jobs.py (train) and you have a
different tokenizer.

To use it:

1. Copy to artifacts/tokenizers/<yourname>/__init__.py and jobs.py.
2. Rename every `Template*` class, and rewrite each artifact's `producer`
   (see the ClassVar on TemplateTokenizer/TemplateTokenizedSource below) to
   the full dotted path of its own Job class in your family's jobs.py. It is
   written out in full, not derived: an artifact hands out that string
   without importing anything, and only `main.run_job` ever turns it into a
   class.
3. Replace the ALGORITHM blocks -- train (in jobs.py) and encode/decode
   (here). Everything else is bookkeeping that stays as it is.
4. Downstream artifacts name their tokenizer type concretely
   (artifacts.dataset.DataSet, artifacts.mappeddataset.MappedDataSet,
   artifacts/models/*/__init__.py), so widen those annotations to accept
   the new family before declaring a run that uses it.

Two things that will bite if changed:

- `uid` must be injective over the parameters. Identity is the folder, so two
  different parameter sets rendering to one uid is one artifact silently
  overwriting another -- hence the digest over whatever can't stay readable.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from artifacts.core.artifact import Artifact, _digest
from artifacts.sources import Source

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

    producer: ClassVar[str] = "artifacts.tokenizers.template.jobs.TemplateTokenizerJob"

    vocab_size: int
    special_tokens: tuple[str, ...]
    sources: frozenset[Source]  # a set: which sources trained it, not in what order

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
        # bare filenames, not paths: declaration creates this folder (and
        # writes manifest.json into it) before the job runs
        return {"tokenizer": "tokenizer.json"}

    # -- the trained state -------------------------------------------------

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: read this artifact's own tokenizer.json --
        the state encode()/decode() run on -- and check it against what this
        artifact declares. The inverse of TemplateTokenizerJob.save. Override
        this in your own family whenever "using" the artifact means holding
        something in memory; `bind(root)` calls it for you.

        object.__setattr__ because the dataclass is frozen -- but by the time
        this runs, `self` is the object Artifact.bind read from the stored
        manifest, never the one bind() was called on, so mutating it here is
        invisible to whoever is still holding the unbound original.
        """
        data = json.loads(self.paths(root)["tokenizer"].read_text())
        if tuple(data["special_tokens"]) != self.special_tokens:
            # the folder is keyed by a digest over special_tokens, so a file
            # that disagrees was written by something other than this job
            raise ValueError(
                f"{self.paths(root)['tokenizer']} has special tokens "
                f"{tuple(data['special_tokens'])}, but {self.uid} declares "
                f"{self.special_tokens}"
            )
        vocab = {int(idx): token for idx, token in data["vocab"].items()}
        object.__setattr__(self, "_vocab", vocab)
        object.__setattr__(self, "_ids", {tok: idx for idx, tok in vocab.items()})

    @property
    def bound(self) -> bool:
        return hasattr(self, "_vocab")

    def _require(self, doing: str) -> None:
        if not self.bound:
            raise RuntimeError(
                f"{self.uid} has no vocab yet -- bind(root) to read the "
                f"tokenizer.json its job wrote before {doing}"
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

    producer: ClassVar[str] = (
        "artifacts.tokenizers.template.jobs.TemplateTokenizeSourceJob"
    )

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

