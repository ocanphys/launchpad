"""BPE: one tokenizer family's artifacts -- what a tokenizer IS, and what it
takes to use one once it's trained. The training loop that produces one lives
in jobs.py; the two share this module's PAT/mergebpairs, which is exactly the
pair that must not drift apart.

`Tokenizer` is a core Artifact first: parameters (vocab_size, special_tokens,
which sources), a uid derived from them, and the folder it owns. That much can
be written down, hashed and declared before anything has been trained.

It is also the thing you tokenize text with. The vocab and merges a training
run produces are not parameters -- they're what the job wrote into the folder
this artifact owns -- so they arrive later, through `bind(root)`, which reads
them back out of that folder. Writing them there in the first place is
`TokenizerJob.save`'s job, not this artifact's: an artifact only ever loads
what's already built. Once bound, `encode`/`decode` work off that state.
Unbound, they say so rather than returning nonsense.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import regex as re

from artifacts.core.artifact import Artifact, _digest
from artifacts.sources import Source

# lifting pretokenizer regex from tiktoken
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def mergebpairs(pretoken: list, pair: tuple, new_index: int, track_diff: bool = True):
    """Apply a single BPE merge to a pretoken, replacing every adjacent occurrence
    of ``pair`` with ``new_index``.

    Input
        pretoken: list[int]  The pretoken as a list of token ids (bytes or merged ids).
        pair: tuple[int, int]  The adjacent pair of token ids to merge.
        new_index: int  The token id that replaces every occurrence of ``pair``.
        track_diff: bool  When True (default), also compute the pairs created and
            destroyed by this merge so the caller can update its pair frequency map
            incrementally. When False, skip building these lists and return empty
            lists in their place — useful when the caller does not need the diff
            (e.g. tokenizer encoding) and wants to avoid the bookkeeping overhead.

    Output
        pretoken: list[int]  The pretoken after the merge has been applied.
        to_add: list[tuple[int, int]]  Pairs newly created by the merge (each
            occurrence contributes one entry). Empty when ``track_diff`` is False.
        to_remove: list[tuple[int, int]]  Pairs destroyed by the merge, including
            one entry of ``pair`` itself for every occurrence merged. Empty when
            ``track_diff`` is False.
    """
    if not pair:
        return pretoken
    else:
        if track_diff:
            to_add = []
            to_remove = []
        i = 0
        while i < len(pretoken):
            if (
                i < len(pretoken) - 1
                and pair[0] == pretoken[i]
                and pair[1] == pretoken[i + 1]
            ):
                left = []
                right = []
                if i > 0:  # has left
                    left = pretoken[:i]
                    if track_diff:
                        to_remove.append((pretoken[i - 1], pretoken[i]))
                        to_add.append((pretoken[i - 1], new_index))
                if i + 2 < len(pretoken):  # has right
                    right = pretoken[i + 2 :]
                    if track_diff:
                        to_remove.append((pretoken[i + 1], pretoken[i + 2]))
                        to_add.append((new_index, pretoken[i + 2]))
                if track_diff:
                    to_remove.append(pair)
                pretoken = left + [new_index] + right
            i += 1
        if track_diff:
            return pretoken, to_add, to_remove
        else:
            return pretoken


def _read_state(data: dict) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """A parsed tokenizer.json, back as vocab and merges. Byte strings
    round-trip through latin-1 text, a 1:1 mapping between byte values 0-255
    and Unicode code points 0-255 -- unlike utf-8, it can encode/decode *any*
    byte sequence, including ones that aren't valid utf-8 (BPE merges routinely
    produce these)."""
    missing = {"vocab", "merges", "special_tokens", "vocab_size"} - data.keys()
    if missing:
        raise ValueError(f"tokenizer.json missing required key(s): {sorted(missing)}")
    vocab = {int(idx): token.encode("latin-1") for idx, token in data["vocab"].items()}
    merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in data["merges"]]
    return vocab, merges


@dataclass(frozen=True)
class Tokenizer(Artifact):
    producer: ClassVar[str] = "artifacts.tokenizers.bpe.jobs.TokenizerJob"

    vocab_size: int
    special_tokens: tuple[str, ...]
    sources: tuple[Source, ...]  # trained on these; order doesn't identify it

    @property
    def uid(self) -> str:
        # digest covers what can't stay readable (special_tokens, sources), so
        # two tokenizers differing only there don't share a folder. Sources are
        # sorted -- *which* sources trained it, not the order they were listed in.
        vocab_label = (
            f"{self.vocab_size / 1000:.1f}k"
            if self.vocab_size > 1000
            else str(self.vocab_size)
        )
        digest = _digest(self.vocab_size, self.special_tokens, sorted(s.uid for s in self.sources))
        return f"bpe-{vocab_label}-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("tokenizers") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {"tokenizer": "tokenizer.json"}

    # -- the trained state -------------------------------------------------
    #
    # Held on the instance and deliberately not a field: it isn't what
    # identifies this tokenizer, so it stays out of ==, hash, and the manifest.
    # object.__setattr__ because the dataclass is frozen -- but by the time
    # this runs, `self` is already the private copy Artifact.bind just made
    # (see bind's own docstring), not the object bind() was called on. Nobody
    # else holds a reference to this particular copy yet, so mutating it here
    # is invisible to everyone but the caller about to receive it back.

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: read this artifact's own tokenizer.json --
        the state encode()/decode() run on -- and check it against what this
        artifact declares. The inverse of TokenizerJob.save.

        object.__setattr__ because the dataclass is frozen -- see the comment
        above this method for why that's safe: `self` here is bind's own
        private copy, not the artifact anyone else is holding.
        """
        data = json.loads(self.paths(root)["tokenizer"].read_text())
        if tuple(data["special_tokens"]) != self.special_tokens:
            # The folder is keyed by a digest over special_tokens, so this file
            # can only disagree if it was written by something other than this
            # artifact's job. Better to say so than to encode with it.
            raise ValueError(
                f"{self.paths(root)['tokenizer']} has special tokens "
                f"{tuple(data['special_tokens'])}, but {self.uid} declares "
                f"{self.special_tokens}"
            )
        vocab, merges = _read_state(data)
        bytes_to_id = {token: idx for idx, token in vocab.items()}
        object.__setattr__(self, "_vocab", vocab)
        object.__setattr__(self, "_merges", merges)
        object.__setattr__(self, "_bytes_to_id", bytes_to_id)
        object.__setattr__(
            self,
            "_merge_dict",
            {
                tuple(bytes_to_id[bytes(b)] for b in merge): bytes_to_id[
                    b"".join(merge)
                ]
                for merge in merges
            },
        )
        object.__setattr__(self, "_cache", {})  # pretoken:str -> list[int]

    @property
    def bound(self) -> bool:
        return hasattr(self, "_vocab")

    def _require(self, doing: str) -> None:
        if not self.bound:
            raise RuntimeError(
                f"{self.uid} has no vocab or merges yet -- bind(root) to read the "
                f"tokenizer.json its job wrote before {doing}"
            )

    @property
    def vocab(self) -> dict[int, bytes]:
        self._require("reading the vocab")
        return self._vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        self._require("reading the merges")
        return self._merges

    # -- tokenizing --------------------------------------------------------

    def encode(self, text_input: str) -> list[int]:
        self._require("encoding")
        # sort by length - if a special token contains another one as a prefix,
        # we do not parse it short
        pattern = (
            "("
            + "|".join(
                re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)
            )
            + ")"
        )
        segments = (
            re.split(pattern, text_input) if self.special_tokens else [text_input]
        )
        ids = []
        for segment in segments:
            if segment in self.special_tokens:
                ids += [self._bytes_to_id[segment.encode("utf-8")]]
            else:
                for pretoken in re.findall(PAT, segment):
                    ids.extend(self.encode_pretoken(pretoken))
        return ids

    def encode_iterable(self, iterable):
        for item in iterable:
            yield from self.encode(item)

    def encode_pass(self, ids: list[int]):
        min_index = float("inf")
        next_merge = None
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            vocab_index = self._merge_dict.get(pair, None)
            if vocab_index and vocab_index < min_index:
                min_index = vocab_index
                next_merge = (pair, vocab_index)  # replace i,i+1 with merge_index
        return next_merge  # a tuple of pair(tuple of ints) and token index created by merge

    def encode_pretoken(self, pretoken: str) -> list[int]:
        if pretoken in self._cache:
            return self._cache[pretoken]
        pretoken_l = [self._bytes_to_id[bytes([b])] for b in pretoken.encode("utf-8")]
        next_merge = self.encode_pass(pretoken_l)
        while next_merge:
            pretoken_l = mergebpairs(pretoken_l, *next_merge, track_diff=False)
            next_merge = self.encode_pass(pretoken_l)
        self._cache[pretoken] = pretoken_l
        return pretoken_l

    def decode(self, ids: list[int]) -> str:
        self._require("decoding")
        return b"".join(self._vocab[id] for id in ids).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class TokenizedSource(Artifact):
    producer: ClassVar[str] = "artifacts.tokenizers.bpe.jobs.TokenizeSourceJob"

    tokenizer: Tokenizer
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

