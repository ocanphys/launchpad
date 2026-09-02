"""BPE: one tokenizer family, whole -- what a tokenizer is, and how one is made.

`Tokenizer` is a dag Artifact first: parameters (vocab_size, special_tokens,
which sources), a uid derived from them, and the folder it owns. That much can
be written down, hashed and declared before anything has been trained.

It is also the thing you tokenize text with. The vocab and merges a training
run produces are not parameters -- they're what the job wrote into the folder
this artifact owns -- so they arrive later, through `bind(root)`, which reads
them back out of that folder. Writing them there in the first place is
`TokenizerJob.save`'s job, not this artifact's: an artifact only ever loads
what's already built. Once bound, `encode`/`decode` work off that state.
Unbound, they say so rather than returning nonsense.

`TokenizerJob` is what fills that folder in, and `TokenizeSourceJob` runs a
built tokenizer over one source. Both live here rather than in a module of
their own: a tokenizer family is small enough to read in one sitting, and the
training loop and the encoder share their innards (the pretokenizer regex, the
merge step), which is exactly the pair that must not drift apart.
"""

import heapq
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import regex as re
from tqdm import tqdm

from dag.artifact import Artifact, _digest
from dag.job import Job
from sources.artifact import Source

if TYPE_CHECKING:
    from runtime import Worker

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


# -- the artifacts -----------------------------------------------------------


@dataclass(frozen=True)
class Tokenizer(Artifact):
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
        digest = _digest(self.special_tokens, sorted(s.uid for s in self.sources))
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
    # object.__setattr__ because the dataclass is frozen, which is exactly the
    # point -- binding state can't change which artifact this is.

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: read this artifact's own tokenizer.json --
        the state encode()/decode() run on -- and check it against what this
        artifact declares. The inverse of TokenizerJob.save.

        object.__setattr__ because the dataclass is frozen, which is exactly
        the point: this state can change without changing which artifact it is.
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


# -- training internals ------------------------------------------------------
#
# Only TokenizerJob below uses these: the corpus reader, and the base-vocabulary
# resolution the trainer needs while it works (merges_dict is the merge tree it
# builds up as it goes, which resolve/decode read back).

merges_dict = {}


def process_chunks(path, chunksize=2**16, special_token="<|endoftext|>"):
    # reads a text file path points to and creates chunks of size chunksize - separated by special token
    # returns an iterable of chunks of bytes
    # TODO: parallelize this.
    delimiter = special_token.encode("utf-8")
    # read bytes from the file
    with open(path, "rb") as f:
        buffer = b""
        while True:
            # read chunksize many bytes
            chunk = f.read(chunksize)
            if not chunk:
                # EOF - flush the buffer and exit the loop.
                yield buffer
                return
            # data after the last delimiter from previous chunk is saved in the buffer
            buffer += chunk
            # find the integer location of the latest delimiter
            id = buffer.rfind(delimiter)
            # if we can't find a delimiter - we keep going until we do.
            # this is in principle dangerous in case we dont'get a delimiter for very long.
            if id == -1:
                continue
            else:
                # if a delimiter is found, split it before the latest one
                delimited = buffer[:id]
                # save the data after that and prepend it to the next chunk.
                buffer = buffer[id:]
                yield delimited


def heap_entry_valid(heap_entry: tuple, pair_map: dict):
    # check if the top node in the heap is stale
    freq = -heap_entry[0]
    pair = heap_entry[1]
    return pair in pair_map and pair_map[pair] == freq


def resolve(index: int):
    # given a token label - resolves it to an array from the base vocabulary
    if index < 256 or index not in merges_dict:
        return [index]
    else:
        return [
            item
            for sublist in [resolve(children) for children in merges_dict[index]]
            for item in sublist
        ]


def decode(tokens: list[int]):
    # breaks down a list of tokens into base representation
    return [bt for resolved in [resolve(index) for index in tokens] for bt in resolved]


def pair_decode(pair, pair_map):
    """Return 'freq  (a, b)  (b'x', b'y')' for a pair."""
    freq = pair_map.get(pair, 0)
    decoded = tuple(bytes(decode([idx])) for idx in pair)
    return f"{freq:>8}  {pair}  {decoded}"


def peek_top_pairs(pair_heap, pair_map, n=5):
    """Return top-n valid pairs as formatted strings without modifying pair_heap."""
    snapshot = pair_heap.copy()
    results = []
    while snapshot and len(results) < n:
        entry = heapq.heappop(snapshot)
        results.append(pair_decode(entry[1], pair_map))
    return results


# -- the jobs ----------------------------------------------------------------


class TokenizerJob(Job):
    artifact: Tokenizer

    def __init__(self, artifact: Tokenizer):
        super().__init__(artifact)
        self.sources = artifact.sources
        self.special_tokens = list(artifact.special_tokens)
        self.vocab_size = artifact.vocab_size

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info(
            f"training BPE tokenizer (vocab_size={self.vocab_size}) "
            f"on {len(self.sources)} source(s)"
        )

        # sorted by uid, matching Tokenizer.uid: if source order doesn't change
        # which tokenizer this is, it mustn't change what gets trained either
        input_paths = [
            source.paths(root)["raw text"]
            for source in sorted(self.sources, key=lambda s: s.uid)
        ]

        # One file or many is the same thing to the frequency map: each file is
        # pretokenized on its own and the counts are summed. Equivalent to fitting
        # on the concatenation, except no pretoken can span a file boundary -- the
        # same guarantee the EOS separator gives inside a single file.
        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)

        freq = {}  # pretoken frequency map
        with tqdm(
            total=sum(os.path.getsize(path) for path in input_paths),
            desc="pretokenizing and building frequency map",
            unit="B",
            unit_scale=True,
        ) as pbar:
            for path in input_paths:
                for chunk in process_chunks(path):
                    text = chunk.decode("utf-8")
                    segments = re.split(pattern, text) if pattern else [text]
                    for segment in segments:
                        for token in re.findall(PAT, segment):
                            freq[token] = freq.get(token, 0) + 1
                    pbar.update(len(chunk))

        pretoken_str, pretoken_freq = zip(*freq.items())
        pretokens = [list(pretoken.encode("utf-8")) for pretoken in pretoken_str]
        # once we have the frequency map - let's fix an index for each pretoken

        pair_map = {}
        for index in range(len(pretokens)):
            token = pretokens[index]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_map[pair] = pair_map.get(pair, 0) + pretoken_freq[index]

        # create a priority queue to keep track of the most frequent pair
        pair_heap = [(-count, pair) for pair, count in pair_map.items()]
        heapq.heapify(pair_heap)

        # LAZY DELETION - pair_map is the source of truth
        # pair heap might have stale entries as pairs get deleted
        # instead of finding and deleting every single pair, we keep them and
        # check if they are still alive by comparing the most frequent pair
        # to the pair_map, which is the source of truth. The point of not
        # deleting from the heap is to maintain the heap property.

        merges = []
        vocab = {i: bytes(resolve(i)) for i in range(256)}
        base_vocab_size = len(vocab)
        for merge_index in tqdm(
            range(self.vocab_size - 256 - len(self.special_tokens)),
            desc="merging pairs",
        ):
            if len(pair_heap) == 0:
                break

            # TODO: both while loops below index pair_heap[0] without checking
            # it's non-empty first. If every remaining heap entry is stale
            # (lazy-delete loop) or all remaining entries share highest_freq
            # (degenerate-stack loop), the heap drains to empty mid-loop and the
            # next pair_heap[0] raises IndexError instead of ending training
            # early -- hit when vocab_size asks for more merges than the corpus
            # has distinct pairs left to give. Fix: guard both with
            # `while pair_heap and ...`, and break out of the merge loop if the
            # lazy-delete pass empties the heap.

            # LAZY DELETE: ensure top of the heap is valid((pair_heap[0][1] in pair_map)
            while not heap_entry_valid(pair_heap[0], pair_map):
                heapq.heappop(pair_heap)

            # find the highest frequency entry that is lexographically largest
            degenerate_stack = []
            highest_freq = pair_heap[0][0]
            while pair_heap[0][0] == highest_freq:
                val = heapq.heappop(pair_heap)
                if heap_entry_valid(val, pair_map):
                    degenerate_stack.append(val)

            # lexographically largest is ill defined because the tokens may not
            # correspond to valid unicode characters because the byte sequence is
            # not necessarily utf decodable (could be incomplete) -- instead of
            # decoding to utf-8 and comparing, we compare the byte sequences
            # instead.
            lex_sort = sorted(
                degenerate_stack,
                key=lambda heap_entry: tuple(
                    bytes(decode([index])) for index in heap_entry[1]
                ),
            )
            most_frequent_pair = lex_sort[-1]  # grab the largest
            new_pair_index = base_vocab_size + merge_index
            merges_dict[new_pair_index] = most_frequent_pair[1]
            merges.append(tuple(bytes(resolve(p)) for p in most_frequent_pair[1]))
            vocab[new_pair_index] = bytes(resolve(new_pair_index))
            # push rest back into the heap
            for pair in lex_sort[:-1]:
                heapq.heappush(pair_heap, pair)

            pair_map_diff = {}
            for i in range(len(pretokens)):
                pretokens[i], created, destroyed = mergebpairs(
                    pretokens[i], most_frequent_pair[1], new_pair_index
                )
                # collect changes for all pretokens
                for pair in created:
                    pair_map_diff[pair] = pair_map_diff.get(pair, 0) + pretoken_freq[i]
                for pair in destroyed:
                    pair_map_diff[pair] = pair_map_diff.get(pair, 0) - pretoken_freq[i]
            # update affected pairs for aggregated diff:
            for pair in pair_map_diff:
                updated_frequency = pair_map.get(pair, 0) + pair_map_diff[pair]
                if updated_frequency == 0 and pair in pair_map:
                    del pair_map[pair]
                else:
                    pair_map[pair] = updated_frequency
                    heapq.heappush(pair_heap, (-updated_frequency, pair))

        # and finally add special tokens.
        offset = len(vocab)
        for index in range(len(self.special_tokens)):
            vocab[offset + index] = bytes(self.special_tokens[index].encode("utf-8"))

        self.save(root, vocab, merges)
        worker.log.info(f"trained, vocab has {len(vocab)} entries")

    def save(
        self, root: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
    ) -> None:
        """Write vocab/merges into the folder self.artifact owns -- the last
        thing training does, and the inverse of Tokenizer._load."""
        self.artifact.paths(root)["tokenizer"].write_text(
            json.dumps(
                {
                    "vocab_size": len(vocab),
                    "special_tokens": list(self.special_tokens),
                    "vocab": {
                        str(idx): token.decode("latin-1")
                        for idx, token in vocab.items()
                    },
                    "merges": [
                        [a.decode("latin-1"), b.decode("latin-1")] for a, b in merges
                    ],
                },
                indent=2,
            )
        )


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
        self.artifact.paths(root)["tokens"].write_text(
            " ".join(map(str, token_ids))
        )  # mock binary encoding as whitespace-joined ids

        worker.log.info(f"wrote {len(token_ids)} tokens for {self.source.name}")
