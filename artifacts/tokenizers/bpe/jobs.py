"""BPE: the job that fills a Tokenizer's folder in. Split from __init__.py
(which holds the artifact itself) because the two are one family read in one
sitting, but PAT/mergebpairs -- the pretokenizer regex and the merge step --
are shared between training here and encoding there, which is exactly the
pair that must not drift apart.
"""

import heapq
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import regex as re

from artifacts.core.job import Job
from artifacts.tokenizers.bpe import PAT, Tokenizer, mergebpairs

if TYPE_CHECKING:
    from system.runtime import Worker

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

        # Progress is reported by mutating `worker.progress`, which costs a
        # local dict write and nothing else -- the heartbeat thread snapshots it
        # onto the wire on its own cadence, so this can be updated per chunk
        # without either loop waiting on the network (system.runtime.Worker).
        total_bytes = sum(os.path.getsize(path) for path in input_paths)
        read_bytes = 0

        # worker.progress says how far along a running call is and is gone once
        # it ends; these lines are what the call's log still holds afterwards.
        for path in input_paths:
            worker.log.info(f"  {path.parent.name}/{path.name}: {os.path.getsize(path)} bytes")
        started = time.perf_counter()
        logged_bytes = 0

        freq = {}  # pretoken frequency map
        for path in input_paths:
            for chunk in process_chunks(path):
                text = chunk.decode("utf-8")
                segments = re.split(pattern, text) if pattern else [text]
                for segment in segments:
                    for token in re.findall(PAT, segment):
                        freq[token] = freq.get(token, 0) + 1
                read_bytes += len(chunk)
                worker.progress.update(
                    {"phase": "pretokenizing", "done": read_bytes, "total": total_bytes}
                )
                if read_bytes - logged_bytes >= total_bytes / 10:
                    logged_bytes = read_bytes
                    worker.log.info(
                        f"pretokenizing {read_bytes}/{total_bytes} bytes, "
                        f"{len(freq)} distinct pretokens so far"
                    )

        worker.log.info(
            f"pretokenized {read_bytes} bytes in {time.perf_counter() - started:.1f}s: "
            f"{len(freq)} distinct pretokens, {sum(freq.values())} occurrences"
        )

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
        total_merges = self.vocab_size - 256 - len(self.special_tokens)

        worker.log.info(f"seeded {len(pair_map)} distinct pairs, merging {total_merges}:")
        for line in peek_top_pairs(pair_heap, pair_map):
            worker.log.info(f"  {line}")
        started = time.perf_counter()
        log_every = max(1, total_merges // 20)

        for merge_index in range(total_merges):
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

            # counted here rather than at the top of the loop: `done` is merges
            # made, so the last one has to be reported after it is made
            merged = merge_index + 1
            worker.progress.update(
                {"phase": "merging", "done": merged, "total": total_merges}
            )
            if merged % log_every == 0 or merged == total_merges:
                left, right = merges[-1]
                elapsed = time.perf_counter() - started
                worker.log.info(
                    f"merge {merged}/{total_merges}: {left} + {right} -> "
                    f"{left + right} at {-most_frequent_pair[0]}, "
                    f"{len(pair_map)} pairs live, {merged / elapsed:.1f}/s"
                )

        worker.log.info(
            f"merged {len(merges)} pairs in {time.perf_counter() - started:.1f}s"
        )

        # and finally add special tokens.
        offset = len(vocab)
        for index in range(len(self.special_tokens)):
            vocab[offset + index] = bytes(self.special_tokens[index].encode("utf-8"))
        worker.log.info(f"appended {len(self.special_tokens)} special tokens at {offset}")

        self.save(root, vocab, merges)
        path = self.artifact.paths(root)["tokenizer"]
        worker.log.info(
            f"trained, vocab has {len(vocab)} entries -- wrote {path.name}, "
            f"{path.stat().st_size} bytes"
        )

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

