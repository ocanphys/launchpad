"""A dataset split once bound: its tokens, and the tag of every token position.

Positions are those of one split's stream, so train and valid each carry their
own tags. Tags are stored as ranges (the position each range starts at and its
tag id), which costs one entry per range rather than one per token; callers
see only per-position slices. A split with no tags is one range: tag 0 from
position 0.

Not an Artifact: a dataset builds one per split when it binds.
"""

from dataclasses import dataclass

import numpy as np


class TagRanges:
    """The tag of every position of a stream of `length` tokens, read in
    contiguous slices.

        TagRanges(np.array([0, 3]), np.array([5, 9]), length=6)[2:5].tolist() -> [5, 9, 9]
    """

    def __init__(self, starts: np.ndarray, tags: np.ndarray, length: int):
        if len(starts) != len(tags) or not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0):
            raise ValueError("tag ranges need one tag per start, with starts increasing from 0")
        self._starts = starts
        self._tags = tags
        self._length = length

    @classmethod
    def untagged(cls, length: int) -> "TagRanges":
        """Returns the tags of a stream with no tags: 0 at every position.

            TagRanges.untagged(4)[:].tolist() -> [0, 0, 0, 0]

        Args:
            length: How many tokens the stream holds.

        Returns:
            One range, tag 0 from position 0.
        """
        return cls(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, key: slice) -> np.ndarray:
        """Returns the tag of each position in `key`, a step-1 slice with
        numpy's semantics for negative and out-of-range bounds."""
        start, stop, step = key.indices(self._length)
        if step != 1:
            raise ValueError("TagRanges only supports contiguous (step=1) slices")
        positions = np.arange(start, stop)
        return self._tags[np.searchsorted(self._starts, positions, side="right") - 1]


@dataclass(frozen=True)
class Split:
    """One split of a bound dataset: its uint16 token stream (a memmap, or a
    TokenStream, which slices like one) and the tags of its positions, the
    same length."""

    tokens: np.ndarray
    tags: TagRanges
