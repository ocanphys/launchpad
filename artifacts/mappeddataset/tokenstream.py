"""TokenStream -- a virtual concatenation of several uint16 token files,
addressable through one global index without physically merging them.

Built by MappedDataSet._load from the sources it already holds as fields, one
np.memmap per file. Opening a memmap is cheap regardless of file size -- pages
load lazily on access -- so this costs only the syscalls to open each file,
never a read of their contents.

Not an Artifact: a plain object, imported lazily (see MappedDataSet._load)
so that importing mappeddatasets.artifact itself never requires numpy --
only a process that actually binds a MappedDataSet does.

Sources never mix. Each tokens.bin is one coherent document; the tail of one
has nothing to do with the head of the next. A window reaching across that
boundary would hand a training example bytes that were never adjacent in any
real text -- the same reasoning tokenizers.bpe.TokenizerJob already applies at
the pretoken level ("no pretoken can span a file boundary"), one level up, at
the window level. So __getitem__ computes a correct index across any number
of sources of any lengths, but refuses to serve a slice that would cross from
one source into the next, rather than silently concatenating across it.
"""

import bisect
from pathlib import Path

import numpy as np


class TokenStream:
    def __init__(self, paths: list[Path]):
        self._streams = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        # cumulative offsets over however many sources there are:
        # [0, len(s0), len(s0)+len(s1), ..., total]
        self._offsets = [0]
        for stream in self._streams:
            self._offsets.append(self._offsets[-1] + len(stream))

    def __len__(self) -> int:
        return self._offsets[-1]

    def locate(self, index: int) -> tuple[int, int]:
        """Which source `index` falls in, and the offset within it.

        locate(0) -> (0, 0)
        """
        i = bisect.bisect_right(self._offsets, index) - 1
        return i, index - self._offsets[i]

    def __getitem__(self, key: slice) -> np.ndarray:
        """The tokens in `key`, a zero-copy memmap slice -- never a copy
        across sources. Raises ValueError if `key` would cross from one
        source into the next, or would reach past the end of the whole
        stream: Python's own slice normalization (`slice.indices`) silently
        clamps an out-of-range stop instead of raising, which would
        otherwise hand back fewer tokens than asked for with nothing to say
        so -- the same silent-wrongness `__getitem__` already refuses at a
        source boundary, just at the tail of the last source instead. A
        batch sampler built on top of this is expected to only ever ask for
        windows that fit inside one source's own length (see `locate`,
        which it needs to compute those).
        """
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("TokenStream only supports contiguous (step=1) slices")
        # only when start still lands on real data: a start already past the
        # end is unambiguously empty (ordinary slicing), nothing to guard --
        # it's specifically a valid start with a stop clamped past it that
        # would otherwise silently hand back fewer tokens than asked for
        if (
            start < len(self)
            and key.stop is not None
            and key.stop >= 0
            and key.stop > len(self)
        ):
            raise ValueError(
                f"[{start}:{key.stop}) reaches past the end of the stream "
                f"({len(self)} tokens total) -- only {len(self) - start} tokens "
                f"available from {start}"
            )
        if start >= stop:
            return np.empty(0, dtype=np.uint16)
        i, local_start = self.locate(start)
        local_stop = local_start + (stop - start)
        if local_stop > len(self._streams[i]):
            raise ValueError(
                f"[{start}:{stop}) crosses out of source {i} "
                f"(only {len(self._streams[i]) - local_start} tokens left in it)"
            )
        return self._streams[i][local_start:local_stop]
