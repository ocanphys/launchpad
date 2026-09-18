"""TokenStream -- several uint16 token files read as one logical array,
without ever writing that array down.

Built by MappedDataSet._load: one np.memmap per file (pages load on access,
so opening costs the syscalls and nothing else), a separator token between
consecutive files when one is given, and the offset of every piece worked out
once here from the memmap lengths. A window inside one file comes back as that
memmap's own slice; a window spanning a boundary is copied, that window only.

Not an Artifact, and imported lazily so that numpy loads only in a process
that binds a MappedDataSet.
"""

import bisect
from pathlib import Path

import numpy as np


class TokenStream:
    dtype = np.uint16

    def __init__(self, paths: list[Path], separator: int | None):
        # an empty file cannot be mapped, and an empty source is a legal one
        pieces = [
            np.memmap(p, dtype=np.uint16, mode="r")
            if p.stat().st_size
            else np.empty(0, dtype=np.uint16)
            for p in paths
        ]
        if separator is not None:
            between = np.array([separator], dtype=np.uint16)
            pieces = [piece for source in pieces for piece in (source, between)][:-1]
        self._segments: list[tuple[int, np.ndarray]] = []
        start = 0
        for piece in pieces:
            self._segments.append((start, piece))
            start += len(piece)
        self._starts = [seg_start for seg_start, _ in self._segments]
        self._total = start

    def __len__(self) -> int:
        return self._total

    @property
    def shape(self) -> tuple[int]:
        return (self._total,)

    def __getitem__(self, key: int | slice) -> np.ndarray:
        """The tokens at `key`, with numpy's own semantics for negative and
        out-of-range positions; only step-1 slices are served."""
        if isinstance(key, int):
            if not -self._total <= key < self._total:
                raise IndexError(f"index {key} out of range for {self._total} tokens")
            key %= self._total
            return self[key : key + 1][0]
        start, stop, step = key.indices(self._total)
        if step != 1:
            raise ValueError("TokenStream only supports contiguous (step=1) slices")
        if start >= stop:
            return np.empty(0, dtype=np.uint16)
        i = bisect.bisect_right(self._starts, start) - 1
        pieces = []
        while start < stop:
            seg_start, segment = self._segments[i]
            pieces.append(segment[start - seg_start : stop - seg_start])
            start = seg_start + len(segment)
            i += 1
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
