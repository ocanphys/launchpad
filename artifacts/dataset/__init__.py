from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from artifacts.core.artifact import Artifact, _digest
from artifacts.sources import Source
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers import Tokenizer

if TYPE_CHECKING:
    import numpy as np

END_OF_TEXT = "<|endoftext|>"


def require_separator(name: str, sources: tuple[TokenizedSource, ...]) -> None:
    """Raises unless every tokenizer in a multi-source split can produce the
    separator that goes between its sources."""
    if len(sources) > 1 and any(
        END_OF_TEXT not in source.tokenizer.special_tokens for source in sources
    ):
        raise ValueError(
            f"DataSet.{name} with multiple sources requires {END_OF_TEXT!r} "
            "in each tokenizer's special_tokens"
        )


@dataclass(frozen=True)
class DataSet(Artifact):
    """A training set built by physically concatenating every source's
    tokens into train.bin/valid.bin, with <|endoftext|> between sources
    (artifacts/dataset/jobs.py's DataSetJob). `from_sources` builds one from
    a tokenizer and raw sources.
    Shared, not run-scoped: its bytes are a pure function of
    train_set/valid_set, so two runs asking for the same tokenizer and
    sources reuse the same folder instead of each paying for their own copy.

    artifacts.mappeddataset.MappedDataSet is a different kind of artifact,
    not a variant of this one -- it reads straight out of the sources' own
    bin files instead of copying them here at all, and lives in its own
    sibling package accordingly.
    """

    producer: ClassVar[str] = "artifacts.dataset.jobs.DataSetJob"

    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        tokenizers = sorted({source.tokenizer.uid for source in (*self.train_set, *self.valid_set)})
        if len(tokenizers) > 1:
            raise ValueError(f"{type(self).__name__} mixes tokenizers {', '.join(tokenizers)}")
        require_separator("train_set", self.train_set)
        require_separator("valid_set", self.valid_set)

    @classmethod
    def from_sources(
        cls,
        tokenizer: Tokenizer,
        train_sources: tuple[Source, ...],
        valid_sources: tuple[Source, ...],
    ) -> DataSet:
        return cls(
            train_set=tuple(TokenizedSource(tokenizer, s) for s in train_sources),
            valid_set=tuple(TokenizedSource(tokenizer, s) for s in valid_sources),
        )

    @property
    def tokenizer(self) -> Tokenizer:
        """Returns the one tokenizer every source was tokenized with."""
        return (*self.train_set, *self.valid_set)[0].tokenizer

    @property
    def uid(self) -> str:
        # order is identifying here, same reasoning as MappedDataSet's uid:
        # concatenation order changes train.bin/valid.bin's actual contents,
        # so this must not sort train_set/valid_set away.
        digest = _digest(
            [ts.uid for ts in self.train_set], [ts.uid for ts in self.valid_set]
        )
        if len(self.train_set) > 1 or len(self.valid_set) > 1:
            # The separator changes the merged bytes and therefore their identity.
            digest = _digest(digest, END_OF_TEXT)
        return f"dataset-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("datasets") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {"training set": "train.bin", "validation set": "valid.bin"}

    # -- the bound view ------------------------------------------------------

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: one memmap per split over the bin its job
        wrote. object.__setattr__ because the dataclass is frozen -- `self`
        here is the object Artifact.bind read from the stored manifest, never
        the one bind() was called on."""
        import numpy as np  # local: only a process that binds pays for numpy

        for name, path in (
            ("_train_tokens", self.paths(root)["training set"]),
            ("_valid_tokens", self.paths(root)["validation set"]),
        ):
            # an empty file cannot be mapped, and an empty split is a legal one
            tokens = (
                np.memmap(path, dtype=np.uint16, mode="r")
                if path.stat().st_size
                else np.empty(0, dtype=np.uint16)
            )
            object.__setattr__(self, name, tokens)

    @property
    def bound(self) -> bool:
        return hasattr(self, "_train_tokens")

    def _require(self, doing: str) -> None:
        if not self.bound:
            raise RuntimeError(
                f"{self.uid} has no token arrays yet -- bind(root) to map the "
                f"train.bin/valid.bin its job wrote before {doing}"
            )

    @property
    def train_tokens(self):
        self._require("reading train_tokens")
        return self._train_tokens

    @property
    def valid_tokens(self):
        self._require("reading valid_tokens")
        return self._valid_tokens

    def train_batches(self, batch_size: int, sequence_length: int, seed: int) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
        """Returns the function from a step number to that step's (inputs,
        targets), int64 arrays of shape (batch_size, sequence_length) drawn at
        random positions of train_tokens.

        The batch is a function of (seed, step) alone, so a resumed run draws
        exactly the batches the interrupted one would have."""
        import numpy as np

        tokens = self.train_tokens

        def get_train_batch(step: int) -> tuple[np.ndarray, np.ndarray]:
            rng = np.random.default_rng((seed, step))
            starts = rng.integers(0, len(tokens) - sequence_length, size=batch_size)
            # int64 because Embedding needs long
            inputs = np.stack([tokens[i : i + sequence_length] for i in starts]).astype(np.int64)
            targets = np.stack([tokens[i + 1 : i + sequence_length + 1] for i in starts]).astype(np.int64)
            return inputs, targets

        return get_train_batch

    def valid_batches(
        self, batch_size: int, sequence_length: int
    ) -> tuple[int, Callable[[int], tuple[np.ndarray, np.ndarray]]]:
        """Returns how many batches cover valid_tokens and the function from a
        batch id below that count to its (inputs, targets), int64 arrays of
        consecutive, non-overlapping sequences of sequence_length.

        A split shorter than one batch is one batch of the sequences it holds,
        so fewer than batch_size rows; the tail too short to fill a batch is in
        no batch."""
        import numpy as np

        tokens = self.valid_tokens
        sequences = (len(tokens) - 1) // sequence_length
        if sequences < 1:
            raise ValueError(
                f"the validation split holds {len(tokens)} tokens, fewer than one "
                f"sequence of {sequence_length} and its next token"
            )
        size = min(batch_size, sequences) * sequence_length

        def get_valid_batch(batch_id: int) -> tuple[np.ndarray, np.ndarray]:
            window = tokens[batch_id * size : (batch_id + 1) * size + 1].astype(np.int64)
            return window[:-1].reshape(-1, sequence_length), window[1:].reshape(-1, sequence_length)

        return (len(tokens) - 1) // size, get_valid_batch
