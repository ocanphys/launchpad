from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from artifacts.core.artifact import Artifact, _digest
from artifacts.sources import Source
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers import Tokenizer

if TYPE_CHECKING:
    import numpy as np

    from artifacts.mappeddataset.tokenstream import TokenStream

END_OF_TEXT = "<|endoftext|>"


def require_separator(name: str, sources: tuple[TokenizedSource, ...]) -> None:
    """Raises unless every tokenizer in a multi-source split can produce the
    separator that goes between its sources."""
    if len(sources) > 1 and any(
        END_OF_TEXT not in source.tokenizer.special_tokens for source in sources
    ):
        raise ValueError(
            f"MappedDataSet.{name} with multiple sources requires {END_OF_TEXT!r} "
            "in each tokenizer's special_tokens"
        )


class MappedDataSet(Artifact):
    """A training set that owns no bytes: it is done exactly when the
    TokenizedSources it depends on are (`completion_paths`), and its bound
    object is a TokenStream per split (artifacts/mappeddataset/tokenstream.py)
    that reads like one memmap over every source's own tokens.bin, with
    <|endoftext|> between sources. `from_sources` builds one from a tokenizer
    and raw sources.

    Its folder (`mappeddatasets/<uid>`) holds only the manifest, which is what
    lets `Artifact.load`, resolution and the notebooks find it like anything
    else. Shared like DataSet: no run_id, identity a digest over the ordered
    source lists.
    """

    producer: ClassVar[None] = None  # nothing to write: done when its sources are

    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        # Completion is borrowed from the sources, and every file of an empty
        # borrowing is present: with neither split filled this artifact would
        # read as done before anything existed. One split may still be empty.
        if not (self.train_set or self.valid_set):
            raise ValueError(
                "MappedDataSet has no sources -- it owns no files, so there would be "
                "nothing anywhere for it to be done by"
            )
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
    ) -> MappedDataSet:
        return cls(
            train_set=tuple(TokenizedSource(tokenizer=tokenizer, source=s) for s in train_sources),
            valid_set=tuple(TokenizedSource(tokenizer=tokenizer, source=s) for s in valid_sources),
        )

    @property
    def tokenizer(self) -> Tokenizer:
        """Returns the one tokenizer every source was tokenized with."""
        return (*self.train_set, *self.valid_set)[0].tokenizer

    @property
    def uid(self) -> str:
        # order is identifying here -- unlike Tokenizer.sources ("order
        # doesn't identify it"), the same sources in a different order stitch
        # into a different stream at every position past the first source,
        # so this must NOT sort train_set/valid_set away.
        digest = _digest(
            [ts.uid for ts in self.train_set], [ts.uid for ts in self.valid_set]
        )
        if len(self.train_set) > 1 or len(self.valid_set) > 1:
            # The separator changes the stream and therefore its identity.
            digest = _digest(digest, END_OF_TEXT)
        return f"mapped-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("mappeddatasets") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {}  # nothing of its own to write -- see completion_paths

    def completion_paths(self, root: Path) -> list[Path]:
        """Done means every dependency's tokens.bin exists -- not anything
        in this artifact's own (empty) folder. Declaration's undeclared check
        still asks about this artifact's own (empty) files, so it can be
        `new` while every one of these already exists."""
        return [ts.paths(root)["tokens"] for ts in (*self.train_set, *self.valid_set)]

    # -- the bound view ------------------------------------------------------

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: one TokenStream per split, offsets worked out
        from the bin files' lengths. The separator id comes from the split's
        tokenizer, whose tokenizer.json every tokens.bin was written from."""
        # local import: numpy only loads when something actually binds this
        from artifacts.mappeddataset.tokenstream import TokenStream

        for name, sources in (("_train_tokens", self.train_set), ("_valid_tokens", self.valid_set)):
            separator = None
            if len(sources) > 1:
                [separator] = sources[0].tokenizer.bind(root).encode(END_OF_TEXT)
            stream = TokenStream([ts.paths(root)["tokens"] for ts in sources], separator)
            # object.__setattr__ because the model is frozen -- `self` here
            # is the object Artifact.bind read from the stored manifest, never
            # the one bind() was called on, so this is invisible to whoever is
            # still holding the unbound original.
            object.__setattr__(self, name, stream)

    @property
    def bound(self) -> bool:
        return hasattr(self, "_train_tokens")

    def _require(self, doing: str) -> None:
        if not self.bound:
            raise RuntimeError(
                f"{self.uid} has no token streams yet -- bind(root) to build them "
                f"from the sources' own tokens.bin files before {doing}"
            )

    @property
    def train_tokens(self) -> TokenStream:
        self._require("reading train_tokens")
        return self._train_tokens

    @property
    def valid_tokens(self) -> TokenStream:
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
            # one contiguous slice: TokenStream serves nothing else
            window = tokens[batch_id * size : (batch_id + 1) * size + 1].astype(np.int64)
            return window[:-1].reshape(-1, sequence_length), window[1:].reshape(-1, sequence_length)

        return (len(tokens) - 1) // size, get_valid_batch
