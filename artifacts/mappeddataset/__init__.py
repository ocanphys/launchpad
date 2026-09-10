from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from artifacts.core.artifact import Artifact, _digest
from artifacts.sources import Source
from artifacts.tokenizers.bpe import TokenizedSource, Tokenizer

if TYPE_CHECKING:
    from artifacts.mappeddataset.tokenstream import TokenStream


@dataclass(frozen=True)
class MappedDataSet(Artifact):
    """A training set that owns no bytes of its own and no folder of its
    own outputs either: it's done exactly when the TokenizedSources it
    depends on are (see `completion_paths` below), and its bound object is
    a TokenStream per split (artifacts/mappeddataset/tokenstream.py) that
    serves windows straight out of each source's own tokens.bin via memmap.

    A different kind of artifact from artifacts.dataset.DataSet, not just
    a variant of it -- one physically copies tokens, this one never writes
    anything, so each lives in its own sibling package the way tokenizers/
    and sources/ are separate from each other.

    It still gets a folder (`mappeddatasets/<uid>`) and still gets declared
    the same way as everything else -- that's what lets `Artifact.at`,
    resolution, and every notebook that already knows how to find an
    artifact by path find this one too. The folder just never holds
    anything a job wrote; the manifest `artifacts.core.resolve.declare`
    puts there is the only file in it.

    Shared, like DataSet -- no run_id, identity a digest over the tokenizer
    and source lists, the same pattern as Tokenizer/TokenizedSource. The
    difference from DataSet is only whether the bytes get copied: DataSet
    still writes its own train.bin/valid.bin (shared across runs, but a real
    file); this one never writes anything at all.
    """

    producer: ClassVar[str] = "artifacts.mappeddataset.jobs.MappedDataSetJob"

    train_set: tuple[TokenizedSource, ...]
    valid_set: tuple[TokenizedSource, ...]

    @classmethod
    def from_sources(
        cls,
        tokenizer: Tokenizer,
        train_sources: list[Source],
        valid_sources: list[Source],
    ) -> MappedDataSet:
        return cls(
            train_set=tuple(TokenizedSource(tokenizer, s) for s in train_sources),
            valid_set=tuple(TokenizedSource(tokenizer, s) for s in valid_sources),
        )

    @property
    def uid(self) -> str:
        # order is identifying here -- unlike Tokenizer.sources ("order
        # doesn't identify it"), the same sources in a different order stitch
        # into a different virtual stream at every position past the first
        # source, so this must NOT sort train_set/valid_set away.
        digest = _digest(
            [ts.uid for ts in self.train_set], [ts.uid for ts in self.valid_set]
        )
        return f"mapped-{digest}"

    @property
    def artifact_path(self) -> Path:
        return Path("mappeddatasets") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {}  # nothing of its own to write -- see completion_paths

    def completion_paths(self, root: Path) -> list[Path]:
        """Done means every dependency's tokens.bin exists -- not anything
        in this artifact's own (empty) folder. See Artifact.completion_paths
        for why the base class needs this hook at all, and artifacts/core/resolve.py's
        `inspect` for why the pre-declaration new/undeclared check still
        looks at this artifact's own folder rather than this."""
        return [ts.paths(root)["tokens"] for ts in (*self.train_set, *self.valid_set)]

    # -- the bound view ------------------------------------------------------
    #
    # _load builds the view directly from self.train_set/self.valid_set
    # (always present, whether this artifact came from a manifest or a
    # fresh construction) plus each TokenizedSource's own
    # paths(root)["tokens"] -- there is nothing of this artifact's own left
    # to read back, by design.

    def _load(self, root: Path) -> None:
        # local import: numpy only loads when something actually binds this
        from artifacts.mappeddataset.tokenstream import TokenStream

        # object.__setattr__ because the dataclass is frozen -- but `self`
        # here is bind's own private copy (see Artifact.bind), never the
        # object bind() was called on, so this is invisible to whoever is
        # still holding the unbound original.
        object.__setattr__(
            self,
            "_train_tokens",
            TokenStream([ts.paths(root)["tokens"] for ts in self.train_set]),
        )
        object.__setattr__(
            self,
            "_valid_tokens",
            TokenStream([ts.paths(root)["tokens"] for ts in self.valid_set]),
        )

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
