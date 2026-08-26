"""Job definitions -- each job imports the artifact type(s) it produces and
registers itself as their producer
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from artifact import Artifact, Checkpoint, DataSet, Source, TokenizedSource, Tokenizer

REGISTRY: dict[
    type[Artifact], type[Job]
] = {}  # artifact type -> the job that produces it


class Job(ABC):
    produces: type[Artifact]  # declared by each subclass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.produces in REGISTRY:
            raise TypeError(
                f"{cls.produces} already registered to {REGISTRY[cls.produces]}"
            )
        REGISTRY[cls.produces] = (
            cls  # registration happens at class-definition time, not lookup time
        )

    @classmethod
    def for_artifact(cls, artifact: Artifact) -> Job:
        """Takes one artifact this job type produces (of cls.produces) and
        returns the job instance that would produce it -- the full output
        set included, even if that means expanding out from the single
        requested artifact (e.g. a discriminated sibling set).

        Default assumes a job with exactly one output, constructed from
        that one artifact -- true for every job below. Override this for a
        job whose real output set is a sibling group: reconstruct it from
        the requested artifact's non-discriminating parameters instead."""
        return cls(artifact)

    @property
    @abstractmethod
    def outputs(self) -> list[Artifact]:
        """The full output set this job produces (usually just one artifact)."""

    @property
    def inputs(self) -> list[Artifact]:
        # extract the list of artifacts the outputs DIRECTLY depend on.
        # Artifact.deps() are the dependencies (Artifact type)
        return [dep for out in self.outputs for dep in out.deps()]

    @abstractmethod
    def run(self, root: Path) -> None:
        """Do the work, writing self.outputs under root."""


class SourceJob(Job):
    produces = Source

    def __init__(self, artifact: Source):
        self.artifact = artifact

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        request = urllib.request.Request(
            self.artifact.url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = (
                response.headers.get_content_type()
            )  # ignores charset, e.g. "text/plain"
            if (
                content_type != "text/plain"
            ):  # text/html etc. would also start with "text/"
                raise ValueError(
                    f"{self.artifact.url} is not a text file (content-type: {content_type})"
                )
            body = response.read().decode("utf-8")

        path = self.artifact.paths(root)["body"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)


class TokenizerJob(Job):
    produces = Tokenizer

    def __init__(self, artifact: Tokenizer):
        self.artifact = artifact

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        tok = self.artifact
        text = "".join(source.paths(root)["body"].read_text() for source in tok.sources)
        vocab = sorted(set(text.split()))[
            : tok.vocab_size
        ]  # mock: first N unique words, not real BPE merges
        path = tok.paths(root)["tokenizer"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": tok.kind,
                    "vocab_size": tok.vocab_size,
                    "special_tokens": list(tok.special_tokens),
                    "trained_on": [source.name for source in tok.sources],
                    "vocab": vocab,
                },
                indent=2,
            )
        )


class TokenizeSourceJob(Job):
    produces = TokenizedSource

    def __init__(self, artifact: TokenizedSource):
        self.artifact = artifact

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        art = self.artifact
        vocab = json.loads(art.tokenizer.paths(root)["tokenizer"].read_text())["vocab"]
        ids = {word: i for i, word in enumerate(vocab)}
        token_ids = [
            ids.get(word, -1)
            for word in art.source.paths(root)["body"].read_text().split()
        ]  # -1 = unk
        path = art.paths(root)["tokens"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            " ".join(map(str, token_ids))
        )  # mock binary encoding as whitespace-joined ids


class DataSetJob(Job):
    produces = DataSet

    def __init__(self, artifact: DataSet):
        self.artifact = artifact

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        ds = self.artifact
        paths = ds.paths(root)
        for path, tokenized_sources in (
            (paths["training set"], ds.train_set),
            (paths["validation set"], ds.valid_set),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            token_ids = " ".join(
                ts.paths(root)["tokens"].read_text() for ts in tokenized_sources
            )
            path.write_text(
                token_ids
            )  # mock: concat the whitespace-joined id strings, same mock encoding as TokenizeSourceJob


class PretrainJob(Job):
    produces = Checkpoint

    def __init__(self, artifact: Checkpoint):
        self.artifact = artifact

    @property
    def outputs(self) -> list[Artifact]:
        return [self.artifact]

    def run(self, root: Path) -> None:
        ck = self.artifact
        train_ids = [
            int(t) for t in ck.dataset.paths(root)["training set"].read_text().split()
        ]
        vocab_size = len(
            json.loads(ck.tokenizer.paths(root)["tokenizer"].read_text())["vocab"]
        )

        loss = 10.0
        for _ in range(ck.step):
            loss *= 0.99  # mock decay, not a real training loop

        path = ck.paths(root)["checkpoint"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "step": ck.step,
                    "config": {
                        "hidden_size": ck.config.hidden_size,
                        "num_layers": ck.config.num_layers,
                        "lr": ck.config.lr,
                        "seed": ck.config.seed,
                    },
                    "vocab_size": vocab_size,
                    "trained_on": ck.dataset.uid,
                    "num_train_tokens": len(train_ids),
                    "final_loss": loss,
                },
                indent=2,
            )
        )
