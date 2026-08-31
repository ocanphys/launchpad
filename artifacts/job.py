"""Job definitions -- each job annotates the artifact type it produces and
registers itself as that type's producer.

One job, one artifact. That artifact may comprise several files, but they
all live in its folder, and the resolver has already created that folder
(writing the manifest into it) before run() is called -- so no job makes
directories of its own.
"""

import json
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from artifact import (
    Artifact,
    DataSet,
    Pretraining,
    Source,
    TokenizedSource,
    Tokenizer,
)

if TYPE_CHECKING:
    # Type-only: `runtime.Worker` pulls in `modal`, and artifact.py/job.py/
    # resolve.py otherwise have no third-party imports of their own (see
    # main.py's CALL_SOURCE comment) -- a real import here would change that
    # for every notebook that imports job.py just to plan or inspect a run.
    from runtime import Worker

REGISTRY: dict[type[Artifact], type["Job"]] = {}  # artifact type -> its producer


class Job(ABC):
    """Produces exactly one artifact, which is all it is given.

    A subclass declares itself with a single line -- `artifact: Tokenizer`.
    That annotation registers the job as Tokenizer's producer and types
    self.artifact, so the job sees its dependencies for what they are.

    A job with dependencies binds them in __init__, under the same names the
    artifact gives them, and run() uses those: self.tokenizer, never
    self.tok. What a job reads is then declared in one place instead of
    being spelled out again at each use, and the artifact's own parameters
    stay on self.artifact -- which keeps the two kinds of field apart.
    """

    artifact: Artifact

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        produces = cls.__annotations__.get("artifact")
        if produces is None:
            raise TypeError(
                f"{cls.__name__} must annotate `artifact:` with the type it produces"
            )
        if produces in REGISTRY:
            raise TypeError(f"{produces} already registered to {REGISTRY[produces]}")
        REGISTRY[produces] = (
            cls  # registration happens at class-definition time, not lookup time
        )

    def __init__(self, artifact: Artifact):
        self.artifact = artifact

    @abstractmethod
    def run(self, root: Path, worker: "Worker") -> None:
        """Do the work, writing self.artifact's files under root.

        `worker` is what main.py's run_job holds for the call this job is
        running under (see runtime.Worker) -- its `.log` is the file that
        call's whole story lands in, boot/heartbeat/done lines included, so
        a job's own narration belongs there too, not in a print() logs.py
        already won't capture.

        This job's own class name is not repeated into each message: it's
        resolved server-side from the artifact's manifest (see main.py's
        artifact_job_name) and shown as its own column when the log is read
        back, so messages here should just be plain narration.
        """


class SourceJob(Job):
    artifact: Source  # no dependencies: a source is downloaded, not derived

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info(f"downloading {self.artifact.name} from {self.artifact.url}")

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

        self.artifact.paths(root)["raw text"].write_text(body)
        worker.log.info(f"wrote {len(body)} chars for {self.artifact.name}")


class TokenizerJob(Job):
    artifact: Tokenizer

    def __init__(self, artifact: Tokenizer):
        super().__init__(artifact)
        self.sources = artifact.sources

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info(
            f"training {self.artifact.kind} tokenizer "
            f"(vocab_size={self.artifact.vocab_size}) on {len(self.sources)} source(s)"
        )

        # sorted by uid, matching Tokenizer.uid: if source order doesn't change
        # which tokenizer this is, it mustn't change what gets trained either
        text = "".join(
            source.paths(root)["raw text"].read_text()
            for source in sorted(self.sources, key=lambda s: s.uid)
        )
        vocab = sorted(set(text.split()))[
            : self.artifact.vocab_size
        ]  # mock: first N unique words, not real BPE merges
        self.artifact.paths(root)["tokenizer"].write_text(
            json.dumps(
                {
                    "kind": self.artifact.kind,
                    "vocab_size": self.artifact.vocab_size,
                    "special_tokens": list(self.artifact.special_tokens),
                    "trained_on": sorted(source.uid for source in self.sources),
                    "vocab": vocab,
                },
                indent=2,
            )
        )
        worker.log.info(f"trained, vocab has {len(vocab)} entries")


class TokenizeSourceJob(Job):
    artifact: TokenizedSource

    def __init__(self, artifact: TokenizedSource):
        super().__init__(artifact)
        self.tokenizer = artifact.tokenizer
        self.source = artifact.source

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info(f"tokenizing {self.source.name}")

        vocab = json.loads(self.tokenizer.paths(root)["tokenizer"].read_text())["vocab"]
        ids = {word: i for i, word in enumerate(vocab)}
        token_ids = [
            ids.get(word, -1)
            for word in self.source.paths(root)["raw text"].read_text().split()
        ]  # -1 = unk
        self.artifact.paths(root)["tokens"].write_text(
            " ".join(map(str, token_ids))
        )  # mock binary encoding as whitespace-joined ids

        unk = sum(1 for i in token_ids if i == -1)
        worker.log.info(
            f"wrote {len(token_ids)} tokens for {self.source.name} ({unk} unk)"
        )


class DataSetJob(Job):
    artifact: DataSet

    def __init__(self, artifact: DataSet):
        super().__init__(artifact)
        self.train_set = artifact.train_set
        self.valid_set = artifact.valid_set

    def run(self, root: Path, worker: "Worker") -> None:
        paths = self.artifact.paths(root)
        for name, path, tokenized_sources in (
            ("training", paths["training set"], self.train_set),
            ("validation", paths["validation set"], self.valid_set),
        ):
            worker.log.info(f"building {name} set from {len(tokenized_sources)} source(s)")
            path.write_text(
                " ".join(
                    tokenized_source.paths(root)["tokens"].read_text()
                    for tokenized_source in tokenized_sources
                )
            )  # mock: concat the id strings, same encoding as TokenizeSourceJob
        worker.log.info("done")


class PretrainJob(Job):
    artifact: Pretraining

    def __init__(self, artifact: Pretraining):
        super().__init__(artifact)
        self.dataset = artifact.dataset
        self.tokenizer = artifact.tokenizer

    def run(self, root: Path, worker: "Worker") -> None:
        train_ids = [
            int(token)
            for token in self.dataset.paths(root)["training set"].read_text().split()
        ]
        vocab_size = len(
            json.loads(self.tokenizer.paths(root)["tokenizer"].read_text())["vocab"]
        )

        model_parameters = self.artifact.model_parameters
        config = self.artifact.config
        folder = root / self.artifact.artifact_path
        every = config.checkpoint_every
        steps = list(range(every, config.total_steps + 1, every))
        if not steps or steps[-1] != config.total_steps:
            steps.append(config.total_steps)

        worker.log.info(
            f"{config.total_steps} steps, "
            f"hidden_size={model_parameters.hidden_size}, num_layers={model_parameters.num_layers}, "
            f"{len(train_ids)} train tokens"
        )

        loss, done = 10.0, 0
        for step in steps:
            for _ in range(step - done):
                loss *= 0.99  # mock decay, not a real training loop
            done = step
            body = json.dumps(
                {
                    "step": step,
                    "model_parameters": {
                        "hidden_size": model_parameters.hidden_size,
                        "num_layers": model_parameters.num_layers,
                    },
                    "config": {
                        "batch_size": config.batch_size,
                        "lr": config.lr,
                        "seed": config.seed,
                    },
                    "vocab_size": vocab_size,
                    "trained_on": self.dataset.uid,
                    "num_train_tokens": len(train_ids),
                    "loss": loss,
                },
                indent=2,
            )
            # only the final step is declared; intermediates are undeclared --
            # written into the folder, absent from `files`, so nothing waits
            # on one that recovery skipped past
            if step == config.total_steps:
                self.artifact.paths(root)["checkpoint"].write_text(body)
            else:
                (folder / f"checkpoint_{step}.txt").write_text(body)
            worker.log.info(f"step {step}/{config.total_steps} loss={loss:.4f}")
        self.artifact.paths(root)["progress"].write_text(
            json.dumps({"step": config.total_steps, "complete": True}, indent=2)
        )  # last, so `done` can't be observed before the checkpoint is durable
        worker.log.info("training complete")
