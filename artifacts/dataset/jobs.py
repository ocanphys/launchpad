from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.dataset import DataSet

if TYPE_CHECKING:
    from system.runtime import Worker


class DataSetJob(Job):
    artifact: DataSet

    def __init__(self, artifact: DataSet):
        super().__init__(artifact)
        self.train_set = artifact.train_set
        self.valid_set = artifact.valid_set

    def run(self, root: Path, worker: "Worker") -> None:
        (root / self.artifact.artifact_path).mkdir(parents=True, exist_ok=True)
        paths = self.artifact.paths(root)
        for name, path, tokenized_sources in (
            ("training", paths["training set"], self.train_set),
            ("validation", paths["validation set"], self.valid_set),
        ):
            worker.log.info(
                f"building {name} set from {len(tokenized_sources)} source(s)"
            )
            # each tokens.bin is already raw uint16 (TokenizeSourceJob), so
            # merging is a byte concatenation in the order given -- no parsing
            merged = array("H")
            for tokenized_source in tokenized_sources:
                merged.frombytes(tokenized_source.paths(root)["tokens"].read_bytes())
            path.write_bytes(merged.tobytes())
        worker.log.info("done")
