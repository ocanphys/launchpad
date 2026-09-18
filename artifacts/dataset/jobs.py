from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.dataset import END_OF_TEXT, DataSet

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
            separator = []
            if len(tokenized_sources) > 1:
                tokenizer = tokenized_sources[0].tokenizer.bind(root)
                separator = tokenizer.encode(END_OF_TEXT)
            # Each tokens.bin is raw uint16, including the inserted separator.
            merged = array("H")
            for index, tokenized_source in enumerate(tokenized_sources):
                if index:
                    merged.extend(separator)
                merged.frombytes(tokenized_source.paths(root)["tokens"].read_bytes())
            path.write_bytes(merged.tobytes())
        worker.log.info("done")
