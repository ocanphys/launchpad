from pathlib import Path
from typing import TYPE_CHECKING

from dag.job import Job
from datasets.artifact import DataSet

if TYPE_CHECKING:
    from runtime import Worker


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
