from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.mappeddataset import MappedDataSet

if TYPE_CHECKING:
    from system.runtime import Worker


class MappedDataSetJob(Job):
    """Exists so MappedDataSet's `producer` has something to point at --
    MappedDataSet declares no files of its own (see its `completion_paths`
    override), so there is nothing for this job to write.
    `artifacts.core.resolve.plan(dag, pending_only=True)` drops any artifact
    that already reads `done`, and a MappedDataSet reads `done` the moment
    its sources are tokenized, before this could ever run -- so in practice
    this body never executes. It exists for the case that isn't practice: a
    plan built and run before its sources exist yet, where something still
    has to occupy this artifact's slot in the order.
    """

    artifact: MappedDataSet

    def run(self, root: Path, worker: "Worker") -> None:
        worker.log.info("virtual dataset -- nothing to write, done follows its sources")
