"""The base job model -- what a job IS, in general: it produces exactly one
artifact, which is all it is given. No concrete job lives here; see
artifacts/sources/, artifacts/tokenizers/, artifacts/tokenized/,
artifacts/dataset/, artifacts/stages/* for those (spec.md, section 4).

One job, one artifact. That artifact may comprise several files, but they
all live in its folder, and declaration (`lab.declare`) has already created
that folder, writing the manifest into it, before run() is called -- so no
job makes its own folder, only subfolders inside it.

Every path in the artifact's `completion_paths` -- the files whose presence
means it is done -- is written through `worker.publishing`, which opens the
file, closes it and renames it into place under a confirmed lease. Nothing a
job writes to the volume that has to appear whole or not at all is written
any other way. A
file that completes nothing is not one of those: a step log is appended to as
the job runs. A checkpoint still goes through it, since a half-written one
would poison the resume that reads it.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.artifact import Artifact

if TYPE_CHECKING:
    # Type-only: `system.runtime.Worker` pulls in `modal`, and the artifact/job
    # modules otherwise have no third-party imports of their own (see
    # main.py's CALL_SOURCE comment) -- a real import here would change that
    # for every notebook that imports a job module just to plan or inspect a
    # run.
    from system.runtime import Worker


class Job(ABC):
    """Produces exactly one artifact, which is all it is given.

    A subclass declares itself with a single line -- `artifact: Tokenizer`.
    That annotation types self.artifact, so the job sees its dependencies
    for what they are. `Artifact.job()` is how a job is found from its
    artifact.

    A job with dependencies binds them in __init__, under the same names the
    artifact gives them, and run() uses those: self.tokenizer, never
    self.tok. What a job reads is then declared in one place instead of
    being spelled out again at each use, and the artifact's own parameters
    stay on self.artifact -- which keeps the two kinds of field apart.
    """

    artifact: Artifact

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "artifact" not in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} must annotate `artifact:` with the type it produces"
            )

    def __init__(self, artifact: Artifact):
        self.artifact = artifact

    @abstractmethod
    def run(self, root: Path, worker: "Worker") -> None:
        """Do the work, writing self.artifact's files under root.

        `worker` is what main.py's run_job holds for the call this job is
        running under (see system.runtime.Worker) -- its `.log` is the logger
        whose records are filed under this call, boot/heartbeat/done lines
        included, so a job's own narration belongs there too, not in a
        print() nothing files.

        A job that reports incremental progress writes it to `worker.progress`
        here, in whatever shape says what it means -- the dashboard shows it
        verbatim, so nothing downstream has to know what a given job counts.
        Its artifact answers the same question off the volume once the call is
        over (`Artifact.durable_progress`).
        """
