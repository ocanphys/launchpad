"""Jobs: the code that builds one artifact.

Each family pairs its artifact class with exactly one job class, in its `jobs.py`: no
alternative producers, no job ids, no per-run config files. A virtual artifact, such as
`MappedDataSet`, has none. The two name each other, the job with a class annotation and the
artifact with a dotted string, which only `Artifact.job()` imports, in the worker about to run it:

    class Tokenizer(Artifact):
        producer: ClassVar[str] = "artifacts.tokenizers.bpe.jobs.TokenizerJob"

    class TokenizerJob(Job):
        artifact: Tokenizer

**What a job may write.** Declaration has created the artifact's folder and written its manifest
before `run` is called, and a job writes only inside that folder:

- Every completion file goes through `worker.publishing`, which hands the job an open temporary
  file, `{filename}.{call_id}.tmp`, and when the job closes it, confirms the lease and renames it
  into place. A file exists whole or not at all, and only the call holding the lease publishes.
  Naming, opening or renaming a temporary path in a job is a defect.
- A published file is never appended to, replaced or deleted.
- Auxiliary files in locations the type owns, a leg's checkpoints and step log, never count
  toward completion. A checkpoint still goes through `publishing`, since a half-written one would
  poison the resume that reads it; the step log is appended to as the job runs.

**Resuming.** Publication is per file, so a crash between files leaves the artifact partial. On a
partial artifact a job checks that what is there is consistent with the definition, keeps it, and
publishes what is missing, or raises.

**By hand.** A job runs from a cell with no lease and console logging:
`artifact.job().run(root, lab.worker)`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.artifact import Artifact

if TYPE_CHECKING:
    # Type-only: `system.runtime.Worker` pulls in `modal`, which a notebook that imports a job
    # module just to inspect a run would otherwise import too.
    from system.runtime import Worker


class Job(ABC):
    """Produces exactly one artifact, which is all it is given.

    A subclass annotates `artifact:` with the type it produces, which types `self.artifact`;
    defining one without it raises TypeError. A job with dependencies binds them in `__init__`
    under the names the artifact gives them, `self.tokenizer` and never `self.tok`, and `run` uses
    those, while the artifact's own parameters stay on `self.artifact`.

    Args:
        artifact: The artifact to produce.
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
        """Writes `self.artifact`'s files under `root`.

        A job narrates through `worker.log`, whose records are filed under this call, never
        `print()`. One that reports progress writes it to `worker.progress`, in whatever shape says
        what it means, since the dashboard shows it verbatim; once the call is over its artifact
        answers the same question from the volume (`Artifact.durable_progress`).

        Args:
            root: The storage root the artifact's folder is under.
            worker: The call this job runs under (`system.runtime.Worker`): its logger, its
                progress and the lease `publishing` confirms.
        """
