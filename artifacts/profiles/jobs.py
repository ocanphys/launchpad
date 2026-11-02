"""The job that runs a Profile: `profiling.util.profile` on the GPU the profile was allocated, each
file it returns published whole."""

from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.profiles import Profile
from profiling.util import profile

if TYPE_CHECKING:
    from system.runtime import Worker


class ProfileJob(Job):
    artifact: Profile  # no dependencies: the tokens are random, drawn from the seed

    def run(self, root: Path, worker: "Worker") -> None:
        paths = self.artifact.paths(root)
        for part, text in profile(self.artifact, worker.log).items():
            with worker.publishing(paths[part]) as out:
                out.write(text.encode())
