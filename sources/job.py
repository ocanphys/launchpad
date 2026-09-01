import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from dag.job import Job
from sources.artifact import Source

if TYPE_CHECKING:
    from runtime import Worker


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
