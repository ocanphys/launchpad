import urllib.request
import zlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from artifacts.core.job import Job
from artifacts.sources import SourceURL

if TYPE_CHECKING:
    from system.runtime import Worker

# zlib's window size for the gzip container, the one wbits value that reads a
# .gz stream rather than a bare zlib or deflate one.
GZIP_WINDOW = 31


class SourceURLJob(Job):
    artifact: SourceURL  # no dependencies: a source is downloaded, not derived

    def run(self, root: Path, worker: "Worker") -> None:
        url = self.artifact.url
        worker.log.info(f"downloading {self.artifact.name} from {url}")

        # Streamed into the file `worker.publishing` hands over, never held
        # in memory: these corpora run to gigabytes.
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        body = self.artifact.paths(root)["raw text"]
        with worker.publishing(body) as out, urllib.request.urlopen(request, timeout=30) as response:
            # The failure this catches is a login wall or a 404 page served
            # with a 200; anything else is taken at its word, since a file
            # behind an LFS redirect arrives as octet-stream whatever it holds.
            if response.headers.get_content_type() == "text/html":
                raise ValueError(f"{url} served a web page, not a file")
            total = int(response.headers.get("Content-Length") or 0)

            gunzip = (
                zlib.decompressobj(GZIP_WINDOW)
                if urlsplit(url).path.endswith(".gz")
                else None
            )
            downloaded = written = 0
            while chunk := response.read(1 << 20):
                downloaded += len(chunk)
                written += out.write(gunzip.decompress(chunk) if gunzip else chunk)
                # done/total count bytes off the wire, the half that has a
                # known end. Unzipping runs in the same pass rather than as
                # a second phase, so what it has written rides along.
                phase = "downloading"
                if gunzip:
                    phase = f"downloading ({written} unzipped)"
                worker.progress.update(
                    {"phase": phase, "done": downloaded, "total": total}
                )
            if gunzip:
                written += out.write(gunzip.flush())
                if not gunzip.eof or gunzip.unused_data:
                    raise ValueError(f"{url} is not a single complete gzip stream")

        worker.log.info(
            f"wrote {written} bytes for {self.artifact.name}, downloaded {downloaded}"
        )
