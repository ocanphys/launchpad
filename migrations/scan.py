"""Whether the code on this checkout reads every manifest under a root and
writes it back unchanged. Run against the volume from a laptop with
`modal run local.py::scan_volume`.
"""

from itertools import zip_longest
from pathlib import Path

from artifacts.core.artifact import Artifact
from artifacts.core.manifest import manifest_json
from artifacts.core.storage import MANIFEST


def scan(root: Path) -> tuple[int, list[str]]:
    """Returns how many manifests are under `root`, and one line for each that
    does not load, or that loads but re-encodes to different bytes, naming the
    first line that differs as stored -> rewritten.

    scan(Path("/storage")) -> (42, [])
    scan(Path("/storage")) -> (42, ['sources/odyssey: line 3: \\'  "commit": "abc"\\' -> ...'])
    """
    manifests = sorted(root.rglob(MANIFEST))
    problems = []
    memo: dict[str, Artifact] = {}
    for path in manifests:
        folder = path.parent.relative_to(root)
        try:
            artifact = Artifact.load(folder, root, memo)
        except (FileNotFoundError, ValueError) as error:
            problems.append(f"{folder}: {error}")
            continue
        stored, rewritten = path.read_text().splitlines(), manifest_json(artifact.to_manifest()).splitlines()
        for number, (before, after) in enumerate(zip_longest(stored, rewritten), start=1):
            if before != after:
                problems.append(f"{folder}: line {number}: {before!r} -> {after!r}")
                break
    return len(manifests), problems
