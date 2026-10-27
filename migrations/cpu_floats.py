"""Rewrites manifests that store `allocated_resources.cpu` as an int, which
code older than the int-to-float rule wrote, in the bytes today's code
writes. The first migration, run before the pydantic refactor so the volume
scan starts from zero problems:

    modal run local.py::canonicalize_manifests            # dry run
    modal run local.py::canonicalize_manifests --apply    # write

Every stored commit and resources value is kept; no path or definition
changes.
"""

import re
from pathlib import Path

from artifacts.core.artifact import MANIFEST, Artifact
from artifacts.core.manifest import manifest_json

CPU_INT = re.compile(r'(\s*"cpu": )(\d+)(,?)')


def plan(root: Path) -> dict[Path, str]:
    """Returns the text today's code writes for every manifest under `root`
    whose stored bytes differ from it only by int `"cpu"` values.

    Raises ValueError, naming each one, if any manifest does not load or
    differs in anything else, so nothing is rewritten that this script was not
    written for.

    plan(Path("/storage"))
      -> {Path("/storage/tokenizers/bpe-800-f90570d566/manifest.json"): '{\\n  "artifact": ...'}
    """
    planned, refused = {}, []
    memo: dict[str, Artifact] = {}
    for path in sorted(root.rglob(MANIFEST)):
        folder = path.parent.relative_to(root)
        try:
            artifact = Artifact.load(folder, root, memo)
        except (FileNotFoundError, ValueError) as error:
            refused.append(f"{folder}: {error}")
            continue
        stored, rewritten = path.read_text(), manifest_json(artifact.to_manifest())
        if rewritten == stored:
            continue
        before, after = stored.splitlines(), rewritten.splitlines()
        changed = [(b, a) for b, a in zip(before, after) if b != a]
        # each int -> float adds exactly ".0", so the lengths also rule out
        # anything splitlines would hide, such as a missing final newline
        if (
            len(before) == len(after)
            and len(rewritten) - len(stored) == 2 * len(changed)
            and all((m := CPU_INT.fullmatch(b)) and a == f"{m[1]}{m[2]}.0{m[3]}" for b, a in changed)
        ):
            planned[path] = rewritten
        else:
            refused.append(f"{folder}: differs in more than an int cpu")
    if refused:
        raise ValueError("refusing to rewrite anything:\n" + "\n".join(refused))
    return planned
