"""Manifest files: where an artifact's manifest lives under a root, reading it
as a dict and writing one. The only module that opens a manifest; the codec
(artifacts.core.manifest) turns the dict into an artifact and back, and
deciding whether a write is allowed is lab.declare's.
"""

import json
from pathlib import Path

import artifacts.core.manifest as codec
from config import STORAGE

MANIFEST = "manifest.json"


def root_of(root: Path | None) -> Path:
    """Returns `root`, or STORAGE when it is None, checked to be an absolute Path.

    A string is refused rather than converted, so one type reaches every join,
    glob and relative_to downstream; a relative root would be discarded by
    `root / artifact_path` and surface as a missing file somewhere else.

        root_of(None) -> Path("/storage")
        root_of(Path("tmp")) -> ValueError (root must be an absolute path)
    """
    resolved = root if root is not None else STORAGE
    if not isinstance(resolved, Path):
        raise TypeError(f"root must be a Path, got {type(resolved).__name__} {resolved!r}")
    if not resolved.is_absolute():
        raise ValueError(f"root must be an absolute path, got {resolved}")
    return resolved


def _manifest_path(root: Path | None, artifact_path: Path | str) -> Path:
    """Returns root / artifact_path / manifest.json, refusing a path that leaves root."""
    wanted = Path(artifact_path)
    if wanted.is_absolute() or ".." in wanted.parts:
        raise ValueError(f"artifact_path must stay under root, got {wanted}")
    return root_of(root) / wanted / MANIFEST


def read_manifest(root: Path | None, artifact_path: Path | str) -> dict:
    """Returns the parsed manifest at root / artifact_path.

    Raises FileNotFoundError when nothing is declared there, and ValueError for
    a path that leaves root or a file that is not JSON, naming the file.

        read_manifest(Path("/storage"), "sources/odyssey")
          -> {"artifact": "artifacts.sources.SourceURL", "declaring_commit": "abc1234", ...}
    """
    path = _manifest_path(root, artifact_path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"nothing declared at {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not readable JSON: {error}") from error


def write_manifest(root: Path | None, artifact_path: Path | str, manifest: dict) -> None:
    """Writes `manifest` to root / artifact_path / manifest.json as
    manifest_json's bytes, creating the folder.

    Overwrites whatever is there: whether a write is allowed is lab.declare's
    decision, made before it calls this.

        write_manifest(Path("/tmp/root"), "sources/odyssey", odyssey.to_manifest())  -> None
    """
    path = _manifest_path(root, artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(codec.manifest_json(manifest))
