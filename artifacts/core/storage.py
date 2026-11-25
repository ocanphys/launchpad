"""Manifest files: where an artifact's manifest lives under a root, reading it as a dict and
writing one. The only module that opens a manifest: the codec turns the dict into an artifact
and back, and whether a write is allowed is `lab.declare`'s decision.
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
        root_of(Path("tmp")) -> ValueError: root must be an absolute path, got tmp

    Args:
        root: A storage root, or None for STORAGE.

    Returns:
        The root to use.
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
    """Returns the parsed manifest at `root / artifact_path`.

        read_manifest(Path("/storage"), "sources/odyssey")
        -> {"artifact": "artifacts.sources.SourceURL", "declaring_commit": "abc1234", ...}

    Args:
        root: The storage root; None for STORAGE.
        artifact_path: The artifact's folder, relative to root.

    Returns:
        The manifest, as a dict.

    Raises:
        FileNotFoundError: Nothing is declared there.
        ValueError: The path leaves root, or the file is not JSON, naming the file.
    """
    path = _manifest_path(root, artifact_path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"nothing declared at {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not readable JSON: {error}") from error


def write_manifest(root: Path | None, artifact_path: Path | str, manifest: dict) -> None:
    """Writes `manifest` to `root / artifact_path / manifest.json` as `manifest_json`'s text,
    creating the folder and overwriting whatever is there.

        write_manifest(Path("/tmp/root"), "sources/odyssey", odyssey.to_manifest())

    Args:
        root: The storage root; None for STORAGE.
        artifact_path: The artifact's folder, relative to root.
        manifest: The manifest to write, as `to_manifest` returns it.
    """
    path = _manifest_path(root, artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(codec.manifest_json(manifest))
