"""Moves a root to the identity change: no `device` in any model, a leg's
`lineage_hash` over its encoded parameters, and `commit` renamed
`declaring_commit` and recorded at the root only. Run from a laptop:

    modal run local.py::migrate_identity            # dry run
    modal run local.py::migrate_identity --apply    # move and write

Every stored declaring commit and resources value is kept. Only pretraining
legs move; every manifest is rewritten, since its root key is renamed.
"""

import json
from collections import Counter
from pathlib import Path

from artifacts.core.artifact import Artifact
from artifacts.core.SGD.training import Training
from artifacts.core.storage import MANIFEST, read_manifest, write_manifest
from config import CALL_HISTORY


def _converted(data: dict, root: bool) -> dict:
    """Returns an old-format manifest in the new format: `device` dropped from
    its model, `commit` renamed at the root, and the recorded fields dropped
    from every embedded copy."""

    def embedded(value: dict | list | None) -> dict | list | None:
        if isinstance(value, list):
            return [_converted(item, False) for item in value]
        return None if value is None else _converted(value, False)

    parameters = dict(data["parameters"])
    if isinstance(parameters.get("model"), dict):
        parameters["model"] = {k: v for k, v in parameters["model"].items() if k != "device"}
    recorded = {"declaring_commit": data["commit"], "allocated_resources": data["allocated_resources"]} if root else {}
    return {
        "artifact": data["artifact"],
        **recorded,
        "parameters": parameters,
        "dependencies": {name: embedded(value) for name, value in data["dependencies"].items()},
    }


def plan(root: Path) -> dict[Path, Artifact]:
    """Returns every folder under `root` whose manifest is in the old format,
    mapped to the artifact it decodes to in the new one; that artifact's
    `artifact_path` is where the folder goes.

    Raises ValueError naming every refusal, before anything is written: a
    manifest that does not decode, two folders that become one path (legs
    that differed only in `device`), or a new path another folder holds.

        plan(Path("/storage"))
          -> {Path("runs/GOLDEN/pretraining/0-1000-639552a92c"): Pretraining(run_id='GOLDEN', ...), ...}
    """
    planned, refused = {}, []
    memo: dict[str, Artifact] = {}
    for path in sorted(root.rglob(MANIFEST)):
        folder = path.parent.relative_to(root)
        try:
            data = read_manifest(root, folder)
            if "commit" not in data:
                continue
            planned[folder] = Artifact.from_manifest(_converted(data, root=True), memo)
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as error:
            refused.append(f"{folder}: does not decode: {error!r}")
    targets = Counter(artifact.artifact_path for artifact in planned.values())
    for folder, artifact in planned.items():
        new = artifact.artifact_path
        if targets[new] > 1:
            refused.append(f"{folder}: becomes {new}, as another folder does")
        elif new != folder and (root / new).exists():
            refused.append(f"{folder}: becomes {new}, which is already there")
    if refused:
        raise ValueError("refusing to migrate anything:\n" + "\n".join(refused))
    return planned


def moves(planned: dict[Path, Artifact]) -> dict[str, str]:
    """Returns the planned folders whose path changes, old path to new.

        moves(plan(Path("/storage")))
          -> {"runs/GOLDEN/pretraining/0-1000-639552a92c": "runs/GOLDEN/pretraining/0-1000-69f057c312"}
    """
    return {
        folder.as_posix(): artifact.artifact_path.as_posix()
        for folder, artifact in planned.items()
        if artifact.artifact_path != folder
    }


def apply(root: Path, planned: dict[Path, Artifact]) -> None:
    """Moves each planned folder, whole, to its artifact's path, writes its new
    manifest there, renames a moved leg's notebook copy, and rekeys
    `call_history.json` from old paths to new.

    A folder moves with one rename, so a leg's model.pt, failsafes, step log
    and call logs go with it. Interrupted, it is finished by planning and
    applying again: a moved folder still holding its old manifest is planned
    in place.

        apply(Path("/storage"), plan(Path("/storage"))) -> None
    """
    for folder, artifact in planned.items():
        new = artifact.artifact_path
        if new != folder:
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            (root / folder).rename(root / new)
            if isinstance(artifact, Training):  # a leg's uid ends in its folder's name
                copies = root / "runs" / artifact.run_id
                old_copy = copies / f"declare-{artifact.uid.removesuffix(new.name)}{folder.name}.ipynb"
                if old_copy.exists():
                    old_copy.rename(copies / f"declare-{artifact.uid}.ipynb")
        write_manifest(root, new, artifact.to_manifest())
    history_file = root / CALL_HISTORY
    if history_file.exists():
        moved = moves(planned)
        history = json.loads(history_file.read_text())
        history_file.write_text(json.dumps({moved.get(path, path): grants for path, grants in history.items()}))
