"""One-time migration of every pretraining leg to
artifacts.stages.pretraining.Pretraining with its model as one map, the
model's class under "class" beside its fields. A leg arrives either as a
per-architecture class (models.transformer.TransformerPretraining, fields
under `model`) or as Pretraining with the class under `model` and the fields
under `model_parameters`.

    uv run python migrate_model_parameters.py            # preview on the volume
    uv run python migrate_model_parameters.py --write    # rewrite on the volume

Delete this file once the volume is migrated.
"""

import sys

import modal

import main

app = modal.App("launchpad-migrate")

PRETRAINING = "artifacts.stages.pretraining.Pretraining"
MODELS = {
    "models.transformer.TransformerPretraining": "models.transformer.Transformer",
    "models.mamba.MambaPretraining": "models.mamba.Mamba",
}


def migrate(root, write: bool) -> list[tuple[str, dict, dict]]:
    """Returns (folder, old manifest, new manifest) for each manifest under
    `root` that changes, rewriting them only with `write`.

    Every manifest under `root`, rewritten or not, is decoded and checked
    against its folder before any file is written, so one that would not load
    afterwards stops the whole run, naming every folder that would not.
    """
    import json

    from artifacts.core.artifact import MANIFEST, Artifact
    from artifacts.core.manifest import manifest_json

    def converted(node: dict) -> dict:
        dependencies = {name: nested(value) for name, value in node["dependencies"].items()}
        parameters = dict(node["parameters"])
        if node["artifact"] in MODELS:
            fields, model_class = parameters.pop("model"), MODELS[node["artifact"]]
        elif node["artifact"] == PRETRAINING and isinstance(parameters.get("model"), str):
            fields, model_class = parameters.pop("model_parameters"), parameters.pop("model")
        else:
            return {**node, "dependencies": dependencies}
        parameters["model"] = dict(sorted({"class": model_class, **fields}.items()))
        return {**node, "artifact": PRETRAINING, "parameters": dict(sorted(parameters.items())), "dependencies": dependencies}

    def nested(value):
        if isinstance(value, list):
            return [converted(item) for item in value]
        return converted(value) if isinstance(value, dict) else value

    pending, unreadable = [], []
    for path in sorted(root.rglob(MANIFEST)):
        old = json.loads(path.read_text())
        new = converted(old)
        folder = path.parent.relative_to(root)
        try:
            decoded = Artifact.from_manifest(new).artifact_path
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as error:
            unreadable.append(f"{folder}: {error!r}")
            continue
        if decoded != folder:
            unreadable.append(f"{folder}: would decode to {decoded}")
        elif new != old:
            pending.append((path, folder, old, new))
    if unreadable:
        raise ValueError("would not load:\n  " + "\n  ".join(unreadable))
    if write:
        for path, _, _, new in pending:
            path.write_text(manifest_json(new))
    return [(str(folder), old, new) for _, folder, old, new in pending]


# serialized, like local._declare: the container never imports this module or `main`
@app.function(image=main.worker_image, volumes={main.STORAGE: main.volume}, serialized=True)
def _on_volume(write: bool) -> list[tuple[str, dict, dict]]:
    from config import STORAGE, VOLUME_NAME

    volume = modal.Volume.from_name(VOLUME_NAME)
    volume.reload()
    changed = migrate(STORAGE, write)
    if write:
        volume.commit()
    return changed


if __name__ == "__main__":
    write = "--write" in sys.argv
    with app.run():
        changed = _on_volume.remote(write)
    print(f"{'rewrote' if write else 'would rewrite'} {len(changed)} manifests")
    for folder, _, _ in changed:
        print(f"  {folder}")
    if changed:
        folder, old, new = changed[0]
        print(f"\nexample, {folder}:")
        print(f"  before {old['artifact']} {old['parameters']}")
        print(f"  after  {new['artifact']} {new['parameters']['model']}")
