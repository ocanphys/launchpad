"""The codec: an artifact as a manifest dict and back. A manifest names its
artifact's class under "artifact", records its declaring commit and resources
at the root only, holds its own fields under "parameters" and a definition per
dependency under "dependencies". A param config whose class is not exactly the
one its field names carries that class under "class", at any depth. Nothing
here opens a file (artifacts.core.storage does); `manifest_json` is the text on
disk.
"""

from __future__ import annotations

import json
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import artifacts.core.artifact as core
from artifacts.core.locate import dotted, locate

if TYPE_CHECKING:
    from artifacts.core.artifact import Artifact

# The fields a manifest records but that never identify the artifact.
RECORDED = ("declaring_commit", "allocated_resources")


def arms(annotation: object) -> tuple:
    """Returns the arms of a union annotation, None included, or the annotation alone.

        arms(Model | None) -> (Model, NoneType)
    """
    return get_args(annotation) if get_origin(annotation) in (UnionType, Union) else (annotation,)


def canonical(value: object) -> Any:
    """Returns `value` as JSON data with keys sorted and -0.0 written as 0.0,
    so equal values give equal JSON.

        canonical({"b": -0.0, "a": (1, 2)}) -> {"a": [1, 2], "b": 0.0}
    """
    return json.loads(
        json.dumps(value),
        parse_float=lambda s: float(s) + 0.0,
        object_pairs_hook=lambda pairs: dict(sorted(pairs)),
    )


def members(value: Any, container: type | None) -> list[Artifact]:
    """Returns a dependency field's artifacts in manifest order: a set's
    sorted by artifact path, never by set iteration.

        members(frozenset({iliad, odyssey}), frozenset) -> [iliad, odyssey]
    """
    if container is None:
        return [] if value is None else [value]
    return sorted(value, key=lambda a: a.artifact_path.as_posix()) if container is frozenset else list(value)


def _with_classes(value: object, annotation: object, data: Any) -> object:
    """Returns `data`, the JSON dump of a field holding `value`, with "class"
    added to every param config whose class is not the one its annotation names."""
    if isinstance(value, core.ParamConfig):
        named = [arm for arm in arms(annotation) if isinstance(arm, type) and issubclass(arm, core.ParamConfig)]
        out = {
            field: _with_classes(getattr(value, field), info.annotation, data[field])
            for field, info in type(value).model_fields.items()
        }
        if named != [type(value)]:
            out["class"] = dotted(type(value))
        return out
    if isinstance(value, tuple):
        items = next((get_args(arm) for arm in arms(annotation) if get_origin(arm) is tuple), ())
        if len(items) == 2 and items[1] is Ellipsis:
            items = (items[0],) * len(value)
        return [_with_classes(v, a, d) for v, a, d in zip(value, items or (None,) * len(value), data)]
    return data


def parameters(artifact: Artifact) -> dict:
    """Returns the artifact's own fields as canonical JSON, sorted by name: the
    "parameters" half of its manifest, without the dependencies' manifests.

    A param config holding a subclass of its annotated class, at any depth,
    carries that subclass's dotted path under "class".

        parameters(SourceURL(name="odyssey", url="https://example.org/odyssey.txt"))
          -> {"name": "odyssey", "url": "https://example.org/odyssey.txt"}
    """
    cls = type(artifact)
    names = [n for n in cls.model_fields if n not in cls.dependencies() and n not in RECORDED]
    dumped = artifact.model_dump(mode="json", include=set(names), serialize_as_any=True)
    return canonical({n: _with_classes(getattr(artifact, n), cls.model_fields[n].annotation, dumped[n]) for n in names})


def _definition_manifest(artifact: Artifact) -> dict:
    """Returns the artifact's class, parameters and its dependencies'
    definitions, recursively: what a parent embeds."""
    deps = {}
    for name, container in type(artifact).dependencies().items():
        value = getattr(artifact, name)
        if container is None:
            deps[name] = None if value is None else _definition_manifest(value)
        else:
            deps[name] = [_definition_manifest(member) for member in members(value, container)]
    return {"artifact": dotted(type(artifact)), "parameters": parameters(artifact), "dependencies": deps}


def to_manifest(artifact: Artifact) -> dict:
    """Returns the artifact as a manifest dict, enough to rebuild it from
    nothing else: its class, declaring commit, resources, own parameters, and
    the definition of each artifact it is built from.

    Only the root records `declaring_commit` and `allocated_resources`; each
    dependency records its own at its own path. The five keys keep this order
    and everything below them is sorted, so the text is a pure function of the
    artifact.

        to_manifest(odyssey) -> {"artifact": "artifacts.sources.SourceURL", "declaring_commit": "abc1234",
                                 "allocated_resources": {...}, "parameters": {...}, "dependencies": {}}
    """
    definition = _definition_manifest(artifact)
    resources = artifact.allocated_resources
    return {
        "artifact": definition["artifact"],
        "declaring_commit": artifact.declaring_commit,
        "allocated_resources": None if resources is None else canonical(resources.model_dump(mode="json")),
        "parameters": definition["parameters"],
        "dependencies": definition["dependencies"],
    }


def _decoded(data: object) -> object:
    """Returns JSON `data` as a parameter's value: arrays as tuples, and an
    object naming a "class" as an instance of that param config."""
    if isinstance(data, list):
        return tuple(_decoded(item) for item in data)
    if not isinstance(data, dict):
        return data
    fields = {key: _decoded(value) for key, value in data.items() if key != "class"}
    if "class" not in data:
        return fields
    cls = locate(data["class"])
    if not (isinstance(cls, type) and issubclass(cls, core.ParamConfig)):
        raise TypeError(f"{data['class']} is not a param config")
    return cls.model_validate(fields)


def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
    """Returns the artifact a manifest describes and the tree under it, unbound.

    The root keeps the declaring commit and resources its manifest records; a
    dependency comes back with both None, since they are recorded at its own
    path. `memo`, one dict shared across calls, hands back the same instance
    for a subtree decoded before (the dataset under every leg of a run).

        from_manifest(odyssey.to_manifest()) == odyssey -> True
    """
    key = json.dumps(data, sort_keys=True) if memo is not None else ""
    if memo is not None and key in memo:
        return memo[key]
    cls = locate(data["artifact"])
    if not (isinstance(cls, type) and issubclass(cls, core.Artifact)):
        raise TypeError(f"manifest target must be an Artifact, got {cls}")
    containers = cls.dependencies()
    inputs = {name: _decoded(value) for name, value in data["parameters"].items()}
    for name, value in data["dependencies"].items():
        container = containers[name]
        if container is None:
            inputs[name] = None if value is None else from_manifest(value, memo)
        else:
            inputs[name] = container(from_manifest(item, memo) for item in value)
    artifact = cls.model_validate(
        {
            **inputs,
            "declaring_commit": data.get("declaring_commit"),
            "allocated_resources": _decoded(data.get("allocated_resources")),
        }
    )
    if memo is not None:
        memo[key] = artifact
    return artifact


def manifest_json(manifest: dict) -> str:
    """Returns one manifest as the text that belongs on disk.

    allow_nan=False because Python writes NaN and Infinity as bare words and
    reads them back, while no other JSON reader accepts either.

        manifest_json({"artifact": "artifacts.sources.SourceURL", ...}) -> '{\\n  "artifact": ...}\\n'
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
