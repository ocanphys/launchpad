"""Manifests: an artifact's definition as JSON, enough to rebuild it and everything it is built
from. Nothing here opens a file; `artifacts.core.storage` does.

Every declared artifact owns `root / artifact_path / "manifest.json"`, with its dependencies'
definitions embedded recursively, so one manifest decodes and compares on its own. `bpe`, as
declared:

    {
      "artifact": "artifacts.tokenizers.bpe.Tokenizer",
      "declaring_commit": "abc1234",
      "allocated_resources": {"cpu": null, "gpu_count": null, "gpu_type": null},
      "tags": {"experiment": ["lr_study"]},
      "parameters": {"special_tokens": ["<pad>"], "vocab_size": 1000},
      "dependencies": {
        "sources": [
          {
            "artifact": "artifacts.sources.SourceURL",
            "parameters": {"name": "odyssey", "url": "https://example.org/odyssey.txt"},
            "dependencies": {}
          }
        ]
      }
    }

`"artifact"` is the concrete class's dotted path. `"parameters"` and `"dependencies"` split the
class's fields by annotation (`Artifact.dependencies`), so `sources=(odyssey,)` becomes an array
of definitions while `special_tokens`, also an array, stays a parameter. An embedded definition
has only `artifact`, `parameters` and `dependencies`: the recorded fields (`Artifact.RECORDED`)
are at the root of each artifact's own manifest.

**Which class a param config is.** A param config whose class is exactly its field's annotation
is written as its fields alone, so that class can be renamed without migrating anything. A
subclass, at any depth, adds its dotted path under `"class"`, and a field annotated with a union
of param configs writes it on every value:

    "model": {"class": "models.transformer.Transformer", "d_model": 128, "num_heads": 16, ...}

A class a manifest names is stored format: moving it leaves the manifest unreadable until it is
rewritten.

**Canonical text.** The text is a pure function of the artifact: the six top-level keys in the
order above (three in an embedded definition), nested maps sorted by key, a set of dependencies
as an array sorted by artifact path, `-0.0` written as `0.0`, NaN and infinity refused,
`indent=2`, UTF-8 unescaped, and a final newline. Two invariants follow:

- `from_manifest(to_manifest(a)) == a`, at the same path, and a decoded manifest re-encodes to
  exactly its own text.
- `==` artifacts embed as identical bytes in any parent, though their own manifests may differ in
  recorded fields. Nothing compares manifest text to decide whether two definitions agree.

There is no format version: one program reads and writes manifests, so a format change
rewrites every manifest on the volume.

**Trusted classes.** Decoding imports the classes a manifest names, with no allow-list: the
volume is written only by `lab.declare`, so a wrong path fails to import, and `Artifact.load`
reports the manifest unreadable.
"""

from __future__ import annotations

import json
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import artifacts.core.artifact as core
from artifacts.core.locate import dotted, locate

if TYPE_CHECKING:
    from artifacts.core.artifact import Artifact


def arms(annotation: object) -> tuple:
    """Returns the arms of a union annotation, None included, or the annotation alone.

        arms(Model | None) -> (Model, NoneType)

    Args:
        annotation: A field's annotation.

    Returns:
        The union's arms, or a one-tuple of the annotation.
    """
    return get_args(annotation) if get_origin(annotation) in (UnionType, Union) else (annotation,)


def canonical(value: object) -> Any:
    """Returns `value` as JSON data with keys sorted and -0.0 written as 0.0, so equal values give
    equal JSON.

        canonical({"b": -0.0, "a": (1, 2)}) -> {"a": [1, 2], "b": 0.0}

    Args:
        value: Any JSON-serializable value.

    Returns:
        The same value as plain JSON data, canonical.
    """
    return json.loads(
        json.dumps(value),
        parse_float=lambda s: float(s) + 0.0,
        object_pairs_hook=lambda pairs: dict(sorted(pairs)),
    )


def members(value: Any, container: type | None) -> list[Artifact]:
    """Returns a dependency field's artifacts in manifest order: a set's sorted by artifact path,
    never by set iteration.

        members(frozenset({iliad, odyssey}), frozenset) -> [iliad, odyssey]

    Args:
        value: The field's value.
        container: The field's container, from `Artifact.dependencies`: tuple, frozenset, or None
            for a single artifact.

    Returns:
        The artifacts it holds, none for a None value.
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
    """Returns the artifact's own fields as canonical JSON, sorted by name: its manifest's
    `"parameters"`, a subclass of a param config's annotated class carrying `"class"`.

        parameters(odyssey) -> {"name": "odyssey", "url": "https://example.org/odyssey.txt"}

    Args:
        artifact: The artifact to encode.

    Returns:
        Each own field's name and its JSON value.
    """
    cls = type(artifact)
    names = [n for n in cls.model_fields if n not in cls.dependencies() and n not in cls.RECORDED]
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
    """Returns the artifact as a manifest dict: its class, recorded fields, own parameters, and the
    definition of each artifact it is built from, without their recorded fields.

        to_manifest(odyssey)
        -> {"artifact": "artifacts.sources.SourceURL", "declaring_commit": None,
            "allocated_resources": {"cpu": None, "gpu_count": None, "gpu_type": None}, "tags": {},
            "parameters": {"name": "odyssey", "url": "https://example.org/odyssey.txt"},
            "dependencies": {}}

    Args:
        artifact: The artifact to encode.

    Returns:
        The manifest, as a JSON-ready dict.
    """
    definition = _definition_manifest(artifact)
    resources, tags = artifact.allocated_resources, artifact.tags
    return {
        "artifact": definition["artifact"],
        "declaring_commit": artifact.declaring_commit,
        "allocated_resources": None if resources is None else canonical(resources.model_dump(mode="json")),
        "tags": None if tags is None else canonical(tags),
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

    The root keeps the recorded fields its manifest holds, and a dependency comes back with all
    three None. Every validator runs, so a manifest that breaks an invariant cannot be loaded. Each
    dependency is decoded from its own `"artifact"` key before the artifact is validated in Python
    mode, never handed to pydantic as a dict, which would guess between classes with the same
    fields.

        from_manifest(odyssey.to_manifest()) == odyssey -> True

    Args:
        data: A manifest, as `to_manifest` returns it.
        memo: Decoded artifacts keyed by manifest, shared across calls, so a subtree many
            manifests embed (the dataset under every leg of a run) is one instance. `bind` never
            passes one, since it attaches files to the instance it loads. None for none.

    Returns:
        The artifact, unbound.

    Raises:
        TypeError: A class the manifest names is not an Artifact or a param config.
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
            "tags": data.get("tags"),
        }
    )
    if memo is not None:
        memo[key] = artifact
    return artifact


def manifest_json(manifest: dict) -> str:
    """Returns one manifest as the text that belongs on disk.

    It refuses NaN and infinity, which Python writes as bare words and reads back while no other
    JSON reader accepts either.

        manifest_json({"artifact": "artifacts.sources.SourceURL"})
        -> '{\\n  "artifact": "artifacts.sources.SourceURL"\\n}\\n'

    Args:
        manifest: A manifest, as `to_manifest` returns it.

    Returns:
        Its canonical text.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
