"""Artifact <-> manifest dict <-> bytes on disk.

A parameter is encoded and decoded by pydantic against its field's annotation
(artifacts.core.schema), so its annotation alone says how to rebuild it; a
dependency is a nested manifest naming its own class. `normalize` is the round
trip every constructor runs, so a concrete artifact already equals what its
manifest decodes to. Nothing here touches storage.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from artifacts.core.locate import dotted, locate
from artifacts.core.schema import annotations, dependencies, is_artifact, polymorphic

if TYPE_CHECKING:
    from artifacts.core.artifact import Artifact

# strict: no str as int, bool as int or list as tuple, though an int still
# becomes a float. revalidate_instances: a record is checked field by field,
# never trusted for its type.
_STRICT = ConfigDict(
    strict=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
)


@cache
def _field(annotation: object) -> type[BaseModel]:
    """Returns a model whose one field, `value`, is annotated `annotation`.

    pydantic takes no config for a bare dataclass, while a model's config
    reaches every record nested in its field.
    """
    return create_model("Field", __config__=_STRICT, value=(annotation, ...))


@contextmanager
def _named(path: str):
    """Reraises a ValidationError as a TypeError, or a ValueError where a
    validator refused the value, naming `path` and the field inside it."""
    try:
        yield
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        inside = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in first["loc"][1:])
        kind = ValueError if first["type"] in ("value_error", "finite_number") else TypeError
        raise kind(f"{path}{inside}: {first['msg']}") from error


def _encode(value: object, annotation: object) -> object:
    """Returns `value`, annotated `annotation`, as JSON data with keys sorted.

    Adding 0.0 turns -0.0 into 0.0: the two compare equal, so they must encode
    equal.
    """
    text = _field(annotation).model_construct(value=value).model_dump_json()
    return json.loads(
        text,
        parse_float=lambda s: float(s) + 0.0,
        object_pairs_hook=lambda pairs: dict(sorted(pairs)),
    )["value"]


def _decode(data: object, annotation: object, path: str) -> object:
    """Returns the value JSON `data` encodes as `annotation`, validated
    strictly; `path` names it in an error."""
    with _named(path):
        return _field(annotation).model_validate_json(json.dumps({"value": data})).value


def normalize(value: object, annotation: object, path: str) -> object:
    """Returns `value` as its manifest decodes it: ints in float fields as
    floats, -0.0 as 0.0, records rebuilt through their constructors.

    Raises unless `value` is exactly an `annotation`: a bool is not an int, a
    list is not a tuple, and a subclass is not the record its field names.
    """
    with _named(path):
        validated = _field(annotation).model_validate({"value": value}).value
    restored = _decode(_encode(validated, annotation), annotation, path)
    if restored != value:
        expected = getattr(annotation, "__qualname__", annotation)
        raise TypeError(f"{path}: expected exactly {expected}, got {value!r}")
    return restored


def members(value: object, container: type | None) -> list[Artifact]:
    """Returns a dependency field's artifacts in manifest order: a set's
    sorted by artifact path, never by set iteration."""
    if container is None:
        return [] if value is None else [value]
    return sorted(value, key=lambda a: a.artifact_path.as_posix()) if container is frozenset else list(value)


def parameters(artifact: Artifact) -> dict:
    """An artifact's own fields as JSON, sorted by name: the `"parameters"`
    half of its manifest, without the dependencies' manifests.

    The parameters/dependencies split is read off the annotations, so it is
    the same split regardless of what a field happens to hold. `commit` and
    `allocated_resources` are neither, being about running and not state.
    """
    cls = type(artifact)
    deps, tagged = dependencies(cls), polymorphic(cls)
    encoded = {}
    for name, annotation in sorted(annotations(cls).items()):
        if name in deps or name in ("commit", "allocated_resources"):
            continue
        value = getattr(artifact, name)
        if name in tagged and value is not None:
            # "class" is a Python keyword, so no field can be named it
            encoded[name] = dict(sorted({"class": dotted(type(value)), **_encode(value, type(value))}.items()))
        else:
            encoded[name] = _encode(value, annotation)
    return encoded


def to_manifest(artifact: Artifact) -> dict:
    """An artifact as a dictionary, enough to rebuild it from nothing else:
    its type, its commit, its resources, its own parameters, and a manifest
    apiece for the artifacts it's built from.

    Key order is fixed -- the five here as written, everything below sorted
    -- so the file stays a pure function of the artifact even when a
    dataclass's fields are reordered. A set of artifacts is an array in
    `members` order, since JSON has no set type.
    """
    from artifacts.core.artifact import Artifact

    Artifact.require_concrete(artifact)
    cls = type(artifact)
    deps = {}
    for name, container in dependencies(cls).items():
        value = getattr(artifact, name)
        if container is None:
            deps[name] = None if value is None else to_manifest(value)
        else:
            deps[name] = [to_manifest(item) for item in members(value, container)]
    return {
        "artifact": dotted(cls),
        "commit": artifact.commit,
        "allocated_resources": _encode(
            artifact.allocated_resources, annotations(cls)["allocated_resources"]
        ),
        "parameters": parameters(artifact),
        "dependencies": deps,
    }


def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
    """The artifact a manifest describes and the tree under it, unbound.

    Each node keeps the commit recorded for it, so a subtree produced by
    older code stays visibly older.

    `memo`, one dict across many calls, hands back the same instance for a
    subtree they share (the dataset under every leg of a run, each leg under
    the next) instead of decoding it again; the instances are frozen, so
    sharing them changes nothing but the work.
    """
    cls = locate(data["artifact"])
    if not is_artifact(cls):
        raise TypeError(f"manifest target must be an Artifact, got {cls}")
    hints, deps, tagged = annotations(cls), dependencies(cls), polymorphic(cls)
    held = {
        name: [from_manifest(item, memo) for item in value] if deps[name]
        else None if value is None else from_manifest(value, memo)
        for name, value in data["dependencies"].items()
    }
    own = {"commit": data["commit"], "allocated_resources": data["allocated_resources"], **data["parameters"]}
    if memo is not None:
        # a node's own fields and the instances its dependencies decoded to,
        # which the memo already keys by their own subtrees
        ids = {name: list(map(id, v)) if isinstance(v, list) else id(v) for name, v in held.items()}
        key = json.dumps([data["artifact"], own, ids], sort_keys=True)
        if key in memo:
            return memo[key]
    inputs = {}
    for name, value in own.items():
        annotation = hints[name]
        if name in tagged and value is not None:
            value = dict(value)
            annotation = locate(value.pop("class"))
        inputs[name] = _decode(value, annotation, f"{cls.__qualname__}.{name}")
    for name, value in held.items():
        inputs[name] = deps[name](value) if deps[name] else value
    artifact = cls(**inputs)
    if memo is not None:
        memo[key] = artifact
    return artifact


def manifest_json(manifest: dict) -> str:
    """One manifest as the bytes that belong on disk.

    allow_nan=False because Python writes NaN and Infinity as bare words and
    reads them back, while no other JSON reader accepts either.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
