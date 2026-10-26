"""Artifact <-> manifest dict <-> bytes on disk.

A parameter is encoded and decoded by pydantic against its field's annotation
(artifacts.core.schema), so its annotation alone says how to rebuild it, except
in a polymorphic field, whose record names its own class under "class"; a
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


def _dump(value: object, annotation: object) -> object:
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


def _load(data: object, annotation: object, path: str) -> object:
    """Returns the value JSON `data` encodes as `annotation`, validated
    strictly; `path` names it in an error."""
    with _named(path):
        return _field(annotation).model_validate_json(json.dumps({"value": data})).value


def _encode(value: object, annotation: object) -> object:
    """Returns a field's `value` as JSON data; a record in a polymorphic field
    is dumped as its own class, which it names under "class"."""
    if isinstance(value, polymorphic(annotation)):
        # "class" is a Python keyword, so no field can be named it
        return dict(sorted({"class": dotted(type(value)), **_dump(value, type(value))}.items()))
    return _dump(value, annotation)


def _decode(data: object, annotation: object, path: str) -> object:
    """Returns the field value JSON `data` encodes; a polymorphic field's record
    is loaded as the class it names, which must subclass the field's."""
    bases = polymorphic(annotation)
    if not bases or data is None:
        return _load(data, annotation, path)
    data = dict(data)
    cls = locate(data.pop("class"))
    if not issubclass(cls, bases):
        raise TypeError(f"{path}: {dotted(cls)} is not a {bases[0].__qualname__}")
    return _load(data, cls, path)


def normalize(value: object, annotation: object, path: str) -> object:
    """Returns `value` as its manifest decodes it: ints in float fields as
    floats, -0.0 as 0.0, records rebuilt through their constructors.

    Raises unless `value` is exactly an `annotation`: a bool is not an int, a
    list is not a tuple, and a subclass is not the record its field names
    unless that record is polymorphic.
    """
    record = type(value) if isinstance(value, polymorphic(annotation)) else annotation
    with _named(path):
        validated = _field(record).model_validate({"value": value}).value
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
    deps = dependencies(cls)
    return {
        name: _encode(getattr(artifact, name), annotation)
        for name, annotation in sorted(annotations(cls).items())
        if name not in deps and name not in ("commit", "allocated_resources")
    }


def to_manifest(artifact: Artifact) -> dict:
    """An artifact as a dictionary, enough to rebuild it from nothing else:
    its type, its commit, its resources, its own parameters, and a manifest
    apiece for the artifacts it's built from.

    Key order is fixed -- the five here as written, everything below sorted
    -- so the file stays a pure function of the artifact even when a
    dataclass's fields are reordered. A set of artifacts is an array in
    `members` order, since JSON has no set type.
    """
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

    `memo` maps a manifest's JSON to the artifact it decoded to. One dict
    across many calls hands back the same instance for a subtree they share
    (the dataset under every leg of a run, each leg under the next) instead
    of decoding it again; the instances are frozen, so sharing them changes
    nothing but the work.
    """
    key = json.dumps(data, sort_keys=True) if memo is not None else None
    if key is not None and key in memo:
        return memo[key]
    cls = locate(data["artifact"])
    if not is_artifact(cls):
        raise TypeError(f"manifest target must be an Artifact, got {cls}")
    hints, deps = annotations(cls), dependencies(cls)
    own = {"commit": data["commit"], "allocated_resources": data["allocated_resources"], **data["parameters"]}
    inputs = {name: _decode(value, hints[name], f"{cls.__qualname__}.{name}") for name, value in own.items()}
    for name, value in data["dependencies"].items():
        if deps[name]:
            inputs[name] = deps[name](from_manifest(item, memo) for item in value)
        else:
            inputs[name] = None if value is None else from_manifest(value, memo)
    artifact = cls(**inputs)
    if key is not None:
        memo[key] = artifact
    return artifact


def manifest_json(manifest: dict) -> str:
    """One manifest as the bytes that belong on disk.

    allow_nan=False because Python writes NaN and Infinity as bare words and
    reads them back, while no other JSON reader accepts either.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
