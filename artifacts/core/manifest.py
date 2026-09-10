"""Artifact <-> manifest dict <-> bytes on disk.

The one place field annotations are interpreted, and the seam to replace if
the codec ever changes: artifact.py calls `to_manifest`, `from_manifest` and
`dependencies`, and nothing else in the repo reaches past those. Nothing
here touches storage.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from functools import cache
from types import UnionType
from typing import TYPE_CHECKING, Union, get_args, get_origin, get_type_hints

from artifacts.core.locate import locate

if TYPE_CHECKING:
    from artifacts.core.artifact import Artifact


def _is_artifact(cls: object) -> bool:
    """Whether `cls` is an Artifact subclass.

    The only thing this module asks of artifact.py, imported at call time
    because artifact.py imports this module.
    """
    from artifacts.core.artifact import Artifact

    return isinstance(cls, type) and issubclass(cls, Artifact)


def _holds_artifacts(annotation: object) -> bool:
    """Whether `annotation` names artifacts, ignoring an optional None arm."""
    if get_origin(annotation) in (UnionType, Union):
        arms = [arm for arm in get_args(annotation) if arm is not type(None)]
    else:
        arms = [annotation]
    return bool(arms) and all(_is_artifact(arm) for arm in arms)


@cache
def dependencies(cls: type) -> dict[str, type | None]:
    """Which of `cls`'s fields hold artifacts, and what holds them: `tuple`,
    `frozenset`, or None for a single artifact. Sorted by field name.

    The container declares whether order identifies the artifact: a tuple is a
    sequence, so its order and its repeats are part of what the artifact is; a
    frozenset is a set, so neither is. Read off annotations, not values, so an
    empty `sources=frozenset()` is still a dependency.
    """
    hints = get_type_hints(cls)
    found: dict[str, type | None] = {}
    for f in sorted(fields(cls), key=lambda f: f.name):
        annotation = hints[f.name]
        container = get_origin(annotation)
        if container in (tuple, frozenset):
            annotation = get_args(annotation)[0]
        else:
            container = None
        if _holds_artifacts(annotation):
            found[f.name] = container
    return found


def _encode(value: object) -> object:
    """A field value as JSON, artifacts nesting as whole manifests.

    A set becomes an array sorted by artifact path, since JSON has no set
    type. That order is a rendering `_decode` cannot notice; it exists to keep
    the file diffable. Every dict comes out sorted by key.
    """
    if _is_artifact(type(value)):
        return to_manifest(value)
    if isinstance(value, frozenset):
        return [_encode(item) for item in sorted(value, key=lambda a: a.artifact_path.as_posix())]
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if is_dataclass(value):
        return {
            f.name: _encode(getattr(value, f.name))
            for f in sorted(fields(value), key=lambda f: f.name)
        }
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in sorted(value.items())}
    return value


def _decode(annotation: object, value: object) -> object:
    """One field value rebuilt from JSON, shaped by its declared type.

    JSON can't tell a tuple from a set from a list, or a nested config from
    any other dict. A union of artifact types isn't ambiguous -- every
    manifest names its own concrete type -- but any other multi-arm union is.

    _decode(tuple[str, ...], ["<pad>", "<unk>"]) -> ("<pad>", "<unk>")
    """
    if get_origin(annotation) in (UnionType, Union):
        if value is None:
            return None
        if _holds_artifacts(annotation):
            return from_manifest(value)
        arms = [arm for arm in get_args(annotation) if arm is not type(None)]
        if len(arms) != 1:
            raise TypeError(f"can't decode into {annotation}: more than one arm")
        return _decode(arms[0], value)
    if get_origin(annotation) is tuple:
        return tuple(_decode(get_args(annotation)[0], item) for item in value)
    if get_origin(annotation) is frozenset:
        return frozenset(_decode(get_args(annotation)[0], item) for item in value)
    if _is_artifact(annotation):
        return from_manifest(value)
    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(**{k: _decode(hints[k], v) for k, v in value.items()})
    return value


def to_manifest(artifact: Artifact) -> dict:
    """An artifact as a dictionary, enough to rebuild it from nothing else:
    its type, its commit, its resources, its own parameters, and a manifest
    apiece for the artifacts it's built from.

    The parameters/dependencies split is read off the annotations, so it is
    the same split regardless of what a field happens to hold. Key order is
    fixed -- the five here as written, everything below sorted -- so the file
    stays a pure function of the artifact even when a dataclass's fields are
    reordered.
    """
    dep_fields = dependencies(type(artifact))
    parameters: dict[str, object] = {}
    deps: dict[str, object] = {}
    for f in fields(artifact):
        if f.name in ("commit", "allocated_resources"):
            continue  # their own top-level keys, being about running and not state
        target = deps if f.name in dep_fields else parameters
        target[f.name] = _encode(getattr(artifact, f.name))
    cls = type(artifact)
    return {
        "artifact": f"{cls.__module__}.{cls.__qualname__}",
        "commit": artifact.commit,
        "allocated_resources": _encode(artifact.allocated_resources),
        "parameters": dict(sorted(parameters.items())),
        "dependencies": dict(sorted(deps.items())),
    }


def from_manifest(data: dict) -> Artifact:
    """The artifact a manifest describes and the tree under it, unbound.

    Each node keeps the commit recorded for it, so a subtree produced by
    older code stays visibly older.
    """
    cls = locate(data["artifact"])
    hints = get_type_hints(cls)
    plugged = {
        "commit": data["commit"],
        "allocated_resources": data["allocated_resources"],
        **data["parameters"],
        **data["dependencies"],
    }
    return cls(**{name: _decode(hints[name], value) for name, value in plugged.items()})


def manifest_json(manifest: dict) -> str:
    """One manifest as the bytes that belong on disk.

    allow_nan=False because Python writes NaN and Infinity as bare words and
    reads them back, while no other JSON reader accepts either.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
