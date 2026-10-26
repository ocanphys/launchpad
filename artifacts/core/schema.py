"""The field schema: what each field of an artifact or parameter record may
hold, read off its annotations.

A parameter is a scalar, a Literal, a tuple with item types, a frozen
dataclass of these, or a union of them with at most one dataclass; a
dependency is A, A | B | None, tuple[A, ...] or frozenset[A] of artifacts. A
field annotated with a record class that declares `polymorphic = True` (such
as `Model`) may hold any subclass of it, and its manifest carries the
subclass's dotted path under "class". manifest.py encodes against this schema,
and Artifact.__post_init__ checks every field against it.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import cache
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints


@cache
def is_artifact(cls: object) -> bool:
    """Returns whether `cls` is an Artifact subclass.

    Artifact is imported at call time, because artifact.py imports this module.
    """
    from artifacts.core.artifact import Artifact

    return isinstance(cls, type) and issubclass(cls, Artifact)


def arms(annotation: object) -> tuple:
    """Returns the arms of a union annotation, None included, or the
    annotation alone."""
    return get_args(annotation) if get_origin(annotation) in (UnionType, Union) else (annotation,)


@cache
def dependencies(cls: type) -> dict[str, type | None]:
    """Returns which of `cls`'s fields hold artifacts, and what holds them:
    `tuple`, `frozenset`, or None for a single artifact. Sorted by field name;
    a parameter record has none.

    The container declares whether order identifies the artifact: a tuple is a
    sequence, so its order and its repeats are part of what the artifact is; a
    frozenset is a set, so neither is. Read off annotations, not values, so an
    empty `sources=frozenset()` is still a dependency.
    """
    if not is_artifact(cls):
        return {}
    hints = get_type_hints(cls)
    found: dict[str, type | None] = {}
    for f in sorted(fields(cls), key=lambda f: f.name):
        annotation, container = hints[f.name], get_origin(hints[f.name])
        if container in (tuple, frozenset):
            annotation = get_args(annotation)[0]
        else:
            container = None
        held = [arm for arm in arms(annotation) if arm is not type(None)]
        if held and all(map(is_artifact, held)):
            found[f.name] = container
    return found


@cache
def annotations(cls: type) -> dict[str, object]:
    """Returns `cls`'s field annotations, each checked to be a dependency or a
    parameter type the codec rebuilds exactly (see `_check_parameter`)."""
    hints, deps = get_type_hints(cls), dependencies(cls)
    for f in fields(cls):
        if f.name not in deps:
            _check_parameter(hints[f.name], f"{cls.__qualname__}.{f.name}")
    return {f.name: hints[f.name] for f in fields(cls)}


def polymorphic(annotation: object) -> tuple[type, ...]:
    """Returns the record class in `annotation` that declares `polymorphic = True`,
    as a one-tuple, or () if it names none: a field so annotated may hold any
    subclass of it."""
    return tuple(arm for arm in arms(annotation) if is_dataclass(arm) and getattr(arm, "polymorphic", False))


def _check_parameter(annotation: object, where: str) -> None:
    """Raises unless `annotation` is a scalar, a Literal, a tuple with item
    types, a frozen dataclass of these, or a union with at most one dataclass."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (bool, int, float, str, type(None)) or origin is Literal:
        return
    if len(arms(annotation)) > 1:
        if sum(map(is_dataclass, args)) > 1:
            raise TypeError(
                f"{where}: a union holds at most one record type; a manifest "
                f"stores a union's record as its fields alone"
            )
        for arm in args:
            _check_parameter(arm, where)
        return
    if origin is tuple and args:
        for arg in args:
            if arg is not Ellipsis:
                _check_parameter(arg, where)
        return
    if is_artifact(annotation):
        raise TypeError(
            f"{where}: artifact hidden in a parameter; a dependency is annotated "
            f"A, A | B | None, tuple[A, ...] or frozenset[A]"
        )
    if is_dataclass(annotation) and annotation.__dataclass_params__.frozen:
        annotations(annotation)
        return
    raise TypeError(
        f"{where}: annotated {annotation}, but the codec needs a concrete type: "
        f"a scalar, Literal, tuple[T, ...], union, or frozen dataclass"
    )
