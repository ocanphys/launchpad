"""Finite declaration choices. Concrete artifacts never contain these objects.

Artifact constructors use DeclarationMeta: inputs holding a Sweep become an
ArtifactSweep, anything else a concrete artifact, and `expand` turns a
declaration into the concrete artifacts it stands for. Parameters are ordinary
frozen dataclasses with concrete annotations, and a Sweep may stand in any
sweepable field; expansion reconstructs them through their constructors,
including their __post_init__ checks. `check` holds constructor inputs to the
field schema (artifacts.core.schema), normalizing every concrete parameter
through the manifest codec.
"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from functools import cache
from inspect import signature
from typing import TYPE_CHECKING, get_args, get_origin

from artifacts.core import manifest, schema

if TYPE_CHECKING:
    from artifacts.core.artifact import Artifact


@dataclass(frozen=True, init=False)
class Sweep[T]:
    """An ordered, nonempty, finite choice; ordinary tuples remain values.

    A type checker reads `Sweep((1e-3, 1e-4))` as a float, so it fits a field
    annotated `float` and a wrongly typed alternative is flagged where it is
    written.
    """

    values: tuple[T, ...]

    def __init__(self, values: Sequence[T]):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(
                "Sweep requires a finite ordered sequence (tuple, list, or range)"
            )
        if not values:
            raise ValueError("Sweep must contain at least one alternative")
        object.__setattr__(self, "values", tuple(values))

    if TYPE_CHECKING:

        def __new__(cls, values: Sequence[T]) -> T: ...


class Record:
    """The base of a parameter record with a `__post_init__` check.

    The check runs only once no field holds a Sweep, so it is written for
    concrete values; expansion rebuilds the record from each combination,
    running it there.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("__post_init__")
        if own is not None:

            def __post_init__(self) -> None:
                if not holds_choice(self):
                    own(self)

            cls.__post_init__ = __post_init__


class UnresolvedDeclarationError(TypeError):
    """A declaration was used where a concrete value is required."""


@dataclass(frozen=True, eq=False)
class ArtifactSweep[A: Artifact]:
    """Internal artifact declaration, deliberately not an Artifact."""

    _target: type[A]
    _inputs: dict[str, object]

    __hash__ = None

    def __getattr__(self, name: str):
        # Pickle probes special methods before restoring the instance state.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._inputs:
            return self._inputs[name]
        raise UnresolvedDeclarationError(
            f"{self._target.__name__}.{name}: unresolved declaration; "
            f"it requires a concrete value; call expand first"
        )


type Declaration = (
    Artifact | ArtifactSweep[Artifact] | Sweep[Artifact | ArtifactSweep[Artifact]]
)


def children(value: object, path: str) -> list[tuple[str, object]]:
    """Returns (path, item) for each value directly inside a sweep, declaration,
    record or collection; a scalar or an artifact has none."""
    if isinstance(value, Sweep):
        return [(f"{path}.values[{i}]", item) for i, item in enumerate(value.values)]
    if isinstance(value, ArtifactSweep):
        return [(f"{path}.{name}", item) for name, item in value._inputs.items()]
    if is_record(value):
        return [(f"{path}.{f.name}", getattr(value, f.name)) for f in fields(value)]
    if isinstance(value, (tuple, frozenset)):
        return [(f"{path}[{i}]", item) for i, item in enumerate(value)]
    return []


def holds_choice(value: object) -> bool:
    """Returns whether a Sweep sits anywhere inside `value`."""
    return isinstance(value, Sweep) or any(holds_choice(item) for _, item in children(value, ""))


def is_record(value: object) -> bool:
    """Returns whether `value` is a parameter record: a dataclass instance that
    is neither an artifact nor a declaration."""
    return (
        is_dataclass(value)
        and not isinstance(value, (type, Sweep, ArtifactSweep))
        and not schema.is_artifact(type(value))
    )


def check(cls, values: dict, path: str, *, sweepable: bool = True) -> dict:
    """Returns each of `cls`'s fields present in `values`, every concrete
    parameter normalized, raising unless they fit `cls`'s field schema: types
    match, artifacts sit only in dependency fields, and sweeps only where
    allowed. `path` names the fields in an error."""
    hints, deps, tagged = schema.annotations(cls), schema.dependencies(cls), schema.polymorphic(cls)
    return {
        f.name: _check(
            values[f.name],
            hints[f.name],
            f"{path}.{f.name}",
            dependency=f.name in deps,
            sweepable=sweepable and f.init and f.metadata.get("sweepable", True),
            own_class=f.name in tagged,
        )
        for f in fields(cls)
        if f.name in values
    }


def _check(value, annotation, path, *, dependency, sweepable, own_class=False) -> object:
    """Returns `value`, normalized where it is a concrete parameter; raises
    where it does not fit `annotation`.

    Artifacts and artifact declarations are leaves: each passed this check
    when it was constructed. With `own_class`, a record may be any subclass of
    its annotated record and is checked as its own class.
    """
    if isinstance(value, Sweep):
        if not sweepable:
            raise UnresolvedDeclarationError(f"{path}: sweeps are not allowed here")
        for i, alternative in enumerate(value.values):
            _check(
                alternative, annotation, f"{path}.values[{i}]",
                dependency=dependency, sweepable=True, own_class=own_class,
            )
        return value
    if own_class and is_record(value) and any(
        isinstance(value, arm) for arm in schema.arms(annotation) if is_dataclass(arm)
    ):
        annotation = type(value)

    if dependency:
        container = get_origin(annotation)
        if container in (tuple, frozenset):
            if not isinstance(value, tuple if container is tuple else (tuple, frozenset)):
                raise TypeError(f"{path}: expected {annotation}, got {type(value).__name__}")
            if isinstance(value, frozenset) and holds_choice(value):
                raise TypeError(f"{path}: supply unresolved set members as an ordered tuple")
            for i, item in enumerate(value):
                _check(item, get_args(annotation)[0], f"{path}[{i}]", dependency=True, sweepable=sweepable)
            return value
        if isinstance(value, ArtifactSweep) and not sweepable:
            raise UnresolvedDeclarationError(f"{path}: unresolved declaration is not allowed here")
        target = value._target if isinstance(value, ArtifactSweep) else type(value)
        if not any(issubclass(target, arm) for arm in schema.arms(annotation)):
            raise TypeError(f"{path}: expected {annotation}, got {target.__name__}")
        return value

    if isinstance(value, ArtifactSweep) or schema.is_artifact(type(value)):
        raise TypeError(f"{path}: artifact hidden in a parameter; use a dependency field")
    if not holds_choice(value):
        return manifest.normalize(value, annotation, path)
    if is_record(value):
        if type(value) not in schema.arms(annotation):
            raise TypeError(f"{path}: expected exactly {annotation}, got {type(value).__name__}")
        check(type(value), {f.name: getattr(value, f.name) for f in fields(value)}, path, sweepable=sweepable)
        return value
    tuple_type = next((arm for arm in schema.arms(annotation) if get_origin(arm) is tuple), None)
    if tuple_type is None or not isinstance(value, tuple):
        raise TypeError(f"{path}: expected {annotation}, got {type(value).__name__}")
    args = get_args(tuple_type)
    item_types = (args[0],) * len(value) if args[-1] is Ellipsis else args
    if len(item_types) != len(value):
        raise TypeError(f"{path}: expected {tuple_type}, got a tuple of length {len(value)}")
    for i, (item, item_type) in enumerate(zip(value, item_types)):
        _check(item, item_type, f"{path}[{i}]", dependency=False, sweepable=sweepable)
    return value


def verify(cls, inputs: dict, built) -> None:
    """Raises unless `built`, constructed from `inputs`, is concrete and still
    holds every chosen input.

    __post_init__ may fill an omitted None; any other change raises.
    """
    concrete = {f.name: getattr(built, f.name) for f in fields(cls)}
    # an input the constructor kept as it was already passed `check`
    changed = {n: v for n, v in concrete.items() if n not in inputs or v is not inputs[n]}
    check(cls, changed, cls.__name__, sweepable=False)
    deps = schema.dependencies(cls)
    for name, value in changed.items():
        original = inputs.get(name, value)  # a field that was no input has nothing to keep
        exempt = name not in deps and original is None
        if not exempt and original != value:
            raise ValueError(f"{cls.__name__}.{name}: __post_init__ changed a chosen input")


@cache
def _signature(cls):
    """Returns `cls`'s constructor signature without `self`."""
    sig = signature(cls.__init__)
    return sig.replace(parameters=tuple(sig.parameters.values())[1:])


class DeclarationMeta(ABCMeta):
    """Dispatch before dataclass initialization, keeping concrete hooks ordinary."""

    @property
    def __signature__(cls):
        return _signature(cls)

    def __call__(cls, *args, **kwargs):
        if not is_dataclass(cls) or cls.__abstractmethods__:
            return super().__call__(*args, **kwargs)
        inputs = _signature(cls).bind(*args, **kwargs).arguments
        # bind has raised for any missing field without a default
        for f in fields(cls):
            if f.init and f.name not in inputs:
                inputs[f.name] = f.default if f.default_factory is MISSING else f.default_factory()
        # A set accepts any iterable, kept ordered while its members may hold choices.
        sets = [name for name, kind in schema.dependencies(cls).items() if kind is frozenset]
        for name in sets:
            if not isinstance(inputs[name], (Sweep, tuple, frozenset)):
                inputs[name] = tuple(inputs[name])
        inputs = check(cls, inputs, cls.__name__)
        if any(holds_choice(value) for value in inputs.values()):
            return ArtifactSweep(cls, inputs)
        for name in sets:
            inputs[name] = frozenset(inputs[name])
        built = super().__call__(**inputs)
        verify(cls, inputs, built)
        return built


def expand(declaration: Declaration) -> list[Artifact]:
    """Returns the flat sweep of concrete artifacts `declaration` stands for.

    A concrete artifact is a sweep of one, and a Sweep of artifacts is a group.
    Every Sweep is its own choice, and all of them combine by Cartesian
    product; a Sweep inside an alternative counts only where that alternative
    is chosen. An artifact declaration used in several places is one node, so
    the sweeps inside it are chosen once per combination. Each combination is
    built through its normal constructors, and duplicates are kept; resolve
    deduplicates them.
    """
    from artifacts.core.artifact import Artifact

    # A choice is keyed by the declaration it sits in and its position there.
    # Entering a declaration starts a fresh position, so a declaration reached
    # twice holds the same keys, while two positions never share one.
    def inside(value, path, key):
        """Returns (path, key, item) for each child, fields by name and items
        by position."""
        scope, at = (id(value), "") if isinstance(value, ArtifactSweep) else key
        items = [(p, (scope, at + p[len(path):]), item) for p, item in children(value, path)]
        return items if isinstance(value, (Sweep, tuple, frozenset)) else sorted(items, key=lambda i: i[0])

    def open_choices(value, path, key, chosen):
        """Yields (key, path, sweep) for each Sweep with no choice yet,
        following the choices made so far, lazily."""
        if isinstance(value, Sweep) and key not in chosen:
            yield key, path, value
            return
        items = inside(value, path, key)
        if isinstance(value, Sweep):
            items = [items[chosen[key]]]
        for p, k, item in items:
            yield from open_choices(item, p, k, chosen)

    def build(value, path, key, chosen):
        """Returns `value` with every Sweep replaced by its chosen alternative."""
        if isinstance(value, Sweep):
            p, k, item = inside(value, path, key)[chosen[key]]
            return build(item, p, k, chosen)
        if isinstance(value, tuple):
            return tuple(build(item, p, k, chosen) for p, k, item in inside(value, path, key))
        if isinstance(value, ArtifactSweep):
            target = value._target
        elif is_record(value) and holds_choice(value):
            target = type(value)
        else:
            return value
        init = {f.name for f in fields(target) if f.init}
        concrete = {}
        for p, k, item in inside(value, path, key):
            name = p[len(path) + 1:]
            if name in init:
                concrete[name] = build(item, p, k, chosen)
        built = target(**concrete)
        if not schema.is_artifact(target):  # artifacts verify on construction
            verify(target, concrete, built)
        return built

    roots = []

    def branch(chosen, choices):
        for key, path, sweep in open_choices(declaration, "root", (None, ""), chosen):
            for i in range(len(sweep.values)):
                branch({**chosen, key: i}, (*choices, f"{path}.values[{i}]"))
            return
        try:
            built = build(declaration, "root", (None, ""), chosen)
        except (TypeError, ValueError) as error:
            error.add_note(f"while building root [{', '.join(choices)}]")
            raise
        roots.append(Artifact.require_concrete(built))

    branch({}, ())
    return roots
