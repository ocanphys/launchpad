"""The manifest codec: every value a constructor accepts survives its manifest,
equal artifacts encode equal, and whatever the codec cannot rebuild exactly
is refused where it is declared or constructed."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from artifacts.core.artifact import MANIFEST, Artifact, Resources, _digest
from artifacts.core.manifest import manifest_json, normalize
from artifacts.core.SGD.training import (
    LoopConfig,
    LRSchedule,
    OptimizerParameters,
    TrainingParameters,
)
from artifacts.core.sweep import Sweep, expand
from models.mamba import Mamba
from models.transformer import Transformer


@dataclass(frozen=True)
class Leaf:
    flag: bool
    count: int
    rate: float
    name: str
    mode: Literal["a", "b"]
    maybe: float | None
    either: int | str


@dataclass(frozen=True)
class Shapes:
    leaf: Leaf
    many: tuple[float, ...]
    pair: tuple[int, str]
    leaves: tuple[Leaf, ...]
    optional_leaf: Leaf | None = None


@dataclass(frozen=True)
class Plain(Artifact):
    """Every parameter shape, and every dependency shape, in one artifact."""

    producer: ClassVar[None] = None
    shapes: Shapes
    one: Plain | None = None
    sequence: tuple[Plain, ...] = ()
    group: frozenset[Plain] = frozenset()

    @property
    def uid(self) -> str:
        # identity the way real artifacts build it: asdict of the parameters
        return _digest(
            asdict(self.shapes),
            self.one.uid if self.one else None,
            [p.uid for p in self.sequence],
            sorted(p.uid for p in self.group),
        )

    @property
    def artifact_path(self) -> Path:
        return Path("plain") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class Real(Artifact):
    """The records the training stages are declared with."""

    producer: ClassVar[None] = None
    training: TrainingParameters
    loop: LoopConfig
    transformer: Transformer
    mamba: Mamba

    @property
    def uid(self) -> str:
        return _digest(asdict(self.training), asdict(self.loop), asdict(self.transformer), asdict(self.mamba))

    @property
    def artifact_path(self) -> Path:
        return Path("real") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {}


# A float field takes an int too, as long as the float holds it exactly.
FLOATS = st.floats(allow_nan=False, allow_infinity=False) | st.integers(-(2**53), 2**53)
LEAVES = st.builds(Leaf, rate=FLOATS, maybe=st.none() | FLOATS)
SHAPES = st.builds(
    Shapes,
    leaf=LEAVES,
    many=st.lists(FLOATS, max_size=4).map(tuple),
    leaves=st.lists(LEAVES, max_size=3).map(tuple),
    optional_leaf=st.none() | LEAVES,
)
POSITIVE = st.integers(1, 2**31)
REALS = st.builds(
    Real,
    training=st.builds(
        TrainingParameters,
        total_steps=POSITIVE,
        max_norm=FLOATS,
        lr_schedule=st.builds(LRSchedule, max_learning_rate=FLOATS, min_learning_rate=FLOATS),
        optimizer_parameters=st.builds(
            OptimizerParameters,
            lr=FLOATS,
            betas=st.lists(FLOATS, max_size=3).map(tuple),
            weight_decay=FLOATS,
            eps=FLOATS,
        ),
    ),
    loop=st.builds(LoopConfig, checkpoint_every=POSITIVE),
    transformer=st.tuples(POSITIVE, st.integers(1, 64)).flatmap(
        lambda kh: st.builds(Transformer, d_model=st.just(kh[0] * kh[1]), num_heads=st.just(kh[1]))
    ),
    mamba=st.builds(Mamba),
)


def plain(shapes: Shapes, **dependencies) -> Plain:
    return Plain(shapes=shapes, commit="c", **dependencies)


def reread(artifact: Artifact) -> tuple[Artifact, str]:
    """Returns `artifact` rebuilt from its manifest's bytes, and those bytes."""
    text = manifest_json(artifact.to_manifest())
    return Artifact.from_manifest(json.loads(text)), text


# Round trips


@settings(deadline=None)
@given(SHAPES)
def test_every_parameter_shape_survives_its_manifest_bytes(shapes):
    artifact = plain(shapes)
    back, text = reread(artifact)
    assert back == artifact
    assert back.artifact_path == artifact.artifact_path
    assert manifest_json(back.to_manifest()) == text


@settings(deadline=None)
@given(SHAPES, SHAPES, SHAPES)
def test_every_dependency_shape_survives_its_manifest_bytes(a, b, c):
    leaf, other = plain(a), plain(b)
    artifact = plain(c, one=leaf, sequence=(other, leaf, other), group=(leaf, other))
    back, text = reread(artifact)
    assert back == artifact
    assert back.sequence == (other, leaf, other)
    assert back.artifact_path == artifact.artifact_path
    assert manifest_json(back.to_manifest()) == text


@settings(deadline=None)
@given(REALS)
def test_the_training_records_survive_their_manifest_bytes(real):
    back, text = reread(real)
    assert back == real
    assert back.artifact_path == real.artifact_path
    assert manifest_json(back.to_manifest()) == text


@settings(deadline=None)
@given(SHAPES)
def test_a_constructed_artifact_already_holds_what_its_manifest_decodes_to(shapes):
    artifact = plain(shapes)
    back, _ = reread(artifact)
    # == is not enough, since 1 == 1.0 and -0.0 == 0.0; repr tells them apart
    assert repr(artifact) == repr(back)


def test_disk_round_trip_through_load(tmp_path):
    artifact = plain(Shapes(Leaf(True, 1, 0.003, "ü", "a", None, "x"), (1e-8,), (2, "b"), ()))
    folder = tmp_path / artifact.artifact_path
    folder.mkdir(parents=True)
    (folder / MANIFEST).write_text(manifest_json(artifact.to_manifest()))
    assert Artifact.load(artifact.artifact_path, root=tmp_path) == artifact


# Equal means one folder


@settings(deadline=None)
@given(SHAPES, st.integers(-(2**53), 2**53))
def test_an_int_in_a_float_field_is_the_float_it_equals(shapes, number):
    as_int = plain(replace(shapes, leaf=replace(shapes.leaf, rate=number), many=(number,)))
    as_float = plain(replace(shapes, leaf=replace(shapes.leaf, rate=float(number)), many=(float(number),)))
    assert as_int == as_float
    assert as_int.artifact_path == as_float.artifact_path
    assert as_int.to_manifest() == as_float.to_manifest()
    assert type(as_int.shapes.leaf.rate) is float


@settings(deadline=None)
@given(SHAPES)
def test_negative_zero_is_zero(shapes):
    negative = plain(replace(shapes, leaf=replace(shapes.leaf, rate=-0.0), many=(-0.0, 1.0)))
    positive = plain(replace(shapes, leaf=replace(shapes.leaf, rate=0.0), many=(0.0, 1.0)))
    assert negative.artifact_path == positive.artifact_path
    assert math.copysign(1, negative.shapes.leaf.rate) == 1


@settings(deadline=None)
@given(SHAPES, st.lists(FLOATS, min_size=1, max_size=4, unique_by=float))
def test_a_swept_value_builds_the_artifact_written_directly(shapes, rates):
    declared = Plain(shapes=replace(shapes, leaf=replace(shapes.leaf, rate=Sweep(rates))), commit="c")
    direct = [plain(replace(shapes, leaf=replace(shapes.leaf, rate=rate))) for rate in rates]
    roots = expand(declared)
    assert roots == direct
    assert [r.artifact_path for r in roots] == [d.artifact_path for d in direct]
    assert all(type(r.shapes.leaf.rate) is float for r in roots)


def test_resources_round_trip_and_stay_out_of_identity():
    first = Plain(shapes=Shapes(Leaf(False, 0, 0.0, "", "b", 1.5, 3), (), (0, ""), ()), allocated_resources=Resources(cpu=2))
    back, _ = reread(first)
    assert back.allocated_resources == Resources(cpu=2.0)
    assert type(back.allocated_resources.cpu) is float


# Refused values

LEAF = Leaf(True, 1, 0.5, "n", "a", None, 1)
SHAPE = Shapes(LEAF, (1.0,), (1, "a"), ())


@dataclass(frozen=True)
class Narrower(Leaf):
    extra: int = 0


@pytest.mark.parametrize(
    ("field", "value", "error", "where"),
    [
        ("leaf", replace(LEAF, count=True), TypeError, "leaf.count"),
        ("leaf", replace(LEAF, flag=1), TypeError, "leaf.flag"),
        ("leaf", replace(LEAF, rate="0.5"), TypeError, "leaf.rate"),
        ("leaf", replace(LEAF, rate=math.nan), ValueError, "leaf.rate"),
        ("leaf", replace(LEAF, rate=math.inf), ValueError, "leaf.rate"),
        ("leaf", replace(LEAF, rate=2**53 + 1), TypeError, "expected exactly Shapes"),
        ("leaf", replace(LEAF, mode="c"), TypeError, "leaf.mode"),
        ("leaf", replace(LEAF, name=None), TypeError, "leaf.name"),
        ("leaf", replace(LEAF, either=1.5), TypeError, "leaf.either"),
        ("many", [1.0], TypeError, "many"),
        ("many", (1.0, "x"), TypeError, "many[1]"),
        ("pair", (1,), TypeError, "pair"),
        ("pair", (1, "a", "b"), TypeError, "pair"),
        ("leaf", asdict(LEAF), TypeError, "leaf"),
        ("leaf", Narrower(**asdict(LEAF)), TypeError, "expected exactly"),
        ("leaves", (LEAF, object()), TypeError, "leaves[1]"),
    ],
)
def test_a_value_that_would_not_come_back_exactly_is_refused(field, value, error, where):
    with pytest.raises(error, match=rf"Plain\.shapes.*{re.escape(where)}"):
        plain(replace(SHAPE, **{field: value}))


def test_a_swept_alternative_is_checked_where_it_is_written():
    with pytest.raises(TypeError, match=r"Plain\.shapes\.leaf\.rate\.values\[1\]"):
        Plain(shapes=replace(SHAPE, leaf=replace(LEAF, rate=Sweep((0.1, "x")))))
    with pytest.raises(TypeError, match=r"Plain\.shapes\.pair\[0\]\.values\[0\]"):
        Plain(shapes=replace(SHAPE, pair=(Sweep(("x",)), "a")))


def test_normalize_is_idempotent_and_exact():
    once = normalize(SHAPE, Shapes, "s")
    assert normalize(once, Shapes, "s") == once
    assert normalize(-0.0, float, "x").hex() == (0.0).hex()
    assert normalize((1, 2), tuple[float, ...], "x") == (1.0, 2.0)


# Refused manifests


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        (lambda m: m["parameters"]["shapes"]["leaf"].update(extra=1), TypeError),
        (lambda m: m["parameters"]["shapes"]["leaf"].pop("rate"), TypeError),
        (lambda m: m["parameters"]["shapes"]["leaf"].update(count="1"), TypeError),
        (lambda m: m["parameters"]["shapes"]["leaf"].update(count=1.0), TypeError),
        (lambda m: m["parameters"]["shapes"]["leaf"].update(rate=math.nan), ValueError),
        (lambda m: m["parameters"]["shapes"].update(pair=[1]), TypeError),
        (lambda m: m["parameters"].update(unknown=1), KeyError),
        (lambda m: m["dependencies"].update(one={"artifact": "builtins.int"}), TypeError),
    ],
)
def test_a_manifest_that_does_not_fit_its_annotations_is_refused(tamper, error):
    manifest = plain(SHAPE).to_manifest()
    tamper(manifest)
    with pytest.raises(error):
        Artifact.from_manifest(manifest)


def test_load_names_the_folder_of_a_manifest_that_does_not_fit(tmp_path):
    artifact = plain(SHAPE)
    manifest = artifact.to_manifest()
    manifest["parameters"]["shapes"]["leaf"]["count"] = True
    folder = tmp_path / artifact.artifact_path
    folder.mkdir(parents=True)
    (folder / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"not a readable manifest.*Plain\.shapes\.leaf\.count"):
        Artifact.load(artifact.artifact_path, root=tmp_path)


# Refused annotations, when a class is first used


@dataclass(frozen=True)
class Two:
    value: int


@dataclass
class Mutable:
    value: int


@dataclass(frozen=True)
class HidesArtifact:
    value: Plain


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (Any, "needs a concrete type"),
        (object, "needs a concrete type"),
        (tuple, "needs a concrete type"),
        (frozenset[str], "needs a concrete type"),
        (list[int], "needs a concrete type"),
        (dict[str, int], "needs a concrete type"),
        (Mutable, "needs a concrete type"),
        (Leaf | Two, "at most one record type"),
        (HidesArtifact, "artifact hidden in a parameter"),
        (tuple[Plain, ...] | None, "artifact hidden in a parameter"),
    ],
)
def test_an_annotation_the_codec_cannot_rebuild_is_refused(annotation, message):
    @dataclass(frozen=True)
    class Holder(Artifact):
        producer: ClassVar[None] = None
        value: Any

        uid = artifact_path = files = None

    Holder.__annotations__["value"] = annotation
    with pytest.raises(TypeError, match=message):
        Holder(value=None)


# A field that may hold any subclass of its record


@dataclass(frozen=True)
class Shape:
    polymorphic: ClassVar[bool] = True

    sides: int


@dataclass(frozen=True)
class Square(Shape):
    width: float


@dataclass(frozen=True)
class Polygon(Shape):
    angles: tuple[float, ...]


@dataclass(frozen=True)
class Drawing(Artifact):
    producer: ClassVar[None] = None
    shape: Shape | None = None

    @property
    def uid(self) -> str:
        return _digest(type(self.shape).__qualname__, asdict(self.shape) if self.shape else None)

    @property
    def artifact_path(self) -> Path:
        return Path("drawings") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {}


@pytest.mark.parametrize("shape", [Square(4, 2), Polygon(3, (60, 60, 60)), Shape(0)])
def test_a_polymorphic_field_survives_as_its_own_class(shape):
    drawing = Drawing(shape=shape, commit="c")
    back, text = reread(drawing)
    assert drawing.to_manifest()["parameters"]["shape"]["class"] == f"{__name__}.{type(shape).__qualname__}"
    assert back == drawing and type(back.shape) is type(shape)
    assert repr(back) == repr(drawing)  # Square's width 2 came back as the float it was normalized to
    assert manifest_json(back.to_manifest()) == text


def test_a_manifest_naming_a_class_outside_the_field_is_refused():
    manifest = Drawing(shape=Square(4, 2.0)).to_manifest()
    manifest["parameters"]["shape"]["class"] = f"{__name__}.Leaf"  # a record, but not a Shape
    with pytest.raises(TypeError, match=r"Drawing\.shape"):
        Artifact.from_manifest(manifest)
    del manifest["parameters"]["shape"]["class"]
    with pytest.raises(KeyError):
        Artifact.from_manifest(manifest)


def test_a_sweep_across_classes_of_a_polymorphic_field():
    roots = expand(Drawing(shape=Sweep((Square(4, Sweep((1, 2))), Polygon(3, (60.0,)))), commit="c"))
    assert [type(r.shape).__name__ for r in roots] == ["Square", "Square", "Polygon"]
    assert all(Artifact.from_manifest(r.to_manifest()) == r for r in roots)
