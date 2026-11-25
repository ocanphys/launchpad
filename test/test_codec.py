"""The manifest codec: every value a constructor accepts survives its manifest,
equal artifacts encode equal definitions, a param config comes back as its own
class at any depth, and whatever the codec cannot rebuild is refused where it
is constructed or decoded."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError, create_model

from artifacts.core.artifact import Artifact, ParamConfig, Resources
from artifacts.core.manifest import manifest_json
from artifacts.core.storage import write_manifest
from artifacts.stages.pretraining.evaluations import TokenLoss
from lab import _leaves
from models.mamba import Mamba
from models.transformer import Transformer
from training.leg import (
    LoopConfig,
    TrainingParameters,
)
from training.lr_schedule import Constant, WarmupCosine
from training.optimizers import SGD, AdamW


class Leaf(ParamConfig):
    flag: bool
    count: int
    rate: float
    name: str
    mode: Literal["a", "b"]
    maybe: float | None
    either: int | str


class Shapes(ParamConfig):
    leaf: Leaf
    many: tuple[float, ...]
    pair: tuple[int, str]
    leaves: tuple[Leaf, ...]
    optional_leaf: Leaf | None = None


class Plain(Artifact):
    """Every parameter shape, and every dependency shape, in one artifact."""

    producer: ClassVar[None] = None
    shapes: Shapes
    one: Plain | None = None
    sequence: tuple[Plain, ...] = ()
    group: frozenset[Plain] = frozenset()

    @property
    def uid(self) -> str:
        # identity the way real artifacts build it: a dump of the parameters
        return self.digest(
            self.shapes.model_dump(),
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


class Real(Artifact):
    """The param configs the training stages are declared with."""

    producer: ClassVar[None] = None
    training: TrainingParameters
    loop: LoopConfig
    transformer: Transformer
    mamba: Mamba

    @property
    def uid(self) -> str:
        return self.digest(self.training.model_dump(), self.loop.model_dump(), self.transformer.model_dump(), self.mamba.model_dump())

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
        lr_schedule=st.builds(WarmupCosine, max_learning_rate=FLOATS, min_learning_rate=FLOATS) | st.builds(Constant, learning_rate=FLOATS),
        optimizer=st.builds(
            AdamW,
            betas=st.tuples(FLOATS, FLOATS),
            weight_decay=FLOATS,
            eps=FLOATS,
        ) | st.just(SGD()),
    ),
    loop=st.builds(
        LoopConfig,
        evaluations=st.lists(st.builds(TokenLoss, eval_every=POSITIVE, batch_size=POSITIVE), max_size=2).map(tuple),
        checkpoint_every=POSITIVE,
        gpu_check_every=POSITIVE,
        compile=st.sampled_from([None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]),
    ),
    transformer=st.tuples(POSITIVE, st.integers(1, 64)).flatmap(
        lambda kh: st.builds(Transformer, d_model=st.just(kh[0] * kh[1]), num_heads=st.just(kh[1]))
    ),
    mamba=st.builds(Mamba),
)


def plain(shapes: Shapes, **dependencies) -> Plain:
    return Plain(shapes=shapes, declaring_commit="c", **dependencies)


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
def test_the_training_param_configs_survive_their_manifest_bytes(real):
    back, text = reread(real)
    assert back == real
    assert back.artifact_path == real.artifact_path
    assert manifest_json(back.to_manifest()) == text


@settings(deadline=None)
@given(SHAPES)
def test_a_constructed_artifact_encodes_as_what_its_manifest_decodes_to(shapes):
    artifact = plain(shapes)
    back, _ = reread(artifact)
    # not repr: an artifact built with -0.0 holds it, and its manifest writes 0.0
    assert back.to_manifest() == artifact.to_manifest()


def test_disk_round_trip_through_load(tmp_path):
    leaf = Leaf(flag=True, count=1, rate=0.003, name="ü", mode="a", maybe=None, either="x")
    artifact = plain(Shapes(leaf=leaf, many=(1e-8,), pair=(2, "b"), leaves=()))
    write_manifest(tmp_path, artifact.artifact_path, artifact.to_manifest())
    assert Artifact.load(artifact.artifact_path, root=tmp_path) == artifact


# Equal means one folder


@settings(deadline=None)
@given(SHAPES, st.integers(-(2**53), 2**53))
def test_an_int_in_a_float_field_is_the_float_it_equals(shapes, number):
    as_int = plain(Shapes(**dict(shapes) | {"leaf": Leaf(**dict(shapes.leaf) | {"rate": number}), "many": (number,)}))
    as_float = plain(Shapes(**dict(shapes) | {"leaf": Leaf(**dict(shapes.leaf) | {"rate": float(number)}), "many": (float(number),)}))
    assert as_int == as_float
    assert as_int.artifact_path == as_float.artifact_path
    assert as_int.to_manifest() == as_float.to_manifest()
    assert type(as_int.shapes.leaf.rate) is float


@settings(deadline=None)
@given(SHAPES)
def test_negative_zero_is_written_as_zero(shapes):
    def with_rate(rate):
        return plain(Shapes(**dict(shapes) | {"leaf": Leaf(**dict(shapes.leaf) | {"rate": rate}), "many": (rate, 1.0)}))

    negative, positive = with_rate(-0.0), with_rate(0.0)
    assert negative.artifact_path == positive.artifact_path
    assert negative.to_manifest() == positive.to_manifest()
    assert math.copysign(1, negative.to_manifest()["parameters"]["shapes"]["leaf"]["rate"]) == 1


def test_resources_round_trip_and_stay_out_of_identity():
    leaf = Leaf(flag=False, count=0, rate=0.0, name="", mode="b", maybe=1.5, either=3)
    first = Plain(shapes=Shapes(leaf=leaf, many=(), pair=(0, ""), leaves=()), allocated_resources=Resources(cpu=2))
    back, _ = reread(first)
    assert back.allocated_resources == Resources(cpu=2.0)
    assert type(back.allocated_resources.cpu) is float


LEAF = Leaf(flag=True, count=1, rate=0.5, name="n", mode="a", maybe=None, either=1)
SHAPE = Shapes(leaf=LEAF, many=(1.0,), pair=(1, "a"), leaves=())


def test_a_recorded_field_changes_the_manifest_but_not_the_definition():
    first, second = plain(SHAPE), Plain(shapes=SHAPE, declaring_commit="d", allocated_resources=Resources(cpu=4))
    assert first == second and hash(first) == hash(second)
    assert first.artifact_path == second.artifact_path
    assert first.to_manifest() != second.to_manifest()
    assert _leaves(first.to_manifest()) == _leaves(second.to_manifest())  # the definition encoding
    assert len(frozenset([first, second])) == 1


# Refused values


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("count", True),  # a bool is not an int
        ("flag", 1),
        ("rate", "0.5"),
        ("rate", math.nan),
        ("rate", math.inf),
        ("mode", "c"),
        ("name", None),
        ("either", 1.5),
    ],
)
def test_a_leaf_value_strict_validation_would_have_to_convert_is_refused(field, value):
    with pytest.raises(ValidationError, match=field):
        Leaf(**dict(LEAF) | {field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("many", [1.0]),  # a list is not a tuple
        ("many", (1.0, "x")),
        ("pair", (1,)),
        ("pair", (1, "a", "b")),
        ("leaves", (LEAF, object())),
    ],
)
def test_a_shapes_value_strict_validation_would_have_to_convert_is_refused(field, value):
    with pytest.raises(ValidationError, match=field):
        Shapes(**dict(SHAPE) | {field: value})


def test_a_dependency_of_the_wrong_type_is_refused():
    with pytest.raises(ValidationError, match="one"):
        plain(SHAPE, one=Drawing(declaring_commit="c"))
    with pytest.raises(ValidationError, match="sequence"):
        plain(SHAPE, sequence=[plain(SHAPE)])


def test_a_set_dependency_takes_any_iterable_and_keeps_one_of_two_commit_only_twins():
    other = plain(Shapes(**dict(SHAPE) | {"many": (2.0,)}))
    assert plain(SHAPE, group=[other]).group == frozenset({other})
    assert plain(SHAPE, group=(other, Plain(shapes=other.shapes, declaring_commit="d"))).group == frozenset({other})


# Refused manifests


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        (lambda m: m["parameters"]["shapes"]["leaf"].update(extra=1), ValidationError),
        (lambda m: m["parameters"]["shapes"]["leaf"].pop("rate"), ValidationError),
        (lambda m: m["parameters"]["shapes"]["leaf"].update(count="1"), ValidationError),
        (lambda m: m["parameters"]["shapes"]["leaf"].update(count=1.0), ValidationError),
        (lambda m: m["parameters"]["shapes"]["leaf"].update(rate=math.nan), ValidationError),
        (lambda m: m["parameters"]["shapes"].update(pair=[1]), ValidationError),
        (lambda m: m["parameters"].update(unknown=1), ValidationError),
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
    write_manifest(tmp_path, artifact.artifact_path, manifest)
    with pytest.raises(ValueError, match=r"(?s)not a readable manifest.*for Plain.*shapes\.leaf\.count"):
        Artifact.load(artifact.artifact_path, root=tmp_path)


# Refused annotations, when a class is first used


class HidesArtifact(ParamConfig):
    value: Plain


class Concrete(Artifact):
    producer: ClassVar[None] = None
    uid = property(lambda self: "concrete")
    artifact_path = property(lambda self: Path("concrete"))
    files = property(lambda self: {})


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (Any, "a parameter holds"),
        (object, "a parameter holds"),
        (frozenset[str], "a parameter holds"),
        (list[int], "a parameter holds"),
        (dict[str, int], "a parameter holds"),
        (HidesArtifact, "artifact hidden in a parameter"),
        (tuple[Plain, ...] | None, "artifact hidden in a parameter"),
    ],
)
def test_an_annotation_the_codec_cannot_rebuild_is_refused(annotation, message):
    holder = create_model("Holder", __base__=Concrete, value=(annotation, None))
    with pytest.raises(TypeError, match=message):
        holder.dependencies()


# A field holding a subclass of its param config, at any depth


class Shape(ParamConfig):
    sides: int


class Square(Shape):
    width: float


class Polygon(Shape):
    angles: tuple[float, ...]


class Gallery(ParamConfig):
    shapes: tuple[Shape, ...]


class Drawing(Artifact):
    producer: ClassVar[None] = None
    shape: Shape | None = None
    gallery: Gallery | None = None

    @property
    def uid(self) -> str:
        return self.digest(manifest_json(self.to_manifest()["parameters"]))

    @property
    def artifact_path(self) -> Path:
        return Path("drawings") / self.uid

    @property
    def files(self) -> dict[str, str]:
        return {}


@pytest.mark.parametrize("shape", [Square(sides=4, width=2), Polygon(sides=3, angles=(60, 60, 60)), Shape(sides=0)])
def test_a_param_config_field_holding_a_subclass_survives_as_its_own_class(shape):
    drawing = Drawing(shape=shape, declaring_commit="c")
    back, text = reread(drawing)
    written = drawing.to_manifest()["parameters"]["shape"]
    assert written.get("class") == (None if type(shape) is Shape else f"{__name__}.{type(shape).__qualname__}")
    assert back == drawing and type(back.shape) is type(shape)
    assert manifest_json(back.to_manifest()) == text


def test_a_subclass_nested_in_a_tuple_inside_a_param_config_survives_as_its_own_class():
    drawing = Drawing(gallery=Gallery(shapes=(Square(sides=4, width=1.5), Shape(sides=3))), declaring_commit="c")
    back, text = reread(drawing)
    assert [type(s) for s in back.gallery.shapes] == [Square, Shape]
    assert back == drawing
    assert manifest_json(back.to_manifest()) == text


class Imposter(ParamConfig):
    """Square's fields, but not a Shape."""

    sides: int
    width: float


def test_a_manifest_naming_a_class_outside_the_field_is_refused():
    manifest = Drawing(shape=Square(sides=4, width=2.0)).to_manifest()
    manifest["parameters"]["shape"]["class"] = f"{__name__}.Imposter"
    with pytest.raises(ValidationError, match="shape"):
        Artifact.from_manifest(manifest)
    del manifest["parameters"]["shape"]["class"]  # read as the base, which has no width
    with pytest.raises(ValidationError, match="width"):
        Artifact.from_manifest(manifest)
