"""Declaration expansion, concrete graph invariants, and batched publication."""

from __future__ import annotations

import pickle
import types
from dataclasses import asdict, dataclass, field, replace
from inspect import signature
from pathlib import Path
from typing import Any, ClassVar

import pytest

import lab
from artifacts.core.artifact import Artifact, Resources, _digest
from artifacts.core.manifest import to_manifest
from artifacts.core.resolve import resolve
from artifacts.core.SGD.training import (
    LoopConfig,
    LRSchedule,
    OptimizerParameters,
    TrainingParameters,
)
from artifacts.core.sweep import (
    ArtifactSweep,
    Record,
    Sweep,
    UnresolvedDeclarationError,
    expand,
)
from artifacts.dataset import DataSet
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.tokenizers.bpe import Tokenizer
from models.transformer import Transformer


@dataclass(frozen=True)
class Config(Record):
    width: int
    depth: int = 1

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("width must be positive")


@dataclass(frozen=True)
class Node(Artifact):
    producer: ClassVar[None] = None
    value: int
    left: Node | None = None
    right: Node | None = None
    items: tuple[Node, ...] = ()
    group: frozenset[Node] = frozenset()
    config: Config | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.value < 0:
            raise ValueError("value must be nonnegative")

    @property
    def uid(self):
        return _digest(
            self.value,
            self.left.uid if self.left else None,
            self.right.uid if self.right else None,
            [node.uid for node in self.items],
            sorted(node.uid for node in self.group),
            asdict(self.config) if self.config else None,
        )

    @property
    def artifact_path(self):
        return Path("nodes") / self.uid

    @property
    def files(self):
        return {}


@dataclass(frozen=True)
class ParameterArtifact(Artifact):
    """A test artifact holding one record, `params`, whose annotation a
    subclass declares (see `holding`)."""

    producer: ClassVar[None] = None

    @property
    def uid(self):
        return _digest(asdict(self.params))

    @property
    def artifact_path(self):
        return Path("parameters") / self.uid

    @property
    def files(self):
        return {}


def holding(record: type) -> type[ParameterArtifact]:
    """Returns a ParameterArtifact whose `params` is annotated `record`."""

    def body(namespace):
        namespace["__annotations__"] = {"params": record}

    return dataclass(frozen=True)(types.new_class(f"{record.__name__}Artifact", (ParameterArtifact,), exec_body=body))


def test_constructor_dispatch_and_concrete_validation():
    concrete = Node(1)
    template = Node(Sweep((1, -1)))
    assert isinstance(concrete, Artifact)
    assert isinstance(template, ArtifactSweep)
    assert not isinstance(template, Artifact)
    assert template.value.values == (1, -1)
    assert "value" in signature(Node).parameters
    assert "self" not in signature(Node).parameters
    with pytest.raises(ValueError, match="nonnegative"):
        Node(-1)
    with pytest.raises(ValueError, match=r"(?s)nonnegative.*root \[root\.value\.values\[1\]\]"):
        resolve(template)
    with pytest.raises(TypeError, match=r"Node.value.values\[1\]"):
        Node(Sweep((1, "wrong")))


def test_children_expand_before_concrete_parent_validation():
    child = Node(1, config=Config(Sweep((2, 4)), depth=Sweep((1, 3))))
    parent = Node(2, left=child)
    assert isinstance(parent, ArtifactSweep)
    roots, graph = expand(parent), resolve(parent)
    assert len(roots) == 4
    assert len(graph) == 8
    assert [(r.left.config.width, r.left.config.depth) for r in roots] == [
        (2, 1),
        (4, 1),
        (2, 3),
        (4, 3),
    ]
    assert all(isinstance(node, Node) for node in graph)
    assert all(Artifact.from_manifest(node.to_manifest()) == node for node in graph)


def test_every_sweep_multiplies_and_a_shared_declaration_is_one_node():
    axis = Sweep((1, 2))
    # the same Sweep in two positions is two choices
    assert len(expand(Node(0, config=Config(axis, axis)))) == 4
    assert len(expand(Node(3, left=Node(axis), right=Node(axis)))) == 4
    # the same declaration in two positions is one node, chosen once
    child = Node(axis)
    request = Node(3, left=child, right=child)
    roots, graph = expand(request), resolve(request)
    assert [(n.left.value, n.right.value) for n in roots] == [(1, 1), (2, 2)]
    assert len(graph) == 4  # two leaves and two parents


def test_a_shared_declaration_sweeps_as_one_through_every_occurrence():
    tokenizer = Tokenizer(1000, Sweep((("<|endoftext|>",), ("<|otherend|>",))), sources=(SourceURL("a", "url"),))
    dataset = DataSet.from_sources(tokenizer, (SourceURL("a", "url"),), (SourceURL("a", "url"),))
    datasets = expand(dataset)
    assert [d.tokenizer.special_tokens for d in datasets] == [("<|endoftext|>",), ("<|otherend|>",)]


def test_nested_alternatives_add_and_other_fields_multiply():
    roots = expand(
        Node(
            Sweep((10, 20)),
            left=Sweep((Node(Sweep((1, 2))), Node(3))),
        )
    )
    assert len(roots) == (2 + 1) * 2
    assert [n.left.value for n in roots] == [1, 1, 2, 2, 3, 3]
    assert len(resolve(Sweep((Sweep((Node(1), Node(2))), Node(3))))) == 3


def test_concrete_artifact_is_a_sweep_of_one_and_groups_are_sweeps():
    first = Node(1, commit="first", allocated_resources=Resources(cpu=1))
    duplicate = replace(first, commit="later", allocated_resources=Resources(cpu=2))
    parent = Node(2, left=duplicate)
    group = Sweep((first, parent, duplicate))
    assert expand(first) == [first]
    assert resolve(first) == [first]
    assert expand(group) == [first, parent, duplicate]
    graph = resolve(group)
    assert graph == [first, parent]
    assert graph[0] is first


def test_conflicting_alternatives_name_both_choices():
    request = SourceURL("same", Sweep(("one", "two")))
    with pytest.raises(
        ValueError,
        match=r"different definitions at sources/same.*roots\[0\].*roots\[1\]",
    ):
        resolve(request)


def test_tuple_set_and_optional_dependency_choices():
    a, b = Node(1), Node(2)
    roots = expand(Node(3, items=(Sweep((a, b)), a)))
    assert [n.items for n in roots] == [(a, a), (b, a)]
    roots = expand(Node(3, items=Sweep(((a, b), (b, a), ()))))
    assert [n.items for n in roots] == [(a, b), (b, a), ()]
    roots = expand(Node(3, group=(Sweep((a, b)), a)))
    assert [n.group for n in roots] == [frozenset((a,)), frozenset((a, b))]
    roots = expand(Node(3, left=Sweep((None, a))))
    assert [n.left for n in roots] == [None, a]


def test_shared_parameter_defaults_are_captured_once():
    calls = []

    def default():
        calls.append(1)
        return 7

    @dataclass(frozen=True)
    class Settings:
        width: int
        stamp: int = field(default_factory=default)

    settings = Settings(Sweep((1, 2)))
    assert calls == [1]
    assert len(resolve(holding(Settings)(settings))) == 2
    # A parameter record is not itself a root artifact.
    with pytest.raises(TypeError, match="concrete Artifact"):
        resolve(settings)
    assert calls == [1]


@pytest.mark.parametrize("values", [(), [], "abc", {1, 2}, iter((1, 2))])
def test_invalid_choice_containers(values):
    with pytest.raises((TypeError, ValueError)):
        Sweep(values)


def test_ranges_snapshot_and_singletons():
    values = [1, 2]
    axis = Sweep(values)
    values.append(3)
    assert axis.values == (1, 2)
    assert len(resolve(Node(Sweep(range(3))))) == 3
    assert resolve(Node(Sweep((1,)))) == resolve(Node(1))


def test_concrete_only_boundaries_and_transport():
    template = Node(2, left=Node(Sweep((0, 1))), config=Config(Sweep((2, 4))))
    concrete = Node(1)
    assert Artifact.require_concrete(concrete) is concrete
    with pytest.raises(UnresolvedDeclarationError, match="concrete Artifact"):
        Artifact.require_concrete(template)
    for name in (
        "uid",
        "artifact_path",
        "deps",
        "parameters",
        "to_manifest",
        "bind",
        "status",
        "job",
    ):
        with pytest.raises(UnresolvedDeclarationError, match="expand"):
            getattr(template, name)
    with pytest.raises(TypeError):
        hash(template)
    with pytest.raises(UnresolvedDeclarationError):
        to_manifest(template)
    restored = pickle.loads(pickle.dumps(template))
    assert resolve(restored) == resolve(template)
    assert isinstance(restored, ArtifactSweep)


def test_base_class_enforces_concrete_fields_without_subclass_checks():
    @dataclass(frozen=True)
    class InjectSweep(Node):
        def __post_init__(self):
            # Even a hook that omits super() cannot return an unresolved Artifact.
            object.__setattr__(self, "value", Sweep((1, 2)))

    with pytest.raises(UnresolvedDeclarationError, match="InjectSweep.value"):
        InjectSweep(1)


def test_metadata_and_non_sweepable_fields_reject_choices():
    with pytest.raises(UnresolvedDeclarationError, match="commit"):
        Node(1, commit=Sweep(("one", "two")))
    with pytest.raises(UnresolvedDeclarationError, match="allocated_resources.cpu"):
        Node(1, allocated_resources=Resources(cpu=Sweep((1, 2))))

    @dataclass(frozen=True)
    class Fixed:
        value: int = field(metadata={"sweepable": False})

    with pytest.raises(UnresolvedDeclarationError, match="params.value"):
        holding(Fixed)(Fixed(Sweep((1, 2))))


def test_wrong_dependency_type_and_hidden_choices():
    with pytest.raises(TypeError, match="Node.left"):
        Node(1, left=SourceURL(Sweep(("a", "b")), "url"))

    @dataclass(frozen=True)
    class Hidden:
        value: Node

    with pytest.raises(TypeError, match="hidden in a parameter"):
        holding(Hidden)(Hidden(Node(Sweep((1, 2)))))


def test_fields_must_name_a_concrete_type():
    @dataclass(frozen=True)
    class Loose:
        value: Any

    @dataclass(frozen=True)
    class Bare:
        value: tuple

    for record in (Loose(1), Bare((1,))):
        with pytest.raises(TypeError, match="needs a concrete type"):
            holding(type(record))(record)


def test_post_init_injection_is_rejected():
    @dataclass(frozen=True)
    class Broken:
        value: int

        def __post_init__(self):
            object.__setattr__(self, "value", Sweep((1, 2)))

    with pytest.raises(UnresolvedDeclarationError):
        resolve(holding(Broken)(Broken(1)))


@pytest.fixture
def pretraining():
    books = SourceURL("books", "books-url")
    code = SourceURL("code", "code-url")
    tokenizer = Tokenizer(1000, ("<pad>", "<|endoftext|>"), sources=(books, code))
    datasets = [
        DataSet.from_sources(tokenizer, (source,), (source,))
        for source in (books, code)
    ]
    return Pretraining(
        run_id="sweep",
        dataset=Sweep(datasets),
        model=Transformer(
            vocab_size=1000,
            sequence_length=8,
            d_model=Sweep((64, 128, 256)),
            num_layers=Sweep((4, 8)),
            d_ff=128,
            num_heads=4,
            rope_theta=10000,
        ),
        training_parameters=TrainingParameters(
            total_steps=4,
            batch_size=4,
            max_norm=1.0,
            lr_schedule=LRSchedule(1e-3, 1e-4, 1, 100),
            optimizer="torch.optim.AdamW",
            optimizer_parameters=OptimizerParameters(1e-3, (0.9, 0.95), 0.1, 1e-8),
        ),
        loop_config=LoopConfig(val_every=2, gpu_check_every=0),
    )


def test_pretraining_twelve_roots_and_shared_dag(pretraining):
    roots, graph = expand(pretraining), resolve(pretraining)
    assert len(roots) == 12
    assert len([n for n in graph if isinstance(n, Tokenizer)]) == 1
    assert len([n for n in graph if isinstance(n, DataSet)]) == 2
    assert all(Artifact.from_manifest(n.to_manifest()) == n for n in roots)


def test_checkpoint_inheritance_and_twenty_four_continuations(pretraining):
    continuation = Pretraining(
        run_id="continued",
        starting_checkpoint=pretraining,
        dataset=pretraining.dataset.values[0],
        training_parameters=replace(
            pretraining.training_parameters, seed=Sweep((10, 20))
        ),
        loop_config=pretraining.loop_config,
    )
    roots, graph = expand(continuation), resolve(continuation)
    assert len(roots) == 24
    assert len([n for n in graph if isinstance(n, Pretraining)]) == 36
    assert all(
        n.model == n.starting_checkpoint.model for n in roots
    )
    assert all(n.start_step == 4 and n.end_step == 8 for n in roots)


def test_declare_preflight_prevents_partial_publication(tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        lab.declare(Node(Sweep((1, -1))), root=tmp_path, commit=True)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="different definitions"):
        lab.declare(SourceURL("same", Sweep(("a", "b"))), root=tmp_path, commit=True)
    assert list(tmp_path.iterdir()) == []


def test_declare_combined_report_and_storage_blocker(tmp_path):
    request = Sweep(
        (SourceURL("a", "url"), SourceURL("b", "url"), SourceURL("a", "url"))
    )
    report = lab.declare(request, root=tmp_path)
    assert report.root_paths == ["sources/a", "sources/b"]
    assert len(report.rows) == 2
    assert list(tmp_path.iterdir()) == []
    lab.declare(SourceURL("b", "different"), root=tmp_path, commit=True, notebook=b"{}")
    with pytest.raises(lab.DeclarationError):
        lab.declare(request, root=tmp_path, commit=True, notebook=b"{}")
    assert not (tmp_path / "sources/a").exists()


def test_notebooks_for_each_new_training_root_and_redeclaration(tmp_path, pretraining):
    report = lab.declare(pretraining, root=tmp_path, commit=True, notebook=b"original")
    assert len(report.root_paths) == 12
    copies = list((tmp_path / "runs/sweep").glob("declare-*.ipynb"))
    assert len(copies) == 12
    assert len(list(tmp_path.rglob("manifest.json"))) == len(report.rows)
    again = lab.declare(pretraining, root=tmp_path, commit=True, notebook=b"later")
    assert not any(row["created"] or row["updated"] for row in again.rows)
    assert all(path.read_bytes() == b"original" for path in copies)


def test_nested_dataset_example_has_eighteen_roots(pretraining):
    books, code = pretraining.dataset.values
    books_variants = DataSet(
        train_set=Sweep((books.train_set, books.train_set + code.train_set)),
        valid_set=books.valid_set,
    )
    inputs = dict(pretraining._inputs)
    inputs["dataset"] = Sweep((books_variants, code))
    request = Pretraining(**inputs)
    roots, graph = expand(request), resolve(request)
    assert len(roots) == 18
    assert len([node for node in graph if isinstance(node, DataSet)]) == 3


def test_invalid_model_product_fails_before_writes(tmp_path, pretraining):
    inputs = dict(pretraining._inputs)
    inputs["model"] = replace(
        pretraining.model, d_model=Sweep((64, 66))
    )
    with pytest.raises(ValueError, match=r"(?s)divisible.*values\[1\]"):
        lab.declare(Pretraining(**inputs), root=tmp_path, commit=True, notebook=b"{}")
    assert list(tmp_path.iterdir()) == []


def test_set_choices_check_collisions_before_normalization():
    original = SourceURL("books", "original")
    conflicting = SourceURL("books", "conflict")
    declaration = Tokenizer(
        1000,
        (),
        sources=(Sweep((original, conflicting)), original),
    )
    with pytest.raises(ValueError, match=r"(?s)more than one definition.*values\[1\]"):
        resolve(declaration)


def test_roots_are_retained_when_also_requested_as_dependencies(tmp_path):
    leaf = Node(1)
    parent = Node(2, left=leaf)
    report = lab.declare(Sweep((parent, leaf)), root=tmp_path)
    assert report.root_paths == [str(parent.artifact_path), str(leaf.artifact_path)]
    assert [row["path"] for row in report.rows] == [
        str(leaf.artifact_path),
        str(parent.artifact_path),
    ]


def test_normalization_cannot_rewrite_choices_or_insert_dependencies():
    @dataclass(frozen=True)
    class Rewrite:
        value: int

        def __post_init__(self):
            if not isinstance(self.value, Sweep):
                object.__setattr__(self, "value", 0)

    with pytest.raises(ValueError, match="changed a chosen input"):
        resolve(holding(Rewrite)(Rewrite(Sweep((1, 2)))))

    @dataclass(frozen=True)
    class InsertDependency(Node):
        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(self, "left", Node(1))

    with pytest.raises(ValueError, match="InsertDependency.left"):
        InsertDependency(2)


@dataclass
class Mutable:
    value: int


@dataclass(frozen=True)
class Unregistered:
    value: int


@dataclass(frozen=True)
class Holder:
    value: Unregistered


def test_mutable_nested_and_mapping_parameter_records():
    with pytest.raises(TypeError, match="frozen dataclass"):
        holding(Mutable)(Mutable(Sweep((1, 2))))
    assert len(resolve(holding(Holder)(Holder(Unregistered(Sweep((1, 2))))))) == 2
    with pytest.raises(TypeError, match="params.value: Input should be an instance of Unregistered"):
        holding(Holder)(Holder({"key": 1}))


def test_modal_transport_and_remote_declaration_wrapper(tmp_path, pretraining):
    from modal._serialization import deserialize, serialize

    import local

    transported = deserialize(serialize(pretraining), None)
    local_report = lab.declare(pretraining, root=tmp_path)
    remote_report = local._declare.get_raw_f()(transported, root=tmp_path)
    returned = deserialize(serialize(remote_report), None)
    assert returned == local_report
    assert list(tmp_path.iterdir()) == []


@dataclass(frozen=True)
class Rates:
    lr: float
    optional: float | None = None
    pair: tuple[float, ...] = (0.003, 1e-8)


@dataclass(frozen=True)
class RateArtifact(ParameterArtifact):
    params: Rates


def test_plain_records_validate_at_artifact_boundary():
    record = Rates(Sweep((0.003, "invalid")))
    assert isinstance(record, Rates)
    with pytest.raises(TypeError, match=r"params.lr.values\[1\]"):
        RateArtifact(record)
    with pytest.raises(ValueError, match="width must be positive"):
        resolve(Node(0, config=Config(Sweep((1, -1)))))


def test_float_json_disk_roundtrip_preserves_bits_and_identity(tmp_path):
    import json
    import math
    import sys

    values = (
        0.003,
        math.nextafter(0.003, math.inf),
        1e-8,
        -0.0,
        math.ulp(0.0),
        sys.float_info.max,
    )
    request = RateArtifact(Rates(Sweep(values), optional=Sweep((None, 0.003))))
    roots = expand(request)
    assert len(roots) == len(values) * 2
    lab.declare(request, root=tmp_path, commit=True, notebook=b"")
    for original in roots:
        loaded = Artifact.load(original.artifact_path, root=tmp_path)
        assert loaded.params.lr.hex() == original.params.lr.hex()
        assert loaded.params.pair == original.params.pair
        assert loaded.params.optional == original.params.optional
        assert loaded.artifact_path == original.artifact_path
        data = (tmp_path / original.artifact_path / "manifest.json").read_text()
        assert (
            json.loads(data)["parameters"]["params"]["lr"].hex()
            == original.params.lr.hex()
        )
        if original.params.lr == 0.003:
            assert "0.003" in data


def test_training_learning_rate_json_roundtrip(tmp_path, pretraining):
    original = expand(pretraining)[0]
    rates = (0.003, 0.0003, 1e-8)
    request = replace(
        original,
        training_parameters=replace(
            original.training_parameters,
            optimizer_parameters=replace(
                original.training_parameters.optimizer_parameters, lr=Sweep(rates)
            ),
        ),
    )
    roots = expand(request)
    lab.declare(request, root=tmp_path, commit=True, notebook=b"{}")
    for root, rate in zip(roots, rates):
        loaded = Artifact.load(root.artifact_path, root=tmp_path)
        assert loaded.training_parameters.optimizer_parameters.lr.hex() == rate.hex()
        assert loaded.artifact_path == root.artifact_path


class Refused(ValueError):
    """An error whose constructor takes more than a message."""

    def __init__(self, field, reason):
        super().__init__(f"{field}: {reason}")


@dataclass(frozen=True)
class Picky(Record):
    width: int

    def __post_init__(self):
        if self.width > 1:
            raise Refused("width", "too wide")


def test_an_expansion_error_keeps_its_class_and_names_its_choice():
    with pytest.raises(Refused, match=r"(?s)width: too wide.*root \[root\.params\.width\.values\[1\]\]"):
        expand(holding(Picky)(Picky(Sweep((1, 2)))))
