import json
import math
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

# Execute only the notebook's helper definitions, never its remote capture cells.
_notebook_path = Path(__file__).resolve().parents[1] / "benchmarking" / "toyV2.ipynb"
_notebook = json.loads(_notebook_path.read_text())
_helpers = {}
for _cell in _notebook["cells"]:
    _source = "".join(_cell["source"])
    if _cell["cell_type"] == "code" and any(
        f"def {name}(" in _source
        for name in ("module_scopes", "module_branches", "aggregate_kernels", "kernel_table")
    ):
        exec(compile(_source, str(_notebook_path), "exec"), _helpers)  # noqa: S102 - local notebook definitions
MODULE_SCOPE = _helpers["MODULE_SCOPE"]
module_scopes = _helpers["module_scopes"]
module_branches = _helpers["module_branches"]
aggregate_kernels = _helpers["aggregate_kernels"]
kernel_table = _helpers["kernel_table"]


@pytest.fixture
def phase_rows():
    def cpu(index, name, start, end, parent=None, flops=0, activity="cpu_op"):
        return {"index": index, "name": name, "start_us": start, "end_us": end,
                "device_type": "DeviceType.CPU", "activity_type": activity,
                "parent": parent, "flops": flops, "stack": ["source.py:42"]}

    def gpu(index, launcher, start, end, activity="kernel"):
        return {"index": index, "name": "same kernel", "start_us": start, "end_us": end,
                "duration_us": end - start, "device_type": "DeviceType.CUDA",
                "activity_type": activity, "launcher": launcher}

    return [
        cpu(100, "<frozen runpy>", 0, 1000, activity="python_function"),
        cpu(101, "aten::outer_wrapper", 1, 999, 100),
        cpu(102, "train::forward", 10, 80, 101, activity="user_annotation"),
        # Module-free forward work, two nested same-named ATen events, two launches.
        cpu(103, "aten::mm", 12, 30, 102, flops=128),
        cpu(104, "aten::mm", 14, 24, 103, flops=256),
        gpu(1, 104, 50, 52), gpu(2, 104, 53, 56),
        cpu(105, MODULE_SCOPE + "model.layer", 31, 70, 102, activity="user_annotation"),
        cpu(106, "aten::add", 35, 50, 105), gpu(3, 106, 70, 72),
        # Detached worker: no train::backward and no Python root.
        cpu(200, "autograd::engine::evaluate_function: MmBackward0", 100, 150),
        cpu(201, _helpers["BACKWARD_MODULE_SCOPE"] + "model.layer", 102, 145, 200,
            activity="user_annotation"),
        cpu(202, "aten::mm", 105, 130, 201, flops=512), gpu(4, 202, 140, 145),
        # Module-free backward and optimizer phases.
        cpu(203, "autograd::engine::evaluate_function: AccumulateGrad", 155, 165),
        cpu(204, "aten::add_", 157, 164, 203), gpu(5, 204, 165, 168),
        cpu(300, "train::optimizer_step", 200, 250, 100, activity="user_annotation"),
        cpu(301, "Optimizer.step#AdamW.step", 201, 249, 300, activity="user_annotation"),
        cpu(302, "aten::mul_", 210, 220, 301), gpu(6, 302, 230, 233),
        # Phase known, but there is no ATen or cpu_op event: CPU timing stays missing.
        cpu(400, "train::loss", 300, 350, 100, activity="user_annotation"),
        cpu(401, "cudaLaunchKernel", 310, 315, 400, activity="cuda_runtime"),
        gpu(7, 401, 330, 332),
        # Truly missing correlation must never be dropped or assigned a made-up CPU time.
        gpu(8, None, 400, 403), gpu(9, 104, 60, 61, "gpu_memcpy"),
    ]


def test_phase_partition_includes_module_free_launches_and_raw_event_lookup(phase_rows):
    kernels = kernel_table(phase_rows, depth=2)
    groups = aggregate_kernels(kernels)
    assert kernels.index.names == ["kernel_index"]
    assert "index" not in kernels.columns and "kernel_index" not in kernels.columns
    assert groups.index.names == ["phase", "module_fqn"]
    assert kernels.query("phase == 'forward' and module_fqn == '<unattributed>'").index.tolist() == [1, 2]
    assert kernels.query("phase == 'backward' and module_fqn == '<unattributed>'").index.tolist() == [5]
    assert kernels.query("phase == 'optimizer_step' and module_fqn == '<unattributed>'").index.tolist() == [6]
    assert kernels.query("phase == 'loss' and module_fqn == '<unattributed>'").index.tolist() == [7]
    assert kernels.query("phase == 'unscoped' and module_fqn == '<unattributed>'").index.tolist() == [8]
    assert groups["launches"].sum() == 8
    assert set(kernels.index.get_level_values("kernel_index")) == set(range(1, 9))
    assert len(kernel_table(phase_rows, kernels_only=False)) == 9
    branch = kernels.loc[1, "branch"]
    events = pd.DataFrame(phase_rows).set_index("index")
    assert events.loc[branch, "name"].tolist() == [
        "same kernel", "aten::mm", "aten::mm", "train::forward",
        "aten::outer_wrapper", "<frozen runpy>",
    ]


def test_cpu_start_is_bounded_by_phase_and_flops_use_max(phase_rows):
    kernels = kernel_table(phase_rows)
    forward = kernels.loc[1]
    assert forward["cpu_start_index"] == 103
    assert forward["cpu_start_us"] == 12  # Not the outer wrapper at 1 or Python root at 0.
    assert forward["cpu_end_us"] == 30
    assert forward["phase_start_us"] == 10
    assert forward["cpu_to_gpu_end_us"] == 40
    assert forward["aten_ops"] == ["aten::outer_wrapper", "aten::mm", "aten::mm"]
    assert forward["aten_event_indices"] == [101, 103, 104]
    assert forward["flops"] == 256
    backward = kernels.loc[4]
    assert backward["cpu_start_us"] == 105
    assert backward["phase_start_us"] == 100
    assert backward["flops"] == 512
    untraced = kernels.loc[7]
    assert untraced["phase"] == "loss"
    assert pd.isna(untraced["cpu_start_source"]) and pd.isna(untraced["cpu_start_index"])
    assert pd.isna(untraced["cpu_start_us"]) and pd.isna(untraced["cpu_to_gpu_end_us"])
    assert untraced["aten_ops"] == [] and untraced["flops"] == 0
    unknown = kernels.loc[8]
    assert pd.isna(unknown["cpu_start_us"]) and pd.isna(unknown["cpu_to_gpu_end_us"])
    bounded = kernels.dropna(subset=["cpu_start_us", "phase_start_us"])
    assert (bounded["cpu_start_us"] >= bounded["phase_start_us"]).all()
    group = aggregate_kernels(kernels).loc[("forward", "<unattributed>")]
    assert group["cpu_start_us"] == 12 and group["gpu_start_us"] == 50 and group["gpu_end_us"] == 56
    assert group["cpu_to_gpu_end_us"] == 44
    assert group["cpu_timed_launches"] == 2 and group["gpu_time_us"] == 5
    # The same operator launches both kernels; do not add its FLOPs twice.
    assert group["max_flops"] == 256


@pytest.mark.parametrize("intervals,span,total", [
    ([(10, 12), (15, 18)], 8, 5),  # Gap is in the span, not in summed GPU work.
    ([(10, 15), (12, 18)], 8, 11),  # Overlap contributes to both launch durations.
])
def test_gpu_span_and_summed_duration_are_distinct(phase_rows, intervals, span, total):
    kernels = kernel_table(phase_rows).loc[[1, 2]].copy()
    kernels["gpu_start_us"] = [start for start, end in intervals]
    kernels["gpu_end_us"] = [end for start, end in intervals]
    kernels["duration_us"] = [end - start for start, end in intervals]
    group = aggregate_kernels(kernels).iloc[0]
    assert group["gpu_span_us"] == span
    assert group["gpu_time_us"] == total
    assert group["launches"] == 2


def test_empty_table_and_missing_phase_rejection(phase_rows):
    empty = kernel_table([])
    assert empty.empty and aggregate_kernels(empty).empty
    for invalid in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="depth"):
            kernel_table([], depth=invalid)
    invalid_phase = kernel_table(phase_rows).iloc[:1].copy()
    invalid_phase["phase"] = None
    with pytest.raises(ValueError, match="nonempty phase"):
        aggregate_kernels(invalid_phase)


@pytest.mark.parametrize("depth", [None, 1, 2, 3, 4, 5, 6, 100])
def test_kernel_groups_are_disjoint_at_every_depth(depth):
    owners = [
        "model", "model.layers.3", "model.layers.3.self_attn",
        "model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.q_proj", "model.layers.3.mlp",
        "model.layers.4.self_attn", None, None,
    ]
    rows = []
    for i, owner in enumerate(owners):
        phase = "forward" if i % 2 else "backward"
        rows.append({"index": 200 + i, "name": "train::" + phase,
                     "device_type": "DeviceType.CPU", "parent": None})
        launcher = 200 + i
        if owner:
            prefix = MODULE_SCOPE if phase == "forward" else _helpers["BACKWARD_MODULE_SCOPE"]
            rows.append({"index": 100 + i, "name": prefix + owner,
                         "device_type": "DeviceType.CPU", "parent": launcher})
            launcher = 100 + i
        rows.append({"index": i, "name": "same kernel", "activity_type": "kernel",
                     "device_type": "DeviceType.CUDA", "launcher": launcher,
                     "duration_us": i + 0.5, "start_us": (10 - i) * 3,
                     "end_us": (10 - i) * 3 + i + 0.5})
    kernels = kernel_table(rows, depth=depth)
    groups = aggregate_kernels(kernels)
    memberships = [set(indices) for indices in groups["event_indices"]]
    for left, right in combinations(memberships, 2):
        assert left.isdisjoint(right)
    assert set().union(*memberships) == set(range(len(owners)))
    for key, group in groups.iterrows():
        members = kernels.loc[group["event_indices"]]
        assert group["launches"] == len(members) == len(group["event_indices"])
        assert group["gpu_time_us"] == members["duration_us"].sum()
        assert group["gpu_start_us"] == members["gpu_start_us"].min()
        assert group["gpu_end_us"] == members["gpu_end_us"].max()
        assert group["gpu_span_us"] == group["gpu_end_us"] - group["gpu_start_us"]
        for i in members.index:
            owner = owners[i]
            expected = ".".join(owner.split(".")[:depth]) if owner and depth else owner
            assert key[1] == (expected or "<unattributed>")
    assert groups["launches"].sum() == len(owners)
    assert set(kernels["phase"]) == {"forward", "backward"}
    # A conflicting phase and module do not turn one event into a new launch.
    duplicate = kernels.iloc[:1].copy()
    duplicate["phase"] = "other_phase"
    duplicate["module_fqn"] = "other_module"
    with pytest.raises(ValueError, match="Duplicate device event"):
        aggregate_kernels(pd.concat([kernels, duplicate]))


def test_exclusive_partition_and_rollup():
    def cpu(index, name, parent=None):
        return {"index": index, "name": name, "parent": parent, "device_type": "DeviceType.CPU"}

    def gpu(index, launcher, kind="kernel"):
        return {"index": index, "name": "same kernel", "launcher": launcher,
                    "activity_type": kind, "device_type": "DeviceType.CUDA",
                    "duration_us": 2.5, "start_us": index, "end_us": index + 2.5}

    rows = [cpu(0, MODULE_SCOPE + "model"),
            cpu(1, MODULE_SCOPE + "model.layers.3", 0),
            cpu(2, MODULE_SCOPE + "model.layers.3.self_attn", 1),
            cpu(3, "aten::mm", 2),
            gpu(4, 3), gpu(5, 3), gpu(6, 1), gpu(7, None),
            gpu(8, 3, "gpu_memcpy"), cpu(9, "outside"),
            cpu(10, "aten::mm", 9), gpu(11, 10)]
    attributed, unattributed = module_branches(rows)
    branches = attributed + unattributed
    assert [b["module_fqn"] for b in attributed] == [
        "model.layers.3.self_attn", "model.layers.3.self_attn",
        "model.layers.3",
    ]
    assert [b["branch"] for b in unattributed] == [[7], [11, 10, 9]]
    for b in unattributed:
        assert b["module_fqn"] is None
        assert b["module_path"] is None
        assert b["kernel"] == "same kernel"
        assert b["duration_us"] == 2.5
        assert b["gpu_start_us"] == b["index"]
        assert b["gpu_end_us"] == b["index"] + 2.5
    assert branches[0]["branch"] == [4, 3, 2, 1, 0]
    for depth in [None, 1, 2, 3, 4]:
        groups = aggregate_kernels(kernel_table(rows, depth=depth))
        ids = [i for indices in groups["event_indices"] for i in indices]
        assert sorted(ids) == [4, 5, 6, 7, 11]
        assert math.isclose(groups["gpu_time_us"].sum(), 12.5)
    rolled = aggregate_kernels(kernel_table(rows, depth=3))
    assert rolled.loc[("forward", "model.layers.3"), "launches"] == 3
    all_attributed, all_unattributed = module_branches(rows, kernels_only=False)
    assert [b["index"] for b in all_attributed] == [4, 5, 6, 8]
    assert all_unattributed == unattributed
    assert module_branches([]) == ([], [])
    assert module_branches([rows[7]]) == ([], unattributed[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_kernels(pd.concat([kernel_table(rows), kernel_table(rows).iloc[:1]]))


def test_real_cpu_scopes_shared_aliases_and_cleanup():
    class Attention(nn.Module):
        def __init__(self, rope):
            super().__init__()
            self.rope = rope

        def forward(self, x):
            return self.rope(x) * 2

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = nn.ReLU()
            self.attention = Attention(self.rope)

        def forward(self, x):
            return self.attention(x) + 1

    model = Model()
    x = torch.tensor([-1.0, 2.0])
    expected = model(x)
    with module_scopes(model), torch.profiler.profile() as prof:
        actual = model(x)
        model(x)
    torch.testing.assert_close(actual, expected)
    scopes = [e for e in prof.events() if e.name.startswith(MODULE_SCOPE)]
    assert [e.name.removeprefix(MODULE_SCOPE) for e in scopes] == [
        "model", "model.attention", "model.attention.rope",
    ] * 2
    for event in prof.events():
        if event.name == "aten::relu":
            assert event.cpu_parent.name == MODULE_SCOPE + "model.attention.rope"
        if event.name == "aten::mul":
            assert event.cpu_parent.name == MODULE_SCOPE + "model.attention"
        if event.name == "aten::add":
            assert event.cpu_parent.name == MODULE_SCOPE + "model"
    for module in model.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_hooks_removed_after_forward_exception():
    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("broken")

    model = nn.Sequential(Broken())
    with pytest.raises(RuntimeError, match="broken"), module_scopes(model), torch.profiler.profile():
        model(torch.ones(1))
    for module in model.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_backward_provenance_preserves_gradients_and_restores_module_paths():
    class Block(nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.shared = shared

        def forward(self, x):
            return self.shared(x).relu()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(3, 3)
            self.left = Block(shared)
            self.right = Block(shared)

        def forward(self, x):
            return self.left(x) + self.right(x)

    model = Model()
    baseline = deepcopy(model)
    x = torch.randn(2, 3, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    baseline(reference_x).square().sum().backward()

    with torch.profiler.profile() as prof, module_scopes(model):
        output = model(x)
        assert output.grad_fn.metadata["launchpad_module"]["module_fqn"] == "model"
        output.square().sum().backward()
    for actual, expected in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    torch.testing.assert_close(x.grad, reference_x.grad)

    backward_prefix = _helpers["BACKWARD_MODULE_SCOPE"]
    events = list(prof.events())
    paths = []
    for event in events:
        if event.name != "aten::mm":
            continue
        path, node = [], event
        while node is not None:
            if node.name.startswith(backward_prefix):
                path.append(node.name.removeprefix(backward_prefix))
            node = node.cpu_parent
        paths.append(path)
    assert paths
    assert all(path in (["model.left.shared", "model.left", "model"],
                        ["model.right.shared", "model.right", "model"]) for path in paths)
    assert {p[0] for p in paths} == {"model.left.shared", "model.right.shared"}

    # Exercise the real recorded CPU ancestry with synthetic GPU launch edges.
    positions = {id(e): i for i, e in enumerate(events)}
    rows = [{"index": i, "name": e.name, "device_type": "DeviceType.CPU",
             "parent": positions.get(id(e.cpu_parent))} for i, e in enumerate(events)]
    for event in events:
        if event.name in ("aten::mm", "aten::addmm"):
            rows.append({"index": len(rows), "name": "test kernel", "activity_type": "kernel",
                         "device_type": "DeviceType.CUDA", "launcher": positions[id(event)],
                         "start_us": event.time_range.start, "end_us": event.time_range.end,
                         "duration_us": event.time_range.elapsed_us()})
    attributed, unattributed = module_branches(rows)
    assert not unattributed
    assert {b["phase"] for b in attributed} == {"forward", "backward"}
    assert all(b["module_path"] == ["model", b["module_fqn"].rsplit(".", 1)[0], b["module_fqn"]]
               for b in attributed)
    for depth in [None, 1, 2, 3, 10]:
        groups = aggregate_kernels(kernel_table(rows, depth=depth))
        indices = [i for ids in groups["event_indices"] for i in ids]
        assert len(indices) == len(set(indices)) == len(attributed)
        assert set(indices) == {b["index"] for b in attributed}


def test_external_graph_is_not_claimed_and_node_hooks_are_removed():
    x = torch.ones(3, requires_grad=True)
    external = x.sin()
    model = nn.ReLU()
    with module_scopes(model):
        y = model(external)
        assert "launchpad_module" not in external.grad_fn.metadata
        assert y.grad_fn.metadata["launchpad_module"]["module_fqn"] == "model"
    # Exiting the context removes node hooks even if backward has not run yet.
    with torch.profiler.profile() as prof:
        y.sum().backward()
    assert not any(e.name.startswith(_helpers["BACKWARD_MODULE_SCOPE"]) for e in prof.events())


def test_retained_graph_reopens_backward_scopes():
    model = nn.Linear(3, 2)
    with torch.profiler.profile() as prof, module_scopes(model):
        y = model(torch.randn(2, 3))
        y.sum().backward(retain_graph=True)
        y.sum().backward()
    backward_scopes = [e for e in prof.events() if e.name == _helpers["BACKWARD_MODULE_SCOPE"] + "model"]
    # AddmmBackward and TBackward each execute on both traversals.
    assert len(backward_scopes) == 4
