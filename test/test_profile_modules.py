import json
import math
from itertools import combinations
from pathlib import Path

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
        for name in ("module_scopes", "module_branches", "aggregate_modules")
    ):
        exec(compile(_source, str(_notebook_path), "exec"), _helpers)  # noqa: S102 - local notebook definitions
MODULE_SCOPE = _helpers["MODULE_SCOPE"]
module_scopes = _helpers["module_scopes"]
module_branches = _helpers["module_branches"]
aggregate_modules = _helpers["aggregate_modules"]


@pytest.mark.parametrize("depth", [None, 1, 2, 3, 4, 5, 6, 100])
def test_module_groups_are_disjoint_at_every_depth(depth):
    # Parent-owned work, siblings, nested children, repeated calls and no owner.
    owners = [
        "model", "model.layers.3", "model.layers.3.self_attn",
        "model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.q_proj", "model.layers.3.mlp",
        "model.layers.4.self_attn", None, None,
    ]
    branches = [
        {"index": i, "module_fqn": owner, "duration_us": i + 0.5,
         "start_us": (10 - i) * 3, "end_us": (10 - i) * 3 + i + 0.5,
         "branch": [i, 100, 101], "kernel": "same kernel"}
        for i, owner in enumerate(owners)
    ]
    groups = aggregate_modules(branches, depth=depth)
    memberships = [set(g["event_indices"]) for g in groups]
    for left, right in combinations(memberships, 2):
        assert left.isdisjoint(right)
    assert set().union(*memberships) == set(range(len(branches)))
    for group, indices in zip(groups, memberships):
        assert group["launches"] == len(group["event_indices"]) == len(indices)
        assert group["gpu_time_us"] == sum(branches[i]["duration_us"] for i in indices)
        assert group["branches"] == [branches[i] for i in group["event_indices"]]
        assert group["start_us"] == min(branches[i]["start_us"] for i in indices)
        assert group["end_us"] == max(branches[i]["end_us"] for i in indices)
        assert group["span_us"] == group["end_us"] - group["start_us"]
        for i in indices:
            owner = owners[i]
            expected = ".".join(owner.split(".")[:depth]) if owner and depth else owner
            assert group["module_fqn"] == expected
    assert sum(g["launches"] for g in groups) == len(branches)
    # Even conflicting ownership must fail rather than put one event in two rows.
    with pytest.raises(ValueError, match="Duplicate device event"):
        aggregate_modules(branches + [{**branches[0], "module_fqn": "other"}], depth=depth)


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
        assert b["start_us"] == b["index"]
        assert b["end_us"] == b["index"] + 2.5
    assert branches[0]["branch"] == [4, 3, 2, 1, 0]
    for depth in [None, 1, 2, 3, 4]:
        groups = aggregate_modules(branches, depth=depth)
        ids = [i for g in groups for i in g["event_indices"]]
        assert sorted(ids) == [4, 5, 6, 7, 11]
        assert math.isclose(sum(g["gpu_time_us"] for g in groups), 12.5)
    rolled = {g["module_fqn"]: g for g in aggregate_modules(branches, depth=3)}
    assert rolled["model.layers.3"]["launches"] == 3
    all_attributed, all_unattributed = module_branches(rows, kernels_only=False)
    assert [b["index"] for b in all_attributed] == [4, 5, 6, 8]
    assert all_unattributed == unattributed
    assert module_branches([]) == ([], [])
    assert module_branches([rows[7]]) == ([], unattributed[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_modules(branches + branches[:1])


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
