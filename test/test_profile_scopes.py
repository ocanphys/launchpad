"""Exercise notebook instrumentation locally; never run the Modal capture."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from profile_events import aggregate_kernels, link_events, module_scopes
from torch import nn


def scope_path(event):
    labels = []
    while event is not None:
        if event.name.startswith(("forward::", "backward::")):
            labels.append(event.name)
        event = event.cpu_parent
    return labels


def assert_no_module_hooks(model):
    for module in model.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


class Block(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, x):
        return self.shared(x).relu()


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.Linear(3, 3)
        self.left = Block(shared)
        self.right = Block(shared)

    def forward(self, x):
        return self.left(x) + self.right(x)


def test_forward_backward_paths_shared_aliases_gradients_and_kernel_links():
    model = SharedModel()
    reference = deepcopy(model)
    x = torch.randn(2, 3, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_loss = reference(reference_x).square().sum()
    reference_loss.backward()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof, module_scopes(model):
        with torch.profiler.record_function("train::forward"):
            output = model(x)
        assert output.grad_fn.metadata["launchpad_module"]["module_fqn"] == "model"
        with torch.profiler.record_function("train::loss"):
            loss = output.square().sum()
        assert "launchpad_module" not in loss.grad_fn.metadata
        with torch.profiler.record_function("train::backward"):
            loss.backward()
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(x.grad, reference_x.grad)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert_no_module_hooks(model)

    events = list(prof.events())
    for phase, operation in [("forward", "aten::addmm"), ("backward", "aten::mm")]:
        paths = [scope_path(event) for event in events if event.name == operation]
        expected = [[f"{phase}::model.{side}.shared", f"{phase}::model.{side}", f"{phase}::model"]
                    for side in ("left", "right")]
        assert paths and all(path in expected for path in paths)
        assert {path[0] for path in paths} == {path[0] for path in expected}

    # Real CPU profiler ancestry + synthetic CUDA edges test the full row lookup
    # on machines without CUDA. They do not claim to test actual GPU capture.
    positions = {id(event): i for i, event in enumerate(events)}
    records = [{"name": event.name, "activity_type": event.activity_type,
                "device_type": event.device_type, "correlation_id": event.id,
                "linked_correlation_id": event.linked_correlation_id,
                "launcher_id": positions.get(id(event.cpu_parent))}
               for event in events]
    expected_kernel_scopes = {}
    for i, event in enumerate(events):
        if event.name not in {"aten::mm", "aten::addmm", "aten::sum"}:
            continue
        launch_id, kernel_id = len(records), len(records) + 1
        correlation = 1_000_000 + kernel_id
        records.extend([
            {"name": "cudaLaunchKernel", "activity_type": "cuda_runtime", "device_type": "DeviceType.CPU",
             "correlation_id": correlation, "linked_correlation_id": event.id, "launcher_id": i},
            {"name": "synthetic kernel", "activity_type": "kernel", "device_type": "DeviceType.CUDA",
             "correlation_id": correlation, "linked_correlation_id": event.id, "launcher_id": None},
        ])
        path = scope_path(event)
        expected_kernel_scopes[kernel_id] = (launch_id, path)
    raw = pd.DataFrame(records)
    rows = link_events(raw)
    assert rows.index.equals(raw.index)
    assert set(rows.columns) == set(raw.columns)
    assert any(not path for _, path in expected_kernel_scopes.values())
    for kernel_id, (launch_id, expected_path) in expected_kernel_scopes.items():
        assert rows.loc[kernel_id, "launcher_id"] == launch_id
        current, labels = kernel_id, []
        while pd.notna(current):
            name = rows.loc[current, "name"]
            if name.startswith(("forward::", "backward::")):
                labels.append(name)
            current = rows.loc[current, "launcher_id"]
        assert labels == expected_path
        if labels:
            phase = labels[0].split("::", 1)[0]
            assert labels[-1] == f"{phase}::model"

    intervals = [SimpleNamespace(start=float(i), end=i + 1.0) for i in range(len(rows))]
    groups, kernels, _, _ = aggregate_kernels(rows.assign(time_range=intervals))
    assert len(kernels) == len(expected_kernel_scopes)
    for kernel_id, (_, expected_path) in expected_kernel_scopes.items():
        assert (kernels.loc[kernel_id, "module_fqn"] == "<unattributed>") == (not expected_path)

    # A module-owned kernel whose launcher chain misses the root stage must raise.
    incomplete = rows.copy()
    incomplete.loc[incomplete["name"].eq("backward::model"), "name"] = "unlabelled"
    with pytest.raises(ValueError, match="not under the model stage"):
        aggregate_kernels(incomplete.assign(time_range=intervals))


def test_external_graph_accumulation_and_optimizer_are_unattributed():
    model = nn.Linear(3, 2)
    x = torch.randn(2, 3, requires_grad=True)
    external = x.sin()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof, module_scopes(model):
        y = model(external)
        assert "launchpad_module" not in external.grad_fn.metadata
        pending, seen = [y.grad_fn], set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.name() == "torch::autograd::AccumulateGrad":
                assert "launchpad_module" not in node.metadata
            pending.extend(parent for parent, _ in node.next_functions if parent is not None)
        y.sum().backward()
        with torch.profiler.record_function("train::optimizer_step"):
            optimizer.step()
    events = list(prof.events())
    cos = [event for event in events if event.name == "aten::cos"]
    assert cos and all(not scope_path(event) for event in cos)
    optimizer_ops = []
    for event in events:
        parent = event.cpu_parent
        while parent is not None and parent.name != "train::optimizer_step":
            parent = parent.cpu_parent
        if parent is not None:
            optimizer_ops.append(event)
    assert optimizer_ops and all(not scope_path(event) for event in optimizer_ops)


def test_hooks_removed_before_later_backward():
    model = nn.Linear(3, 2)
    with module_scopes(model):
        y = model(torch.randn(2, 3))
    assert_no_module_hooks(model)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        y.sum().backward()
    assert not any(event.name.startswith("backward::") for event in prof.events())


def test_retained_graph_reopens_scopes():
    model = nn.Linear(3, 2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof, module_scopes(model):
        y = model(torch.randn(2, 3))
        y.sum().backward(retain_graph=True)
        y.sum().backward()
    backward = [event for event in prof.events() if event.name == "backward::model"]
    assert len(backward) == 4  # AddmmBackward and TBackward on both traversals.
    assert_no_module_hooks(model)


def test_forward_exception_closes_scopes_and_removes_hooks():
    class Broken(nn.Module):
        def forward(self, x):
            x.relu()
            raise RuntimeError("forward failed")

    model = nn.Sequential(Broken())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with pytest.raises(RuntimeError, match="forward failed"), module_scopes(model):
            model(torch.ones(1))
        with torch.profiler.record_function("after_failure"):
            torch.ones(1).cos()
    assert_no_module_hooks(model)
    assert all(not scope_path(event) for event in prof.events() if event.name in {"after_failure", "aten::cos"})


def test_backward_exception_closes_open_node_scopes_and_removes_hooks():
    model = nn.Linear(3, 2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with pytest.raises(RuntimeError, match="backward failed"), module_scopes(model):
            y = model(torch.randn(2, 3))

            def fail(grad_outputs):
                raise RuntimeError("backward failed")

            # Registered after instrumentation: fail after our scope pre-hook,
            # preventing the scope post-hook from running.
            handle = y.grad_fn.register_prehook(fail)
            try:
                y.sum().backward()
            finally:
                handle.remove()
        with torch.profiler.record_function("after_failure"):
            torch.ones(1).cos()
    assert_no_module_hooks(model)
    assert any(event.name == "backward::model" for event in prof.events())
    assert all(not scope_path(event) for event in prof.events() if event.name in {"after_failure", "aten::cos"})
