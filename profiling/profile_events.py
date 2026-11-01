"""Module scopes for a profiled training step, its capture as JSON, and the tables built from it.

The capture context managers turn what torch hands back into plain JSON before it leaves the
container: `event_table` the profiler's events, `capture_memory` the allocator snapshot. Every
table after that is built from that JSON alone, on any machine. The profile container imports
this module for `module_scopes` and the capture context managers, so pandas and matplotlib,
which the worker image lacks, are imported where they are used.
"""

from __future__ import annotations

import math
import random
from contextlib import ExitStack, contextmanager
from threading import local
from typing import NamedTuple

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils._pytree import tree_leaves, tree_map

PHASE_ORDER = {p: i for i, p in enumerate(
    ["zero_grad", "forward", "loss", "backward", "clip_grad", "optimizer_step", "unscoped"])}


@contextmanager
def module_scopes(model, root="model"):
    """Yields while every eager module call, and the backward of the work it did, is recorded as a module stage.

        with module_scopes(network):
            run_step(tokens)

    A stage is a record_function annotation named <phase>::<fqn>: forward::model.tblocks.0
    around the module's forward call, and backward::model, ..., backward::model.tblocks.0 around
    each autograd node that call created. Keep the context open for forward and backward both;
    on exit every hook is removed and every open stage closed, even after an exception.
    docs/profiling/attribution.md says what is claimed, and how kernels are attributed to stages.

    Args:
        model: The network to instrument; no tensor or gradient is replaced.
        root: The FQN of the network's own stages; a submodule's is root.<named_modules path>.
    """
    # Discover module identities and their fully qualified aliases.
    aliases = {}
    for name, module in model.named_modules(remove_duplicate=False):
        aliases.setdefault(module, []).append(f"{root}.{name}" if name else root)

    state = local()
    handles, owned_nodes, open_scopes = [], set(), {}

    # Emit stages from outermost module to owner; ExitStack closes in reverse.
    def open_path(phase, path):
        scopes = ExitStack()
        open_scopes[scopes] = None
        for fqn in path:
            scopes.enter_context(record_function(f"{phase}::{fqn}"))
        return scopes

    def close_path(scopes):
        try:
            scopes.close()
        finally:
            open_scopes.pop(scopes, None)

    # Restore backward stages on the thread executing each owned autograd node.
    def attach_node(node, path):
        # Retain node objects: Python object IDs may be reused between operations.
        owned_nodes.add(node)
        node.metadata["launchpad_module"] = {
            "module_fqn": path[-1], "module_path": list(path),
        }
        running = local()

        def before(grad_outputs):
            if not hasattr(running, "stack"):
                running.stack = []
            running.stack.append(open_path("backward", path))

        def after(grad_inputs, grad_outputs):
            close_path(running.stack.pop())

        handles.append(node.register_prehook(before))
        handles.append(node.register_hook(after))

    def grad_nodes(value):
        nodes = set()
        for tensor in tree_leaves(value):
            if isinstance(tensor, torch.Tensor) and (node := tensor.grad_fn) is not None:
                nodes.add(node)
        return nodes

    # During forward, assign new nodes once to the active module path.
    class CaptureAutogradNodes(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            stack = getattr(state, "stack", [])
            if not stack:
                return func(*args, **kwargs)

            path = state.path  # Cached on module entry/exit, not rebuilt per operation.
            boundary = grad_nodes((args, kwargs))
            output = func(*args, **kwargs)
            pending = list(grad_nodes(output))
            while pending:
                node = pending.pop()
                if node in boundary or node in owned_nodes:
                    continue
                if node.name() == "torch::autograd::AccumulateGrad":
                    continue
                attach_node(node, path)
                pending.extend(parent for parent, _ in node.next_functions if parent is not None)
            return output

    # Forward hooks open/close one stage per module invocation.
    def enter(module, args):
        if not hasattr(state, "stack"):
            state.stack = []
        paths = aliases[module]
        caller = state.stack[-1][1] if state.stack else None
        candidates = [path for path in paths if caller and path.startswith(caller + ".")]
        fqn = (candidates or paths)[0]
        state.stack.append((module, fqn, open_path("forward", [fqn])))
        state.path = getattr(state, "path", ()) + (fqn,)

    def leave(module, args, output):
        stack = getattr(state, "stack", [])
        if stack and stack[-1][0] is module:
            close_path(stack.pop()[2])
            state.path = state.path[:-1]

    # Install instrumentation for this context only; clean up even on failure.
    try:
        for module in aliases:
            handles.append(module.register_forward_pre_hook(enter, prepend=True))
            handles.append(module.register_forward_hook(leave, always_call=True))
        with CaptureAutogradNodes():
            yield
    finally:
        for handle in handles:
            handle.remove()
        # An autograd exception may skip a node's post-hook.
        for scopes in reversed(list(open_scopes)):
            close_path(scopes)
        owned_nodes.clear()


def event_table(function_events):
    """Returns profiler events as JSON columns: one list per FunctionEvent attribute, in the
    events' order, so an event's row ID is its position.

        event_table(prof.events())["name"]  # a CPU profile of torch.ones(2) + 1
        -> ['aten::ones', 'aten::empty', 'aten::fill_', 'aten::add', 'aten::to', 'aten::_to_copy',
            'aten::empty_strided', 'aten::copy_']

    Four attributes hold objects JSON cannot: time_range becomes start_us and end_us, device_type
    its str ("DeviceType.CPU"), cpu_parent the parent's position or None, and cpu_children and
    kernels, the inverses of links the rows keep, are left out. Any other value JSON cannot hold,
    such as a symbolic size torch.compile recorded or an infinite fill value, is written as its str.

    Args:
        function_events: One step's events, as `prof.events()` returns them.

    Returns:
        Maps each attribute to its column, in torch's order; empty without events.
    """
    position = {id(event): i for i, event in enumerate(function_events)}

    def plain(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        return value if value is None or isinstance(value, bool | int | str) else str(value)

    def columns(event):
        for name, value in vars(event).items():
            if name == "time_range":
                yield "start_us", value.start
                yield "end_us", value.end
            elif name == "device_type":
                yield name, str(value)
            elif name == "cpu_parent":
                yield name, position.get(id(value))
            elif name not in ("cpu_children", "kernels"):
                yield name, tree_map(plain, value)

    records = [dict(columns(event)) for event in function_events]
    return {name: [record[name] for record in records] for name in records[0]} if records else {}


@contextmanager
def capture_profiler_events(enabled):
    """Yields a dict that holds the block's profiler events once it exits; empty when not enabled.

    It synchronizes before the profiler stops, so every kernel launched inside the block is recorded.

    Args:
        enabled: Record the block's CPU ops and CUDA kernels, with their FLOPs, under torch.profiler.

    Yields:
        The dict: "trace_start_unix_us", the trace's start on the unix clock, which the events'
        times count from, and "events", event_table's columns.
    """
    profiled = {}
    if not enabled:
        yield profiled
        return
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        yield profiled
        torch.cuda.synchronize()
    profiled["trace_start_unix_us"] = prof.profiler.kineto_results.trace_start_ns() / 1e3
    profiled["events"] = event_table(prof.events())


@contextmanager
def capture_memory(peak=False, trace=False, max_entries=100_000):
    """Yields a dict that holds the block's memory readings once it exits; empty when both are off.

    The allocator records no tracebacks, so the snapshot is plain JSON and the block is not
    slowed by stack capture; which scope allocated a block is still in its annotations.

    Args:
        peak: Add peak_allocated_bytes and peak_reserved_bytes.
        trace: Add "snapshot", the allocator snapshot of just this block, as torch returns it.
        max_entries: The most trace entries the allocator keeps; past it, the earliest are dropped.

    Yields:
        The dict.
    """
    memory = {}
    if peak:
        torch.cuda.reset_peak_memory_stats()
    if trace:
        torch.cuda.memory._record_memory_history(context=None, max_entries=max_entries, clear_history=True)
    try:
        yield memory
        if trace:
            memory["snapshot"] = torch.cuda.memory._snapshot()
    finally:
        if trace:
            torch.cuda.memory._record_memory_history(enabled=None)
    if peak:  # allocator statistics are kept on the CPU, so no synchronize is needed
        memory["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        memory["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()


def _forest_order(parents, cpu_ids):
    """Returns each event's children and every event in parent-first order.

    It raises ValueError unless every launcher is a CPU event in parents and no link closes a cycle.

    Args:
        parents: Maps each row ID to its launcher's row ID, or None for a root.
        cpu_ids: The row IDs of CPU events, the only events that launch others.

    Returns:
        (children, order): children maps each row ID to the row IDs it launched, in row order;
        order lists every row ID after its launcher.
    """
    children = {index: [] for index in parents}
    roots = []
    for index, parent in parents.items():
        if parent is None:
            roots.append(index)
        elif parent not in cpu_ids:
            raise ValueError(f"Event {index} refers to missing or non-CPU launcher {parent}")
        else:
            children[parent].append(index)

    order, pending = [], list(reversed(roots))
    while pending:
        index = pending.pop()
        order.append(index)
        pending.extend(reversed(children[index]))
    # With at most one parent per node, a cycle cannot be reached from a root.
    if len(order) != len(parents):
        raise ValueError("Launcher links contain a cycle")
    return children, order


def _correlation_index(events):
    """Returns the CPU events by (node_id, correlation_id), one lookup per id namespace.

    PyTorch ids (ops and scopes) and CUPTI ids (launch calls and GPU work) come from two counters
    that overlap, so each counter has its own lookup and an id is only compared with ids of its
    own kind. An id two events in one lookup share maps to None, never to either.

    Args:
        events: link_events' itertuples rows, with is_cpu and node_id.

    Returns:
        (ops, launches): ops maps the PyTorch ids of cpu_op and user_annotation rows, launches
        those of cuda_runtime and cuda_driver rows, each to a row ID or None.
    """
    import pandas as pd

    ops, launches = {}, {}
    for event in events:
        if not event.is_cpu or pd.isna(event.correlation_id) or event.correlation_id == 0:
            continue
        if event.activity_type in {"cpu_op", "user_annotation"}:
            lookup = ops
        elif event.activity_type in {"cuda_runtime", "cuda_driver"}:
            lookup = launches
        else:
            continue
        key = (event.node_id, event.correlation_id)
        lookup[key] = event.Index if key not in lookup else None
    return ops, launches


def link_events(frame):
    """Returns frame with every GPU event's launcher_id set to the CPU event that launched it.

    CPU rows keep the launcher_id they come with. Kernels, copies and memsets link to the launch
    call whose CUPTI id equals theirs; failing that, any GPU row links to the op or scope whose
    PyTorch id its linked_correlation_id holds. A GPU row without exactly
    one match stays a root (launcher_id NA): links are never guessed from time.
    It raises ValueError unless row IDs are unique and the links form a forest of CPU launchers.

    Args:
        frame: One row per profiler event, with unique row IDs; its CPU rows' launcher_id is
            already their cpu_parent's row ID.

    Returns:
        A copy of frame, every row, column and the order kept, with launcher_id (nullable
        Int64) set for its GPU rows.
    """
    import pandas as pd

    if not frame.index.is_unique:
        raise ValueError("Event row indices must be unique")

    # Only read linking fields, leaving large metadata (stacks, shapes) untouched.
    columns = ["activity_type", "correlation_id", "linked_correlation_id", "launcher_id"]
    events = list(frame[columns].assign(
        is_cpu=frame["device_type"].map(str).eq("DeviceType.CPU"),
        node_id=frame.get("node_id", pd.Series(-1, index=frame.index)).fillna(-1),
    ).itertuples())
    ops, launches = _correlation_index(events)

    parents = {}
    for event in events:
        if event.is_cpu:  # PyTorch's cpu_parent, already a row ID
            parents[event.Index] = int(event.launcher_id) if pd.notna(event.launcher_id) else None
            continue
        launcher = None
        # Annotation spans must not match unrelated CUDA launch IDs.
        if event.activity_type in {"kernel", "gpu_memcpy", "gpu_memset"} and pd.notna(event.correlation_id):
            launcher = launches.get((event.node_id, event.correlation_id))
        if launcher is None and pd.notna(event.linked_correlation_id):
            launcher = ops.get((event.node_id, event.linked_correlation_id))
        parents[event.Index] = launcher

    _forest_order(parents, {event.Index for event in events if event.is_cpu})
    return frame.assign(launcher_id=pd.array(list(parents.values()), dtype="Int64"))


def events_to_rows(table):
    """Returns one linked row per profiler event, indexed by its row ID, its position in the step's events.

        rows = events_to_rows(events[0]["events"])  # events.json's step 0
        rows.loc[532, "launcher_id"] -> 531

    Args:
        table: One step's events as event_table's columns.

    Returns:
        The event table: one row per event with its attributes, `id` renamed correlation_id and
        `cpu_parent` renamed launcher_id, which for a GPU event is the row ID of the CPU event
        link_events finds.
    """
    import pandas as pd

    if not table:
        raise ValueError("no events captured: profile with kernel_events=True")
    frame = pd.DataFrame(table).rename(columns={"id": "correlation_id", "cpu_parent": "launcher_id"})
    return link_events(frame.astype({"launcher_id": "Int64"}).rename_axis("index"))


LAUNCH_CHAIN_COLUMNS = [
    "name", "activity_type", "device_type",  # what the event is
    "start_us", "end_us",  # timing
    "launcher_id", "correlation_id", "linked_correlation_id",  # relations
    "flops",
    "input_shapes", "input_dtypes",  # shapes
]


def launch_chain(rows, kernel_id, format=True):
    """Returns the rows on kernel_id's launch sequence, from its CPU root down to the kernel.

        launch_chain(rows, 532).index.tolist() -> [5, 7, 16, 501, 502, 510, 529, 531, 532]

    Args:
        rows: The table from events_to_rows.
        kernel_id: The row ID of the event to start from; one with no launcher gives just its row.
        format: Keep only LAUNCH_CHAIN_COLUMNS; False keeps every column.

    Returns:
        The rows of the chain, the root first.
    """
    import pandas as pd

    chain, current = [], kernel_id
    while pd.notna(current):
        chain.append(current)
        current = rows.loc[current, "launcher_id"]
    chain = rows.loc[chain[::-1]]
    return chain[LAUNCH_CHAIN_COLUMNS] if format else chain


class _LaunchForest(NamedTuple):
    """The events of a linked table, as the forest their launcher_id links make."""

    events: dict  # row ID: its itertuples row (name, device_type, activity_type, launcher_id, flops)
    parents: dict  # row ID: its launcher's row ID, or None for a root
    children: dict  # row ID: the row IDs it launched, in row order
    order: list  # every row ID, each after its launcher
    cpu: set  # the row IDs of CPU events
    kernels: list  # the row IDs of kernels, in row order


def _launch_forest(rows):
    """Returns the launch forest of a linked event table, checked to be one.

    Args:
        rows: The table from events_to_rows or link_events.

    Returns:
        The table's _LaunchForest.
    """
    import pandas as pd

    columns = ["name", "device_type", "activity_type", "launcher_id", "flops"]
    events = {event.Index: event for event in rows.reindex(columns=columns).itertuples()}
    parents = {index: int(event.launcher_id) if pd.notna(event.launcher_id) else None
               for index, event in events.items()}
    cpu = {index for index, event in events.items() if str(event.device_type) == "DeviceType.CPU"}
    children, order = _forest_order(parents, cpu)
    kernels = [index for index, event in events.items() if event.activity_type == "kernel"]
    return _LaunchForest(events, parents, children, order, cpu, kernels)


def _execution_context(event, inherited, root):
    """Returns the (phase, owner FQN, owner stage row ID) a CPU event passes to what it launches.

    Args:
        event: A CPU event's row from the launch forest.
        inherited: The context its launcher passed down.
        root: The FQN module_scopes gave the model.

    Returns:
        A train::<phase> annotation's phase with no owner; a <phase>::<fqn> module stage's phase,
        FQN and row ID; optimizer_step outside any module for an Optimizer.step# annotation;
        backward for an autograd node, keeping the owner only inside backward; else inherited.
    """
    phase, owner, stage_id = inherited
    name = event.name
    if event.activity_type == "user_annotation":
        if name.startswith("train::"):
            return name.removeprefix("train::"), None, None
        label_phase, separator, fqn = name.partition("::")
        if separator and (fqn == root or fqn.startswith(root + ".")):
            return label_phase, fqn, event.Index
        if name.startswith("Optimizer.step#") and owner is None:
            return "optimizer_step", None, None
    if name.startswith("autograd::engine::evaluate_function:"):
        return "backward", owner if phase == "backward" else None, stage_id if phase == "backward" else None
    return inherited


def _kernel_owners(forest, root):
    """Returns each kernel's (phase, owner FQN, owner stage row ID), passed down its launch chain.

    The owner is the nearest module stage above the kernel, None without one. A kernel whose
    owner's chain never reaches the root stage raises ValueError: its module path is incomplete.

    Args:
        forest: The table's _LaunchForest.
        root: The FQN module_scopes gave the model.

    Returns:
        Maps each kernel's row ID, in row order, to its context.
    """
    contexts, rooted = {}, {}
    for index in forest.order:
        event, parent = forest.events[index], forest.parents[index]
        context = contexts.get(parent, ("unscoped", None, None))
        contexts[index] = _execution_context(event, context, root) if index in forest.cpu else context
        rooted[index] = rooted.get(parent, False) or (
            index in forest.cpu and event.activity_type == "user_annotation"
            and event.name.partition("::")[2] == root)
    owners = {index: contexts[index] for index in forest.kernels}
    unrooted = [index for index in forest.order if index in owners and owners[index][1] is not None and not rooted[index]]
    if unrooted:
        raise ValueError(f"Kernels owned by a module stage but not under the {root} stage: {unrooted}")
    return owners


def _module_group(owner, depth, root):
    """Returns the module group of a kernel with this owner: the owner's first depth FQN components.

        _module_group("model.tblocks.0.ffn", 3, "model") -> "model.tblocks.0"

    Args:
        owner: The kernel's owner FQN, or None for a kernel no module stage owns.
        depth: How many FQN components to keep, root included; None keeps them all.
        root: The FQN module_scopes gave the model, one component however many dots it has.

    Returns:
        The group's FQN, or <unattributed> without an owner.
    """
    if owner is None:
        return "<unattributed>"
    if depth is None:
        return owner
    components = [root] + owner[len(root) + 1:].split(".") if owner != root else [root]
    return ".".join(components[:depth])


def _reported_flops(forest):
    """Returns the FLOP estimates the profiler reported and the outermost reported ancestor of each.

    An estimate is reported when a CPU aten:: event has a positive finite flops value.

    Args:
        forest: The table's _LaunchForest.

    Returns:
        (reported, suppressed_by): reported maps each reporting row ID to its FLOPs, in
        parent-first order; suppressed_by maps the same IDs to the outermost reporting ATen
        ancestor's row ID, None for an estimate no reporting ancestor encloses.
    """
    import pandas as pd

    reported, suppressed_by, outermost = {}, {}, {}
    for index in forest.order:
        event, ancestor = forest.events[index], outermost.get(forest.parents[index])
        if (index in forest.cpu and event.name.startswith("aten::")
                and pd.notna(event.flops) and event.flops > 0 and math.isfinite(event.flops)):
            reported[index] = float(event.flops)
            suppressed_by[index] = ancestor
            if ancestor is None:
                ancestor = index
        outermost[index] = ancestor
    return reported, suppressed_by


_MIXED = object()  # the destination of a subtree whose kernels fall into more than one group


def _subtree_summaries(forest, group_keys, reported):
    """Returns, for every event, the kernels below it, the group they fall into and the FLOPs it carries.

    A subtree carries its reported estimate if it has one, else the sum of what its children
    carry; a subtree without kernels carries none.

    Args:
        forest: The table's _LaunchForest.
        group_keys: Maps each kernel's row ID to its (phase, module_fqn) group.
        reported: Maps each reporting ATen row ID to its FLOPs.

    Returns:
        (kernel_counts, destinations, effective_flops), each keyed by row ID: the kernels in the
        subtree; their one group key, None without kernels or _MIXED for several groups; and the
        FLOPs the subtree carries.
    """
    kernel_counts, destinations, effective_flops = {}, {}, {}
    for index in reversed(forest.order):
        destination, count, child_flops = group_keys.get(index), int(index in group_keys), 0.0
        for child in forest.children[index]:
            count += kernel_counts[child]
            child_flops += effective_flops[child]
            if destinations[child] is None:
                continue
            if destination is None:
                destination = destinations[child]
            elif destination != destinations[child]:
                destination = _MIXED
        kernel_counts[index], destinations[index] = count, destination
        effective_flops[index] = reported.get(index, child_flops) if count else 0.0
    return kernel_counts, destinations, effective_flops


def _flop_sources(forest, group_keys):
    """Returns one row per reported FLOP estimate with kernels below it, and the group it counts toward.

    An estimate is suppressed when an outer reported estimate encloses it, unallocated when its
    kernels fall into several groups, and otherwise allocated to its kernels' one group. It raises
    ValueError unless allocated plus unallocated FLOPs equal what the forest's roots carry.

    Args:
        forest: The table's _LaunchForest.
        group_keys: Maps each kernel's row ID to its (phase, module_fqn) group.

    Returns:
        A DataFrame indexed by the estimate's row ID (aten_id), in parent-first order, with name,
        reported_flops, contribution_flops (reported_flops when allocated, else 0), status,
        suppressed_by, spans_groups, kernel_count, and the phase and module_fqn it is allocated to.
    """
    import pandas as pd

    reported, suppressed_by = _reported_flops(forest)
    kernel_counts, destinations, effective_flops = _subtree_summaries(forest, group_keys, reported)
    records = []
    for index, amount in reported.items():
        if not kernel_counts[index]:
            continue
        blocker, spans_groups = suppressed_by[index], destinations[index] is _MIXED
        status = "suppressed" if blocker is not None else "unallocated" if spans_groups else "allocated"
        phase, fqn = (None, None) if spans_groups else destinations[index]
        records.append((index, forest.events[index].name, amount, amount if status == "allocated" else 0.0,
                        status, blocker, spans_groups, kernel_counts[index], phase, fqn))
    sources = pd.DataFrame(records, columns=[
        "aten_id", "name", "reported_flops", "contribution_flops", "status", "suppressed_by",
        "spans_groups", "kernel_count", "phase", "module_fqn",
    ]).set_index("aten_id")
    sources["suppressed_by"] = pd.array(sources["suppressed_by"], dtype="Int64")
    sources["spans_groups"] = sources["spans_groups"].astype(bool)

    counted = sources["contribution_flops"].sum() + sources.loc[sources["status"].eq("unallocated"), "reported_flops"].sum()
    if not math.isclose(counted, sum(effective_flops[index] for index in forest.order if forest.parents[index] is None)):
        raise ValueError("FLOP sources must conserve the bottom-up estimate")
    return sources


def _kernel_table(rows, owners, group_keys):
    """Returns the kernel rows of rows, in row order, with their owner, group and GPU interval.

    It raises ValueError for a kernel whose start_us and end_us are not a finite interval.

    Args:
        rows: The linked event table.
        owners: Maps each kernel's row ID, in row order, to its (phase, owner FQN, owner stage row ID).
        group_keys: Maps each kernel's row ID to its (phase, module_fqn) group.

    Returns:
        The kernel rows with phase, module_fqn, owner_fqn, owner_stage_id, gpu_start_us,
        gpu_end_us and duration_us.
    """
    import pandas as pd

    kernel_ids = list(owners)
    kernels = rows.loc[kernel_ids].copy()
    starts, ends = [], []
    for index, start, end in zip(kernels.index, kernels["start_us"], kernels["end_us"]):
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Invalid GPU interval for kernel {index}")
        starts.append(start)
        ends.append(end)
    kernels["phase"] = [group_keys[index][0] for index in kernel_ids]
    kernels["module_fqn"] = [group_keys[index][1] for index in kernel_ids]
    kernels["owner_fqn"] = [owners[index][1] for index in kernel_ids]
    kernels["owner_stage_id"] = pd.array([owners[index][2] for index in kernel_ids], dtype="Int64")
    kernels["gpu_start_us"], kernels["gpu_end_us"] = starts, ends
    kernels["duration_us"] = kernels["gpu_end_us"] - kernels["gpu_start_us"]
    return kernels


def _phase_rank(level):
    """Returns a sort key that orders a phase index level or column by PHASE_ORDER, any other as it is.

    Args:
        level: An index level or column, as sort_index and sort_values pass it.

    Returns:
        The phases' ranks, or level unchanged.
    """
    return level.map(PHASE_ORDER) if level.name == "phase" else level


def _group_table(kernels, flop_sources):
    """Returns one row per (phase, module_fqn) group of kernels, in phase execution order.

    Args:
        kernels: The kernel table, in row order.
        flop_sources: The FLOP estimates and the groups they are allocated to.

    Returns:
        A DataFrame indexed by (phase, module_fqn) with kernel_ids, owner_stage_ids, kernel_count,
        total_est_flops (the sum of the group's allocated estimates), one
        <op>_est_flops column per ATen op with an allocated estimate, in name order
        (aten::bmm's is bmm_est_flops), flop_source_ids, gpu_start_us, gpu_end_us,
        gpu_span_us (end minus start, gaps included) and gpu_time_us (the sum of kernel durations).
    """
    import pandas as pd

    allocated = flop_sources[flop_sources["status"].eq("allocated")]
    columns = allocated["name"].str.removeprefix("aten::") + "_est_flops"
    breakdown = sorted(set(columns))
    group_flops, group_sources = {}, {}
    for index, phase, fqn, column, amount in zip(allocated.index, allocated["phase"], allocated["module_fqn"],
                                                 columns, allocated["contribution_flops"]):
        group_flops.setdefault((phase, fqn), dict.fromkeys(breakdown, 0.0))[column] += amount
        group_sources.setdefault((phase, fqn), []).append(index)

    summaries = []
    for key, members in kernels.groupby(["phase", "module_fqn"], sort=False, dropna=False):
        start, end = members["gpu_start_us"].min(), members["gpu_end_us"].max()
        flops = group_flops.get(key, dict.fromkeys(breakdown, 0.0))
        summaries.append((*key, members.index.tolist(), members["owner_stage_id"].dropna().unique().tolist(),
                          len(members), sum(flops.values(), 0.0), *flops.values(), group_sources.get(key, []),
                          start, end, end - start, members["duration_us"].sum()))
    return pd.DataFrame(summaries, columns=[
        "phase", "module_fqn", "kernel_ids", "owner_stage_ids", "kernel_count", "total_est_flops", *breakdown,
        "flop_source_ids", "gpu_start_us", "gpu_end_us", "gpu_span_us", "gpu_time_us",
    ]).set_index(["phase", "module_fqn"]).sort_index(key=_phase_rank)


def aggregate_kernels(rows, depth=None, *, root="model"):
    """Returns the kernels of a linked event table grouped by phase and module, with the FLOPs of each group.

        groups, kernels, unattributed_kernels, flop_sources = aggregate_kernels(rows, depth=3)
        groups.loc[("forward", "model.embedding"), ["kernel_ids", "total_est_flops"]].tolist() -> [[14], 0.0]

    Every kernel is in exactly one group and every reported FLOP estimate counts toward at most
    one; docs/profiling/attribution.md shows how. Copies and memsets are not kernels here.

    Args:
        rows: The table from events_to_rows, which is not changed.
        depth: How many FQN components a group keeps, root included: 1 groups everything under
            model, 3 under model.tblocks.0; None keeps each kernel's full owner FQN.
        root: The FQN module_scopes gave the model.

    Returns:
        (groups, kernels, unattributed_kernels, flop_sources):

        - groups: one row per (phase, module_fqn) in phase execution order, with kernel_ids,
          owner_stage_ids, kernel_count, total_est_flops and its <op>_est_flops
          breakdown, flop_source_ids, gpu_start_us, gpu_end_us, gpu_span_us and gpu_time_us.
        - kernels: the kernel rows, sorted by phase, module_fqn and gpu_start_us, with phase,
          module_fqn, owner_fqn, owner_stage_id, gpu_start_us, gpu_end_us and duration_us.
        - unattributed_kernels: the rows of kernels no module stage owns.
        - flop_sources: one row per reported ATen estimate with kernels below it, indexed by its
          row ID, with its status: allocated, suppressed or unallocated.
    """
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
        raise ValueError("depth must be a positive integer or None")
    if not isinstance(root, str) or not root:
        raise ValueError("root must be a nonempty FQN")
    if not rows.index.is_unique:
        raise ValueError("Event row indices must be unique")

    forest = _launch_forest(rows)
    owners = _kernel_owners(forest, root)
    group_keys = {index: (phase, _module_group(owner, depth, root)) for index, (phase, owner, _) in owners.items()}
    flop_sources = _flop_sources(forest, group_keys)
    kernels = _kernel_table(rows, owners, group_keys)
    groups = _group_table(kernels, flop_sources)
    kernels = kernels.sort_values(["phase", "module_fqn", "gpu_start_us"], key=_phase_rank)
    return groups, kernels, kernels[kernels["module_fqn"].eq("<unattributed>")], flop_sources


def allocations_by_label(memory_snapshot, device=0):
    """Return one row per allocation in the allocator trace, labelled with its innermost scope.

    Columns are addr, size, start_us, end_us and label. start_us is the alloc entry's time_us and
    end_us is the time_us of the free_requested entry for the same addr, NaN if the trace never
    frees it. label is the name of the most recently started record_function annotation still
    open at start_us, or <none>. Annotations carry no thread, so concurrent scopes on different
    threads resolve to whichever started last.
    """
    import pandas as pd

    annotations = sorted((a for a in memory_snapshot["external_annotations"] if a["device"] == device),
                         key=lambda a: a["time_us"])
    open_scopes, allocations, live, i = [], [], {}, 0
    # The trace is a ring buffer, so sort it by time before replaying the annotations.
    for entry in sorted(memory_snapshot["device_traces"][device], key=lambda e: e["time_us"]):
        while i < len(annotations) and annotations[i]["time_us"] <= entry["time_us"]:
            annotation, i = annotations[i], i + 1
            if annotation["stage"] == "START":
                open_scopes.append(annotation["name"])
            elif annotation["name"] in open_scopes:
                del open_scopes[len(open_scopes) - 1 - open_scopes[::-1].index(annotation["name"])]
        if entry["action"] == "alloc":
            live[entry["addr"]] = {"addr": entry["addr"], "size": entry["size"], "start_us": entry["time_us"],
                                   "end_us": float("nan"), "label": open_scopes[-1] if open_scopes else "<none>"}
            allocations.append(live[entry["addr"]])
        elif entry["action"] == "free_requested" and entry["addr"] in live:
            live.pop(entry["addr"])["end_us"] = entry["time_us"]
    return pd.DataFrame(allocations, columns=["addr", "size", "start_us", "end_us", "label"])


def memory_over_time(memory_snapshot):
    """Return {device: DataFrame of allocated and reserved bytes after each allocator trace entry, indexed by time_us}.

    allocated changes on alloc and free_requested entries; reserved changes on segment_alloc and
    segment_free. The starting values are the device's final totals in the snapshot minus the
    trace's net change, so they are exact only if the trace ring buffer did not wrap.
    """
    import pandas as pd

    deltas = {"alloc": (1, 0), "free_requested": (-1, 0), "segment_alloc": (0, 1), "segment_free": (0, -1)}
    usage = {}
    for device, entries in enumerate(memory_snapshot["device_traces"]):
        trace = sorted((e for e in entries if e["action"] in deltas), key=lambda e: e["time_us"])
        steps = pd.DataFrame({
            "time_us": [e["time_us"] for e in trace],
            "allocated": [deltas[e["action"]][0] * e["size"] for e in trace],
            "reserved": [deltas[e["action"]][1] * e["size"] for e in trace],
        }).set_index("time_us")
        segments = [s for s in memory_snapshot["segments"] if s["device"] == device]
        final = pd.Series({"allocated": sum(s["allocated_size"] for s in segments),
                           "reserved": sum(s["total_size"] for s in segments)})
        usage[device] = steps.cumsum() + (final - steps.sum())
    return usage


def dots_height(phase_count):
    """Returns the height in inches of the kernel duration dots under the kernel timeline: a lane per phase.

    Args:
        phase_count: The phases with kernels.

    Returns:
        The height in inches.
    """
    return 0.8 * (0.5 + 0.3 * phase_count)


def _kernel_panel_height(group_count):
    """Returns the height in inches of the kernel timeline: a row per group.

    Args:
        group_count: The groups plotted.

    Returns:
        The height in inches.
    """
    return max(3, 0.3 * group_count + 1.2)


def timeline_height(groups=None, memory=None, phases=None, peaks=None):
    """Returns the height in inches draw_timeline wants for these groups, memory, phases and peaks.

    Args:
        groups: The groups draw_timeline plots, or None.
        memory: The memory over time, or None.
        phases: The phase spans, or None.
        peaks: The peak memory, or None.

    Returns:
        The height in inches; 0 when there is nothing to draw.
    """
    has_kernels = groups is not None and not groups.empty
    kernels_height = (_kernel_panel_height(len(groups))
                      + dots_height(groups.index.get_level_values("phase").nunique())) if has_kernels else 0
    height = kernels_height + 3 * (memory is not None or peaks is not None)
    return height or 1.5 * (phases is not None)  # phases alone get a strip of their own


def _phase_colors(phases, seed):
    """Returns a color for each phase: unscoped gray, the others from a palette in a shuffled order.

    Args:
        phases: The phase names drawn, repeats allowed.
        seed: Seeds the shuffle, so the same phases get the same colors.

    Returns:
        Maps each phase, and unscoped, to a color.
    """
    shuffled = ["#15803d", "#c75a0a", "#1d4ed8", "#7c3aed", "#0f766e", "#b91c1c", "#be185d", "#a16207"]
    random.Random(seed).shuffle(shuffled)
    colors = {phase: shuffled[i % len(shuffled)]
              for i, phase in enumerate(sorted(phase for phase in set(phases) if phase != "unscoped"))}
    colors["unscoped"] = "#6b7280"
    return colors


def _plotted_groups(groups, kernels):
    """Returns the groups in plot order, earliest GPU start first, and the kernels they plot.

    It raises ValueError for a repeated group or kernel index, or a kernel in two groups.

    Args:
        groups: The groups to plot.
        kernels: The kernel table their kernel_ids index.

    Returns:
        (selected, plotted): the groups sorted by gpu_start_us, and the rows of their kernels.
    """
    if not groups.index.is_unique or not kernels.index.is_unique:
        raise ValueError("Group and kernel indices must be unique")
    selected = groups.sort_values("gpu_start_us", kind="stable")
    kernel_ids = [index for ids in selected["kernel_ids"] for index in ids]
    if len(kernel_ids) != len(set(kernel_ids)):
        raise ValueError("A kernel cannot appear in more than one plotted group")
    return selected, kernels.loc[kernel_ids]


def _time_extent(selected, memory, phases):
    """Returns the earliest and latest time drawn, infinite when nothing drawn has a time.

    Args:
        selected: The plotted groups, or None.
        memory: The memory over time, or None.
        phases: The phase spans, or None.

    Returns:
        (left, right) in us.
    """
    left, right = math.inf, -math.inf
    if selected is not None:
        left, right = selected["gpu_start_us"].min(), selected["gpu_end_us"].max()
    if memory is not None:
        left, right = min(left, memory.index.min()), max(right, memory.index.max())
    if phases is not None:
        for phase in phases.itertuples():
            left, right = min(left, phase.start_us), max(right, phase.end_us)
    return left, right


def _heading(title, info):
    """Returns title with the info lines joined on one line under it.

    Args:
        title: The panel's title.
        info: Text lines, or none.

    Returns:
        The heading text.
    """
    return "\n".join([title, " | ".join(info)] if info else [title])


def _draw_kernel_rows(ax, selected, kernels, kernel_phases, colors, heading):
    """Draws a row per group into ax: its GPU span in gray and every one of its kernels in its phase's color.

    Rectangles are batched per group, one collection each, without merging any kernels.

    Args:
        ax: The kernel timeline's axes.
        selected: The groups, in plot order.
        kernels: The kernel table their kernel_ids index.
        kernel_phases: The phases with kernels, in plot order, for the legend.
        colors: Maps each phase to its color.
        heading: The panel's title, or None when the memory panel above holds it.
    """
    from matplotlib.patches import Patch

    background, labels = "#d1d5db", []
    for y, ((phase, module_fqn), group) in enumerate(selected.iterrows()):
        launches = kernels.loc[group["kernel_ids"]]
        labels.append(f"{phase}::{module_fqn}")
        ax.barh(y, group["gpu_end_us"] - group["gpu_start_us"], left=group["gpu_start_us"], height=0.8,
                color=background, linewidth=0, zorder=1)
        intervals = list(zip(launches["gpu_start_us"], launches["duration_us"]))
        bars = ax.broken_barh(intervals, (y - 0.32, 0.64), facecolors=colors[phase], edgecolors=colors[phase],
                              linewidth=0.6, zorder=3)
        bars.set_gid(f"kernels:{phase}::{module_fqn}")
    ax.set_yticks(range(len(selected)), labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time (us)")
    if heading is not None:
        ax.set_title(heading)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    handles = [Patch(color=colors[phase], label=f"{phase.replace('_', ' ').capitalize()} GPU")
               for phase in kernel_phases]
    handles.append(Patch(color=background, label="Group GPU span"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, borderpad=0.4, labelspacing=0.3, framealpha=0.9)
    ax.margins(y=0.02)


def _draw_flops(flop_ax, selected, colors):
    """Draws each group's estimated FLOPs as a bar in its phase's color, beside its kernel row.

    Args:
        flop_ax: The FLOPs panel's axes, sharing the kernel timeline's rows.
        selected: The groups, in plot order.
        colors: Maps each phase to its color.
    """
    from matplotlib.ticker import EngFormatter

    estimates = selected["total_est_flops"].to_numpy(dtype=float)
    phase_colors = [colors[phase] for phase in selected.index.get_level_values("phase")]
    flop_bars = flop_ax.barh(range(len(selected)), estimates, height=0.64, color=phase_colors, linewidth=0, zorder=3)
    formatter = EngFormatter(places=1, sep="")
    flop_ax.bar_label(flop_bars, labels=[formatter(value) for value in estimates], padding=3, fontsize=8)
    flop_ax.set_xlim(0, max(float(estimates.max()) * 1.35, 1))
    flop_ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    flop_ax.locator_params(axis="x", nbins=3)
    flop_ax.tick_params(axis="y", left=False, labelleft=False)
    flop_ax.set_xlabel("Estimated FLOPs per stage\n0 = no allocated estimate", fontsize=8)  # a title would widen the gap above
    flop_ax.grid(axis="x", alpha=0.2)
    flop_ax.set_axisbelow(True)


def _draw_durations(dots_ax, plotted, kernel_phases, colors):
    """Draws a lane per phase with every kernel's duration as a dot on a log axis, a black tick at the median.

    Args:
        dots_ax: The duration panel's axes.
        plotted: The rows of the plotted kernels.
        kernel_phases: The phases with kernels, in plot order, one lane each.
        colors: Maps each phase to its color.
    """
    jitter = np.random.default_rng(0)  # only spreads the dots vertically within a lane
    for lane, phase in enumerate(kernel_phases):
        durations = plotted.loc[plotted["phase"] == phase, "duration_us"]
        dots_ax.scatter(durations, lane + jitter.uniform(-0.3, 0.3, len(durations)), s=8, alpha=0.5,
                        color=colors[phase], linewidths=0)
        dots_ax.vlines(durations.median(), lane - 0.4, lane + 0.4, color="k", linewidth=1.5)  # the median
    dots_ax.set_yticks(range(len(kernel_phases)),
                       [f"{phase} (n={(plotted['phase'] == phase).sum()})" for phase in kernel_phases], fontsize=7)
    dots_ax.invert_yaxis()
    dots_ax.set_xscale("log")
    dots_ax.set_xlabel("kernel duration (us), black tick = median", fontsize=8)
    dots_ax.tick_params(axis="x", labelsize=7)
    dots_ax.grid(axis="x", alpha=0.2)


def _draw_memory(memory_ax, memory, peaks, memory_baseline, total_memory, persistent_memory, estimated_total_memory):
    """Draws allocated and reserved memory over time in MiB, with the lines for its peaks and totals.

    Args:
        memory_ax: The memory panel's axes.
        memory: allocated and reserved bytes indexed by time, or None.
        peaks: The peak allocated and reserved bytes, or None.
        memory_baseline: The allocated and reserved bytes before the model was built, or None.
        total_memory: The GPU's bytes, or None.
        persistent_memory: The bytes of weights, gradients and optimizer state, or None.
        estimated_total_memory: The estimated persistent plus activation bytes, or None.
    """
    highest = []
    for index, name in enumerate(["allocated", "reserved"]):
        color = f"C{index}"
        if memory is not None:
            memory_ax.step(memory.index, memory[name] / 2**20, where="post", color=color, label=name)
            highest.append(memory[name].max())
        if peaks is not None:
            memory_ax.axhline(peaks[name] / 2**20, color=color, linestyle=":",
                              label=f"peak {name} ({peaks[name] / 2**20:.1f} MiB)")
            highest.append(peaks[name])
        if memory_baseline is not None:
            memory_ax.axhline(memory_baseline[name] / 2**20, color=color, alpha=0.5,
                              label=f"{name} before model ({memory_baseline[name] / 2**20:.1f} MiB)")
    memory_ax.set_ylim(0, 1.1 * max(highest) / 2**20)  # a larger total is clipped, not rescaled
    if total_memory is not None:
        memory_ax.axhline(total_memory / 2**20, color="k", linestyle="--",
                          label=f"total VRAM ({total_memory / 2**30:.1f} GiB)")
    if persistent_memory is not None:
        memory_ax.axhline(persistent_memory / 2**20, color="grey", linestyle="--", alpha=0.5,
                          label=f"persistent ({persistent_memory / 2**20:.1f} MiB)")
    if estimated_total_memory is not None:
        memory_ax.axhline(estimated_total_memory / 2**20, color="C2", linestyle="--", alpha=0.6,
                          label=f"total, est. ({estimated_total_memory / 2**20:.1f} MiB)")
    memory_ax.set_ylabel("MiB")
    memory_ax.grid(axis="x", alpha=0.2)


def _shade_phases(panels, phases, colors):
    """Shades each phase's span behind every panel in its color, labelled at the top of the first.

    Args:
        panels: The axes to shade, the top one first.
        phases: The phase spans: name, start_us and end_us.
        colors: Maps each phase to its color.
    """
    for phase in phases.itertuples():
        for panel in panels:
            panel.axvspan(phase.start_us, phase.end_us, color=colors[phase.name], alpha=0.1, linewidth=0, zorder=0)
        panels[0].text((phase.start_us + phase.end_us) / 2, 0.98, phase.name, transform=panels[0].get_xaxis_transform(),
                       ha="center", va="top", fontsize=7, color=colors[phase.name])


def draw_timeline(
    fig, area, groups=None, kernels=None, title="Training step", *,
    color_seed=0, show_flops=True, memory=None, total_memory=None,
    memory_baseline=None, persistent_memory=None, estimated_total_memory=None, phases=None, peaks=None, info=(),
):
    """Draws a step's panels into area of fig: memory on top, then a row per kernel group with its FLOPs, then kernel durations.

    Each panel is drawn only when its data is there; phases alone get a strip of their own. Every
    time is us from one shared t = 0, so the panels line up, and the view spans everything drawn,
    padded by 5% on each side. Color depends only on phase. Details stay in pandas:
    kernels.loc[groups.loc[key, "kernel_ids"]].

    Args:
        fig: A figure with constrained layout, which spaces the panels and their labels.
        area: The SubplotSpec of fig to draw into; size it with timeline_height.
        groups: The groups from aggregate_kernels, or a subset of them; each kernel is drawn once.
        kernels: The kernel table their kernel_ids index.
        title: The top panel's title.
        color_seed: Seeds the order phases take their colors in.
        show_flops: Adds the panel of each group's total_est_flops beside its row.
        memory: allocated and reserved bytes indexed by time, drawn as steps in MiB.
        total_memory: The GPU's bytes, a dashed line.
        memory_baseline: The allocated and reserved bytes before the model was built, faded lines.
        persistent_memory: The bytes of weights, gradients and optimizer state, a dashed grey line.
        estimated_total_memory: The estimated persistent plus activation bytes, a dashed line.
        phases: The phase spans (name, start_us, end_us), shaded behind every panel.
        peaks: The peak allocated and reserved bytes, dotted lines; they draw the memory panel alone.
        info: Text lines under the memory panel's legend, or in the title without one.
    """
    has_kernels = groups is not None and not groups.empty
    if not has_kernels and memory is None and phases is None and peaks is None:
        return
    show_flops = show_flops and has_kernels
    selected, plotted = _plotted_groups(groups, kernels) if has_kernels else (None, None)
    kernel_phases = list(dict.fromkeys(selected.index.get_level_values("phase"))) if has_kernels else []
    colors = _phase_colors(kernel_phases + (list(phases["name"]) if phases is not None else []), color_seed)

    top = int(memory is not None or peaks is not None)  # the memory panel
    rows = max(top + 2 * has_kernels, 1)  # memory, kernels, kernel duration dots: one grid, so their edges align
    ratios = [3] * top + ([_kernel_panel_height(len(selected)), dots_height(len(kernel_phases))] if has_kernels else [])
    grid = area.subgridspec(rows, 2 if show_flops else 1, width_ratios=[4, 1.15] if show_flops else None,
                            height_ratios=ratios if rows > 1 else None)
    ax = None
    if has_kernels:
        ax = fig.add_subplot(grid[top, 0])
        if show_flops:
            _draw_flops(fig.add_subplot(grid[top, 1], sharey=ax), selected, colors)
        _draw_kernel_rows(ax, selected, kernels, kernel_phases, colors, None if top else _heading(title, info))
        _draw_durations(fig.add_subplot(grid[top + 1, 0]), plotted, kernel_phases, colors)
    if top:
        memory_ax = fig.add_subplot(grid[0, 0], sharex=ax)
        _draw_memory(memory_ax, memory, peaks, memory_baseline, total_memory, persistent_memory, estimated_total_memory)
        memory_ax.set_title(title)
        # The legend sits to the right of the memory panel, above the FLOPs panel when there is one.
        if show_flops:
            legend_ax = fig.add_subplot(grid[0, 1])
            legend_ax.axis("off")
            legend = legend_ax.legend(*memory_ax.get_legend_handles_labels(), loc="upper left", borderaxespad=0, fontsize=8)
            note_ax = legend_ax
        else:
            legend = memory_ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
            note_ax = memory_ax
        if info:  # just below the legend, wherever the layout puts it
            note_ax.annotate("\n".join(info), xy=(0, 0), xycoords=legend, xytext=(0, -6), textcoords="offset points",
                             va="top", ha="left", fontsize=8, annotation_clip=False)
        if has_kernels:
            memory_ax.tick_params(axis="x", labelbottom=False)
        else:
            memory_ax.set_xlabel("Time (us)")
    panels = ([memory_ax] if top else []) + ([ax] if has_kernels else [])
    if not panels:  # phases alone: a strip with just their bands
        strip = fig.add_subplot(grid[0, 0])
        strip.set_title(_heading(title, info))
        strip.set_yticks([])
        strip.set_xlabel("Time (us)")
        panels = [strip]
    if phases is not None:
        _shade_phases(panels, phases, colors)
    left, right = _time_extent(selected, memory, phases)
    if math.isfinite(left):  # peaks alone have no times
        pad = 0.05 * (right - left)  # the view, not the data
        panels[-1].set_xlim(left - pad, right + pad)
