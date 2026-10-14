"""Module scopes for a profiled training step, and the tables built from its events.

A notebook's capture returns `list(prof.events())` unchanged: `FunctionEvent` objects pickle
with their `cpu_parent` and `cpu_children` references intact, so every step after profiling
runs locally. The capture container imports this module for `module_scopes` and the capture context
managers, so its image must add this source, and pandas and matplotlib, which the worker image lacks, are imported where
they are used.
"""

from __future__ import annotations

import math
import random
from contextlib import ExitStack, contextmanager
from threading import local

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils._pytree import tree_leaves

PHASE_ORDER = {p: i for i, p in enumerate(
    ["zero_grad", "forward", "loss", "backward", "optimizer_step", "unscoped"])}


@contextmanager
def module_scopes(model, root="model"):
    """Record module scoped execution stages for eager modules.

    Convention and organization
    ---------------------------
    A module scoped execution stage is a CPU user_annotation event created by
    record_function. Its name is <phase>::<module_fqn>, where phase is forward
    or backward and module_fqn is a named_modules path prefixed by root:

        forward::model                  # root module
        forward::model.tblocks.0        # submodule
        backward::model.tblocks.0       # same submodule during backward

    The same convention applies at every module depth. Forward stages nest
    according to actual module calls. Backward stages restore the recorded
    root-to-owner module path around each owned autograd node's execution.
    A stage name can therefore occur in multiple event rows; the original row
    index identifies each occurrence, not the stage name or module FQN.

    train::<phase> annotations describe whole training phases; module scoped
    execution stages describe module work within a phase. Backward workers may
    lack the outer train::backward ancestor, so the stage name encodes its phase
    directly. These annotations label execution; they are not autograd nodes.
    Following a kernel's launcher_id chain finds its nearest module scoped
    execution stage. If none exists, that kernel stays module-unattributed.

    Recording and ownership
    -----------------------
    Forward hooks track the active module path. TorchFunctionMode observes each
    operation's new autograd nodes and attaches that path to them once. Node
    pre/post hooks restore the path on the backward worker thread, surrounding
    its CPU operations and CUDA launches with backward::<FQN> annotations.
    Keep this context open for both forward and backward.

    Input graph nodes and nodes already owned by a child module are not claimed.
    AccumulateGrad is deliberately unowned: a shared parameter can accumulate
    contributions from several modules. Loss/optimizer work outside the module
    calls gets no module label. A kernel without a module annotation anywhere
    on its launcher_id chain remains unattributed.

    Shared modules use an alias below the active caller when possible, otherwise
    the first named_modules path. Aliases under the same caller are ambiguous
    to module hooks and use the first path. Direct .forward() calls bypass hooks.
    Compiled graphs and checkpoint recomputation are outside this eager workflow.
    All installed hooks and still-open scopes are cleaned up on context exit.
    No tensors or gradients are replaced.
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


@contextmanager
def capture_profiler_events(enabled):
    """Yield a dict whose "events" holds the block's profiler events once it exits; empty when not enabled.

    Event times are us from the trace's start; "trace_start_us" is that start on the unix clock.
    It synchronizes before the profiler stops, so every kernel launched inside the block is recorded.
    """
    profiled = {"events": []}
    if not enabled:
        yield profiled
        return
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        yield profiled
        torch.cuda.synchronize()
    profiled["events"].extend(prof.events())
    profiled["trace_start_us"] = prof.profiler.kineto_results.trace_start_ns() / 1e3


@contextmanager
def capture_memory(peak=False, trace=False, max_entries=100_000):
    """Yield a dict that holds the block's memory readings once it exits; empty when both are off.

    peak adds peak_allocated and peak_reserved bytes; trace adds the allocator snapshot of just
    this block under "snapshot".
    """
    memory = {}
    if peak:
        torch.cuda.reset_peak_memory_stats()
    if trace:
        torch.cuda.memory._record_memory_history(max_entries=max_entries, clear_history=True)
    try:
        yield memory
        if trace:
            memory["snapshot"] = torch.cuda.memory._snapshot()
    finally:
        if trace:
            torch.cuda.memory._record_memory_history(enabled=None)
    if peak:  # allocator statistics are kept on the CPU, so no synchronize is needed
        memory["peak_allocated"] = torch.cuda.max_memory_allocated()
        memory["peak_reserved"] = torch.cuda.max_memory_reserved()


def _validate_launchers(parents, cpu_ids):
    """Validate the one-parent forest; return children and parent-first order."""
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


def link_events(frame):
    """Connect events to their immediate CPU origin using original row IDs.

    ID reference
    ------------
    frame.index
        Unique integer row ID assigned when building the original event table.
        This is the identity used for navigation with frame.loc[row_id]. It is
        NOT a profiler correlation ID. Preserve it when filtering the table.
    launcher_id
        A row ID in this same table, or NA when no parent is known. On input,
        CPU rows must contain their PyTorch cpu_parent converted from an object
        reference to its row ID. On output, CPU parents are preserved and GPU
        rows point to their matched CPU launch call or frontend operation.
        For a CPU row, "launcher" means its enclosing CPU parent scope; it
        does not imply that every CPU event launches GPU work.
    correlation_id
        The original FunctionEvent.id, renamed during table construction.
        CPU frontend operations (cpu_op) and scopes (user_annotation) use
        PyTorch's counter. CUDA runtime/driver calls and GPU work use CUPTI's
        counter: a GPU activity shares its launch call's correlation ID.
        These counters can collide, so frontend and launch lookups are kept
        separate. A correlation ID is not a unique event-row identifier.
    linked_correlation_id
        For CUDA runtime/driver calls and GPU work, identifies the originating
        CPU frontend operation's correlation_id. It is not that operation's
        row ID, and does not identify the intermediate CUDA launch row.
        Zero or missing values mean there is no usable correlation link.
    node_id
        Profiler node identifier used to keep correlations from different
        nodes separate. Every lookup uses (node_id, correlation_id); absent
        or missing node IDs use -1. This is neither a row ID nor a GPU ID.
    external_id
        Raw profiler external-correlation metadata, retained unchanged if
        present. It is not a row ID and this function does not use it.

    Linking rules
    -------------
    1. CPU events keep the cpu_parent row links already computed by PyTorch.
    2. GPU kernels, copies and memsets first match correlation_id against CPU
       cuda_runtime/cuda_driver rows, inserting the CUDA API call in the path.
    3. If there is no unique launch match, match linked_correlation_id against
       CPU cpu_op/user_annotation rows instead. Other device events only use
       this fallback; GPU annotation spans must not join unrelated launch IDs.

    Missing or duplicate correlation matches are unresolved, never selected
    arbitrarily. After fallback, unresolved events retain launcher_id=NA.
    No time-based parent inference or cross-thread CPU ancestry is added.

    For example, following launcher_id from a kernel might visit original row
    IDs 50 -> 40 -> 30 -> 20, representing:
        GPU kernel -> cudaLaunchKernel -> aten::mm -> enclosing CPU scope
    Read the path in reverse for execution provenance. To find direct children
    of row 30, use rows.loc[rows["launcher_id"].eq(30)].

    Returns a new DataFrame in O(n) time and space, preserving every row, its
    index, order and other columns. Only launcher_id is updated (nullable
    Int64); no groups or duplicated ancestor rows are created. Raises
    ValueError for non-unique row IDs, missing/non-CPU parents, or cycles.
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

    # PyTorch frontend IDs and CUPTI launch IDs are separate namespaces; duplicates map to None.
    frontend, launches = {}, {}
    for event in events:
        if not event.is_cpu or pd.isna(event.correlation_id) or event.correlation_id == 0:
            continue
        if event.activity_type in {"cpu_op", "user_annotation"}:
            lookup = frontend
        elif event.activity_type in {"cuda_runtime", "cuda_driver"}:
            lookup = launches
        else:
            continue
        key = (event.node_id, event.correlation_id)
        lookup[key] = event.Index if key not in lookup else None

    gpu_work = {"kernel", "gpu_memcpy", "gpu_memset"}
    parents = {}
    for event in events:
        if event.is_cpu:
            parents[event.Index] = int(event.launcher_id) if pd.notna(event.launcher_id) else None
            continue

        launcher = None
        # Annotation spans must not match unrelated CUDA launch IDs.
        if event.activity_type in gpu_work and pd.notna(event.correlation_id):
            launcher = launches.get((event.node_id, event.correlation_id))
        if launcher is None and pd.notna(event.linked_correlation_id):
            launcher = frontend.get((event.node_id, event.linked_correlation_id))
        parents[event.Index] = launcher

    _validate_launchers(parents, {event.Index for event in events if event.is_cpu})
    return frame.assign(launcher_id=pd.array(list(parents.values()), dtype="Int64"))


def events_to_rows(function_events):
    """Return one linked row per profiler event, indexed by its position in function_events.

    cpu_parent object references become row IDs in launcher_id, then link_events adds the
    GPU launch links.
    """
    import pandas as pd
    
    position = {id(event): i for i, event in enumerate(function_events)}
    raw_rows = (
        pd.DataFrame([vars(event) for event in function_events])
        .rename(columns={"id": "correlation_id"})
        .drop(columns=["cpu_parent", "cpu_children", "kernels"])
        .assign(launcher_id=pd.array(
            [position.get(id(event.cpu_parent)) for event in function_events], dtype="Int64"))
        .rename_axis("index")
    )
    rows = link_events(raw_rows)
    assert rows.index.equals(raw_rows.index)  # every event keeps its row index
    return rows


LAUNCH_CHAIN_COLUMNS = [
    "name", "activity_type", "device_type",# "device_index",  # what the event is
    "start_us", "end_us",# "thread",  # timing
    "launcher_id", "correlation_id", "linked_correlation_id", #"sequence_nr", "fwd_thread", "scope",  # relations
    "flops",
    "input_shapes", "input_dtypes",  # shapes
    #"cpu_memory_usage", "device_memory_usage",  # memory
]


def launch_chain(rows, kernel_id, format=True):
    """Return the rows on kernel_id's launcher_id path, ordered from the CPU root down to the kernel.

    rows is the table from events_to_rows. A kernel with no launcher returns just its own row.
    format=True keeps LAUNCH_CHAIN_COLUMNS, with start_us and end_us expanded from time_range;
    format=False returns the raw rows.
    """
    import pandas as pd

    chain, current = [], kernel_id
    while pd.notna(current):
        chain.append(current)
        current = rows.loc[current, "launcher_id"]
    chain = rows.loc[chain[::-1]]
    if not format:
        return chain
    return chain.assign(start_us=chain["time_range"].map(lambda r: r.start),
                        end_us=chain["time_range"].map(lambda r: r.end))[LAUNCH_CHAIN_COLUMNS]


def _execution_context(event, inherited, root):
    """Return (phase, full owner FQN, owner stage row ID) for one tree node."""
    phase, owner, stage_id = inherited
    if str(event.device_type) != "DeviceType.CPU":
        return inherited
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


def aggregate_kernels(rows, depth=None, *, root="model"):
    """Partition kernel events by phase/module depth and roll up ATen FLOPs.

    Input
    -----
    rows is the complete event table returned by link_events, with unique
    original integer row IDs and launcher_id parent links. No events are
    removed from or added to rows. Module scoped execution stages use labels
    <phase>::<module_fqn>. GPU timestamps come from time_range.start/end in us.

    depth counts FQN components, including root: 1 = model, 2 = model.tblocks,
    3 = model.tblocks.0; None retains the full owning FQN. A container prefix
    need not have a recorded scope: it is a logical group, not a fabricated
    event. Direct work in a shallower module stays in that module's group.
    Repeated occurrences of the same phase/FQN merge. Kernel membership is
    exclusive: no group includes kernels already assigned to another group.

    Ownership and timing
    --------------------
    The nearest module stage owns a kernel. Without one, module_fqn is
    <unattributed>. Phase comes from the stage, train::<phase>, or a backward
    engine/optimizer scope; otherwise it is unscoped. All kernel rows appear
    once, including those with missing links. Copies and memsets are excluded.
    GPU span = latest end - earliest start (including gaps). GPU time is the
    sum of kernel durations (overlapping launches contribute independently).

    FLOPs
    -----
    Only CPU aten:: operations with descendant kernels contribute. A positive
    finite reported flops value replaces the sum of all child estimates.
    Otherwise, child estimates pass up through intermediate event nodes.
    Thus nested reported estimates are suppressed by the outermost reported
    ancestor, siblings add, and one ATen launching many kernels counts once.
    Every contribution is identified by its original ATen row ID, never by
    operation name. No FLOP total is replicated on individual kernel rows.

    Returns
    -------
    groups: DataFrame indexed by (phase, module_fqn) in phase execution order,
        with kernel_ids, owner_stage_ids, kernel_count, estimated_flops,
        flop_source_ids, gpu_start_us, gpu_end_us, gpu_span_us and gpu_time_us.
    kernels: Original kernel rows, at their original unique indices, sorted by
        phase execution order, module_fqn and gpu_start_us, augmented
        with phase, module_fqn, owner_fqn, owner_stage_id, gpu_start_us,
        gpu_end_us and duration_us. groups.kernel_ids references this table.
    unattributed_kernels: The rows of kernels whose module_fqn is <unattributed>.
    flop_sources: Positive reported ATen estimates with descendant kernels,
        indexed by original ATen row ID. status is allocated, suppressed or
        unallocated. suppressed_by identifies the overriding ATen ancestor;
        contribution_flops is nonzero only for allocated estimates.
        spans_groups checks the caveat below, including suppressed estimates.

    Limitations
    -----------
    FLOPs are partial operation estimates, not measured GPU instructions.
    Zero/missing/nonfinite values mean unavailable here; zero output does not
    prove zero work. CPU-only branches without kernels are excluded, but a
    reported operation estimate cannot separate mixed CPU/GPU computation.

    Usually module stages enclose ATen operations. Hypothetically an estimated
    ATen ancestor could instead enclose several module stages whose kernels
    fall into different requested groups. Multiple kernels or shared weights
    alone do NOT imply this. We check actual descendant group membership and
    flag spans_groups=True. If such an estimate is unsuppressed, it remains
    unallocated rather than being divided or duplicated; its descendants are
    still suppressed. Inspect rows.loc[flop_sources.index] for the real source
    events. Allocated group totals can consequently change with depth; the
    allocated plus unallocated estimate stays conserved. Suppressed reports
    must never be added back to totals.

    Missing ancestry stays unattributed; no phase/module links are invented.
    A kernel owned by a module stage whose launcher chain never reaches the
    root stage raises ValueError: its attribution would be incomplete.
    A merged group's time span can include gaps between repeated invocations.
    The shared validator builds the forest once. Ownership passes down and
    FLOPs pass up in O(events + kernels); no per-kernel
    ancestor walks or copies of entire subtrees are required.
    """
    import pandas as pd

    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
        raise ValueError("depth must be a positive integer or None")
    if not isinstance(root, str) or not root:
        raise ValueError("root must be a nonempty FQN")
    if not rows.index.is_unique:
        raise ValueError("Event row indices must be unique")

    # Read just the fields needed to build the forest and compute estimates.
    columns = ["name", "device_type", "activity_type", "launcher_id", "time_range", "flops"]
    events = {event.Index: event for event in rows.reindex(columns=columns).itertuples()}
    parents = {index: int(event.launcher_id) if pd.notna(event.launcher_id) else None
               for index, event in events.items()}
    cpu_ids = {index for index, event in events.items() if str(event.device_type) == "DeviceType.CPU"}
    children, order = _validate_launchers(parents, cpu_ids)
    kernel_ids = [index for index, event in events.items() if event.activity_type == "kernel"]
    kernel_set = set(kernel_ids)

    # One top-down pass carries module ownership and the overriding FLOP source.
    contexts, group_keys, reported, suppressed_by, active_source = {}, {}, {}, {}, {}
    rooted, unrooted = {}, []
    owner_groups = {None: "<unattributed>"}
    for index in order:
        event, parent = events[index], parents[index]
        context = contexts.get(parent, ("unscoped", None, None))
        if index in cpu_ids and (event.activity_type == "user_annotation"
                                 or event.name.startswith("autograd::engine::evaluate_function:")):
            context = _execution_context(event, context, root)
        contexts[index] = context
        rooted[index] = rooted.get(parent, False) or (
            index in cpu_ids and event.activity_type == "user_annotation"
            and event.name.partition("::")[2] == root)
        if index in kernel_set:
            phase, owner, _ = context
            if owner is not None and not rooted[index]:
                unrooted.append(index)
            if owner not in owner_groups:
                components = [root] + owner[len(root) + 1:].split(".") if owner != root else [root]
                owner_groups[owner] = owner if depth is None else ".".join(components[:depth])
            group_keys[index] = phase, owner_groups[owner]

        ancestor = active_source.get(parent)
        if (index in cpu_ids and event.name.startswith("aten::")
                and pd.notna(event.flops) and event.flops > 0 and math.isfinite(event.flops)):
            reported[index] = float(event.flops)
            suppressed_by[index] = ancestor
            if ancestor is None:
                ancestor = index
        active_source[index] = ancestor

    if unrooted:
        raise ValueError(f"Kernels owned by a module stage but not under the {root} stage: {unrooted}")

    # A subtree needs just one group key or a mixed marker, not a set of kernels.
    mixed = object()
    destinations, kernel_counts, effective_flops = {}, {}, {}
    for index in reversed(order):
        destination = group_keys[index] if index in kernel_set else None
        count = int(index in kernel_set)
        child_flops = 0.0
        for child in children[index]:
            count += kernel_counts[child]
            child_flops += effective_flops[child]
            child_destination = destinations[child]
            if child_destination is not None:
                if destination is None:
                    destination = child_destination
                elif destination != child_destination:
                    destination = mixed
        destinations[index], kernel_counts[index] = destination, count
        effective_flops[index] = reported.get(index, child_flops) if count else 0.0

    # Only estimated ATen nodes need source records, not every event in the tree.
    source_records, group_flops, group_sources = [], {}, {}
    for index, amount in reported.items():
        if not kernel_counts[index]:
            continue
        blocker = suppressed_by[index]
        destination = destinations[index]
        spans_groups = destination is mixed
        if blocker is not None:
            status = "suppressed"
        else:
            status = "unallocated" if spans_groups else "allocated"
        phase, fqn = (None, None) if spans_groups else destination
        contribution = amount if status == "allocated" else 0.0
        source_records.append((index, events[index].name, amount, contribution,
                               status, blocker, spans_groups, kernel_counts[index], phase, fqn))
        if status == "allocated":
            group_flops[destination] = group_flops.get(destination, 0.0) + amount
            group_sources.setdefault(destination, []).append(index)

    flop_sources = pd.DataFrame(source_records, columns=[
        "aten_id", "name", "reported_flops", "contribution_flops", "status", "suppressed_by",
        "spans_groups", "kernel_count", "phase", "module_fqn",
    ]).set_index("aten_id")
    flop_sources["suppressed_by"] = pd.array(flop_sources["suppressed_by"], dtype="Int64")
    flop_sources["spans_groups"] = flop_sources["spans_groups"].astype(bool)

    # Keep one detail row per original kernel, including its GPU interval.
    kernels = rows.loc[kernel_ids].copy()
    starts, ends = [], []
    for index, interval in kernels["time_range"].items():
        start, end = float(interval.start), float(interval.end)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Invalid GPU interval for kernel {index}")
        starts.append(start)
        ends.append(end)
    kernels["phase"] = [group_keys[index][0] for index in kernel_ids]
    kernels["module_fqn"] = [group_keys[index][1] for index in kernel_ids]
    kernels["owner_fqn"] = [contexts[index][1] for index in kernel_ids]
    kernels["owner_stage_id"] = pd.array([contexts[index][2] for index in kernel_ids], dtype="Int64")
    kernels["gpu_start_us"], kernels["gpu_end_us"] = starts, ends
    kernels["duration_us"] = kernels["gpu_end_us"] - kernels["gpu_start_us"]

    summaries = []
    for key, members in kernels.groupby(["phase", "module_fqn"], sort=False, dropna=False):
        start, end = members["gpu_start_us"].min(), members["gpu_end_us"].max()
        summaries.append((*key, members.index.tolist(), members["owner_stage_id"].dropna().unique().tolist(),
                          len(members), group_flops.get(key, 0.0), group_sources.get(key, []),
                          start, end, end - start, members["duration_us"].sum()))
    groups = pd.DataFrame(summaries, columns=[
        "phase", "module_fqn", "kernel_ids", "owner_stage_ids", "kernel_count", "estimated_flops",
        "flop_source_ids", "gpu_start_us", "gpu_end_us", "gpu_span_us", "gpu_time_us",
    ]).set_index(["phase", "module_fqn"])

    assigned = [index for members in groups["kernel_ids"] for index in members]
    if len(assigned) != len(kernel_ids) or set(assigned) != set(kernel_ids):
        raise ValueError("Kernel groups must partition all input kernels exactly once")

    # Sort by phase in execution order, then module, then GPU start.
    phase_rank = lambda level: level.map(PHASE_ORDER) if level.name == "phase" else level
    groups = groups.sort_index(key=phase_rank)
    kernels = kernels.sort_values(["phase", "module_fqn", "gpu_start_us"], key=phase_rank)
    unattributed_kernels = kernels[kernels["module_fqn"].eq("<unattributed>")]

    allocated = flop_sources["contribution_flops"].sum()
    unallocated = flop_sources.loc[flop_sources["status"].eq("unallocated"), "reported_flops"].sum()
    if not math.isclose(allocated + unallocated, sum(effective_flops[index] for index in order if parents[index] is None)):
        raise ValueError("FLOP sources must conserve the bottom-up estimate")
    return groups, kernels, unattributed_kernels, flop_sources


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


def transformer_matmul_comparison(transformer, tokens, rows):
    """Returns (per-phase, per-module) tables of analytical vs profiled matmul FLOPs for a TransformerLM.

    `rows` must be a capture of `transformer` run on `tokens` under `module_scopes(model)` with
    the default root, since the analytical rows are keyed by `model.tblocks.<i>.attention` and
    friends. Matmul estimates exclude normalization, activations, loss and optimizer work.
    """
    import pandas as pd

    # Multiply + add = 2 FLOPs. All B*S tokens count: logits are computed for every position
    # before the loss slices off the last. Attention forms full S x S matrices, then masks them.
    batch, sequence = tokens.shape
    n_tokens = batch * sequence
    d_model, d_ff, vocab = transformer.d_model, transformer.d_ff, transformer.vocab_size
    expected = {}
    # Backward takes gradients for both inputs of every matmul: twice the forward work.
    for phase, multiplier in [("forward", 1), ("backward", 2)]:
        for layer in range(transformer.num_layers):
            # Four projections, then QK^T and AV.
            expected[phase, f"model.tblocks.{layer}.attention"] = multiplier * (
                8 * n_tokens * d_model**2 + 4 * batch * sequence**2 * d_model
            )
            # Three SwiGLU projections.
            expected[phase, f"model.tblocks.{layer}.ffn"] = multiplier * 6 * n_tokens * d_model * d_ff
        expected[phase, "model.linear"] = multiplier * 2 * n_tokens * d_model * vocab

    # Full module ownership, independent of any plotted depth.
    groups, _, _, sources = aggregate_kernels(rows)
    matmuls = sources.loc[
        sources["status"].eq("allocated") & sources["name"].isin(["aten::mm", "aten::bmm", "aten::addmm"])
    ]
    # Outer join keeps unexpected profiled matmul work instead of hiding it.
    stages = (
        pd.Series(expected, name="analytical_matmul_flops")
        .rename_axis(["phase", "module_fqn"])
        .to_frame()
        .join(matmuls.groupby(["phase", "module_fqn"])["contribution_flops"].sum()
              .rename("profiler_matmul_flops"), how="outer")
        .fillna(0)
    )
    stages["difference"] = stages["profiler_matmul_flops"] - stages["analytical_matmul_flops"]

    by_phase = stages.groupby(level="phase").sum()
    phases = pd.DataFrame({
        "analytical_matmul_flops": by_phase["analytical_matmul_flops"],
        "profiler_matmul_flops": by_phase["profiler_matmul_flops"],
        "profiler_total_flops": groups.groupby(level="phase")["estimated_flops"].sum(),
    }).reindex(pd.Index(["forward", "backward"], name="phase"), fill_value=0)
    phases["profiler_other_flops"] = phases["profiler_total_flops"] - phases["profiler_matmul_flops"]
    phases["matmul_difference"] = phases["profiler_matmul_flops"] - phases["analytical_matmul_flops"]
    phases.loc["forward + backward"] = phases.sum()
    return phases, stages


def dots_height(phase_count):
    """Return the height in inches of the kernel duration dots under the kernel timeline: a lane per phase."""
    return 0.8 * (0.5 + 0.3 * phase_count)


def timeline_height(groups=None, memory=None, phases=None, peaks=None):
    """Return the height in inches draw_timeline wants for these groups, memory, phases and peaks."""
    has_kernels = groups is not None and not groups.empty
    kernels_height = (max(3, 0.3 * len(groups) + 1.2)
                      + dots_height(groups.index.get_level_values("phase").nunique())) if has_kernels else 0
    height = kernels_height + 3 * (memory is not None or peaks is not None)
    return height or 1.5 * (phases is not None)  # phases alone get a strip of their own


def draw_timeline(
    fig, area, groups=None, kernels=None, title="Training step", *,
    color_seed=0, show_flops=True, memory=None, total_memory=None,
    memory_baseline=None, persistent_memory=None, phases=None, peaks=None, info=(),
):
    """Draw group GPU spans, one bar per original kernel event, and memory above them, into area of fig.

    area is a SubplotSpec of a figure with constrained layout, which spaces the panels and their
    labels; size it with timeline_height. title goes on the top panel. Every time
    (kernels' gpu_start_us and gpu_end_us, memory's index, phases) is us from one shared t = 0, so
    the panels line up; step_timeline puts a captured step's data on one.

    Each panel is drawn only when its data is there: without groups (or with none selected) there
    is no kernel timeline and no FLOPs panel, and without memory there is no memory panel. With
    neither, the phases get a strip of their own.

    Rows are ordered by earliest GPU start. Gray backgrounds show each group's
    enclosing interval; colored rectangles show exact individual kernel intervals.
    Rectangles are batched by stage for faster rendering without merging events.
    Color depends only on phase, including for module-unattributed kernels.
    Unscoped is gray and appears in the legend only when present. Other phases
    draw from a shuffled palette; color_seed makes the assignment reproducible.

    show_flops adds a right-hand panel with one estimated_flops bar per stage,
    using exactly the same groups, depth, row order and phase colors. These are
    allocated estimates from aggregate_kernels, never a sum of per-kernel FLOPs.
    A zero means no allocated estimate, not necessarily zero computation.
    Stage labels contain only phase and module FQN. Rounded values in the
    right-hand panel are for display; groups.estimated_flops holds exact totals.

    memory adds a top panel with one step line per column (allocated, reserved) in MiB, sharing the
    time axis. peaks ({"allocated", "reserved"} bytes) draws each as a dotted line in that panel, which
    then appears even without memory. total_memory, in bytes, adds a dashed line.
    memory_baseline maps a memory column to its bytes before the model was built, drawn as a faded line
    in that column's color. persistent_memory, in bytes, adds a dashed grey line for the memory that stays
    allocated through the step (weights, gradients and optimizer state).

    phases shades each phase of the step (zero_grad, forward, ...) behind every panel, in its phase's
    color and labelled at the top. It is a DataFrame of name, start_us and end_us.

    With kernels, a thin panel under the kernel timeline has a lane per phase with every kernel's
    duration as a dot on a log axis, in the phase colors of the kernel legend and a black tick at the
    lane's median. info is a list of text lines shown just below the legend of the top panel; without
    a memory panel they join the title.

    Details remain in pandas: kernels.loc[groups.loc[key, "kernel_ids"]].
    Filter groups to plot a subset; each selected kernel is drawn at most once.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import EngFormatter

    has_kernels = groups is not None and not groups.empty
    if not has_kernels and memory is None and phases is None and peaks is None:
        return
    show_flops = show_flops and has_kernels
    height = 0
    if has_kernels:
        if not groups.index.is_unique or not kernels.index.is_unique:
            raise ValueError("Group and kernel indices must be unique")
        selected = groups.sort_values("gpu_start_us", kind="stable")
        kernel_ids = [index for ids in selected["kernel_ids"] for index in ids]
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("A kernel cannot appear in more than one plotted group")
        height = max(3, 0.3 * len(selected) + 1.2)  # the kernel panel; the dots panel is dots_height

    palette = {"green": "#15803d", "orange": "#c75a0a", "blue": "#1d4ed8",
               "purple": "#7c3aed", "teal": "#0f766e", "red": "#b91c1c",
               "pink": "#be185d", "gold": "#a16207"}
    kernel_phases = list(dict.fromkeys(selected.index.get_level_values("phase"))) if has_kernels else []
    named = dict.fromkeys(kernel_phases + (list(phases["name"]) if phases is not None else []))
    shuffled = list(palette.values())
    random.Random(color_seed).shuffle(shuffled)
    colors = {phase: shuffled[i % len(shuffled)]
              for i, phase in enumerate(sorted(phase for phase in named if phase != "unscoped"))}
    colors["unscoped"] = "#6b7280"

    top = int(memory is not None or peaks is not None)  # the memory panel
    rows = max(top + 2 * has_kernels, 1)  # memory, kernels, kernel duration dots: one grid, so their edges align
    ratios = [3] * top + [height, dots_height(len(kernel_phases))] * has_kernels
    grid = area.subgridspec(rows, 2 if show_flops else 1, width_ratios=[4, 1.15] if show_flops else None,
                            height_ratios=ratios if rows > 1 else None)
    left, right = math.inf, -math.inf  # the view spans the earliest to the latest time drawn
    if has_kernels:
        background = "#d1d5db"
        labels = []
        formatter = EngFormatter(places=1, sep="")

        ax = fig.add_subplot(grid[top, 0])
        if show_flops:
            flop_ax = fig.add_subplot(grid[top, 1], sharey=ax)
        for y, ((phase, module_fqn), group) in enumerate(selected.iterrows()):
            launches = kernels.loc[group["kernel_ids"]]
            labels.append(f"{phase}::{module_fqn}")
            ax.barh(y, group["gpu_end_us"] - group["gpu_start_us"],
                    left=group["gpu_start_us"], height=0.8,
                    color=background, linewidth=0, zorder=1)
            # One collection per stage; each kernel still has its own exact
            # rectangle. This avoids hundreds of separate Matplotlib artists.
            intervals = list(zip(launches["gpu_start_us"], launches["duration_us"]))
            bars = ax.broken_barh(intervals, (y - 0.32, 0.64),
                                 facecolors=colors[phase], edgecolors=colors[phase],
                                 linewidth=0.6, zorder=3)
            bars.set_gid(f"kernels:{phase}::{module_fqn}")

        if show_flops:
            estimates = selected["estimated_flops"].to_numpy(dtype=float)
            phase_colors = [colors[phase] for phase in selected.index.get_level_values("phase")]
            flop_bars = flop_ax.barh(range(len(selected)), estimates, height=0.64,
                                     color=phase_colors, linewidth=0, zorder=3)
            flop_ax.bar_label(flop_bars, labels=[formatter(value) for value in estimates],
                              padding=3, fontsize=8)
            flop_ax.set_xlim(0, max(float(estimates.max()) * 1.35, 1))
            flop_ax.xaxis.set_major_formatter(EngFormatter(sep=""))
            flop_ax.locator_params(axis="x", nbins=3)
            flop_ax.tick_params(axis="y", left=False, labelleft=False)
            flop_ax.set_xlabel("Estimated FLOPs per stage\n0 = no allocated estimate", fontsize=8)  # a title would widen the gap above
            flop_ax.grid(axis="x", alpha=0.2)
            flop_ax.set_axisbelow(True)

        ax.set_yticks(range(len(selected)), labels=labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Time (us)")
        if not top:
            ax.set_title("\n".join([title, " | ".join(info)] if info else [title]))
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        handles = [Patch(color=colors[phase], label=f"{phase.replace('_', ' ').capitalize()} GPU")
                   for phase in kernel_phases]
        handles.append(Patch(color=background, label="Group GPU span"))
        ax.legend(handles=handles, loc="upper right", fontsize=8,
                  borderpad=0.4, labelspacing=0.3, framealpha=0.9)
        ax.margins(y=0.02)
        left, right = selected["gpu_start_us"].min(), selected["gpu_end_us"].max()

        plotted = kernels.loc[kernel_ids]
        dots_ax = fig.add_subplot(grid[top + 1, 0])
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
    if top:
        memory_ax = fig.add_subplot(grid[0, 0], sharex=ax if has_kernels else None)
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
        if memory is not None:
            left, right = min(left, memory.index.min()), max(right, memory.index.max())
        memory_ax.set_ylim(0, 1.1 * max(highest) / 2**20)  # a larger total is clipped, not rescaled
        if total_memory is not None:
            memory_ax.axhline(total_memory / 2**20, color="k", linestyle="--",
                              label=f"total VRAM ({total_memory / 2**30:.1f} GiB)")
        if persistent_memory is not None:
            memory_ax.axhline(persistent_memory / 2**20, color="grey", linestyle="--", alpha=0.5,
                              label=f"persistent ({persistent_memory / 2**20:.1f} MiB)")
        memory_ax.set_ylabel("MiB")
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
        memory_ax.grid(axis="x", alpha=0.2)
        if has_kernels:
            memory_ax.tick_params(axis="x", labelbottom=False)
        else:
            memory_ax.set_xlabel("Time (us)")
    panels = ([memory_ax] if top else []) + ([ax] if has_kernels else [])
    if not panels:  # phases alone: a strip with just their bands
        strip = fig.add_subplot(grid[0, 0])
        strip.set_title("\n".join([title, " | ".join(info)] if info else [title]))
        strip.set_yticks([])
        strip.set_xlabel("Time (us)")
        panels = [strip]
    if phases is not None:
        for phase in phases.itertuples():
            left, right = min(left, phase.start_us), max(right, phase.end_us)
            for panel in panels:
                panel.axvspan(phase.start_us, phase.end_us, color=colors[phase.name], alpha=0.1, linewidth=0, zorder=0)
            panels[0].text((phase.start_us + phase.end_us) / 2, 0.98, phase.name, transform=panels[0].get_xaxis_transform(),
                           ha="center", va="top", fontsize=7, color=colors[phase.name])
    if math.isfinite(left):  # peaks alone have no times
        # Pad the view by 5% of the plotted time on each side; the data is untouched.
        pad = 0.05 * (right - left)
        panels[-1].set_xlim(left - pad, right + pad)


def step_timeline(readings, depth=3):
    """Return draw_timeline's data for one captured step, in us from the step's start on the unix clock.

    readings is one step of capture's result; depth is the module depth of the kernel rows. The
    result has "groups" and "kernels" with kernel_events, "memory" with trace, "peaks" with peak
    and "phases" with wall_clock. The allocator, the profiler (through its trace start) and the phases all stamp unix
    time, so they share one clock; the step starts at the earliest time any of them recorded.
    """
    import pandas as pd

    timeline, starts = {}, []
    if readings["events"]:
        timeline["groups"], timeline["kernels"], _, _ = aggregate_kernels(
            events_to_rows(readings["events"]), depth=depth)
        starts.append(timeline["kernels"]["gpu_start_us"].min() + readings["trace_start_us"])
    if "snapshot" in readings:
        timeline["memory"] = memory_over_time(readings["snapshot"])[0]
        starts.append(timeline["memory"].index.min())
    if "wall_clock_phase_spans" in readings:
        timeline["phases"] = pd.DataFrame(readings["wall_clock_phase_spans"], index=["start_us", "end_us"]).T.rename_axis("name").reset_index()
        starts.append(timeline["phases"]["start_us"].min())
    if "peak_allocated" in readings:
        timeline["peaks"] = {"allocated": readings["peak_allocated"], "reserved": readings["peak_reserved"]}
    if not starts:  # nothing with a time, so nothing to move
        return timeline
    start = min(starts)

    for key in ("groups", "kernels"):  # profiler times count from its trace start
        if key in timeline:
            shift = readings["trace_start_us"] - start
            timeline[key] = timeline[key].assign(gpu_start_us=timeline[key]["gpu_start_us"] + shift,
                                                 gpu_end_us=timeline[key]["gpu_end_us"] + shift)
    if "memory" in timeline:
        timeline["memory"] = timeline["memory"].set_axis(timeline["memory"].index - start)
    if "phases" in timeline:
        timeline["phases"] = timeline["phases"].assign(start_us=timeline["phases"]["start_us"] - start,
                                                     end_us=timeline["phases"]["end_us"] - start)
    return timeline


def display_steps(captured, steps=None, depth=3):
    """Display each captured step's timeline, stacked top to bottom in one figure.

    captured is capture's result; steps picks step numbers (all by default) and depth the module
    depth of the kernel rows. A step's panels appear only if capture recorded their data: kernels
    with kernel_events, memory with trace, peak lines with peak, the shaded phases with wall_clock.
    Persistent memory and total VRAM are drawn wherever the memory panel is. All of them are on the
    unix clock, from the step's earliest recorded time (step_timeline). Allocations are stamped
    when the CPU issues them and kernels when the GPU runs them, so a kernel can start after the
    allocation for its output.

    Under each step's legend: the GPU, the step's wall time, the GPU's dense peak TFLOPS for the
    precision capture ran in (gpus.peaks), the achieved TFLOPS and the MFU. Achieved is capture's
    step_flops over the step's time, so it counts matmul and attention FLOPs only, and the time
    includes any instruments that were on.
    """
    import matplotlib.pyplot as plt
    from gpus.peaks import PRECISIONS, peak_tflops, spec_row

    gpu, precision = captured["gpu"], captured["precision"]
    peak = peak_tflops(gpu, precision)
    infos = {}
    for step in steps if steps is not None else captured["steps"]:
        step_ms = captured["steps"][step]["step_ms"]
        achieved = captured["step_flops"] / (step_ms / 1e3) / 1e12  # TFLOPS
        infos[step] = [f"GPU: {gpu} [{spec_row(gpu)}]",
                       f"step time: {step_ms:.1f} ms",
                       f"peak: {peak:,.1f} TFLOPS {PRECISIONS[precision]}",
                       f"achieved: {achieved:,.1f} TFLOPS",
                       f"MFU: {achieved / peak:.1%}"]
    timelines = {step: step_timeline(captured["steps"][step], depth)
                 for step in (steps if steps is not None else captured["steps"])}
    heights = {step: timeline_height(timeline.get("groups"), timeline.get("memory"), timeline.get("phases"), timeline.get("peaks"))
               for step, timeline in timelines.items()}
    timelines = {step: timeline for step, timeline in timelines.items() if heights[step]}  # nothing recorded, no plot
    if not timelines:
        return
    heights = [heights[step] for step in timelines]
    # ioff prevents ipympl from displaying once on creation and again on show().
    with plt.ioff():
        fig = plt.figure(figsize=(12, sum(heights)), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)  # tight, in inches and fractions
        for area, (step, timeline) in zip(fig.add_gridspec(len(heights), 1, height_ratios=heights), timelines.items()):
            draw_timeline(fig, area, title=f"step {step}", total_memory=captured["device_vram"],
                          persistent_memory=captured["persistent_memory"], info=infos[step], **timeline)
    plt.show()
