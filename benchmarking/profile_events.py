"""Module scopes for a profiled training step, and the tables built from its events.

A notebook's capture returns `list(prof.events())` unchanged: `FunctionEvent` objects pickle
with their `cpu_parent` and `cpu_children` references intact, so every step after profiling
runs locally. The capture container imports this module for `module_scopes`, so its image must
add this source, and pandas and matplotlib, which the worker image lacks, are imported where
they are used.
"""

from __future__ import annotations

import math
import random
from contextlib import ExitStack, contextmanager
from threading import local

import torch
from torch.overrides import TorchFunctionMode
from torch.profiler import record_function
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


def display_timeline(
    groups, kernels, title="Module scoped execution stages", *,
    color_seed=0, figsize=None, show_flops=True,
):
    """Plot group GPU spans with one separate bar per original kernel event.

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

    Details remain in pandas: kernels.loc[groups.loc[key, "kernel_ids"]].
    Filter groups to plot a subset; each selected kernel is drawn at most once.
    The function displays one figure and returns None to avoid duplicate output.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import EngFormatter

    if not groups.index.is_unique or not kernels.index.is_unique:
        raise ValueError("Group and kernel indices must be unique")
    selected = groups.sort_values("gpu_start_us", kind="stable")
    kernel_ids = [index for ids in selected["kernel_ids"] for index in ids]
    if len(kernel_ids) != len(set(kernel_ids)):
        raise ValueError("A kernel cannot appear in more than one plotted group")
    if selected.empty:
        return

    palette = {"green": "#15803d", "orange": "#c75a0a", "blue": "#1d4ed8",
               "purple": "#7c3aed", "teal": "#0f766e", "red": "#b91c1c",
               "pink": "#be185d", "gold": "#a16207"}
    phases = list(dict.fromkeys(selected.index.get_level_values("phase")))
    shuffled = list(palette.values())
    random.Random(color_seed).shuffle(shuffled)
    colors = {phase: shuffled[i % len(shuffled)]
              for i, phase in enumerate(sorted(phase for phase in phases if phase != "unscoped"))}
    if "unscoped" in phases:
        colors["unscoped"] = "#6b7280"
    background = "#d1d5db"
    origin = selected["gpu_start_us"].min()
    labels = []
    formatter = EngFormatter(places=1, sep="")

    # ioff prevents ipympl from displaying once on creation and again on show().
    with plt.ioff():
        height = max(3, 0.3 * len(selected) + 1.2)
        if show_flops:
            fig, (ax, flop_ax) = plt.subplots(
                1, 2, sharey=True, figsize=figsize or (12, height),
                gridspec_kw={"width_ratios": [4, 1.15]},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or (12, height))
        for y, ((phase, module_fqn), group) in enumerate(selected.iterrows()):
            launches = kernels.loc[group["kernel_ids"]]
            labels.append(f"{phase}::{module_fqn}")
            ax.barh(y, group["gpu_end_us"] - group["gpu_start_us"],
                    left=group["gpu_start_us"] - origin, height=0.8,
                    color=background, linewidth=0, zorder=1)
            # One collection per stage; each kernel still has its own exact
            # rectangle. This avoids hundreds of separate Matplotlib artists.
            intervals = list(zip(launches["gpu_start_us"] - origin, launches["duration_us"]))
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
            flop_ax.set_title("Estimated FLOPs per stage", fontsize=10)
            flop_ax.set_xlabel("0 = no allocated estimate", fontsize=8)
            flop_ax.grid(axis="x", alpha=0.2)
            flop_ax.set_axisbelow(True)

        ax.set_yticks(range(len(selected)), labels=labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Time since earliest displayed kernel (us); origin = {origin:,.3f} us")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        handles = [Patch(color=colors[phase], label=f"{phase.replace('_', ' ').capitalize()} GPU")
                   for phase in phases]
        handles.append(Patch(color=background, label="Group GPU span"))
        ax.legend(handles=handles, loc="upper right", fontsize=8,
                  borderpad=0.4, labelspacing=0.3, framealpha=0.9)
        ax.margins(x=0.01, y=0.02)
        fig.tight_layout(pad=0.6)
    plt.show()
