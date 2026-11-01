"""Pictures of event trees: `render` draws a linked event table, or one event in it, as an SVG.

The table is a profiled step's, from `events_to_rows`, or one of `CASES`, the small tables on
docs/profiling/attribution.md: raw rows as `events_to_rows` builds them, run through
`link_events` the same way. `render` runs it through `aggregate_kernels` at every depth from 1
to its deepest module stage and at None, and every launch arrow, kernel group and FLOP status
drawn is read from their output, so a picture shows what the code does. Positions are the
table's times; the cases' are schematic. Redraw the docs' pictures after changing attribution:

    uv run python -m profiling.attribution_diagrams
"""

from html import escape
from pathlib import Path

import pandas as pd

from profiling.profile_events import aggregate_kernels, launch_chain, link_events

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "attribution"

WIDTH, X0, RIGHT, TOP = 760, 50, 14, 10
PITCH, BAR, TALL, GAP, LINE = 24, 19, 31, 40, 12.5  # px: CPU row, bar, two-line bar, CPU to GPU, text line
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO_CHAR, SANS_CHAR = 0.61 * 10.5, 0.56 * 9.5  # px per character
INK, MUTED, RED, DEPTH = "#1f2937", "#6b7280", "#dc2626", "#6d28d9"
PYTORCH, CUPTI = "#1d4ed8", "#c2410c"  # the colors of PyTorch and CUPTI ids
STYLES = {  # fill, stroke, text
    "phase": ("#f3f4f6", "#9ca3af", "#374151"),
    "stage": ("#ede9fe", "#8b5cf6", "#4c1d95"),
    "aten": ("#dbeafe", "#3b82f6", "#1e3a8a"),
    "node": ("#ccfbf1", "#14b8a6", "#134e4a"),
    "launch": ("#ffedd5", "#f97316", "#7c2d12"),
    "kernel": ("#dcfce7", "#22c55e", "#14532d"),
    "copy": ("#fef9c3", "#eab308", "#713f12"),
    "gpu_annotation": ("#ffffff", "#9ca3af", "#4b5563"),
}
CHIPS = {"allocated": ("#dcfce7", "#15803d"), "suppressed": ("#f3f4f6", "#6b7280"),
         "unallocated": ("#fef3c7", "#b45309"), "depth": ("#ffffff", DEPTH)}
GPU_KINDS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
LAUNCHES = {"cuda_runtime", "cuda_driver"}
ROOT = "model"
ARROW_HEAD = ('<defs><marker id="head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
              f'orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="{INK}"/></marker></defs>')


def event(index, name, kind, start, end, parent=None, corr=0, linked=0, flops=0.0):
    """Returns one raw event row, as events_to_rows has it before linking.

        event(4, "cuLaunchKernel", "cuda_driver", 30, 58, parent=3, corr=39520, linked=18825)["device_type"]
        -> 'DeviceType.CPU'

    Args:
        index: The row ID.
        name: The event's name.
        kind: Its activity_type.
        start: Its start, in schematic time.
        end: Its end, in schematic time.
        parent: A CPU event's cpu_parent row ID; GPU events get theirs from link_events.
        corr: Its correlation_id.
        linked: Its linked_correlation_id.
        flops: The FLOPs the profiler reports for it.

    Returns:
        The row as a dict.
    """
    return {"index": index, "name": name, "activity_type": kind,
            "device_type": "DeviceType.CUDA" if kind in GPU_KINDS else "DeviceType.CPU",
            "correlation_id": corr, "linked_correlation_id": linked, "launcher_id": parent, "flops": float(flops),
            "start_us": float(start), "end_us": float(end)}


def _style(row):
    """Returns the STYLES key an event is drawn with.

    Args:
        row: The event's row.

    Returns:
        A STYLES key.
    """
    kind, name = row["activity_type"], row["name"]
    if kind == "user_annotation":
        return "phase" if name.startswith("train::") else "stage" if _fqn_depth(name) else "node"
    if kind == "cpu_op":
        return "aten" if name.startswith("aten::") else "node"
    if kind in LAUNCHES:
        return "launch"
    return {"kernel": "kernel", "gpu_user_annotation": "gpu_annotation"}.get(kind, "copy")


def _fqn_depth(name):
    """Returns the depth of a module stage: its FQN's components, the root included.

        _fqn_depth("forward::model.tblocks.0") -> 3

    Args:
        name: An event's name.

    Returns:
        The depth, 0 for an event that is no module stage.
    """
    fqn = name.partition("::")[2]
    return 1 + fqn.count(".") if fqn == ROOT or fqn.startswith(ROOT + ".") else 0


def _si(value):
    """Returns value with an SI prefix and three significant digits.

    Args:
        value: A count.

    Returns:
        The text, as 134M.
    """
    for exponent, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if value >= exponent:
            return f"{value / exponent:.3g}{prefix}"
    return f"{value:.3g}"


def _text(x, y, content, size=10.5, fill=INK, family=MONO, weight="normal", anchor="start"):
    """Returns one SVG text element.

    Args:
        x: Its anchor's x.
        y: Its baseline.
        content: The text.
        size: The font size in px.
        fill: The color.
        family: The font family.
        weight: The font weight.
        anchor: start, middle or end.

    Returns:
        The element.
    """
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{escape(content)}</text>')


def _pieces(pieces, gap=5):
    """Returns tspans for (content, color, weight) pieces, gap px apart.

    Args:
        pieces: The pieces, in order.
        gap: The space before each piece but the first.

    Returns:
        The tspans.
    """
    return "".join(f'<tspan dx="{gap if i else 0}" fill="{color}" font-weight="{weight}">{escape(content)}</tspan>'
                   for i, (content, color, weight) in enumerate(pieces))


def _line(x, y, pieces, size=9.5, anchor="start"):
    """Returns one line of text made of (content, color, weight) pieces.

    Args:
        x: Its anchor's x.
        y: Its baseline.
        pieces: The pieces, in order.
        size: The font size in px.
        anchor: start or end.

    Returns:
        The text element.
    """
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" '
            f'text-anchor="{anchor}">{_pieces(pieces)}</text>')


def _chip(right, y, content, kind):
    """Returns a rounded chip whose right edge is at right, and its width.

    Args:
        right: Its right edge.
        y: Its top.
        content: Its text.
        kind: A CHIPS key.

    Returns:
        (markup, width).
    """
    fill, ink = CHIPS[kind]
    width = len(content) * 5.6 + 10
    return (f'<rect x="{right - width:.1f}" y="{y:.1f}" width="{width:.1f}" height="14" rx="7" fill="{fill}" '
            f'stroke="{ink}" stroke-width="0.7"/>'
            + _text(right - width / 2, y + 10.2, content, 9, ink, SANS, "600", "middle"), width)


def _ids(row, own):
    """Returns the id pieces of an event: its correlation_id named by its counter, and its linked id.

    Args:
        row: The event's row.
        own: Show its own correlation_id.

    Returns:
        (content, color, weight) pieces.
    """
    cupti = row["activity_type"] in LAUNCHES | GPU_KINDS - {"gpu_user_annotation"}
    pieces = [(f"CUPTI {row.correlation_id}", CUPTI, "600") if cupti
              else (f"PyTorch {row.correlation_id}", PYTORCH, "600")] if own else []
    if row.linked_correlation_id:
        pieces += [("·", MUTED, "normal"), (f"linked {row.linked_correlation_id}", PYTORCH, "600")]
    return pieces


def _bar(x, y, width, row, chips=(), tall=False, dashed=False, own_id=False):
    """Returns one event's bar with its name, ids and FLOPs, and chips at its right end.

    Args:
        x: Its left edge.
        y: Its top.
        width: Its width.
        row: The event's row.
        chips: (content, CHIPS key) pairs, drawn right to left from the bar's end.
        tall: Two lines, the ids on the second.
        dashed: A dashed border, for an event with no launcher.
        own_id: Show the event's own correlation_id.

    Returns:
        The markup.
    """
    fill, stroke, ink = STYLES[_style(row)]
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    parts = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{TALL if tall else BAR}" rx="3" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{1.2 if dashed else 0.9}"{dash}/>']
    right = x + width - 3
    for content, kind in reversed(chips):
        markup, used = _chip(right, y + 2.5, content, kind)
        parts.append(markup)
        right -= used + 4
    ids = _ids(row, own_id)
    after = ([] if tall else ids) + ([(f"{_si(row.flops)} FLOPs", MUTED, "normal")] if row.flops > 0 else [])
    room = right - x - 14 - sum(len(content) + 1 for content, _, _ in after) * SANS_CHAR
    label = row.label if len(row.label) * MONO_CHAR <= room else row.label[:max(int(room / MONO_CHAR) - 1, 1)] + "…"
    trailer = f'<tspan dx="8" font-family="{SANS}" font-size="9.5">{_pieces(after)}</tspan>' if after else ""
    parts.append(f'<text x="{x + 6:.1f}" y="{y + 13.2:.1f}" font-family="{MONO}" font-size="10.5" fill="{ink}">'
                 f'{escape(label)}{trailer}</text>')
    if tall and ids:
        parts.append(_line(x + 6, y + 26, ids))
    return "".join(parts)


def _runs(values):
    """Returns (depth label, value) for each run of equal values over the depths 1, 2, ..., None.

        _runs({1: "model", 2: "model.ffn", None: "model.ffn"}) -> [("depth 1", "model"), ("depth 2+", "model.ffn")]

    Args:
        values: Maps each depth, in order and ending with None, to a value.

    Returns:
        One pair per run.
    """
    depths, found, start = list(values), [], 0
    for i in range(1, len(depths) + 1):
        if i < len(depths) and values[depths[i]] == values[depths[start]]:
            continue
        first, last = depths[start], depths[i - 1]
        if start == 0 and last is None:
            label = "every depth"
        elif last is None:
            label = f"depth {first}+"
        else:
            label = f"depth {first}" if first == last else f"depth {first}-{last}"
        found.append((label, values[depths[start]]))
        start = i
    return found


def _lanes(gpu):
    """Returns a lane per GPU event, kernels first, packing events that do not overlap into one lane.

    Args:
        gpu: The GPU rows, with start and end.

    Returns:
        Maps each row ID to its lane.
    """
    assigned, ends = {}, []
    for index, row in gpu.assign(other=gpu.activity_type.ne("kernel")).sort_values(["other", "start"]).iterrows():
        lane = next((i for i, end in enumerate(ends) if end <= row.start), len(ends))
        ends[lane:lane + 1] = [row.end]
        assigned[index] = lane
    return assigned


def render(rows, event_id=None, width=WIDTH, gpu_work_only=True):
    """Returns the SVG of an event tree: CPU events as nested bars, the GPU work below, linked by arrows.

        render(CASES["launch"]).startswith("<svg") -> True

    Module stages carry their depth; under each kernel is its phase and its group at every depth;
    every ATen op carries its FLOP status. A CPU annotation shows its id only when another event
    carries the same number.

    Args:
        rows: A linked event table: a profiled step's from events_to_rows, or one of CASES.
        event_id: Draws only this event, the events below it and the launch chain above it, the
            chain cut to the span of the rest; None draws the whole table.
        width: The picture's width in px.
        gpu_work_only: Leaves out the events that lead to no GPU work (views, reshapes,
            allocations, launch calls that queue nothing); event_id and its chain stay.

    Returns:
        The SVG document.
    """
    chain = []  # event_id's launch chain, root first
    if event_id is not None:
        chain = launch_chain(rows, event_id, format=False).index.tolist()
        children = {}
        for index, parent in rows["launcher_id"].dropna().items():
            children.setdefault(int(parent), []).append(index)
        below, pending = [], [event_id]
        while pending:
            below.append(pending.pop())
            pending.extend(children.get(below[-1], []))
        rows = rows[rows.index.isin(chain + below)]
    if gpu_work_only:  # every event on the way from GPU work up to its root
        drawn = set(chain)
        for index in rows.index[rows.activity_type.isin(GPU_KINDS)]:
            while pd.notna(index) and index not in drawn:
                drawn.add(index)
                index = rows.loc[index, "launcher_id"]
        rows = rows[rows.index.isin(drawn)]
    if chain:
        rest = rows.loc[~rows.index.isin(chain[:-1])]
        rows = rows.assign(start_us=rows["start_us"].clip(lower=rest["start_us"].min()),
                           end_us=rows["end_us"].clip(upper=rest["end_us"].max()))
    deepest = max([_fqn_depth(name) for name in rows["name"]] + [1])
    results = {depth: aggregate_kernels(rows, depth) for depth in [*range(1, deepest + 1), None]}
    _, kernels, _, sources = results[None]
    rows = rows.assign(start=rows["start_us"], end=rows["end_us"],
                       label=rows["name"].str.replace("autograd::engine::", "").str.replace("torch::autograd::", ""))
    stages = {name.partition("::")[2] for name in rows["name"] if _fqn_depth(name)}

    cpu, gpu = rows[~rows.activity_type.isin(GPU_KINDS)], rows[rows.activity_type.isin(GPU_KINDS)]
    level = {index: len(launch_chain(rows, index, format=False)) - 1 for index in cpu.index}
    groups = {index: _runs({depth: result[1].loc[index, "module_fqn"] for depth, result in results.items()})
              for index in kernels.index}
    lane_of = _lanes(gpu)
    lane_height = {}
    for index, lane in lane_of.items():
        lines = 1 + len(groups.get(index, []))
        lane_height[lane] = max(lane_height.get(lane, 0), TALL + 18 + LINE * lines)

    t0, t1 = rows.start.min(), rows.end.max()
    x = lambda t: X0 + (t - t0) * (width - X0 - RIGHT) / (t1 - t0)  # noqa: E731
    cpu_bottom = TOP + max(level[i] * PITCH + (TALL if cpu.loc[i, "activity_type"] in LAUNCHES else BAR) for i in cpu.index)
    gpu_top = cpu_bottom + GAP
    lane_top = {lane: gpu_top + sum(lane_height[i] for i in range(lane)) for lane in lane_height}
    height = gpu_top + sum(lane_height.values())
    divider = cpu_bottom + GAP / 2 - 4
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:.0f}" '
             f'viewBox="0 0 {width} {height:.0f}">', ARROW_HEAD,
             f'<rect width="{width}" height="{height:.0f}" fill="#ffffff"/>',
             f'<line x1="8" x2="{width - 8}" y1="{divider:.1f}" y2="{divider:.1f}" stroke="#e5e7eb" stroke-dasharray="2 4"/>',
             _text(10, TOP + 13.5, "CPU", 10, MUTED, SANS, "700"), _text(10, gpu_top + 13.5, "GPU", 10, MUTED, SANS, "700")]

    boxes = {}
    for index, row in cpu.iterrows():
        left, y, tall = x(row.start), TOP + level[index] * PITCH, row.activity_type in LAUNCHES
        boxes[index] = (left, y, x(row.end) - left, TALL if tall else BAR)
        if _fqn_depth(row["name"]):
            chips = ((f"depth {_fqn_depth(row['name'])}", "depth"),)
        elif index in sources.index:
            statuses = _runs({depth: result[3].loc[index, "status"] for depth, result in results.items()})
            count = sources.loc[index, "kernel_count"]
            chips = (((statuses[0][1] + (f" · {count} kernels" if count > 1 else "")), statuses[0][1]),) \
                if len(statuses) == 1 else tuple((f"{status} at {label}", status) for label, status in statuses)
        else:
            chips = ()
        others = rows.drop(index)
        shared = others.correlation_id.eq(row.correlation_id).any() or others.linked_correlation_id.eq(row.correlation_id).any()
        own_id = row.activity_type != "user_annotation" or (shared and not _fqn_depth(row["name"]))
        parts.append(_bar(left, y, x(row.end) - left, row, chips, tall, own_id=own_id))

    arrows, notes = [], []
    for index, row in gpu.iterrows():
        left, y, launcher = x(row.start), lane_top[lane_of[index]], row.launcher_id
        parts.append(_bar(left, y, x(row.end) - left, row, tall=True, dashed=pd.isna(launcher), own_id=True))
        below = y + TALL + 13
        if index in groups:
            notes.append(_line(left + 2, below, [(kernels.loc[index, "phase"], INK, "700")]))
            for i, (label, group) in enumerate(groups[index], 1):
                virtual = group not in stages and group != "<unattributed>"
                notes.append(_line(left + 2, below + i * LINE, [
                    (label, DEPTH, "600"), (group + (" (no stage)" if virtual else ""), MUTED if virtual else INK, "normal")]))
        else:
            notes.append(_line(left + 2, below, [("not a kernel: in no group", MUTED, "normal")]))
        if pd.isna(launcher):
            notes.append(_text(left + 2, y - 5, "no launcher: a root", 9.5, RED, SANS, "600"))
            continue
        bx, by, bw, bh = boxes[launcher]
        by_launch = rows.loc[launcher, "activity_type"] in LAUNCHES
        sx = bx + bw / 2
        if not by_launch:  # an op linked directly: leave from a part of its bar no deeper bar covers
            deeper = [box for i, box in boxes.items() if level[i] > level[launcher]]
            sx = next((p for p in (bx + bw - 6 - k for k in range(int(bw) - 12))
                       if not any(b[0] - 3 <= p <= b[0] + b[2] + 3 for b in deeper)), sx)
        sy, ex, ey = by + bh, left + 12, y - 1
        dash = "" if by_launch else ' stroke-dasharray="4 3"'
        arrows.append(f'<path d="M{sx:.1f},{sy:.1f} C{sx:.1f},{sy + (ey - sy) * 0.45:.1f} {ex:.1f},'
                      f'{ey - (ey - sy) * 0.45:.1f} {ex:.1f},{ey:.1f}" fill="none" stroke="{INK}" stroke-width="1.1"'
                      f'{dash} marker-end="url(#head)"/>')
        matched = ([("rule 2 · same", MUTED, "normal"), (f"CUPTI {row.correlation_id}", CUPTI, "600")]
                   if by_launch else
                   [("rule 3 ·", MUTED, "normal"), (f"linked {row.linked_correlation_id}", PYTORCH, "600"),
                    ("=", MUTED, "normal"), (f"PyTorch {row.linked_correlation_id}", PYTORCH, "600")])
        right_side = sx <= ex
        arrows.append(_line(ex + (7 if right_side else -7), ey - 5, matched, 9, "start" if right_side else "end"))
    return "\n".join(parts + arrows + notes + ["</svg>"])


def legend():
    """Returns the SVG of the key to render's pictures: event colors, ids, links, depth and FLOP statuses.

        legend().startswith("<svg") -> True

    Returns:
        The SVG document.
    """
    items = [("phase", "train::<phase>"), ("stage", "module stage"), ("aten", "ATen op"),
             ("node", "autograd node, optimizer"), ("launch", "launch call"), ("kernel", "kernel"),
             ("copy", "copy, memset"), ("gpu_annotation", "GPU annotation")]
    parts = []
    for i, (key, label) in enumerate(items):
        fill, stroke, _ = STYLES[key]
        x, y = 10 + (i % 4) * 185, 8 + (i // 4) * 22
        parts.append(f'<rect x="{x}" y="{y}" width="14" height="12" rx="2" fill="{fill}" stroke="{stroke}"/>'
                     + _text(x + 19, y + 10, label, 10, INK, SANS))
    y = 64
    for content, color, meaning in [
            ("PyTorch 18825", PYTORCH, "correlation_id from PyTorch's counter: ops, scopes and the GPU spans of scopes"),
            ("CUPTI 39520", CUPTI, "correlation_id from CUPTI's counter: launch calls and the GPU work they queue"),
            ("linked 18825", PYTORCH, "linked_correlation_id: the PyTorch id of the op that issued the launch")]:
        parts.append(_line(10, y, [(content, color, "600"), (meaning, INK, "normal")], 10))
        y += 18
    y += 4
    parts.append(f'<path d="M10,{y + 6} L44,{y + 6}" stroke="{INK}" stroke-width="1.1" marker-end="url(#head)"/>'
                 + _line(50, y + 10, [("rule 2:", MUTED, "normal"),
                                      ("the GPU work's CUPTI id is its launch call's", INK, "normal")], 10))
    parts.append(f'<path d="M370,{y + 6} L404,{y + 6}" stroke="{INK}" stroke-width="1.1" stroke-dasharray="4 3" '
                 'marker-end="url(#head)"/>'
                 + _line(410, y + 10, [("rule 3:", MUTED, "normal"),
                                       ("its linked id is an op's PyTorch id", INK, "normal")], 10))
    y += 26
    markup, used = _chip(10 + 7 * 5.6 + 10, y, "depth 3", "depth")
    parts.append(markup + _line(used + 18, y + 10.5, [("a module stage's FQN components, the root included", INK,
                                                       "normal")], 10))
    y += 20
    parts.append(_line(10, y + 10.5, [("depth 2+", DEPTH, "600"), ("model.ffn", INK, "normal"),
                                      ("under a kernel: its group at depth 2 and deeper, and at None", INK, "normal")], 10))
    y += 22
    x = 10
    for status in ("allocated", "suppressed", "unallocated"):
        markup, used = _chip(x + len(status) * 5.6 + 10, y, status, status)
        parts.append(markup)
        x += used + 8
    parts.append(_text(x + 4, y + 10.5, "FLOP source status", 10, MUTED, SANS))
    parts.append(_text(10, y + 34, "Vertical position is nesting in the launch tree, not depth.", 10, MUTED, SANS))
    height = y + 44
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">'
            f'{ARROW_HEAD}<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>' + "".join(parts) + "</svg>")


E = event
CASES = {name: link_events(pd.DataFrame(events).set_index("index").astype({"launcher_id": "Int64"}))
         for name, events in {
    "launch": [  # the launch sequence on docs/profiling/attribution.md, its rows and ids as captured
        E(5, "train::forward", "user_annotation", 0, 100, corr=18435),
        E(7, "forward::model", "user_annotation", 1, 99, 5, corr=18436),
        E(16, "forward::model.tblocks.0", "user_annotation", 2, 98, 7, corr=18442),
        E(501, "forward::model.tblocks.0.ffn", "user_annotation", 3, 97, 16, corr=18800),
        E(502, "aten::einsum", "cpu_op", 4, 80, 501, corr=18801),
        E(510, "aten::einsum", "cpu_op", 6, 78, 502, corr=18806),
        E(529, "aten::bmm", "cpu_op", 8, 74, 510, corr=18825, flops=134217728),
        E(531, "cuLaunchKernel", "cuda_driver", 32, 60, 529, corr=39520, linked=18825),
        E(532, "cutlass_80_tensorop_bf16_s16816gemm", "kernel", 63, 100, corr=39520, linked=18825),
    ],
    "depth": [
        E(0, "forward::model", "user_annotation", 0, 100, corr=1),
        E(1, "forward::model.tblocks.0", "user_annotation", 1, 99, 0, corr=2),
        E(2, "forward::model.tblocks.0.ffn", "user_annotation", 2, 60, 1, corr=3),
        E(3, "aten::bmm", "cpu_op", 3, 59, 2, corr=12, flops=134217728),
        E(4, "cudaLaunchKernel", "cuda_runtime", 24, 54, 3, corr=102, linked=12),
        E(5, "aten::add", "cpu_op", 63, 98, 1, corr=13),
        E(6, "cudaLaunchKernel", "cuda_runtime", 67, 95, 5, corr=103, linked=13),
        E(7, "gemm", "kernel", 57, 83, corr=102, linked=12),
        E(8, "add", "kernel", 97, 123, corr=103, linked=13),
    ],
    "several-launches": [
        E(0, "forward::model", "user_annotation", 0, 100, corr=1),
        E(1, "forward::model.attn", "user_annotation", 1, 99, 0, corr=2),
        E(2, "aten::einsum", "cpu_op", 2, 66, 1, corr=11, flops=2.1e6),
        E(3, "cudaLaunchKernel", "cuda_runtime", 5, 28, 2, corr=101, linked=11),
        E(4, "cudaLaunchKernel", "cuda_runtime", 35, 58, 2, corr=102, linked=11),
        E(5, "aten::copy_", "cpu_op", 68, 99, 1, corr=12),
        E(6, "cudaMemcpyAsync", "cuda_runtime", 71, 96, 5, corr=103, linked=12),
        E(7, "gemm", "kernel", 30, 55, corr=101, linked=11),
        E(8, "softmax", "kernel", 61, 86, corr=102, linked=11),
        E(9, "memcpy", "gpu_memcpy", 97, 122, corr=103, linked=12),
    ],
    "nested-estimates": [
        E(0, "forward::model", "user_annotation", 0, 100, corr=1),
        E(1, "forward::model.ffn", "user_annotation", 1, 99, 0, corr=2),
        E(2, "aten::einsum", "cpu_op", 1.5, 49.5, 1, corr=21),
        E(3, "aten::bmm", "cpu_op", 2, 49, 2, corr=22, flops=134217728),
        E(4, "cudaLaunchKernel", "cuda_runtime", 14, 40, 3, corr=201, linked=22),
        E(5, "aten::outer", "cpu_op", 51, 98.5, 1, corr=23, flops=300),
        E(6, "aten::inner", "cpu_op", 52, 98, 5, corr=24, flops=200),
        E(7, "cudaLaunchKernel", "cuda_runtime", 62, 88, 6, corr=202, linked=24),
        E(8, "gemm", "kernel", 41, 65, corr=201, linked=22),
        E(9, "kernel", "kernel", 89, 113, corr=202, linked=24),
    ],
    "spanning-estimate": [
        E(0, "forward::model", "user_annotation", 0, 100, corr=1),
        E(1, "aten::outer", "cpu_op", 1, 99, 0, corr=31, flops=300),
        E(2, "forward::model.left", "user_annotation", 2, 49, 1, corr=2),
        E(3, "aten::left", "cpu_op", 3, 48, 2, corr=32, flops=100),
        E(4, "cudaLaunchKernel", "cuda_runtime", 18, 45, 3, corr=301, linked=32),
        E(5, "forward::model.right", "user_annotation", 51, 98, 1, corr=3),
        E(6, "aten::right", "cpu_op", 52, 97, 5, corr=33, flops=200),
        E(7, "cudaLaunchKernel", "cuda_runtime", 66, 93, 6, corr=302, linked=33),
        E(8, "kernel", "kernel", 46, 70, corr=301, linked=32),
        E(9, "kernel", "kernel", 94, 116, corr=302, linked=33),
    ],
    "ambiguous-launch": [
        E(0, "forward::model", "user_annotation", 0, 100, corr=1),
        E(1, "aten::mm", "cpu_op", 2, 47, 0, corr=41),
        E(2, "cudaLaunchKernel", "cuda_runtime", 12, 40, 1, corr=77, linked=41),
        E(3, "aten::mm", "cpu_op", 53, 98, 0, corr=42),
        E(4, "cudaLaunchKernel", "cuda_runtime", 62, 90, 3, corr=77, linked=42),
        E(5, "gemm", "kernel", 52, 80, corr=77, linked=41),
    ],
    "no-match": [
        E(0, "forward::model", "user_annotation", 0, 100, corr=1),
        E(1, "aten::mul", "cpu_op", 2, 58, 0, corr=51),
        E(2, "cudaLaunchKernel", "cuda_runtime", 20, 52, 1, corr=501, linked=51),
        E(3, "kernel", "kernel", 55, 81, corr=501, linked=51),
        E(4, "kernel", "kernel", 88, 114, corr=777, linked=52),
    ],
    "gpu-annotation": [
        E(0, "train::optimizer_step", "user_annotation", 0, 100, corr=1),
        E(1, "Optimizer.step#AdamW.step", "user_annotation", 1, 99, 0, corr=1912),
        E(2, "aten::_foreach_add_", "cpu_op", 3, 97, 1, corr=61),
        E(3, "cudaLaunchKernel", "cuda_runtime", 40, 72, 2, corr=1912, linked=61),
        E(4, "Optimizer.step#AdamW.step", "gpu_user_annotation", 1, 99, corr=1912),
        E(5, "kernel", "kernel", 74, 100, corr=1912, linked=61),
    ],
    "backward": [
        E(0, "autograd::engine::evaluate_function: MmBackward0", "cpu_op", 0, 50, corr=1),
        E(1, "backward::model", "user_annotation", 1, 49, 0, corr=2),
        E(2, "backward::model.ffn", "user_annotation", 2, 48, 1, corr=3),
        E(3, "aten::mm", "cpu_op", 3, 47, 2, corr=71, flops=268435456),
        E(4, "cudaLaunchKernel", "cuda_runtime", 18, 44, 3, corr=701, linked=71),
        E(5, "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad", "cpu_op", 53, 102, corr=4),
        E(6, "aten::add_", "cpu_op", 55, 100, 5, corr=72),
        E(7, "cudaLaunchKernel", "cuda_runtime", 62, 90, 6, corr=702, linked=72),
        E(8, "gemm", "kernel", 46, 70, corr=701, linked=71),
        E(9, "add", "kernel", 92, 116, corr=702, linked=72),
    ],
    "outside-modules": [
        E(0, "train::loss", "user_annotation", 0, 46, corr=1),
        E(1, "aten::cross_entropy_loss", "cpu_op", 1, 45, 0, corr=81),
        E(2, "cudaLaunchKernel", "cuda_runtime", 14, 42, 1, corr=801, linked=81),
        E(3, "train::optimizer_step", "user_annotation", 50, 100, corr=2),
        E(4, "Optimizer.step#AdamW.step", "user_annotation", 51, 99, 3, corr=3),
        E(5, "aten::_foreach_mul_", "cpu_op", 53, 97, 4, corr=82),
        E(6, "cudaLaunchKernel", "cuda_runtime", 66, 95, 5, corr=802, linked=82),
        E(7, "nll_loss", "kernel", 44, 70, corr=801, linked=81),
        E(8, "foreach_mul", "kernel", 97, 123, corr=802, linked=82),
    ],
}.items()}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legend.svg").write_text(legend())
    for name, rows in CASES.items():
        (OUT / f"{name}.svg").write_text(render(rows, gpu_work_only=False))
