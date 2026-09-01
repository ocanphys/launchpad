"""Render a resolve() plan as an SVG graph, inline in a notebook cell.

Style follows dag/dag.py: hand-built SVG (no graphviz/dot/networkx), rows by
rank, one shape per node kind. Artifacts are boxes (they're data at rest);
jobs are ellipses (they're the verb that produces them).
"""

from __future__ import annotations

from pathlib import Path

from dag.artifact import Artifact
from dag.job import Job
from dag.resolve import Status, status

_ARTIFACT_COLORS = {
    "Source": "#cfe3ff",
    "Tokenizer": "#bfe6e0",
    "TokenizedSource": "#c9d6f7",
    "DataSet": "#d9d0f7",
    "Pretraining": "#f7d6c6",
}
_JOB_COLORS = {
    "SourceJob": "#ffe4b3",
    "TokenizerJob": "#ffcfa8",
    "TokenizeSourceJob": "#f7c6e0",
    "DataSetJob": "#e8c6f7",
    "PretrainJob": "#ffc6c6",
}
_DEFAULT_COLOR = "#e8e8e8"
_STATUS_COLORS: dict[Status, str] = {
    "done": "#2a8f2a",
    "declared": "#2a6f9f",  # manifest written, work not started
    "partial": "#8f2a8f",  # some outputs -- interrupted, or still running
    "new": "#8a8a8a",  # not declared yet
    "conflict": "#c22a2a",  # a manifest there describes something else
    "undeclared": "#c9820a",  # outputs nobody declared
}


class SVG(str):
    """A str Jupyter renders as an image when it's a cell's last value."""

    def _repr_svg_(self) -> str:
        return str(self)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _collect(
    plan: dict,
) -> tuple[dict[str, tuple[str, str, Job | Artifact]], set[tuple[str, str]]]:
    """Flatten the plan into bipartite nodes (kind, label, obj) keyed by
    id, plus the artifact<->job edges between them. Ids are prefixed so an
    artifact and its producing job never collide even though they share a
    folder."""
    nodes: dict[str, tuple[str, str, Job | Artifact]] = {}
    edges: set[tuple[str, str]] = set()

    def walk(node: dict) -> None:
        artifact = node["artifact"]
        job = node["job"]
        a_id, j_id = f"a:{artifact.artifact_path}", f"j:{artifact.artifact_path}"
        nodes[a_id] = ("artifact", type(artifact).__name__, artifact)
        nodes[j_id] = ("job", type(job).__name__, job)
        edges.add((j_id, a_id))  # job produces artifact
        for child in node["dependencies"]:
            edges.add((f"a:{child['artifact'].artifact_path}", j_id))  # feeds job
            walk(child)

    walk(plan)
    return nodes, edges


def _ranks(nodes: dict, edges: set[tuple[str, str]]) -> dict[str, int]:
    """Longest path from a root (no incoming edge) -- same convention as
    dag.py's topo_sort-derived rank, so leaves (here: sources with nothing
    to download first) land at rank 0."""
    incoming: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        incoming[dst].append(src)

    rank: dict[str, int] = {}

    def r(n: str) -> int:
        if n not in rank:
            rank[n] = 1 + max((r(s) for s in incoming[n]), default=-1)
        return rank[n]

    for n in nodes:
        r(n)
    return rank


def visualize(
    plan: dict,
    root: Path | None = None,
    node_h: int = 40,
    char_w: float = 6.2,
    row_gap: int = 50,
    col_gap: int = 20,
) -> SVG:
    """Render a plan (from resolve.resolve()) as a bipartite artifact/job
    graph: boxes for artifacts, ellipses for jobs, fill color by type, and
    -- when root is given -- outline color by resolve.status(). Just call
    it as a cell's last line in Jupyter.
    """
    nodes, edges = _collect(plan)
    rank = _ranks(nodes, edges)

    rows: dict[int, list[str]] = {}
    for n in nodes:
        rows.setdefault(rank[n], []).append(n)

    def _display(kind: str, type_name: str, obj: Job | Artifact) -> tuple[str, ...]:
        # full uid, untruncated, on its own line -- uid formats vary now (a bare
        # name, or a readable-prefix-plus-hash), so a fixed-width slice can chop
        # the hash off, and a shared line with the type name would force the box
        # wide enough for both combined.
        return (type_name, obj.uid) if kind == "artifact" else (type_name,)

    widths = {
        n: max(
            56, int(max(len(line) for line in _display(kind, label, obj)) * char_w) + 16
        )
        for n, (kind, label, obj) in nodes.items()
    }

    row_w = {
        r: sum(widths[n] for n in ns) + col_gap * (len(ns) - 1)
        for r, ns in rows.items()
    }
    max_row_w = max(row_w.values(), default=0.0)

    pos: dict[str, tuple[float, float, float, float]] = {}
    for r, ns in rows.items():
        x = (max_row_w - row_w[r]) / 2  # center this row within the widest one
        for n in ns:
            w = widths[n]
            pos[n] = (x, r * (node_h + row_gap), w, node_h)
            x += w + col_gap

    legend_h = 24
    width = max_row_w + 40
    height = (
        (max(rank.values(), default=0) + 1) * (node_h + row_gap)
        - row_gap
        + 40
        + legend_h
    )
    ox, oy = 20, 20 + legend_h

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'font-family="monospace" font-size="11">'
        ),
        (
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#888"/></marker></defs>'
        ),
    ]

    # legend: shape convention (kind) + outline convention (status), so the
    # two color/shape systems don't get confused for one another.
    lx = ox
    parts.append(
        f'<rect x="{lx}" y="4" width="14" height="10" rx="2" fill="#ddd" stroke="#666"/>'
    )
    parts.append(f'<text x="{lx + 18}" y="13">artifact</text>')
    lx += 18 + len("artifact") * 6 + 16
    parts.append(
        f'<ellipse cx="{lx + 7}" cy="9" rx="7" ry="5" fill="#ddd" stroke="#666"/>'
    )
    parts.append(f'<text x="{lx + 18}" y="13">job</text>')
    lx += 18 + len("job") * 6 + 16
    if root is not None:
        for st, color in _STATUS_COLORS.items():
            parts.append(
                f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(f'<text x="{lx + 14}" y="13">{st}</text>')
            lx += 14 + len(st) * 6 + 16

    # Fan edges across a node's width instead of bunching every edge at its
    # center: when a row is centered, same-column nodes share an x, and
    # center-to-center edges land exactly on top of each other.
    def _fan(center_x: float, w: float, n: int) -> list[float]:
        if n <= 1:
            return [center_x]
        margin = min(w * 0.3, 10)
        span = w - 2 * margin
        return [center_x - w / 2 + margin + span * i / (n - 1) for i in range(n)]

    out_edges: dict[str, list[str]] = {}
    in_edges: dict[str, list[str]] = {}
    for src, dst in edges:
        out_edges.setdefault(src, []).append(dst)
        in_edges.setdefault(dst, []).append(src)

    src_anchor: dict[tuple[str, str], float] = {}
    for src, dsts in out_edges.items():
        sx, _, sw, _ = pos[src]
        order = sorted(dsts, key=lambda d: pos[d][0])
        for d, ax in zip(order, _fan(sx + ox + sw / 2, sw, len(order))):
            src_anchor[(src, d)] = ax

    dst_anchor: dict[tuple[str, str], float] = {}
    for dst, srcs in in_edges.items():
        dx, _, dw, _ = pos[dst]
        order = sorted(srcs, key=lambda s: pos[s][0])
        for s, ax in zip(order, _fan(dx + ox + dw / 2, dw, len(order))):
            dst_anchor[(s, dst)] = ax

    for src, dst in edges:
        _, sy, _, sh = pos[src]
        _, dy, _, _ = pos[dst]
        x1, y1 = src_anchor[(src, dst)], sy + oy + sh
        x2, y2 = dst_anchor[(src, dst)], dy + oy
        ym = (y1 + y2) / 2
        parts.append(
            f'<path d="M{x1:.0f},{y1:.0f} C{x1:.0f},{ym:.0f} {x2:.0f},{ym:.0f} {x2:.0f},{y2:.0f}" '
            f'fill="none" stroke="#999" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    for n, (kind, label, obj) in nodes.items():
        x, y, w, h = pos[n]
        x, y = x + ox, y + oy
        lines = _display(kind, label, obj)
        if kind == "artifact":
            fill = _ARTIFACT_COLORS.get(label, _DEFAULT_COLOR)
            stroke = _STATUS_COLORS[status(obj, root)] if root is not None else "#666"
            shape = f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        else:
            fill = _JOB_COLORS.get(label, _DEFAULT_COLOR)
            stroke = (
                _STATUS_COLORS[status(obj.artifact, root)]
                if root is not None
                else "#666"
            )
            cx, cy = x + w / 2, y + h / 2
            shape = f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{w / 2:.0f}" ry="{h / 2:.0f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        display = ": ".join(lines)
        title = (
            display
            if root is None
            else f"{display} [{status(obj.artifact if kind == 'job' else obj, root)}]"
        )
        cx = x + w / 2
        # multi-line labels stack around vertical center; a lone line just sits on it
        first_dy = -6 * (len(lines) - 1)
        tspans = "".join(
            f'<tspan x="{cx:.0f}" dy="{first_dy if i == 0 else 12}">{_esc(line)}</tspan>'
            for i, line in enumerate(lines)
        )
        parts.append(
            f"<g><title>{_esc(title)}</title>{shape}"
            f'<text x="{cx:.0f}" y="{y + h / 2 + 4:.0f}" text-anchor="middle">{tspans}</text></g>'
        )

    parts.append("</svg>")
    return SVG("".join(parts))
