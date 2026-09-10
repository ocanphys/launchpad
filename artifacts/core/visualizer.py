"""Render a resolved Dag as an SVG graph, inline in a notebook cell.

Hand-built SVG (no graphviz/dot/networkx), rows by rank, one box per artifact.

Artifacts only. A job produces exactly one artifact, so drawing a box for the
artifact and an ellipse for its job said the same thing twice with twice the
nodes -- and it meant constructing a Job (and importing its family's torch or
numpy) just to read a class name off it. Rank, color and outline all come off
the Dag the caller already resolved, so drawing touches no filesystem either.
"""

from __future__ import annotations

from artifacts.core.resolve import Dag, Status

_ARTIFACT_COLORS = {
    "Source": "#cfe3ff",
    "Tokenizer": "#bfe6e0",
    "TokenizedSource": "#c9d6f7",
    "DataSet": "#d9d0f7",
    "MappedDataSet": "#d9d0f7",
    "Pretraining": "#f7d6c6",
    "MambaPretraining": "#f7d6c6",
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
_UNCHECKED_COLOR = "#666"  # resolved without a target: no status to show


class SVG(str):
    """A str Jupyter renders as an image when it's a cell's last value."""

    def _repr_svg_(self) -> str:
        return str(self)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ranks(dag: Dag) -> dict[str, int]:
    """Longest path from a leaf, so artifacts with nothing to build first
    (sources) land at rank 0 and each dependent sits below everything it
    needs.

    One forward pass, no recursion: `dag` is already in dependency order, so
    every dependency's rank is known by the time its dependent is reached.
    """
    rank: dict[str, int] = {}
    for node in dag:
        key = node.path.as_posix()
        rank[key] = 1 + max(
            (rank[dep.as_posix()] for dep in node.deps), default=-1
        )
    return rank


def visualize(
    dag: Dag,
    node_h: int = 40,
    char_w: float = 6.2,
    row_gap: int = 50,
    col_gap: int = 20,
) -> SVG:
    """Render a resolved graph: one box per artifact, fill color by type,
    outline color by status, arrows pointing from a dependency to what needs
    it. Just call it as a cell's last line in Jupyter.

    A `dag` resolved with a target is drawn with its statuses; one resolved
    without is drawn in a neutral outline, with no status legend.
    """
    checked = dag.target is not None
    rank = _ranks(dag)
    labels = {
        node.path.as_posix(): (type(node.artifact).__name__, node.artifact.uid)
        for node in dag
    }
    strokes = {
        node.path.as_posix(): (
            _STATUS_COLORS[node.status] if checked else _UNCHECKED_COLOR
        )
        for node in dag
    }
    titles = {
        node.path.as_posix(): (
            f"{': '.join(labels[node.path.as_posix()])}"
            + (f" [{node.status}]" if checked else "")
        )
        for node in dag
    }
    edges = {
        (dep.as_posix(), node.path.as_posix())  # dependency -> dependent
        for node in dag
        for dep in node.deps
    }

    rows: dict[int, list[str]] = {}
    for key in labels:
        rows.setdefault(rank[key], []).append(key)

    widths = {
        key: max(56, int(max(len(line) for line in lines) * char_w) + 16)
        for key, lines in labels.items()
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

    legend_h = 24 if checked else 0
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

    # legend: outline convention only. With one node kind there is no shape
    # convention left to explain.
    if checked:
        lx = ox
        for status, color in _STATUS_COLORS.items():
            parts.append(
                f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            parts.append(f'<text x="{lx + 14}" y="13">{status}</text>')
            lx += 14 + len(status) * 6 + 16

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
            f'<path d="M{x1:.0f},{y1:.0f} C{x1:.0f},{ym:.0f} {x2:.0f},{ym:.0f} '
            f'{x2:.0f},{y2:.0f}" fill="none" stroke="#999" stroke-width="1.5" '
            f'marker-end="url(#arrow)"/>'
        )

    for key, lines in labels.items():
        x, y, w, h = pos[key]
        x, y = x + ox, y + oy
        fill = _ARTIFACT_COLORS.get(lines[0], _DEFAULT_COLOR)
        shape = (
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" '
            f'rx="6" fill="{fill}" stroke="{strokes[key]}" stroke-width="2"/>'
        )
        cx = x + w / 2
        # multi-line labels stack around vertical center
        first_dy = -6 * (len(lines) - 1)
        tspans = "".join(
            f'<tspan x="{cx:.0f}" dy="{first_dy if i == 0 else 12}">{_esc(line)}</tspan>'
            for i, line in enumerate(lines)
        )
        parts.append(
            f"<g><title>{_esc(titles[key])}</title>{shape}"
            f'<text x="{cx:.0f}" y="{y + h / 2 + 4:.0f}" text-anchor="middle">'
            f"{tspans}</text></g>"
        )

    parts.append("</svg>")
    return SVG("".join(parts))
