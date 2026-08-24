"""Render a resolve() manifest as an SVG graph, inline in a notebook cell.

Style follows dag/dag.py: hand-built SVG (no graphviz/dot/networkx), rows by
rank, one shape per node kind. Artifacts are boxes (they're data at rest);
jobs are ellipses (they're the verb that produces them).
"""

from __future__ import annotations

from pathlib import Path

from artifact import Artifact
from job import Job
from resolve import Status, status

_ARTIFACT_COLORS = {"Source": "#cfe3ff", "CombinedSource": "#d7f3d7"}
_JOB_COLORS = {"SourceJob": "#ffe4b3", "CombineJob": "#f5d6ff"}
_DEFAULT_COLOR = "#e8e8e8"
_STATUS_COLORS: dict[Status, str] = {
    "done": "#2a8f2a",
    "runnable": "#2a6f9f",
    "blocked": "#c9820a",
}


class SVG(str):
    """A str Jupyter renders as an image when it's a cell's last value."""

    def _repr_svg_(self) -> str:
        return str(self)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _collect(manifest: dict) -> tuple[dict[str, tuple[str, str, Job | Artifact]], set[tuple[str, str]]]:
    """Flatten the manifest into bipartite nodes (kind, label, obj) keyed by
    id, plus the artifact<->job edges between them. Ids are prefixed so an
    artifact and its producing job never collide even though they share a
    relpath."""
    nodes: dict[str, tuple[str, str, Job | Artifact]] = {}
    edges: set[tuple[str, str]] = set()

    def walk(node: dict) -> None:
        artifact = node["outputs"][0]
        job = node["job"]
        a_id, j_id = f"a:{artifact.relpath()}", f"j:{artifact.relpath()}"
        nodes[a_id] = ("artifact", type(artifact).__name__, artifact)
        nodes[j_id] = ("job", type(job).__name__, job)
        edges.add((j_id, a_id))  # job produces artifact
        for child in node["inputs"]:
            child_artifact = child["outputs"][0]
            in_id = f"a:{child_artifact.relpath()}"
            edges.add((in_id, j_id))  # artifact feeds job
            walk(child)

    walk(manifest)
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
    manifest: dict,
    root: Path | None = None,
    node_h: int = 40,
    char_w: float = 6.8,
    row_gap: int = 50,
    col_gap: int = 20,
) -> SVG:
    """Render manifest (from resolve.resolve()) as a bipartite artifact/job
    graph: boxes for artifacts, ellipses for jobs, fill color by type, and
    -- when root is given -- outline color by resolve.status(). Just call
    it as a cell's last line in Jupyter.
    """
    nodes, edges = _collect(manifest)
    rank = _ranks(nodes, edges)

    rows: dict[int, list[str]] = {}
    for n in nodes:
        rows.setdefault(rank[n], []).append(n)

    widths = {n: max(70, int(len(label) * char_w) + 24) for n, (_, label, _) in nodes.items()}

    pos: dict[str, tuple[float, float, float, float]] = {}
    max_row_w = 0.0
    for r, ns in rows.items():
        x = 0.0
        for n in ns:
            w = widths[n]
            pos[n] = (x, r * (node_h + row_gap), w, node_h)
            x += w + col_gap
        max_row_w = max(max_row_w, x - col_gap)

    legend_h = 24
    width = max_row_w + 40
    height = (max(rank.values(), default=0) + 1) * (node_h + row_gap) - row_gap + 40 + legend_h
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
    parts.append(f'<rect x="{lx}" y="4" width="14" height="10" rx="2" fill="#ddd" stroke="#666"/>')
    parts.append(f'<text x="{lx + 18}" y="13">artifact</text>')
    lx += 18 + len("artifact") * 6 + 16
    parts.append(f'<ellipse cx="{lx + 7}" cy="9" rx="7" ry="5" fill="#ddd" stroke="#666"/>')
    parts.append(f'<text x="{lx + 18}" y="13">job</text>')
    lx += 18 + len("job") * 6 + 16
    if root is not None:
        for st, color in _STATUS_COLORS.items():
            parts.append(f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{lx + 14}" y="13">{st}</text>')
            lx += 14 + len(st) * 6 + 16

    for src, dst in edges:
        sx, sy, sw, sh = pos[src]
        dx, dy, dw, _ = pos[dst]
        x1, y1 = sx + ox + sw / 2, sy + oy + sh
        x2, y2 = dx + ox + dw / 2, dy + oy
        ym = (y1 + y2) / 2
        parts.append(
            f'<path d="M{x1:.0f},{y1:.0f} C{x1:.0f},{ym:.0f} {x2:.0f},{ym:.0f} {x2:.0f},{y2:.0f}" '
            f'fill="none" stroke="#999" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    for n, (kind, label, obj) in nodes.items():
        x, y, w, h = pos[n]
        x, y = x + ox, y + oy
        if kind == "artifact":
            fill = _ARTIFACT_COLORS.get(label, _DEFAULT_COLOR)
            stroke = _STATUS_COLORS["done"] if (root is not None and obj.exists(root)) else "#666"
            shape = f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        else:
            fill = _JOB_COLORS.get(label, _DEFAULT_COLOR)
            stroke = _STATUS_COLORS[status(obj, root)] if root is not None else "#666"
            cx, cy = x + w / 2, y + h / 2
            shape = f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{w / 2:.0f}" ry="{h / 2:.0f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        title = label if root is None else f"{label} [{status(obj, root) if kind == 'job' else ('done' if obj.exists(root) else 'pending')}]"
        parts.append(
            f"<g><title>{_esc(title)}</title>{shape}"
            f'<text x="{x + w / 2:.0f}" y="{y + h / 2 + 4:.0f}" text-anchor="middle">{_esc(label)}</text></g>'
        )

    parts.append("</svg>")
    return SVG("".join(parts))
