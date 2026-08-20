"""DAG-level operations over the Job objects defined in jobs.py: walking a
set of jobs into topological order, and computing each one's status from
what's actually on disk. No execution lives here -- see job-spec.md, whose
scope is deliberately structural.
"""

from typing import Literal

from jobs import Job

Status = Literal["done", "runnable", "blocked"]


def topo_sort(*roots: Job) -> list[Job]:
    """DFS post-order over every dependency list, deduped by id --
    dependencies always come before whatever depends on them. Frozen nodes
    stop recursion on their own (their .deps is already {}), so they need
    no special case here.
    """
    seen: dict[str, Job] = {}
    order: list[Job] = []

    def visit(job: Job) -> None:
        if job.id in seen:
            return
        seen[job.id] = job
        for deps in job.deps.values():
            for dep in deps:
                visit(dep)
        order.append(job)

    for root in roots:
        visit(root)
    return order


def status(job: Job) -> Status:
    """done: every output list is non-empty and every path in it exists
    (the non-empty check matters for glob outputs -- nothing materialized
    yet globs to [], which must not vacuously count as done). runnable:
    not done, and every input path exists. blocked: otherwise.

    status(download_source_before_running) -> "runnable"
    """
    outputs = [p for paths in job.output_paths().values() for p in paths]
    if outputs and all(p.exists() for p in outputs):
        return "done"
    # Same non-empty guard per named input: an input sourced from a glob
    # family that hasn't been materialized yet resolves to [], which must
    # block rather than vacuously satisfy "every path exists".
    if all(paths and all(p.exists() for p in paths) for paths in job.input_paths().values()):
        return "runnable"
    return "blocked"


def scheduled(job: Job, statuses: dict[str, Status], _memo: dict[str, bool] | None = None) -> bool:
    """A node is scheduled if it isn't done, or anything upstream of it is
    scheduled. A frozen node is never scheduled, regardless of its status --
    it's taken as given, not something this DAG would (re)run.
    """
    if _memo is None:
        _memo = {}
    if job.id in _memo:
        return _memo[job.id]
    if job.frozen:
        result = False
    elif statuses[job.id] != "done":
        result = True
    else:
        result = any(
            scheduled(dep, statuses, _memo) for deps in job.deps.values() for dep in deps
        )
    _memo[job.id] = result
    return result


class SVG(str):
    """A str Jupyter renders as an image when it's a cell's last value."""

    def _repr_svg_(self) -> str:
        return str(self)


def _esc(s: str) -> str:
    """Escape text going into SVG so a path or id can't break the markup."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TYPE_COLORS = {
    "DownloadSource": "#cfe3ff",
    "TrainTokenizer": "#ffe4b3",
    "TokenizeSource": "#d7f3d7",
    "BuildSplit": "#f5d6ff",
    "Pretrain": "#eeceff",
    "SFT": "#ffd6e7",
}
_DEFAULT_COLOR = "#e8e8e8"

_STATUS_COLORS: dict[Status, str] = {
    "done": "#2a8f2a",
    "runnable": "#2a6f9f",
    "blocked": "#c9820a",
}


def _reasons_for(job: Job, statuses: dict[str, Status]) -> list[str]:
    """Why a blocked/runnable job isn't done yet: which of its own
    dependencies isn't done. Empty once the job itself is done."""
    if statuses[job.id] == "done":
        return []
    return [
        f"{type(dep).__name__} {dep.id[:8]} not done"
        for deps in job.deps.values()
        for dep in deps
        if statuses.get(dep.id, "blocked") != "done"
    ]


def dag_svg(
    *roots: Job,
    node_h: int = 40,
    char_w: float = 6.8,
    reason_char_w: float = 5.6,
    line_h: int = 11,
    row_gap: int = 28,
    col_gap: int = 14,
) -> SVG:
    """Render the DAG reachable from roots as a top-to-bottom SVG: one box
    per job, colored by job type and outlined by its dag.status() (green
    done, blue runnable, amber blocked; a frozen leaf gets a dashed
    border), with the job type and a short id on two lines and arrows from
    each dependency to its dependent. A blocked/runnable job lists which
    of its own dependencies isn't done yet, below its box. Just call it as
    a cell's last line in Jupyter -- no need to import IPython.display.SVG.
    """
    order = topo_sort(*roots)
    statuses = {j.id: status(j) for j in order}

    # order is topo (deps before dependents), so a single forward pass
    # gives rank = longest path from a root.
    rank: dict[str, int] = {}
    for j in order:
        deps = [d for jobs in j.deps.values() for d in jobs]
        rank[j.id] = 1 + max((rank[d.id] for d in deps), default=-1)

    rows: dict[int, list[Job]] = {}
    for j in order:
        rows.setdefault(rank[j.id], []).append(j)

    reasons = {j.id: _reasons_for(j, statuses) for j in order}
    labels = {j.id: (type(j).__name__, j.id[:8]) for j in order}
    widths = {
        j.id: max(
            70,
            int(max(len(labels[j.id][0]), len(labels[j.id][1])) * char_w) + 16,
            int(max((len(line) for line in reasons[j.id]), default=0) * reason_char_w) + 16,
        )
        for j in order
    }

    # Rows whose jobs list blocking reasons need extra height for those
    # lines, so row y-offsets accumulate rather than using a fixed stride.
    row_lines = {r: max((len(reasons[j.id]) for j in js), default=0) for r, js in rows.items()}
    row_y: dict[int, float] = {}
    y_cursor = 0.0
    for r in sorted(rows):
        row_y[r] = y_cursor
        y_cursor += node_h + row_lines[r] * line_h + row_gap

    pos: dict[str, tuple[float, float, float, float]] = {}  # id -> (x, y, w, h)
    max_row_w = 0.0
    for r, js in rows.items():
        x = 0.0
        for j in js:
            w = widths[j.id]
            pos[j.id] = (x, row_y[r], w, node_h)
            x += w + col_gap
        max_row_w = max(max_row_w, x - col_gap)

    legend_h = 24
    width = max_row_w + 40
    height = y_cursor - row_gap + 40 + legend_h
    ox, oy = 20, 20 + legend_h  # origin: left margin, below the legend

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

    lx = ox
    for st, color in _STATUS_COLORS.items():
        parts.append(
            f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{lx + 14}" y="13">{st}</text>'
        )
        lx += 14 + len(st) * 6 + 16
    parts.append(
        f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="none" '
        f'stroke="#666" stroke-width="1.5" stroke-dasharray="3 2"/>'
        f'<text x="{lx + 14}" y="13">frozen</text>'
    )

    for j in order:
        x, y, w, h = pos[j.id]
        x, y = x + ox, y + oy
        for deps in j.deps.values():
            for dep in deps:
                dx, dy, dw, dh = pos[dep.id]
                x1, y1 = dx + ox + dw / 2, dy + oy + dh
                x2, y2 = x + w / 2, y
                ym = (y1 + y2) / 2
                parts.append(
                    f'<path d="M{x1:.0f},{y1:.0f} C{x1:.0f},{ym:.0f} '
                    f'{x2:.0f},{ym:.0f} {x2:.0f},{y2:.0f}" fill="none" '
                    f'stroke="#999" stroke-width="1.5" marker-end="url(#arrow)"/>'
                )

    for j in order:
        x, y, w, h = pos[j.id]
        x, y = x + ox, y + oy
        fill = _TYPE_COLORS.get(type(j).__name__, _DEFAULT_COLOR)
        st = statuses[j.id]
        stroke = _STATUS_COLORS[st]
        dash = ' stroke-dasharray="4 2"' if j.frozen else ""
        title = f"{type(j).__name__} {j.id} [{st}]" + (" (frozen)" if j.frozen else "")
        label, short_id = labels[j.id]
        parts.append(
            f'<g><title>{_esc(title)}</title>'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" '
            f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
            f'<text x="{x + w / 2:.0f}" y="{y + h / 2 - 3:.0f}" '
            f'text-anchor="middle">{_esc(label)}</text>'
            f'<text x="{x + w / 2:.0f}" y="{y + h / 2 + 11:.0f}" '
            f'text-anchor="middle" font-size="9" fill="#777">{_esc(short_id)}</text></g>'
        )
        for i, line in enumerate(reasons[j.id]):
            ry = y + h + 12 + i * line_h
            parts.append(f'<text x="{x:.0f}" y="{ry:.0f}" font-size="9" fill="#888">{_esc(line)}</text>')

    parts.append("</svg>")
    return SVG("".join(parts))
