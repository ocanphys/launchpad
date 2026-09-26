"""Tree of PyTorch operators from chrome traces, with the kernels each one launched.

`build` takes one parsed trace and the id of the worker that produced it, and returns the
roots. Every node is an `Op`; `cat` says whether it is an operator, a runtime call, a
kernel, or profiler overhead, and `thread` identifies it as (worker, pid, tid).

Host nesting is time containment within one thread, which is exact: a `cpu_op` record is
a RecordFunction scope on the C++ stack, so it cannot outlive its caller. Device events
attach to the runtime call that launched them by CUPTI correlation id, falling back to
the operator named by `External id`. A kernel that matches neither becomes a root rather
than disappearing.

Correlation and external ids are only unique within one trace, and `ts` is a boot-relative
clock private to the machine, so traces must be built one worker at a time. `build` shifts
`ts` onto the wall clock recorded in `baseTimeNanoseconds`, which makes nodes from
different workers comparable to within whatever NTP skew separates their hosts.

Kineto records its own cost on a separate track, so `Activity Buffer Request` comes back
as a root that overlaps the operators it interrupted. Its time is still inside their
durations, which is why `self_time` can exceed the Self CPU that `key_averages` reports
for the same operator on the first captured step.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

TEMPLATE = re.compile(r"^(?:void\s+)?([\w:]+)<")
HOST_CATS = ("cpu_op", "cuda_runtime", "overhead", "user_annotation")
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
EPSILON = 1e-3  # Kineto writes microseconds to three decimals.


@dataclass
class Op:
    name: str
    cat: str
    worker: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict = field(repr=False)
    parent: Op | None = field(default=None, repr=False, compare=False)
    children: list[Op] = field(default_factory=list, repr=False)

    @property
    def thread(self) -> tuple[str, int, int]:
        """The (worker, pid, tid) this node ran on."""
        return (self.worker, self.pid, self.tid)

    @property
    def end(self) -> float:
        return self.ts + self.dur

    @property
    def short(self) -> str:
        """The name without C++ template arguments."""
        match = TEMPLATE.match(self.name)
        return match.group(1) if match else self.name

    @property
    def shapes(self) -> list:
        return self.args.get("Input Dims", [])

    @property
    def stack(self) -> list[str]:
        """The Python frames that reached this operator, innermost first."""
        return [f for f in self.args.get("Call stack", "").split(";") if f]

    @property
    def self_time(self) -> float:
        """Host microseconds in this node but not in a nested host node."""
        return self.dur - sum(c.dur for c in self.children if c.cat not in DEVICE_CATS)

    @property
    def device_time(self) -> float:
        """Device microseconds launched by this node and everything below it."""
        return sum(o.dur for _, o in self.walk() if o.cat in DEVICE_CATS)

    def walk(self, depth: int = 0):
        """Yields (depth, op) for this node and every descendant, in start order."""
        yield depth, self
        for child in self.children:
            yield from child.walk(depth + 1)

    def ancestors(self):
        """Yields the enclosing nodes, innermost first."""
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


def build(trace: dict, worker: str = "") -> list[Op]:
    """Returns the root nodes of one worker's trace, on the wall clock."""
    events = trace["traceEvents"]
    offset = trace.get("baseTimeNanoseconds", 0) / 1000

    def node(e: dict) -> Op:
        return Op(
            name=e["name"],
            cat=e["cat"],
            worker=worker,
            pid=e["pid"],
            tid=e["tid"],
            ts=offset + e["ts"],
            dur=e["dur"],
            args=e.get("args", {}),
        )

    roots: list[Op] = []
    by_correlation: dict[int, Op] = {}
    by_external: dict[int, Op] = {}

    host = sorted(
        (e for e in events if e.get("cat") in HOST_CATS and "dur" in e),
        key=lambda e: (e["pid"], e["tid"], e["ts"], -e["dur"]),
    )
    stack: list[Op] = []
    thread = None
    for e in host:
        op = node(e)
        if op.thread != thread:
            thread, stack = op.thread, []
        while stack and stack[-1].end < op.end - EPSILON:
            stack.pop()

        op.parent = stack[-1] if stack else None
        (op.parent.children if op.parent else roots).append(op)
        stack.append(op)

        if "correlation" in op.args:
            by_correlation[op.args["correlation"]] = op
        if op.cat == "cpu_op" and "External id" in op.args:
            by_external.setdefault(op.args["External id"], op)

    device = sorted(
        (e for e in events if e.get("cat") in DEVICE_CATS and "dur" in e),
        key=lambda e: e["ts"],
    )
    for e in device:
        op = node(e)
        op.parent = by_correlation.get(op.args.get("correlation")) or by_external.get(
            op.args.get("External id")
        )
        (op.parent.children if op.parent else roots).append(op)

    return roots


def from_events(rows: list[dict], worker: str = "", pid: int = 0) -> list[Op]:
    """Returns the root nodes of one worker's `prof.events()` rows, linked by `parent`.

    Device rows are skipped: the same kernels arrive through each operator's `kernels`,
    which is the association PyTorch made from the linked correlation id.

    A `Kernel` records only a name, a device and a duration, so a kernel node takes its
    launching operator's `ts` and reports `tid` as -1; read the tree built from a trace for
    when a kernel ran, on which stream, and with what launch geometry. `cat` is inferred
    from the name, because a `FunctionEvent` does not carry Kineto's category.
    """
    nodes = {
        r["index"]: Op(
            name=r["name"],
            cat="cuda_runtime" if r["name"].startswith("cuda") else "cpu_op",
            worker=worker,
            pid=pid,
            tid=r["thread"],
            ts=r["start_us"],
            dur=r["end_us"] - r["start_us"],
            args={"Input Dims": r["input_shapes"], "Call stack": ";".join(r["stack"])},
        )
        for r in rows
        if r["device_type"] == "DeviceType.CPU"
    }

    roots = []
    for r in rows:
        op = nodes.get(r["index"])
        if op is None:
            continue
        op.parent = nodes.get(r["parent"]) if r["parent"] is not None else None
        (op.parent.children if op.parent else roots).append(op)
        for k in r["kernels"]:
            op.children.append(
                Op(
                    name=k["name"],
                    cat="kernel",
                    worker=worker,
                    pid=k["device"],
                    tid=-1,
                    ts=op.ts,
                    dur=k["duration_us"],
                    args={},
                    parent=op,
                )
            )
    return roots


def render(ops: list[Op]) -> str:
    """Returns the trees as indented text, one line per node, timed from the first root."""
    if not ops:
        return ""
    origin = min(op.ts for op in ops)
    lines = []
    for root in ops:
        for depth, op in root.walk():
            detail = f"  {op.shapes}" if op.shapes else ""
            if "grid" in op.args:
                detail = f"  grid{op.args['grid']} block{op.args['block']}"
            lines.append(
                f"{'  ' * depth}{op.short}  +{op.ts - origin:.1f} {op.dur:.1f}us "
                f"(self {op.self_time:.1f}) [{op.cat}]{detail}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    roots = []
    for path in sys.argv[1:]:
        with open(path) as f:
            roots += build(json.load(f), worker=path)
    print(render(roots))
