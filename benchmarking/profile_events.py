"""Serialized `torch.profiler` events: the CPU call tree and the GPU activities under it.

`serialize` returns one row per event of `prof.events()`, in that order, and `index` is
position in that list. Two edge sets join the rows: `parent`/`children` is synchronous CPU
nesting, `launcher`/`gpu_activities` is the asynchronous launch edge from a CPU event to
the device activities CUPTI correlated with it. Both are edges between single events;
`subtrees` closes them over the CPU tree, which is what an operator's own work is. `capture`
runs one forward pass on a Modal GPU and returns the rows; `render` prints them.

`event.id` is the CUPTI correlation id and is 0 for every python_function, so it cannot be
the identity of a row; it is kept as `correlation_id` because the launch edge is built from
it. Device rows come from `prof.events()` rather than from each operator's `event.kernels`,
which keeps only a name, a device and a duration: the start and end of a kernel, its stream
and its device are on the device row alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("profile-events")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "einops")
    .add_local_python_source("profile_events", "models", "artifacts")
)

def serialize(prof) -> list[dict]:
    """Returns one row per event of `prof.events()`, linked by index."""
    from torch.autograd import DeviceType

    ordered = list(prof.events())
    position = {id(event): i for i, event in enumerate(ordered)}
    rows = [
        {
            "index": i,
            "name": event.name,
            "device_type": str(event.device_type),
            "activity_type": str(event.activity_type) if event.activity_type else None,
            "parent": position.get(id(event.cpu_parent)) if event.cpu_parent else None,
            "children": [],
            "launcher": None,
            "gpu_activities": [],
            "correlation_id": event.id,
            "linked_correlation_id": event.linked_correlation_id,
            "external_id": event.external_id,
            "start_us": event.time_range.start,
            "end_us": event.time_range.end,
            "duration_us": event.time_range.elapsed_us(),
            "thread": event.thread,
            "device_index": event.device_index,
            "device_resource_id": event.device_resource_id,
            "sequence_nr": event.sequence_nr,
            "flops": event.flops,
            "python_id": event.python_id,
            "python_parent_id": event.python_parent_id,
            "python_module_id": event.python_module_id,
            "input_shapes": event.input_shapes,
            "stack": list(event.stack or []),
            "scope": event.scope,
            "is_async": event.is_async,
        }
        for i, event in enumerate(ordered)
    ]

    # An operator's correlation id is what a device activity's `linked_correlation_id`
    # names, so the launch edge is that lookup. Only a RecordFunction scope carries an
    # external id, and only those correlation ids are unique: a python_function has 0,
    # and profiler overhead reuses the id of an operator it interrupted.
    by_correlation = {
        event.id: i
        for i, event in enumerate(ordered)
        if event.device_type == DeviceType.CPU and event.external_id
    }
    for row, event in zip(rows, ordered):
        if row["parent"] is not None:
            rows[row["parent"]]["children"].append(row["index"])
        if event.device_type != DeviceType.CPU:
            row["launcher"] = by_correlation.get(event.linked_correlation_id)
            if row["launcher"] is not None:
                rows[row["launcher"]]["gpu_activities"].append(row["index"])
    return rows


@app.function(image=image, gpu="T4", timeout=300)
def capture(batch_size: int = 8, warmup: int = 5) -> list[dict]:
    """Returns the serialized events of one profiled forward pass on a GPU."""
    import torch
    from torch import nn
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2)).cuda()
    x = torch.randn(batch_size, 64, device="cuda")

    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()  # finish warmup before the capture opens

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        model(x)
        torch.cuda.synchronize()  # include completion of the launched kernels
    return serialize(prof)


@app.local_entrypoint()
def main(out: str = "events.json"):
    rows = capture.remote()
    Path(out).write_text(json.dumps(rows, indent=1))
    launched = sum(len(row["gpu_activities"]) for row in rows)
    print(f"{len(rows)} events, {launched} GPU activities, written to {out}\n")
    print(render(rows))
