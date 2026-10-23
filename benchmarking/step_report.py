"""The report of a capture: `build_report` turns capture's result into one JSON-serializable dict.

The report holds everything `display_steps` shows and what the capture ran with, and nothing else:
no events, no colors, no kernel names. It is the one place the analysis lives, so plotting only
reads it. This module is local-only (it needs pandas and gpus.peaks); the Modal worker never
imports it.

Layout (steps are keyed by step number as a string, times are us from the step's start):

    {"schema_version": 1,
     "run":   {"gpu", "model", "optimizer", "capture", "precision", "peak",
               "step_flops", "sources", "persistent_memory_bytes"},
     "steps": {"0": {"start_unix_us", "step_ms", "mfu", "phases", "memory", "kernels", "groups"}}}

Units are in the names: `step_flops` and `estimated_flops` are counts of FLOPs, `*_flops_per_s` are rates
(FLOPs per second), `*_ms` are milliseconds, `*_us` microseconds, `*_bytes` bytes. Every FLOP number is an
algorithmic estimate; `run["sources"]` says how each was obtained. The per-step `mfu` block holds the
step's FLOP count, its wall time, the achieved and peak rates and their ratio.

`run["model"]` is {"class", "parameters", "size"}: the declaration's class name, its fields (vocab_size,
d_model, ...) and the weights' count and bytes.

`phases` and `groups` are lists of records. `kernels` and `memory["series"]` are columnar: one list
per field, so `pd.DataFrame(table)` gives the table back. A block is present only when capture
recorded it: kernels and groups with kernel_events, memory["series"] with trace, the memory peaks
with peak, phases with wall_clock_phase_spans.

Kernel ids are the row ids `events_to_rows` gives the step's events, so they index the same rows the
appendix's `launch_chain(rows, kernel_id)` follows.
"""

import math

from gpus.peaks import PRECISIONS, peak_tflops, spec_row
from profile_events import aggregate_kernels, events_to_rows, memory_over_time

SCHEMA_VERSION = 1

# Where each FLOP number comes from. None of them is read from a hardware counter.
SOURCES = {
    "step_flops": ("algorithmic estimate: the model declaration's step_flops, an analytic count of one forward "
                   "and backward pass from the architecture's shapes"),
    "kernel_flops": "algorithmic estimate: the torch profiler's per-op formulas from tensor shapes, rolled up per group",
    "peak_flops_per_s": "vendor datasheet, dense tensor-core rate (gpus/peaks.py)",
    "achieved_flops_per_s": "step_flops divided by the step's measured wall time",
    "mfu": "achieved_flops_per_s divided by peak_flops_per_s",
}


def build_report(captured, depth=3):
    """Return the JSON-serializable report of capture's result; depth is the module depth of the kernel rows."""
    gpu, precision = captured["gpu"], captured["precision"]
    peak = peak_tflops(gpu, precision) * 1e12
    peak = None if math.isnan(peak) else float(peak)  # NaN: the GPU has no unit for this precision
    sizes = captured["size_estimation"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "gpu": {"name": gpu, "spec_row": spec_row(gpu), "vram_bytes": captured["device_vram"]},
            "model": {"class": captured["model_class"], "parameters": captured["model_parameters"],
                      "size": {"parameters": sizes["model"]["parameters"], "bytes": sizes["model"]["bytes"]}},
            "optimizer": {"class": captured["optimizer"], "state_parameters": sizes["optimizer"]["parameters"],
                          "state_bytes": sizes["optimizer"]["bytes"]},
            "capture": captured["parameters"],
            "precision": precision,
            "peak": {"dtype": PRECISIONS[precision], "flops_per_s": peak},
            "step_flops": captured["step_flops"],
            "sources": SOURCES,
            "persistent_memory_bytes": captured["persistent_memory"],
        },
        "steps": {str(step): _step(readings, captured["step_flops"], peak, depth)
                  for step, readings in captured["steps"].items()},
    }


def _step(readings, step_flops, peak, depth):
    """Return one step's report on the unix clock, from the earliest time any of its blocks recorded."""
    step = {"step_ms": readings["step_ms"]}
    starts = []  # the earliest time of each block, unix us
    if readings["events"]:
        groups, kernels, _, _ = aggregate_kernels(events_to_rows(readings["events"]), depth=depth)
        shift = readings["trace_start_us"]  # the profiler counts from its trace start
        starts.append(kernels["gpu_start_us"].min() + shift)
    if "snapshot" in readings:
        series = memory_over_time(readings["snapshot"])[0]
        starts.append(series.index.min())
    if "wall_clock_phase_spans" in readings:
        spans = readings["wall_clock_phase_spans"]
        starts.append(min(start for start, _ in spans.values()))
    start = min(starts) if starts else None
    if start is not None:
        step["start_unix_us"] = float(start)

    achieved = step_flops / (readings["step_ms"] / 1e3)  # FLOPs per second
    step["mfu"] = {"step_flops": step_flops, "step_ms": readings["step_ms"], "achieved_flops_per_s": achieved,
                   "peak_flops_per_s": peak, "mfu": None if peak is None else achieved / peak}

    if "wall_clock_phase_spans" in readings:
        step["phases"] = [{"name": name, "start_us": begin - start, "end_us": end - start,
                           "duration_ms": (end - begin) / 1e3}
                          for name, (begin, end) in readings["wall_clock_phase_spans"].items()]
    memory = {}
    if "peak_allocated" in readings:
        memory["peak_allocated_bytes"] = readings["peak_allocated"]
        memory["peak_reserved_bytes"] = readings["peak_reserved"]
    if "snapshot" in readings:
        memory["series"] = {"time_us": (series.index - start).tolist(),
                            "allocated": series["allocated"].tolist(), "reserved": series["reserved"].tolist()}
    if memory:
        step["memory"] = memory
    if readings["events"]:
        move = shift - start
        step["kernels"] = {
            "id": [int(i) for i in kernels.index], "phase": kernels["phase"].tolist(),
            "module": kernels["module_fqn"].tolist(),
            "start_us": (kernels["gpu_start_us"] + move).tolist(), "end_us": (kernels["gpu_end_us"] + move).tolist(),
            "duration_us": kernels["duration_us"].tolist()}
        step["groups"] = [
            {"phase": phase, "module": module, "kernel_ids": [int(i) for i in group.kernel_ids],
             "start_us": group.gpu_start_us + move, "end_us": group.gpu_end_us + move,
             "gpu_time_us": group.gpu_time_us, "estimated_flops": group.estimated_flops}
            for (phase, module), group in groups.iterrows()]
    return step
