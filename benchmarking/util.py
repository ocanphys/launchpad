"""The profiling capture: `capture` trains a model under the profiler on a Modal GPU, and `phase` is
what the step it is passed is built from."""

import dataclasses
import time
from contextlib import contextmanager, nullcontext

import modal
import profile_events
import torch
from torch.profiler import record_function

import main
from artifacts.core.locate import locate

app = modal.App("profile-events")
# The worker image, so its torch is the uv.lock version that unpickles the events locally. Modal adds
# this module itself; the container also imports main, for this image, and profile_events.
image = main.worker_image.add_local_python_source("main", "profile_events")


@contextmanager
def phase(name, spans):
    """Yields inside train::<name>, timed between synchronizes; spans[name] becomes its (start, end) in us on the unix clock."""
    with record_function(f"train::{name}"):
        torch.cuda.synchronize()
        start = time.time_ns() / 1e3
        yield
        torch.cuda.synchronize()
        spans[name] = start, time.time_ns() / 1e3


# No gpu here: the caller requests it with Resources, as a production run does.
@app.function(image=image, timeout=600)
def capture(
    model_declaration, training_parameters, profiling_step_loop, batch_size, warmup_steps, profile_steps, autocast,
    kernel_events, module_scopes, peak_memory, trace_memory,
) -> dict:
    """Return the measured loop's results after a plain warmup loop over the same model and optimizer.

    It needs a GPU, so call it as capture.with_options(**resource_options(Resources(gpu_type="L4"))).remote(...).
    model_declaration is the Model to build; it decides whether the model is compiled. Of
    training_parameters, the optimizer, optimizer_parameters and seed are what capture trains with;
    the rest are not used. Each step calls profiling_step_loop(model, optimizer, tokens, autocast=...,
    spans=...) on the same batch_size random sequences of the model's sequence_length, and it wraps each
    phase in phase(name, spans). autocast is the dtype forward and loss run in, or None; warmup and the
    measured loop share it, so warmup prepares the same kernels. The switches pick the instruments
    around every measured step: kernel_events (torch profiler), module_scopes (forward::/backward::
    annotations), peak_memory (peak allocated and reserved bytes) and trace_memory (the allocator trace).
    Every step has "wall_clock_phase_times", each phase's duration in ms, and "wall_clock_phase_spans", its
    (start, end) in us on the unix clock, both keyed by phase name. Every measured step ends in a
    synchronize. The result has "steps" ({step: {"events",
    "trace_start_us", and capture_memory's readings}}), "size_estimation" ({"model", "optimizer"}, each with
    parameters and bytes), "persistent_memory" (bytes of weights, gradients and optimizer state),
    "device_vram" (total bytes on the GPU), "step_flops" (the declaration's step_flops for
    batch_size, an analytic estimate in FLOPs), "precision" (str of the autocast dtype, else of the model's dtype)
    and "gpu" (torch.cuda.get_device_name()). Each step also has "step_ms", the wall time of the
    step, instruments included. The last five are one-off quantities, measured once per call.
    It also returns what it ran with, for step_report.build_report: "parameters" (its own
    arguments), "model_class" and "model_parameters" (the declaration's class name and fields)
    and "optimizer" (its class name).
    """
    model = model_declaration.build()
    tokens = torch.randint(
        0, model_declaration.vocab_size, (batch_size, model_declaration.sequence_length),
        generator=torch.Generator().manual_seed(training_parameters.seed)).to(model_declaration.device)

    model.train()
    optimizer = locate(training_parameters.optimizer)(
        model.parameters(), **dataclasses.asdict(training_parameters.optimizer_parameters))
    spans = {}  # name: (start, end) of the latest step's phases, in us on the unix clock

    def size_of(tensors):
        tensors = list(tensors)
        return {"parameters": sum(t.numel() for t in tensors),
                "bytes": sum(t.numel() * t.element_size() for t in tensors)}

    def run_step():
        profiling_step_loop(model, optimizer, tokens, autocast=autocast, spans=spans)

    for _ in range(warmup_steps):
        run_step()
    # Finish warmup first, so none of its queued kernels land in the measured loop.
    torch.cuda.synchronize()

    # Optimizer state exists only after the first step, which warmup took.
    size_estimation = {
        "model": size_of(model.parameters()),
        "optimizer": size_of(t for state in optimizer.state.values() for t in state.values() if torch.is_tensor(t)),
    }

    steps = {}
    for step in range(profile_steps):
        # Entered top to bottom and closed in reverse: module scopes close before the profiler stops.
        with (
            profile_events.capture_memory(peak_memory, trace_memory) as readings,
            profile_events.capture_profiler_events(kernel_events) as profiled,
            profile_events.module_scopes(model) if module_scopes else nullcontext(),
        ):
            begin = time.perf_counter()
            run_step()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - begin) * 1e3
        steps[step] = {
            **profiled, **readings, "step_ms": step_ms,
            "wall_clock_phase_times": {name: (end - start) / 1e3 for name, (start, end) in spans.items()},
            "wall_clock_phase_spans": dict(spans),
        }

    return {
        "steps": steps,
        "size_estimation": size_estimation,
        # Gradients are the size of the weights, so persistent memory counts the model twice.
        "persistent_memory": 2 * size_estimation["model"]["bytes"] + size_estimation["optimizer"]["bytes"],
        "device_vram": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        "step_flops": model_declaration.step_flops(batch_size),
        "precision": str(autocast or model_declaration.dtype),  # the dtype the matmuls ran in
        "gpu": torch.cuda.get_device_name(),
        "parameters": {
            "batch_size": batch_size, "warmup_steps": warmup_steps, "profile_steps": profile_steps,
            "autocast": str(autocast) if autocast else None, "kernel_events": kernel_events,
            "module_scopes": module_scopes, "peak": peak_memory, "trace": trace_memory,
        },
        "model_class": type(model_declaration).__name__,
        "model_parameters": dataclasses.asdict(model_declaration),
        "optimizer": type(optimizer).__name__,
    }
