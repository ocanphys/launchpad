"""The profiling run: `profile` trains a model under the profiler, and `phase` is what the step
it is passed is built from. `run_profile` runs `profile` on a Modal GPU through main.profile."""

import time
from contextlib import contextmanager, nullcontext

import modal
import torch
from torch.profiler import record_function

from artifacts.core.locate import locate
from config import APP_NAME, HEARTBEAT_SECONDS
from launcher.resources import resource_options
from profiling import profile_events
from system.lease_protocol import call_logs
from system.logs import AMBIENT, LIVE, PROFILER, call_logger, channel, current_call_id


def run_profile(resources, **parameters) -> dict:
    """Returns main.profile's result on the GPU `resources` requests, printing its log rows as the call
    publishes them to the launcher's live channels."""
    call = modal.Function.from_name(APP_NAME, "profile").with_options(**resource_options(resources)).spawn(**parameters)
    print(f"profile call {call.object_id}")
    shown = dict.fromkeys((PROFILER, AMBIENT), 0)  # the only sources a profile call writes

    def show():
        for source in shown:
            rows = call_logs.get(channel(call.object_id, LIVE, source)) or []
            for row in rows[shown[source]:]:
                print(f"{row['level']} {row['logger']}: {row['msg']}")
            shown[source] = len(rows)

    while True:
        try:
            result = call.get(timeout=HEARTBEAT_SECONDS)
        except TimeoutError:
            show()
            continue
        show()  # main.profile's stop_logging publishes the last rows before it returns
        return result


@contextmanager
def phase(name, spans):
    """Yields inside train::<name>, timed between synchronizes; spans[name] becomes its (start, end) in us on the unix clock."""
    with record_function(f"train::{name}"):
        torch.cuda.synchronize()
        start = time.time_ns() / 1e3
        yield
        torch.cuda.synchronize()
        spans[name] = start, time.time_ns() / 1e3


def profile(
    model_declaration, training_parameters, profiling_step_loop, batch_size, warmup_steps, profile_steps, autocast,
    kernel_events, module_scopes, peak_memory, trace_memory,
) -> dict:
    """Return the measured loop's results after a plain warmup loop over the same model and optimizer.

    It needs a GPU: notebooks call it through run_profile.
    model_declaration is the Model to build; it decides whether the model is compiled. Of
    training_parameters, the optimizer, optimizer_parameters and seed are what profile trains with;
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
    log = call_logger("profile", PROFILER, current_call_id())
    log.info(f"{type(model_declaration).__name__} on {torch.cuda.get_device_name()}, batch_size {batch_size}")
    model = model_declaration.build()
    tokens = torch.randint(
        0, model_declaration.vocab_size, (batch_size, model_declaration.sequence_length),
        generator=torch.Generator().manual_seed(training_parameters.seed)).to(model_declaration.device)

    model.train()
    optimizer = locate(training_parameters.optimizer)(
        model.parameters(), **training_parameters.optimizer_parameters.model_dump())
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
    log.info(f"warmup: {warmup_steps} steps")

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
        log.info(f"step {step}: {step_ms:.1f} ms")
        steps[step] = {
            **profiled, **readings, "step_ms": step_ms,
            "wall_clock_phase_times": {name: (end - start) / 1e3 for name, (start, end) in spans.items()},
            "wall_clock_phase_spans": dict(spans),
        }

    log.info("done")
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
        "model_parameters": model_declaration.model_dump(),
        "optimizer": type(optimizer).__name__,
    }
