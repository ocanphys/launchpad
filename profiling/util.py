"""The profiling run: `profile` runs a Profile's model under the profiler one step at a time, a full
training step or its forward alone, each phase timed by `phase`, and returns the JSON of the
profile's files. A ProfileJob runs it for a declared profile; `run_profile` runs it on a Modal GPU,
through main.profile, for one that is never stored."""

import json
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta

import modal
import torch
from torch.profiler import record_function

from artifacts.core.locate import locate
from artifacts.profiles import Profile
from config import APP_NAME, HEARTBEAT_SECONDS
from launcher.resources import resource_options
from profiling import profile_events


def run_profile(spec: Profile) -> Profile:
    """Returns `spec` holding its results, `profile` run on a Modal GPU, printing Modal's log of
    the call (its stdout and stderr) as it runs and, until it starts, Modal's messages about the
    function, such as a wait for a GPU. Nothing is stored: to keep a profile, declare it.

        result = run_profile(Profile(run_id="scratch", resources=Resources(gpu_type="T4"), model=model,
                                     optimizer=SGD(), batch_size=4, seed=0, warmup_steps=5, profile_steps=3,
                                     autocast=None, compile="default", kernel_events=True, module_scopes=False,
                                     peak_memory=True, trace_memory=True))
        result.display_steps(steps=[1])

    Args:
        spec: The profile to run, declared or not; its `resources` name the GPU.

    Returns:
        A copy of spec holding every file's JSON in memory, so its methods run as a bound profile's do.
    """
    function = modal.Function.from_name(APP_NAME, "profile")
    since = datetime.now(UTC)
    call = function.with_options(**resource_options(spec.resources)).spawn(spec)
    print(f"profile call {call.object_id}")
    started = threading.Event()

    def report_queue(since):  # a call waiting for a container is in the function's log, not the call's
        while not started.wait(HEARTBEAT_SECONDS):
            for entry in function.logs.fetch(since=since, source="system"):
                print(entry.message, end="")
                since = entry.timestamp + timedelta(microseconds=1)

    threading.Thread(target=report_queue, args=(since,), daemon=True).start()
    try:
        for entry in call.logs.stream():  # what the call writes from now on, until it finishes
            started.set()
            print(entry.message, end="")
    finally:
        started.set()
    return spec.with_results(call.get())


@contextmanager
def phase(name, spans):
    """Yields inside train::<name>, timed between synchronizes; spans[name] becomes its (start, end) in us on the unix clock."""
    with record_function(f"train::{name}"):
        torch.cuda.synchronize()
        start = time.time_ns() / 1e3
        yield
        torch.cuda.synchronize()
        spans[name] = start, time.time_ns() / 1e3


def profile(spec: Profile, log: logging.Logger | logging.LoggerAdapter) -> dict[str, str]:
    """Returns the JSON text of each of `spec`'s files: the measurements of `spec.profile_steps`
    steps, taken after `spec.warmup_steps` unmeasured ones.

    It needs a GPU: a ProfileJob runs it for a declared profile, and `run_profile` for one never
    stored. Every step, warmup or measured, runs on the same `batch_size` random sequences of
    the model's `sequence_length`, and every measured step ends in a synchronize. Warmup runs
    under `autocast` and `compile` too, so it prepares the same kernels. docs/profiling.md shows
    each file's shape.

    Args:
        spec: The profile to run. Its model and optimizer are declarations, built here on the GPU.
        log: Where its progress lines go: the call's logger in a job, stdout under run_profile.

    Returns:
        Each file's JSON text, keyed as `spec.files`: "report" always, "events" with
        `kernel_events` and "memory" with `trace_memory`.
    """
    model, batch_size = spec.model, spec.batch_size
    log.info(f"{type(model).__name__} on {torch.cuda.get_device_name()}, batch_size {batch_size}")
    network = model.build(torch.device("cuda"))
    tokens = torch.randint(
        0, model.vocab_size, (batch_size, model.sequence_length),
        generator=torch.Generator().manual_seed(spec.seed)).to("cuda")

    network.train()
    torch_optimizer = spec.optimizer.build(network.parameters())
    forward = network if spec.compile is None else torch.compile(network, mode=spec.compile)
    autocast_dtype = locate(spec.autocast) if spec.autocast else None
    spans = {}  # name: (start, end) of the latest step's phases, in us on the unix clock

    def forward_step():
        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=bool(spec.autocast)), phase("forward", spans):
            forward(tokens)

    def full_step():
        with phase("zero_grad", spans):
            torch_optimizer.zero_grad(set_to_none=True)
        # autocast covers forward and loss only, as a leg runs them
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=bool(spec.autocast)):
            with phase("forward", spans):
                logits = forward(tokens)
            with phase("loss", spans):
                # each position predicts the following input token
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
        with phase("backward", spans):
            loss.backward()
        with phase("optimizer_step", spans):
            torch_optimizer.step()

    run_step = {"forward": forward_step, "full": full_step}[spec.step_kind]
    for _ in range(spec.warmup_steps):
        run_step()
    # Finish warmup first, so none of its queued kernels land in the measured loop.
    torch.cuda.synchronize()
    log.info(f"warmup: {spec.warmup_steps} steps")

    # Gradients and optimizer state exist only after a full step, which warmup took; a forward
    # step leaves none. The last warmup step left its gradients allocated: nothing has zeroed them yet.
    measured_size = {
        "model": model.measure_size(network),
        "gradients": model.measure_gradient_size(network),
        "optimizer": spec.optimizer.measure_size(torch_optimizer),
    }

    steps, events, memory = [], [], []
    for index in range(spec.profile_steps):
        # Entered top to bottom and closed in reverse: module scopes close before the profiler stops.
        with (
            profile_events.capture_memory(spec.peak_memory, spec.trace_memory) as readings,
            profile_events.capture_profiler_events(spec.kernel_events) as profiled,
            profile_events.module_scopes(network) if spec.module_scopes else nullcontext(),
        ):
            begin = time.perf_counter()
            run_step()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - begin) * 1e3
        log.info(f"step {index}: {step_ms:.1f} ms")
        memory.append(readings.pop("snapshot", None))
        events.append(profiled)
        steps.append({"step_ms": step_ms, "phases_unix_us": {name: [start, end] for name, (start, end) in spans.items()},
                      **readings})

    log.info("done")
    parts = {
        "report": {
            "gpu": torch.cuda.get_device_name(),
            "device_vram_bytes": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
            "measured_size": measured_size,
            "steps": steps,
        },
        "events": events,
        "memory": memory,
    }
    return {part: json.dumps(parts[part], allow_nan=False) for part in spec.files}
