"""MFU and peak memory of a leg's steps on CUDA, as step metrics once per
window of steps, without the loop ever waiting on the GPU.
"""

import math
from typing import TYPE_CHECKING

import torch

from gpus.peaks import PRECISIONS, gpu_specs, spec_row

if TYPE_CHECKING:
    from artifacts.core.SGD.training import Model
    from system.runtime import Worker


class GpuMonitor:
    """Measures "mfu", and "peak_allocated_pct" and "peak_reserved_pct" as
    percentages of the device's VRAM, for each window of `every` steps;
    measures nothing off CUDA.

    A window's peaks are final when it closes, since the allocator counts on the
    host; its time is final once the GPU reaches the event that closed it, which
    later calls to `record` poll for, so a window's metrics land together a few
    steps after it closes. "mfu" is never measured when the GPU, or its unit for
    the model's dtype, is not in gpus/peaks.py.
    """

    def __init__(self, model: "Model", batch_size: int, every: int, step: int, worker: "Worker"):
        self.device = torch.device(model.device)
        self.every = every
        self.worker = worker
        self.window = None
        if self.device.type != "cuda":
            return
        try:
            gpu = spec_row(torch.cuda.get_device_name(self.device))
            worker.log.info(f"Starting training on {gpu}")
        except KeyError as error:
            worker.log.warning(f"MFU unknown: {error}")
            gpu = "unknown"
        self.peak_flops = gpu_specs.get(gpu, {}).get(PRECISIONS.get(model.dtype), math.nan) * 1e12
        self.step_flops = model.step_flops(batch_size)
        self.vram = torch.cuda.get_device_properties(self.device).total_memory
        self.start, self.start_step = torch.cuda.Event(enable_timing=True), step
        self.start.record()
        torch.cuda.reset_peak_memory_stats(self.device)

    def record(self, step: int) -> dict[str, float]:
        """Returns the last closed window's peaks and mfu at the first `step` the
        GPU has run it, empty on every other step, and closes a new window on
        every `every`th step.

        The same values go to worker.progress: "mfu", and "peak_alloc" as the
        peak allocated share of the device's VRAM, e.g. "37% (80 GB)"."""
        if self.device.type != "cuda":
            return {}
        metrics = {}
        if self.window is not None and self.window[1].query():
            start, end, steps, metrics = self.window
            self.window = None
            if not math.isnan(self.peak_flops):
                metrics["mfu"] = round(steps * self.step_flops / (start.elapsed_time(end) / 1e3) / self.peak_flops, 3)
                self.worker.progress["mfu"] = metrics["mfu"]
            self.worker.progress["peak_alloc"] = f"{metrics['peak_allocated_pct']:.0f}% ({self.vram / 2**30:.0f} GB)"
        if step % self.every == 0:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            peaks = {
                "peak_allocated_pct": round(100 * torch.cuda.max_memory_allocated(self.device) / self.vram, 1),
                "peak_reserved_pct": round(100 * torch.cuda.max_memory_reserved(self.device) / self.vram, 1),
            }
            torch.cuda.reset_peak_memory_stats(self.device)
            self.window = (self.start, end, step - self.start_step, peaks)
            self.start, self.start_step = end, step
        return metrics
