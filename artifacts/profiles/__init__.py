"""Profiles: a few profiled steps of a model on one GPU, kept on the volume as JSON.

`Profile` declares one run: the model, the optimizer, the GPU, the kind of step and every switch
of `profiling.util.profile`. Its job writes report.json, plus events.json and memory.json when
their switches are on. A bound Profile, or `run_profile`'s result, which holds the same files in
memory, carries the analysis as methods. docs/profiling.md shows each file's shape.

Every process that decodes a manifest imports this module, so it imports nothing heavier than
pydantic; the methods import pandas, matplotlib and `profiling` when they are called.
"""

from __future__ import annotations

import json
import logging
import math
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import Field, PrivateAttr, model_validator

from artifacts.core.artifact import Artifact, Resources
from artifacts.core.SGD.optimizers import Optimizer
from models import Model

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


class Profile(Artifact):
    """A few profiled steps of `model` on one GPU, at profiles/<run_id>/<hash>. Every input that
    changes what is measured is a field, the GPU included; the tokens are random, drawn from
    `seed`, so a profile is built from nothing.

        Profile(run_id="scratch", resources=Resources(gpu_type="T4"), model=model, optimizer=SGD(),
                batch_size=4, seed=0, warmup_steps=5, profile_steps=3, autocast=None, compile="default",
                kernel_events=False, module_scopes=False, peak_memory=True, trace_memory=True).artifact_path
        -> PosixPath('profiles/scratch/6fe970b07b')

    The examples on its methods are one profile with every switch on, holding the files
    docs/profiling.md shows.
    """

    producer: ClassVar[str] = "artifacts.profiles.jobs.ProfileJob"

    run_id: str  # names a series of profiles: the first folder of each one's path
    resources: Resources  # the GPU it measures; allocated_resources is pinned to it
    model: Model
    optimizer: Optimizer
    # "full" is a next-token training step: zero_grad, forward, loss, backward, optimizer_step.
    # "forward" is the forward alone, without gradients.
    step_kind: Literal["full", "forward"] = "full"
    autocast: Literal["torch.bfloat16", "torch.float16"] | None  # what forward and loss run in; None is model.dtype
    compile: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] | None = None
    batch_size: int = Field(gt=0)
    seed: int  # seeds the random tokens every step trains on
    warmup_steps: int = Field(ge=0)  # unmeasured steps first, enough for a compiled step to finish compiling
    profile_steps: int = Field(gt=0)
    kernel_events: bool  # every CPU op and CUDA kernel, in events.json
    module_scopes: bool  # each module's work annotated, so kernels are attributed to submodules
    peak_memory: bool  # each step's peak allocated and reserved bytes, in report.json
    trace_memory: bool  # each step's allocator trace, in memory.json

    _parts: dict[str, Any] = PrivateAttr(default_factory=dict)  # each file's JSON, by part, once read
    _root: Path | None = PrivateAttr(default=None)  # the root a bound profile reads the rest from

    @model_validator(mode="before")
    @classmethod
    def _allocated_the_gpu_it_measures(cls, data: Any) -> Any:
        """Returns `data` with allocated_resources set to resources when it names none."""
        if isinstance(data, dict) and "allocated_resources" not in data and "resources" in data:
            return {**data, "allocated_resources": data["resources"]}
        return data

    @model_validator(mode="after")
    def _runs_as_declared(self) -> Self:
        """Refuses a profile that would not run as it says: allocated anything but its resources,
        on anything but one GPU, with module scopes under torch.compile, which they do not work under,
        or as its model cannot run (Model.check_artifact)."""
        if self.allocated_resources not in (None, self.resources):
            raise ValueError(f"allocated_resources ({self.allocated_resources}) must be resources ({self.resources}): "
                             f"a profile is allocated the GPU it measures")
        if self.resources.gpu_type is None or self.resources.gpu_count not in (None, 1):
            raise ValueError(f"resources must name one GPU, got {self.resources}")
        if self.compile is not None and self.module_scopes:
            raise ValueError("module_scopes do not work under torch.compile: declare module_scopes=False")
        self.model.check_artifact(self)
        return self

    @cached_property
    def parameters_hash(self) -> str:
        """Returns a short id for everything this profile measures: its parameters, run_id aside, so
        one profile under two run_ids shares it.

            Artifact.load("profiles/scratch/6fe970b07b").parameters_hash -> "6fe970b07b"
        """
        return self.digest({name: value for name, value in self.parameters().items() if name != "run_id"})

    @property
    def uid(self) -> str:
        return f"{self.run_id}-profile-{self.parameters_hash}"

    @property
    def artifact_path(self) -> Path:
        return Path("profiles") / self.run_id / self.parameters_hash

    @property
    def files(self) -> dict[str, str]:
        files = {"report": "report.json"}
        if self.kernel_events:
            files["events"] = "events.json"
        if self.trace_memory:
            files["memory"] = "memory.json"
        return files

    def _load(self, root: Path) -> None:
        """Artifact._load's hook: reads report.json. events.json and memory.json are read on first
        use, so binding a series reads its reports alone."""
        self._root = root
        self._parts = {"report": json.loads(self.paths(root)["report"].read_text())}

    def with_results(self, texts: dict[str, str]) -> Self:
        """Returns a copy of this profile holding `texts`, so its methods run without a volume.

        Args:
            texts: Each file's JSON text, keyed as `files`, as `profiling.util.profile` returns them.

        Returns:
            The copy, every file held in memory.
        """
        held = self.model_copy()
        held._parts = {part: json.loads(text) for part, text in texts.items()}
        return held

    def _part(self, part: str) -> Any:
        """Returns one file's JSON, read on first use from the root this profile was bound at."""
        if part not in self._parts:
            if part not in self.files:
                switch = {"events": "kernel_events", "memory": "trace_memory"}[part]
                raise ValueError(f"{self.uid} recorded no {part}: profile with {switch}=True")
            if self._root is None:
                raise ValueError(f"{self.uid} holds no results: bind it, or run it with run_profile")
            self._parts[part] = json.loads(self.paths(self._root)[part].read_text())
        return self._parts[part]

    @property
    def report(self) -> dict:
        """Returns report.json: the GPU, its VRAM, the measured sizes, and each step's wall time,
        phase spans and, with peak_memory, its peaks."""
        return self._part("report")

    @property
    def precision(self) -> str:
        """Returns the dtype the matmuls ran in: autocast, else the model's dtype."""
        return self.autocast or self.model.dtype

    @property
    def step_flops(self) -> float:
        """Returns the declaration's analytic FLOPs for one measured step: the model's step_flops at
        batch_size, or its forward_flops for a forward step; NaN for a model that counts neither."""
        count = getattr(self.model, "forward_flops" if self.step_kind == "forward" else "step_flops", None)
        return math.nan if count is None else count(self.batch_size)

    @property
    def peak_flops_per_s(self) -> float:
        """Returns the GPU's dense FLOPs per second at `precision`, from its datasheet
        (gpus/peaks.py); NaN when it has no unit for that precision or no row for the GPU."""
        from gpus.peaks import peak_tflops

        try:
            return float(peak_tflops(self.report["gpu"], self.precision)) * 1e12
        except KeyError:  # spec_row found no row for the GPU
            return math.nan

    @property
    def estimated_memory(self) -> dict[str, int]:
        """Returns the bytes the declaration estimates for a full step: "persistent" (weights,
        gradients and optimizer state) and "activation", each only if the model and optimizer
        can estimate it, and neither for a forward step."""
        estimated = {}
        if self.step_kind != "full":
            return estimated
        if hasattr(self.model, "estimate_size") and hasattr(self.optimizer, "estimate_size"):
            weights = self.model.estimate_size()  # gradients are as large as the weights
            estimated["persistent"] = 2 * weights["bytes"] + self.optimizer.estimate_size(weights)["bytes"]
        if hasattr(self.model, "activation_memory"):
            estimated["activation"] = self.model.activation_memory(self.batch_size, autocast=bool(self.autocast))
        return estimated

    def events(self, step: int) -> pd.DataFrame:
        """Returns one step's events as events_to_rows' table, each GPU event linked to the CPU
        event that launched it.

            rows = profile.events(0)
            rows.loc[rows.loc[3, "launcher_id"], "name"] -> 'cudaLaunchKernel'  # the launch call that queued kernel 3

        Args:
            step: The step's number, from 0.

        Returns:
            One row per event, indexed by its row ID.
        """
        from profiling.profile_events import events_to_rows

        return events_to_rows(self._part("events")[step]["events"])

    def aggregate_kernels(self, step: int, depth: int | None = None) -> tuple:
        """Returns aggregate_kernels of one step's events: its kernels grouped by phase and module,
        with the FLOPs of each group.

            groups, kernels, unattributed_kernels, flop_sources = profile.aggregate_kernels(0)
            groups["total_est_flops"].sum() -> 201326592.0

        Args:
            step: The step's number, from 0.
            depth: How many FQN components a group keeps, root included; None keeps them all.

        Returns:
            (groups, kernels, unattributed_kernels, flop_sources), as aggregate_kernels returns them.
        """
        from profiling.profile_events import aggregate_kernels

        return aggregate_kernels(self.events(step), depth)

    def memory_over_time(self, step: int) -> pd.DataFrame:
        """Returns the allocated and reserved bytes after each allocator trace entry of one step,
        indexed by time_us on the unix clock.

            profile.memory_over_time(0)["allocated"].tolist() -> [12866048, 12866560, 12866048]

        Args:
            step: The step's number, from 0.

        Returns:
            One row per entry that changes either total.
        """
        from profiling.profile_events import memory_over_time

        return memory_over_time(self._part("memory")[step])[0]

    def allocations_by_label(self, step: int) -> pd.DataFrame:
        """Returns one row per allocation in one step's allocator trace, labelled with the innermost
        record_function scope open when it was made.

            profile.allocations_by_label(0)["label"].tolist() -> ['<none>']

        Args:
            step: The step's number, from 0.

        Returns:
            The allocations: addr, size, start_us, end_us and label.
        """
        from profiling.profile_events import allocations_by_label

        return allocations_by_label(self._part("memory")[step])

    def summary(self, agg: str | None = None, verbose: bool = False) -> pd.DataFrame:
        """Returns one row per measured step, or with `agg` one row for the profile: its parameters,
        flattened ("model.d_model"), autocast filled with model.dtype and compile with "no" where
        they are None, what it measured, and what the declaration estimates and derives from that.

            profile.summary().loc[0, "mfu"] -> 0.1308677610611617

        Measured: gpu, device_vram_bytes, persistent_measured_bytes (weights, gradients and
        optimizer state), step_ms, each phase's <phase>_ms and, with peak_memory, the peaks.
        Estimated and derived: persistent_estimated_bytes and activation_estimated_bytes where the
        model and optimizer can estimate them, model_parameter_count (N, without the input
        embedding), step_flops, peak_flops_per_s, cost_per_ms (Modal's base price of
        resources.gpu_type, gpus/pricing.py, in dollars), achieved_flops_per_s (step_flops over
        step_ms), mfu and flops_per_dollar. A series is one table:
        `pd.concat([p.summary("mean") for p in profiles])`. Every column is always there: a value the
        profile did not record, or the model, the datasheet or the price table cannot give, is NaN,
        and with verbose summary logs which columns hold one.

        `agg` reduces the steps' timings and takes the max of their peaks; the rates are then
        derived from the reduced step_ms, so mfu is the mean step's, not the mean of each step's.

        Args:
            agg: A pandas reduction of the steps ("mean", "median", "min"); None keeps a row per step.
            verbose: Whether to log a warning naming the columns that hold NaN.

        Returns:
            The rows: one per step, step numbers in "step", or one row without "step".
        """
        import pandas as pd

        from gpus.pricing import gpu_price_per_s

        report, model, estimated = self.report, self.model, self.estimated_memory
        steps = pd.DataFrame([{
            "step_ms": readings["step_ms"],
            **{f"{name}_ms": (end - start) / 1e3 for name, (start, end) in readings["phases_unix_us"].items()},
            **{key: readings.get(key, math.nan) for key in ("peak_allocated_bytes", "peak_reserved_bytes")},
        } for readings in report["steps"]])
        if agg is not None:
            steps = steps.agg({column: "max" if column.startswith("peak_") else agg for column in steps}).to_frame().T
        parameters = {**pd.json_normalize(self.parameters()).iloc[0].to_dict(),
                      "autocast": self.precision, "compile": self.compile or "no"}
        given = {**parameters, "gpu": report["gpu"], "device_vram_bytes": report["device_vram_bytes"],
                 "persistent_measured_bytes": sum(size["bytes"] for size in report["measured_size"].values())}
        table = pd.concat([pd.DataFrame([given] * len(steps)), steps], axis=1).assign(
            persistent_estimated_bytes=estimated.get("persistent", math.nan),
            activation_estimated_bytes=estimated.get("activation", math.nan),
            model_parameter_count=model.parameter_count(embedding=False) if hasattr(model, "parameter_count") else math.nan,
            step_flops=self.step_flops, peak_flops_per_s=self.peak_flops_per_s,
            cost_per_ms=gpu_price_per_s.get(self.resources.gpu_type, math.nan) / 1e3)
        table["achieved_flops_per_s"] = table["step_flops"] / (table["step_ms"] / 1e3)
        table["mfu"] = table["achieved_flops_per_s"] / table["peak_flops_per_s"]
        table["flops_per_dollar"] = table["step_flops"] / (table["cost_per_ms"] * table["step_ms"])
        missing = [column for column in table if column not in parameters and table[column].isna().any()]
        if verbose and missing:
            log.warning(f"{self.uid}: no {', '.join(missing)}; NaN in the summary")
        if agg is None:
            table.insert(0, "step", table.index)
        return table

    def display_steps(self, steps: list[int] | None = None, depth: int = 3) -> None:
        """Displays each step as a timeline, stacked top to bottom in one figure: memory with its
        peaks and totals, a row per kernel group with its estimated FLOPs, and kernel durations,
        each phase shaded behind them.

            profile.display_steps(steps=[1])

        A panel appears only for what the profile recorded: kernels with kernel_events, the memory
        curves with trace_memory, the peak lines with peak_memory. Times are us from the start of
        the step's first phase. Every FLOP figure is an algorithmic estimate, and a footnote says so.

        Args:
            steps: The step numbers to show; None shows every step.
            depth: How many FQN components a kernel group keeps, as aggregate_kernels takes it.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        from gpus.peaks import PRECISIONS, spec_row
        from profiling.profile_events import draw_timeline, timeline_height

        report, summary = self.report, self.summary()
        estimated = self.estimated_memory
        estimated_total = estimated["persistent"] + estimated["activation"] if len(estimated) == 2 else None
        timelines = {}
        for step in range(len(report["steps"])) if steps is None else steps:
            readings, row = report["steps"][step], summary.loc[step]
            phases = pd.DataFrame([(name, start, end) for name, (start, end) in readings["phases_unix_us"].items()],
                                  columns=["name", "start_us", "end_us"])
            zero = phases["start_us"].min()
            timeline = {"phases": phases.assign(start_us=phases["start_us"] - zero, end_us=phases["end_us"] - zero)}
            if self.kernel_events:
                groups, kernels, _, _ = self.aggregate_kernels(step, depth)
                move = self._part("events")[step]["trace_start_unix_us"] - zero  # event times count from the trace's start
                timeline["groups"] = groups.assign(gpu_start_us=groups["gpu_start_us"] + move,
                                                   gpu_end_us=groups["gpu_end_us"] + move)
                timeline["kernels"] = kernels.assign(gpu_start_us=kernels["gpu_start_us"] + move,
                                                     gpu_end_us=kernels["gpu_end_us"] + move)
            if self.trace_memory:
                memory = self.memory_over_time(step)
                timeline["memory"] = memory.set_axis(memory.index - zero)
            if self.peak_memory:
                timeline["peaks"] = {"allocated": readings["peak_allocated_bytes"],
                                     "reserved": readings["peak_reserved_bytes"]}
            timeline["info"] = [
                f"GPU: {row['gpu']} [{spec_row(row['gpu'])}]",
                f"step time: {row['step_ms']:.1f} ms",
                f"step: {_with_prefix(row['step_flops'], 'FLOPs')} (est.)",
                f"achieved: {_with_prefix(row['achieved_flops_per_s'], 'FLOPs/s')} (est.)",
                "peak: n/a" if math.isnan(row["peak_flops_per_s"])
                else f"peak: {_with_prefix(row['peak_flops_per_s'], 'FLOPs/s')} {PRECISIONS[self.precision]}",
                "MFU: n/a" if math.isnan(row["mfu"]) else f"MFU: {row['mfu']:.1%} (est.)",
            ]
            timelines[step] = timeline
        heights = [timeline_height(timeline.get("groups"), timeline.get("memory"), timeline["phases"], timeline.get("peaks"))
                   for timeline in timelines.values()]
        # ioff prevents ipympl from displaying once on creation and again on show().
        with plt.ioff():
            fig = plt.figure(figsize=(12, sum(heights)), layout="constrained")
            fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)  # tight, in inches and fractions
            for area, (step, timeline) in zip(fig.add_gridspec(len(heights), 1, height_ratios=heights), timelines.items()):
                draw_timeline(fig, area, title=f"step {step}", total_memory=report["device_vram_bytes"],
                              persistent_memory=summary.loc[step, "persistent_measured_bytes"],
                              estimated_total_memory=estimated_total, **timeline)
            fig.supxlabel("(est.): algorithmic estimates from shapes (the model's step_flops, the profiler's per-op "
                          "formulas), not hardware counters. Peak: vendor datasheet, dense.", fontsize=7, color="0.4")
        plt.show()


def _with_prefix(value: float, unit: str) -> str:
    """Returns value in a unit with an SI prefix that keeps it readable.

        _with_prefix(228.8e9, "FLOPs") -> "228.8 GFLOPs"

    Args:
        value: The number.
        unit: Its unit.

    Returns:
        The value with one decimal and its prefixed unit.
    """
    for exponent, prefix in ((15, "P"), (12, "T"), (9, "G"), (6, "M"), (3, "K")):
        if abs(value) >= 10**exponent:
            return f"{value / 10**exponent:,.1f} {prefix}{unit}"
    return f"{value:,.1f} {unit}"
