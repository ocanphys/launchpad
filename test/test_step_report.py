"""Tests for build_report and the display_steps that plots its report."""
import json

import pytest
import step_report
from profile_events import display_steps
from step_report import build_report
from test_profile_aggregation import rows  # noqa: F401  the shared event table fixture


@pytest.fixture
def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def captured_with(steps, **extra):
    """Return a capture result around steps, with every one-off field build_report reads."""
    return {
        "steps": steps, "device_vram": 2**30, "persistent_memory": 4, "step_flops": 1e12,
        "precision": "torch.bfloat16", "gpu": "NVIDIA L4",
        "parameters": {"batch_size": 8, "warmup_steps": 5, "profile_steps": len(steps), "autocast": "torch.bfloat16",
                       "kernel_events": False, "module_scopes": False, "peak": True, "trace": True, "wall_clock": True},
        "model_class": "Transformer", "model_parameters": {"vocab_size": 100, "d_model": 16, "dtype": "torch.float32"},
        "optimizer": "AdamW", "compile": True,
        "size_estimation": {"model": {"parameters": 10, "bytes": 40}, "optimizer": {"parameters": 20, "bytes": 80}},
        **extra,
    }


def step_readings(start_us=1.7e15, **extra):
    """A step with a trace and wall-clock phases: zero_grad frees 2 us in, forward allocates 6 us in."""
    note = lambda stage, name, t: {"stage": stage, "name": name, "device": 0, "time_us": t}
    trace = [{"action": "free_requested", "addr": 1, "size": 4, "time_us": start_us + 2},
             {"action": "alloc", "addr": 2, "size": 8, "time_us": start_us + 6}]
    return {"events": [], "step_ms": 20.0, "peak_allocated": 2**20, "peak_reserved": 2**21,
            "snapshot": {"segments": [{"device": 0, "allocated_size": 8, "total_size": 16}], "device_traces": [trace],
                         "external_annotations": []},
            "wall_clock_phase_spans": {"zero_grad": (start_us, start_us + 3), "forward": (start_us + 5, start_us + 9)},
            **extra}


def test_report_is_plain_json_and_holds_the_run_and_every_step():
    report = build_report(captured_with({0: step_readings(), 1: step_readings(1.7e15 + 5e4)}))
    assert json.loads(json.dumps(report, allow_nan=False)) == report  # no numpy, no NaN
    assert report["schema_version"] == 1
    run = report["run"]
    assert run["gpu"] == {"name": "NVIDIA L4", "spec_row": "L4", "vram_bytes": 2**30}
    assert run["model"] == {"class": "Transformer", "parameters": {"vocab_size": 100, "d_model": 16, "dtype": "torch.float32"},
                            "size": {"parameters": 10, "bytes": 40}}
    assert run["optimizer"] == {"class": "AdamW", "state_parameters": 20, "state_bytes": 80}
    assert run["capture"]["batch_size"] == 8 and run["capture"]["autocast"] == "torch.bfloat16" and run["compile"] is True
    assert run["precision"] == "torch.bfloat16" and run["peak"] == {"dtype": "bf16", "flops_per_s": 121e12}
    assert run["step_flops"] == 1e12
    # every FLOP number says where it comes from, and none comes from a hardware counter
    assert set(run["sources"]) == {"step_flops", "kernel_flops", "peak_flops_per_s", "achieved_flops_per_s", "mfu"}
    assert "algorithmic estimate" in run["sources"]["step_flops"] and "datasheet" in run["sources"]["peak_flops_per_s"]
    assert list(report["steps"]) == ["0", "1"]  # indexed by step number


def test_each_block_is_in_the_report_only_when_capture_recorded_it():
    report = build_report(captured_with({0: step_readings(), 1: {"events": [], "step_ms": 10.0}}))
    assert {"memory", "phases", "mfu", "start_unix_us"} <= set(report["steps"]["0"])
    assert "kernels" not in report["steps"]["0"] and "groups" not in report["steps"]["0"]  # no kernel_events
    assert set(report["steps"]["1"]) == {"step_ms", "mfu"}  # nothing else was recorded


def test_times_are_us_from_the_steps_start_with_the_unix_start_kept():
    step = build_report(captured_with({0: step_readings(1.7e15)}))["steps"]["0"]
    assert step["start_unix_us"] == 1.7e15
    assert step["memory"]["series"] == {"time_us": [2.0, 6.0], "allocated": [0, 8], "reserved": [16, 16]}
    assert step["phases"] == [{"name": "zero_grad", "start_us": 0.0, "end_us": 3.0, "duration_ms": 0.003},
                              {"name": "forward", "start_us": 5.0, "end_us": 9.0, "duration_ms": 0.004}]
    assert step["memory"]["peak_allocated_bytes"] == 2**20


def test_mfu_is_achieved_over_the_peak_of_the_precision_that_ran():
    mfu = build_report(captured_with({0: step_readings()}))["steps"]["0"]["mfu"]
    assert mfu == {"step_flops": 1e12, "step_ms": 20.0, "achieved_flops_per_s": pytest.approx(50e12),  # 1 TFLOP in 20 ms
                   "peak_flops_per_s": 121e12, "mfu": pytest.approx(50 / 121)}
    no_unit = build_report(captured_with({0: step_readings()}, gpu="Tesla T4"))["steps"]["0"]["mfu"]
    assert no_unit["peak_flops_per_s"] is None and no_unit["mfu"] is None  # a T4 has no bf16 unit, not a zero peak


def test_kernels_are_columnar_and_groups_are_records_on_the_steps_clock(rows, monkeypatch):
    monkeypatch.setattr(step_report, "events_to_rows", lambda events: rows)  # the fixture's rows stand in for events
    readings = step_readings(1.7e15, events=[object()], trace_start_us=1.7e15 - 100.0)
    step = build_report(captured_with({0: readings}))["steps"]["0"]
    kernels = step["kernels"]
    assert set(kernels) == {"id", "phase", "module", "start_us", "end_us", "duration_us"}
    assert len({len(column) for column in kernels.values()}) == 1  # one list per field, all as long as the table
    assert all(isinstance(group["kernel_ids"], list) and group["phase"] and group["module"] for group in step["groups"])
    assert sorted(i for group in step["groups"] for i in group["kernel_ids"]) == sorted(kernels["id"])
    # the profiler's times count from its trace start, 100 us before the step's start on the unix clock
    start = min(kernels["start_us"] + step["memory"]["series"]["time_us"] + [p["start_us"] for p in step["phases"]])
    assert start == pytest.approx(0.0)
    assert json.loads(json.dumps(step, allow_nan=False)) == step


def test_display_steps_plots_only_what_the_report_holds(plotting, monkeypatch):
    shown = []
    monkeypatch.setattr(plotting, "show", lambda: shown.append(plotting.gcf()))
    report = build_report(captured_with({0: step_readings(), 1: step_readings(1.7e15 + 5e4)}))
    display_steps(json.loads(json.dumps(report)))  # from the JSON alone
    (fig,) = shown
    first, second = fig.axes  # one memory panel per step, stacked
    assert [first.get_title(), second.get_title()] == ["step 0", "step 1"]
    for axis in (first, second):
        assert axis.lines[0].get_xdata().tolist() == [2.0, 6.0]  # the step's own clock
        assert [patch.get_x() for patch in axis.patches] == [0.0, 5.0]  # the phases, where they ran
    shown.clear()
    display_steps(report, steps=[1])
    assert [axis.get_title() for axis in shown[0].axes] == ["step 1"]
    shown.clear()
    display_steps(build_report(captured_with({0: {"events": [], "step_ms": 10.0}})))
    assert shown == []  # nothing recorded, nothing drawn


def test_display_steps_draws_phases_and_peaks_alone(plotting, monkeypatch):
    shown = []
    monkeypatch.setattr(plotting, "show", lambda: shown.append(plotting.gcf()))
    only_phases = {"events": [], "step_ms": 10.0, "wall_clock_phase_spans": {"forward": (1.7e15, 1.7e15 + 9)}}
    display_steps(build_report(captured_with({0: only_phases})))
    ((strip,),) = [fig.axes for fig in shown]
    title, info = strip.get_title().split("\n")  # no legend here, so the info joins the title
    assert title == "step 0" and "MFU: 82.6% (est.)" in info
    shown.clear()
    only_peaks = {"events": [], "step_ms": 10.0, "peak_allocated": 2**20, "peak_reserved": 2**21}
    display_steps(build_report(captured_with({0: only_peaks}, device_vram=2**40, persistent_memory=2**19)))
    ((memory_ax,),) = [fig.axes for fig in shown]
    assert [line.get_label() for line in memory_ax.lines] == [
        "peak allocated (1.0 MiB)", "peak reserved (2.0 MiB)", "total VRAM (1024.0 GiB)", "persistent (0.5 MiB)"]


def test_display_steps_shows_the_mfu_block_under_the_legend(plotting, monkeypatch):
    shown = []
    monkeypatch.setattr(plotting, "show", lambda: shown.append(plotting.gcf()))
    display_steps(build_report(captured_with({0: step_readings()})))
    (memory_ax, _) = shown[-1].axes if len(shown[-1].axes) == 2 else (shown[-1].axes[0], None)
    (note,) = [text for text in memory_ax.texts if "MFU" in text.get_text()]
    assert note.get_text().splitlines() == [  # a count is FLOPs, a rate is FLOPs/s, and the estimates say so
        "GPU: NVIDIA L4 [L4]", "step time: 20.0 ms", "step: 1.0 TFLOPs (est.)", "achieved: 50.0 TFLOPs/s (est.)",
        "peak: 121.0 TFLOPs/s bf16", "MFU: 41.3% (est.)"]
    assert "algorithmic estimates" in shown[-1].get_suptitle() or any(
        "algorithmic estimates" in text.get_text() for text in shown[-1].texts)  # the footnote
    assert note.xycoords is memory_ax.get_legend()  # anchored to the legend


def test_report_kernel_ids_are_the_event_rows_the_launch_chain_follows(rows, monkeypatch):
    from profile_events import aggregate_kernels, launch_chain

    monkeypatch.setattr(step_report, "events_to_rows", lambda events: rows)
    step = build_report(captured_with({0: step_readings(events=[object()], trace_start_us=1.7e15)}))["steps"]["0"]
    _, kernels, _, _ = aggregate_kernels(rows, depth=3)
    assert sorted(step["kernels"]["id"]) == sorted(kernels.index)  # the same rows aggregate_kernels indexes
    for kernel_id, module in zip(step["kernels"]["id"], step["kernels"]["module"]):
        assert kernels.loc[kernel_id, "module_fqn"] == module
        chain = launch_chain(rows, kernel_id, format=False)  # reaches the kernel from its CPU root
        assert chain.index[-1] == kernel_id and chain.loc[kernel_id, "activity_type"] == "kernel"
