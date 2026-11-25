"""Profile: its identity and checks, the job that writes its files, and the analysis a bound
profile, or run_profile's result, carries. Nothing here runs on a GPU: the files are written by hand."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pandas as pd
import pytest
from pydantic import ValidationError

from artifacts.core.artifact import Artifact, Resources
from artifacts.core.manifest import manifest_json
from artifacts.core.storage import write_manifest
from artifacts.profiles import Profile, _with_prefix, jobs
from artifacts.profiles.jobs import ProfileJob
from gpus.pricing import gpu_price_per_s
from models.mamba import Mamba
from models.transformer import Transformer
from models.transformerFA import TransformerFA
from profiling.util import run_profile
from system.runtime import Worker
from training.optimizers import SGD, AdamW

MODEL = Transformer(vocab_size=64, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2, rope_theta=10000)
FIELDS = {"run_id": "sweep", "resources": Resources(gpu_type="L4"), "model": MODEL, "optimizer": AdamW(), "batch_size": 4,
          "seed": 0, "warmup_steps": 1, "profile_steps": 2, "autocast": "torch.bfloat16", "kernel_events": True,
          "module_scopes": True, "peak_memory": True, "trace_memory": True}
START = 1.7e15  # each step's first phase starts here, on the unix clock; the second step 50 ms later


def declared(**changes) -> Profile:
    return Profile(**FIELDS | changes)


def readings(start):
    return {"step_ms": 20.0, "phases_unix_us": {"zero_grad": [start, start + 3], "forward": [start + 5, start + 9]},
            "peak_allocated_bytes": 2**20, "peak_reserved_bytes": 2**21}


def events(trace_start):
    """One forward: train::forward > forward::model > aten::mm > cudaLaunchKernel, which queued kernel 4."""
    return {"trace_start_unix_us": trace_start, "events": {
        "id": [1, 2, 3, 100, 100],
        "node_id": [-1] * 5,
        "name": ["train::forward", "forward::model", "aten::mm", "cudaLaunchKernel", "gemm"],
        "activity_type": ["user_annotation", "user_annotation", "cpu_op", "cuda_runtime", "kernel"],
        "device_type": ["DeviceType.CPU"] * 4 + ["DeviceType.CUDA"],
        "start_us": [0.0, 1.0, 2.0, 3.0, 5.0],
        "end_us": [10.0, 9.0, 8.0, 4.0, 7.0],
        "cpu_parent": [None, 0, 1, 2, None],
        "linked_correlation_id": [0, 0, 0, 3, 3],
        "flops": [0, 0, 64, 0, 0],
    }}


def snapshot(start):
    """zero_grad frees 4 bytes 2 us in, and forward, open from 5 us, allocates 8 bytes 6 us in."""
    return {"segments": [{"device": 0, "allocated_size": 8, "total_size": 16}],
            "device_traces": [[{"action": "free_requested", "addr": 1, "size": 4, "time_us": start + 2},
                               {"action": "alloc", "addr": 2, "size": 8, "time_us": start + 6}]],
            "external_annotations": [{"device": 0, "name": "train::forward", "stage": "START", "time_us": start + 5}]}


REPORT = {"gpu": "NVIDIA L4", "device_vram_bytes": 2**30,
          "measured_size": {"model": {"parameters": 10, "bytes": 40}, "gradients": {"parameters": 10, "bytes": 40},
                            "optimizer": {"parameters": 20, "bytes": 80}},
          "steps": [readings(START), readings(START + 5e4)]}
TEXTS = {"report": json.dumps(REPORT),
         "events": json.dumps([events(START - 100), events(START + 5e4 - 100)]),
         "memory": json.dumps([snapshot(START), snapshot(START + 5e4)])}


def worker() -> Worker:
    """A real Worker with the lease no-opped, so the job publishes as it would on Modal."""
    return Worker(artifact_path="", call_id="local", log=Mock(), confirm_lease=Mock(), progress={})


@pytest.fixture
def shown(monkeypatch):
    """The figures plt.show would display, in order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    yield figures
    plt.close("all")


def test_a_profile_comes_back_from_its_manifest_with_its_classes():
    profile = declared()
    back = Artifact.from_manifest(json.loads(manifest_json(profile.to_manifest())))
    assert back == profile and back.artifact_path == profile.artifact_path
    assert type(back.model) is Transformer and type(back.optimizer) is AdamW
    assert back.allocated_resources == back.resources == Resources(gpu_type="L4")


def test_what_it_measures_is_its_identity():
    profile = declared()
    assert profile.artifact_path == Path("profiles/sweep") / profile.parameters_hash
    assert profile.uid == f"sweep-profile-{profile.parameters_hash}"
    for change in ({"resources": Resources(gpu_type="A10")}, {"optimizer": SGD()}, {"autocast": None},
                   {"step_kind": "forward"}, {"seed": 1}):
        assert declared(**change).artifact_path != profile.artifact_path, change
    other = declared(run_id="other")  # the same profile in another series
    assert other.parameters_hash == profile.parameters_hash and other.artifact_path != profile.artifact_path
    assert declared() == profile and hash(declared()) == hash(profile)


def test_a_profile_is_allocated_the_one_gpu_it_measures_and_refuses_what_would_not_run():
    assert declared().allocated_resources == Resources(gpu_type="L4")
    with pytest.raises(ValidationError, match="allocated the GPU it measures"):
        declared(allocated_resources=Resources(gpu_type="A100"))
    for resources in (Resources(), Resources(gpu_type="L4", gpu_count=2)):
        with pytest.raises(ValidationError, match="one GPU"):
            declared(resources=resources)
    with pytest.raises(ValidationError, match="module_scopes do not work under torch.compile"):
        declared(compile="default")
    assert declared(compile="default", module_scopes=False).compile == "default"
    flash = TransformerFA(**dict(MODEL))
    with pytest.raises(ValidationError, match="flash attention takes fp16 or bf16, not torch.float32"):
        declared(model=flash, autocast=None)
    assert declared(model=flash).precision == "torch.bfloat16"
    assert declared(model=TransformerFA(**dict(MODEL) | {"dtype": "torch.bfloat16"}), autocast=None).autocast is None
    for change in ({"autocast": "torch.float32"}, {"compile": "fast"}, {"batch_size": 0}, {"profile_steps": 0},
                   {"warmup_steps": -1}):
        with pytest.raises(ValidationError):
            declared(**change)


def test_its_files_follow_its_switches():
    assert declared().files == {"report": "report.json", "events": "events.json", "memory": "memory.json"}
    assert declared(kernel_events=False, trace_memory=False).files == {"report": "report.json"}


def test_the_job_publishes_each_file_and_bind_reads_the_rest_on_first_use(tmp_path, monkeypatch):
    profile = declared()
    ran = []
    monkeypatch.setattr(jobs, "profile", lambda spec, log: ran.append((spec, log)) or TEXTS)
    write_manifest(tmp_path, profile.artifact_path, profile.to_manifest())
    job_worker = worker()
    ProfileJob(profile).run(tmp_path, job_worker)
    assert ran == [(profile, job_worker.log)]
    assert profile.status(tmp_path).complete
    assert {path.name for path in (tmp_path / profile.artifact_path).iterdir()} == {
        "manifest.json", "report.json", "events.json", "memory.json"}  # no temporary file left behind
    bound = profile.bind(tmp_path)
    assert bound.report == REPORT
    changed = json.loads(TEXTS["events"])
    changed[0]["events"]["name"][4] = "read after bind"
    (tmp_path / profile.artifact_path / "events.json").write_text(json.dumps(changed))
    assert bound.events(0).loc[4, "name"] == "read after bind"  # bind read report.json alone
    held = profile.with_results(TEXTS)  # what run_profile returns for the same files
    pd.testing.assert_frame_equal(bound.summary(), held.summary())


def test_methods_say_what_a_profile_lacks():
    profile = declared(kernel_events=False, trace_memory=False)
    held = profile.with_results({"report": TEXTS["report"]})
    with pytest.raises(ValueError, match="holds no results: bind it, or run it with run_profile"):
        profile.summary()  # with_results returned a copy and left the declaration as it was
    with pytest.raises(ValueError, match="kernel_events=True"):
        held.events(0)
    with pytest.raises(ValueError, match="trace_memory=True"):
        held.memory_over_time(0)


def test_events_kernels_and_memory_come_from_each_steps_files():
    held = declared().with_results(TEXTS)
    assert held.events(0).loc[4, "launcher_id"] == 3  # the kernel, linked to its launch call by correlation id
    groups, _, unattributed, sources = held.aggregate_kernels(1)
    assert groups.loc[("forward", "model"), "kernel_ids"] == [4] and groups["total_est_flops"].sum() == 64
    assert unattributed.empty and sources.loc[2, "status"] == "allocated"
    memory = held.memory_over_time(0)
    assert memory.index.tolist() == [START + 2, START + 6]  # the unix clock
    assert memory["allocated"].tolist() == [0, 8] and memory["reserved"].tolist() == [16, 16]
    assert held.allocations_by_label(0)["label"].tolist() == ["train::forward"]


def test_the_summary_is_a_row_per_step_of_parameters_measurements_and_rates():
    summary = declared().with_results(TEXTS).summary()
    assert summary["step"].tolist() == [0, 1]
    row = summary.loc[0]
    assert row["model.class"] == "models.transformer.Transformer" and row["model.d_model"] == 16
    assert row["optimizer.class"] == "training.optimizers.AdamW" and row["resources.gpu_type"] == "L4"
    assert row["autocast"] == "torch.bfloat16" and row["compile"] == "no"
    assert declared(autocast=None).with_results(TEXTS).summary().loc[0, "autocast"] == MODEL.dtype  # None runs in it
    assert row["gpu"] == "NVIDIA L4" and row["device_vram_bytes"] == 2**30
    assert row["persistent_measured_bytes"] == 160  # weights, gradients and optimizer state
    assert row[["step_ms", "zero_grad_ms", "forward_ms"]].tolist() == [20.0, 0.003, 0.004]
    assert row[["peak_allocated_bytes", "peak_reserved_bytes"]].tolist() == [2**20, 2**21]
    weights = MODEL.estimate_size()
    assert row["persistent_estimated_bytes"] == 2 * weights["bytes"] + AdamW().estimate_size(weights)["bytes"]
    assert row["activation_estimated_bytes"] == MODEL.activation_memory(4, autocast=True)
    step_flops = MODEL.step_flops(4)
    assert row["step_flops"] == step_flops and row["achieved_flops_per_s"] == pytest.approx(step_flops / 0.02)
    assert row["peak_flops_per_s"] == 121e12  # an L4's dense bf16 rate
    assert row["mfu"] == pytest.approx(step_flops / 0.02 / 121e12)
    assert row["model_parameter_count"] == MODEL.parameter_count(embedding=False)
    assert row["cost_per_ms"] == gpu_price_per_s["L4"] / 1e3
    assert row["flops_per_dollar"] == pytest.approx(step_flops / (gpu_price_per_s["L4"] * 0.02))
    assert len(pd.concat([summary, summary])) == 4  # a series is one table


def test_an_aggregated_summary_is_one_row_with_rates_from_the_reduced_step():
    slower = REPORT | {"steps": [readings(START), readings(START + 5e4) | {"step_ms": 40.0, "peak_allocated_bytes": 2**22}]}
    held = declared().with_results({"report": json.dumps(slower)})
    per_step, mean = held.summary(), held.summary("mean")
    assert "step" not in mean and len(mean) == 1
    row = mean.loc[0]
    assert row["step_ms"] == 30.0 and row["peak_allocated_bytes"] == 2**22  # a peak is the steps' max
    assert row["mfu"] == pytest.approx(MODEL.step_flops(4) / 0.03 / 121e12)  # the mean step's, not the mean mfu
    assert row["mfu"] != pytest.approx(per_step["mfu"].mean())
    assert row["model.d_model"] == 16 and row["model_parameter_count"] == per_step.loc[0, "model_parameter_count"]
    assert held.summary("min").loc[0, "step_ms"] == 20.0


def test_mfu_is_nan_without_a_unit_and_a_forward_step_estimates_no_memory():
    on_a_t4 = json.dumps(REPORT | {"gpu": "Tesla T4"})  # a T4 has no bf16 unit
    row = declared().with_results(TEXTS | {"report": on_a_t4}).summary().loc[0]
    assert math.isnan(row["peak_flops_per_s"]) and math.isnan(row["mfu"])
    forward = declared(step_kind="forward", kernel_events=False, trace_memory=False)
    assert forward.step_flops == MODEL.forward_flops(4) and forward.estimated_memory == {}
    held = forward.with_results({"report": TEXTS["report"]})
    assert held.summary()[["persistent_estimated_bytes", "activation_estimated_bytes"]].isna().all(axis=None)


def test_what_a_summary_cannot_give_is_nan_and_logged_when_verbose(caplog):
    unpeaked = {**REPORT, "gpu": "Some New GPU", "steps": [{"step_ms": 20.0, "phases_unix_us": {}}]}
    mamba = Mamba(vocab_size=64, sequence_length=8, num_layers=1, d_model=16)
    held = declared(model=mamba, peak_memory=False).with_results({"report": json.dumps(unpeaked)})
    row = held.summary("mean").loc[0]
    nan = ["peak_allocated_bytes", "peak_reserved_bytes", "persistent_estimated_bytes", "activation_estimated_bytes",
           "model_parameter_count", "step_flops", "peak_flops_per_s", "achieved_flops_per_s", "mfu", "flops_per_dollar"]
    assert row[nan].isna().all() and row["cost_per_ms"] == gpu_price_per_s["L4"] / 1e3
    assert caplog.messages == []  # quiet unless verbose
    held.summary(verbose=True)
    assert caplog.messages == [f"{held.uid}: no {', '.join(nan)}; NaN in the summary"]
    caplog.clear()
    declared().with_results(TEXTS).summary(verbose=True)
    assert caplog.messages == []  # a profile with everything logs nothing


def test_display_steps_draws_each_step_on_its_first_phases_clock(shown):
    held = declared(kernel_events=False).with_results({"report": TEXTS["report"], "memory": TEXTS["memory"]})
    held.display_steps()
    (fig,) = shown
    first, second = fig.axes  # one memory panel per step, stacked
    assert [first.get_title(), second.get_title()] == ["step 0", "step 1"]
    for axis in (first, second):
        assert axis.lines[0].get_xdata().tolist() == [2.0, 6.0]  # allocated, on the step's own clock
        assert [patch.get_x() for patch in axis.patches] == [0.0, 5.0]  # the phases, where they ran
    shown.clear()
    held.display_steps(steps=[1])
    assert [axis.get_title() for axis in shown[0].axes] == ["step 1"]


def test_display_steps_puts_kernels_on_the_same_clock_and_the_rates_under_the_legend(shown):
    declared().with_results(TEXTS).display_steps(steps=[0])
    (fig,) = shown
    timeline, memory_ax = fig.axes[0], fig.axes[3]  # kernels, FLOPs, durations, then memory and its legend
    (kernel,) = timeline.collections[0].get_paths()
    assert kernel.vertices[:, 0].min() == -95.0  # 5 us after a trace that started 100 us before the step
    assert any("total, est." in line.get_label() for line in memory_ax.lines)  # both estimates exist
    (note,) = [text for axis in fig.axes for text in axis.texts if "MFU" in text.get_text()]
    step_flops = MODEL.step_flops(4)
    assert note.get_text().splitlines() == [
        "GPU: NVIDIA L4 [L4]", "step time: 20.0 ms", f"step: {_with_prefix(step_flops, 'FLOPs')} (est.)",
        f"achieved: {_with_prefix(step_flops / 0.02, 'FLOPs/s')} (est.)", "peak: 121.0 TFLOPs/s bf16",
        f"MFU: {step_flops / 0.02 / 121e12:.1%} (est.)"]
    assert any("algorithmic estimates" in text.get_text() for text in fig.texts)  # the footnote
    shown.clear()
    declared().with_results(TEXTS | {"report": json.dumps(REPORT | {"gpu": "Tesla T4"})}).display_steps(steps=[0])
    (note,) = [text for axis in shown[0].axes for text in axis.texts if "MFU" in text.get_text()]
    assert note.get_text().splitlines()[-2:] == ["peak: n/a", "MFU: n/a"]
    assert _with_prefix(228.8e9, "FLOPs") == "228.8 GFLOPs"


def test_run_profile_spawns_the_spec_on_its_gpu_and_returns_a_copy_holding_the_results(monkeypatch):
    spawned = []

    class Call:
        object_id = "fc-test"
        logs = SimpleNamespace(stream=lambda: iter([SimpleNamespace(message="step 0: 20.0 ms\n")]))

        def get(self):
            return TEXTS

    class Function:
        logs = SimpleNamespace(fetch=lambda **kwargs: [])

        def with_options(self, **options):
            spawned.append(options)
            return self

        def spawn(self, spec):
            spawned.append(spec)
            return Call()

    monkeypatch.setattr(modal.Function, "from_name", lambda app, name: Function())
    spec = declared()
    result = run_profile(spec)
    assert spawned == [{"gpu": "L4"}, spec]
    assert result == spec and result.report == REPORT
    with pytest.raises(ValueError, match="holds no results"):
        spec.summary()
