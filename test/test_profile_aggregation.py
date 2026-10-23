"""Partition and FLOP accounting tests for the notebook aggregator."""
from types import SimpleNamespace

import pandas as pd
import pytest

from profiling.profile_events import (
    LAUNCH_CHAIN_COLUMNS,
    PHASE_ORDER,
    aggregate_kernels,
    draw_timeline,
    launch_chain,
)


def event(index, name, parent=None, flops=0, kind="cpu_op"):
    return {"index": index, "name": name, "launcher_id": parent, "flops": flops,
            "activity_type": kind,
            "device_type": "DeviceType.CUDA" if kind in {"kernel", "gpu_memcpy"} else "DeviceType.CPU",
            "time_range": SimpleNamespace(start=float(index), end=float(index + 2))}


def stage(index, name, parent=None):
    return event(index, name, parent, kind="user_annotation")


def kernel(index, parent=None):
    return event(index, "same kernel name", parent, kind="kernel")


def frame(records):
    return pd.DataFrame(records).set_index("index").astype({"launcher_id": "Int64"})


@pytest.fixture
def rows():
    return frame([
        stage(0, "train::forward"), stage(1, "forward::model", 0),
        stage(2, "forward::model.left", 1), event(3, "aten::mm", 2, 100),
        event(4, "aten::mm", 3, 200), kernel(10, 4), kernel(11, 4),
        event(20, "aten::unknown", 2), event(21, "aten::child", 20, 30), kernel(12, 21),
        event(22, "aten::child", 20, 40), kernel(13, 22),
        stage(30, "forward::model.right", 1), event(31, "aten::mm", 30, 20), kernel(14, 31),
        event(40, "aten::direct", 1, 10), kernel(15, 40),
        stage(50, "train::loss"), event(51, "aten::loss", 50, 5), kernel(16, 51),
        kernel(17), stage(60, "forward::model"), stage(61, "forward::model.left", 60),
        event(62, "aten::mm", 61, 7), kernel(18, 62),
        event(63, "aten::cpu_only", 61, 999), event(19, "copy", 3, kind="gpu_memcpy"),
    ])


@pytest.mark.parametrize("depth", [1, 2, 3, 100, None])
def test_partition_and_flops_are_conserved_at_every_depth(rows, depth):
    original = rows.copy(deep=True)
    groups, kernels, _, sources = aggregate_kernels(rows, depth)
    expected = rows.index[rows.activity_type.eq("kernel")]
    assert kernels.index.sort_values().equals(expected)
    assert kernels.index.is_unique
    memberships = [index for ids in groups.kernel_ids for index in ids]
    assert len(memberships) == len(set(memberships)) == len(expected)
    assert set(memberships) == set(expected)
    assert groups.kernel_count.sum() == len(expected)
    assert groups.estimated_flops.sum() == 212
    assert sources.loc[4, "status"] == "suppressed"
    assert sources.loc[4, "suppressed_by"] == 3
    assert sources.loc[3, "kernel_count"] == 2
    assert sources.loc[3, "contribution_flops"] == 100
    assert 63 not in sources.index  # No descendant kernels.
    assert not sources.spans_groups.any()
    assert sources.contribution_flops.sum() == groups.estimated_flops.sum()
    assert kernels.loc[17, "phase"] == "unscoped"
    assert kernels.loc[16, "phase"] == "loss"
    assert kernels.loc[16, "module_fqn"] == "<unattributed>"
    for _, group in groups.iterrows():
        members = kernels.loc[group.kernel_ids]
        assert group.gpu_start_us == members.gpu_start_us.min()
        assert group.gpu_end_us == members.gpu_end_us.max()
        assert group.gpu_span_us == group.gpu_end_us - group.gpu_start_us
        assert group.gpu_time_us == members.duration_us.sum()
        assert group.estimated_flops == sources.loc[group.flop_source_ids, "contribution_flops"].sum()
    pd.testing.assert_frame_equal(rows, original)


def test_direct_work_stays_exclusive_and_repeated_stages_merge(rows):
    groups, kernels, _, _ = aggregate_kernels(rows)
    assert groups.loc[("forward", "model"), "kernel_ids"] == [15]
    assert groups.loc[("forward", "model.left"), "kernel_ids"] == [10, 11, 12, 13, 18]
    assert groups.loc[("forward", "model.left"), "owner_stage_ids"] == [2, 61]
    assert groups.loc[("forward", "model.left"), "estimated_flops"] == 177
    assert kernels.loc[18, "owner_stage_id"] == 61
    assert kernels.loc[18, "owner_fqn"] == "model.left"
    assert (kernels.gpu_end_us - kernels.gpu_start_us).eq(2).all()


def test_spanning_aten_is_diagnosed_and_not_split_or_duplicated():
    rows = frame([
        stage(0, "forward::model"), event(1, "aten::outer", 0, 300),
        stage(2, "forward::model.left", 1), event(3, "aten::left", 2, 100), kernel(4, 3),
        stage(5, "forward::model.right", 1), event(6, "aten::right", 5, 200), kernel(7, 6),
    ])
    groups, kernels, _, sources = aggregate_kernels(rows)
    assert sources.loc[1, "spans_groups"]
    assert sources.loc[1, "status"] == "unallocated"
    assert sources.loc[[3, 6], "status"].eq("suppressed").all()
    assert sources.loc[[3, 6], "suppressed_by"].eq(1).all()
    assert groups.estimated_flops.sum() == 0
    assert kernels.index.tolist() == [4, 7]
    coarse, coarse_kernels, _, coarse_sources = aggregate_kernels(rows, depth=1)
    assert coarse.estimated_flops.sum() == 300
    assert coarse_sources.loc[1, "status"] == "allocated"
    assert not coarse_sources.spans_groups.any()
    assert coarse_kernels.index.equals(kernels.index)


def test_phase_fallbacks_virtual_prefixes_and_unknown_flops():
    rows = frame([
        stage(0, "forward::model"), stage(1, "forward::model.layers.3.attn", 0),
        event(2, "aten::unknown", 1, float("nan")), event(3, "aten::unknown", 2, float("inf")),
        event(4, "aten::mm", 3, 80), kernel(5, 4),
        event(10, "autograd::engine::evaluate_function: MmBackward0"),
        event(11, "aten::mm", 10, 160), kernel(12, 11),
        stage(20, "Optimizer.step#AdamW.step"), kernel(21, 20),
        stage(29, "backward::model"), stage(30, "backward::model.layers.3.attn", 29), event(31, "aten::mm", 30, 160), kernel(32, 31),
    ])
    groups, kernels, _, sources = aggregate_kernels(rows, 2)
    assert kernels.loc[5, "module_fqn"] == "model.layers"
    assert kernels.loc[12, "phase"] == "backward"
    assert kernels.loc[12, "module_fqn"] == "<unattributed>"
    assert kernels.loc[21, "phase"] == "optimizer_step"
    assert kernels.loc[32, "phase"] == "backward"
    assert groups.estimated_flops.sum() == 400
    assert set(sources.index) == {4, 11, 31}


def test_module_owned_kernel_must_reach_the_root_stage():
    orphan = frame([stage(0, "forward::model.left"), kernel(1, 0)])
    with pytest.raises(ValueError, match=r"not under the model stage: \[1\]"):
        aggregate_kernels(orphan)
    nested = frame([stage(0, "forward::model"), event(1, "aten::mm", 0), stage(2, "forward::model.left", 1),
                    kernel(3, 2)])
    assert aggregate_kernels(nested)[1].loc[3, "module_fqn"] == "model.left"


def test_mismatched_root_leaves_every_kernel_unattributed():
    rows = frame([stage(0, "forward::model"), stage(1, "forward::model.left", 0), kernel(2, 1)])
    _, kernels, _, _ = aggregate_kernels(rows, root="other")
    assert kernels["module_fqn"].eq("<unattributed>").all()


def test_empty_input_and_invalid_inputs(rows):
    groups, kernels, _, sources = aggregate_kernels(rows.iloc[:0])
    assert groups.empty and kernels.empty and sources.empty
    for depth in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="depth"):
            aggregate_kernels(rows, depth)
    with pytest.raises(ValueError, match="unique"):
        aggregate_kernels(pd.concat([rows, rows.iloc[:1]]))
    broken = rows.copy()
    broken.loc[1, "launcher_id"] = 2
    with pytest.raises(ValueError, match="cycle"):
        aggregate_kernels(broken)
    broken.loc[1, "launcher_id"] = 99999
    with pytest.raises(ValueError, match="missing"):
        aggregate_kernels(broken)
    invalid_interval = rows.copy()
    invalid_interval.loc[10, "time_range"] = SimpleNamespace(start=10, end=9)
    with pytest.raises(ValueError, match="GPU interval"):
        aggregate_kernels(invalid_interval)


def test_row_order_does_not_affect_ownership_or_flops(rows):
    groups, kernels, _, sources = aggregate_kernels(rows)
    shuffled_groups, shuffled_kernels, _, shuffled_sources = aggregate_kernels(rows.iloc[::-1])
    pd.testing.assert_series_equal(groups.estimated_flops.sort_index(), shuffled_groups.estimated_flops.sort_index())
    pd.testing.assert_series_equal(kernels.module_fqn.sort_index(), shuffled_kernels.module_fqn.sort_index())
    pd.testing.assert_series_equal(sources.contribution_flops.sort_index(), shuffled_sources.contribution_flops.sort_index())


def test_outputs_are_in_phase_order_and_unattributed_kernels_are_split_out(rows):
    groups, kernels, unattributed_kernels, _ = aggregate_kernels(rows)
    assert groups.index.get_level_values("phase").map(PHASE_ORDER).is_monotonic_increasing
    assert kernels["phase"].map(PHASE_ORDER).is_monotonic_increasing
    assert not unattributed_kernels.empty
    assert unattributed_kernels.index.equals(kernels.index[kernels["module_fqn"].eq("<unattributed>")])


def test_launch_chain_runs_from_the_cpu_root_to_the_kernel(rows):
    assert launch_chain(rows, 10, format=False).index.tolist() == [0, 1, 2, 3, 4, 10]
    assert launch_chain(rows, 17, format=False).index.tolist() == [17]  # no launcher


def test_formatted_launch_chain_keeps_the_key_columns_and_expands_time_range(rows):
    wide = rows.reindex(columns=[*rows.columns, *(c for c in LAUNCH_CHAIN_COLUMNS if c not in rows.columns)])
    chain = launch_chain(wide, 10)
    assert chain.columns.tolist() == LAUNCH_CHAIN_COLUMNS
    assert chain.loc[10, ["start_us", "end_us"]].tolist() == [10.0, 12.0]
    assert launch_chain(wide, 10, format=False).columns.tolist() == wide.columns.tolist()


@pytest.fixture
def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def draw(plt, *args, **kwargs):
    """Return a figure with one draw_timeline in it."""
    fig = plt.figure()
    draw_timeline(fig, fig.add_gridspec(1, 1)[0], *args, **kwargs)
    return fig


def test_timeline_keeps_every_kernel_interval_and_aligned_flops(rows, plotting):
    groups, kernels, _, _ = aggregate_kernels(rows)
    timeline, flop_ax, _ = draw(plotting, groups, kernels).axes
    ordered = groups.sort_values("gpu_start_us", kind="stable")
    seen = []
    for collection, (key, group) in zip(timeline.collections, ordered.iterrows()):
        assert collection.get_gid() == f"kernels:{key[0]}::{key[1]}"
        assert len(collection.get_paths()) == len(group.kernel_ids)
        for path, index in zip(collection.get_paths(), group.kernel_ids):
            seen.append(index)
            assert path.vertices[:, 0].min() == pytest.approx(kernels.loc[index, "gpu_start_us"])  # as given
            assert path.vertices[:, 0].max() == pytest.approx(kernels.loc[index, "gpu_end_us"])
    assert len(seen) == len(set(seen)) == len(kernels)
    assert set(seen) == set(kernels.index)
    assert [bar.get_width() for bar in flop_ax.patches] == ordered.estimated_flops.tolist()
    assert timeline.get_ylim() == flop_ax.get_ylim()
    start, end = ordered.gpu_start_us.min(), ordered.gpu_end_us.max()
    assert timeline.get_xlim() == pytest.approx((start - 0.05 * (end - start), end + 0.05 * (end - start)))  # padded
    assert [label.get_text() for label in timeline.get_yticklabels()] == [f"{p}::{m}" for p, m in ordered.index]


def test_timeline_memory_panel_shares_the_time_axis(rows, plotting):
    groups, kernels, _, _ = aggregate_kernels(rows)
    memory = pd.DataFrame({"allocated": [1, 2**20], "reserved": [2**20, 2**20]}, index=[-2.0, 5.0])
    fig = draw(plotting, groups, kernels, memory=memory, total_memory=2**40,
               memory_baseline={"allocated": 0, "reserved": 2**19}, persistent_memory=2**19)
    timeline, flop_ax, _, memory_ax, _ = fig.axes
    assert memory_ax.get_shared_x_axes().joined(memory_ax, timeline)
    assert min(memory_ax.lines[0].get_xdata()) == -2.0  # memory is drawn at its own times, not re-zeroed
    span = groups.gpu_end_us.max() + 2.0  # from the memory's first entry at -2 to the last kernel
    assert timeline.get_xlim() == pytest.approx((-2.0 - 0.05 * span, groups.gpu_end_us.max() + 0.05 * span))
    assert memory_ax.get_ylim() == (0, pytest.approx(1.1))  # the larger total is clipped
    curves = [line for line in memory_ax.lines if line.get_label() in ("allocated", "reserved")]
    assert len(curves) == 2
    faded = [line for line in memory_ax.lines if "before model" in line.get_label()]
    assert [line.get_label() for line in memory_ax.lines if "persistent" in line.get_label()] == ["persistent (0.5 MiB)"]
    assert [line.get_alpha() for line in faded] == [0.5, 0.5]
    assert [line.get_color() for line in faded] == [line.get_color() for line in curves]


def test_timeline_draws_only_the_panels_it_has_data_for(plotting):
    memory = pd.DataFrame({"allocated": [1, 2**20], "reserved": [2**20, 2**20]}, index=[0.0, 5.0])
    (memory_ax,) = draw(plotting, memory=memory, total_memory=2**40).axes  # no kernel timeline, no FLOPs panel
    assert memory_ax.get_title() == "Training step"
    assert memory_ax.get_xlim() == pytest.approx((-0.25, 5.25))
    assert draw(plotting).axes == []  # nothing to draw


def test_timeline_shades_phases_where_they_are(plotting):
    memory = pd.DataFrame({"allocated": [1, 2**20], "reserved": [2**20, 2**20]}, index=[0.0, 5.0])
    phases = pd.DataFrame({"name": ["zero_grad", "forward"], "start_us": [-2.0, 0.0], "end_us": [-1.0, 8.0]})
    (memory_ax,) = draw(plotting, memory=memory, phases=phases).axes
    assert [patch.get_x() for patch in memory_ax.patches] == [-2.0, 0.0]
    assert [text.get_text() for text in memory_ax.texts] == ["zero_grad", "forward"]
    assert memory_ax.get_xlim() == pytest.approx((-2.0 - 0.05 * 10.0, 8.0 + 0.05 * 10.0))  # reaches every phase


def test_timeline_has_a_lane_of_duration_dots_per_phase_in_the_phase_colors(rows, plotting):
    groups, kernels, _, _ = aggregate_kernels(rows)
    timeline, _, dots_ax = draw(plotting, groups, kernels).axes
    phases = list(dict.fromkeys(groups.index.get_level_values("phase")))
    assert dots_ax.get_xscale() == "log"
    assert dots_ax.get_position().height < timeline.get_position().height  # thin
    assert [label.get_text().split(" (")[0] for label in dots_ax.get_yticklabels()] == phases  # a lane per phase
    legend = {text.get_text(): handle.get_facecolor()[:3] for handle, text in
              zip(timeline.get_legend().legend_handles, timeline.get_legend().get_texts())}
    from matplotlib.collections import LineCollection, PathCollection

    scatters = [c for c in dots_ax.collections if isinstance(c, PathCollection)]
    ticks = [c for c in dots_ax.collections if isinstance(c, LineCollection)]
    for lane, (phase, dots) in enumerate(zip(phases, scatters)):
        durations = kernels.loc[kernels["phase"] == phase, "duration_us"]
        assert sorted(dots.get_offsets()[:, 0]) == pytest.approx(sorted(durations))  # one dot per kernel, at its duration
        assert all(abs(y - lane) <= 0.3 for y in dots.get_offsets()[:, 1])  # in its own lane
        assert tuple(dots.get_facecolor()[0][:3]) == pytest.approx(legend[f"{phase.replace('_', ' ').capitalize()} GPU"])
    medians = [segment[0][0] for collection in ticks for segment in collection.get_segments()]
    assert medians == pytest.approx([kernels.loc[kernels["phase"] == phase, "duration_us"].median() for phase in phases])
