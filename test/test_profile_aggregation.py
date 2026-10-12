"""Partition and FLOP accounting tests for the notebook aggregator."""
from types import SimpleNamespace

import pandas as pd
import pytest
from profile_events import aggregate_kernels, display_timeline


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
    groups, kernels, sources = aggregate_kernels(rows, depth)
    expected = rows.index[rows.activity_type.eq("kernel")]
    assert kernels.index.equals(expected)
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
    groups, kernels, _ = aggregate_kernels(rows)
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
    groups, kernels, sources = aggregate_kernels(rows)
    assert sources.loc[1, "spans_groups"]
    assert sources.loc[1, "status"] == "unallocated"
    assert sources.loc[[3, 6], "status"].eq("suppressed").all()
    assert sources.loc[[3, 6], "suppressed_by"].eq(1).all()
    assert groups.estimated_flops.sum() == 0
    assert kernels.index.tolist() == [4, 7]
    coarse, coarse_kernels, coarse_sources = aggregate_kernels(rows, depth=1)
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
    groups, kernels, sources = aggregate_kernels(rows, 2)
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
    _, kernels, _ = aggregate_kernels(rows, root="other")
    assert kernels["module_fqn"].eq("<unattributed>").all()


def test_empty_input_and_invalid_inputs(rows):
    groups, kernels, sources = aggregate_kernels(rows.iloc[:0])
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
    groups, kernels, sources = aggregate_kernels(rows)
    shuffled_groups, shuffled_kernels, shuffled_sources = aggregate_kernels(rows.iloc[::-1])
    pd.testing.assert_series_equal(groups.estimated_flops.sort_index(), shuffled_groups.estimated_flops.sort_index())
    pd.testing.assert_series_equal(kernels.module_fqn.sort_index(), shuffled_kernels.module_fqn.sort_index())
    pd.testing.assert_series_equal(sources.contribution_flops.sort_index(), shuffled_sources.contribution_flops.sort_index())


def test_timeline_keeps_every_kernel_interval_and_aligned_flops(rows, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, kernels, _ = aggregate_kernels(rows)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    try:
        display_timeline(groups, kernels)
        assert shown == [True]
        timeline, flop_ax = plt.gcf().axes
        ordered = groups.sort_values("gpu_start_us", kind="stable")
        origin = ordered.gpu_start_us.min()
        seen = []
        for collection, (key, group) in zip(timeline.collections, ordered.iterrows()):
            assert collection.get_gid() == f"kernels:{key[0]}::{key[1]}"
            assert len(collection.get_paths()) == len(group.kernel_ids)
            for path, index in zip(collection.get_paths(), group.kernel_ids):
                seen.append(index)
                assert path.vertices[:, 0].min() == pytest.approx(kernels.loc[index, "gpu_start_us"] - origin)
                assert path.vertices[:, 0].max() == pytest.approx(kernels.loc[index, "gpu_end_us"] - origin)
        assert len(seen) == len(set(seen)) == len(kernels)
        assert set(seen) == set(kernels.index)
        assert [bar.get_width() for bar in flop_ax.patches] == ordered.estimated_flops.tolist()
        assert timeline.get_ylim() == flop_ax.get_ylim()
        assert [label.get_text() for label in timeline.get_yticklabels()] == [f"{p}::{m}" for p, m in ordered.index]
    finally:
        plt.close("all")
