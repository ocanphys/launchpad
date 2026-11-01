import pandas as pd
import pytest

from profiling.profile_events import events_to_rows, link_events


def event(index, kind, corr=0, linked=0, parent=None, node=-1):
    return {"index": index, "name": kind, "activity_type": kind,
            "device_type": "DeviceType.CUDA" if kind in {"kernel", "gpu_user_annotation", "gpu_memcpy", "gpu_memset"} else "DeviceType.CPU",
            "correlation_id": corr, "linked_correlation_id": linked, "launcher_id": parent,
            "node_id": node}


def frame(events):
    return pd.DataFrame(events).set_index("index").astype({"launcher_id": "Int64"})


def test_cpu_trees_and_runtime_gpu_links_preserve_all_rows():
    raw = frame([
        event(50, "kernel", 100, linked=3),  # Child before its parent.
        event(10, "user_annotation", 1),
        event(20, "cpu_op", 2, parent=10),
        event(30, "cpu_op", 3, parent=20),
        event(40, "cuda_runtime", 100, linked=3, parent=30),
        event(51, "kernel", 100, linked=3),
        event(60, "user_annotation", 100, parent=10),
        event(70, "gpu_user_annotation", 100),  # No linked ID: independent root.
        event(80, "cpu_op", 4, parent=10),  # Sibling ops share the same root.
        event(90, "gpu_memcpy", 101, linked=4),
        event(91, "gpu_memset", 102, linked=4),
    ])
    rows = link_events(raw)
    original_columns = raw.columns.drop("launcher_id")
    pd.testing.assert_frame_equal(rows[original_columns], raw[original_columns])
    assert rows.loc[50, "launcher_id"] == 40  # Runtime launch, not frontend 30.
    assert rows.loc[51, "launcher_id"] == 40
    assert rows.loc[40, "launcher_id"] == 30
    assert pd.isna(rows.loc[70, "launcher_id"])
    assert rows.index.equals(raw.index)
    assert set(rows.columns) == set(raw.columns)
    assert rows.loc[[20, 30, 80, 90, 91], "launcher_id"].tolist() == [10, 20, 10, 80, 80]
    current, branch = 50, []
    while pd.notna(current):
        branch.append(current)
        current = rows.loc[current, "launcher_id"]
    assert branch == [50, 40, 30, 20, 10]
    pd.testing.assert_frame_equal(link_events(rows), rows)


def test_unmatched_and_ambiguous_links_and_cpu_parent_authority():
    raw = frame([
        event(1, "cpu_op", 7),
        event(2, "kernel", 999, linked=7),
        event(3, "kernel", 999, linked=888),
        event(4, None),
        event(5, "cuda_runtime", 10, linked=7),  # CPU uses only cpu_parent.
        event(6, "cpu_op", 8), event(7, "cpu_op", 8),
        event(8, "kernel", 777, linked=8),
        event(9, "cuda_runtime", 7),  # Does not collide with frontend ID 7.
        event(10, "kernel", 777, linked=0),
    ])
    rows = link_events(raw)
    assert rows.loc[2, "launcher_id"] == 1
    for index in [3, 4, 5, 8, 10]:
        assert pd.isna(rows.loc[index, "launcher_id"])


def test_correlation_ids_are_local_to_a_node():
    rows = link_events(frame([
        event(0, "cpu_op", 7, node=0), event(1, "cpu_op", 7, node=1),
        event(2, "kernel", linked=7, node=0), event(3, "kernel", linked=7, node=1),
    ]))
    assert rows.loc[[2, 3], "launcher_id"].tolist() == [0, 1]


def test_invalid_graphs():
    raw = frame([event(1, "cpu_op", 1)])
    with pytest.raises(ValueError, match="unique"):
        link_events(pd.concat([raw, raw]))
    with pytest.raises(ValueError, match="missing or non-CPU launcher"):
        link_events(frame([event(1, "cpu_op", parent=999)]))
    with pytest.raises(ValueError, match="cycle"):
        link_events(frame([event(1, "cpu_op", parent=2), event(2, "cpu_op", parent=1)]))
    with pytest.raises(ValueError, match="cycle"):
        link_events(frame([event(1, "cpu_op", parent=1)]))


def test_deep_reverse_order_and_empty_input():
    raw = frame([event(i, "cpu_op", parent=i - 1 if i else None) for i in reversed(range(5000))])
    rows = link_events(raw)
    assert rows.index.equals(raw.index)
    pd.testing.assert_series_equal(rows["launcher_id"], raw["launcher_id"])
    empty = link_events(raw.iloc[:0])
    assert empty.empty
    assert str(empty["launcher_id"].dtype) == "Int64"


def test_launch_namespace_nodes_ambiguity_and_device_work():
    rows = link_events(frame([
        event(0, "cpu_op", 7, node=0),
        event(1, "cuda_driver", 99, parent=0, node=0),
        event(2, "cuda_runtime", 99, node=1),
        event(3, "kernel", 99, linked=7, node=0),
        event(4, "gpu_memcpy", 99, linked=7, node=0),
        event(5, "gpu_memset", 99, linked=7, node=0),
        event(6, "kernel", 99, node=1),
        event(7, "cuda_runtime", 100, node=0),
        event(8, "cuda_driver", 100, node=0),
        event(9, "kernel", 100, linked=7, node=0),  # Ambiguous launch: frontend fallback.
        event(10, "kernel", 100, node=0),  # Ambiguous launch, no frontend: root.
        event(11, "gpu_user_annotation", 99, node=0),  # Numeric collision is not a launch.
    ]))
    assert rows.loc[[3, 4, 5, 6, 9], "launcher_id"].tolist() == [1, 1, 1, 2, 0]
    assert rows.loc[[10, 11], "launcher_id"].isna().all()


def test_no_events_says_kernel_events_was_off():
    with pytest.raises(ValueError, match="kernel_events=True"):
        events_to_rows({})
