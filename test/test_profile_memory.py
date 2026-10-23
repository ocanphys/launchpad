"""Tests for grouping allocator trace entries by record_function labels."""
from profiling.profile_events import allocations_by_label, memory_over_time


def test_allocations_take_the_innermost_open_label():
    def note(stage, name, t):
        return {"stage": stage, "name": name, "device": 0, "time_us": t}

    def alloc(t, size):
        return {"action": "alloc", "addr": t, "size": size, "stream": 0, "time_us": t}

    def free(t, addr):
        return {"action": "free_requested", "addr": addr, "size": 0, "stream": 0, "time_us": t}

    snapshot = {
        "external_annotations": [
            note("START", "train::forward", 10), note("START", "forward::model", 11),
            note("END", "forward::model", 20), note("END", "train::forward", 30),
        ],
        # Out of order, as after a ring-buffer wrap.
        "device_traces": [[alloc(25, 3), alloc(5, 1), free(27, 5), alloc(15, 2), alloc(40, 4), free(35, 15)]],
    }
    rows = allocations_by_label(snapshot)
    assert rows["start_us"].tolist() == [5, 15, 25, 40]
    assert rows["end_us"].fillna(-1).tolist() == [27, 35, -1, -1]  # NaN: never freed in the trace
    assert rows["label"].tolist() == ["<none>", "forward::model", "train::forward", "<none>"]


def test_memory_over_time_ends_at_each_devices_snapshot_totals():
    def entry(action, t, size):
        return {"action": action, "addr": t, "size": size, "stream": 0, "time_us": t}

    snapshot = {
        "segments": [{"device": 0, "allocated_size": 60, "total_size": 200},
                     {"device": 1, "allocated_size": 5, "total_size": 50}],
        "device_traces": [
            [entry("alloc", 3, 30), entry("segment_alloc", 1, 100), entry("free_requested", 4, 10),
             entry("free_completed", 5, 10), entry("alloc", 2, 40)],
            [entry("alloc", 7, 5)],
        ],
    }
    usage = memory_over_time(snapshot)
    assert usage[0].index.tolist() == [1, 2, 3, 4]
    assert usage[0]["allocated"].tolist() == [0, 40, 70, 60]  # starts at 60 - (40 + 30 - 10)
    assert usage[0]["reserved"].tolist() == [200, 200, 200, 200]  # 100 allocated at t=1, so 100 before it
    assert usage[1].index.tolist() == [7]
    assert usage[1]["allocated"].tolist() == [5]
    assert usage[1]["reserved"].tolist() == [50]
