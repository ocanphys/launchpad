"""Tests for capture's context managers, with the CUDA calls faked so they run without a GPU."""
import json

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from profiling.profile_events import capture_memory, capture_profiler_events, event_table, events_to_rows


@pytest.fixture
def cuda(monkeypatch):
    """Record the CUDA calls the context managers make."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: calls.append("reset_peaks"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 20)
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history",
                        lambda enabled="all", context="all", **k: calls.append(f"history:{enabled}:{context}"))
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda *a: {"segments": []})
    return calls


def test_everything_off_does_nothing(cuda):
    with capture_memory() as memory, capture_profiler_events(False) as profiled:
        pass
    assert memory == {} and profiled == {}
    assert cuda == []


def test_peaks_reset_before_the_block_and_read_after(cuda):
    with capture_memory(peak=True) as memory:
        assert memory == {} and cuda == ["reset_peaks"]
    assert memory == {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20}


def test_trace_snapshots_the_block_without_tracebacks_and_stops_recording_even_when_it_fails(cuda):
    with capture_memory(trace=True) as memory:
        pass
    assert memory == {"snapshot": {"segments": []}}
    assert cuda[0] == "history:all:None"  # recording, with no tracebacks
    assert cuda[-1].startswith("history:None:")  # stopped
    cuda.clear()
    with pytest.raises(RuntimeError, match="step failed"):
        with capture_memory(trace=True):
            raise RuntimeError("step failed")
    assert cuda[-1].startswith("history:None:")


def test_profiler_events_are_a_table_and_the_trace_start_is_on_the_unix_clock(cuda):
    with capture_profiler_events(True) as profiled:
        with record_function("train::forward"):
            torch.ones(2) + 1
    assert list(profiled) == ["trace_start_unix_us", "events"]
    assert "train::forward" in profiled["events"]["name"]
    assert profiled["trace_start_unix_us"] > 1e15  # unix microseconds
    assert cuda == ["synchronize"]  # the profiler's barrier before it stops


def test_the_event_table_is_json_with_every_parent_a_row_id():
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        with record_function("train::forward"):
            torch.zeros(2).masked_fill(torch.tensor([True, False]), float("-inf"))
    events = prof.events()
    table = json.loads(json.dumps(event_table(events), allow_nan=False))  # no object, NaN or infinity in it
    assert {len(column) for column in table.values()} == {len(events)}
    assert {"time_range", "cpu_children", "kernels"}.isdisjoint(table)
    position = {id(event): i for i, event in enumerate(events)}
    assert table["cpu_parent"] == [position.get(id(event.cpu_parent)) for event in events]
    assert table["start_us"] == [event.time_range.start for event in events]
    assert table["end_us"] == [event.time_range.end for event in events]
    assert set(table["device_type"]) == {"DeviceType.CPU"}
    fill = table["concrete_inputs"][table["name"].index("aten::masked_fill")]
    assert fill[-1] == "-inf"  # a value JSON cannot hold, written as its str
    links = {i: parent for i, parent in enumerate(table["cpu_parent"]) if parent is not None}
    assert events_to_rows(table)["launcher_id"].dropna().to_dict() == links  # a CPU row's launcher is its cpu_parent
    assert event_table([]) == {}
