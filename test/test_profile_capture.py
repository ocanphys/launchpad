"""Tests for capture's context managers, with the CUDA calls faked so they run without a GPU."""
import pytest
import torch
from profile_events import capture_memory, capture_profiler_events
from torch.profiler import record_function


@pytest.fixture
def cuda(monkeypatch):
    """Record the CUDA calls the context managers make."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: calls.append("reset_peaks"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 20)
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history",
                        lambda enabled="all", **k: calls.append(f"history:{enabled}"))
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda *a: {"segments": []})
    return calls


def test_everything_off_does_nothing(cuda):
    with capture_memory() as memory, capture_profiler_events(False) as profiled:
        pass
    assert memory == {} and profiled == {"events": []}
    assert cuda == []


def test_peaks_reset_before_the_block_and_read_after(cuda):
    with capture_memory(peak=True) as memory:
        assert memory == {} and cuda == ["reset_peaks"]
    assert memory == {"peak_allocated": 10, "peak_reserved": 20}


def test_trace_snapshots_the_block_and_stops_recording_even_when_it_fails(cuda):
    with capture_memory(trace=True) as memory:
        pass
    assert memory == {"snapshot": {"segments": []}}
    assert cuda == ["history:all", "history:None"]
    cuda.clear()
    with pytest.raises(RuntimeError, match="step failed"):
        with capture_memory(trace=True):
            raise RuntimeError("step failed")
    assert cuda[-1] == "history:None"


def test_profiler_events_hold_the_blocks_scopes_and_the_trace_start_on_the_unix_clock(cuda):
    with capture_profiler_events(True) as profiled:
        with record_function("train::forward"):
            torch.ones(2) + 1
    assert any(event.name == "train::forward" for event in profiled["events"])
    assert profiled["trace_start_us"] > 1e15  # unix microseconds
    assert cuda == ["synchronize"]  # the profiler's barrier before it stops
