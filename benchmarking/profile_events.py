"""The Modal app and image a `torch.profiler` capture runs on.

A capture returns `list(prof.events())` unchanged: `FunctionEvent` objects pickle with their
`cpu_parent` and `cpu_children` references intact, so every step after profiling runs locally.
torch is pinned so the profiler behaviour the notebook relies on matches the source it cites,
and so the local torch that unpickles the events is the one that produced them.
"""

from __future__ import annotations

import modal

app = modal.App("profile-events")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "einops")
    .add_local_python_source("profile_events", "models", "artifacts")
)
