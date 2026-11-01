# Profiling

Profiling trains a model for a few steps on a Modal GPU and measures them: the time of every
CUDA kernel, attributed to the model's submodules, the FLOPs, the memory, and how long each phase
of the step takes. It is not a launched job: no artifact, no lease, nothing written to the volume.

## Profile a model on a GPU

In a notebook:

```python
from artifacts.core.artifact import Resources
from artifacts.core.SGD.optimizers import AdamW
from models.transformer import Transformer
from profiling.profile_events import display_steps
from profiling.step_report import build_report
from profiling.util import run_profile

model = Transformer(
    vocab_size=10000,
    sequence_length=512,
    dtype="torch.float32",
    num_layers=4,
    d_model=128,
    d_ff=256,
    num_heads=4,
    rope_theta=10000,
)
optimizer = AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)

captured = run_profile(
    Resources(gpu_type="A10"),
    model=model,
    optimizer=optimizer,
    batch_size=4,
    seed=0,
    warmup_steps=5,
    profile_steps=3,
    autocast="torch.bfloat16",
    kernel_events=True,
    module_scopes=True,
    peak_memory=True,
    trace_memory=True,
)
```

Modal's log of the call prints as it runs, the profiler's own `profiler_start` and `profiler_stop`
lines included:

```text
profile call fc-01M6B2GH5QK95N6Z58CPECDJ7J
2026-11-01T08:09:43.914Z INFO Transformer on NVIDIA A10, batch_size 4
2026-11-01T08:10:08.412Z INFO warmup: 5 steps
USDT:2026-11-01 08:10:08 2:2 SyncActivityProfilerHandler.cpp:52] profiler_start
USDT:2026-11-01 08:10:08 2:2 SyncActivityProfilerHandler.cpp:59] profiler_stop
2026-11-01T08:10:10.054Z INFO step 0: 20.5 ms
...
2026-11-01T08:10:10.295Z INFO step 2: 18.7 ms
2026-11-01T08:10:10.295Z INFO done
```

`captured` holds the raw measurements. Turn it into a report and plot a step:

```python
report = build_report(captured)
display_steps(report, steps=[0])
```

![Step 0 of the run above: memory over time, kernels per module and phase, estimated FLOPs per group, and kernel durations per phase](img/profiling-step.png)

From top to bottom:

- **Memory:** allocated and reserved bytes over the step, with their peaks, the persistent
  memory (weights, gradients, optimizer state) and the VRAM. The shading marks each phase.
- **Kernels:** one row per phase and module, one tick per kernel on the GPU's clock. The bars
  on the right are each row's estimated FLOPs.
- **Kernel durations:** every kernel's duration per phase, on a log scale.

The box beside the memory panel gives the GPU, the step's time, its FLOPs and the MFU.

`report` is a plain, JSON-serializable dict. From a run with the settings above:

```text
report
├── schema_version: 1
├── run                                    # what was profiled, once per call
│   ├── gpu: {name: 'NVIDIA A10', spec_row: 'A10'}
│   ├── model
│   │   ├── class: 'Transformer'
│   │   ├── parameters: {vocab_size: 10000, d_model: 128, num_layers: 4, ...}
│   │   ├── size: {parameters: 3216512, bytes: 12866048}
│   │   └── gradient_size: {parameters: 3216512, bytes: 12866048}
│   ├── optimizer: {class: 'AdamW', state_parameters: 6433063, state_bytes: 25732252}
│   ├── capture: {batch_size: 4, warmup_steps: 5, profile_steps: 3, autocast: 'torch.bfloat16', ...}
│   ├── precision: 'torch.bfloat16'
│   ├── peak: {dtype: 'bf16', flops_per_s: 1.25e14}
│   ├── step_flops: 30224154624
│   ├── sources: {step_flops: 'algorithmic estimate: ...', kernel_flops: ..., mfu: ...}
│   └── memory
│       ├── device_vram_bytes: 23684186112
│       ├── persistent: {measured_bytes: 51464348, estimated_bytes: 51464192}
│       └── activation: {estimated_bytes: 272302080}
└── steps                                  # one entry per measured step
    ├── "0"
    │   ├── step_ms: 196.76
    │   ├── start_unix_us: 1793480963794424.8
    │   ├── mfu: {step_flops: 30224154624, step_ms: 196.76, achieved_flops_per_s: 1.54e11,
    │   │         peak_flops_per_s: 1.25e14, mfu: 0.0012}
    │   ├── phases: [5 records]            # zero_grad, forward, loss, backward, optimizer_step
    │   │   └── {name: 'zero_grad', start_us: 0.0, end_us: 751.5, duration_ms: 0.75}
    │   ├── memory
    │   │   ├── peak_allocated_bytes: 456828416
    │   │   ├── peak_reserved_bytes: 595591168
    │   │   └── series                     # columnar: 1632 values per field
    │   │       ├── time_us: [157.25, ...]
    │   │       ├── allocated: [64710656, ...]
    │   │       └── reserved: [595591168, ...]
    │   ├── kernels                        # columnar: 840 values per field, one per kernel
    │   │   ├── id: [14, ...]
    │   │   ├── phase: ['forward', ...]
    │   │   ├── module: ['model.embedding', ...]
    │   │   ├── start_us: [1278.27, ...]
    │   │   ├── end_us: [1283.39, ...]
    │   │   └── duration_us: [5.12, ...]
    │   └── groups: [17 records]           # one per phase and module, the plot's rows
    │       └── {phase: 'forward', module: 'model.embedding', kernel_ids: [14], start_us: 1278.27,
    │            end_us: 1283.39, gpu_time_us: 5.12, total_est_flops: 0.0}
    ├── "1"                                # same shape
    └── "2"
```

Times in a step are us from the step's start. Columnar blocks turn back into a table with
`pd.DataFrame(report["steps"]["0"]["kernels"])`.

## What is passed in

`model` and `optimizer` are declarations, not live objects: no weights, no optimizer state.
They are sent to the profiling container, where `model.build` materializes the network and
`optimizer.build` creates the torch optimizer over its parameters. To profile a declared leg,
pass its fields: `optimizer=leg.training_parameters.optimizer`, and the same for `batch_size`
and `seed`.

## Running a profile

`run_profile` sends the call to Modal and prints Modal's live log of it, the call's stdout and
stderr in order: `profile`'s log rows, and whatever a library writes past `logging`, such as
Inductor's autotuning under `compile="max-autotune"`. The rows reach nothing else: `main.profile`
logs to stdout only, never to `call_logs`. Until the call writes its first line,
`run_profile` also prints Modal's own messages about `profile`, so a call queued for a GPU says so:

```text
Function 'profile' (fu-8amyGm50LUR45l2eHnFfTq) is waiting to be scheduled on a GPU_L4 worker. We are actively working on acquiring more capacity for your workload.
```

`profile` is what runs on the GPU: a warmup loop, then the measured steps, each under the
instruments its switches turn on.

### The step and compile

Two arguments choose what a step runs, and how:

```python
captured = run_profile(Resources(gpu_type="A10"), model=model, optimizer=optimizer, batch_size=4,
                       seed=0, warmup_steps=5, profile_steps=3, autocast="torch.bfloat16",
                       kernel_events=True, module_scopes=False, peak_memory=True,
                       trace_memory=True, step="forward", compile="max-autotune")
```

- **`step`.** `"full"`, the default, is a next-token training step in five phases, each labelled
  `train::<name>` and timed: `zero_grad`, `forward`, `loss`, `backward` and `optimizer_step`.
  `"forward"` is the forward alone, under `torch.no_grad()`, in one phase, `forward`. Its
  `step_flops` is the model's `forward_flops`, so MFU counts the forward only, and it leaves no
  gradients or optimizer state, so the report has no memory estimates.
- **`compile`.** None, the default, runs the model eagerly. A mode (`"default"`,
  `"reduce-overhead"`, `"max-autotune"`) runs it through `torch.compile(network, mode=compile)`,
  compiled once, in the container; the loss and the optimizer step stay eager. The first steps
  compile, so give `warmup_steps` enough of them that no compiling lands in a measured step. The
  worker image has `gcc`, which Triton needs to build its kernel launchers.
- **Not both `compile` and `module_scopes`.** Module stages do not work under `torch.compile`,
  so `profile` asserts `module_scopes=False` when `compile` is set, and kernels group by phase
  only.

::: profiling.util.run_profile
    options:
      show_root_heading: true
      heading_level: 3

::: profiling.util.profile
    options:
      show_root_heading: true
      heading_level: 3

## Processing events

`events_to_rows` turns a step's events into a table, `launch_chain` shows how one kernel in it
was launched, and `aggregate_kernels` groups its kernels by phase and module with the FLOPs of
each group. How each step works, from the raw events to FLOP sources, is on
[Attribution](profiling/attribution.md).

## Compile and Triton

TBD.

- Eager mode versus `torch.compile`: what Dynamo does.
- How attribution works when compile and Triton are involved.

## Where things live

- **`profiling/util.py`**: `run_profile`, `profile` (the loop that runs on the GPU, and its two
  steps) and `phase`.
- **`profiling/profile_events.py`**: the capture context managers, the tables built from the
  events (each kernel linked to the CPU op that launched it, kernels by phase and module depth,
  allocations by scope) and `display_steps`.
- **`profiling/attribution_diagrams.py`**: `render`, the event-tree picture of a profiled step
  or one event in it, and the cases drawn on [Attribution](profiling/attribution.md), redrawn
  with `uv run python -m profiling.attribution_diagrams`.
- **`profiling/step_report.py`**: `build_report`, the one place the analysis lives.

`declarations/profiling.ipynb` is the minimal template: a model, an optimizer, one
`run_profile` call, its plot, the kernel tables, a launch chain and one event drawn with `render`. `profiling/profile_model.ipynb` runs all of it. Two traps in reading profiler events are in
LESSONS.md.
