# launchpad

Launchpad enables rapid experimentation with hyperparameters and model architectures through reproducible ML pipelines running on Modal. Any stage of the model training pipeline can be described as a parameterized artifact, from data preparation and tokenization to pre-training, post-training, and model serving. Launchpad automatically resolves the full dependency graph, executes jobs across distributed workers, reuses existing artifacts, and tracks progress through a dashboard.

New here? Start with [Framework](framework.md), which explains the idea, then
[Getting started](getting-started.md): setup, a first workflow, and a training
example. Each other page describes one part, and is the only place that part
is described.

| Page | What it covers |
| --- | --- |
| [How it fits together](core.md) | How artifacts, manifests and jobs interact, and why. |
| [Artifacts](artifacts.md) | Defining, validating and identifying artifacts; resolution. |
| [Manifests](manifests.md) | How a definition is written to disk and read back. |
| [Lab](lab.md) | The notebook API: declaration, loading, binding. |
| [Training a Model](training.md), [Models](models.md) | Training legs, checkpoints, architectures. |
| [Jobs](jobs.md), [Workers](workers.md) | The code that builds an artifact, and what runs around it. |
| [Launcher](launcher.md), [Dashboard](dashboard.md), [Logging](logging.md) | Launching, the state map, the web page, logs. |
| [Repository](repository.md) | What lives where: the volume, the Dicts, the code. |
| [Artifact Repository](artifact-repository.md) | Every artifact family under `artifacts/`, one page each. |

Every Markdown file in `docs/` is a page here. `mkdocs serve` previews them
locally; `mkdocs build --strict && modal deploy main.py` publishes them at `/guide/`.
