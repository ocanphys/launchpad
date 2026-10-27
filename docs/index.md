# launchpad

Launchpad runs ML pipelines on Modal. You describe what should exist (a
dataset, a tokenizer, a trained model), Launchpad works out what has to be
built first, runs each piece as its own job, and shows progress on the
dashboard.

New here? Start with [Getting started](getting-started.md): setup, a first
workflow, and a training example. [Framework](framework.md) explains the idea;
each other page describes one part, and is the only place that part is
described.

| Page | What it covers |
| --- | --- |
| [Artifacts](artifacts.md) | Defining, validating and identifying artifacts; resolution. |
| [Manifests](manifests.md) | How a definition is written to disk and read back. |
| [Lab](lab.md) | The notebook API: declaration, loading, binding. |
| [Training a Model](training.md), [Models](models.md) | Training legs, checkpoints, architectures. |
| [Jobs](jobs.md), [Workers](workers.md) | The code that builds an artifact, and what runs around it. |
| [Launcher](launcher.md), [Dashboard](dashboard.md), [Logging](logging.md) | Launching, the state map, the web page, logs. |
| [Artifact Repository](repository.md) | What lives where: the volume, the Dicts, the code. |

Every Markdown file in `docs/` is a page here. `mkdocs serve` previews them
locally; `modal deploy main.py` publishes them at `/guide/`.
