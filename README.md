# launchpad

Launchpad runs ML pipelines on [Modal](https://modal.com): you describe what
should exist (a dataset, a tokenizer, a trained model) as Python objects, and it
works out what has to be built first, runs each piece as its own job against
one shared volume, and shows progress on a dashboard.

Everything about how it works is in the docs, under `docs/`:

```bash
uv sync
mkdocs serve        # http://127.0.0.1:8000
```

A deployed app (`modal deploy main.py`) also serves them at `/guide/`. Start
with [docs/getting-started.md](docs/getting-started.md).
