# launchpad

Launchpad runs ML pipelines on Modal. You describe what should exist (a
dataset, a tokenizer, a trained model), launchpad works out what has to be
built first, runs each piece as its own job, and shows progress on the
dashboard.

Every Markdown file in `docs/` is a page here. `mkdocs serve` previews them
locally; `modal deploy main.py` publishes them at `/guide/`.
