---
name: code-style
description: Code style and module-organization guidelines for this project (launchpad) — docstring format, when a function is justified, and how to structure modules. Load before writing or reviewing Python code here.
---

# Code style

Guidelines for writing and reviewing code in this project.

## Docstrings

Keep docstrings short and clear. State what the function does, then give one
input/output example — that example does more for readability than a
paragraph of prose.

```python
def flatline_age(last_beat: float, now: float) -> float:
    """Seconds since the last heartbeat.

    flatline_age(last_beat=100.0, now=105.0) -> 5.0
    """
    return now - last_beat
```

Skip the docstring entirely if the function name and signature already make
the behavior obvious — an empty explanation is worse than none.

## Justify every function

Before splitting code into a new function, be able to state why in one
sentence. A function earns its existence when it is:

- **Explainable** — its purpose fits in one sentence, without "and."
- **Atomic** — it does one thing; it doesn't need a partial rollback if
  interrupted halfway through.
- **Fewer, not more** — prefer fewer, slightly larger functions over many
  tiny ones. Each extra piece is a piece the reader has to hold in their
  head; only split when the split reduces that load, not just to shorten a
  block.

If you can't justify the split, keep the code inline.

## Organize by module, not by function

Think about tasks and functionality in terms of modules first. A new
capability should usually land in the module that owns its concern
(`config`, `logs`, `lease_protocol`, `runtime`, `jobs`, etc.) rather than as
a new top-level file or a grab-bag of helpers. Before adding a function, ask
which module's responsibility it belongs to — that answer usually also
answers whether it should be a new function at all.
