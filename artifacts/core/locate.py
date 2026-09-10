"""Resolve a dotted path (e.g. "torch.float32", "torch.optim.AdamW",
"numpy.float32") to the live object it names, importing modules along the
way as needed.

Lets artifact definitions store framework types -- torch dtypes, optimizer
classes, numpy dtypes, and so on -- as plain strings, so af.py modules never
need to import torch/numpy just to type a field. The actual import only
happens where the string gets resolved, which is always worker-side code
where the framework is guaranteed installed.
"""

import importlib
from typing import Any


def locate(path: str) -> Any:
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        prefix = ".".join(parts[:split])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as exc:
            # exc.name is the specific module Python couldn't find -- this
            # prefix itself, or (when a middle segment is missing) a shorter
            # one Python needed just to look for this prefix. Either way,
            # nothing of this prefix exists -- try a shorter one. Anything
            # else means a module along the way was found and failed on its
            # own terms, which must surface, not get silently treated as
            # "keep shrinking the prefix".
            if exc.name == prefix or prefix.startswith(exc.name + "."):
                continue
            raise
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"could not resolve {path!r}")
