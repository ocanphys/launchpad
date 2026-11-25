"""The per-step record of a leg: train.jsonl in its folder, one JSON row per
completed step, tagged with the attempt that took it. Every attempt appends,
so steps redone after a crash appear once per attempt that took them. The
file reaches anyone else only when the worker commits under its lease.
"""

import json
import math
from pathlib import Path

import torch

from config import TRAIN_LOG


class StepLog:
    """Records one row per step; `flush` appends them to the file.

    `record` keeps tensor metrics as the device tensors they are and `flush`
    reads them all back at once, so the loop pays one GPU sync per flush
    rather than one per step.
    """

    def __init__(self, root: Path, artifact_path: str):
        self.path = root / artifact_path / TRAIN_LOG
        self.rows: list[dict] = []
        # one past the highest attempt on file; 1 for a leg's first
        previous = [json.loads(line)["attempt"] for line in self.path.read_text().splitlines()] if self.path.exists() else []
        self.attempt = max(previous, default=0) + 1

    def record(self, step: int, metrics: dict[str, torch.Tensor | float]) -> None:
        """Adds one row for `step` holding exactly `metrics`; a metric left out
        of a step is absent from its row.

        Every tensor recorded under one name must share its shape and device,
        since flush stacks them."""
        self.rows.append(
            {
                "step": step,
                "attempt": self.attempt,
                **{name: value.detach() if isinstance(value, torch.Tensor) else value for name, value in metrics.items()},
            }
        )

    def flush(self) -> None:
        """Appends every row recorded since the last flush, tensors read back as
        numbers (nested lists for non-scalars) and a non-finite scalar as null."""
        rows, self.rows = self.rows, []
        if not rows:
            return
        for name in {name for row in rows for name, value in row.items() if isinstance(value, torch.Tensor)}:
            holding = [row for row in rows if isinstance(row.get(name), torch.Tensor)]
            for row, value in zip(holding, torch.stack([row[name] for row in holding]).tolist()):
                row[name] = value
        with self.path.open("a") as f:
            # JSON has no NaN or infinity, and a reader that holds to that fails
            # on the whole file: a diverged value reads as null
            f.writelines(
                json.dumps({name: None if isinstance(value, float) and not math.isfinite(value) else value for name, value in row.items()})
                + "\n"
                for row in rows
            )
