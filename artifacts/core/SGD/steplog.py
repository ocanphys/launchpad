"""The per-step record of a leg: train.jsonl in its folder, one JSON row per
completed step, tagged with the attempt that took it. Every attempt appends,
so steps redone after a crash appear once per attempt that took them.
"""

import json
from pathlib import Path

import torch


class StepLog:
    """Records one row per step and appends them to `path` on flush.

    `record` keeps loss and gradient norm as the device tensors they are and
    `flush` reads them all back at once, so the loop pays one GPU sync per
    flush rather than one per step.
    """

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        # one past the highest attempt on file; 1 for a leg's first
        previous = [json.loads(line)["attempt"] for line in path.read_text().splitlines()] if path.exists() else []
        self.attempt = max(previous, default=0) + 1

    def record(self, step: int, loss: torch.Tensor, grad_norm: torch.Tensor, learning_rate: float) -> None:
        self.rows.append(
            {
                "step": step,
                "attempt": self.attempt,
                "loss": loss.detach(),
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
            }
        )

    def flush(self) -> None:
        """Appends every row recorded since the last flush, tensors read back as floats."""
        if not self.rows:
            return
        for name in ("loss", "grad_norm"):
            for row, value in zip(self.rows, torch.stack([row[name] for row in self.rows]).tolist()):
                row[name] = value
        with self.path.open("a") as f:
            f.writelines(json.dumps(row) + "\n" for row in self.rows)
        self.rows = []
