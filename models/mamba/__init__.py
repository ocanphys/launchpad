"""Mamba as a training leg's model: Mamba is what a leg declares, and its
`build` makes the MambaLM in src/. Free of torch at import so declaring an
artifact never pays for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from artifacts.core.SGD import training


@dataclass(frozen=True)
class Mamba(training.Model):
    vocab_size: int
    sequence_length: int  # batch shape only: the recurrence has no context limit
    num_layers: int
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dtype: str = (
        "torch.float32"  # resolved worker-side via artifacts.core.locate.locate
    )

    def build(self):
        """Returns a freshly initialized MambaLM on this model's device."""
        import torch

        from artifacts.core.locate import locate
        from models.mamba.src.model import MambaLM

        return MambaLM(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            device=torch.device(self.device),
            dtype=locate(self.dtype),
        )
