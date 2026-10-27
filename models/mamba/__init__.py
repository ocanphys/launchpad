"""Mamba as a training leg's model: Mamba is what a leg declares, and its
`build` makes the MambaLM in src/. Free of torch at import so declaring an
artifact never pays for it.
"""

from __future__ import annotations

from pydantic import Field

from artifacts.core.SGD import training


class Mamba(training.Model):
    """A Mamba LM: the fields are its shapes, and `build` makes it. Its
    `sequence_length` is the batch shape only, since the recurrence has no
    context limit.

        Mamba(vocab_size=1000, sequence_length=32, num_layers=2, d_model=64)
          -> Mamba(vocab_size=1000, sequence_length=32, ..., d_state=16, d_conv=4, expand=2)
    """

    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_state: int = Field(default=16, gt=0)
    d_conv: int = Field(default=4, gt=0)
    expand: int = Field(default=2, gt=0)

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
