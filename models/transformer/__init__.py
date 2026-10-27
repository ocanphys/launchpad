"""The transformer as a training leg's model: Transformer is what a leg
declares, and its `build` makes the TransformerLM in src/. Free of torch at
import so declaring an artifact never pays for it.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from artifacts.core.SGD import training


class Transformer(training.Model):
    """A decoder-only transformer LM: the fields are its shapes, and `build` makes it.

        Transformer(vocab_size=1000, sequence_length=32, num_layers=4, d_model=128, d_ff=256, num_heads=16,
                    rope_theta=10000)
        Transformer(..., d_model=100, num_heads=16) -> ValidationError (d_model must be divisible by num_heads)
    """

    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    rope_theta: int = Field(gt=0)

    @model_validator(mode="after")
    def _heads_divide_d_model(self) -> Self:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self

    def build(self):
        """Returns a freshly initialized TransformerLM on this model's device."""
        import torch

        from artifacts.core.locate import locate
        from models.transformer.src.model import TransformerLM

        return TransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.sequence_length,
            num_layers=self.num_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            rope_theta=self.rope_theta,
            device=torch.device(self.device),
            dtype=locate(self.dtype),
        )

    def step_flops(self, batch_size: int) -> int:
        """Returns the matmul FLOPs of one forward and backward pass over batch_size sequences of sequence_length.

        Backward is twice forward. Attention counts the full sequence x sequence scores the causal mask
        then discards; normalization, activations, the loss and the optimizer are not counted.
        """
        tokens = batch_size * self.sequence_length
        attention = 8 * tokens * self.d_model**2 + 4 * tokens * self.sequence_length * self.d_model  # projections, QK^T and AV
        ffn = 6 * tokens * self.d_model * self.d_ff  # three SwiGLU projections
        head = 2 * tokens * self.d_model * self.vocab_size
        return 3 * (self.num_layers * (attention + ffn) + head)
