"""The transformer as a training leg's model: Transformer is what a leg
declares, and its `build` makes the TransformerLM in src/. Free of torch at
import so declaring an artifact never pays for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from artifacts.core.SGD import training


@dataclass(frozen=True)
class Transformer(training.Model):
    vocab_size: int
    sequence_length: int
    num_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    rope_theta: int
    dtype: str = (
        "torch.float32"  # resolved worker-side via artifacts.core.locate.locate
    )

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError(
                "d_model must be positive and divisible by positive num_heads"
            )

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
