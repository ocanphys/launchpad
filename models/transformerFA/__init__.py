"""The transformer with flash attention as a training leg's model: TransformerFA
is what a leg declares, and its `build` makes the TransformerLM in src/. Free of
torch at import so declaring an artifact never pays for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import Field, model_validator

from models import Model

if TYPE_CHECKING:
    import torch


class TransformerFA(Model):
    """A decoder-only transformer LM whose attention runs PyTorch's flash attention on CUDA: the
    fields are its shapes, and `build` makes it.

        TransformerFA(vocab_size=1000, sequence_length=32, num_layers=4, d_model=128, d_ff=256, num_heads=16,
                      rope_theta=10000)
        TransformerFA(..., d_model=100, num_heads=16) -> ValidationError (d_model must be divisible by num_heads)
    """

    vocab_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    dtype: str = "torch.float32"  # resolved worker-side via artifacts.core.locate.locate
    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    rope_theta: int = Field(gt=0)
    tie_embeddings: bool = False  # the head and the input embedding are one matrix

    @model_validator(mode="after")
    def _heads_divide_d_model(self) -> Self:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self

    def build(self, device: torch.device):
        """Returns a freshly initialized TransformerLM on `device`."""
        import torch

        from artifacts.core.locate import locate
        from models.transformerFA.src.model import TransformerLM

        return TransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.sequence_length,
            num_layers=self.num_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            rope_theta=self.rope_theta,
            tie_embeddings=self.tie_embeddings,
            device=device,
            dtype=locate(self.dtype),
        )

    def parameter_count(self, embedding: bool = False) -> int:
        """Returns the number of trainable parameters without the input embedding, N in
        docs/model-repository/transformer.md, or with it, P. An untied head is counted either way: it
        is a matmul, where the embedding is a lookup. A tied head is the embedding, so it is in P alone.

            TransformerFA(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2,
                          rope_theta=10000).parameter_count() -> 18608

        Args:
            embedding: Whether to count the input embedding's vocab_size * d_model weights.

        Returns:
            N, or P when embedding is True.
        """
        L, V, d_model, d_ff = self.num_layers, self.vocab_size, self.d_model, self.d_ff
        attention = 4 * d_model**2  # W_Q W_K W_V W_O
        swiglu = 3 * d_model * d_ff  # W_1 W_2 W_3
        norms = 2 * d_model  # two RMSNorm gains per layer
        final_norm = d_model
        head = 0 if self.tie_embeddings else V * d_model  # a tied head is the embedding
        return L * (attention + swiglu + norms) + final_norm + head + (V * d_model if embedding else 0)

    def estimate_size(self) -> dict[str, int]:
        """Returns the parameters (P in docs/model-repository/transformer.md) and the bytes of the weights at
        `dtype`: the theoretical counterpart of `measure_size`.

            TransformerFA(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2,
                          rope_theta=10000).estimate_size() -> {'parameters': 34608, 'bytes': 138432}
        """
        import torch

        from artifacts.core.locate import locate

        parameters = self.parameter_count(embedding=True)
        return {"parameters": parameters, "bytes": parameters * torch.empty((), dtype=locate(self.dtype)).element_size()}

    def forward_flops(self, batch_size: int) -> int:
        """Returns F in docs/model-repository/transformer.md: the matmul FLOPs of one forward pass over batch_size
        sequences, counting the full T x T attention scores the causal mask then discards. Normalization,
        activations, the loss and the optimizer are not counted.

            TransformerFA(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2,
                          rope_theta=10000).forward_flops(4) -> 1204224
        """
        L, B, T, d_model, d_ff, V = (
            self.num_layers, batch_size, self.sequence_length, self.d_model, self.d_ff, self.vocab_size)
        attention = 8 * d_model**2 + 4 * T * d_model  # W_Q W_K W_V W_O; Q K^T and P V over the full T x T scores
        swiglu = 6 * d_model * d_ff  # W_1 W_2 W_3
        head = 2 * V * d_model
        return B * T * (L * (attention + swiglu) + head)

    def backward_flops(self, batch_size: int) -> int:
        """Returns the FLOPs of one backward pass, twice the forward's: one matmul for the gradient of
        each input and one for each weight.

            TransformerFA(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2,
                          rope_theta=10000).backward_flops(4) -> 2408448
        """
        return 2 * self.forward_flops(batch_size)

    def step_flops(self, batch_size: int) -> int:
        """Returns the FLOPs of one forward and backward pass over batch_size sequences.

            TransformerFA(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2,
                          rope_theta=10000).step_flops(4) -> 3612672
        """
        return self.forward_flops(batch_size) + self.backward_flops(batch_size)
