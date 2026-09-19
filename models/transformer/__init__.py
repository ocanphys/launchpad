"""The transformer as a training leg's `model`: ModelParameters is what a leg
declares, `build` turns it into the TransformerLM in src/. Free of torch at
import so declaring an artifact never pays for it; every package under
models/ exports these two names.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParameters:
    vocab_size: int
    sequence_length: int
    num_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    rope_theta: int
    device: str = "cuda"
    dtype: str = "torch.float32"  # resolved worker-side via artifacts.core.locate.locate


def build(parameters: ModelParameters):
    """A freshly initialized TransformerLM from `parameters`, on its device."""
    import torch

    from artifacts.core.locate import locate
    from models.transformer.src.model import TransformerLM

    return TransformerLM(
        vocab_size=parameters.vocab_size,
        context_length=parameters.sequence_length,
        num_layers=parameters.num_layers,
        d_model=parameters.d_model,
        d_ff=parameters.d_ff,
        num_heads=parameters.num_heads,
        rope_theta=parameters.rope_theta,
        device=torch.device(parameters.device),
        dtype=locate(parameters.dtype),
    )
