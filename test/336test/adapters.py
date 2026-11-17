"""Adapters from the CS336 assignment 1 test interface to models.transformer.src.model.

The tests pass weights under the assignment's state-dict names; each adapter loads them into
our module under its own parameter names.
"""

import torch
from torch import Tensor

from models.transformer.src.model import (
    SPDA,
    Embedding,
    Linear,
    MultiHeadAttention,
    RMSNorm,
    RoPE,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
)

BLOCK_KEYS = {
    "attn.q_proj.weight": "attention.WQ",
    "attn.k_proj.weight": "attention.WK",
    "attn.v_proj.weight": "attention.WV",
    "attn.output_proj.weight": "attention.WO",
    "ln1.weight": "norm1.gain",
    "ln2.weight": "norm2.gain",
    "ffn.w1.weight": "ffn.W1",
    "ffn.w2.weight": "ffn.W2",
    "ffn.w3.weight": "ffn.W3",
}


def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    layer = Linear(d_in, d_out)
    layer.load_state_dict({"W": weights})
    return layer(in_features)


def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    layer = Embedding(vocab_size, d_model)
    layer.load_state_dict({"weight": weights})
    return layer(token_ids)


def run_swiglu(
    d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor
) -> Tensor:
    layer = SwiGLU(d_ff=d_ff, d_model=d_model)
    layer.load_state_dict({"W1": w1_weight, "W2": w2_weight, "W3": w3_weight})
    return layer(in_features)


def run_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    return SPDA(Q, K, V, mask)[0]


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    layer = MultiHeadAttention(d_model, num_heads, max_seq_len=in_features.shape[-2])
    layer.load_state_dict({"WQ": q_proj_weight, "WK": k_proj_weight, "WV": v_proj_weight, "WO": o_proj_weight})
    return layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor,
) -> Tensor:
    rope = RoPE(theta, d_model // num_heads, max_seq_len)
    layer = MultiHeadAttention(d_model, num_heads, max_seq_len, rope=rope)
    layer.load_state_dict({"WQ": q_proj_weight, "WK": k_proj_weight, "WV": v_proj_weight, "WO": o_proj_weight})
    return layer(in_features, token_positions)


def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    return RoPE(theta, d_k, max_seq_len)(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    block.load_state_dict({BLOCK_KEYS[k]: v for k, v in weights.items()})
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, d_ff, num_heads, rope_theta, tie_embeddings=False)
    state = {
        "embedding.weight": weights["token_embeddings.weight"],
        "norm.gain": weights["ln_final.weight"],
        "linear.W": weights["lm_head.weight"],
    }
    for i in range(num_layers):
        for theirs, ours in BLOCK_KEYS.items():
            state[f"tblocks.{i}.{ours}"] = weights[f"layers.{i}.{theirs}"]
    model.load_state_dict(state)
    return model(in_indices)


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    layer = RMSNorm(d_model, eps)
    layer.load_state_dict({"gain": weights})
    return layer(in_features)
