# Transformer

`models.transformer.Transformer`: a decoder-only language model in the usual
modern shape, built from scratch in `models/transformer/network.py`.

## Architecture

Token embedding, `num_layers` pre-norm blocks, a final RMSNorm, and a linear
head onto the vocabulary, untied unless `tie_embeddings` is set. Each block is
two residual sublayers:

- `x + attention(norm1(x))`: multi-head causal self-attention with rotary
  position embeddings (RoPE) on queries and keys, `d_model / num_heads` per head.
- `y + ffn(norm2(y))`: a SwiGLU feed-forward with three matrices (`W1`, `W3`
  up, `W2` down) of width `d_ff`.

Attention is the plain scaled dot product with a causal mask, computed in torch
on every device. With `capture_internals` set, the module also keeps the
attention scores, which is what makes this version the one to inspect.

## Fields

| Field | Meaning |
| --- | --- |
| `num_layers` | Number of blocks. |
| `d_model` | Residual stream width. Must be divisible by `num_heads`. |
| `d_ff` | SwiGLU hidden width, conventionally about `8/3 * d_model`. |
| `num_heads` | Attention heads. |
| `rope_theta` | RoPE base frequency. |
| `tie_embeddings` | `True` makes the head and the input embedding one `(vocab_size, d_model)` matrix, initialized as the head is; `False` by default. |

Plus `vocab_size`, `sequence_length` and `dtype`, as on every model (see
[what readers expect](../models.md#what-readers-expect-of-a-model)). `sequence_length` is also the
context length: the causal mask and RoPE tables are sized to it.

## Notation

The same symbols are used in the code (`parameter_count`, `forward_flops`,
`activation_memory`) and below.

| Symbol | Field or argument | Meaning |
| --- | --- | --- |
| $L$ | `num_layers` | number of transformer layers |
| $B$ | `batch_size` | batch size (sequences) |
| $T$ | `sequence_length` | sequence length (tokens) |
| $d_{model}$ | `d_model` | model width |
| $d_{ff}$ | `d_ff` | MLP hidden width |
| $n_H$ | `num_heads` | number of attention heads |
| $d_h$ | `d_model // num_heads` | head dimension, $d_{model} = n_H\,d_h$ |
| $V$ | `vocab_size` | vocabulary size |

## Parameters

`parameter_count(embedding)`: the count of each submodule times its parameters,
summed. By default it leaves out the input embedding, a lookup with no FLOPs, and
returns $N$; `embedding=True` returns every trainable parameter, $P$, which is
what `estimate_size` sizes the weights from. An untied head is in both: it is
a matmul, counted in the forward FLOPs. A tied head is the embedding, stored
once, so it is in $P$ alone.

| Submodule | Count | Parameters each |
| --- | --- | --- |
| Embedding | $1$ | $Vd_{model}$ |
| Attention ($W_Q, W_K, W_V, W_O$) | $L$ | $4d_{model}^2$ |
| SwiGLU ($W_1, W_2, W_3$) | $L$ | $3d_{model}d_{ff}$ |
| RMSNorm gain, before attention and before SwiGLU | $2L$ | $d_{model}$ |
| Final RMSNorm gain | $1$ | $d_{model}$ |
| Head, untied only | $1$ | $Vd_{model}$ |

$$
N = L\,d_{model}\left(4d_{model} + 3d_{ff} + 2\right) + d_{model} + h\,V d_{model}
\qquad
P = N + V\,d_{model}
$$

where $h$ is 1 for an untied head and 0 for a tied one. The forward FLOPs are
the same either way: a tied head is still a matmul.

## Forward FLOPs

`forward_flops(B)`: the count of each submodule times its FLOPs, summed. Matmuls
only, at 2 FLOPs per weight per token, over the $BT$ tokens of a batch.

| Submodule | Count | FLOPs each |
| --- | --- | --- |
| Embedding (a lookup) | $1$ | $0$ |
| Attention projections ($W_Q, W_K, W_V, W_O$) | $L$ | $8BTd_{model}^2$ |
| Attention scores ($QK^\top$, all $n_H$ heads of width $d_h$, full $T \times T$) | $L$ | $2BT^2d_{model}$ |
| Attention values ($PV$, same shape) | $L$ | $2BT^2d_{model}$ |
| SwiGLU ($W_1, W_2, W_3$) | $L$ | $6BTd_{model}d_{ff}$ |
| RMSNorm, all | | $0$ |
| Head | $1$ | $2BTVd_{model}$ |

$$
F = 2BT\left[L\left(4d_{model}^2 + 2Td_{model} + 3d_{model}d_{ff}\right) + Vd_{model}\right]
$$

The score and value matmuls span the full $T \times T$ matrix, which the causal
mask then discards; $n_Hd_h = d_{model}$. The backward pass is $2F$ (one matmul
for the gradient of each input and one for each weight) and a step is $3F$:
`backward_flops` and `step_flops`. Normalization, activations, the loss and the
optimizer are not counted.

## Activation memory

`activation_memory(B, autocast)`: bytes saved for backward, no recomputation.

**fp32:**

$$
M_{\text{fp32}} = 4BT\left[L\left(8d_{model} + 4d_{ff} + n_H T\right) + 2d_{model} + 2V\right]
$$

**bf16 autocast:**

$$
M_{\text{bf16}} = BT\left[L\left(28d_{model} + 8d_{ff} + 6n_H T\right) + 10d_{model} + 6V\right]
$$

with $n_H T = d_{model}T/d_h$.

- $d_{model}$ terms: RMSNorm inputs and outputs, $Q, K, V$, attention output,
  bf16 casts of matmul inputs
- $d_{ff}$ terms: $W_1$ out, SiLU out, $W_3$ out, $W_2$ input
- $n_H T$ term: softmax probabilities, one $T \times T$ matrix per head (fp32,
  plus a bf16 copy under autocast)
- final terms: last RMSNorm, logits, and loss softmax

Under autocast, matmul inputs and outputs are saved as bf16 while the residual
stream, RMSNorm, softmax and the loss stay fp32; autocast also saves bf16 copies
of the fp32 tensors fed into matmuls (the normalized inputs to $W_Q, W_K, W_V$,
$W_1, W_3$ and the head, and the softmax probabilities fed into $PV$).

This is an estimate of the dominant saved tensors, not a measurement: autograd
also saves small extras (RMSNorm scales, RoPE tables) the formula leaves out.

## FLOPs per byte of saved activations

Per layer, from $F$ and $M$ above:

$$
\begin{aligned}
\text{fp32:}\quad & \frac{F_{\text{lin}}}{M_{\text{lin}}} = \frac{2\left(4d_{model}^2 + 3d_{model}d_{ff}\right)}{32d_{model} + 16d_{ff}},
\qquad \frac{F_{\text{attn}}}{M_{\text{attn}}} = \frac{4T\,n_H\,d_h}{4n_H T} = d_h \\[6pt]
\text{bf16:}\quad & \frac{F_{\text{lin}}}{M_{\text{lin}}} = \frac{2\left(4d_{model}^2 + 3d_{model}d_{ff}\right)}{28d_{model} + 8d_{ff}},
\qquad \frac{F_{\text{attn}}}{M_{\text{attn}}} = \frac{4T\,n_H\,d_h}{6n_H T} = \tfrac{2}{3}\,d_h
\end{aligned}
$$

The linear ratio scales with $d_{model}$. The attention ratio depends only on
$d_h$, so at fixed $d_{model}$, more heads means more activation memory per FLOP.

## Checks

`test/test_transformer_counts.py` pins $P$ to the parameter count of the built
network and $F$ to torch's `FlopCounterMode` on its forward pass.

## Reference

::: models.transformer.Transformer
