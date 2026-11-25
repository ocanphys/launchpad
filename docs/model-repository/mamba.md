# Mamba

`models.mamba.Mamba`: a selective state space language model after Gu & Dao
(2023), in `models/mamba/network.py`.

## Architecture

Token embedding, `num_layers` pre-norm residual blocks, a final RMSNorm and an
untied linear head. Each block is the Mamba mixer:

1. RMSNorm, then an input projection to `2 * expand * d_model`, split into a
   signal `u` and a gate `z`.
2. A causal depthwise convolution of width `d_conv` over `u`, then SiLU.
3. Input-dependent `delta`, `B` and `C` projected from `u`; `A` is the S4D-real
   initialization (`-[1 .. d_state]` per channel), and `delta`'s bias starts
   log-uniform in `[1e-3, 1e-1]`.
4. The selective scan, `h_t = exp(delta_t A) h_{t-1} + delta_t B_t x_t`,
   `y_t = C_t h_t + D x_t`.
5. `y * silu(z)` through an output projection, added to the residual.

The scan is a Python loop over the sequence, so its cost grows with sequence
length in kernel launches, not just FLOPs. There is no fused kernel.

## Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `num_layers` | required | Number of blocks. |
| `d_model` | required | Residual stream width. |
| `d_state` | 16 | State size per channel. |
| `d_conv` | 4 | Convolution width. |
| `expand` | 2 | Inner width is `expand * d_model`. |

Plus `vocab_size`, `sequence_length` and `dtype`, as on every model (see
[what readers expect](../models.md#what-readers-expect-of-a-model)). The recurrence has no context
limit, so `sequence_length` only sets the shape of each batch.

## Counts

`Mamba` defines neither `parameter_count` nor `forward_flops`, so both raise
`NotImplementedError`: a leg trains it, but MFU and the profiler's FLOP report
are unavailable until they are written.

## Reference

::: models.mamba.Mamba
