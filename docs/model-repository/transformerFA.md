# Transformer FA

`models.transformerFA.TransformerFA`: the [Transformer](transformer.md) with
PyTorch's flash attention. Same fields, same parameters, same FLOP count.

## What differs

Architecture, initialization and parameter layout are identical to
[Transformer](transformer.md): from the same seed, `build` makes the same
weights, and on the CPU the two give the same logits. The only change is the
attention call. On a CUDA tensor, when internals are not being captured,
attention runs `torch.nn.functional.scaled_dot_product_attention` with
`is_causal=True`, under `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, so the
$T \times T$ score matrix is never materialized. Everywhere else (the CPU, or
`capture_internals`) it runs the same dense scaled dot product as
[Transformer](transformer.md).

Only the flash backend is allowed, so where it cannot run, the forward pass
raises instead of falling back to another kernel:

- **Precision.** Flash attention takes fp16 or bf16: declare
  `dtype="torch.bfloat16"`, or run under bf16 autocast, as profiling's
  `autocast="torch.bfloat16"` does. An fp32 network without autocast raises.
  RoPE's tables are fp32, so the rotated queries and keys are cast to the
  values' dtype first.
- **GPU.** An Ampere GPU or later (A10, L4, A100, H100); a T4 raises.
- **Head dimension.** `d_model / num_heads` at most 256.

Parameters and forward, backward and step FLOPs are those of
[Transformer](transformer.md#parameters), the full $T \times T$ attention
matmuls included, so MFU compares between the two. The torch profiler gives
the flash kernels no FLOP estimate, so a profiled step's kernel FLOPs leave
attention out. `TransformerFA` has no `activation_memory`: with flash
attention the $T \times T$ probabilities are never saved, so the Transformer's
estimate does not apply.

## Two classes with the same fields

`models.transformer.Transformer` and `models.transformerFA.TransformerFA` have
identical fields, so only the `"class"` path in a manifest tells them apart.
That is why a leg's model is a subclass and not a union of field sets
([Why a subclass per architecture](../models.md#why-a-subclass-per-architecture)).
A leg trained with one is a different leg from the same shapes trained with the
other.

## Checks

`test/test_transformer_fa.py` pins the weights and CPU logits to
Transformer's, and, on an Ampere or later GPU, the flash logits to the dense
ones under bf16 autocast, and an fp32 forward to raising.
`test/test_transformer_counts.py` pins the counts to the built network, as for
Transformer.

## Reference

The class docstrings are on [Models](../models.md#reference).
