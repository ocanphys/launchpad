"""Causal flash attention as a torch autograd function with Triton forward and backward kernels."""

import torch
import triton
import triton.language as tl

BLOCK = 64
LOG2E = 1.4426950408889634


@triton.jit
def _fwd_kernel(Q, K, V, O, L, N, qk_scale: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m, bh = tl.program_id(0), tl.program_id(1)
    base = bh * N * D
    rows = m * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + base + rows[:, None] * D + d[None, :], mask=rows[:, None] < N, other=0.0)

    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(0, min((m + 1) * BM, N), BN):
        cols = start + tl.arange(0, BN)
        k = tl.load(K + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < N, other=0.0)
        v = tl.load(V + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where((rows[:, None] >= cols[None, :]) & (cols[None, :] < N), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + base + rows[:, None] * D + d[None, :], acc.to(O.dtype.element_ty), mask=rows[:, None] < N)
    tl.store(L + bh * N + rows, m_i + tl.log2(l_i), mask=rows < N)  # log2-domain logsumexp


@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, DO, L, DELTA, DK, DV, N, scale: tl.constexpr, qk_scale: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr
):
    n, bh = tl.program_id(0), tl.program_id(1)
    base = bh * N * D
    cols = n * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    k = tl.load(K + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < N, other=0.0)
    v = tl.load(V + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < N, other=0.0)

    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    for start in range((n * BN) // BM * BM, N, BM):
        rows = start + tl.arange(0, BM)
        q = tl.load(Q + base + rows[:, None] * D + d[None, :], mask=rows[:, None] < N, other=0.0)
        do = tl.load(DO + base + rows[:, None] * D + d[None, :], mask=rows[:, None] < N, other=0.0)
        l = tl.load(L + bh * N + rows, mask=rows < N, other=0.0)
        delta = tl.load(DELTA + bh * N + rows, mask=rows < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = (rows[:, None] >= cols[None, :]) & (rows[:, None] < N) & (cols[None, :] < N)
        p = tl.where(keep, tl.exp2(s - l[:, None]), 0.0)
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        ds = p * (tl.dot(do, tl.trans(v)) - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

    tl.store(DK + base + cols[:, None] * D + d[None, :], (dk * scale).to(DK.dtype.element_ty), mask=cols[:, None] < N)
    tl.store(DV + base + cols[:, None] * D + d[None, :], dv.to(DV.dtype.element_ty), mask=cols[:, None] < N)


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, DO, L, DELTA, DQ, N, scale: tl.constexpr, qk_scale: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr
):
    m, bh = tl.program_id(0), tl.program_id(1)
    base = bh * N * D
    rows = m * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + base + rows[:, None] * D + d[None, :], mask=rows[:, None] < N, other=0.0)
    do = tl.load(DO + base + rows[:, None] * D + d[None, :], mask=rows[:, None] < N, other=0.0)
    l = tl.load(L + bh * N + rows, mask=rows < N, other=0.0)
    delta = tl.load(DELTA + bh * N + rows, mask=rows < N, other=0.0)

    dq = tl.zeros((BM, D), tl.float32)
    for start in range(0, min((m + 1) * BM, N), BN):
        cols = start + tl.arange(0, BN)
        k = tl.load(K + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < N, other=0.0)
        v = tl.load(V + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = (rows[:, None] >= cols[None, :]) & (cols[None, :] < N)
        p = tl.where(keep, tl.exp2(s - l[:, None]), 0.0)
        ds = p * (tl.dot(do, tl.trans(v)) - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(DQ + base + rows[:, None] * D + d[None, :], (dq * scale).to(DQ.dtype.element_ty), mask=rows[:, None] < N)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, N, D = q.shape
        assert D >= 16 and D & (D - 1) == 0, "head dim must be a power of two >= 16"
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)
        L = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
        ctx.scale = D**-0.5
        _fwd_kernel[(triton.cdiv(N, BLOCK), B * H)](
            q, k, v, o, L, N, ctx.scale * LOG2E, D=D, BM=BLOCK, BN=BLOCK
        )
        ctx.save_for_backward(q, k, v, o, L)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        B, H, N, D = q.shape
        do = do.to(o.dtype).contiguous()
        delta = (do.float() * o.float()).sum(-1).reshape(B * H, N)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        args = (ctx.scale, ctx.scale * LOG2E)
        grid = (triton.cdiv(N, BLOCK), B * H)
        _bwd_dkdv_kernel[grid](q, k, v, do, L, delta, dk, dv, N, *args, D=D, BM=BLOCK, BN=BLOCK)
        _bwd_dq_kernel[grid](q, k, v, do, L, delta, dq, N, *args, D=D, BM=BLOCK, BN=BLOCK)
        return dq, dk, dv


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal softmax(q k^T / sqrt(d)) v for (batch, heads, seq, d_head) CUDA tensors, computed in v's dtype.

    RoPE's fp32 tables promote q and k to fp32 while v stays bf16, so q and k are cast to v's dtype.
    """
    return FlashAttention.apply(q.to(v.dtype), k.to(v.dtype), v)
