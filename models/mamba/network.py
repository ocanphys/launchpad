import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0,
                std=std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0,
                std=1,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # squaring overflows in narrow dtypes
        rms = ((x**2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        return (x / rms).to(in_dtype) * self.gain.to(in_dtype)


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Returns y (batch seq d_inner) of the recurrence h_t = exp(delta_t A) h_{t-1} + delta_t B_t x_t,
    y_t = C_t h_t + D x_t, with x, delta (batch seq d_inner), A (d_inner d_state),
    B, C (batch seq d_state), D (d_inner).

    Runs a Python loop over seq, so its cost grows with sequence length in launches, not just FLOPs.
    """
    dA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
    dBx = einsum(delta, B, x, "b l d, b l n, b l d -> b l d n")
    h = torch.zeros_like(dA[:, 0])
    ys = []
    for t in range(x.shape[1]):
        h = dA[:, t] * h + dBx[:, t]
        ys.append(einsum(h, C[:, t], "b d n, b n -> b d"))
    return torch.stack(ys, dim=1) + x * D


class MambaBlock(nn.Module):
    """Pre-norm residual block around the Mamba mixer of Gu & Dao (2023)."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.in_proj = Linear(d_model, 2 * d_inner, device=device, dtype=dtype)
        self.conv = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1, device=device, dtype=dtype
        )
        self.x_proj = Linear(d_inner, self.dt_rank + 2 * d_state, device=device, dtype=dtype)
        self.dt_proj = Linear(self.dt_rank, d_inner, device=device, dtype=dtype)
        # softplus(dt_bias) starts log-uniform in [1e-3, 1e-1]: the paper's step-size range
        dt = torch.exp(torch.empty(d_inner, device=device).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter((dt + torch.log(-torch.expm1(-dt))).to(dtype))
        # A = -exp(A_log) = -[1, 2, ..., d_state] per channel (S4D-real)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, device=device, dtype=torch.float32))
            .repeat(d_inner, 1)
            .to(dtype)
        )
        self.D = nn.Parameter(torch.ones(d_inner, device=device, dtype=dtype))
        self.out_proj = Linear(d_inner, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.conv(rearrange(u, "b l d -> b d l"))[..., :seq_len]  # drop the right padding: causal
        u = F.silu(rearrange(u, "b d l -> b l d"))
        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt) + self.dt_bias)
        y = selective_scan(u, delta, -torch.exp(self.A_log), B, C, self.D)
        return x + self.out_proj(y * F.silu(z))


class MambaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_features)
        for block in self.blocks:
            x = block(x)
        return self.linear(self.norm(x))
