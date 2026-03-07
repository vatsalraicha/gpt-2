"""Transformer building blocks: CausalSelfAttention, MLP, TransformerBlock.

Architecture: GPT-2 style (pre-norm) with no bias on linear layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses a single fused projection for Q, K, V, then splits into heads.
    Causal mask prevents attending to future positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Fused QKV: (B, T, 3*C) -> 3x (B, T, C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # PyTorch's scaled_dot_product_attention handles the causal mask efficiently
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        return self.resid_dropout(self.c_proj(attn_out))


class MLP(nn.Module):
    """Feed-forward network with GELU activation.

    Standard GPT-2: d_model -> 4*d_model -> d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff, bias=bias)
        self.c_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

    Pre-norm (GPT-2 style):
        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, bias)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
