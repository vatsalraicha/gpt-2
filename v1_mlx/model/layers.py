"""Transformer building blocks for MLX: CausalSelfAttention, MLP, TransformerBlock.

Architecture: GPT-2 style (pre-norm) with no bias on linear layers.
MLX port of v1/model/layers.py.
"""

import math
import mlx.core as mx
import mlx.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses separate Q, K, V projections (MLX convention), then merges heads.
    Causal mask prevents attending to future positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Separate Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) / self.scale

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        attn_weights = attn_weights + mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, C)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)

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

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.c_fc(x))
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

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
