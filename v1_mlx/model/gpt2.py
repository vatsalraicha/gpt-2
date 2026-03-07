"""GPT-2 model definition for MLX.

Architecture (identical to v1 PyTorch version):
  - Token embeddings + learned absolute position embeddings
  - N transformer blocks (pre-norm)
  - Final LayerNorm
  - Output projection (weight-tied with token embeddings)

Weight initialization:
  - MLX default init is uniform, we override with N(0, 0.02)
  - Residual projections scaled by 1/sqrt(2*n_layers)
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from v1_mlx.model.layers import TransformerBlock


@dataclass
class GPT2Config:
    vocab_size: int = 32305
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    context_length: int = 2048
    dropout: float = 0.1
    bias: bool = False

    def num_params(self, count_embeddings: bool = True) -> int:
        """Estimate total parameter count."""
        emb = self.vocab_size * self.d_model
        pos = self.context_length * self.d_model
        # Per block: Q,K,V projections + output proj + MLP + 2x LayerNorm
        attn = 3 * self.d_model * self.d_model + self.d_model * self.d_model
        mlp = self.d_model * self.d_ff + self.d_ff * self.d_model
        ln = 2 * self.d_model
        block = attn + mlp + ln
        final_ln = self.d_model

        total = pos + block * self.n_layers + final_ln
        if count_embeddings:
            total += emb
        return total


class GPT2(nn.Module):
    """GPT-2 language model in MLX."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ]

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with N(0, 0.02) and scale residual projections."""
        std = 0.02
        residual_std = std / math.sqrt(2 * self.config.n_layers)

        new_params = []
        for name, param in tree_flatten(self.parameters()):
            shape = param.shape
            if "weight" in name and len(shape) >= 2:
                if "c_proj" in name:
                    # Residual projection: scale by 1/sqrt(2*n_layers)
                    new_params.append((name, mx.random.normal(shape=shape) * residual_std))
                else:
                    new_params.append((name, mx.random.normal(shape=shape) * std))
            elif "weight" in name and len(shape) == 1:
                # LayerNorm weights: init to 1
                new_params.append((name, mx.ones(shape)))
            elif "bias" in name:
                new_params.append((name, mx.zeros(shape)))
            else:
                new_params.append((name, mx.random.normal(shape=shape) * std))

        self.load_weights(new_params)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # Embeddings
        positions = mx.arange(T)
        tok_emb = self.wte(input_ids)       # (B, T, d_model)
        pos_emb = self.wpe(positions)        # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.ln_f(x)

        # Output projection: weight tying — reuse wte embedding weights
        logits = x @ self.wte.weight.T       # (B, T, vocab_size)

        return logits

    def loss(self, input_ids: mx.array, targets: mx.array) -> mx.array:
        """Compute cross-entropy loss.

        Args:
            input_ids: (B, T) input token IDs
            targets: (B, T) target token IDs

        Returns:
            Scalar loss
        """
        logits = self(input_ids)  # (B, T, vocab_size)
        # Reshape for cross_entropy
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        return nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int = 278,
    ) -> mx.array:
        """Autoregressive text generation.

        Args:
            input_ids: (1, T) prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling (0 = no filtering)
            eos_token_id: Stop generation when this token is produced

        Returns:
            (1, T + generated) token IDs including prompt
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = input_ids[:, -self.config.context_length:]

            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # (1, vocab_size)

            if temperature == 0:
                idx_next = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, logits.shape[-1])
                    top_vals = mx.sort(logits, axis=-1)[:, -k:]
                    threshold = top_vals[:, 0:1]
                    logits = mx.where(logits < threshold, float("-inf"), logits)

                # Sample
                idx_next = mx.random.categorical(logits, axis=-1)
                idx_next = idx_next.reshape(1, 1)

            input_ids = mx.concatenate([input_ids, idx_next], axis=1)
            mx.eval(input_ids)

            # Stop at EOS
            if idx_next.item() == eos_token_id:
                break

        return input_ids

    def save_pretrained(self, path: str):
        """Save model weights and config."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights as safetensors
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(save_dir / "model.safetensors"), weights)

        # Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "d_ff": self.config.d_ff,
            "context_length": self.config.context_length,
            "dropout": self.config.dropout,
            "bias": self.config.bias,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "GPT2":
        """Load model from saved checkpoint."""
        load_dir = Path(path)

        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)

        config = GPT2Config(**config_dict)
        model = cls(config)

        weights = mx.load(str(load_dir / "model.safetensors"))
        model.load_weights(list(weights.items()))

        return model
