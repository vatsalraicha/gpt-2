"""GPT-2 model definition with configuration.

Architecture:
  - Token embeddings + learned absolute position embeddings
  - N transformer blocks (pre-norm)
  - Final LayerNorm
  - Output projection (weight-tied with token embeddings)

Weight initialization:
  - All Linear/Embedding: N(0, 0.02)
  - Residual projections (c_proj in attn and mlp): scaled by 1/sqrt(2*n_layers)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from v1_nvidia.model.layers import TransformerBlock


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
        # Token embeddings
        emb = self.vocab_size * self.d_model
        # Position embeddings
        pos = self.context_length * self.d_model
        # Per transformer block
        # Attention: c_attn (d_model -> 3*d_model) + c_proj (d_model -> d_model)
        attn = self.d_model * 3 * self.d_model + self.d_model * self.d_model
        # MLP: c_fc (d_model -> d_ff) + c_proj (d_ff -> d_model)
        mlp = self.d_model * self.d_ff + self.d_ff * self.d_model
        # LayerNorms: 2 per block, each has d_model params (weight only if no bias)
        ln = 2 * self.d_model
        block = attn + mlp + ln
        # Final LayerNorm
        final_ln = self.d_model
        # Output head is tied with embeddings, so no extra params

        total = pos + block * self.n_layers + final_ln
        if count_embeddings:
            total += emb
        return total


class GPT2(nn.Module):
    """GPT-2 language model."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head — weight tied with wte
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) — GPT-2 convention
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        """Initialize weights with N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs
            targets: (B, T) target token IDs for loss computation

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context length {self.config.context_length}"
        )

        # Embeddings
        positions = torch.arange(T, device=input_ids.device)
        tok_emb = self.wte(input_ids)       # (B, T, d_model)
        pos_emb = self.wpe(positions)        # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)             # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int = 278,
    ) -> torch.Tensor:
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
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = input_ids[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature == 0:
                # Greedy
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, idx_next], dim=1)

            # Stop at EOS
            if idx_next.item() == eos_token_id:
                break

        return input_ids

    def save_pretrained(self, path: str):
        """Save model weights and config."""
        import json
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_dir / "model.pt")

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
    def from_pretrained(cls, path: str, device: torch.device | None = None) -> "GPT2":
        """Load model from saved checkpoint."""
        import json
        from pathlib import Path

        load_dir = Path(path)

        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)

        config = GPT2Config(**config_dict)
        model = cls(config)

        state_dict = torch.load(
            load_dir / "model.pt",
            map_location=device or "cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        if device:
            model = model.to(device)

        return model
