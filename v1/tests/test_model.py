"""Tests for GPT-2 model architecture."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1.model.gpt2 import GPT2, GPT2Config
from v1.model.layers import CausalSelfAttention, MLP, TransformerBlock


class TestGPT2Config:
    def test_default_config(self):
        config = GPT2Config()
        assert config.vocab_size == 32305
        assert config.d_model == 512
        assert config.n_layers == 8
        assert config.n_heads == 8
        assert config.d_ff == 2048
        assert config.context_length == 2048

    def test_num_params(self):
        config = GPT2Config()
        n = config.num_params()
        # ~43M expected
        assert 40_000_000 < n < 50_000_000, f"Expected ~43M params, got {n:,}"

    def test_num_params_no_embeddings(self):
        config = GPT2Config()
        with_emb = config.num_params(count_embeddings=True)
        without_emb = config.num_params(count_embeddings=False)
        assert with_emb > without_emb
        # Embedding params = vocab_size * d_model
        diff = with_emb - without_emb
        assert diff == config.vocab_size * config.d_model


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(d_model=64, n_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_causal_masking(self):
        """Verify that changing future tokens doesn't affect past outputs."""
        attn = CausalSelfAttention(d_model=64, n_heads=4, dropout=0.0)
        attn.eval()

        x1 = torch.randn(1, 5, 64)
        x2 = x1.clone()
        x2[0, 3:, :] = torch.randn(2, 64)  # Change last 2 positions

        out1 = attn(x1)
        out2 = attn(x2)

        # First 3 positions should be identical
        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5)

    def test_head_dim_assertion(self):
        with pytest.raises(AssertionError):
            CausalSelfAttention(d_model=65, n_heads=4)


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(d_model=64, d_ff=256, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = mlp(x)
        assert out.shape == (2, 10, 64)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_residual_connection(self):
        """Output should be different from input (non-trivial transform)."""
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert not torch.allclose(x, out)


class TestGPT2:
    @pytest.fixture
    def small_model(self):
        config = GPT2Config(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4,
            d_ff=128, context_length=32, dropout=0.0,
        )
        return GPT2(config)

    def test_forward_shape(self, small_model):
        x = torch.randint(0, 100, (2, 16))
        logits, loss = small_model(x)
        assert logits.shape == (2, 16, 100)
        assert loss is None

    def test_forward_with_targets(self, small_model):
        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss = small_model(x, targets=targets)
        assert logits.shape == (2, 16, 100)
        assert loss is not None
        assert loss.item() > 0

    def test_context_length_assertion(self, small_model):
        x = torch.randint(0, 100, (1, 64))  # Exceeds context_length=32
        with pytest.raises(AssertionError):
            small_model(x)

    def test_weight_tying(self, small_model):
        """lm_head weight should be the same object as wte weight."""
        assert small_model.lm_head.weight is small_model.wte.weight

    def test_generate(self, small_model):
        small_model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = small_model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10, eos_token_id=-1)
        assert out.shape[1] == 15  # 5 prompt + 10 generated

    def test_generate_greedy(self, small_model):
        """Greedy generation should be deterministic."""
        small_model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out1 = small_model.generate(prompt.clone(), max_new_tokens=10, temperature=0, eos_token_id=-1)
        out2 = small_model.generate(prompt.clone(), max_new_tokens=10, temperature=0, eos_token_id=-1)
        assert torch.equal(out1, out2)

    def test_save_load_roundtrip(self, small_model, tmp_path):
        """Save and load should produce identical outputs."""
        small_model.eval()
        x = torch.randint(0, 100, (1, 10))

        logits1, _ = small_model(x)

        small_model.save_pretrained(str(tmp_path / "test_ckpt"))
        loaded = GPT2.from_pretrained(str(tmp_path / "test_ckpt"))
        loaded.eval()

        logits2, _ = loaded(x)
        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_residual_scaling(self, small_model):
        """Residual projection weights should have smaller std than other weights."""
        proj_stds = []
        other_stds = []
        for name, p in small_model.named_parameters():
            if "c_proj" in name and "weight" in name:
                proj_stds.append(p.std().item())
            elif "c_attn" in name and "weight" in name:
                other_stds.append(p.std().item())

        # c_proj should have smaller std due to 1/sqrt(2*n_layers) scaling
        avg_proj = sum(proj_stds) / len(proj_stds)
        avg_other = sum(other_stds) / len(other_stds)
        assert avg_proj < avg_other, (
            f"c_proj std ({avg_proj:.4f}) should be smaller than c_attn std ({avg_other:.4f})"
        )
