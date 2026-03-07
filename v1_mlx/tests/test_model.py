"""Tests for MLX GPT-2 model architecture."""

import sys
from pathlib import Path

import pytest
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1_mlx.model.gpt2 import GPT2, GPT2Config
from v1_mlx.model.layers import CausalSelfAttention, MLP, TransformerBlock


class TestGPT2Config:
    def test_default_config(self):
        config = GPT2Config()
        assert config.vocab_size == 32305
        assert config.d_model == 512
        assert config.n_layers == 8

    def test_num_params(self):
        config = GPT2Config()
        n = config.num_params()
        assert 40_000_000 < n < 50_000_000, f"Expected ~43M params, got {n:,}"


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(d_model=64, n_heads=4, dropout=0.0)
        x = mx.random.normal((2, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_causal_masking(self):
        """Verify that changing future tokens doesn't affect past outputs."""
        attn = CausalSelfAttention(d_model=64, n_heads=4, dropout=0.0)

        x1 = mx.random.normal((1, 5, 64))
        x2 = mx.concatenate([x1[:, :3, :], mx.random.normal((1, 2, 64))], axis=1)

        out1 = attn(x1)
        out2 = attn(x2)
        mx.eval(out1, out2)

        diff = mx.abs(out1[0, :3] - out2[0, :3]).max().item()
        assert diff < 1e-5, f"Causal masking failed, diff={diff}"


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(d_model=64, d_ff=256, dropout=0.0)
        x = mx.random.normal((2, 10, 64))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
        x = mx.random.normal((2, 10, 64))
        out = block(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)


class TestGPT2:
    @pytest.fixture
    def small_model(self):
        config = GPT2Config(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4,
            d_ff=128, context_length=32, dropout=0.0,
        )
        return GPT2(config)

    def test_forward_shape(self, small_model):
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        logits = small_model(x)
        mx.eval(logits)
        assert logits.shape == (1, 16, 100)

    def test_loss(self, small_model):
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]])
        loss = small_model.loss(x, targets)
        mx.eval(loss)
        assert loss.item() > 0

    def test_generate(self, small_model):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        out = small_model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10, eos_token_id=-1)
        mx.eval(out)
        assert out.shape[1] == 15  # 5 prompt + 10 generated

    def test_generate_greedy_deterministic(self, small_model):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        out1 = small_model.generate(prompt, max_new_tokens=10, temperature=0, eos_token_id=-1)
        out2 = small_model.generate(prompt, max_new_tokens=10, temperature=0, eos_token_id=-1)
        mx.eval(out1, out2)
        assert mx.array_equal(out1, out2)

    def test_save_load_roundtrip(self, small_model, tmp_path):
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        logits1 = small_model(x)
        mx.eval(logits1)

        small_model.save_pretrained(str(tmp_path / "test_ckpt"))
        loaded = GPT2.from_pretrained(str(tmp_path / "test_ckpt"))

        logits2 = loaded(x)
        mx.eval(logits2)

        diff = mx.abs(logits1 - logits2).max().item()
        assert diff < 1e-5, f"Save/load roundtrip failed, diff={diff}"
