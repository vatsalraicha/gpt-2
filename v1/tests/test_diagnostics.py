"""Tests for model diagnostics (v1/scripts/diagnose.py)."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1.model.gpt2 import GPT2, GPT2Config
from v1.scripts.diagnose import (
    extract_attention_weights,
    analyze_attention,
    analyze_embeddings,
    analyze_token_predictions,
    analyze_capacity,
    load_sample_passages,
)


@pytest.fixture
def small_model():
    """Small model for fast testing."""
    config = GPT2Config(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        context_length=32,
        dropout=0.0,
        bias=False,
    )
    model = GPT2(config)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Random token IDs for testing."""
    return torch.randint(0, 100, (1, 16), dtype=torch.long)


@pytest.fixture
def sample_passages():
    """Random passages for testing."""
    return np.random.randint(0, 100, size=(3, 16), dtype=np.int64)


class TestExtractAttentionWeights:
    def test_shapes(self, small_model, sample_input):
        """Attention maps should have correct shapes."""
        attn_maps = extract_attention_weights(small_model, sample_input)
        assert len(attn_maps) == 2  # n_layers=2
        for attn in attn_maps:
            assert attn.shape == (1, 4, 16, 16)  # (B, n_heads, T, T)

    def test_valid_distributions(self, small_model, sample_input):
        """Each row of attention should sum to ~1.0 and be non-negative."""
        attn_maps = extract_attention_weights(small_model, sample_input)
        for attn in attn_maps:
            # Non-negative
            assert (attn >= 0).all()
            # Rows sum to 1
            row_sums = attn.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask(self, small_model, sample_input):
        """Attention should be zero for future positions (upper triangle)."""
        attn_maps = extract_attention_weights(small_model, sample_input)
        for attn in attn_maps:
            T = attn.shape[-1]
            for t in range(T):
                # Positions after t should have zero attention from position t
                future_attn = attn[0, :, t, t + 1:]
                assert torch.allclose(future_attn, torch.zeros_like(future_attn), atol=1e-6)


class TestAnalyzeAttention:
    def test_output_keys(self, small_model, sample_passages):
        """Should return all expected keys."""
        result = analyze_attention(small_model, sample_passages, torch.device("cpu"))
        assert "entropy_matrix" in result
        assert "locality_matrix" in result
        assert "head_types" in result
        assert "avg_entropy" in result
        assert "avg_locality" in result

    def test_matrix_shapes(self, small_model, sample_passages):
        """Entropy and locality matrices should be n_layers x n_heads."""
        result = analyze_attention(small_model, sample_passages, torch.device("cpu"))
        entropy = np.array(result["entropy_matrix"])
        locality = np.array(result["locality_matrix"])
        assert entropy.shape == (2, 4)  # (n_layers, n_heads)
        assert locality.shape == (2, 4)

    def test_entropy_range(self, small_model, sample_passages):
        """Normalized entropy should be in [0, 1]."""
        result = analyze_attention(small_model, sample_passages, torch.device("cpu"))
        entropy = np.array(result["entropy_matrix"])
        assert (entropy >= 0).all() and (entropy <= 1).all()

    def test_locality_range(self, small_model, sample_passages):
        """Locality should be in [0, 1]."""
        result = analyze_attention(small_model, sample_passages, torch.device("cpu"))
        locality = np.array(result["locality_matrix"])
        assert (locality >= 0).all() and (locality <= 1).all()


class TestAnalyzeEmbeddings:
    def test_output_keys(self, small_model):
        """Should return all expected keys."""
        # Mock tokenizer with encode method
        class MockTokenizer:
            def encode(self, text):
                # Return a single token for simplicity
                return [hash(text) % 100]

        result = analyze_embeddings(small_model, MockTokenizer())
        assert "vocab_size" in result
        assert "d_model" in result
        assert "singular_values_top50" in result
        assert "dims_for_90pct" in result
        assert "dims_for_95pct" in result
        assert "dims_for_99pct" in result
        assert "random_baseline_similarity" in result

    def test_svd_dimension_ordering(self, small_model):
        """dims_90 <= dims_95 <= dims_99 <= d_model."""
        class MockTokenizer:
            def encode(self, text):
                return [hash(text) % 100]

        result = analyze_embeddings(small_model, MockTokenizer())
        assert result["dims_for_90pct"] <= result["dims_for_95pct"]
        assert result["dims_for_95pct"] <= result["dims_for_99pct"]
        assert result["dims_for_99pct"] <= result["d_model"]


class TestAnalyzeTokenPredictions:
    def test_output_structure(self, small_model, sample_passages):
        """Should return samples and aggregate."""
        result = analyze_token_predictions(small_model, sample_passages, torch.device("cpu"))
        assert "samples" in result
        assert "aggregate" in result
        assert len(result["samples"]) == len(sample_passages)

    def test_accuracy_range(self, small_model, sample_passages):
        """All accuracy values should be in [0, 1]."""
        result = analyze_token_predictions(small_model, sample_passages, torch.device("cpu"))
        for s in result["samples"]:
            assert 0 <= s["accuracy"] <= 1
            assert 0 <= s["first_half_accuracy"] <= 1
            assert 0 <= s["second_half_accuracy"] <= 1
        agg = result["aggregate"]
        assert 0 <= agg["overall_accuracy"] <= 1

    def test_position_array_length(self, small_model, sample_passages):
        """Position arrays should match passage length - 1."""
        result = analyze_token_predictions(small_model, sample_passages, torch.device("cpu"))
        # All passages are same length, so min_len = passage_length - 1
        expected_len = sample_passages.shape[1] - 1
        assert len(result["aggregate"]["accuracy_by_position"]) == expected_len


class TestAnalyzeCapacity:
    def test_output_structure(self, small_model):
        """Should return global stats and per-layer info."""
        result = analyze_capacity(small_model)
        assert "global" in result
        assert "layers" in result
        assert result["global"]["total_params"] > 0

    def test_layer_count(self, small_model):
        """Should have entries for weight parameters."""
        result = analyze_capacity(small_model)
        # Count weight parameters in the model
        n_weights = sum(1 for name, _ in small_model.named_parameters() if "weight" in name)
        assert len(result["layers"]) == n_weights

    def test_rank_utilization_range(self, small_model):
        """Rank utilization should be in (0, 1]."""
        result = analyze_capacity(small_model)
        for layer in result["layers"]:
            if "rank_utilization" in layer:
                assert 0 < layer["rank_utilization"] <= 1.0

    def test_near_zero_fraction_range(self, small_model):
        """Near-zero fraction should be in [0, 1]."""
        result = analyze_capacity(small_model)
        assert 0 <= result["global"]["near_zero_fraction"] <= 1
        for layer in result["layers"]:
            assert 0 <= layer["near_zero_fraction"] <= 1


class TestLoadSamplePassages:
    def test_shape(self, tmp_path):
        """Should return correct shape."""
        tokens = np.arange(10000, dtype=np.int32)
        np.save(tmp_path / "tokens.npy", tokens)
        passages = load_sample_passages(tmp_path / "tokens.npy", n_passages=5, passage_length=100)
        assert passages.shape == (5, 100)

    def test_deterministic(self, tmp_path):
        """Same seed should give same passages."""
        tokens = np.arange(10000, dtype=np.int32)
        np.save(tmp_path / "tokens.npy", tokens)
        p1 = load_sample_passages(tmp_path / "tokens.npy", n_passages=5, passage_length=100, seed=42)
        p2 = load_sample_passages(tmp_path / "tokens.npy", n_passages=5, passage_length=100, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds(self, tmp_path):
        """Different seeds should give different passages."""
        tokens = np.arange(10000, dtype=np.int32)
        np.save(tmp_path / "tokens.npy", tokens)
        p1 = load_sample_passages(tmp_path / "tokens.npy", n_passages=5, passage_length=100, seed=42)
        p2 = load_sample_passages(tmp_path / "tokens.npy", n_passages=5, passage_length=100, seed=99)
        assert not np.array_equal(p1, p2)
