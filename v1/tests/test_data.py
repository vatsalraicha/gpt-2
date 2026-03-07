"""Tests for data pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1.data.dataset import TextChunkDataset, chunk_tokens, train_val_split


class TestChunkTokens:
    def test_basic_chunking(self):
        tokens = np.arange(100, dtype=np.int32)
        chunks = chunk_tokens(tokens, context_length=10)
        assert chunks.shape == (10, 10)
        assert np.array_equal(chunks[0], np.arange(10))
        assert np.array_equal(chunks[1], np.arange(10, 20))

    def test_drops_remainder(self):
        tokens = np.arange(25, dtype=np.int32)
        chunks = chunk_tokens(tokens, context_length=10)
        assert chunks.shape == (2, 10)  # 25 // 10 = 2, drops last 5

    def test_empty_if_too_short(self):
        tokens = np.arange(5, dtype=np.int32)
        chunks = chunk_tokens(tokens, context_length=10)
        assert chunks.shape == (0, 10)


class TestTrainValSplit:
    def test_split_ratio(self):
        chunks = np.arange(100).reshape(10, 10)
        train, val = train_val_split(chunks, val_fraction=0.2, seed=42)
        assert len(train) == 8
        assert len(val) == 2

    def test_no_overlap(self):
        chunks = np.arange(200).reshape(20, 10)
        train, val = train_val_split(chunks, val_fraction=0.2, seed=42)
        # Check no rows are shared
        for v_row in val:
            for t_row in train:
                assert not np.array_equal(v_row, t_row)

    def test_deterministic(self):
        chunks = np.arange(200).reshape(20, 10)
        train1, val1 = train_val_split(chunks, val_fraction=0.2, seed=42)
        train2, val2 = train_val_split(chunks, val_fraction=0.2, seed=42)
        assert np.array_equal(train1, train2)
        assert np.array_equal(val1, val2)

    def test_at_least_one_val(self):
        chunks = np.arange(20).reshape(2, 10)
        train, val = train_val_split(chunks, val_fraction=0.1, seed=42)
        assert len(val) >= 1


class TestTextChunkDataset:
    def test_len(self):
        chunks = np.arange(100, dtype=np.int32).reshape(10, 10)
        ds = TextChunkDataset(chunks)
        assert len(ds) == 10

    def test_getitem_shapes(self):
        chunks = np.arange(100, dtype=np.int32).reshape(10, 10)
        ds = TextChunkDataset(chunks)
        item = ds[0]
        assert item["input_ids"].shape == (9,)   # chunk[:-1]
        assert item["labels"].shape == (9,)       # chunk[1:]

    def test_shift_by_one(self):
        chunks = np.arange(10, dtype=np.int32).reshape(1, 10)
        ds = TextChunkDataset(chunks)
        item = ds[0]
        # input_ids = [0,1,2,3,4,5,6,7,8], labels = [1,2,3,4,5,6,7,8,9]
        assert item["input_ids"][0].item() == 0
        assert item["labels"][0].item() == 1
        assert item["input_ids"][-1].item() == 8
        assert item["labels"][-1].item() == 9


class TestTokenizeCorpus:
    """Test tokenization with real corpus (if available)."""

    @pytest.fixture
    def corpus_path(self):
        path = ROOT / "corpus" / "books.jsonl"
        if not path.exists():
            pytest.skip("Corpus not found")
        return path

    @pytest.fixture
    def tokenizer_dir(self):
        path = ROOT / "tokenizer_output"
        if not (path / "vocab.json").exists():
            pytest.skip("Tokenizer not found")
        return path

    def test_tokenize_first_record(self, corpus_path, tokenizer_dir):
        """Verify we can tokenize at least one record."""
        import json
        from bpe.tokenizer import Tokenizer

        tok = Tokenizer.from_files(str(tokenizer_dir))

        with open(corpus_path, "r") as f:
            first = json.loads(f.readline())

        ids = tok.encode(first["text"])
        assert len(ids) > 0

        # Should start with <|source:book|> (ID 299)
        assert ids[0] == 299, f"Expected first token to be <|source:book|> (299), got {ids[0]}"

        # Should end with <|endoftext|> (ID 278)
        assert ids[-1] == 278, f"Expected last token to be <|endoftext|> (278), got {ids[-1]}"

        # Roundtrip
        decoded = tok.decode(ids)
        assert decoded == first["text"]
