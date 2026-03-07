"""Data pipeline: load corpus, tokenize, chunk, split into train/val.

Pipeline:
  1. Load JSONL corpus (each line has "text" field with special tokens inline)
  2. Tokenize each record with the BPE tokenizer
  3. Concatenate all token IDs into one long sequence
     (documents already separated by <|endoftext|> tokens)
  4. Chunk into fixed-length sequences of context_length
  5. Split chunks into train and val sets

Each chunk becomes:
  input_ids = chunk[:-1]   (context_length - 1 tokens)
  targets   = chunk[1:]    (shifted by 1 for next-token prediction)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to path so we can import the bpe package
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class TextChunkDataset(Dataset):
    """Dataset of fixed-length token chunks for language modeling."""

    def __init__(self, chunks: np.ndarray):
        """
        Args:
            chunks: (N, context_length) array of token IDs
        """
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": targets}


def tokenize_corpus(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
) -> np.ndarray:
    """Load and tokenize the entire corpus.

    Args:
        corpus_path: Path to books.jsonl
        tokenizer_dir: Path to tokenizer output directory

    Returns:
        1D numpy array of all token IDs concatenated
    """
    from bpe.tokenizer import Tokenizer

    tok = Tokenizer.from_files(str(tokenizer_dir))

    all_ids = []
    n_records = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ids = tok.encode(record["text"])
            all_ids.extend(ids)
            n_records += 1

    all_ids = np.array(all_ids, dtype=np.int32)
    logger.info(f"Tokenized {n_records} records -> {len(all_ids):,} tokens")
    return all_ids


def chunk_tokens(
    token_ids: np.ndarray,
    context_length: int,
) -> np.ndarray:
    """Split a long token sequence into fixed-length chunks.

    Drops the last chunk if it's shorter than context_length.

    Args:
        token_ids: 1D array of token IDs
        context_length: Length of each chunk (includes both input and target)

    Returns:
        (N, context_length) array of chunks
    """
    n_tokens = len(token_ids)
    # We need context_length tokens per chunk (input[:-1] + target = context_length)
    n_chunks = n_tokens // context_length
    # Trim to exact multiple
    trimmed = token_ids[: n_chunks * context_length]
    chunks = trimmed.reshape(n_chunks, context_length)
    logger.info(
        f"Chunked {n_tokens:,} tokens into {n_chunks:,} chunks "
        f"of {context_length} tokens ({n_tokens - len(trimmed)} tokens dropped)"
    )
    return chunks


def train_val_split(
    chunks: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split chunks into train and validation sets.

    Shuffles before splitting so val set isn't all from the end.

    Args:
        chunks: (N, context_length) array
        val_fraction: Fraction to hold out for validation
        seed: Random seed for reproducibility

    Returns:
        (train_chunks, val_chunks) tuple
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(chunks))
    n_val = max(1, int(len(chunks) * val_fraction))
    n_train = len(chunks) - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_chunks = chunks[train_indices]
    val_chunks = chunks[val_indices]

    logger.info(f"Split: {n_train} train, {n_val} val ({val_fraction:.0%} held out)")
    return train_chunks, val_chunks


def prepare_datasets(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
    context_length: int = 2048,
    val_fraction: float = 0.1,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> tuple[TextChunkDataset, TextChunkDataset]:
    """Full data preparation pipeline.

    Optionally caches tokenized data to avoid re-tokenizing on every run.

    Args:
        corpus_path: Path to books.jsonl
        tokenizer_dir: Path to tokenizer output directory
        context_length: Chunk size (each chunk is context_length tokens)
        val_fraction: Fraction for validation
        seed: Random seed
        cache_dir: If provided, cache tokenized array here

    Returns:
        (train_dataset, val_dataset) tuple
    """
    corpus_path = Path(corpus_path)
    tokenizer_dir = Path(tokenizer_dir)

    # Try loading cached tokenized data
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "tokenized.npy"
        if cache_path.exists():
            logger.info(f"Loading cached tokens from {cache_path}")
            all_ids = np.load(cache_path)
            logger.info(f"Loaded {len(all_ids):,} cached tokens")
        else:
            all_ids = tokenize_corpus(corpus_path, tokenizer_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, all_ids)
            logger.info(f"Cached tokens to {cache_path}")
    else:
        all_ids = tokenize_corpus(corpus_path, tokenizer_dir)

    chunks = chunk_tokens(all_ids, context_length)
    train_chunks, val_chunks = train_val_split(chunks, val_fraction, seed)

    return TextChunkDataset(train_chunks), TextChunkDataset(val_chunks)
