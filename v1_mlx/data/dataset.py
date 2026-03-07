"""Data pipeline for MLX: load corpus, tokenize, chunk, split, iterate.

MLX doesn't have a DataLoader, so we implement a simple batch iterator.
Tokenization and chunking are identical to the PyTorch version.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def tokenize_corpus(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
) -> np.ndarray:
    """Load and tokenize the entire corpus. Returns 1D numpy array of token IDs."""
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


def chunk_tokens(token_ids: np.ndarray, context_length: int) -> np.ndarray:
    """Split a long token sequence into fixed-length chunks."""
    n_tokens = len(token_ids)
    n_chunks = n_tokens // context_length
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
    """Split chunks into train and validation sets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(chunks))
    n_val = max(1, int(len(chunks) * val_fraction))
    n_train = len(chunks) - n_val

    train_chunks = chunks[indices[:n_train]]
    val_chunks = chunks[indices[n_train:]]

    logger.info(f"Split: {n_train} train, {n_val} val ({val_fraction:.0%} held out)")
    return train_chunks, val_chunks


def batch_iterator(chunks: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 42):
    """Yield batches of (input_ids, targets) as MLX arrays.

    Each chunk is split: input_ids = chunk[:-1], targets = chunk[1:]

    Args:
        chunks: (N, context_length) numpy array
        batch_size: Batch size
        shuffle: Whether to shuffle each epoch
        seed: Random seed for shuffling

    Yields:
        (input_ids, targets) as mx.array, each (B, context_length-1)
    """
    n = len(chunks)
    rng = np.random.RandomState(seed)

    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = chunks[batch_idx]

        input_ids = mx.array(batch[:, :-1])
        targets = mx.array(batch[:, 1:])
        yield input_ids, targets


def prepare_data(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
    context_length: int = 2048,
    val_fraction: float = 0.1,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full data preparation pipeline.

    Returns:
        (train_chunks, val_chunks) as numpy arrays
    """
    corpus_path = Path(corpus_path)
    tokenizer_dir = Path(tokenizer_dir)

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
    return train_chunks, val_chunks
