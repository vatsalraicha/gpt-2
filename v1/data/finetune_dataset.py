"""Q&A finetuning dataset.

Loads tokenized Q&A pairs from qa_train.jsonl, pads to a fixed max length,
and masks padding positions in labels with -100 so they're excluded from loss.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class QAFinetuneDataset(Dataset):
    """Dataset for Q&A finetuning.

    Each sample is a tokenized Q&A pair padded to max_length.
    Labels use -100 for padding positions (ignored by cross-entropy loss).
    """

    def __init__(self, token_ids: list[list[int]], max_length: int, pad_id: int = 278):
        """
        Args:
            token_ids: List of token ID sequences (already tokenized).
            max_length: Fixed sequence length to pad/truncate to.
            pad_id: Token ID used for padding (default: EOS=278).
        """
        self.max_length = max_length
        self.pad_id = pad_id
        self.lengths = []  # actual length of each sequence before padding

        # Pad and store as numpy array for Trainer compatibility
        padded = []
        for ids in token_ids:
            seq_len = min(len(ids), max_length)
            self.lengths.append(seq_len)
            if len(ids) >= max_length:
                padded.append(ids[:max_length])
            else:
                padded.append(ids + [pad_id] * (max_length - len(ids)))

        # chunks attribute: (N, max_length) — Trainer uses .chunks.shape[1]
        self.chunks = np.array(padded, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.chunks[idx]
        length = self.lengths[idx]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mask padding: positions >= length-1 in shifted labels are padding
        if length < self.max_length:
            labels[length - 1:] = -100

        return {"input_ids": input_ids, "labels": labels}


def load_qa_data(
    jsonl_path: str | Path,
    tokenizer,
    max_length: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
    pad_id: int = 278,
) -> tuple[QAFinetuneDataset, QAFinetuneDataset]:
    """Load Q&A data from JSONL and create train/val datasets.

    Args:
        jsonl_path: Path to qa_train.jsonl.
        tokenizer: Tokenizer instance with .encode() method.
        max_length: Fixed sequence length (pad/truncate).
        val_fraction: Fraction for validation split.
        seed: Random seed for split.
        pad_id: Padding token ID.

    Returns:
        (train_dataset, val_dataset)
    """
    jsonl_path = Path(jsonl_path)
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} Q&A pairs from {jsonl_path}")

    # Tokenize all records
    all_token_ids = []
    token_lengths = []
    for rec in records:
        ids = tokenizer.encode(rec["text"])
        all_token_ids.append(ids)
        token_lengths.append(len(ids))

    logger.info(
        f"Token lengths: min={min(token_lengths)}, max={max(token_lengths)}, "
        f"avg={sum(token_lengths)/len(token_lengths):.0f}, max_length={max_length}"
    )

    # Warn about truncation
    n_truncated = sum(1 for l in token_lengths if l > max_length)
    if n_truncated > 0:
        logger.warning(f"{n_truncated} sequences will be truncated to {max_length} tokens")

    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_token_ids))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ids = [all_token_ids[i] for i in train_indices]
    val_ids = [all_token_ids[i] for i in val_indices]

    train_dataset = QAFinetuneDataset(train_ids, max_length, pad_id)
    val_dataset = QAFinetuneDataset(val_ids, max_length, pad_id)

    logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")

    return train_dataset, val_dataset
