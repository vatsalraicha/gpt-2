"""Shared helpers for I/O, timing, and data loading."""

import json
import time
from pathlib import Path


def load_corpus(jsonl_path: str) -> list[str]:
    """Load all text fields from the JSONL corpus."""
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def load_special_tokens(json_path: str) -> list[str]:
    """Load the special tokens list from special_tokens.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["special_tokens"]


def save_json(obj, path: str) -> None:
    """Write a JSON file with UTF-8 encoding."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_merges(merges: list[tuple[bytes, bytes]], path: str) -> None:
    """Write merge rules to a text file, one per line.
    Each line is two space-separated hex-encoded byte sequences."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for b1, b2 in merges:
            f.write(f"{b1.hex()} {b2.hex()}\n")


def load_merges(path: str) -> list[tuple[bytes, bytes]]:
    """Read merge rules from a text file."""
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h1, h2 = line.split()
            merges.append((bytes.fromhex(h1), bytes.fromhex(h2)))
    return merges


class Timer:
    """Simple context manager for timing blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[{self.label}] {self.elapsed:.2f}s")
