"""
Tokenizer: encode text to token IDs and decode back.

Encoding pipeline:
  1. Split on special tokens (exact match)
  2. Pre-tokenize non-special segments (cl100k_base regex)
  3. For each chunk, convert to bytes and apply merge rules in priority order
  4. Map resulting byte sequences to token IDs

Decoding pipeline:
  1. Map token IDs to byte sequences (or special token strings)
  2. Concatenate all bytes
  3. Decode as UTF-8

Roundtrip guarantee: decode(encode(text)) == text for all valid UTF-8 input.
"""

from pathlib import Path
from bpe.pretokenizer import PreTokenizer
from bpe.utils import load_merges, load_special_tokens, Timer
import json


class Tokenizer:
    """BPE tokenizer with encode/decode supporting special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str],
    ):
        self.vocab = vocab  # id → bytes
        self.id_to_bytes = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges
        # Merge priority: index in list (lower = higher priority)
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = set(special_tokens)
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        for token_str in special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes in self.bytes_to_id:
                tid = self.bytes_to_id[token_bytes]
                self.special_token_to_id[token_str] = tid
                self.id_to_special_token[tid] = token_str

        self.pre_tokenizer = PreTokenizer(list(special_tokens))

    @classmethod
    def from_files(cls, output_dir: str) -> "Tokenizer":
        """Load a trained tokenizer from output directory."""
        output_dir = Path(output_dir)

        # Load config
        with open(output_dir / "tokenizer_config.json", "r") as f:
            config = json.load(f)

        special_tokens = config["special_tokens"]

        # Load vocab
        with open(output_dir / "vocab.json", "r") as f:
            vocab_raw = json.load(f)

        special_set = set(special_tokens)
        vocab: dict[int, bytes] = {}
        for id_str, value in vocab_raw.items():
            token_id = int(id_str)
            if value in special_set:
                vocab[token_id] = value.encode("utf-8")
            else:
                vocab[token_id] = bytes.fromhex(value)

        # Load merges
        merges = load_merges(str(output_dir / "merges.txt"))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        if not text:
            return []

        ids: list[int] = []
        chunks = self.pre_tokenizer.pre_tokenize(text)

        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_token_to_id[chunk])
            else:
                chunk_ids = self._encode_chunk(chunk)
                ids.extend(chunk_ids)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        parts: list[bytes] = []
        for tid in ids:
            if tid in self.id_to_special_token:
                parts.append(self.id_to_special_token[tid].encode("utf-8"))
            elif tid in self.vocab:
                parts.append(self.vocab[tid])
            else:
                raise ValueError(f"Unknown token ID: {tid}")
        return b"".join(parts).decode("utf-8")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def token_to_id(self) -> dict[str, int]:
        """Map displayable token strings to their IDs."""
        result = {}
        for tid, tbytes in self.vocab.items():
            try:
                result[tbytes.decode("utf-8")] = tid
            except UnicodeDecodeError:
                result[repr(tbytes)] = tid
        return result

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a single pre-tokenized chunk (not a special token)."""
        # Convert to list of single-byte tokens
        tokens = [bytes([b]) for b in chunk.encode("utf-8")]

        if len(tokens) < 2:
            return [self.bytes_to_id[t] for t in tokens]

        # Repeatedly merge the highest-priority pair until no more merges apply
        while len(tokens) >= 2:
            # Find the pair with the lowest merge rank (= highest priority)
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # no more applicable merges

            # Merge all occurrences of best_pair
            merged = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.bytes_to_id[t] for t in tokens]
