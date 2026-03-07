"""
Pre-tokenizer: splits raw text into chunks before BPE operates.

Two-step process:
  1. Split on special token boundaries (exact string match) so they are never
     fragmented by the regex.
  2. Apply the cl100k_base regex to each non-special segment, splitting text
     into words, numbers, punctuation, and whitespace chunks.

BPE merges can only happen within a chunk — never across chunk boundaries.
"""

import json
import regex

# GPT-4 / cl100k_base pre-tokenization pattern.
# Requires the `regex` library (not stdlib `re`) for possessive quantifiers
# (?+, ++) and inline flags (?i:...).
PRETOK_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
)


def _build_special_token_pattern(special_tokens: list[str]) -> regex.Pattern:
    """Build a regex that matches any special token, longest-first."""
    # Sort by length descending so longer tokens match first
    # (e.g. <|code_start:python|> before <|code_start|>)
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped = [regex.escape(t) for t in sorted_tokens]
    return regex.compile("|".join(escaped))


class PreTokenizer:
    """Splits text into chunks suitable for BPE training/encoding."""

    def __init__(self, special_tokens: list[str]):
        self.special_tokens = set(special_tokens)
        self._special_pattern = _build_special_token_pattern(special_tokens)

    @classmethod
    def from_json(cls, path: str) -> "PreTokenizer":
        """Load special tokens from a JSON file and return a PreTokenizer."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["special_tokens"])

    def pre_tokenize(self, text: str) -> list[str]:
        """Split text into chunks. Special tokens are returned as-is;
        regular text segments are split by the cl100k_base regex."""
        if not text:
            return []

        chunks: list[str] = []
        last_end = 0

        for m in self._special_pattern.finditer(text):
            start, end = m.span()

            # Process regular text before this special token
            if start > last_end:
                segment = text[last_end:start]
                chunks.extend(self._regex_split(segment))

            # Add the special token as-is
            chunks.append(m.group())
            last_end = end

        # Process any remaining regular text after the last special token
        if last_end < len(text):
            chunks.extend(self._regex_split(text[last_end:]))

        return chunks

    def _regex_split(self, text: str) -> list[str]:
        """Apply the cl100k_base regex to a segment of regular text."""
        return PRETOK_PATTERN.findall(text)
