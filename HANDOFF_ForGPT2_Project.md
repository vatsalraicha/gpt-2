# BPE Tokenizer — Handoff Document for GPT-2 Pretraining

> **Purpose of this document**: This is a complete briefing for a Claude Code session
> whose job is to **build and pretrain a GPT-2 model** using the custom BPE tokenizer
> and curated corpus produced by this project. Everything you need to know about the
> tokenizer, its vocabulary, its file formats, how to load and use it, and the corpus
> it was trained on is in this file.

---

## Table of Contents

1. [Project Lifecycle](#1-project-lifecycle)
2. [What Was Built](#2-what-was-built)
3. [Tokenizer Statistics](#3-tokenizer-statistics)
4. [Files to Copy to the GPT-2 Project](#4-files-to-copy-to-the-gpt-2-project)
5. [Token ID Layout](#5-token-id-layout)
6. [Vocabulary File Formats](#6-vocabulary-file-formats)
7. [How to Load the Tokenizer in Python](#7-how-to-load-the-tokenizer-in-python)
8. [Encoding and Decoding API](#8-encoding-and-decoding-api)
9. [Special Tokens — Full Specification](#9-special-tokens--full-specification)
10. [Special Tokens Relevant to Pretraining](#10-special-tokens-relevant-to-pretraining)
11. [Corpus Format — What the Model Trains On](#11-corpus-format--what-the-model-trains-on)
12. [Corpus Statistics](#12-corpus-statistics)
13. [Pre-tokenization — How Text is Split Before BPE](#13-pre-tokenization--how-text-is-split-before-bpe)
14. [Stray Token Escaping](#14-stray-token-escaping)
15. [Evaluation Results — Tokenizer Quality](#15-evaluation-results--tokenizer-quality)
16. [Design Decisions and Their Rationale](#16-design-decisions-and-their-rationale)
17. [How to Tokenize the Corpus for Training](#17-how-to-tokenize-the-corpus-for-training)
18. [Key Considerations for Pretraining](#18-key-considerations-for-pretraining)
19. [Tokenizer Source Code Architecture](#19-tokenizer-source-code-architecture)
20. [Dependencies](#20-dependencies)

---

## 1. Project Lifecycle

We are pretraining a GPT-2 scale language model on technical books. The full
lifecycle is:

1. **Curate text** from 72 EPUB ebooks — DONE (previous session)
2. **Train a BPE tokenizer** on the curated text — DONE (this project)
3. **Pretrain the GPT-2 model** using the tokenizer and the corpus — YOUR job (next session)

---

## 2. What Was Built

A complete BPE tokenizer built from scratch in Python — no external tokenizer
libraries (`tiktoken`, `sentencepiece`, HuggingFace `tokenizers` were NOT used).

The tokenizer:
- Uses the GPT-4 / cl100k_base pre-tokenization regex pattern
- Was trained with 32,000 BPE merges on a 46 MB corpus of 62 technical books
- Has 32,305 total vocabulary entries
- Supports 49 special tokens (inline structural markers for code, headings, figures, etc.)
- Has 100% roundtrip accuracy on all 1,094 corpus records
- Achieves 1.47 tokens/word fertility and 4.77 bytes/token compression

---

## 3. Tokenizer Statistics

| Metric | Value |
|---|---|
| Total vocabulary size | 32,305 |
| Byte-level tokens | 256 (IDs 0–255) |
| Special tokens | 49 (IDs 256–304) |
| BPE merge tokens | 32,000 (IDs 305–32,304) |
| Fertility (tokens/word) | 1.47 (corpus average) |
| Compression (bytes/token) | 4.77 |
| Vocab coverage | 97.2% (only 2.8% dead tokens) |
| Corpus records | 1,094 chapters from 62 books |
| Total corpus tokens | 9,973,151 |
| Total corpus words | 6,804,045 |
| Training time | 100.5 seconds |
| Roundtrip accuracy | 100% on all 1,094 records |

---

## 4. Files to Copy to the GPT-2 Project

You need to copy **two directories** from the BPE Tokenizer project:

### From `BPE_Tokenizer/bpe/` — The tokenizer package (4 files)

These are the Python source files that implement the tokenizer. Copy the entire
`bpe/` directory into your GPT-2 project:

| File | Description |
|---|---|
| `bpe/__init__.py` | Empty init (makes it a package) |
| `bpe/pretokenizer.py` | Pre-tokenization: special token splitting + cl100k_base regex |
| `bpe/tokenizer.py` | `Tokenizer` class with `encode()`, `decode()`, `from_files()` |
| `bpe/utils.py` | I/O helpers: `load_corpus()`, `load_merges()`, `load_special_tokens()`, `Timer` |

**Note:** `bpe/trainer.py` is NOT needed for pretraining — it was only used to
train the tokenizer. You can skip it.

### From `BPE_Tokenizer/output/` — The trained tokenizer artifacts (3 files)

These are the serialized vocabulary, merge rules, and configuration. Copy the
entire `output/` directory (or at minimum these 3 files):

| File | Size | Description |
|---|---|---|
| `output/vocab.json` | 865 KB | Token ID → hex-encoded byte string (or special token string) |
| `output/merges.txt` | 463 KB | 32,000 merge rules, one per line, hex-encoded |
| `output/tokenizer_config.json` | 1.5 KB | Metadata: vocab size, special tokens list, regex pattern |

### From `BPE_Tokenizer/corpus/` — The training corpus (already exists)

The corpus files live at `/Users/vr/Code/LLMs/BPE_Tokenizer/corpus/`. These were
produced by the CurateText project and are shared between the tokenizer and GPT-2
projects:

| File | Size | Description |
|---|---|---|
| `corpus/books.jsonl` | 46 MB | Primary corpus: 1,094 JSON records (text + meta) |
| `corpus/special_tokens.json` | 1.4 KB | 49 special token strings |
| `corpus/token_counts.csv` | 777 B | Special token frequency table |

### Summary — What to copy

```
GPT2_Project/
  bpe/                          # Copy from BPE_Tokenizer/bpe/
    __init__.py
    pretokenizer.py
    tokenizer.py
    utils.py

  tokenizer_output/             # Copy from BPE_Tokenizer/output/
    vocab.json
    merges.txt
    tokenizer_config.json

  corpus/                       # Copy from BPE_Tokenizer/corpus/
    books.jsonl
    special_tokens.json
    token_counts.csv
```

---

## 5. Token ID Layout

```
IDs 0–255:       Byte-level tokens (raw bytes 0x00–0xFF)
IDs 256–304:     Special tokens (49 tokens, sorted alphabetically)
IDs 305–32,304:  BPE merge tokens (learned during training)
```

**Total: 32,305 token IDs.**

This layout means:
- The embedding layer in GPT-2 should have 32,305 entries
- IDs 0–255 represent individual bytes — the base vocabulary
- IDs 256–304 represent special tokens — fixed, never merged
- IDs 305+ represent learned subword tokens from BPE merges

---

## 6. Vocabulary File Formats

### `vocab.json`

A JSON object mapping string token IDs to their values. Values are either:
- **Hex-encoded byte strings** for byte tokens and BPE merges
- **Literal special token strings** for special tokens

```json
{
  "0": "00",                        // byte 0x00
  "32": "20",                       // byte 0x20 (space)
  "97": "61",                       // byte 0x61 ('a')
  "256": "<|/figure|>",             // special token (literal string, not hex)
  "278": "<|endoftext|>",           // special token
  "299": "<|source:book|>",         // special token
  "305": "2074",                    // BPE merge: bytes 0x20 0x74 = ' t'
  "306": "696e",                    // BPE merge: bytes 0x69 0x6e = 'in'
  "500": "726573",                  // BPE merge: 'res'
  "1000": "2073616d65"              // BPE merge: ' same'
}
```

**How to distinguish special tokens from hex:** If the value is in the special
tokens list (from `tokenizer_config.json`), it's a literal string. Otherwise,
it's hex-encoded bytes. The `Tokenizer.from_files()` method handles this
automatically.

### `merges.txt`

One merge rule per line. Each line has two space-separated hex-encoded byte
sequences representing the pair that was merged:

```
20 74       ← merge #1:  bytes(0x20) + bytes(0x74) = ' ' + 't' → ' t'
69 6e       ← merge #2:  bytes(0x69) + bytes(0x6e) = 'i' + 'n' → 'in'
20 61       ← merge #3:  bytes(0x20) + bytes(0x61) = ' ' + 'a' → ' a'
68 65       ← merge #4:  bytes(0x68) + bytes(0x65) = 'h' + 'e' → 'he'
61 74       ← merge #5:  bytes(0x61) + bytes(0x74) = 'a' + 't' → 'at'
```

Merge priority is determined by line order — merge #1 (line 1) has the highest
priority. During encoding, when multiple merges could apply, the one with the
lowest line number wins.

32,000 lines total.

### `tokenizer_config.json`

```json
{
  "vocab_size": 32305,
  "num_byte_tokens": 256,
  "num_special_tokens": 49,
  "num_merges": 32000,
  "special_tokens": ["<|/figure|>", "<|book_title|>", ... ],  // all 49, sorted
  "pretok_pattern": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+..."
}
```

---

## 7. How to Load the Tokenizer in Python

### Option A: Use the `bpe` package directly (recommended)

```python
from bpe.tokenizer import Tokenizer

# Load from the output directory
tok = Tokenizer.from_files("tokenizer_output/")

# Encode text → token IDs
ids = tok.encode("Hello world")
# [13022, 2716]

# Decode token IDs → text
text = tok.decode(ids)
# "Hello world"

# Vocabulary size (for embedding layer)
print(tok.vocab_size)  # 32305
```

### Option B: Load the artifacts manually

If you want to build a PyTorch embedding without using the `bpe` package:

```python
import json

# Load config
with open("tokenizer_output/tokenizer_config.json") as f:
    config = json.load(f)

vocab_size = config["vocab_size"]  # 32305 — use this for nn.Embedding
special_tokens = config["special_tokens"]  # list of 49 strings

# Load vocab for ID ↔ bytes mapping
with open("tokenizer_output/vocab.json") as f:
    vocab_raw = json.load(f)

special_set = set(special_tokens)
vocab = {}
for id_str, value in vocab_raw.items():
    token_id = int(id_str)
    if value in special_set:
        vocab[token_id] = value.encode("utf-8")
    else:
        vocab[token_id] = bytes.fromhex(value)
```

---

## 8. Encoding and Decoding API

### `Tokenizer.encode(text: str) → list[int]`

Encoding pipeline:
1. **Split on special tokens** — find all 49 special tokens by exact string match (longest first)
2. **Pre-tokenize** non-special segments with the cl100k_base regex
3. **Apply BPE merges** to each chunk: convert to bytes, repeatedly merge the highest-priority pair
4. **Map** resulting byte sequences to token IDs

```python
tok.encode("<|source:book|>\nHello world")
# → [299, 10, 13022, 2716]
#    ^^^  ^^  ^^^^^  ^^^^
#    special  \n   Hello  _world
```

### `Tokenizer.decode(ids: list[int]) → str`

Decoding pipeline:
1. **Map** each token ID to its byte sequence (or special token string)
2. **Concatenate** all bytes
3. **Decode** as UTF-8

### `Tokenizer.encode_batch(texts: list[str]) → list[list[int]]`

Encodes multiple texts. Convenience method.

### Key properties

```python
tok.vocab_size           # 32305 (int)
tok.special_tokens       # set of 49 strings
tok.special_token_to_id  # {"<|endoftext|>": 278, ...}
tok.id_to_special_token  # {278: "<|endoftext|>", ...}
tok.vocab                # {0: b'\x00', 1: b'\x01', ..., 278: b'<|endoftext|>', ...}
```

---

## 9. Special Tokens — Full Specification

All 49 special tokens. Every one is registered as a single, unsplittable token.
Each encodes to exactly 1 token ID.

### Document-Level (5 tokens)

| Token | ID | Purpose |
|---|---|---|
| `<|startoftext|>` | 304 | BOS token. **Reserved for inference only** — NOT in training corpus. |
| `<|endoftext|>` | 278 | End of document. Every record ends with this. |
| `<|source:book|>` | 299 | Source type. Every record starts with this. |
| `<|book_title|>` | 257 | Next line is the book title. |
| `<|chapter|>` | 258 | Next line is the chapter title. |

### Code (21 tokens)

| Token | ID | Purpose |
|---|---|---|
| `<|code_start|>` | 277 | Code block (language unknown) |
| `<|code_start:python|>` | 268 | Python code block |
| `<|code_start:r|>` | 269 | R code block |
| `<|code_start:sql|>` | 272 | SQL code block |
| `<|code_start:shell|>` | 271 | Shell/bash code block |
| `<|code_start:javascript|>` | 265 | JavaScript code block |
| `<|code_start:json|>` | 267 | JSON code block |
| `<|code_start:yaml|>` | 276 | YAML code block |
| `<|code_start:dockerfile|>` | 262 | Dockerfile code block |
| `<|code_start:java|>` | 266 | Java (reserved, unused) |
| `<|code_start:scala|>` | 270 | Scala (reserved, unused) |
| `<|code_start:html|>` | 263 | HTML (reserved, unused) |
| `<|code_start:xml|>` | 275 | XML (reserved, unused) |
| `<|code_start:css|>` | 261 | CSS (reserved, unused) |
| `<|code_start:toml|>` | 274 | TOML (reserved, unused) |
| `<|code_start:ini|>` | 264 | INI (reserved, unused) |
| `<|code_start:text|>` | 273 | Text (reserved, unused) |
| `<|code_start:bash|>` | 260 | Bash (reserved, unused) |
| `<|code_end|>` | 259 | End of ANY code block (shared) |
| `<|inline_code_start|>` | 287 | Inline code within prose |
| `<|inline_code_end|>` | 286 | End of inline code |

### Headings (6 tokens)

| Token | ID | Purpose |
|---|---|---|
| `<|heading:1|>` | 280 | Level 1 (chapter titles) |
| `<|heading:2|>` | 281 | Level 2 (sections) |
| `<|heading:3|>` | 282 | Level 3 (subsections) |
| `<|heading:4|>` | 283 | Level 4 |
| `<|heading:5|>` | 284 | Level 5 |
| `<|heading:6|>` | 285 | Level 6 |

### Figures / Tables (4 tokens)

| Token | ID | Purpose |
|---|---|---|
| `<|figure|>` | 279 | Start of figure |
| `<|/figure|>` | 256 | End of figure |
| `<|table_start|>` | 303 | Start of table |
| `<|table_end|>` | 302 | End of table |

### Notes (6 tokens)

| Token | ID | Purpose |
|---|---|---|
| `<|note_start|>` | 293 | Generic (reserved, unused) |
| `<|note_start:note|>` | 291 | Note callout |
| `<|note_start:tip|>` | 294 | Tip callout |
| `<|note_start:warning|>` | 295 | Warning (reserved, unused) |
| `<|note_start:info|>` | 290 | Info callout |
| `<|note_start:sidebar|>` | 292 | Sidebar |
| `<|note_end|>` | 289 | End of any note (shared) |

### Quotes (2 tokens)

| Token | ID | Purpose |
|---|---|---|
| `<|quote_start|>` | 297 | Blockquote start |
| `<|quote_end|>` | 296 | Blockquote end |

### Math (1 token)

| Token | ID | Purpose |
|---|---|---|
| `<|math|>` | 288 | Math expression placeholder |

### Future Sources (3 tokens, reserved, unused)

| Token | ID | Purpose |
|---|---|---|
| `<|source:article|>` | 298 | Blog post/article |
| `<|source:docs|>` | 300 | Technical docs |
| `<|source:paper|>` | 301 | Academic paper |

---

## 10. Special Tokens Relevant to Pretraining

For the GPT-2 model, these are the most important special tokens to understand:

### `<|endoftext|>` (ID 278)
Every training record ends with this token. The model should learn to stop
generating when it produces this token. During training, this serves as the
document separator. You can concatenate multiple records and the model learns
document boundaries from this token.

### `<|startoftext|>` (ID 304)
**Not present in the training corpus.** Reserved as a BOS (beginning-of-sequence)
token for inference. During inference, prepend this token to signal "start
generating." During training, documents start with `<|source:book|>` instead.

### `<|source:book|>` (ID 299)
Every training record starts with this. The model learns this as "a new document
is beginning." Future corpus expansion may add `<|source:article|>`, etc.

### `<|code_start:*|>` / `<|code_end|>`
These bracket code blocks. The model will learn to switch between prose mode and
code mode. Inside code blocks, whitespace (indentation) is semantically
significant and preserved exactly.

### Padding Token
**There is no padding token defined.** If you need padding for batched training,
you have two options:
1. Use `<|endoftext|>` (ID 278) as the pad token and mask it in the loss
2. Define a new padding token at a new ID (32,305) — but this changes the
   embedding layer size

---

## 11. Corpus Format — What the Model Trains On

### JSONL format (`corpus/books.jsonl`)

Each line is a JSON object with two fields:

```json
{
  "text": "<|source:book|>\n<|book_title|>\nBook Title Here\n<|chapter|>\nChapter Title\n\n<|heading:1|> Chapter Title\n\nProse text here...\n\n<|endoftext|>",
  "meta": {
    "source": "book",
    "title": "Book Title Here",
    "author": "Author Name",
    "isbn": "978...",
    "publisher": "Publisher",
    "chapter": "Chapter Title",
    "chapter_number": 1,
    "content_types": ["code:python", "code_block", "heading", "table"],
    "format": "epub",
    "word_count": 5000,
    "language": "en-GB",
    "template": "template_a"
  }
}
```

**Key points:**
- `text` is what the model trains on — contains all special tokens inline
- `meta` is never seen by the model — use for sampling weights, filtering, analysis
- Every record starts with `<|source:book|>` and ends with `<|endoftext|>`
- `meta.content_types` lists what content types appear in the chapter
- `meta.word_count` is useful for weighted sampling

### Document structure

```
<|source:book|>
<|book_title|>
The Actual Book Title
<|chapter|>
Chapter Title

<|heading:1|> Chapter Title

Plain English prose (the default — no wrapping tokens).

<|heading:2|> A Subsection

More prose.

<|code_start:python|>
def example():
    return 42
<|code_end|>

Use <|inline_code_start|>example()<|inline_code_end|> to call it.

<|figure|>
[Figure]
Caption: Some caption
<|/figure|>

<|table_start|>
Col A || Col B || Col C
val 1 | val 2 | val 3
<|table_end|>

<|note_start:tip|>
This is a tip callout.
<|note_end|>

<|endoftext|>
```

**Critical:** Plain prose has NO wrapping tokens. It is the default, unmarked
content. Special tokens only mark "mode switches" into structured content.

---

## 12. Corpus Statistics

| Metric | Value |
|---|---|
| Books | 62 |
| Chapter records | 1,094 |
| Total words | 6,804,045 |
| Total tokens (encoded) | 9,973,151 |
| Total bytes | 47,543,428 (46 MB) |
| Special tokens defined | 49 (34 used, 15 reserved) |
| Content types | 20 (prose, code in 9 languages, tables, figures, math, etc.) |

### Sequence length distribution (tokens per record)

| Metric | Value |
|---|---|
| Min | 84 |
| Max | 110,306 |
| Mean | 9,116 |
| Median | 8,335 |
| Std | 7,796 |
| P90 | 17,091 |
| P95 | 19,819 |
| P99 | 35,710 |

**Implication for context window:** Most records fit in a 16K or 32K context
window. The longest record (110K tokens) will need truncation or chunking.

### Content-type fertility

| Category | Records | Fertility (tok/word) |
|---|---|---|
| Prose only | 63 | 1.420 |
| Code heavy | 936 | 1.466 |
| Has Python | 790 | 1.487 |
| Has SQL | 105 | 1.417 |
| Has tables | 868 | 1.480 |

---

## 13. Pre-tokenization — How Text is Split Before BPE

The tokenizer uses the **GPT-4 / cl100k_base** regex pattern for pre-tokenization:

```python
import regex
PRETOK_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
```

This requires the `regex` library (not stdlib `re`) for possessive quantifiers
and inline flags.

**Two-step process:**
1. Split on special token boundaries (exact match, longest first)
2. Apply the regex to each non-special segment

This ensures special tokens are never fragmented and BPE merges never cross
word boundaries.

---

## 14. Stray Token Escaping

Books about LLMs discuss tokens like `<|endoftext|>` in their text. These are
escaped with Unicode box-drawing vertical `│` (U+2502):

```
Original in book:  <|endoftext|>
Escaped in corpus:  <│endoftext│>
```

The tokenizer treats `<│...│>` as regular text (multiple subword tokens), while
real `<|...|>` special tokens are atomic (1 token each).

**This means:** The byte `│` (U+2502, 3 bytes: `E2 94 82`) will appear in the
training data. The model will learn it as a regular character. It has no special
meaning to the tokenizer.

---

## 15. Evaluation Results — Tokenizer Quality

All checks passed:

| Check | Result | Details |
|---|---|---|
| Roundtrip | PASS | 1,094/1,094 records byte-identical |
| Special token atomicity | PASS | All 49 special tokens → exactly 1 ID |
| Escaped token non-atomicity | PASS | `<│endoftext│>` → 3 IDs (correct) |
| Balanced pairs | PASS | All 6 pair types perfectly balanced |
| Fertility | 1.47 tok/word | Prose: 1.42, Code: 1.47 |
| Compression | 4.77 bytes/tok | |
| Vocab coverage | 97.2% | 896 dead tokens (2.8%) |

### Token frequency — top 10

| ID | Token | Frequency |
|---|---|---|
| 312 | ` the` | 360,577 |
| 44 | `,` | 332,987 |
| 46 | `.` | 250,528 |
| 32 | ` ` (space) | 198,135 |
| 337 | ` to` | 155,822 |
| 341 | ` of` | 151,357 |
| 342 | ` and` | 148,570 |
| 307 | ` a` | 132,241 |
| 333 | ` in` | 94,843 |
| 364 | ` is` | 91,853 |

The frequency distribution follows Zipf's law (verified — plot saved in
`output/plots/zipf_distribution.png`).

---

## 16. Design Decisions and Their Rationale

### Vocab size: 32,305

- Corpus is ~8.8M tokens — relatively small
- 32K BPE merges is sufficient for English-only technical content
- GPT-2 original used 50,257 for a much larger corpus
- Final: 256 bytes + 49 special + 32,000 merges = 32,305

### Pre-tokenization: cl100k_base (not GPT-2 pattern)

We chose the GPT-4 / cl100k_base pattern over GPT-2's because:
- Case-insensitive contractions (`DON'T` → `DON` + `'T`)
- 3-digit max numbers (forces digit composition)
- Dedicated newline handling (important for code blocks)
- Possessive quantifiers (faster matching)

### Byte-level BPE

The tokenizer operates at the byte level (not character level). This means:
- Every possible input can be encoded (no unknown tokens)
- UTF-8 multi-byte characters are handled naturally
- The base vocabulary is exactly 256 entries

### Special tokens get IDs 256–304

Placed right after bytes, before BPE merges. This means they always exist in
the vocabulary regardless of how many merges are trained. They are never
created or destroyed by the merge process.

### No `<|startoftext|>` in training data

Reserved for inference as a BOS token. During training, `<|source:book|>` serves
as the document start marker. This mirrors GPT-2's design.

### Shared end tokens

`<|code_end|>` is shared across all code languages. `<|note_end|>` is shared
across all note types. This keeps the vocabulary small — the language/type
information is on the start token.

---

## 17. How to Tokenize the Corpus for Training

```python
import json
from bpe.tokenizer import Tokenizer

# Load tokenizer
tok = Tokenizer.from_files("tokenizer_output/")

# Load and encode corpus
encoded_records = []
with open("corpus/books.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        ids = tok.encode(record["text"])
        encoded_records.append(ids)

# Total tokens
total_tokens = sum(len(r) for r in encoded_records)
print(f"Total tokens: {total_tokens:,}")  # 9,973,151

# For pretraining, you'll typically:
# 1. Concatenate all records with <|endoftext|> already at boundaries
# 2. Split into fixed-length chunks (e.g., 1024 tokens)
# 3. Create input/target pairs (shifted by 1)

# Example: concatenate and chunk
import itertools
all_ids = list(itertools.chain.from_iterable(encoded_records))
context_length = 1024
chunks = [all_ids[i:i+context_length] for i in range(0, len(all_ids) - context_length, context_length)]
print(f"Training chunks: {len(chunks):,}")  # ~9,700 chunks of 1024
```

---

## 18. Key Considerations for Pretraining

### Embedding layer size
```python
vocab_size = 32305  # From tokenizer_config.json
nn.Embedding(vocab_size, d_model)
```

### Special token IDs you'll reference in code

```python
EOS_TOKEN_ID = 278   # <|endoftext|> — used as document separator, generation stop
BOS_TOKEN_ID = 304   # <|startoftext|> — prepend at inference time
PAD_TOKEN_ID = 278   # Use EOS as pad (mask in loss), or define a new one
```

### Context window sizing

- Mean record length: 9,116 tokens
- P95: 19,819 tokens
- Recommended context: 1024 or 2048 (standard GPT-2 sizes)
- Records longer than context will be chunked automatically when you
  concatenate and split into fixed-length sequences

### Loss masking

You may want to:
- Mask padding tokens in the loss
- Optionally mask special tokens in the loss (debatable — the model should
  learn to produce them, so including them in the loss is typically correct)

### The `meta` field

Not used for tokenization, but useful for pretraining:
- `meta.word_count` — for weighted sampling (upweight longer chapters)
- `meta.content_types` — for curriculum learning (e.g., prose first, then code)
- `meta.title` — for debugging and analysis

---

## 19. Tokenizer Source Code Architecture

```
bpe/
  __init__.py          # Empty
  pretokenizer.py      # PreTokenizer class
    - PRETOK_PATTERN: compiled cl100k_base regex
    - _build_special_token_pattern(): builds regex for special tokens
    - PreTokenizer.__init__(special_tokens): stores set + builds pattern
    - PreTokenizer.pre_tokenize(text) → list[str]: the main split function
    - PreTokenizer._regex_split(text) → list[str]: applies PRETOK_PATTERN

  tokenizer.py         # Tokenizer class
    - Tokenizer.__init__(vocab, merges, special_tokens)
    - Tokenizer.from_files(output_dir) → Tokenizer: loads all 3 artifacts
    - Tokenizer.encode(text) → list[int]
    - Tokenizer.decode(ids) → str
    - Tokenizer.encode_batch(texts) → list[list[int]]
    - Tokenizer._encode_chunk(chunk) → list[int]: BPE merge application

  utils.py             # Shared helpers
    - load_corpus(jsonl_path) → list[str]
    - load_special_tokens(json_path) → list[str]
    - load_merges(path) → list[tuple[bytes, bytes]]
    - save_json(obj, path)
    - save_merges(merges, path)
    - Timer: context manager for timing
```

### Dependencies required by the `bpe` package

```
regex      # For cl100k_base pre-tokenization (possessive quantifiers)
```

That's the only runtime dependency. `numpy`, `matplotlib`, `rich`, `flask` are
only needed for evaluation and visualization scripts — not for encoding/decoding.

---

## 20. Dependencies

### Required for tokenizer encode/decode (minimal)

```
regex
```

### Required for evaluation and visualization (optional)

```
regex
numpy
matplotlib
rich
flask
```

### Python version

Python 3.12 (tested on 3.12.12 via conda-forge on macOS arm64).

---

## Appendix: Book Titles in the Corpus (62 titles)

- 15 Math Concepts Every Data Scientist Should Know
- AI Agents and Applications
- AI Agents in Practice
- AI Engineering
- AI and ML for Coders in PyTorch
- AI and Machine Learning for Coders
- AWS Certified Machine Learning - Specialty (MLS-C01) Certification Guide
- AWS Certified Solutions Architect - Associate (SAA-C03) Exam Guide
- All of Statistics
- Artificial Intelligence for Cybersecurity
- Bayesian Analysis with Python - Third Edition
- Build a Large Language Model (From Scratch)
- Building Agentic AI Systems
- Building LLM Powered Applications
- Causal Inference and Discovery in Python
- Causal Inference in R
- Computational Intelligence and Data Analytics
- Data Analysis with Python and PySpark
- Data Engineering on Azure
- Data Engineering with Databricks Cookbook
- Data Science from Scratch
- Databricks Certified Associate Developer for Apache Spark Using Python
- Databricks ML in Action
- Deep Learning with PyTorch
- Deep Learning with PyTorch, Second Edition
- Deep Reinforcement Learning Hands-On
- Designing Machine Learning Systems
- Essential GraphRAG
- Financial Data Engineering
- From Unimodal to Multimodal Machine Learning
- Generative AI with Amazon Bedrock
- Grokking Algorithms, Second Edition
- Hands-On Genetic Algorithms with Python, Second Edition
- Hands-On Large Language Models
- Hands-On Web Scraping with Python
- Knowledge Graphs and LLMs in Action
- LLM Engineer's Handbook
- Machine Learning Algorithms in Depth
- Machine Learning Engineering with Python
- Machine Learning with PyTorch and Scikit-Learn
- Machine Learning with R, Fourth Edition
- Mastering NLP from Foundations to LLMs
- Mastering PyTorch, Second Edition
- Mastering Transformers
- Mathematics of Machine Learning
- Modern Computer Vision with PyTorch, Second Edition
- Modern Time Series Forecasting with Python, Second Edition
- Multimodal and Tensor Data Analytics
- Outlier Detection in Python
- Practical Statistics for Data Scientists, 2nd Edition
- Pretrain Vision and Large Language Models in Python
- Python Feature Engineering Cookbook
- Python Machine Learning By Example, Fourth Edition
- Python for Algorithmic Trading Cookbook
- RAG-Driven Generative AI
- Retrieval Augmented Generation Using LangChain
- SQL for Data Analytics
- The Machine Learning Solutions Architect Handbook
- TinyML Cookbook
- Transformers for Natural Language Processing and Computer Vision, Third Edition
- XGBoost for Regression Predictive Modeling and Time Series Analysis

---

*End of handoff document. The next session should have everything it needs to
pretrain a GPT-2 model using this tokenizer and corpus.*
