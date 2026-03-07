# GPT-2 Pretraining — Version Plan

> This document defines the versioned approach to building our GPT-2 model.
> Each version isolates specific variables so we can measure what helps and what doesn't.

---

## v1 — Baseline GPT-2 (clean, proven architecture)

**Goal:** Establish a solid baseline with well-understood components. No experiments.
Every choice is battle-tested.

### Architecture

| Component | Choice | Rationale |
|---|---|---|
| d_model | 512 | Conservative, proven. 38% embedding fraction — healthy for 10M tokens. |
| n_layers | 8 | Enough depth without overfitting risk. |
| n_heads | 8 | 64 dims/head, standard. |
| d_ff | 2048 | 4x expansion, standard GELU FFN. |
| context_len | 2048 | Retains document structure. |
| activation | GELU | GPT-2 standard, well-understood. |
| normalization | LayerNorm (pre-norm) | GPT-2 standard, proven stable. |
| positions | Learned absolute | Simplest, proven. |
| weight init | N(0, 0.02) + residual scaling 1/sqrt(2*n_layers) | GPT-2 convention. |
| weight tying | Yes | Shares embedding params with output head. |
| bias | False | Fewer params, cleaner. |
| dropout | 0.1 | Standard regularization. |
| attention masking | Standard causal | Simple. No document boundary masking yet. |
| **Total params** | **~43M** | |

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| max_epochs | **100** | Early stopping decides when to stop, not a fixed count. 100 is the ceiling. |
| batch_size | 8 | Fits in ~21GB on M4 Pro. |
| grad_accum | 4 | Effective batch = 32. |
| learning_rate | 6e-4 | Peak LR, cosine decay to 6e-5. Chinchilla 70M used 1e-3; GPT-3 125M used 6e-4. Our 43M model is well within range. |
| warmup | 5% of total steps | ~300 steps warmup at 100 epochs. |
| weight_decay | 0.01 | BookGPT showed 0.1 was too aggressive. |
| grad_clip | 1.0 | Standard. |
| AdamW betas | (0.9, 0.95) | Standard for LM pretraining. |
| patience | 10 | Stop if val loss doesn't improve for 10 epochs. |
| min_delta | 0.001 | Minimum improvement to count as progress. |
| val_split | 10% | ~487 chunks held out. |

### Training Scale

```
Chunks per epoch:        4,869
Steps per epoch:           608  (batch=8)
Weight updates per epoch:  152  (accum=4)
Tokens per weight update:  65,536

At 100 epochs (ceiling):
  Total steps:           60,800
  Total weight updates:  15,200
  Total tokens seen:     ~997M (each of the 10M tokens seen ~100 times)
  Effective tok/param:   23.2 (matches Chinchilla ratio)
  Estimated time:        ~3.4 hours on M4 Pro

Early stopping with patience=10 will likely stop between 30-70 epochs.
We let the validation loss decide, not an arbitrary epoch count.
```

### Deliverables

- Model architecture + training code
- Data pipeline (tokenize, concatenate, chunk, train/val split)
- Pretraining with next-token prediction
- Diagnostic plots (loss curves, attention entropy, embedding utilization, token prediction accuracy)
- Text generation (chat with pretrained model)
- Checkpoint preservation (pretrained copy saved before any fine-tuning)
- Fine-tuning (if applicable)
- Comparison: pretrained vs fine-tuned side by side
- v1 report: what worked, what didn't, what to change

---

## v2 — Modern Architecture Improvements

**Goal:** Test whether modern architectural upgrades improve quality.
Same data and training setup as v1. Isolates the effect of architecture changes.

### Changes from v1

| Component | v1 | v2 | Why test this |
|---|---|---|---|
| d_model | 512 | **768** | More capacity for 32K vocab + diverse content. |
| n_layers | 8 | **10** | More depth for head specialization. |
| n_heads | 8 | **12** | 64 dims/head maintained at 768 width. |
| activation | GELU | **SwiGLU** | Gated FFN, consistently better in post-2020 research. |
| normalization | LayerNorm | **RMSNorm** | Simpler, no mean subtraction, fewer params. |
| **Total params** | ~43M | **~96M** | |

### Deliverables

- Same diagnostic plots as v1
- Direct comparison: v1 vs v2 (val loss, perplexity, generation quality, attention patterns, embedding utilization)
- v2 report: did scaling + modern FFN/norm help? By how much?

---

## v3 — Training Signal Quality

**Goal:** Keep best architecture from v2, improve how we train.
Same model size, same data. Only the training signal quality changes.

### Changes from v2

| Component | v2 | v3 | Why test this |
|---|---|---|---|
| attention masking | Standard causal | **Document boundary masking** | Prevents cross-document contamination. LLaMA 3 does this. |
| position reset | No | **Yes — reset at doc boundaries** | Positions restart at each `<\|endoftext\|>`. |
| QK-Norm | No | **Yes** | Cheap stability insurance, prevents attention logit drift. |

### Deliverables

- Same diagnostics as v1/v2
- Comparison: v2 vs v3 — does document masking improve coherence at chapter boundaries?
- Attention pattern analysis near `<|endoftext|>` tokens
- v3 report

---

## v4 — Positional Encoding Experiment

**Goal:** Definitive test of RoPE vs learned absolute positions at our scale.

### Changes from v3

| Component | v3 | v4 | Why test this |
|---|---|---|---|
| positions | Learned absolute | **RoPE** | BookGPT said it didn't help — but that was per-book with 170K tokens. At 10M tokens with document masking, the result may differ. |

### Deliverables

- Same diagnostics
- Definitive comparison of position encoding strategies
- v4 report

---

## v5+ — Based on Findings

Depends on what v1-v4 reveal. Possible directions:

- Curriculum learning (prose first, then code-heavy chapters)
- Attention diversity loss (penalize head similarity)
- Different context lengths (1024 vs 2048 vs 4096)
- Fine-tuning strategies (instruction tuning, DPO)
- Data augmentation or expansion

---

## Summary

| Version | Focus | Key Changes | Params | Goal |
|---|---|---|---|---|
| **v1** | Baseline | Standard GPT-2 | ~43M | Prove pipeline works. Baseline metrics. |
| **v2** | Scale + modern arch | d=768, 10L, SwiGLU, RMSNorm | ~96M | Does more capacity + modern components help? |
| **v3** | Training quality | Doc boundary masking, QK-Norm | ~96M | Does cleaner training signal improve coherence? |
| **v4** | Position encoding | RoPE | ~95M | Learned vs rotary — definitive answer. |
| **v5+** | TBD | Based on findings | TBD | Curriculum, fine-tuning, alignment |

Each version produces: diagnostic plots, generation samples, a comparison report, and
a preserved checkpoint for side-by-side chat. Each version's code lives in its own
directory. No version destroys another.

---

## Rules (non-negotiable)

1. No decisions without explicit user approval ("Yes, I agree").
2. Free to write code, not to change data/approach without approval.
3. No data quality claims without proofs.
4. Versioned directories from the start.
5. Test cases for critical components.
6. Meaningful visualizations mandatory.
7. Logs to files, per version.
8. Python venv for this project.
9. v2 starts as a copy of v1 code, then modified.
10. Each version is complete only after pretrain + finetune + diagnostics + plots.
11. Post-version document: what worked, what didn't, what next.
12. Preserve pretrained checkpoint before finetuning — both available for side-by-side chat.
13. Diagnostic plots for every stage (pretrain, finetune, etc.).
