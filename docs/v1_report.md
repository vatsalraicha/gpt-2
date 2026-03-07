# v1 Report: What We Built, What We Learned, What's Next

> Written: March 7, 2026

---

## 1. What We Set Out To Do

Train a GPT-2-style language model from scratch on a curated corpus of 62 technical books (ML/AI/data science), using a custom BPE tokenizer. The goal: establish a solid baseline with a well-understood architecture, then iterate with modern improvements in subsequent versions.

This project exists because of a previous attempt ("BookGPT," a 7M parameter model) that failed in instructive ways. Every design decision in v1 was informed by those failures.

---

## 2. What We Built

### Corpus & Tokenizer
- **62 books** (72 EPUBs curated, 5 removed as out-of-domain, 5 duplicates)
- **1,094 chapter records**, each wrapped with `<|source:book|>` ... `<|endoftext|>` markers
- **Custom BPE tokenizer**: 32,305 vocab (256 byte tokens + 49 special tokens + 32,000 learned merges)
- **9.97M tokens**, 6.8M words, tokenizer fertility 1.47 tokens/word
- Median record: 8,335 tokens, mean: 9,116, P95: 19,819

### Model Architecture (43M parameters)

| Component | Choice |
|---|---|
| d_model | 512 |
| n_layers | 8 |
| n_heads | 8 (64 dims/head) |
| d_ff | 2,048 (4x expansion) |
| context_length | 2,048 |
| activation | GELU |
| normalization | Pre-LayerNorm |
| positions | Learned absolute |
| weight tying | Yes (embedding ↔ output head) |
| bias | None |
| dropout | 0.1 |
| Total params | ~42.8M |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Peak LR | 6e-4, cosine decay to 6e-5 |
| Warmup | 5% of total steps |
| Weight decay | 0.01 |
| Batch size | 8 (gradient accumulation 4 → effective 32) |
| Grad clip | 1.0 |
| Early stopping | Patience 10, min_delta 0.001 |
| Val split | 10% |

### Infrastructure
- **Three backends implemented**: MPS (Apple M4 Pro), NVIDIA (Colab A100), MLX (Apple Silicon — code ready, not yet trained)
- **Unified dashboard**: Root-level Flask app with version/backend dropdowns, auto-discovers all `v{N}` and `v{N}_{backend}` directories
- **Diagnostic framework**: 4-dimensional analysis (attention entropy, attention locality, embedding utilization, token prediction)
- **Runtime controls**: Graceful stop (`STOP` file), LR override (`LR` file), checkpoint resume

---

## 3. Training Results

### v1 MPS (Apple M4 Pro 48GB)

| | Value |
|---|---|
| Epochs trained | 17 (early stopped; patience was not yet 10 in initial run) |
| Best epoch | 17 |
| Best val loss | **3.1667** |
| Best val PPL | **23.73** |
| Total time | **35.5 hours** |
| Avg epoch time | ~35-60 min (varied due to thermal throttling and background load) |

The MPS training was our first run. It revealed the M4 Pro's thermal limitations — epoch times varied wildly from 48 min to 11 hours (some epochs ran overnight with system sleep interruptions). But it converged to a reasonable baseline.

### v1 NVIDIA (Colab Pro A100 80GB)

| Epoch | Train Loss | Val Loss | Val PPL |
|---|---|---|---|
| 1 | 8.3295 | 6.5584 | 705.16 |
| 5 | 4.4166 | 4.3743 | 79.38 |
| 10 | 3.2139 | 3.4523 | 31.57 |
| 15 | 2.6713 | 3.2050 | 24.66 |
| **19** | **2.4008** | **3.1756** | **23.94** |
| 25 | 2.1079 | 3.2320 | 25.33 |
| 29 | 1.9589 | 3.2989 | 27.08 |

| | Value |
|---|---|
| Epochs trained | 29 (early stopped at patience 10 after best epoch 19) |
| Best epoch | 19 |
| Best val loss | **3.1756** |
| Best val PPL | **23.94** |
| Total time | **~82 minutes** |
| Avg epoch time | ~170 seconds (~2.8 min) |
| Throughput | ~52,000 tok/sec |

**Key observation**: MPS and NVIDIA converged to nearly identical val loss (~3.17) and PPL (~23.8). This confirms the model architecture and hyperparameters are the bottleneck, not the hardware. The A100 just got there **26x faster** (82 min vs 35.5 hours).

**Overfitting became clear from epoch 20 onward**: train loss kept falling (2.40 → 1.96) while val loss rose (3.18 → 3.30). The model memorized training data past the point of generalization.

### Finetuning (250 Q&A pairs)

| Backend | Best Epoch | Best Val Loss | Best Val PPL | Epochs Trained |
|---|---|---|---|---|
| MPS | 2 | 2.8521 | 17.32 | 10 (early stopped) |
| NVIDIA | 2 | 2.9047 | 18.26 | 10 (early stopped) |

Finetuning showed rapid learning (best at epoch 2) followed by severe overfitting — by epoch 10, train PPL dropped to ~1.2 while val PPL exploded to ~55. With only 250 Q&A pairs, this is expected. The finetuned model was qualitatively better at Q&A format (coherent answers to "What is gradient descent?") while the pretrained model degenerated into loops.

---

## 4. What Worked

1. **Combined corpus training** — All 62 books in one dataset, with document boundary markers. Unlike BookGPT's per-book training, the model saw diverse vocabulary and writing styles every epoch.

2. **Weight decay 0.01** — BookGPT used 0.1, which was too aggressive. The lower value allowed better convergence.

3. **Cosine LR schedule with warmup** — Smooth training dynamics, no instability.

4. **Early stopping with patience 10** — Let the model find its natural convergence point rather than arbitrary epoch limits.

5. **Gradient accumulation** — Effective batch size 32 from physical batch 8. Good balance of memory efficiency and gradient quality.

6. **Diagnostic framework** — Attention entropy/locality heatmaps and embedding analysis gave concrete evidence of model behavior, not just loss curves.

7. **Infrastructure** — Graceful stop, LR override, checkpoint resume, unified dashboard with version/backend switching. Made long training runs manageable.

---

## 5. What Didn't Work (or Needs Improvement)

1. **Overfitting after epoch ~19** — The gap between train loss (2.40) and val loss (3.18) at the best epoch already signaled early overfitting. With only 10M tokens and 43M parameters, we're in a data-limited regime. The Chinchilla-optimal token count for 43M params would be ~860M tokens (20x what we have).

2. **Finetuning with 250 Q&A pairs** — Far too few. The model memorized them by epoch 3. We need 2,000-5,000 Q&A pairs minimum, or use data augmentation techniques.

3. **Attention entropy still high** — Mean entropy ~0.77 across all heads (BookGPT was 0.81). We wanted <0.5, indicating sharp, specialized attention patterns. The heads are still somewhat diffuse, though better than BookGPT's uniformly high entropy.

4. **No document boundary masking** — The model can attend across document boundaries within a context window. This means it could learn spurious patterns between the end of one chapter and the start of another.

5. **Generation quality** — The pretrained model produces topically relevant text but degenerates into repetition loops after ~50 tokens. This is a known issue at this scale without nucleus sampling or repetition penalties.

6. **MPS training speed** — 35.5 hours for 17 epochs is painful for iteration. The A100 proved that identical results can be achieved 26x faster.

---

## 6. Diagnostics Summary

The diagnostic framework measures four dimensions across all 64 attention heads (8 layers x 8 heads):

- **Attention Entropy**: How diffuse vs. focused each head's attention pattern is (0 = perfectly focused, 1 = uniform). Our heads averaged ~0.77, indicating moderate but not sharp specialization.
- **Attention Locality**: How much each head focuses on nearby tokens vs. distant ones. Some heads showed strong local patterns (likely syntax/grammar), others were more global (likely semantic).
- **Embedding Utilization**: How much of the 512-dimensional embedding space the model actually uses. Better than BookGPT's 10% utilization, but room for improvement.
- **Token Prediction Accuracy**: Where the model succeeds/fails at next-token prediction across content types.

---

## 7. What's Next: The Roadmap

### Immediate: v1 MLX (Apple Silicon)

**Why**: Same v1 architecture, same data, same hyperparameters — but running natively on Apple Silicon via MLX instead of PyTorch+MPS. We expect **identical results** (same model, same data), but need to benchmark **MLX training speed** against MPS (PyTorch) on the same hardware. This tells us which framework to use for local development going forward.

**When**: After v2_nvidia completes on Colab.

### v2: Modern Architecture Improvements

**Goal**: Test whether modern architectural upgrades improve quality. Same data, same training setup. Only the architecture changes, so we isolate the effect.

| Component | v1 | v2 | Why |
|---|---|---|---|
| d_model | 512 | **768** | More capacity for 32K vocab + diverse content |
| n_layers | 8 | **10** | More depth for head specialization |
| n_heads | 8 | **12** | 64 dims/head maintained at 768 width |
| activation | GELU | **SwiGLU** | Gated FFN, consistently outperforms GELU in post-2020 research |
| normalization | LayerNorm | **RMSNorm** | Simpler (no mean subtraction), fewer params, used by LLaMA |
| **Total params** | **~43M** | **~96M** | 2.2x increase |

**Training plan**: Run v2 on **Colab A100** first (v2_nvidia). Based on v1's experience, a 96M model with 10M tokens will overfit even sooner — we may need to increase dropout or reduce epochs. But the diagnostic comparison (v1 vs v2 attention patterns, embedding utilization, generation quality) will tell us if the extra capacity helps.

### v3: Training Signal Quality

**Goal**: Keep the best architecture from v2, improve how the training signal reaches the model.

| Change | What it does |
|---|---|
| **Document boundary masking** | Prevent attention from crossing `<\|endoftext\|>` boundaries within a context window. LLaMA 3 does this. |
| **Position reset at boundaries** | Restart position embeddings at each document start, so the model doesn't learn spurious positional patterns across documents |
| **QK-Norm** | Normalize query and key vectors before attention dot product. Prevents attention logit drift at scale. Cheap stability insurance. |

### v4: Positional Encoding Experiment

| Change | What it does |
|---|---|
| **RoPE** (Rotary Position Embeddings) | Replace learned absolute positions. BookGPT said RoPE didn't help, but that was per-book with 170K tokens. At 10M tokens with document masking from v3, the result may differ. This is the definitive test. |

### v5+: Based on Findings

Possible directions depending on what v1-v4 reveal:
- Curriculum learning (prose chapters first, code-heavy chapters later)
- Attention diversity loss (explicitly penalize heads for being too similar)
- Context length experiments (1024 vs 2048 vs 4096)
- More finetuning data (2,000-5,000 Q&A pairs)
- DPO/RLHF alignment (currently incomplete for v1)

---

## 8. Execution Order

| Step | What | Where | Estimated Time |
|---|---|---|---|
| 1 | Create v2 code (copy v1, apply arch changes) | Local | 1-2 hours |
| 2 | v2_nvidia pretrain | Colab A100 | ~3-4 hours (2.2x params, similar epoch count) |
| 3 | v2_nvidia finetune | Colab A100 | Minutes |
| 4 | v2 diagnostics + comparison report (v1 vs v2) | Local | 1 hour |
| 5 | Decide: does v2 architecture help enough to keep? | Decision point | — |
| 6 | v1_mlx training (benchmark MLX speed) | Local M4 Pro | ~35 hours (expect similar to MPS) |
| 7 | v3 onwards based on v2 findings | TBD | TBD |

**Note**: v1_mlx runs locally and can happen in parallel with Colab training. It's a speed benchmark, not a new experiment — we expect identical convergence to v1 MPS and v1 NVIDIA.

---

## 9. Hardware & Cost Summary

| Backend | Hardware | Training Time | Cost |
|---|---|---|---|
| v1 MPS | Apple M4 Pro 48GB | 35.5 hours (17 epochs) | $0 (local) |
| v1 NVIDIA | Colab Pro A100 80GB | 82 min (29 epochs) | ~$12/month subscription + ~10 compute units |
| v1 MLX | Apple M4 Pro 48GB (MLX) | Not yet run | $0 (local) |

---

## 10. Key Numbers to Beat in v2

| Metric | v1 Best | Target for v2 |
|---|---|---|
| Val Loss | 3.17 | < 3.0 |
| Val PPL | 23.9 | < 20 |
| Attention Entropy (mean) | 0.77 | < 0.6 |
| Generation coherence | Loops after ~50 tokens | Coherent for 100+ tokens |
| Finetuned val PPL | 17.3 | < 15 (with more Q&A data) |

---

## 11. Lessons for the Paper

If we pursue a publication, the narrative is:

1. **Systematic ablation study** of modern LLM components at small scale (43M-96M params)
2. **BookGPT → GPT-2 progression**: how fixing per-book training, embedding utilization, weight decay, and data diversity transforms results
3. **Diagnostic framework** that goes beyond loss curves: attention entropy, locality, embedding rank, token prediction analysis
4. **Cross-backend reproducibility**: identical convergence on MPS, CUDA, and (soon) MLX
5. **Version-isolated experiments**: each version changes exactly one category of variables

The paper would need v2 and v3 results to be complete. v4 (RoPE) adds a definitional answer to a debated question at small scale.

---

*This document will be updated after each version completes.*
