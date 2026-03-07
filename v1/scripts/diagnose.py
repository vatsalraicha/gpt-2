"""Model diagnostics: Analyze what a trained GPT-2 model learned.

Investigates four dimensions:
1. Attention Pattern Analysis — Are heads learning meaningful relationships?
2. Embedding Space Analysis — Are semantically similar tokens clustered?
3. Token Prediction Analysis — Where exactly does next-token prediction break down?
4. Capacity Saturation — Are the weights "full" or is there unused capacity?

Usage:
    python -m v1.scripts.diagnose                          # pretrain (default)
    python -m v1.scripts.diagnose --stage finetune         # finetune checkpoint
    python -m v1.scripts.diagnose --checkpoint path/to/ckpt
    python -m v1.scripts.diagnose --no-plots
    python -m v1.scripts.diagnose --n-samples 20 --passage-length 1024
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1.model.gpt2 import GPT2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────

def load_sample_passages(
    cache_path: Path,
    n_passages: int = 15,
    passage_length: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Load random chunks from cached tokens for analysis.

    Returns:
        (n_passages, passage_length) int32 array of token IDs
    """
    all_tokens = np.load(cache_path)
    rng = np.random.RandomState(seed)
    max_start = len(all_tokens) - passage_length
    starts = rng.randint(0, max_start, size=n_passages)
    passages = np.stack([all_tokens[s:s + passage_length] for s in starts])
    return passages


# ─────────────────────────────────────────────────────
# 1. ATTENTION PATTERN ANALYSIS
# ─────────────────────────────────────────────────────

def extract_attention_weights(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Run forward pass with hooks to capture attention weights from all layers.

    Since the model uses F.scaled_dot_product_attention (no weights exposed),
    we recompute Q@K^T / sqrt(d_k) inside the hook, apply causal mask, softmax.

    Args:
        model: GPT2 model in eval mode
        input_ids: (B, T) tensor

    Returns:
        List of (B, n_heads, T, T) attention weight tensors, one per layer.
        All tensors on CPU float32.
    """
    attention_maps = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0].detach().cpu().float()  # (B, T, C)
            B, T, C = x.shape

            # Recompute QKV on CPU
            c_attn_w = module.c_attn.weight.detach().cpu().float()
            qkv_out = F.linear(x, c_attn_w)  # (B, T, 3*d_model)

            d_model = module.d_model
            n_heads = module.n_heads
            head_dim = module.head_dim

            q, k, _v = qkv_out.split(d_model, dim=2)
            q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, n_heads, head_dim).transpose(1, 2)

            # Scaled dot-product
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

            # Causal mask: can't attend to future
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            att = F.softmax(att, dim=-1)
            attention_maps.append(att)  # already on CPU

        return hook_fn

    model.eval()
    for i, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return attention_maps


def analyze_attention(
    model,
    passages: np.ndarray,
    device: torch.device,
) -> dict:
    """Analyze attention patterns across sample passages.

    Returns dict with entropy and locality matrices (n_layers x n_heads),
    averaged across all sample passages.
    """
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads

    # Accumulate across passages
    entropy_sum = np.zeros((n_layers, n_heads))
    locality_sum = np.zeros((n_layers, n_heads))
    n_passages = len(passages)

    for i, passage in enumerate(passages):
        input_ids = torch.from_numpy(np.array([passage])).long().to(device)
        attn_maps = extract_attention_weights(model, input_ids)

        T = len(passage)
        locality_window = 10

        # Precompute locality mask once per passage
        locality_mask = torch.zeros(T, T)
        for pos in range(T):
            start = max(0, pos - locality_window)
            end = min(T, pos + 1)  # causal: only past + self
            locality_mask[pos, start:end] = 1.0

        for layer_idx, attn in enumerate(attn_maps):
            attn = attn[0]  # remove batch dim: (n_heads, T, T)

            for head_idx in range(n_heads):
                head_attn = attn[head_idx]  # (T, T)

                # Entropy: how spread out is attention?
                eps = 1e-10
                entropy = -(head_attn * (head_attn + eps).log()).sum(dim=-1)  # (T,)
                avg_entropy = entropy.mean().item()
                max_entropy = math.log(T)
                normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0

                # Locality: fraction of attention on nearby tokens
                local_attn = (head_attn * locality_mask).sum(dim=-1).mean().item()

                entropy_sum[layer_idx, head_idx] += normalized_entropy
                locality_sum[layer_idx, head_idx] += local_attn

        # Free attention maps
        del attn_maps

    # Average across passages
    entropy_matrix = entropy_sum / n_passages
    locality_matrix = locality_sum / n_passages

    # Classify head types
    head_types = []
    for l in range(n_layers):
        row = []
        for h in range(n_heads):
            loc = locality_matrix[l, h]
            if loc > 0.7:
                row.append("local")
            elif loc < 0.3:
                row.append("long_range")
            else:
                row.append("mixed")
        head_types.append(row)

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_samples": n_passages,
        "entropy_matrix": entropy_matrix.tolist(),
        "locality_matrix": locality_matrix.tolist(),
        "head_types": head_types,
        "avg_entropy": float(entropy_matrix.mean()),
        "avg_locality": float(locality_matrix.mean()),
    }


# ─────────────────────────────────────────────────────
# 2. EMBEDDING SPACE ANALYSIS
# ─────────────────────────────────────────────────────

# ML/AI concept groups for our technical book corpus
CONCEPT_GROUPS = {
    "neural_network": ["neural", "network", "layer", "neuron", "activation"],
    "optimization": ["gradient", "loss", "learning", "optimizer", "convergence"],
    "statistics": ["probability", "distribution", "variance", "mean", "sample"],
    "linear_algebra": ["matrix", "vector", "dimension", "eigenvalue", "transpose"],
    "programming": ["function", "algorithm", "variable", "class", "method"],
}


def analyze_embeddings(model, tokenizer) -> dict:
    """Analyze the token embedding space.

    Checks:
    - Effective dimensionality via SVD
    - Semantic clustering of ML/AI concept groups
    - Cross-group vs within-group similarity
    """
    embeddings = model.wte.weight.detach().cpu().numpy()  # (vocab_size, d_model)
    vocab_size, n_embd = embeddings.shape

    # 1. SVD: effective dimensionality
    _U, S, _Vt = np.linalg.svd(embeddings, full_matrices=False)
    total_variance = (S ** 2).sum()
    cumulative_variance = np.cumsum(S ** 2) / total_variance

    dims_90 = int(np.searchsorted(cumulative_variance, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative_variance, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumulative_variance, 0.99)) + 1

    # 2. Embedding norms
    norms = np.linalg.norm(embeddings, axis=1)

    # 3. Semantic group clustering
    group_similarities = {}
    for group_name, words in CONCEPT_GROUPS.items():
        token_ids = []
        found_words = []
        for word in words:
            ids = tokenizer.encode(word)
            if len(ids) == 1:
                token_ids.append(ids[0])
                found_words.append(word)
            elif len(ids) > 0:
                # Multi-token: use first token as approximation
                token_ids.append(ids[0])
                found_words.append(f"{word}[0]")

        if len(token_ids) >= 2:
            group_embs = embeddings[token_ids]
            norms_g = np.linalg.norm(group_embs, axis=1, keepdims=True)
            normalized = group_embs / (norms_g + 1e-8)
            sim_matrix = normalized @ normalized.T
            n = len(token_ids)
            mask = ~np.eye(n, dtype=bool)
            avg_sim = float(sim_matrix[mask].mean())
            group_similarities[group_name] = {
                "words": found_words,
                "avg_cosine_similarity": avg_sim,
                "token_ids": token_ids,
            }

    # 4. Cross-group similarity
    all_group_embs = {}
    for group_name, info in group_similarities.items():
        all_group_embs[group_name] = embeddings[info["token_ids"]].mean(axis=0)

    cross_sims = {}
    group_names = list(all_group_embs.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1:]:
            e1 = all_group_embs[g1]
            e2 = all_group_embs[g2]
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
            cross_sims[f"{g1}-{g2}"] = sim

    # 5. Random baseline
    rng = np.random.RandomState(42)
    random_pairs = rng.choice(vocab_size, size=(200, 2), replace=True)
    random_sims = []
    for i, j in random_pairs:
        e1, e2 = embeddings[i], embeddings[j]
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        random_sims.append(float(sim))

    return {
        "vocab_size": vocab_size,
        "d_model": n_embd,
        "singular_values_top50": S[:50].tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "dims_for_90pct": dims_90,
        "dims_for_95pct": dims_95,
        "dims_for_99pct": dims_99,
        "embedding_norms": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "group_similarities": group_similarities,
        "cross_group_similarities": cross_sims,
        "random_baseline_similarity": float(np.mean(random_sims)),
    }


# ─────────────────────────────────────────────────────
# 3. TOKEN PREDICTION ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_token_predictions(
    model,
    passages: np.ndarray,
    device: torch.device,
) -> dict:
    """Teacher-forced next-token prediction analysis.

    For each passage, compute:
    - Per-position accuracy and confidence
    - First-half vs second-half accuracy (error compounding)
    """
    model.eval()
    results = []
    all_position_correct = []
    all_position_confidence = []

    for i in range(len(passages)):
        passage = passages[i]
        # Input: all tokens except last. Target: all tokens except first.
        input_ids = torch.from_numpy(np.array([passage[:-1]])).long().to(device)
        targets = passage[1:]

        with torch.no_grad():
            logits, _ = model(input_ids)

        logits_cpu = logits[0].cpu().float()  # (T-1, vocab_size)
        probs = F.softmax(logits_cpu, dim=-1)

        T = len(targets)
        half = T // 2

        # Per-position accuracy and confidence
        predicted = logits_cpu.argmax(dim=-1).numpy()
        correct = (predicted == targets).astype(float)
        confidences = probs[range(T), targets].numpy()

        first_half_acc = float(correct[:half].mean()) if half > 0 else 0
        second_half_acc = float(correct[half:].mean()) if (T - half) > 0 else 0

        results.append({
            "sample_idx": i,
            "total_tokens": T,
            "accuracy": float(correct.mean()),
            "first_half_accuracy": first_half_acc,
            "second_half_accuracy": second_half_acc,
            "avg_confidence": float(confidences.mean()),
        })

        all_position_correct.append(correct)
        all_position_confidence.append(confidences)

    # Aggregate accuracy by position (averaged across samples)
    min_len = min(len(c) for c in all_position_correct)
    acc_by_pos = np.mean([c[:min_len] for c in all_position_correct], axis=0)
    conf_by_pos = np.mean([c[:min_len] for c in all_position_confidence], axis=0)

    overall_acc = np.mean([r["accuracy"] for r in results])
    overall_1st = np.mean([r["first_half_accuracy"] for r in results])
    overall_2nd = np.mean([r["second_half_accuracy"] for r in results])

    return {
        "n_samples": len(passages),
        "passage_length": int(passages.shape[1]),
        "samples": results,
        "aggregate": {
            "overall_accuracy": float(overall_acc),
            "first_half_accuracy": float(overall_1st),
            "second_half_accuracy": float(overall_2nd),
            "overall_confidence": float(np.mean([r["avg_confidence"] for r in results])),
            "accuracy_by_position": acc_by_pos.tolist(),
            "confidence_by_position": conf_by_pos.tolist(),
        },
    }


# ─────────────────────────────────────────────────────
# 4. CAPACITY SATURATION ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_capacity(model) -> dict:
    """Analyze whether the model's parameters are saturated.

    Per-layer: weight std, dead weight fraction, rank utilization (for 2D matrices).
    """
    all_weights = []
    total_params = 0
    near_zero_params = 0

    for _name, param in model.named_parameters():
        w = param.detach().cpu().numpy().flatten()
        all_weights.extend(w.tolist())
        total_params += len(w)
        near_zero_params += int((np.abs(w) < 0.001).sum())

    all_weights_arr = np.array(all_weights)

    global_stats = {
        "total_params": total_params,
        "mean": float(all_weights_arr.mean()),
        "std": float(all_weights_arr.std()),
        "min": float(all_weights_arr.min()),
        "max": float(all_weights_arr.max()),
        "near_zero_fraction": near_zero_params / total_params,
    }

    # Per-layer analysis
    layers = []
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        w = param.detach().cpu().numpy()
        flat = w.flatten()

        layer_info = {
            "name": name,
            "shape": list(w.shape),
            "n_params": len(flat),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "abs_mean": float(np.abs(flat).mean()),
            "near_zero_fraction": float((np.abs(flat) < 0.001).sum() / len(flat)),
            "max_abs": float(np.abs(flat).max()),
        }

        # For 2D weight matrices: rank utilization via SVD
        if len(w.shape) == 2:
            try:
                sv = np.linalg.svd(w, compute_uv=False)
                total_var = (sv ** 2).sum()
                cumvar = np.cumsum(sv ** 2) / total_var
                effective_rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
                max_rank = min(w.shape)
                layer_info["effective_rank_90pct"] = effective_rank_90
                layer_info["max_rank"] = max_rank
                layer_info["rank_utilization"] = effective_rank_90 / max_rank
            except Exception:
                pass

        layers.append(layer_info)

    return {
        "global": global_stats,
        "layers": layers,
    }


# ─────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────

def plot_attention(results: dict, output_dir: Path):
    """Attention entropy and locality heatmaps."""
    if not HAS_MATPLOTLIB:
        return

    n_layers = results["n_layers"]
    n_heads = results["n_heads"]
    entropy = np.array(results["entropy_matrix"])
    locality = np.array(results["locality_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Entropy heatmap
    im0 = axes[0].imshow(entropy, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Attention Entropy\n(1.0 = uniform/unfocused, 0.0 = sharp/focused)")
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    # Annotate cells
    for l in range(n_layers):
        for h in range(n_heads):
            axes[0].text(h, l, f"{entropy[l, h]:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if entropy[l, h] > 0.3 else "white")
    plt.colorbar(im0, ax=axes[0])

    # Locality heatmap
    im1 = axes[1].imshow(locality, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].set_title("Locality Score\n(1.0 = only nearby tokens, 0.0 = long-range)")
    axes[1].set_xticks(range(n_heads))
    axes[1].set_yticks(range(n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            axes[1].text(h, l, f"{locality[l, h]:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if locality[l, h] < 0.7 else "white")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle("Attention Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "attention_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embeddings(results: dict, output_dir: Path):
    """Embedding SVD spectrum, group similarity, cross-group similarity."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Singular value spectrum
    svs = results["singular_values_top50"]
    axes[0].bar(range(len(svs)), svs, color="steelblue", alpha=0.8)
    dim90 = results["dims_for_90pct"]
    if dim90 - 1 < len(svs):
        axes[0].axhline(y=svs[dim90 - 1], color="red", linestyle="--",
                         label=f"90% variance (dim {dim90})")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title(
        f"Embedding Singular Values\n"
        f"(90%: {dim90}, 95%: {results['dims_for_95pct']}, "
        f"99%: {results['dims_for_99pct']} of {results['d_model']} dims)"
    )
    if dim90 - 1 < len(svs):
        axes[0].legend()

    # Within-group similarities
    groups = results["group_similarities"]
    group_names = list(groups.keys())
    within_sims = [groups[g]["avg_cosine_similarity"] for g in group_names]
    random_bl = results["random_baseline_similarity"]

    axes[1].bar(range(len(group_names)), within_sims, color="coral", alpha=0.8)
    axes[1].axhline(y=random_bl, color="gray", linestyle="--",
                     label=f"Random baseline: {random_bl:.3f}")
    axes[1].set_xticks(range(len(group_names)))
    axes[1].set_xticklabels(group_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Avg Cosine Similarity")
    axes[1].set_title("Within-Group Token Similarity\n(Higher = better clustering)")
    axes[1].legend()

    # Cross-group similarities
    cross = results["cross_group_similarities"]
    cross_names = list(cross.keys())
    cross_vals = list(cross.values())

    axes[2].barh(range(len(cross_names)), cross_vals, color="mediumpurple", alpha=0.8)
    axes[2].set_yticks(range(len(cross_names)))
    axes[2].set_yticklabels(cross_names, fontsize=7)
    axes[2].axvline(x=random_bl, color="gray", linestyle="--", label="Random baseline")
    axes[2].set_xlabel("Cosine Similarity")
    axes[2].set_title("Cross-Group Similarity\n(Lower = better separation)")
    axes[2].legend()

    fig.suptitle("Embedding Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(results: dict, output_dir: Path):
    """Token prediction accuracy and confidence plots."""
    if not HAS_MATPLOTLIB:
        return

    agg = results["aggregate"]
    samples = results["samples"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy by position (smoothed)
    acc_by_pos = np.array(agg["accuracy_by_position"])
    if len(acc_by_pos) > 20:
        # Rolling average with window=20
        kernel = np.ones(20) / 20
        smoothed = np.convolve(acc_by_pos, kernel, mode="valid")
        x = range(len(smoothed))
        axes[0].plot(x, smoothed, "b-", alpha=0.8, linewidth=1.5)
    else:
        axes[0].plot(range(len(acc_by_pos)), acc_by_pos, "b-o", markersize=3)
    axes[0].set_xlabel("Position in Passage")
    axes[0].set_ylabel("Accuracy (Top-1 Match)")
    axes[0].set_title("Prediction Accuracy by Position\n(Does it get worse deeper into the passage?)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Confidence by position (smoothed)
    conf_by_pos = np.array(agg["confidence_by_position"])
    if len(conf_by_pos) > 20:
        smoothed_c = np.convolve(conf_by_pos, kernel, mode="valid")
        axes[1].plot(range(len(smoothed_c)), smoothed_c, "r-", alpha=0.8, linewidth=1.5)
    else:
        axes[1].plot(range(len(conf_by_pos)), conf_by_pos, "r-o", markersize=3)
    axes[1].set_xlabel("Position in Passage")
    axes[1].set_ylabel("P(correct token)")
    axes[1].set_title("Confidence in Correct Token\n(How sure is it about the right answer?)")
    axes[1].grid(True, alpha=0.3)

    # First half vs second half
    first_halves = [s["first_half_accuracy"] for s in samples]
    second_halves = [s["second_half_accuracy"] for s in samples]
    x = range(len(samples))
    width = 0.35
    axes[2].bar([i - width / 2 for i in x], first_halves, width,
                label="First half", color="steelblue", alpha=0.8)
    axes[2].bar([i + width / 2 for i in x], second_halves, width,
                label="Second half", color="coral", alpha=0.8)
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("First Half vs Second Half Accuracy\n(Measures error compounding)")
    axes[2].legend()
    axes[2].set_ylim(0, 1.05)

    fig.suptitle("Token Prediction Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "token_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity(results: dict, output_dir: Path):
    """Capacity saturation analysis: weight spread, dead weights, rank utilization."""
    if not HAS_MATPLOTLIB:
        return

    layers = results["layers"]
    layer_names = [l["name"].replace("blocks.", "b").replace(".weight", "")
                   for l in layers]
    stds = [l["std"] for l in layers]
    near_zeros = [l["near_zero_fraction"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Weight std
    axes[0].barh(range(len(layer_names)), stds, color="steelblue", alpha=0.8)
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels(layer_names, fontsize=6)
    axes[0].set_xlabel("Weight Std Dev")
    axes[0].set_title("Weight Spread by Layer\n(Low std = not learning much)")
    axes[0].axvline(x=0.02, color="red", linestyle="--", label="Init std (0.02)")
    axes[0].legend()

    # Dead weight fraction
    axes[1].barh(range(len(layer_names)), near_zeros, color="coral", alpha=0.8)
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names, fontsize=6)
    axes[1].set_xlabel("Fraction Near Zero (<0.001)")
    axes[1].set_title("Dead Weight Fraction\n(High = wasted capacity)")

    # Rank utilization (only for 2D matrices)
    rank_layers = [l for l in layers if "rank_utilization" in l]
    if rank_layers:
        r_names = [l["name"].replace("blocks.", "b").replace(".weight", "")
                   for l in rank_layers]
        r_utils = [l["rank_utilization"] for l in rank_layers]
        axes[2].barh(range(len(r_names)), r_utils, color="mediumpurple", alpha=0.8)
        axes[2].set_yticks(range(len(r_names)))
        axes[2].set_yticklabels(r_names, fontsize=6)
        axes[2].set_xlabel("Rank Utilization (90% variance)")
        axes[2].set_title("Matrix Rank Utilization\n(1.0 = fully used, low = spare capacity)")
        axes[2].set_xlim(0, 1.05)

    fig.suptitle("Capacity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────

def print_report(all_results: dict):
    """Print a formatted diagnostic summary to console."""

    print(f"\n{'=' * 80}")
    print(f"  MODEL DIAGNOSTIC REPORT")
    print(f"  Checkpoint: {all_results['checkpoint']}")
    print(f"  Timestamp: {all_results['timestamp']}")
    print(f"{'=' * 80}")

    # ── Attention ──
    attn = all_results["attention"]
    print(f"\n{'─' * 80}")
    print("  1. ATTENTION PATTERN ANALYSIS")
    print(f"{'─' * 80}")
    print(f"    Avg Normalized Entropy: {attn['avg_entropy']:.3f} (1.0=uniform, 0.0=focused)")
    print(f"    Avg Locality Score:     {attn['avg_locality']:.3f} (1.0=nearby only, 0.0=long-range)")

    if attn["avg_entropy"] > 0.85:
        print("    WARNING: Most heads are attending almost uniformly — not learning specific patterns")
    elif attn["avg_entropy"] > 0.7:
        print("    MODERATE: Some focus developing, but still quite diffuse")
    else:
        print("    GOOD: Heads are learning focused attention patterns")

    # Count head types
    flat_types = [t for row in attn["head_types"] for t in row]
    n_local = flat_types.count("local")
    n_long = flat_types.count("long_range")
    n_mixed = flat_types.count("mixed")
    total_heads = len(flat_types)
    print(f"    Head types: {n_local} local ({n_local/total_heads:.0%}), "
          f"{n_long} long-range ({n_long/total_heads:.0%}), "
          f"{n_mixed} mixed ({n_mixed/total_heads:.0%})")

    # ── Embeddings ──
    emb = all_results["embeddings"]
    print(f"\n{'─' * 80}")
    print("  2. EMBEDDING SPACE ANALYSIS")
    print(f"{'─' * 80}")
    print(f"    Embedding dim: {emb['d_model']}, Vocab size: {emb['vocab_size']}")
    print(f"    Effective dimensions (90% variance): {emb['dims_for_90pct']} / {emb['d_model']}")
    print(f"    Effective dimensions (95% variance): {emb['dims_for_95pct']} / {emb['d_model']}")
    print(f"    Effective dimensions (99% variance): {emb['dims_for_99pct']} / {emb['d_model']}")

    utilization = emb["dims_for_90pct"] / emb["d_model"]
    if utilization < 0.3:
        print(f"    WARNING: Only {utilization:.0%} of dimensions carry 90% of info — underused")
    elif utilization > 0.7:
        print(f"    SATURATED: {utilization:.0%} of dimensions needed for 90% — no room to grow")
    else:
        print(f"    MODERATE: {utilization:.0%} utilization — reasonable use of space")

    random_bl = emb["random_baseline_similarity"]
    print(f"\n    Concept clustering (within-group cosine similarity):")
    print(f"    {'Group':<20} {'Similarity':<12} {'vs Random':<12} {'Status'}")
    print(f"    {'─' * 60}")
    for group, info in emb["group_similarities"].items():
        sim = info["avg_cosine_similarity"]
        delta = sim - random_bl
        status = "Clustered" if delta > 0.05 else "Not clustered" if delta < 0.01 else "Weak"
        print(f"    {group:<20} {sim:<12.4f} {delta:+<12.4f} {status}")
    print(f"    Random baseline: {random_bl:.4f}")

    # ── Predictions ──
    pred = all_results["predictions"]
    print(f"\n{'─' * 80}")
    print("  3. TOKEN PREDICTION ANALYSIS")
    print(f"{'─' * 80}")
    agg = pred["aggregate"]
    print(f"    Overall top-1 accuracy (teacher-forced): {agg['overall_accuracy']:.1%}")
    print(f"    First half accuracy:  {agg['first_half_accuracy']:.1%}")
    print(f"    Second half accuracy: {agg['second_half_accuracy']:.1%}")
    print(f"    Avg confidence (P(correct)): {agg['overall_confidence']:.3f}")

    if agg["second_half_accuracy"] < agg["first_half_accuracy"] * 0.7:
        drop = (agg["first_half_accuracy"] - agg["second_half_accuracy"]) / agg["first_half_accuracy"] * 100
        print(f"    WARNING: Accuracy drops {drop:.0f}% from first to second half")
    elif agg["second_half_accuracy"] < agg["first_half_accuracy"] * 0.9:
        print("    MODERATE: Some degradation in later positions")
    else:
        print("    STABLE: Accuracy holds across passage length")

    # ── Capacity ──
    cap = all_results["capacity"]
    print(f"\n{'─' * 80}")
    print("  4. CAPACITY SATURATION ANALYSIS")
    print(f"{'─' * 80}")
    g = cap["global"]
    print(f"    Total parameters: {g['total_params']:,}")
    print(f"    Weight std: {g['std']:.4f} (init was 0.02)")
    print(f"    Near-zero fraction: {g['near_zero_fraction']:.1%}")

    if g["near_zero_fraction"] > 0.3:
        print("    WARNING: >30% near-zero weights — significant wasted capacity")
    elif g["near_zero_fraction"] > 0.15:
        print("    MODERATE: 15-30% near-zero weights")
    else:
        print("    GOOD: Most parameters are being used")

    rank_layers = [l for l in cap["layers"] if "rank_utilization" in l]
    if rank_layers:
        avg_rank = np.mean([l["rank_utilization"] for l in rank_layers])
        print(f"    Avg rank utilization: {avg_rank:.1%}")
        saturated = sum(1 for l in rank_layers if l["rank_utilization"] > 0.8)
        spare = sum(1 for l in rank_layers if l["rank_utilization"] < 0.3)
        print(f"    Saturated layers (>80%): {saturated}/{len(rank_layers)}")
        print(f"    Spare capacity layers (<30%): {spare}/{len(rank_layers)}")

    print(f"\n{'=' * 80}")
    print("  END OF DIAGNOSTIC REPORT")
    print(f"{'=' * 80}\n")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model Diagnostics")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "finetune"],
                        help="Stage: pretrain or finetune (determines checkpoint & output paths)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-detected from --stage if not set)")
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Number of sample passages for analysis")
    parser.add_argument("--passage-length", type=int, default=512,
                        help="Token length of each sample passage")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plot generation")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU (default behavior; MPS not recommended for diagnostics)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (auto-detected from --stage if not set)")
    args = parser.parse_args()

    # Resolve checkpoint and output paths from stage
    if args.checkpoint is None:
        if args.stage == "pretrain":
            args.checkpoint = str(ROOT / "v1" / "checkpoints" / "pretrained" / "best")
        else:
            args.checkpoint = str(ROOT / "v1" / "checkpoints" / "finetuned" / "best")
    if args.output is None:
        args.output = str(ROOT / "v1" / "logs" / args.stage / "diagnostics.json")

    print(f"Stage: {args.stage}")

    # Always use CPU for diagnostics — avoids MPS memory issues
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = GPT2.from_pretrained(args.checkpoint, device=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params:,} params)")

    # Load tokenizer
    from bpe.tokenizer import Tokenizer
    tokenizer = Tokenizer.from_files(str(ROOT / "tokenizer_output"))

    # Load sample passages
    cache_path = ROOT / "v1" / "data" / "cache" / "tokenized.npy"
    print(f"Loading {args.n_samples} sample passages ({args.passage_length} tokens each)...")
    passages = load_sample_passages(cache_path, args.n_samples, args.passage_length)
    print(f"Loaded {passages.shape[0]} passages")

    # Run analyses
    t0 = time.time()

    print("\n[1/4] Analyzing attention patterns...")
    attention_results = analyze_attention(model, passages, device)
    print(f"  Done ({time.time() - t0:.1f}s)")

    t1 = time.time()
    print("[2/4] Analyzing embedding space...")
    embedding_results = analyze_embeddings(model, tokenizer)
    print(f"  Done ({time.time() - t1:.1f}s)")

    t2 = time.time()
    print("[3/4] Analyzing token predictions...")
    prediction_results = analyze_token_predictions(model, passages, device)
    print(f"  Done ({time.time() - t2:.1f}s)")

    t3 = time.time()
    print("[4/4] Analyzing capacity...")
    capacity_results = analyze_capacity(model)
    print(f"  Done ({time.time() - t3:.1f}s)")

    total_time = time.time() - t0
    print(f"\nAll analyses complete ({total_time:.1f}s total)")

    # Combine results
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "stage": args.stage,
        "n_samples": args.n_samples,
        "passage_length": args.passage_length,
        "attention": attention_results,
        "embeddings": embedding_results,
        "predictions": prediction_results,
        "capacity": capacity_results,
    }

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Generate plots
    if not args.no_plots:
        plot_dir = ROOT / "v1" / "plots" / "diagnostics" / args.stage
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating plots in {plot_dir}...")
        plot_attention(attention_results, plot_dir)
        plot_embeddings(embedding_results, plot_dir)
        plot_predictions(prediction_results, plot_dir)
        plot_capacity(capacity_results, plot_dir)
        print(f"Plots saved to {plot_dir}/")

    # Print report
    print_report(all_results)


if __name__ == "__main__":
    main()
