"""Training loop for GPT-2 pretraining with MLX.

MLX differences from PyTorch:
  - Uses mlx.nn.value_and_grad for forward+backward in one call
  - Lazy evaluation: mx.eval() forces computation
  - No DataLoader: we use a simple batch iterator
  - Optimizer updates via optimizer.update(model, grads)
  - Gradient accumulation done manually by averaging grads
"""

import json
import logging
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from v1_mlx.model.gpt2 import GPT2
from v1_mlx.data.dataset import batch_iterator
from v1_mlx.utils.device import get_memory_stats

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with metrics logging for the dashboard."""

    def __init__(
        self,
        model: GPT2,
        train_chunks,
        val_chunks,
        config: dict,
        save_dir: str | Path,
        metrics_path: str | Path | None = None,
    ):
        self.model = model
        self.train_chunks = train_chunks
        self.val_chunks = val_chunks
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics file for dashboard
        self.metrics_path = Path(metrics_path) if metrics_path else self.save_dir / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("")

        # Optimizer
        lr = config.get("learning_rate", 0.0006)
        betas = config.get("betas", [0.9, 0.95])
        weight_decay = config.get("weight_decay", 0.01)
        self.optimizer = optim.AdamW(
            learning_rate=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        # Loss + grad function
        self.loss_and_grad_fn = nn.value_and_grad(model, model.loss)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine decay with warmup."""
        warmup_fraction = self.config.get("warmup_fraction", 0.05)
        warmup_steps = int(total_steps * warmup_fraction)
        lr = self.config.get("learning_rate", 0.0006)
        min_lr = self.config.get("min_lr", 0.00006)

        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        elif step >= total_steps:
            return min_lr
        else:
            decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (lr - min_lr)

    def _log_metric(self, data: dict):
        """Append a metric entry to the JSONL file."""
        data["timestamp"] = time.time()
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def evaluate(self) -> float:
        """Evaluate on validation set, return average loss."""
        batch_size = self.config.get("batch_size", 8)
        total_loss = 0.0
        n_batches = 0

        for input_ids, targets in batch_iterator(
            self.val_chunks, batch_size, shuffle=False
        ):
            loss = self.model.loss(input_ids, targets)
            mx.eval(loss)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self) -> dict:
        """Run the full training loop."""
        batch_size = self.config.get("batch_size", 8)
        max_epochs = self.config.get("max_epochs", 100)
        grad_clip = self.config.get("grad_clip", 1.0)
        patience = self.config.get("patience", 10)
        min_delta = self.config.get("min_delta", 0.001)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)

        n_train = len(self.train_chunks)
        steps_per_epoch = n_train // batch_size
        total_steps = steps_per_epoch * max_epochs
        weight_updates_per_epoch = steps_per_epoch // grad_accum_steps
        context_length = self.train_chunks.shape[1]
        tokens_per_update = batch_size * grad_accum_steps * (context_length - 1)

        logger.info(
            f"Training: {max_epochs} max epochs, {steps_per_epoch} steps/epoch, "
            f"{weight_updates_per_epoch} weight updates/epoch, "
            f"batch={batch_size}, accum={grad_accum_steps}, "
            f"effective_batch={batch_size * grad_accum_steps}"
        )

        n_params = sum(p.size for _, p in tree_flatten(self.model.parameters()))
        self._log_metric({
            "type": "start",
            "max_epochs": max_epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_train_chunks": n_train,
            "total_val_chunks": len(self.val_chunks),
            "context_length": context_length,
            "total_params": n_params,
        })

        start_time = time.time()
        final_epoch = 0

        for epoch in range(max_epochs):
            final_epoch = epoch + 1
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            n_batches = 0
            n_updates = 0
            epoch_start = time.time()

            # Accumulated gradients
            accum_grads = None

            for step, (input_ids, targets) in enumerate(
                batch_iterator(self.train_chunks, batch_size, shuffle=True, seed=42 + epoch)
            ):
                # Update learning rate
                lr = self._get_lr(self.global_step, total_steps)
                self.optimizer.learning_rate = lr

                # Forward + backward
                loss, grads = self.loss_and_grad_fn(input_ids, targets)
                mx.eval(loss)

                # Accumulate gradients
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map_add(accum_grads, grads)

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                    # Average accumulated gradients
                    n_accum = min(grad_accum_steps, (step % grad_accum_steps) + 1)
                    accum_grads = tree_map_scale(accum_grads, 1.0 / n_accum)

                    # Gradient clipping
                    grad_norm = compute_grad_norm(accum_grads)
                    if grad_clip > 0 and grad_norm > grad_clip:
                        accum_grads = tree_map_scale(accum_grads, grad_clip / grad_norm)

                    epoch_grad_norm += grad_norm

                    # Update weights
                    self.optimizer.update(self.model, accum_grads)
                    mx.eval(self.model.parameters(), self.optimizer.state)
                    accum_grads = None
                    n_updates += 1

                    # Log step metrics
                    self._log_metric({
                        "type": "step",
                        "epoch": epoch + 1,
                        "step": self.global_step,
                        "train_loss": loss.item(),
                        "lr": lr,
                        "grad_norm": grad_norm,
                    })

                epoch_loss += loss.item()
                n_batches += 1
                self.global_step += 1

                # Log progress every 100 steps
                if self.global_step % 100 == 0:
                    avg_loss = epoch_loss / n_batches
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - start_time
                    tokens_seen = self.global_step * batch_size * (context_length - 1)
                    tokens_per_sec = tokens_seen / elapsed
                    logger.info(
                        f"Step {self.global_step} | Epoch {epoch+1}/{max_epochs} | "
                        f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | "
                        f"Time: {elapsed:.0f}s | {tokens_per_sec:.0f} tok/s"
                    )

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            avg_grad_norm = epoch_grad_norm / max(n_updates, 1)
            train_ppl = math.exp(min(avg_epoch_loss, 20))

            # Validation
            val_loss = self.evaluate()
            val_ppl = math.exp(min(val_loss, 20))

            self.train_losses.append({"epoch": epoch + 1, "loss": avg_epoch_loss})
            self.val_losses.append({"epoch": epoch + 1, "loss": val_loss})

            mem = get_memory_stats()

            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Train Loss: {avg_epoch_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                f"Time: {epoch_time:.1f}s | Mem: {mem['allocated_gb']:.1f}GB"
            )

            is_best = val_loss < self.best_val_loss - min_delta
            if is_best:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
                logger.info(f"  New best (val_loss={val_loss:.4f}, improved by {improvement:.4f})")
            else:
                self.patience_counter += 1
                gap = val_loss - self.best_val_loss
                logger.info(
                    f"  No improvement (gap={gap:+.4f}, "
                    f"patience={self.patience_counter}/{patience})"
                )

            self._log_metric({
                "type": "epoch",
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "lr": lr,
                "grad_norm": avg_grad_norm,
                "is_best": is_best,
                "patience": self.patience_counter,
                "epoch_time": epoch_time,
                "memory_gb": mem["allocated_gb"],
            })

            if patience > 0 and self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

        self._save_checkpoint("final")
        self._save_training_log()

        total_time = time.time() - start_time
        stats = {
            "total_steps": self.global_step,
            "total_epochs": final_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": math.exp(min(self.best_val_loss, 20)),
            "total_time_seconds": total_time,
        }
        self._log_metric({"type": "end", **stats})
        logger.info(f"Training complete: {stats}")
        return stats

    def _save_checkpoint(self, name: str):
        ckpt_dir = self.save_dir / name
        self.model.save_pretrained(str(ckpt_dir))
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _save_training_log(self):
        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        log_path = self.save_dir / "training_log.json"
        log_path.write_text(json.dumps(log, indent=2))


# --- Tree utilities for gradient manipulation ---

def tree_map_add(tree_a, tree_b):
    """Element-wise add two parameter trees."""
    result = {}
    for key in tree_a:
        if isinstance(tree_a[key], dict):
            result[key] = tree_map_add(tree_a[key], tree_b[key])
        else:
            result[key] = tree_a[key] + tree_b[key]
    return result


def tree_map_scale(tree, scale: float):
    """Scale all arrays in a parameter tree."""
    result = {}
    for key in tree:
        if isinstance(tree[key], dict):
            result[key] = tree_map_scale(tree[key], scale)
        else:
            result[key] = tree[key] * scale
    return result


def compute_grad_norm(grads) -> float:
    """Compute the global L2 norm of all gradients."""
    flat = tree_flatten(grads)
    total = 0.0
    for _, g in flat:
        total += mx.sum(g * g).item()
    return math.sqrt(total)
