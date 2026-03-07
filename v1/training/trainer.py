"""Training loop for GPT-2 pretraining.

Features:
  - Cosine LR schedule with warmup
  - Gradient accumulation
  - Gradient clipping
  - Early stopping with patience and min_delta
  - Metrics logging to JSONL (for real-time dashboard)
  - Per-head weight statistics logging
  - Graceful stop signal (echo "STOP" > v1/STOP to stop training)
  - Checkpoint resumption (model + optimizer + training state)
  - Periodic generation samples
  - MPS memory management
"""

import gc
import json
import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from v1.model.gpt2 import GPT2
from v1.utils.device import mps_empty_cache, mps_synchronize, get_memory_stats

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with metrics logging for the dashboard."""

    def __init__(
        self,
        model: GPT2,
        train_dataset,
        val_dataset,
        config: dict,
        save_dir: str | Path,
        device: torch.device,
        metrics_path: str | Path | None = None,
        stop_file: str | Path | None = None,
        resume: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Metrics file for dashboard
        self.metrics_path = Path(metrics_path) if metrics_path else self.save_dir / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Weight stats file (per-head, per-layer stats each epoch)
        self.weight_stats_path = self.metrics_path.parent / "weight_stats.jsonl"

        if not resume:
            # Fresh start — clear previous metrics
            self.metrics_path.write_text("")
            self.weight_stats_path.write_text("")

        # Graceful stop signal — echo "STOP" > this file to stop training cleanly
        self.stop_file = Path(stop_file) if stop_file else Path("v1/STOP")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        # Resume from checkpoint if requested
        if resume:
            self._load_resume_checkpoint()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW with weight decay only on eligible params."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay on: LayerNorm, biases, position embeddings
            if "ln_" in name or "bias" in name or "wpe" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.get("weight_decay", 0.01)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        lr = self.config.get("learning_rate", 6e-4)
        betas = tuple(self.config.get("betas", [0.9, 0.95]))

        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine decay with warmup."""
        warmup_fraction = self.config.get("warmup_fraction", 0.05)
        warmup_steps = int(total_steps * warmup_fraction)
        lr = self.config.get("learning_rate", 6e-4)
        min_lr = self.config.get("min_lr", 6e-5)

        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        elif step >= total_steps:
            return min_lr
        else:
            decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (lr - min_lr)

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _log_metric(self, data: dict):
        """Append a metric entry to the JSONL file."""
        data["timestamp"] = time.time()
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _tensor_stats(self, t: torch.Tensor) -> dict:
        """Compute summary statistics for a weight tensor."""
        t_flat = t.float().flatten()
        return {
            "min": round(t_flat.min().item(), 6),
            "max": round(t_flat.max().item(), 6),
            "mean": round(t_flat.mean().item(), 6),
            "median": round(t_flat.median().item(), 6),
            "std": round(t_flat.std().item(), 6),
            "l2_norm": round(t_flat.norm(2).item(), 4),
            "numel": t_flat.numel(),
        }

    @torch.no_grad()
    def _compute_weight_stats(self, epoch: int):
        """Compute per-head and per-layer weight statistics.

        Logs to weight_stats.jsonl with structure:
        - wte, wpe: embedding stats
        - blocks[i].attn_heads[h].{q,k,v,o}: per-head attention stats
        - blocks[i].{mlp_fc, mlp_proj, ln_1, ln_2}: per-layer stats
        - ln_f: final layernorm stats
        """
        model = self.model
        d_model = model.config.d_model
        n_heads = model.config.n_heads
        head_dim = d_model // n_heads

        stats = {"type": "weight_stats", "epoch": epoch}

        # Embeddings
        stats["wte"] = self._tensor_stats(model.wte.weight)
        stats["wpe"] = self._tensor_stats(model.wpe.weight)

        # Per block
        blocks_stats = []
        for i, block in enumerate(model.blocks):
            block_stats = {"layer": i}

            # Attention: per-head Q, K, V from fused c_attn
            # c_attn.weight shape: (3*d_model, d_model) -> (3, n_heads, head_dim, d_model)
            qkv_w = block.attn.c_attn.weight.view(3, n_heads, head_dim, d_model)

            # Output projection: per-head O
            # c_proj.weight shape: (d_model, d_model) -> (d_model, n_heads, head_dim)
            o_w = block.attn.c_proj.weight.view(d_model, n_heads, head_dim)

            heads = []
            for h in range(n_heads):
                heads.append({
                    "head": h,
                    "q": self._tensor_stats(qkv_w[0, h]),   # (head_dim, d_model)
                    "k": self._tensor_stats(qkv_w[1, h]),   # (head_dim, d_model)
                    "v": self._tensor_stats(qkv_w[2, h]),   # (head_dim, d_model)
                    "o": self._tensor_stats(o_w[:, h, :]),   # (d_model, head_dim)
                })
            block_stats["attn_heads"] = heads

            # MLP weights
            block_stats["mlp_fc"] = self._tensor_stats(block.mlp.c_fc.weight)
            block_stats["mlp_proj"] = self._tensor_stats(block.mlp.c_proj.weight)

            # LayerNorms
            block_stats["ln_1"] = self._tensor_stats(block.ln_1.weight)
            block_stats["ln_2"] = self._tensor_stats(block.ln_2.weight)

            blocks_stats.append(block_stats)

        stats["blocks"] = blocks_stats
        stats["ln_f"] = self._tensor_stats(model.ln_f.weight)
        stats["timestamp"] = time.time()

        with open(self.weight_stats_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # Clean up temporary tensors from stats computation
        del stats
        gc.collect()

        logger.info(f"  Weight stats logged for epoch {epoch}")

    def _check_stop_signal(self) -> bool:
        """Check if user has signaled to stop training via STOP file.

        Usage:
            echo "STOP" > v1/STOP   — stop after current epoch
            echo "" > v1/STOP       — clear the signal (no-op)

        File is never deleted by code. User controls its lifecycle.
        """
        if not self.stop_file.exists():
            return False
        try:
            content = self.stop_file.read_text().strip()
            return content == "STOP"
        except OSError:
            return False

    def _check_lr_override(self) -> float | None:
        """Check if user has requested a learning rate change via LR file.

        Usage:
            echo "0.001" > v1/LR   — set base LR to 1e-3 (persists until removed)
            rm v1/LR               — revert to config.yaml / checkpoint LR

        File is never deleted by code. User controls its lifecycle.
        """
        lr_file = self.stop_file.parent / "LR"
        if not lr_file.exists():
            return None
        try:
            new_lr = float(lr_file.read_text().strip())
            if not (0 < new_lr < 1):
                logger.warning(f"  LR override ignored: {new_lr} not in (0, 1)")
                return None
            old_lr = self.config.get("learning_rate", 6e-4)
            if abs(new_lr - old_lr) > 1e-10:
                self.config["learning_rate"] = new_lr
                old_min = self.config.get("min_lr", 6e-5)
                self.config["min_lr"] = new_lr * (old_min / old_lr)
                logger.info(
                    f"  LR override: {old_lr:.6f} -> {new_lr:.6f} "
                    f"(min_lr: {old_min:.6f} -> {self.config['min_lr']:.6f})"
                )
            return new_lr
        except (ValueError, OSError) as e:
            logger.warning(f"  LR override failed: {e}")
            return None

    def _save_resume_checkpoint(self, epoch: int):
        """Save full training state for resumption.

        Saves model weights, optimizer state, and all training counters
        so training can be resumed exactly where it left off.
        """
        resume_dir = self.save_dir / "resume"
        resume_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Persist LR overrides so they survive process restarts
            "learning_rate": self.config.get("learning_rate"),
            "min_lr": self.config.get("min_lr"),
        }
        torch.save(state, resume_dir / "training_state.pt")
        logger.info(f"  Resume checkpoint saved (epoch {epoch})")

    def _load_resume_checkpoint(self):
        """Load training state from resume checkpoint.

        Restores model weights, optimizer state (including momentum buffers),
        and all training counters.
        """
        resume_path = self.save_dir / "resume" / "training_state.pt"
        if not resume_path.exists():
            logger.warning(f"No resume checkpoint found at {resume_path} — starting fresh")
            return

        logger.info(f"Loading resume checkpoint from {resume_path}")
        state = torch.load(resume_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            logger.info("  No optimizer state — starting with fresh optimizer (momentum will restart)")
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]
        self.patience_counter = state["patience_counter"]
        self.train_losses = state["train_losses"]
        self.val_losses = state["val_losses"]

        # Restore LR overrides if they were saved
        if state.get("learning_rate") is not None:
            old_lr = self.config.get("learning_rate")
            self.config["learning_rate"] = state["learning_rate"]
            if state.get("min_lr") is not None:
                self.config["min_lr"] = state["min_lr"]
            if old_lr != state["learning_rate"]:
                logger.info(
                    f"  Restored LR override: {old_lr} -> {state['learning_rate']} "
                    f"(min_lr: {state.get('min_lr')})"
                )

        logger.info(
            f"Resumed from epoch {self.start_epoch} | "
            f"global_step={self.global_step} | "
            f"best_val_loss={self.best_val_loss:.4f} | "
            f"patience={self.patience_counter}"
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set, return average loss."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            _, loss = self.model(input_ids, targets=labels)
            total_loss += loss.detach()
            n_batches += 1

        self.model.train()
        # Single .item() call at end of validation instead of per-batch
        return (total_loss / max(n_batches, 1)).item()

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with training statistics.
        """
        batch_size = self.config.get("batch_size", 8)
        max_epochs = self.config.get("max_epochs", 100)
        grad_clip = self.config.get("grad_clip", 1.0)
        patience = self.config.get("patience", 10)
        min_delta = self.config.get("min_delta", 0.001)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)

        n_train = len(self.train_dataset)
        if n_train == 0:
            logger.warning("Training dataset is empty — skipping")
            return {"total_steps": 0, "total_epochs": 0, "best_val_loss": float("inf")}

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * max_epochs
        weight_updates_per_epoch = steps_per_epoch // grad_accum_steps
        context_length = self.train_dataset.chunks.shape[1]
        tokens_per_update = batch_size * grad_accum_steps * (context_length - 1)

        logger.info(
            f"Training: {max_epochs} max epochs, {steps_per_epoch} steps/epoch, "
            f"{weight_updates_per_epoch} weight updates/epoch, "
            f"batch={batch_size}, accum={grad_accum_steps}, "
            f"effective_batch={batch_size * grad_accum_steps}"
        )
        logger.info(
            f"Data: {n_train} train chunks, {len(self.val_dataset)} val chunks, "
            f"context={context_length}, tokens/update={tokens_per_update:,}"
        )

        # Log training start/resume
        self._log_metric({
            "type": "resume" if self.start_epoch > 0 else "start",
            "max_epochs": max_epochs,
            "start_epoch": self.start_epoch,
            "steps_per_epoch": steps_per_epoch,
            "total_train_chunks": n_train,
            "total_val_chunks": len(self.val_dataset),
            "context_length": context_length,
            "total_params": sum(p.numel() for p in self.model.parameters()),
        })

        if self.start_epoch > 0:
            logger.info(f"Resuming from epoch {self.start_epoch + 1}")

        self.model.train()
        start_time = time.time()
        start_step = self.global_step  # Track session start for accurate tok/s
        last_log_time = start_time  # Track time per 100-step interval
        final_epoch = self.start_epoch

        for epoch in range(self.start_epoch, max_epochs):
            final_epoch = epoch + 1
            # Accumulate loss as tensor to avoid .item() in hot loop.
            # .item() forces MPS synchronization (GPU→CPU copy) every call,
            # which stalls when MPS memory pressure is high (43+ GB on 48 GB).
            epoch_loss_tensor = torch.tensor(0.0, device=self.device)
            n_batches = 0
            n_updates = 0
            epoch_start = time.time()

            for step, batch in enumerate(train_loader):
                # Update learning rate
                lr = self._get_lr(self.global_step, total_steps)
                self._set_lr(lr)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward
                _, loss = self.model(input_ids, targets=labels)
                loss = loss / grad_accum_steps

                # Backward
                loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                    # Gradient clipping
                    # NOTE: Do NOT call .item() on clip_grad_norm_ return value —
                    # this causes MPS memory leak (PyTorch issue #154329).
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )

                    # Snapshot grad norm BEFORE zero_grad (on GPU, no .item()).
                    # Overwrites each update; final value = last update's grad norm.
                    last_grad_norm_sq = torch.tensor(0.0, device=self.device)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            last_grad_norm_sq += p.grad.data.float().norm(2) ** 2

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    n_updates += 1

                # Accumulate on GPU — no .item() sync!
                epoch_loss_tensor += loss.detach() * grad_accum_steps
                n_batches += 1
                self.global_step += 1

                # Log progress every 100 steps (console + metrics file)
                # Single .item() call per 100 steps instead of every step
                if self.global_step % 100 == 0:
                    now = time.time()
                    avg_loss = (epoch_loss_tensor / n_batches).item()
                    ppl = math.exp(min(avg_loss, 20))
                    interval_time = now - last_log_time
                    tokens_interval = 100 * batch_size * (context_length - 1)
                    tokens_per_sec = tokens_interval / max(interval_time, 1)
                    last_log_time = now
                    logger.info(
                        f"Step {self.global_step} | Epoch {epoch+1}/{max_epochs} | "
                        f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | "
                        f"Time: {interval_time:.0f}s | {tokens_per_sec:.0f} tok/s"
                    )
                    self._log_metric({
                        "type": "step",
                        "epoch": epoch + 1,
                        "step": self.global_step,
                        "train_loss": avg_loss,
                        "lr": lr,
                    })

            # End of epoch — single .item() call to get epoch loss from GPU
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = (epoch_loss_tensor / max(n_batches, 1)).item()
            del epoch_loss_tensor  # Free GPU memory
            train_ppl = math.exp(min(avg_epoch_loss, 20))

            # Grad norm from the last weight update (captured before zero_grad).
            # Single .item() call here.
            epoch_grad_norm = last_grad_norm_sq.sqrt().item()
            del last_grad_norm_sq

            # Validation
            val_loss = self.evaluate(val_loader)
            val_ppl = math.exp(min(val_loss, 20))

            self.train_losses.append({"epoch": epoch + 1, "loss": avg_epoch_loss})
            self.val_losses.append({"epoch": epoch + 1, "loss": val_loss})

            # Memory stats
            mem = get_memory_stats()

            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Train Loss: {avg_epoch_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                f"Time: {epoch_time:.1f}s | Mem: {mem['allocated_gb']:.1f}GB"
            )

            # Best model check
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

            # Log epoch metrics
            self._log_metric({
                "type": "epoch",
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "lr": lr,
                "grad_norm": round(epoch_grad_norm, 4),
                "is_best": is_best,
                "patience": self.patience_counter,
                "epoch_time": epoch_time,
                "memory_gb": mem["allocated_gb"],
            })

            # Per-head weight statistics
            self._compute_weight_stats(epoch + 1)

            # Check for LR override — user can `echo "0.0003" > v1/LR`
            self._check_lr_override()

            # Save resume checkpoint (every epoch)
            self._save_resume_checkpoint(epoch + 1)

            # Graceful stop signal — user can `echo "STOP" > v1/STOP`
            if self._check_stop_signal():
                logger.info(f"STOP signal detected at epoch {epoch+1} — saving and exiting")
                break

            # Early stopping
            if patience > 0 and self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

            gc.collect()
            mps_empty_cache()

        # Save final checkpoint + training log
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
        """Save model checkpoint."""
        ckpt_dir = self.save_dir / name
        self.model.save_pretrained(str(ckpt_dir))
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _save_training_log(self):
        """Save training loss history."""
        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        log_path = self.save_dir / "training_log.json"
        log_path.write_text(json.dumps(log, indent=2))
