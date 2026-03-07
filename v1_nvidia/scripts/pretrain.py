"""Pretrain GPT-2 v1 on the curated corpus.

Usage:
    python -m v1_nvidia.scripts.pretrain
    python -m v1_nvidia.scripts.pretrain --resume       # Resume from last checkpoint
    python -m v1_nvidia.scripts.pretrain --force-cpu
    python -m v1_nvidia.scripts.pretrain --dashboard    # Also start the dashboard
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml
import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1_nvidia.model.gpt2 import GPT2, GPT2Config
from v1_nvidia.data.dataset import prepare_datasets
from v1_nvidia.training.trainer import Trainer
from v1_nvidia.utils.device import get_device, set_seed


def setup_logging(log_dir: Path, resume: bool = False):
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pretrain.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler — append when resuming, overwrite on fresh start
    fh = logging.FileHandler(log_file, mode="a" if resume else "w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)

    return log_file


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 v1_nvidia")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (model + optimizer + training state)")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard alongside training")
    parser.add_argument("--config", type=str, default=str(ROOT / "v1_nvidia" / "config.yaml"),
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    # Setup
    log_file = setup_logging(ROOT / paths_cfg["logs"], resume=args.resume)
    logger = logging.getLogger("pretrain")
    set_seed(train_cfg["seed"])
    device = get_device(force_cpu=args.force_cpu)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # Start dashboard in background if requested
    dashboard_proc = None
    if args.dashboard:
        logger.info("Starting dashboard on port %d...", cfg["dashboard"]["port"])
        dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "v1_nvidia.scripts.dashboard", "--config", args.config],
            cwd=str(ROOT),
        )
        time.sleep(1)
        logger.info(f"Dashboard running at http://localhost:{cfg['dashboard']['port']}")

    try:
        # Model
        config = GPT2Config(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            n_heads=model_cfg["n_heads"],
            d_ff=model_cfg["d_ff"],
            context_length=model_cfg["context_length"],
            dropout=model_cfg["dropout"],
            bias=model_cfg["bias"],
        )
        model = GPT2(config)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {config}")
        logger.info(f"Parameters: {n_params:,}")

        # Data
        logger.info("Preparing datasets...")
        corpus_path = ROOT / data_cfg["corpus_path"]
        tokenizer_dir = ROOT / data_cfg["tokenizer_dir"]
        cache_dir = ROOT / "v1_nvidia" / "data" / "cache"

        train_dataset, val_dataset = prepare_datasets(
            corpus_path=corpus_path,
            tokenizer_dir=tokenizer_dir,
            context_length=model_cfg["context_length"],
            val_fraction=train_cfg["val_split"],
            seed=train_cfg["seed"],
            cache_dir=cache_dir,
        )
        logger.info(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")

        # Train
        metrics_path = ROOT / paths_cfg["metrics"]
        stop_file = ROOT / "v1_nvidia" / "STOP"
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=train_cfg,
            save_dir=ROOT / paths_cfg["checkpoints"] / "pretrained",
            device=device,
            metrics_path=metrics_path,
            stop_file=stop_file,
            resume=args.resume,
        )

        if args.resume:
            logger.info("Resuming pretraining...")
        else:
            logger.info("Starting pretraining...")
        logger.info(f'  Graceful stop: echo "STOP" > {stop_file}')
        stats = trainer.train()

        logger.info("=" * 60)
        logger.info("PRETRAINING COMPLETE")
        logger.info(f"  Epochs: {stats['total_epochs']}")
        logger.info(f"  Best val loss: {stats['best_val_loss']:.4f}")
        logger.info(f"  Best val PPL: {stats['best_val_ppl']:.2f}")
        logger.info(f"  Time: {stats['total_time_seconds']:.0f}s ({stats['total_time_seconds']/3600:.1f}h)")
        logger.info(f"  Checkpoint: {ROOT / paths_cfg['checkpoints'] / 'pretrained' / 'best'}")
        logger.info(f"  Log: {log_file}")
        logger.info("=" * 60)

    finally:
        if dashboard_proc:
            dashboard_proc.terminate()
            logger.info("Dashboard stopped.")


if __name__ == "__main__":
    main()
