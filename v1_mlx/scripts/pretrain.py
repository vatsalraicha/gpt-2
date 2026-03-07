"""Pretrain GPT-2 v1 using MLX backend.

Usage:
    python -m v1_mlx.scripts.pretrain
    python -m v1_mlx.scripts.pretrain --dashboard
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1_mlx.model.gpt2 import GPT2, GPT2Config
from v1_mlx.data.dataset import prepare_data
from v1_mlx.training.trainer import Trainer
from v1_mlx.utils.device import set_seed


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pretrain.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)

    return log_file


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 v1 (MLX)")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--config", type=str, default=str(ROOT / "v1_mlx" / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    log_file = setup_logging(ROOT / paths_cfg["logs"])
    logger = logging.getLogger("pretrain")
    set_seed(train_cfg["seed"])
    logger.info(f"Device: {mx.default_device()}")
    logger.info(f"Backend: MLX {mx.__version__}")

    # Dashboard
    dashboard_proc = None
    if args.dashboard:
        dashboard_port = cfg["dashboard"]["port"]
        logger.info(f"Starting dashboard on port {dashboard_port}...")
        dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "v1.scripts.dashboard",
             "--config", args.config, "--metrics", str(ROOT / paths_cfg["metrics"])],
            cwd=str(ROOT),
        )
        time.sleep(1)
        logger.info(f"Dashboard running at http://localhost:{dashboard_port}")

    try:
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
        n_params = sum(p.size for _, p in mx.utils.tree_flatten(model.parameters()))
        logger.info(f"Model: {config}")
        logger.info(f"Parameters: {n_params:,}")

        # Data (reuse v1 cache if available)
        logger.info("Preparing datasets...")
        corpus_path = ROOT / data_cfg["corpus_path"]
        tokenizer_dir = ROOT / data_cfg["tokenizer_dir"]
        cache_dir = ROOT / "v1" / "data" / "cache"  # Shared cache with v1

        train_chunks, val_chunks = prepare_data(
            corpus_path=corpus_path,
            tokenizer_dir=tokenizer_dir,
            context_length=model_cfg["context_length"],
            val_fraction=train_cfg["val_split"],
            seed=train_cfg["seed"],
            cache_dir=cache_dir,
        )
        logger.info(f"Train: {len(train_chunks)} chunks, Val: {len(val_chunks)} chunks")

        # Train
        metrics_path = ROOT / paths_cfg["metrics"]
        trainer = Trainer(
            model=model,
            train_chunks=train_chunks,
            val_chunks=val_chunks,
            config=train_cfg,
            save_dir=ROOT / paths_cfg["checkpoints"] / "pretrained",
            metrics_path=metrics_path,
        )

        logger.info("Starting pretraining (MLX)...")
        stats = trainer.train()

        logger.info("=" * 60)
        logger.info("PRETRAINING COMPLETE (MLX)")
        logger.info(f"  Epochs: {stats['total_epochs']}")
        logger.info(f"  Best val loss: {stats['best_val_loss']:.4f}")
        logger.info(f"  Best val PPL: {stats['best_val_ppl']:.2f}")
        logger.info(f"  Time: {stats['total_time_seconds']:.0f}s ({stats['total_time_seconds']/3600:.1f}h)")
        logger.info("=" * 60)

    finally:
        if dashboard_proc:
            dashboard_proc.terminate()
            logger.info("Dashboard stopped.")


if __name__ == "__main__":
    main()
