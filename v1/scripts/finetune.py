"""Fine-tune the pretrained GPT-2 model on Q&A data.

Loads the best pretrained checkpoint, trains on qa_train.jsonl,
and saves the finetuned model to v1/checkpoints/finetuned/.

Usage:
    python -m v1.scripts.finetune
    python -m v1.scripts.finetune --resume
    python -m v1.scripts.finetune --lr 1e-4 --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from bpe.tokenizer import Tokenizer
from v1.data.finetune_dataset import load_qa_data
from v1.model.gpt2 import GPT2
from v1.training.trainer import Trainer
from v1.utils.device import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Q&A data")
    parser.add_argument("--config", type=str, default=str(ROOT / "v1" / "config.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume from finetune checkpoint")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--min-lr", type=float, default=None, help="Min LR (default: lr/10)")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--max-length", type=int, default=256, help="Max token sequence length (default: 256)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (default: 8)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained checkpoint to load (default: v1/checkpoints/pretrained/best)")
    parser.add_argument("--qa-data", type=str, default=None,
                        help="Path to qa_train.jsonl (default: v1/data/finetune/qa_train.jsonl)")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard subprocess")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Paths
    pretrained_dir = Path(args.checkpoint) if args.checkpoint else ROOT / "v1" / "checkpoints" / "pretrained" / "best"
    qa_data_path = Path(args.qa_data) if args.qa_data else ROOT / "v1" / "data" / "finetune" / "qa_train.jsonl"
    save_dir = ROOT / "v1" / "checkpoints" / "finetuned"
    metrics_path = ROOT / "v1" / "logs" / "finetune" / "metrics.jsonl"
    log_path = ROOT / "v1" / "logs" / "finetune" / "finetune.log"
    tokenizer_dir = ROOT / config["data"]["tokenizer_dir"]

    # Ensure directories exist
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Logging
    log_mode = "a" if args.resume else "w"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), mode=log_mode),
        ],
    )
    logger = logging.getLogger("finetune")

    # Device
    device = get_device(force_cpu=args.force_cpu)
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Device: {device}, seed: {seed}")

    # Load pretrained model
    logger.info(f"Loading pretrained model from {pretrained_dir}")
    model = GPT2.from_pretrained(str(pretrained_dir), device=device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")

    # Load tokenizer
    tokenizer = Tokenizer.from_files(str(tokenizer_dir))
    eos_id = config["special_tokens"]["eos_id"]
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load and prepare Q&A data
    logger.info(f"Loading Q&A data from {qa_data_path}")
    train_dataset, val_dataset = load_qa_data(
        jsonl_path=qa_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_fraction=0.1,
        seed=seed,
        pad_id=eos_id,
    )
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info(f"Sequence length: {args.max_length} tokens")

    # Build training config with finetune-specific hyperparameters
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    train_config = {
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "min_lr": min_lr,
        "warmup_fraction": 0.1,     # 10% warmup (more warmup for small dataset)
        "weight_decay": config["training"].get("weight_decay", 0.01),
        "grad_clip": config["training"].get("grad_clip", 1.0),
        "betas": config["training"].get("betas", [0.9, 0.95]),
        "patience": args.patience,
        "min_delta": 0.001,
        "seed": seed,
    }

    logger.info(f"Finetune config: lr={train_config['learning_rate']}, "
                f"min_lr={train_config['min_lr']}, epochs={train_config['max_epochs']}, "
                f"batch={train_config['batch_size']}, accum={train_config['gradient_accumulation_steps']}, "
                f"patience={train_config['patience']}")

    # Optionally launch dashboard
    if args.dashboard:
        import subprocess
        dashboard_cmd = [
            sys.executable, "-m", "v1.scripts.dashboard",
            "--config", args.config,
            "--metrics", str(metrics_path),
            "--port", str(config["dashboard"].get("port", 5000)),
        ]
        subprocess.Popen(dashboard_cmd, cwd=str(ROOT))
        logger.info("Dashboard launched")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        save_dir=save_dir,
        device=device,
        metrics_path=metrics_path,
        stop_file=ROOT / "v1" / "STOP",
        resume=args.resume,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Final report
    logger.info("Fine-tuning complete!")
    logger.info(f"Best checkpoint: {save_dir / 'best'}")
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"Weight stats: {metrics_path.parent / 'weight_stats.jsonl'}")
    logger.info(f"Log: {log_path}")
    logger.info("")
    logger.info("Run diagnostics:")
    logger.info(f"  python -m v1.scripts.diagnose --stage finetune")
    logger.info("View in dashboard:")
    logger.info(f"  http://localhost:{config['dashboard'].get('port', 5000)}/?stage=finetune")


if __name__ == "__main__":
    main()
