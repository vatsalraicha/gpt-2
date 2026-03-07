# Training Runtime Commands

All commands are run from the project root (`/Users/vr/Code/LLMs/GPT-2`).

## Start / Resume Training

```bash
# Start fresh
python -m v1_nvidia.scripts.pretrain

# Resume from last checkpoint
python -m v1_nvidia.scripts.pretrain --resume
```

## Graceful Stop (after current epoch)

```bash
# Stop after current epoch finishes
echo "STOP" > v1_nvidia/STOP

# Clear the signal before next run
echo "" > v1_nvidia/STOP
```

Training finishes the current epoch, saves checkpoint, then exits.
File is **not** deleted by the trainer. Clear it with `echo "" > v1_nvidia/STOP` before restarting.

## Learning Rate Override

```bash
# Set LR to 1e-3 (takes effect at end of current epoch, persists every epoch)
echo "0.001" > v1_nvidia/LR

# Remove override (reverts to config.yaml / checkpoint LR)
rm v1_nvidia/LR
```

File is **not** deleted by the trainer. As long as it exists with a value, it overrides every epoch.

## Dashboard

```bash
# Start unified dashboard (auto-discovers all versions/backends)
python dashboard.py

# Custom port
python dashboard.py --port 5001
```

Dashboard URL: http://localhost:5000 (or whatever port is specified)

## Model Diagnostics

```bash
# Pretrain diagnostics (default)
python -m v1_nvidia.scripts.diagnose --stage pretrain

# Finetune diagnostics
python -m v1_nvidia.scripts.diagnose --stage finetune

# Custom checkpoint
python -m v1_nvidia.scripts.diagnose --checkpoint v1_nvidia/checkpoints/pretrained/best

# Skip matplotlib plots (JSON only)
python -m v1_nvidia.scripts.diagnose --no-plots

# Adjust sample count and passage length
python -m v1_nvidia.scripts.diagnose --n-samples 20 --passage-length 1024
```

Results saved to:
- JSON: `v1_nvidia/logs/{stage}/diagnostics.json`
- Plots: `v1_nvidia/plots/diagnostics/{stage}/*.png`
- Dashboard: http://localhost:5000/diagnostics?stage=pretrain
- Dashboard: http://localhost:5000/diagnostics?stage=finetune

## Fine-tuning

```bash
# Start fine-tuning (loads pretrained/best checkpoint)
python -m v1_nvidia.scripts.finetune

# Resume from last finetune checkpoint
python -m v1_nvidia.scripts.finetune --resume

# Custom hyperparameters
python -m v1_nvidia.scripts.finetune --lr 1e-4 --epochs 50 --batch-size 4 --patience 5

# Custom checkpoint or data
python -m v1_nvidia.scripts.finetune --checkpoint v1_nvidia/checkpoints/pretrained/best --qa-data v1_nvidia/data/finetune/qa_train.jsonl

# Launch dashboard alongside training
python -m v1_nvidia.scripts.finetune --dashboard
```

Finetuning output:
- Checkpoint: `v1_nvidia/checkpoints/finetuned/best/`
- Metrics: `v1_nvidia/logs/finetune/metrics.jsonl`
- Weight stats: `v1_nvidia/logs/finetune/weight_stats.jsonl`
- Log: `v1_nvidia/logs/finetune/finetune.log`
- Dashboard: http://localhost:5000/?stage=finetune

Graceful stop and LR override work the same as pretraining (see above).

## Text Generation

```bash
# Interactive REPL
python -m v1_nvidia.scripts.generate

# Single prompt
python -m v1_nvidia.scripts.generate --prompt "The gradient of"

# Adjust parameters
python -m v1_nvidia.scripts.generate --temperature 0.5 --top-k 30 --max-tokens 200
```

## Kill Training (immediate, no checkpoint save)

```bash
# Find PID
ps aux | grep pretrain | grep -v grep

# Kill
kill <PID>
```

Use `echo "STOP" > v1_nvidia/STOP` instead when possible — it saves a checkpoint before exiting.
