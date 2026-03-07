"""Device and seed utilities for MLX.

MLX runs on Apple GPU by default — no device management needed.
"""

import mlx.core as mx


def set_seed(seed: int = 42):
    """Set MLX random seed for reproducibility."""
    mx.random.seed(seed)


def get_memory_stats() -> dict:
    """Get current memory usage stats.

    Note: MLX doesn't expose memory tracking like PyTorch MPS.
    We report what's available.
    """
    try:
        info = mx.metal.get_active_memory()
        return {"allocated_gb": round(info / 1e9, 2), "device": "gpu"}
    except Exception:
        return {"allocated_gb": 0.0, "device": "gpu"}
