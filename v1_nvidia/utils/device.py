"""Device detection, seeding, and memory utilities for NVIDIA CUDA GPUs."""

import random
import torch
import numpy as np


def get_device(force_cpu: bool = False) -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_empty_cache():
    """Clear CUDA memory cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_synchronize():
    """Synchronize CUDA operations if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_memory_stats() -> dict:
    """Get current memory usage stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        return {"allocated_gb": round(allocated, 2), "device": "cuda"}
    return {"allocated_gb": 0.0, "device": "cpu"}
