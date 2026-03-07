"""Device detection, seeding, and memory utilities for Apple Silicon."""

import random
import torch
import numpy as np


def get_device(force_cpu: bool = False) -> torch.device:
    """Get the best available device (MPS > CPU)."""
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def mps_empty_cache():
    """Clear MPS memory cache if available."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def mps_synchronize():
    """Synchronize MPS operations if available."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_memory_stats() -> dict:
    """Get current memory usage stats."""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1e9
        return {"allocated_gb": round(allocated, 2), "device": "mps"}
    return {"allocated_gb": 0.0, "device": "cpu"}
