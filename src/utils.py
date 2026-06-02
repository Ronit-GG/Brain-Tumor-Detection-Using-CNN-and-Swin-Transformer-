"""
General-purpose utilities: random seeding, device selection, plotting
helpers, Grad-CAM, attention rollout.

Implemented progressively from STEP 6 onward.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Make a run reproducible.

    Sets seeds for Python's `random`, NumPy, and (if installed) PyTorch
    on both CPU and CUDA. Optionally configures cuDNN for deterministic
    operation (slower but exactly repeatable).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
        else:
            torch.backends.cudnn.benchmark     = True
    except ImportError:
        pass


def get_device(prefer_cuda: bool = True) -> "torch.device":           # noqa: F821
    """Return `cuda` if available and requested, else `cpu`."""
    import torch
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model, trainable_only: bool = True) -> int:
    """Return the number of (trainable) parameters in a torch.nn.Module."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def human_format(num: int) -> str:
    """Format a large integer like '5.2M' or '1.3K'."""
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}".rstrip("0").rstrip(".")
        num /= 1000
    return f"{num:.1f}T"
