"""
utils/seed.py
-------------
Sets all random seeds for full reproducibility across Python, NumPy,
PyTorch (CPU and CUDA).

Note: CUDA fully-deterministic mode (torch.use_deterministic_algorithms(True))
can significantly slow down training on some hardware; it is left as an opt-in
flag below.  For M.Tech / research reproducibility purposes, setting the seeds
as shown here is sufficient for the vast majority of experiments.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed:          The integer seed value.
        deterministic: If True, enable CUDA deterministic mode.
                       This may hurt performance but guarantees bit-exact
                       reproducibility across runs on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # multi-GPU

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        # benchmark=True finds the fastest conv algorithm per input shape.
        # This is fine when input shapes are constant (fixed sequence length
        # and image size), which is the case here.
        torch.backends.cudnn.benchmark = True
