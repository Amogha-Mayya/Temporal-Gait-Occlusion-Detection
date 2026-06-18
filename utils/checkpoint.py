"""
utils/checkpoint.py
-------------------
Save and load model checkpoints with full training state for resumable runs.

Saved state dict contains:
    - model state dict
    - optimizer state dict
    - scheduler state dict
    - scaler state dict (AMP)
    - current epoch
    - best validation metric
    - config snapshot
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    cfg: dict,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        path:        File path to write the .pth file.
        model:       The model (nn.Module) to save.
        optimizer:   Current optimizer.
        scheduler:   Current LR scheduler.
        scaler:      AMP GradScaler.
        epoch:       Current epoch (0-based).
        best_metric: Best validation metric seen so far.
        cfg:         Config dict (saved for reference).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":        epoch,
        "best_metric":  best_metric,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler is not None else None,
        "scaler":       scaler.state_dict()    if scaler    is not None else None,
        "cfg":          cfg,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        path:      Path to the .pth checkpoint file.
        model:     Model to load weights into.
        optimizer: If given, restore optimizer state.
        scheduler: If given, restore scheduler state.
        scaler:    If given, restore AMP scaler state.
        device:    Device to map tensors to.

    Returns:
        Dict with keys 'epoch', 'best_metric', 'cfg'.
    """
    state = torch.load(path, map_location=device)

    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    return {
        "epoch":       state.get("epoch", 0),
        "best_metric": state.get("best_metric", 0.0),
        "cfg":         state.get("cfg", {}),
    }
