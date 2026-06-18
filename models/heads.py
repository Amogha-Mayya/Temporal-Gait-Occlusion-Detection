"""
models/heads.py
---------------
Prediction heads for the two output tasks:

1. DetectionHead  — per-frame binary occlusion detection (BCEWithLogitsLoss)
2. SeverityHead   — per-frame occlusion severity regression (SmoothL1Loss)

Both heads share the same MLP structure:

    hidden_dim → 128 → 1

The output is a single scalar per frame:
  - Detection: raw logit  (apply sigmoid for probability, threshold at 0.5)
  - Severity:  raw value  (apply sigmoid to bound output to [0, 1])

Design note on sigmoid for severity:
    We apply sigmoid in the forward pass of SeverityHead during *inference*
    but NOT during training because SmoothL1Loss expects unbounded targets.
    Actually, since target severities are in [0, 1], we use sigmoid here to
    match the target range. This is a valid design: sigmoid squashes output
    to (0,1) matching the label range, and SmoothL1 operates on the
    sigmoid-transformed output vs. the [0,1] target.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Binary occlusion classifier operating frame-by-frame.

    Input:  (B, T, hidden_dim)
    Output: (B, T)  raw logits (not probabilities)

    Loss: BCEWithLogitsLoss (numerically stable sigmoid + BCE combined)
    """

    def __init__(self, hidden_dim: int, head_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim)

        Returns:
            logits: (B, T)
        """
        return self.mlp(x).squeeze(-1)   # (B, T)


class SeverityHead(nn.Module):
    """
    Occlusion severity regressor operating frame-by-frame.

    Input:  (B, T, hidden_dim)
    Output: (B, T)  values in (0, 1) via sigmoid

    Loss: SmoothL1Loss (Huber loss) — robust to outlier severity values
    """

    def __init__(self, hidden_dim: int, head_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim)

        Returns:
            severity: (B, T) in (0, 1)
        """
        raw = self.mlp(x).squeeze(-1)    # (B, T) — unbounded
        return torch.sigmoid(raw)        # map to (0, 1)
