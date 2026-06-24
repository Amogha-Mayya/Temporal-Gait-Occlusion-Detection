"""
models/heads.py
---------------
Prediction heads for the four output tasks:

1. DetectionHead   — per-frame binary occlusion detection (BCEWithLogitsLoss)
2. SeverityHead    — per-frame occlusion severity regression (SmoothL1Loss)
3. RegionHead      — per-frame quadrant-wise occlusion (upper/lower/left/right)
4. ConfidenceHead  — per-frame self-confidence estimation (SmoothL1Loss)

All heads share the same MLP structure:

    hidden_dim → 128 → (1 or 4)

Output semantics:
  - Detection:   raw logit  (apply sigmoid for probability, threshold at 0.5)
  - Severity:    sigmoid-bounded value in (0, 1)
  - Region:      4 raw logits, one per quadrant, NOT mutually exclusive
                 (apply sigmoid independently per region, threshold at 0.5)
  - Confidence:  sigmoid-bounded value in (0, 1)

Design note on sigmoid for severity:
    We apply sigmoid in the forward pass of SeverityHead during *inference*
    but NOT during training because SmoothL1Loss expects unbounded targets.
    Actually, since target severities are in [0, 1], we use sigmoid here to
    match the target range. This is a valid design: sigmoid squashes output
    to (0,1) matching the label range, and SmoothL1 operates on the
    sigmoid-transformed output vs. the [0,1] target.

Design note on RegionHead and ConfidenceHead (added post-training):
    These two heads are NEW additions on top of an already-trained
    backbone + transformer + detection/severity heads. They are randomly
    initialised and must be fine-tuned (head-only, frozen backbone) before
    their output is meaningful — see finetune_new_heads.py.
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


class RegionHead(nn.Module):
    """
    Region-wise (quadrant) occlusion classifier, operating frame-by-frame.

    Predicts FOUR independent binary outputs per frame:
        [upper, lower, left, right]
    Each is 1 if that half-region of the silhouette is occluded.
    Independent sigmoids are used (not softmax) because regions are NOT
    mutually exclusive — e.g. an occlusion in the upper-left corner is
    simultaneously 'upper' AND 'left'.

    Input:  (B, T, hidden_dim)
    Output: (B, T, 4)  raw logits (apply sigmoid for probabilities)

    Loss: BCEWithLogitsLoss (multi-label, one logit per region)
    """

    def __init__(self, hidden_dim: int, head_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 4),   # [upper, lower, left, right]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim)

        Returns:
            logits: (B, T, 4)
        """
        return self.mlp(x)   # (B, T, 4)


class ConfidenceHead(nn.Module):
    """
    Per-frame confidence estimation head.

    Predicts a scalar confidence score in (0, 1) representing how reliable
    the model considers its own detection/severity prediction for that
    frame to be. Trained as a calibration target (see finetune_new_heads.py
    for how the target is constructed from prediction-consistency signals),
    NOT as a free-floating uncertainty estimate with no grounding.

    Input:  (B, T, hidden_dim)
    Output: (B, T)  in (0, 1) via sigmoid

    Loss: SmoothL1Loss against the constructed confidence target.
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
            confidence: (B, T) in (0, 1)
        """
        raw = self.mlp(x).squeeze(-1)
        return torch.sigmoid(raw)
