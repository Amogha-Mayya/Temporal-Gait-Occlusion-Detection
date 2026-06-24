"""
models/model.py
---------------
Top-level model: assembles CNN Backbone, Temporal Transformer, Detection Head,
and Severity Head into a single nn.Module.

Forward pass data-flow:
    Input: (B, T, 1, H, W)
        ↓  reshape to (B*T, 1, H, W)
        ↓  CNNBackbone  → (B*T, D_cnn)
        ↓  reshape to   (B, T, D_cnn)
        ↓  TemporalTransformer → (B, T, D_hidden)
        ↓  DetectionHead       → (B, T)  logits
        ↓  SeverityHead        → (B, T)  [0,1]

Loss computation (separate method for clarity):
    det_loss = BCEWithLogitsLoss(det_logits, det_labels.float())
    sev_loss = SmoothL1Loss(sev_preds, sev_labels)
    total    = λ_det * det_loss + λ_sev * sev_loss
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import CNNBackbone, _BACKBONE_OUTPUT_DIM
from models.transformer import TemporalTransformer
from models.heads import DetectionHead, SeverityHead


class OcclusionDetectionModel(nn.Module):
    """
    Temporal gait occlusion detection model.

    Args:
        backbone_name:   'resnet18' | 'resnet34' | 'resnet50'
        pretrained:      Load ImageNet weights for backbone.
        transformer_cfg: Dict with keys: hidden_dim, num_layers, num_heads,
                         dropout, dim_feedforward.
        head_hidden:     Hidden dimension for MLP heads.
        lambda_det:      Detection loss weight.
        lambda_sev:      Severity loss weight.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        transformer_cfg: dict = None,
        head_hidden: int = 128,
        lambda_det: float = 1.0,
        lambda_sev: float = 1.0,
    ):
        super().__init__()

        t_cfg = transformer_cfg or {}

        # ---- CNN Backbone --------------------------------------------------
        self.backbone = CNNBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        cnn_out_dim = _BACKBONE_OUTPUT_DIM[backbone_name]

        # ---- Temporal Transformer ------------------------------------------
        self.transformer = TemporalTransformer(
            input_dim=cnn_out_dim,
            hidden_dim=t_cfg.get("hidden_dim", 256),
            num_layers=t_cfg.get("num_layers", 4),
            num_heads=t_cfg.get("num_heads", 8),
            dropout=t_cfg.get("dropout", 0.1),
            dim_feedforward=t_cfg.get("dim_feedforward", 512),
        )
        hidden_dim = self.transformer.output_dim

        # ---- Prediction Heads ----------------------------------------------
        self.det_head = DetectionHead(hidden_dim, head_hidden)
        self.sev_head = SeverityHead(hidden_dim, head_hidden)

        # ---- Losses --------------------------------------------------------
        self.det_loss_fn = nn.BCEWithLogitsLoss()
        self.sev_loss_fn = nn.SmoothL1Loss()

        # ---- Loss weights --------------------------------------------------
        self.lambda_det = lambda_det
        self.lambda_sev = lambda_sev

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: (B, T, 1, H, W)

        Returns:
            det_logits: (B, T)   — raw detection logits
            sev_preds:  (B, T)   — severity predictions in (0, 1)
        """
        B, T, C, H, W = frames.shape

        # ---- Per-frame CNN features ----------------------------------------
        frames_flat = frames.view(B * T, C, H, W)          # (B*T, 1, H, W)
        cnn_feats   = self.backbone(frames_flat)            # (B*T, D_cnn)
        cnn_feats   = cnn_feats.view(B, T, -1)             # (B, T, D_cnn)

        # ---- Temporal context via Transformer ------------------------------
        ctx_feats = self.transformer(cnn_feats)             # (B, T, D_hidden)

        # ---- Heads ---------------------------------------------------------
        det_logits = self.det_head(ctx_feats)               # (B, T)
        sev_preds  = self.sev_head(ctx_feats)               # (B, T)

        return det_logits, sev_preds

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        det_logits: torch.Tensor,
        sev_preds: torch.Tensor,
        det_labels: torch.Tensor,
        sev_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined multi-task loss.

        Args:
            det_logits:  (B, T) raw logits
            sev_preds:   (B, T) sigmoid severity predictions
            det_labels:  (B, T) int64  {0, 1}
            sev_labels:  (B, T) float32 [0, 1]

        Returns:
            Dict with keys: 'total', 'det', 'sev'
        """
        det_loss = self.det_loss_fn(
            det_logits, det_labels.float()
        )
        sev_loss = self.sev_loss_fn(
            sev_preds, sev_labels
        )
        total = self.lambda_det * det_loss + self.lambda_sev * sev_loss

        return {
            "total": total,
            "det":   det_loss,
            "sev":   sev_loss,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        det_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return binary predictions + severity estimates.

        Args:
            frames:        (B, T, 1, H, W) or (T, 1, H, W) for single clip.
            det_threshold: Sigmoid threshold for binary detection.

        Returns:
            dict with:
                'det_prob'   : (B, T)  — detection probability
                'det_binary' : (B, T)  — 0/1 predicted label
                'sev'        : (B, T)  — severity estimate
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)   # add batch dim

        det_logits, sev_preds = self.forward(frames)
        det_prob   = torch.sigmoid(det_logits)
        det_binary = (det_prob >= det_threshold).long()

        return {
            "det_prob":   det_prob,
            "det_binary": det_binary,
            "sev":        sev_preds,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> OcclusionDetectionModel:
    """Build the model from the full config dict."""
    m_cfg = cfg["model"]
    t_cfg = m_cfg["transformer"]
    tr_cfg = cfg["training"]

    return OcclusionDetectionModel(
        backbone_name=m_cfg["backbone"],
        pretrained=True,
        transformer_cfg={
            "hidden_dim":      t_cfg["hidden_dim"],
            "num_layers":      t_cfg["num_layers"],
            "num_heads":       t_cfg["num_heads"],
            "dropout":         t_cfg["dropout"],
            "dim_feedforward": t_cfg["dim_feedforward"],
        },
        head_hidden=m_cfg["heads"]["hidden_dim"],
        lambda_det=tr_cfg["lambda_det"],
        lambda_sev=tr_cfg["lambda_sev"],
    )
