"""
models/model.py
---------------
Top-level model: assembles CNN Backbone, Temporal Transformer, and up to
FOUR prediction heads into a single nn.Module.

Forward pass data-flow:
    Input: (B, T, 1, H, W)
        ↓  reshape to (B*T, 1, H, W)
        ↓  CNNBackbone  → (B*T, D_cnn)
        ↓  reshape to   (B, T, D_cnn)
        ↓  TemporalTransformer → (B, T, D_hidden)
        ├─ DetectionHead   → (B, T)     logits
        ├─ SeverityHead    → (B, T)     [0,1]
        ├─ RegionHead      → (B, T, 4)  logits  [upper, lower, left, right]
        └─ ConfidenceHead  → (B, T)     [0,1]

BACKWARD COMPATIBILITY
-----------------------
This model started with only Detection + Severity heads (trained for 60
epochs, checkpoint saved as best_model.pth). RegionHead and ConfidenceHead
were added LATER, after that training run completed.

To avoid breaking the existing checkpoint / pipeline:
  - `enable_extra_heads=False` (default) builds the ORIGINAL 2-head model.
    `load_state_dict` on an old checkpoint works unchanged.
  - `enable_extra_heads=True` additionally builds RegionHead and
    ConfidenceHead. Loading an old checkpoint with `strict=False` will
    populate backbone/transformer/det_head/sev_head from the checkpoint
    and leave region_head/confidence_head at their (random) initialisation
    — exactly the state expected by finetune_new_heads.py, which then
    trains ONLY those two new heads with everything else frozen.
  - `forward()` is unchanged (returns the original 2-tuple) so existing
    train.py / validate.py / test.py / trainer.py / evaluator.py keep
    working with zero modification.
  - `forward_all()` is the NEW entry point that returns all four head
    outputs; used by finetune_new_heads.py and the new inference script.

Loss computation (separate methods for clarity):
    det_loss   = BCEWithLogitsLoss(det_logits, det_labels.float())
    sev_loss   = SmoothL1Loss(sev_preds, sev_labels)
    region_loss = BCEWithLogitsLoss(region_logits, region_labels.float())
    conf_loss  = SmoothL1Loss(conf_preds, conf_targets)
    total      = λ_det*det_loss + λ_sev*sev_loss + λ_region*region_loss + λ_conf*conf_loss
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import CNNBackbone, _BACKBONE_OUTPUT_DIM
from models.transformer import TemporalTransformer
from models.heads import DetectionHead, SeverityHead, RegionHead, ConfidenceHead


class OcclusionDetectionModel(nn.Module):
    """
    Temporal gait occlusion detection model.

    Args:
        backbone_name:      'resnet18' | 'resnet34' | 'resnet50'
        pretrained:         Load ImageNet weights for backbone.
        transformer_cfg:    Dict with keys: hidden_dim, num_layers, num_heads,
                            dropout, dim_feedforward.
        head_hidden:        Hidden dimension for MLP heads.
        lambda_det:         Detection loss weight.
        lambda_sev:         Severity loss weight.
        enable_extra_heads: If True, also build RegionHead and ConfidenceHead.
                            Default False preserves the original 2-head
                            architecture exactly (for loading old checkpoints
                            without any extra unused parameters).
        lambda_region:      Region loss weight (only used if extra heads enabled).
        lambda_conf:        Confidence loss weight (only used if extra heads enabled).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        transformer_cfg: dict = None,
        head_hidden: int = 128,
        lambda_det: float = 1.0,
        lambda_sev: float = 1.0,
        enable_extra_heads: bool = False,
        lambda_region: float = 1.0,
        lambda_conf: float = 1.0,
    ):
        super().__init__()

        t_cfg = transformer_cfg or {}
        self.enable_extra_heads = enable_extra_heads

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

        # ---- Original Prediction Heads --------------------------------------
        self.det_head = DetectionHead(hidden_dim, head_hidden)
        self.sev_head = SeverityHead(hidden_dim, head_hidden)

        # ---- Original Losses -------------------------------------------------
        self.det_loss_fn = nn.BCEWithLogitsLoss()
        self.sev_loss_fn = nn.SmoothL1Loss()

        # ---- Original Loss weights ---------------------------------------
        self.lambda_det = lambda_det
        self.lambda_sev = lambda_sev

        # ---- NEW Prediction Heads (only built if requested) ------------------
        if self.enable_extra_heads:
            self.region_head     = RegionHead(hidden_dim, head_hidden)
            self.confidence_head = ConfidenceHead(hidden_dim, head_hidden)

            self.region_loss_fn = nn.BCEWithLogitsLoss()
            self.conf_loss_fn   = nn.SmoothL1Loss()

            self.lambda_region = lambda_region
            self.lambda_conf   = lambda_conf

    # ------------------------------------------------------------------
    # Shared encoder (CNN + Transformer) — used by both forward paths
    # ------------------------------------------------------------------

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Run the CNN backbone + Temporal Transformer to obtain contextualised
        per-frame features. Shared by forward() and forward_all() so the
        feature extraction code is never duplicated.

        Args:
            frames: (B, T, 1, H, W)

        Returns:
            ctx_feats: (B, T, hidden_dim)
        """
        B, T, C, H, W = frames.shape

        # ---- Per-frame CNN features ----------------------------------------
        frames_flat = frames.view(B * T, C, H, W)          # (B*T, 1, H, W)
        cnn_feats   = self.backbone(frames_flat)            # (B*T, D_cnn)
        cnn_feats   = cnn_feats.view(B, T, -1)              # (B, T, D_cnn)

        # ---- Temporal context via Transformer ------------------------------
        ctx_feats = self.transformer(cnn_feats)             # (B, T, D_hidden)
        return ctx_feats

    # ------------------------------------------------------------------
    # Forward (ORIGINAL 2-head interface — unchanged, fully backward compatible)
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
        ctx_feats = self.encode(frames)                     # (B, T, D_hidden)

        # ---- Heads ---------------------------------------------------------
        det_logits = self.det_head(ctx_feats)               # (B, T)
        sev_preds  = self.sev_head(ctx_feats)               # (B, T)

        return det_logits, sev_preds

    # ------------------------------------------------------------------
    # Forward ALL (NEW 4-head interface — used after enable_extra_heads=True)
    # ------------------------------------------------------------------

    def forward_all(
        self,
        frames: torch.Tensor,
        ctx_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run all four heads. Requires enable_extra_heads=True at construction.

        Args:
            frames:    (B, T, 1, H, W). Ignored if ctx_feats is provided.
            ctx_feats: Optionally pass pre-computed transformer features
                       (B, T, hidden_dim) directly, skipping CNN+Transformer
                       recomputation — useful in finetune_new_heads.py where
                       backbone/transformer are frozen and features can be
                       cached per batch.

        Returns:
            dict with:
                'det_logits'    : (B, T)
                'sev_preds'     : (B, T)
                'region_logits' : (B, T, 4)
                'conf_preds'    : (B, T)
                'ctx_feats'     : (B, T, hidden_dim)  — returned for reuse
        """
        if not self.enable_extra_heads:
            raise RuntimeError(
                "forward_all() requires the model to be built with "
                "enable_extra_heads=True."
            )

        if ctx_feats is None:
            ctx_feats = self.encode(frames)

        det_logits    = self.det_head(ctx_feats)
        sev_preds     = self.sev_head(ctx_feats)
        region_logits = self.region_head(ctx_feats)
        conf_preds    = self.confidence_head(ctx_feats)

        return {
            "det_logits":    det_logits,
            "sev_preds":     sev_preds,
            "region_logits": region_logits,
            "conf_preds":    conf_preds,
            "ctx_feats":     ctx_feats,
        }

    # ------------------------------------------------------------------
    # Freezing helper — used by finetune_new_heads.py
    # ------------------------------------------------------------------

    def freeze_pretrained_components(self) -> None:
        """
        Freeze backbone, transformer, and the ORIGINAL detection/severity
        heads so that only region_head and confidence_head receive
        gradients during head-only fine-tuning. Also switches frozen
        submodules to eval() mode so BatchNorm/Dropout statistics in the
        backbone are not perturbed.
        """
        for module in [self.backbone, self.transformer, self.det_head, self.sev_head]:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

        if self.enable_extra_heads:
            for p in self.region_head.parameters():
                p.requires_grad = True
            for p in self.confidence_head.parameters():
                p.requires_grad = True
            self.region_head.train()
            self.confidence_head.train()

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

    def compute_loss_all(
        self,
        outputs: Dict[str, torch.Tensor],
        det_labels: torch.Tensor,
        sev_labels: torch.Tensor,
        region_labels: torch.Tensor,
        conf_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined 4-head loss. Requires enable_extra_heads=True.

        Args:
            outputs:       dict returned by forward_all()
            det_labels:    (B, T)    int64  {0, 1}
            sev_labels:    (B, T)    float32 [0, 1]
            region_labels: (B, T, 4) float32 {0, 1}  [upper, lower, left, right]
            conf_targets:  (B, T)    float32 [0, 1]  constructed confidence target

        Returns:
            Dict with keys: 'total', 'det', 'sev', 'region', 'conf'
        """
        if not self.enable_extra_heads:
            raise RuntimeError(
                "compute_loss_all() requires enable_extra_heads=True."
            )

        det_loss    = self.det_loss_fn(outputs["det_logits"], det_labels.float())
        sev_loss    = self.sev_loss_fn(outputs["sev_preds"], sev_labels)
        region_loss = self.region_loss_fn(outputs["region_logits"], region_labels.float())
        conf_loss   = self.conf_loss_fn(outputs["conf_preds"], conf_targets)

        total = (
            self.lambda_det * det_loss
            + self.lambda_sev * sev_loss
            + self.lambda_region * region_loss
            + self.lambda_conf * conf_loss
        )

        return {
            "total":  total,
            "det":    det_loss,
            "sev":    sev_loss,
            "region": region_loss,
            "conf":   conf_loss,
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

    @torch.no_grad()
    def predict_all(
        self,
        frames: torch.Tensor,
        det_threshold: float = 0.5,
        region_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference with all four heads. Requires enable_extra_heads=True.

        Args:
            frames:            (B, T, 1, H, W) or (T, 1, H, W) for single clip.
            det_threshold:     Sigmoid threshold for binary detection.
            region_threshold:  Sigmoid threshold for binary region labels.

        Returns:
            dict with:
                'det_prob'     : (B, T)     — detection probability
                'det_binary'   : (B, T)     — 0/1 predicted label
                'sev'          : (B, T)     — severity estimate
                'region_prob'  : (B, T, 4)  — region probabilities [u,l,le,r]
                'region_binary': (B, T, 4)  — 0/1 per region
                'confidence'   : (B, T)     — self-confidence estimate
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)   # add batch dim

        outputs = self.forward_all(frames)

        det_prob   = torch.sigmoid(outputs["det_logits"])
        det_binary = (det_prob >= det_threshold).long()

        region_prob   = torch.sigmoid(outputs["region_logits"])
        region_binary = (region_prob >= region_threshold).long()

        return {
            "det_prob":      det_prob,
            "det_binary":    det_binary,
            "sev":           outputs["sev_preds"],
            "region_prob":   region_prob,
            "region_binary": region_binary,
            "confidence":    outputs["conf_preds"],
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_model(cfg: dict, enable_extra_heads: bool = False) -> OcclusionDetectionModel:
    """
    Build the model from the full config dict.

    Args:
        cfg:                 Full config dict.
        enable_extra_heads:  If True, also construct RegionHead and
                             ConfidenceHead. Default False reproduces the
                             exact original 2-head architecture used during
                             the initial 60-epoch training run.
    """
    m_cfg = cfg["model"]
    t_cfg = m_cfg["transformer"]
    tr_cfg = cfg["training"]

    # Extra-head loss weights are optional in config; default to 1.0 each.
    extra_cfg = cfg.get("extra_heads", {})

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
        enable_extra_heads=enable_extra_heads,
        lambda_region=extra_cfg.get("lambda_region", 1.0),
        lambda_conf=extra_cfg.get("lambda_conf", 1.0),
    )


def build_model_with_pretrained_base(
    cfg: dict,
    checkpoint_path: str,
    device: str = "cpu",
) -> OcclusionDetectionModel:
    """
    Build the 4-head model and load backbone/transformer/det_head/sev_head
    weights from an EXISTING 2-head checkpoint (e.g. checkpoints/best_model.pth
    from the original 60-epoch run). region_head and confidence_head are left
    at their random initialisation, ready for head-only fine-tuning.

    This is the standard entry point used by finetune_new_heads.py and by
    any inference script that wants all four heads available.

    Args:
        cfg:             Full config dict.
        checkpoint_path: Path to the original 2-head .pth checkpoint.
        device:          Device string to map tensors to while loading.

    Returns:
        OcclusionDetectionModel with enable_extra_heads=True.
    """
    model = build_model(cfg, enable_extra_heads=True)

    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model"] if "model" in state else state

    # strict=False: region_head / confidence_head keys are absent from the
    # old checkpoint and will simply keep their random initialisation.
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    expected_missing = {
        k for k in missing
        if k.startswith("region_head.") or k.startswith("confidence_head.")
    }
    truly_missing = set(missing) - expected_missing
    if truly_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading base checkpoint: {truly_missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint not present in model: {unexpected}"
        )

    return model
