"""
utils/metrics.py
----------------
Evaluation metrics for detection (classification) and severity (regression).

Detection metrics:
    Accuracy, Precision, Recall, F1-score
    Computed over the flattened (B*T,) predictions.

Severity metrics:
    MAE  (Mean Absolute Error)
    RMSE (Root Mean Squared Error)
    Computed only on frames that are actually occluded (det_label == 1),
    so we measure how accurately we estimate *how bad* the occlusion is,
    not on clean frames where severity is trivially 0.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def compute_detection_metrics(
    det_preds: torch.Tensor,   # (N,) long  0/1 predicted binary labels
    det_labels: torch.Tensor,  # (N,) long  0/1 ground-truth labels
) -> Dict[str, float]:
    """
    Args:
        det_preds:  Binary predictions, shape (N,).
        det_labels: Ground-truth labels, shape (N,).

    Returns:
        Dict with 'accuracy', 'precision', 'recall', 'f1'.
    """
    preds  = det_preds.cpu().numpy().astype(int)
    labels = det_labels.cpu().numpy().astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n = len(preds)
    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy":  float(accuracy),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def compute_severity_metrics(
    sev_preds: torch.Tensor,   # (N,) float predicted severities
    sev_labels: torch.Tensor,  # (N,) float ground-truth severities
    mask: torch.Tensor = None, # (N,) bool — if given, compute only on True entries
) -> Dict[str, float]:
    """
    Args:
        sev_preds:  Predicted severity values, shape (N,).
        sev_labels: Ground-truth severity values, shape (N,).
        mask:       Optional boolean mask to restrict computation.

    Returns:
        Dict with 'mae', 'rmse'.
    """
    p = sev_preds.cpu().float()
    t = sev_labels.cpu().float()

    if mask is not None:
        m = mask.cpu().bool()
        p = p[m]
        t = t[m]

    if len(p) == 0:
        return {"mae": 0.0, "rmse": 0.0}

    diff = p - t
    mae  = float(diff.abs().mean().item())
    rmse = float(diff.pow(2).mean().sqrt().item())

    return {"mae": mae, "rmse": rmse}


class RegionConfidenceMetricAccumulator:
    """
    Accumulates predictions for the two NEW heads (RegionHead, ConfidenceHead)
    across batches, for use in finetune_new_heads.py.

    Region metrics are computed independently per quadrant (since regions
    are multi-label, not mutually exclusive) and then averaged.

    Usage:
        acc = RegionConfidenceMetricAccumulator()
        for batch in loader:
            acc.update(region_logits, region_labels, conf_preds, conf_targets, loss=...)
        results = acc.compute()
    """

    REGION_NAMES = ["upper", "lower", "left", "right"]

    def __init__(self):
        self._region_preds:  list = []
        self._region_labels: list = []
        self._conf_preds:    list = []
        self._conf_targets:  list = []
        self._losses:        list = []
        self._region_losses: list = []
        self._conf_losses:   list = []

    def update(
        self,
        region_logits: torch.Tensor,   # (B, T, 4)
        region_labels: torch.Tensor,   # (B, T, 4)
        conf_preds: torch.Tensor,      # (B, T)
        conf_targets: torch.Tensor,    # (B, T)
        total_loss: float = 0.0,
        region_loss: float = 0.0,
        conf_loss: float = 0.0,
    ) -> None:
        region_binary = (torch.sigmoid(region_logits) >= 0.5).long()
        self._region_preds.append(region_binary.reshape(-1, 4).cpu())
        self._region_labels.append(region_labels.reshape(-1, 4).long().cpu())
        self._conf_preds.append(conf_preds.reshape(-1).cpu())
        self._conf_targets.append(conf_targets.reshape(-1).cpu())
        self._losses.append(total_loss)
        self._region_losses.append(region_loss)
        self._conf_losses.append(conf_loss)

    def compute(self) -> Dict[str, float]:
        region_preds  = torch.cat(self._region_preds)    # (N, 4)
        region_labels = torch.cat(self._region_labels)   # (N, 4)
        conf_preds    = torch.cat(self._conf_preds)       # (N,)
        conf_targets  = torch.cat(self._conf_targets)     # (N,)

        results: Dict[str, float] = {}

        # Per-region metrics + macro average
        f1s = []
        accs = []
        for i, name in enumerate(self.REGION_NAMES):
            m = compute_detection_metrics(region_preds[:, i], region_labels[:, i])
            results[f"region_{name}_accuracy"] = m["accuracy"]
            results[f"region_{name}_f1"]       = m["f1"]
            f1s.append(m["f1"])
            accs.append(m["accuracy"])

        results["region_macro_f1"]       = float(np.mean(f1s))
        results["region_macro_accuracy"] = float(np.mean(accs))

        # Confidence regression metrics
        conf_metrics = compute_severity_metrics(conf_preds, conf_targets)
        results["confidence_mae"]  = conf_metrics["mae"]
        results["confidence_rmse"] = conf_metrics["rmse"]

        # results["loss"]        = float(np.mean(self._losses)) if self._losses else 0.0
        results["loss"]        = float(np.mean(self._losses))        if self._losses else 0.0
        results["region_loss"] = float(np.mean(self._region_losses)) if self._region_losses else 0.0
        results["conf_loss"]   = float(np.mean(self._conf_losses))   if self._conf_losses else 0.0

        return results

    def reset(self) -> None:
        self._region_preds  = []
        self._region_labels = []
        self._conf_preds    = []
        self._conf_targets  = []
        self._losses        = []
        self._region_losses = []
        self._conf_losses   = []


class MetricAccumulator:
    """
    Accumulates predictions and labels across batches, then computes metrics
    at epoch end.

    Usage:
        acc = MetricAccumulator()
        for batch in loader:
            acc.update(det_logits, sev_preds, det_labels, sev_labels)
        results = acc.compute()
        acc.reset()
    """

    def __init__(self):
        self._det_preds:  list = []
        self._det_labels: list = []
        self._sev_preds:  list = []
        self._sev_labels: list = []
        self._losses:     list = []
        self._det_losses: list = []
        self._sev_losses: list = []

    def update(
        self,
        det_logits:  torch.Tensor,   # (B, T) raw logits
        sev_preds:   torch.Tensor,   # (B, T) [0,1]
        det_labels:  torch.Tensor,   # (B, T) long
        sev_labels:  torch.Tensor,   # (B, T) float
        total_loss:  float = 0.0,
        det_loss:    float = 0.0,
        sev_loss:    float = 0.0,
    ) -> None:
        det_binary = (torch.sigmoid(det_logits) >= 0.5).long()
        self._det_preds.append(det_binary.view(-1).cpu())
        self._det_labels.append(det_labels.view(-1).cpu())
        self._sev_preds.append(sev_preds.view(-1).cpu())
        self._sev_labels.append(sev_labels.view(-1).cpu())
        self._losses.append(total_loss)
        self._det_losses.append(det_loss)
        self._sev_losses.append(sev_loss)

    def compute(self) -> Dict[str, float]:
        all_det_preds  = torch.cat(self._det_preds)
        all_det_labels = torch.cat(self._det_labels)
        all_sev_preds  = torch.cat(self._sev_preds)
        all_sev_labels = torch.cat(self._sev_labels)

        det_metrics = compute_detection_metrics(all_det_preds, all_det_labels)

        # Compute severity metrics only on occluded frames
        occ_mask = all_det_labels.bool()
        sev_metrics = compute_severity_metrics(
            all_sev_preds, all_sev_labels, mask=occ_mask
        )

        results = {
            **det_metrics,
            **sev_metrics,
            "loss":     float(np.mean(self._losses)) if self._losses else 0.0,
            "det_loss": float(np.mean(self._det_losses)) if self._det_losses else 0.0,
            "sev_loss": float(np.mean(self._sev_losses)) if self._sev_losses else 0.0,
        }
        return results

    def reset(self) -> None:
        self._det_preds  = []
        self._det_labels = []
        self._sev_preds  = []
        self._sev_labels = []
        self._losses     = []
        self._det_losses = []
        self._sev_losses = []
