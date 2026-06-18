"""
engine/evaluator.py
-------------------
Validation / test evaluation loop.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import MetricAccumulator


class Evaluator:
    """
    Runs model evaluation (validation or test) on a DataLoader.

    Args:
        model:  OcclusionDetectionModel
        device: torch.device
        amp:    Use mixed precision for forward pass (no backward).
        split:  'val' | 'test' — used for tqdm labels.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        amp: bool = True,
        split: str = "val",
    ):
        self.model  = model
        self.device = device
        self.amp    = amp
        self.split  = split

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """
        Run one full evaluation pass.

        Returns:
            dict with accuracy, precision, recall, f1, mae, rmse, loss.
        """
        self.model.eval()
        acc = MetricAccumulator()

        pbar = tqdm(
            loader,
            desc=f"[{self.split.upper()} Epoch {epoch}]",
            leave=False,
        )

        for batch in pbar:
            frames     = batch["frames"].to(self.device)
            det_labels = batch["det_labels"].to(self.device)
            sev_labels = batch["sev_labels"].to(self.device)

            with autocast(enabled=self.amp):
                det_logits, sev_preds = self.model(frames)
                losses = self.model.compute_loss(
                    det_logits, sev_preds, det_labels, sev_labels
                )

            acc.update(
                det_logits, sev_preds,
                det_labels, sev_labels,
                total_loss=losses["total"].item(),
                det_loss=losses["det"].item(),
                sev_loss=losses["sev"].item(),
            )

            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        metrics = acc.compute()
        return metrics
