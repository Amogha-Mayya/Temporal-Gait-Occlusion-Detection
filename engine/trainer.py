"""
engine/trainer.py
-----------------
Core training loop with AMP, gradient clipping, and TensorBoard logging.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import MetricAccumulator


class Trainer:
    """
    Encapsulates one training epoch and provides the outer epoch loop.

    Args:
        model:      OcclusionDetectionModel
        optimizer:  AdamW optimizer
        scheduler:  LR scheduler (stepped per epoch)
        scaler:     AMP GradScaler
        device:     torch.device
        grad_clip:  Max gradient norm
        amp:        Whether to use mixed precision
        logger:     Logger instance
        log_interval: Log loss every N batches
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        device: torch.device,
        grad_clip: float = 1.0,
        amp: bool = True,
        logger=None,
        log_interval: int = 10,
    ):
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.scaler       = scaler
        self.device       = device
        self.grad_clip    = grad_clip
        self.amp          = amp
        self.logger       = logger
        self.log_interval = log_interval
        self.global_step  = 0

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Run one full training epoch.

        Returns:
            dict of epoch-averaged metrics.
        """
        self.model.train()
        acc = MetricAccumulator()

        pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            frames     = batch["frames"].to(self.device)       # (B, T, 1, H, W)
            det_labels = batch["det_labels"].to(self.device)   # (B, T)
            sev_labels = batch["sev_labels"].to(self.device)   # (B, T)

            self.optimizer.zero_grad(set_to_none=True)

            # ---- Forward (AMP) -----------------------------------------
            with autocast(enabled=self.amp):
                det_logits, sev_preds = self.model(frames)
                losses = self.model.compute_loss(
                    det_logits, sev_preds, det_labels, sev_labels
                )

            # ---- Backward + Grad clip + Step ---------------------------
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ---- Accumulate metrics ------------------------------------
            total_l = losses["total"].item()
            det_l   = losses["det"].item()
            sev_l   = losses["sev"].item()

            acc.update(
                det_logits.detach(), sev_preds.detach(),
                det_labels, sev_labels,
                total_loss=total_l,
                det_loss=det_l,
                sev_loss=sev_l,
            )

            pbar.set_postfix(loss=f"{total_l:.4f}")

            # ---- TensorBoard batch-level logging -----------------------
            if self.logger and (self.global_step % self.log_interval == 0):
                self.logger.log_scalars(
                    {
                        "train/batch_loss":     total_l,
                        "train/batch_det_loss": det_l,
                        "train/batch_sev_loss": sev_l,
                    },
                    step=self.global_step,
                )
                cur_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_lr(cur_lr, step=self.global_step)

            self.global_step += 1

        # Step LR scheduler once per epoch
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = acc.compute()
        return metrics
