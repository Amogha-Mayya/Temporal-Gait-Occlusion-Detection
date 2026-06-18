#!/usr/bin/env python
"""
train.py
--------
Main training script for temporal gait occlusion detection on CASIA-B.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/last_model.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# ---- Local imports ---------------------------------------------------------
from data.casia_dataset import build_dataset
from data.transforms import build_transforms
from engine.evaluator import Evaluator
from engine.trainer import Trainer
from models.model import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import Logger
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train temporal gait occlusion detection model on CASIA-B"
    )
    p.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML configuration file."
    )
    p.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume training from."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Early-stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = None
        self.counter    = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load config -------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Resolve resume path from config if not overridden -----------------
    resume = args.resume or cfg["checkpoint"].get("resume", "")

    # ---- Reproducibility ---------------------------------------------------
    set_seed(cfg["training"]["seed"])

    # ---- Device ------------------------------------------------------------
    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda" and cfg["gpu"].get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    # ---- Output directories ------------------------------------------------
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint"]["dir"], exist_ok=True)

    # ---- Logger ------------------------------------------------------------
    logger = Logger(
        log_file=cfg["logging"]["log_file"],
        tensorboard_dir=cfg["logging"]["tensorboard_dir"],
    )
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Datasets & DataLoaders --------------------------------------------
    train_transform = build_transforms(cfg, split="train")
    val_transform   = build_transforms(cfg, split="val")

    logger.info("Building datasets...")
    train_dataset = build_dataset(cfg, split="train", transform=train_transform)
    val_dataset   = build_dataset(cfg, split="val",   transform=val_transform)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model -------------------------------------------------------------
    logger.info("Building model...")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---- LR Scheduler ------------------------------------------------------
    sch_cfg   = cfg["training"]["scheduler"]
    max_eps   = cfg["training"]["epochs"]
    warmup_eps = sch_cfg.get("warmup_epochs", 3)

    # Cosine annealing with warmup (manual linear warmup + CosineAnnealingLR)
    def warmup_lambda(epoch):
        if epoch < warmup_eps:
            return float(epoch + 1) / float(warmup_eps)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_eps - warmup_eps,
        eta_min=sch_cfg.get("eta_min", 1e-6),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_eps],
    )

    # ---- AMP Scaler --------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"])

    # ---- Resume ------------------------------------------------------------
    start_epoch = 0
    best_metric = 0.0
    if resume and os.path.isfile(resume):
        logger.info(f"Resuming from checkpoint: {resume}")
        info = load_checkpoint(
            resume, model, optimizer, scheduler, scaler,
            device=str(device),
        )
        start_epoch  = info["epoch"] + 1
        best_metric  = info["best_metric"]
        logger.info(f"  Resumed at epoch {start_epoch}, best metric = {best_metric:.4f}")

    # ---- Training objects --------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        grad_clip=cfg["training"]["grad_clip"],
        amp=cfg["training"]["amp"],
        logger=logger,
        log_interval=cfg["logging"]["log_interval"],
    )

    evaluator = Evaluator(
        model=model,
        device=device,
        amp=cfg["training"]["amp"],
        split="val",
    )

    # ---- Early stopping ----------------------------------------------------
    es_cfg = cfg["training"]["early_stopping"]
    monitor = es_cfg.get("monitor", "val_f1")
    es_mode = "min" if "loss" in monitor or "mae" in monitor or "rmse" in monitor else "max"
    early_stop = EarlyStopping(
        patience=es_cfg["patience"],
        min_delta=es_cfg["min_delta"],
        mode=es_mode,
    )

    ckpt_dir = cfg["checkpoint"]["dir"]
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    # ---- Training loop -----------------------------------------------------
    logger.info(f"Starting training for {max_eps} epochs (starting from {start_epoch})")

    for epoch in range(start_epoch, max_eps):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        logger.log_epoch_metrics(train_metrics, epoch, prefix="train")
        logger.info(
            f"[Epoch {epoch:03d}] Train  loss={train_metrics['loss']:.4f}  "
            f"acc={train_metrics['accuracy']:.4f}  "
            f"f1={train_metrics['f1']:.4f}  "
            f"mae={train_metrics['mae']:.4f}"
        )

        # Validate
        val_metrics = evaluator.evaluate(val_loader, epoch)
        logger.log_epoch_metrics(val_metrics, epoch, prefix="val")
        logger.info(
            f"[Epoch {epoch:03d}] Val    loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"f1={val_metrics['f1']:.4f}  "
            f"mae={val_metrics['mae']:.4f}  "
            f"rmse={val_metrics['rmse']:.4f}"
        )

        # Determine monitored metric
        metric_key_map = {
            "val_f1":   val_metrics["f1"],
            "val_loss": -val_metrics["loss"],  # negate so "max" works
            "val_mae":  -val_metrics["mae"],
        }
        monitored_value = metric_key_map.get(monitor, val_metrics["f1"])
        raw_value = val_metrics["f1"] if monitor == "val_f1" else val_metrics.get(
            monitor.replace("val_", ""), val_metrics["f1"]
        )

        # Save best
        if monitored_value > best_metric:
            best_metric = monitored_value
            if cfg["checkpoint"]["save_best"]:
                save_checkpoint(
                    best_path, model, optimizer, scheduler, scaler,
                    epoch, best_metric, cfg
                )
                logger.info(f"  ✓ Saved best model (epoch={epoch}, metric={raw_value:.4f})")

        # Save last
        if cfg["checkpoint"]["save_last"]:
            save_checkpoint(
                last_path, model, optimizer, scheduler, scaler,
                epoch, best_metric, cfg
            )

        # Early stopping check
        if early_stop.step(monitored_value):
            logger.info(
                f"Early stopping triggered after {epoch+1} epochs "
                f"(patience={es_cfg['patience']})."
            )
            break

    logger.info("Training complete.")
    logger.info(f"Best {monitor}: {best_metric:.4f}")
    logger.info(f"Best checkpoint saved at: {best_path}")
    logger.close()


if __name__ == "__main__":
    main()
