#!/usr/bin/env python
"""
train_oumvlp.py
---------------
Training entry point for OU-MVLP. Identical to train.py in every way except
it uses OUMVLPDataset instead of CasiaGaitDataset.

Usage:
    python3 train_oumvlp.py --config configs/oumvlp.yaml
    python3 train_oumvlp.py --config configs/oumvlp.yaml --resume checkpoints/oumvlp/last_model.pth
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from data.oumvlp_dataset import build_oumvlp_dataset   # ← only difference
from data.transforms import build_transforms
from engine.evaluator import Evaluator
from engine.trainer import Trainer
from models.model import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import Logger
from utils.seed import set_seed


class EarlyStopping:
    def __init__(self, patience, min_delta, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def parse_args():
    p = argparse.ArgumentParser(description="Train gait occlusion detector on OU-MVLP")
    p.add_argument("--config", type=str, default="configs/oumvlp.yaml")
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    resume = args.resume or cfg["checkpoint"].get("resume", "")
    set_seed(cfg["training"]["seed"])

    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda" and cfg["gpu"].get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint"]["dir"], exist_ok=True)

    logger = Logger(
        log_file=cfg["logging"]["log_file"],
        tensorboard_dir=cfg["logging"]["tensorboard_dir"],
    )
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Datasets -----------------------------------------------------------
    train_transform = build_transforms(cfg, split="train")
    val_transform   = build_transforms(cfg, split="val")

    logger.info("Building OU-MVLP datasets...")
    train_dataset = build_oumvlp_dataset(cfg, split="train", transform=train_transform)
    val_dataset   = build_oumvlp_dataset(cfg, split="val",   transform=val_transform)
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

    # ---- Model --------------------------------------------------------------
    logger.info("Building model...")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ---- Optimizer + Scheduler ----------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    sch_cfg    = cfg["training"]["scheduler"]
    max_eps    = cfg["training"]["epochs"]
    warmup_eps = sch_cfg.get("warmup_epochs", 3)

    def warmup_lambda(epoch):
        return float(epoch + 1) / float(warmup_eps) if epoch < warmup_eps else 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_eps - warmup_eps, eta_min=sch_cfg.get("eta_min", 1e-6)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_eps]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"])

    # ---- Resume -------------------------------------------------------------
    start_epoch = 0
    best_metric = 0.0
    if resume and os.path.isfile(resume):
        logger.info(f"Resuming from: {resume}")
        info = load_checkpoint(resume, model, optimizer, scheduler, scaler, device=str(device))
        start_epoch = info["epoch"] + 1
        best_metric = info["best_metric"]
        logger.info(f"  Resumed at epoch {start_epoch}, best metric = {best_metric:.4f}")

    # ---- Training loop ------------------------------------------------------
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        device=device, grad_clip=cfg["training"]["grad_clip"],
        amp=cfg["training"]["amp"], logger=logger,
        log_interval=cfg["logging"]["log_interval"],
    )
    evaluator = Evaluator(model=model, device=device, amp=cfg["training"]["amp"], split="val")

    es_cfg = cfg["training"]["early_stopping"]
    monitor = es_cfg.get("monitor", "val_f1")
    es_mode = "min" if any(k in monitor for k in ["loss", "mae", "rmse"]) else "max"
    early_stop = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"], mode=es_mode)

    ckpt_dir  = cfg["checkpoint"]["dir"]
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    logger.info(f"Starting training for {max_eps} epochs")

    for epoch in range(start_epoch, max_eps):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        logger.log_epoch_metrics(train_metrics, epoch, prefix="train")
        logger.info(
            f"[Epoch {epoch:03d}] Train  loss={train_metrics['loss']:.4f}  "
            f"f1={train_metrics['f1']:.4f}  mae={train_metrics['mae']:.4f}"
        )

        val_metrics = evaluator.evaluate(val_loader, epoch)
        logger.log_epoch_metrics(val_metrics, epoch, prefix="val")
        logger.info(
            f"[Epoch {epoch:03d}] Val    loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  f1={val_metrics['f1']:.4f}  "
            f"mae={val_metrics['mae']:.4f}  rmse={val_metrics['rmse']:.4f}"
        )

        metric_val = val_metrics["f1"] if monitor == "val_f1" else -val_metrics.get(
            monitor.replace("val_", ""), val_metrics["f1"]
        )

        if metric_val > best_metric:
            best_metric = metric_val
            if cfg["checkpoint"]["save_best"]:
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_metric, cfg)
                logger.info(f"  ✓ Saved best model (epoch={epoch}, f1={val_metrics['f1']:.4f})")

        if cfg["checkpoint"]["save_last"]:
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_metric, cfg)

        if early_stop.step(metric_val):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    logger.info("Training complete.")
    logger.info(f"Best checkpoint: {best_path}")
    logger.close()


if __name__ == "__main__":
    main()
