#!/usr/bin/env python
"""
train_sustech1k.py
------------------
Training entry point for SUSTech1K.
Trains on 00-nm sequences with synthetic occlusion injection.
Identical to train_oumvlp.py except uses SUSTech1KDataset.

Usage:
    python train_sustech1k.py --config configs/sustech1k.yaml
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from data.sustech1k_dataset import build_sustech1k_dataset
from data.transforms import build_transforms
from engine.evaluator import Evaluator
from engine.trainer import Trainer
from models.model import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import Logger
from utils.seed import set_seed


class EarlyStopping:
    def __init__(self, patience, min_delta, mode="max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        improved = (
            value > self.best + self.min_delta if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best   = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/sustech1k.yaml")
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg["gpu"].get("cudnn_benchmark", True)

    os.makedirs(cfg["output"]["dir"],      exist_ok=True)
    os.makedirs(cfg["checkpoint"]["dir"],  exist_ok=True)

    logger = Logger(
        log_file=cfg["logging"]["log_file"],
        tensorboard_dir=cfg["logging"]["tensorboard_dir"],
    )
    logger.info(f"Config: {args.config} | Device: {device}")

    # ---- Datasets -----------------------------------------------------------
    train_transform = build_transforms(cfg, split="train")
    val_transform   = build_transforms(cfg, split="val")

    logger.info("Building SUSTech1K datasets (NM condition, synthetic occlusion)...")
    train_dataset = build_sustech1k_dataset(cfg, "train", transform=train_transform)

    # Val: use a subset of test subjects with NM only for training monitoring
    # (Official split has no val — we use first 50 test subjects as proxy val)
    import copy
    val_cfg = copy.deepcopy(cfg)
    val_cfg["split"]["train_end"] = cfg["split"]["train_end"] - 50  # 0000–0749 train
    val_cfg["split"]["val_end"]   = cfg["split"]["train_end"]       # 0750–0799 val
    val_cfg["split"]["test_end"]  = cfg["split"]["train_end"]       # stop here
    val_dataset = build_sustech1k_dataset(val_cfg, "val", transform=val_transform,
                                        occlusion_enabled=True)

    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model + Optimizer --------------------------------------------------
    model = build_model(cfg).to(device)
    logger.info(f"Trainable params: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sch_cfg    = cfg["training"]["scheduler"]
    warmup_eps = sch_cfg.get("warmup_epochs", 3)
    max_eps    = cfg["training"]["epochs"]

    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: (e + 1) / warmup_eps if e < warmup_eps else 1.0
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_eps - warmup_eps, eta_min=sch_cfg.get("eta_min", 1e-6)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_eps]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"])

    # ---- Resume -------------------------------------------------------------
    start_epoch = 0
    best_metric = 0.0
    resume = args.resume or cfg["checkpoint"].get("resume", "")
    if resume and os.path.isfile(resume):
        info = load_checkpoint(resume, model, optimizer, scheduler, scaler,
                               device=str(device))
        start_epoch = info["epoch"] + 1
        best_metric = info["best_metric"]
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop ------------------------------------------------------
    trainer   = Trainer(model=model, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, device=device,
                        grad_clip=cfg["training"]["grad_clip"],
                        amp=cfg["training"]["amp"], logger=logger,
                        log_interval=cfg["logging"]["log_interval"])
    evaluator = Evaluator(model=model, device=device,
                          amp=cfg["training"]["amp"], split="val")

    es_cfg = cfg["training"]["early_stopping"]
    early_stop = EarlyStopping(es_cfg["patience"], es_cfg["min_delta"])

    best_path = os.path.join(cfg["checkpoint"]["dir"], "best_model.pth")
    last_path = os.path.join(cfg["checkpoint"]["dir"], "last_model.pth")

    for epoch in range(start_epoch, max_eps):
        train_m = trainer.train_epoch(train_loader, epoch)
        logger.log_epoch_metrics(train_m, epoch, prefix="train")
        logger.info(f"[Epoch {epoch:03d}] Train loss={train_m['loss']:.4f} "
                    f"f1={train_m['f1']:.4f} mae={train_m['mae']:.4f}")

        val_m = evaluator.evaluate(val_loader, epoch)
        logger.log_epoch_metrics(val_m, epoch, prefix="val")
        logger.info(f"[Epoch {epoch:03d}] Val   loss={val_m['loss']:.4f} "
                    f"acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f} "
                    f"mae={val_m['mae']:.4f}")

        if val_m["f1"] > best_metric:
            best_metric = val_m["f1"]
            save_checkpoint(best_path, model, optimizer, scheduler,
                            scaler, epoch, best_metric, cfg)
            logger.info(f"  ✓ Saved best (epoch={epoch}, f1={best_metric:.4f})")

        save_checkpoint(last_path, model, optimizer, scheduler,
                        scaler, epoch, best_metric, cfg)

        if early_stop.step(val_m["f1"]):
            logger.info("Early stopping.")
            break

    logger.info(f"Training complete. Best F1: {best_metric:.4f}")
    logger.close()


if __name__ == "__main__":
    main()
