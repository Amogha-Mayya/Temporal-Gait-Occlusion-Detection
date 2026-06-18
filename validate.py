#!/usr/bin/env python
"""
validate.py
-----------
Run validation on the validation split and print full metrics.

Usage:
    python validate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from data.casia_dataset import build_dataset
from data.transforms import build_transforms
from engine.evaluator import Evaluator
from models.model import build_model
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Validate occlusion detection model.")
    p.add_argument("--config",     type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pth).")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # ---- Dataset -----------------------------------------------------------
    transform = build_transforms(cfg, split="val")
    dataset   = build_dataset(cfg, split="val", transform=transform)
    loader    = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    print(f"Validation samples: {len(dataset)}")

    # ---- Model -------------------------------------------------------------
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Evaluate ----------------------------------------------------------
    evaluator = Evaluator(model, device, amp=cfg["training"]["amp"], split="val")
    metrics   = evaluator.evaluate(loader, epoch=0)

    # ---- Print results -----------------------------------------------------
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"  Loss       : {metrics['loss']:.6f}")
    print(f"  Det Loss   : {metrics['det_loss']:.6f}")
    print(f"  Sev Loss   : {metrics['sev_loss']:.6f}")
    print("-" * 50)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print("-" * 50)
    print(f"  MAE        : {metrics['mae']:.4f}")
    print(f"  RMSE       : {metrics['rmse']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
