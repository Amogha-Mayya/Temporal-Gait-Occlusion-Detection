#!/usr/bin/env python
"""
test.py
-------
Evaluate the best checkpoint on the held-out test split (subjects 100-124).
Generates a full evaluation report saved to outputs/test_report.txt.

Usage:
    python test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Test occlusion detection model on CASIA-B test set.")
    p.add_argument("--config",     type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                   help="Path to best model checkpoint.")
    p.add_argument("--output",     type=str, default="outputs/test_report.txt",
                   help="Path to save the evaluation report.")
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

    print(f"Device: {device}")

    # ---- Dataset -----------------------------------------------------------
    transform = build_transforms(cfg, split="test")
    dataset   = build_dataset(cfg, split="test", transform=transform)
    loader    = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    print(f"Test samples: {len(dataset)}")

    # ---- Model -------------------------------------------------------------
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Evaluate ----------------------------------------------------------
    evaluator = Evaluator(model, device, amp=cfg["training"]["amp"], split="test")
    metrics   = evaluator.evaluate(loader, epoch=0)

    # ---- Report ------------------------------------------------------------
    lines = []
    lines.append("=" * 60)
    lines.append("TEMPORAL GAIT OCCLUSION DETECTION — TEST REPORT")
    lines.append(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Checkpoint : {args.checkpoint}")
    lines.append(f"Config     : {args.config}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("DETECTION METRICS")
    lines.append("-" * 40)
    lines.append(f"  Accuracy  : {metrics['accuracy']:.4f}")
    lines.append(f"  Precision : {metrics['precision']:.4f}")
    lines.append(f"  Recall    : {metrics['recall']:.4f}")
    lines.append(f"  F1 Score  : {metrics['f1']:.4f}")
    lines.append("")
    lines.append("SEVERITY REGRESSION METRICS (on occluded frames)")
    lines.append("-" * 40)
    lines.append(f"  MAE       : {metrics['mae']:.4f}")
    lines.append(f"  RMSE      : {metrics['rmse']:.4f}")
    lines.append("")
    lines.append("LOSS")
    lines.append("-" * 40)
    lines.append(f"  Total     : {metrics['loss']:.6f}")
    lines.append(f"  Detection : {metrics['det_loss']:.6f}")
    lines.append(f"  Severity  : {metrics['sev_loss']:.6f}")
    lines.append("=" * 60)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report_str + "\n")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
