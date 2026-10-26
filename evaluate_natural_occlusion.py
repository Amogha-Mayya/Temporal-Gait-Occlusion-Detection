#!/usr/bin/env python
"""
evaluate_natural_occlusion.py
------------------------------
Evaluate the trained model on SUSTech1K's NATURAL OCCLUSION sequences.

This is the key novel evaluation for your paper — the model was trained
ONLY on synthetic occlusion (NM sequences with injected patterns) but is
now tested on REAL naturally occurring occlusion (01-oc condition sequences)
WITHOUT any fine-tuning or adaptation.

If the model detects natural occlusion accurately, it demonstrates that:
  1. The temporal Transformer learned a generalizable occlusion signal
  2. Synthetic training transfers to real-world occlusion patterns
  3. The 4-head framework works on wild outdoor data

WHAT THIS SCRIPT DOES
-----------------------
1. Loads clean NM sequences from test subjects → ground truth det=0
2. Loads natural OC sequences from test subjects → ground truth det=1
3. Runs the trained model on both WITHOUT any occlusion injection
4. Reports Detection Accuracy, Precision, Recall, F1 against natural GT
5. Saves a detailed per-condition breakdown report

Usage:
    # Using the 2-head model
    python evaluate_natural_occlusion.py \
        --config configs/sustech1k.yaml \
        --checkpoint checkpoints/sustech1k/best_model.pth

    # Using the 4-head model (after finetune_new_heads_sustech1k.py)
    python evaluate_natural_occlusion.py \
        --config configs/sustech1k.yaml \
        --checkpoint checkpoints/sustech1k/best_model_4heads.pth \
        --four_heads

    # Cross-dataset: use CASIA-B trained model on SUSTech1K OC
    python evaluate_natural_occlusion.py \
        --config configs/sustech1k.yaml \
        --checkpoint checkpoints/best_model.pth \
        --report_name casia_to_sustech1k
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset

from data.sustech1k_dataset import (
    OC_CONDITIONS,
    NM_CONDITION,
    build_sustech1k_oc_eval_dataset,
    build_sustech1k_subject_split,
)
from data.transforms import build_transforms
from models.model import build_model, build_model_with_pretrained_base
from utils.metrics import MetricAccumulator
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate gait occlusion model on natural OC sequences."
    )
    p.add_argument("--config",      type=str, default="configs/sustech1k.yaml")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--four_heads",  action="store_true",
                   help="Load as 4-head model (after finetune_new_heads)")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--report_name", type=str, default="natural_oc_eval",
                   help="Name for the output report file")
    p.add_argument("--output_dir",  type=str, default="outputs")
    return p.parse_args()


def evaluate_loader(model, loader, device, amp, four_heads=False):
    """Run model on a DataLoader, return predictions and labels."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            frames     = batch["frames"].to(device)
            det_labels = batch["det_labels"].to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                if four_heads:
                    out = model.forward_all(frames)
                    det_logits = out["det_logits"]
                else:
                    det_logits, _ = model(frames)

            det_prob   = torch.sigmoid(det_logits)         # (B, T)
            det_binary = (det_prob >= 0.5).long()

            # Flatten batch and temporal dimensions
            all_preds.append(det_binary.reshape(-1).cpu())
            all_labels.append(det_labels.reshape(-1).cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return preds, labels


def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, F1."""
    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy":  accuracy.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
        "f1":        f1.item(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total": int(tp + tn + fp + fn),
    }


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    amp = cfg["training"]["amp"]

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ---------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    if args.four_heads:
        model = build_model_with_pretrained_base(
            cfg, args.checkpoint, device=str(device)
        ).to(device)
    else:
        model = build_model(cfg).to(device)
        state = torch.load(args.checkpoint, map_location=str(device))
        model_state = state["model"] if "model" in state else state
        model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"Model loaded. 4-head mode: {args.four_heads}")

    # ---- Build evaluation datasets ------------------------------------------
    val_transform = build_transforms(cfg, split="val")

    split_map    = build_sustech1k_subject_split(
        train_end=cfg["split"]["train_end"],
        val_end=cfg["split"]["val_end"],
        test_end=cfg["split"]["test_end"],
    )
    test_subjects = split_map["test"]
    print(f"Test subjects: {len(test_subjects)} (IDs {min(test_subjects)}–{max(test_subjects)})")

    # Clean NM sequences → GT label = 0
    print("Building NM (clean) eval dataset...")
    nm_dataset = build_sustech1k_oc_eval_dataset(
        cfg, conditions=[NM_CONDITION],
        subjects=test_subjects, transform=val_transform,
    )
    print(f"  NM samples: {len(nm_dataset)}")

    # Pure OC sequences → GT label = 1
    print("Building OC (natural occlusion) eval dataset...")
    oc_dataset = build_sustech1k_oc_eval_dataset(
        cfg, conditions=["01-oc"],
        subjects=test_subjects, transform=val_transform,
    )
    print(f"  OC samples: {len(oc_dataset)}")

    # Combined OC (including cr-oc, uf-oc)
    print("Building combined OC eval dataset...")
    all_oc_dataset = build_sustech1k_oc_eval_dataset(
        cfg, conditions=list(OC_CONDITIONS),
        subjects=test_subjects, transform=val_transform,
    )
    print(f"  All OC samples: {len(all_oc_dataset)}")

    def make_loader(ds):
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

    # ---- Run evaluation -----------------------------------------------------
    print("\nRunning inference on NM (clean) sequences...")
    nm_preds, nm_labels = evaluate_loader(
        model, make_loader(nm_dataset), device, amp, args.four_heads
    )

    print("Running inference on pure OC sequences...")
    oc_preds, oc_labels = evaluate_loader(
        model, make_loader(oc_dataset), device, amp, args.four_heads
    )

    print("Running inference on all OC sequences (including combined conditions)...")
    all_oc_preds, all_oc_labels = evaluate_loader(
        model, make_loader(all_oc_dataset), device, amp, args.four_heads
    )

    # Combined: NM + OC together (full binary evaluation)
    combined_preds  = torch.cat([nm_preds,  oc_preds])
    combined_labels = torch.cat([nm_labels, oc_labels])

    # ---- Compute metrics ----------------------------------------------------
    nm_metrics         = compute_metrics(nm_preds, nm_labels)
    oc_metrics         = compute_metrics(oc_preds, oc_labels)
    all_oc_metrics     = compute_metrics(all_oc_preds, all_oc_labels)
    combined_metrics   = compute_metrics(combined_preds, combined_labels)

    # ---- Print and save report ----------------------------------------------
    report_lines = []

    def pr(line=""):
        print(line)
        report_lines.append(line)

    pr("=" * 70)
    pr("  NATURAL OCCLUSION EVALUATION — SUSTech1K")
    pr(f"  Checkpoint : {args.checkpoint}")
    pr(f"  Config     : {args.config}")
    pr(f"  4-head     : {args.four_heads}")
    pr(f"  Test subjects : {len(test_subjects)}")
    pr("=" * 70)
    pr()
    pr("IMPORTANT: Model was trained on SYNTHETIC occlusion only.")
    pr("This evaluation uses REAL NATURAL occlusion sequences (no injection).")
    pr("A good result here demonstrates genuine generalisation.")
    pr()

    pr("─" * 70)
    pr("  1. CLEAN SEQUENCES (00-nm condition) — Expected: model says clean")
    pr("─" * 70)
    pr(f"  Samples     : {nm_metrics['total']:,} frames")
    pr(f"  Accuracy    : {nm_metrics['accuracy']:.4f}")
    pr(f"  False Positive Rate : {nm_metrics['fp'] / (nm_metrics['fp'] + nm_metrics['tn'] + 1e-8):.4f}")
    pr(f"  (FP = model incorrectly flagging a clean frame as occluded)")
    pr()

    pr("─" * 70)
    pr("  2. PURE OCCLUSION (01-oc condition) — Expected: model says occluded")
    pr("─" * 70)
    pr(f"  Samples     : {oc_metrics['total']:,} frames")
    pr(f"  Accuracy    : {oc_metrics['accuracy']:.4f}")
    pr(f"  Precision   : {oc_metrics['precision']:.4f}")
    pr(f"  Recall      : {oc_metrics['recall']:.4f}")
    pr(f"  F1 Score    : {oc_metrics['f1']:.4f}")
    pr(f"  TP={oc_metrics['tp']}  FP={oc_metrics['fp']}  "
       f"TN={oc_metrics['tn']}  FN={oc_metrics['fn']}")
    pr()

    pr("─" * 70)
    pr("  3. ALL OC CONDITIONS (01-oc + 01-cr-oc + 01-uf-oc)")
    pr("─" * 70)
    pr(f"  Samples     : {all_oc_metrics['total']:,} frames")
    pr(f"  Accuracy    : {all_oc_metrics['accuracy']:.4f}")
    pr(f"  Precision   : {all_oc_metrics['precision']:.4f}")
    pr(f"  Recall      : {all_oc_metrics['recall']:.4f}")
    pr(f"  F1 Score    : {all_oc_metrics['f1']:.4f}")
    pr()

    pr("─" * 70)
    pr("  4. COMBINED (NM + OC) — Full binary detection evaluation")
    pr("─" * 70)
    pr(f"  Total frames : {combined_metrics['total']:,}")
    pr(f"  Accuracy     : {combined_metrics['accuracy']:.4f}")
    pr(f"  Precision    : {combined_metrics['precision']:.4f}")
    pr(f"  Recall       : {combined_metrics['recall']:.4f}")
    pr(f"  F1 Score     : {combined_metrics['f1']:.4f}")
    pr(f"  TP={combined_metrics['tp']}  FP={combined_metrics['fp']}  "
       f"TN={combined_metrics['tn']}  FN={combined_metrics['fn']}")
    pr()
    pr("=" * 70)

    # Save report
    report_path = os.path.join(args.output_dir, f"{args.report_name}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
