#!/usr/bin/env python
"""
inference.py
------------
Run per-frame occlusion detection on an arbitrary sequence folder.

Usage:
    python inference.py \\
        --sequence path/to/001/nm-01 \\
        --checkpoint checkpoints/best_model.pth \\
        --config configs/config.yaml \\
        --output outputs/inference_results

Output:
    - Frame-by-frame predictions printed to console
    - outputs/inference_results/pred_*.png  — annotated frames
    - outputs/inference_results/predictions.csv
    - outputs/inference_results/temporal_plot.png
    - outputs/inference_results/occluded.gif
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import yaml

from engine.inference_engine import InferenceEngine
from models.model import build_model
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed
from utils.visualization import (
    plot_temporal,
    save_gif,
    save_inference_frames,
)


def parse_args():
    p = argparse.ArgumentParser(description="Inference on a gait sequence folder.")
    p.add_argument("--sequence",   type=str, required=True,
                   help="Path to the sequence folder containing PNG frames.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--config",     type=str, default="configs/config.yaml")
    p.add_argument("--output",     type=str, default="outputs/inference_results",
                   help="Directory to save inference outputs.")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Detection probability threshold.")
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

    # ---- Load model --------------------------------------------------------
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # ---- Inference engine --------------------------------------------------
    engine = InferenceEngine(
        model=model,
        cfg=cfg,
        device=device,
        det_threshold=args.threshold,
    )

    print(f"Running inference on: {args.sequence}")
    results = engine.run(args.sequence)

    frames     = results["frames"]
    det_prob   = results["det_prob"]
    det_binary = results["det_binary"]
    sev        = results["sev"]
    n_frames   = len(frames)

    # ---- Console output ----------------------------------------------------
    print("\n" + "=" * 55)
    print(f"{'Frame':>6}  {'Detection':>10}  {'Det Prob':>8}  {'Severity':>8}")
    print("-" * 55)
    for i in range(n_frames):
        status = "Occluded" if det_binary[i] == 1 else "Clean   "
        print(f"{i:>6}  {status:>10}  {det_prob[i]:>8.3f}  {sev[i]:>8.3f}")
    print("=" * 55)

    n_occ = sum(det_binary)
    print(f"\nSummary: {n_occ}/{n_frames} frames predicted as occluded.")
    if n_occ > 0:
        avg_sev = sum(s for d, s in zip(det_binary, sev) if d == 1) / n_occ
        print(f"Mean severity (occluded frames): {avg_sev:.3f}")

    # ---- Save outputs ------------------------------------------------------
    out_dir = args.output
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Annotated frames
    save_inference_frames(frames, det_prob, det_binary, sev, out_dir)
    print(f"Annotated frames saved to: {out_dir}/pred_*.png")

    # Temporal plot (no ground truth available, so use zeros as dummy GT)
    gt_dummy_det = [0] * n_frames
    gt_dummy_sev = [0.0] * n_frames
    plot_temporal(
        gt_det=gt_dummy_det, pred_det=det_prob,
        gt_sev=gt_dummy_sev, pred_sev=sev,
        save_path=os.path.join(out_dir, "temporal_plot.png"),
        title=f"Inference: {Path(args.sequence).name}",
    )
    print(f"Temporal plot saved to: {out_dir}/temporal_plot.png")

    # GIF
    save_gif(frames, os.path.join(out_dir, "sequence.gif"), fps=8)
    print(f"Sequence GIF saved to: {out_dir}/sequence.gif")

    # CSV
    csv_path = os.path.join(out_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "frame_path", "det_binary",
                         "det_prob", "severity"])
        for i in range(n_frames):
            fp = results["frame_paths"][i] if i < len(results["frame_paths"]) else ""
            writer.writerow([i, fp, det_binary[i],
                             f"{det_prob[i]:.6f}", f"{sev[i]:.6f}"])
    print(f"Predictions CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
