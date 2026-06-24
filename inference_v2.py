#!/usr/bin/env python
"""
inference_v2.py
----------------
Upgraded inference tool that solves the "I can only feed clean silhouettes
and have no way to check results" problem.

WHAT THIS GIVES YOU THAT inference.py DOESN'T
-------------------------------------------------
inference.py only runs the model on whatever frames are already on disk.
If you point it at a clean sequence, you get predictions but NO ground
truth to check them against (because there's no occlusion in clean data,
so "did it detect correctly" is unanswerable beyond "it should say clean").

inference_v2.py lets YOU decide what occlusion to inject (type, frame
range, target severity), generates it the exact same way training data is
generated (via OcclusionGenerator), and then runs the model and shows you
GROUND TRUTH vs PREDICTED side by side for ALL FOUR heads:
  - Detection      (GT occluded/clean        vs predicted probability)
  - Severity       (GT pixel-ratio severity  vs predicted severity)
  - Region         (GT quadrant labels       vs predicted quadrant probs)
  - Confidence     (model's self-assessed reliability per frame — no GT,
                    see utils/confidence.py docstring for what it means)

Two modes:
  1. --mode clean        : run on the sequence exactly as stored on disk
                            (no injected occlusion; useful as a sanity
                            check that the model says "clean" everywhere).
  2. --mode inject        : load the clean sequence, inject a synthetic
                            occlusion event YOU control, then compare.

By default this loads checkpoints/best_model_4heads.pth (produced by
finetune_new_heads.py). If you only have the original 2-head checkpoint,
pass --base_checkpoint and this script will build the 4-head model on the
fly with randomly-initialised region/confidence heads (their output will
be meaningless until you run finetune_new_heads.py — the script will warn
you loudly if so).

USAGE
-----
# Run on a sequence exactly as-is (no injection), full 4-head dashboard:
python inference_v2.py \\
    --sequence /data/CASIA-B/050/nm-03 \\
    --checkpoint checkpoints/best_model_4heads.pth \\
    --mode clean \\
    --output outputs/inference_v2/050_nm03_clean

# Inject a rectangle occlusion on frames 10-20 targeting severity ~0.5:
python inference_v2.py \\
    --sequence /data/CASIA-B/050/nm-03 \\
    --checkpoint checkpoints/best_model_4heads.pth \\
    --mode inject \\
    --occlusion_type rectangle \\
    --start_frame 10 --end_frame 20 \\
    --target_severity 0.5 \\
    --output outputs/inference_v2/050_nm03_injected

OUTPUT FILES (in --output directory)
--------------------------------------
  dashboard.png          4-panel GT-vs-Predicted plot for all heads
  full_overlay/*.png     per-frame rich overlay (quadrant guide + all heads)
  clean.gif              the clean (pre-injection) sequence
  occluded.gif           the sequence actually fed to the model
  paired.gif             clean | occluded side-by-side single GIF
  predictions.csv        full per-frame table: GT + predicted, all heads
  summary.txt            console summary saved to disk
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from data.occlusion_generator import OcclusionGenerator
from data.transforms import SilhouetteTransform, SequenceTransform
from models.model import build_model, build_model_with_pretrained_base
from utils.seed import set_seed
from utils.visualization import (
    plot_four_head_dashboard,
    save_full_inference_frames,
    save_gif,
    save_paired_gif,
)

from PIL import Image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Rich 4-head inference with optional on-the-fly occlusion injection."
    )
    p.add_argument("--sequence", type=str, required=True,
                   help="Path to a folder of PNG silhouette frames.")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model_4heads.pth",
                   help="4-head checkpoint (produced by finetune_new_heads.py).")
    p.add_argument("--base_checkpoint", type=str, default=None,
                   help="If --checkpoint doesn't exist yet, fall back to building "
                        "the 4-head model from this 2-head checkpoint (region/"
                        "confidence heads will be UNTRAINED — output will be noted "
                        "as such).")
    p.add_argument("--output", type=str, default="outputs/inference_v2/result")
    p.add_argument("--mode", type=str, choices=["clean", "inject"], default="inject",
                   help="'clean': run sequence as-is. 'inject': inject synthetic "
                        "occlusion you control before running the model.")

    # Injection controls (only used in --mode inject)
    p.add_argument("--occlusion_type", type=str, default="rectangle",
                   choices=["rectangle", "random_blocks", "moving_temporal",
                            "partial_body", "multi_block"])
    p.add_argument("--start_frame", type=int, default=10,
                   help="First frame index of the injected occlusion event.")
    p.add_argument("--end_frame", type=int, default=20,
                   help="Last frame index (inclusive) of the injected occlusion event.")
    p.add_argument("--target_severity", type=float, default=None,
                   help="Approximate desired severity in [0,1]. If omitted, "
                        "uses default config-driven random sizing.")

    p.add_argument("--det_threshold", type=float, default=0.5)
    p.add_argument("--region_threshold", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clean_frames(folder: str):
    """Load sorted (H, W) uint8 frames from a folder."""
    p = Path(folder)
    paths = sorted(
        f for f in p.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not paths:
        raise RuntimeError(f"No image frames found in {folder}")
    frames = [np.array(Image.open(str(fp)).convert("L"), dtype=np.uint8) for fp in paths]
    return frames, [str(fp) for fp in paths]


def run_model_on_frames(model, frames_uint8, cfg, device, seq_len):
    """
    Run the 4-head model on a list of (H, W) uint8 frames using a sliding
    window pass (pads/truncates to seq_len, matching training-time
    sequence length so the Transformer sees a familiar input length).

    Returns:
        dict with det_prob, det_binary, sev, region_prob, region_binary,
        confidence — all as plain python lists of length len(frames_uint8).
    """
    ft = SilhouetteTransform(
        height=cfg["dataset"]["image_height"],
        width=cfg["dataset"]["image_width"],
        augment=False,
        normalize=True,
    )
    transform = SequenceTransform(ft)

    N = len(frames_uint8)
    T = seq_len

    det_sum    = np.zeros(N, dtype=np.float32)
    sev_sum    = np.zeros(N, dtype=np.float32)
    region_sum = np.zeros((N, 4), dtype=np.float32)
    conf_sum   = np.zeros(N, dtype=np.float32)
    counts     = np.zeros(N, dtype=np.float32)

    starts = list(range(0, max(1, N - T + 1)))
    if N < T:
        starts = [0]

    model.eval()
    for start in starts:
        window = list(frames_uint8[start:start + T])
        while len(window) < T:
            window.append(window[-1])

        frames_tensor = transform(window).unsqueeze(0).to(device)  # (1,T,1,H,W)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model.predict_all(frames_tensor)

        det_prob_w    = pred["det_prob"][0].float().cpu().numpy()
        sev_w         = pred["sev"][0].float().cpu().numpy()
        region_prob_w = pred["region_prob"][0].float().cpu().numpy()  # (T,4)
        conf_w        = pred["confidence"][0].float().cpu().numpy()

        actual_len = min(T, N - start)
        for i in range(actual_len):
            fi = start + i
            det_sum[fi]    += det_prob_w[i]
            sev_sum[fi]    += sev_w[i]
            region_sum[fi] += region_prob_w[i]
            conf_sum[fi]   += conf_w[i]
            counts[fi]     += 1.0

    counts = np.maximum(counts, 1.0)
    det_prob    = (det_sum / counts).tolist()
    sev         = (sev_sum / counts).tolist()
    region_prob = (region_sum / counts[:, None]).tolist()
    confidence  = (conf_sum / counts).tolist()

    return {
        "det_prob":    det_prob,
        "det_binary":  [int(p >= 0.5) for p in det_prob],
        "sev":         sev,
        "region_prob": region_prob,
        "confidence":  confidence,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # ---- Load model -----------------------------------------------------
    heads_are_trained = True
    if os.path.isfile(args.checkpoint):
        model = build_model(cfg, enable_extra_heads=True).to(device)
        state = torch.load(args.checkpoint, map_location=str(device))
        model.load_state_dict(state["model"])
        print(f"Loaded fully fine-tuned 4-head checkpoint: {args.checkpoint}")
    elif args.base_checkpoint and os.path.isfile(args.base_checkpoint):
        model = build_model_with_pretrained_base(
            cfg, args.base_checkpoint, device=str(device)
        ).to(device)
        heads_are_trained = False
        print(
            f"WARNING: {args.checkpoint} not found. Built 4-head model from "
            f"{args.base_checkpoint} instead — region_head and confidence_head "
            f"are RANDOMLY INITIALISED and their output is NOT meaningful yet. "
            f"Run finetune_new_heads.py first for valid region/confidence results."
        )
    else:
        raise FileNotFoundError(
            f"Neither --checkpoint ({args.checkpoint}) nor a valid "
            f"--base_checkpoint was found. Provide one of them."
        )
    model.eval()

    # ---- Load clean sequence frames --------------------------------------
    clean_frames, frame_paths = load_clean_frames(args.sequence)
    N = len(clean_frames)
    print(f"Loaded {N} frames from {args.sequence}")

    # ---- Build ground truth (either all-clean, or inject occlusion) -------
    if args.mode == "clean":
        occluded_frames = [f.copy() for f in clean_frames]
        gt_det    = [0] * N
        gt_sev    = [0.0] * N
        gt_region = [[0, 0, 0, 0]] * N
        print("Mode: clean — no occlusion injected. Ground truth is all-clean.")
    else:
        gen = OcclusionGenerator(cfg.get("occlusion", {}))
        results = gen.apply_controlled_occlusion(
            clean_frames,
            occlusion_type=args.occlusion_type,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            target_severity=args.target_severity,
        )
        occluded_frames = [r.frame for r in results]
        gt_det    = [r.is_occluded for r in results]
        gt_sev    = [r.severity for r in results]
        gt_region = [
            [r.region_upper, r.region_lower, r.region_left, r.region_right]
            for r in results
        ]
        n_occ = sum(gt_det)
        peak_sev = max(gt_sev) if gt_sev else 0.0
        print(
            f"Mode: inject — type={args.occlusion_type}  "
            f"frames=[{args.start_frame},{args.end_frame}]  "
            f"occluded_frames={n_occ}/{N}  peak_severity={peak_sev:.3f}"
        )

    # ---- Run the model ------------------------------------------------------
    seq_len = cfg["dataset"]["sequence_length"]
    pred = run_model_on_frames(model, occluded_frames, cfg, device, seq_len)

    # ---- Console table -------------------------------------------------------
    print("\n" + "=" * 95)
    header = (f"{'Frame':>5}  {'GT_Det':>6}  {'Pred_P':>7}  {'GT_Sev':>6}  {'Pred_Sev':>8}  "
              f"{'GT_U/L/Le/R':>12}  {'Pred_U/L/Le/R':>16}  {'Conf':>5}")
    print(header)
    print("-" * 95)
    for i in range(N):
        gt_r = "".join(str(v) for v in gt_region[i])
        pred_r = "/".join(f"{v:.2f}" for v in pred["region_prob"][i])
        print(
            f"{i:>5}  {gt_det[i]:>6}  {pred['det_prob'][i]:>7.3f}  "
            f"{gt_sev[i]:>6.3f}  {pred['sev'][i]:>8.3f}  "
            f"{gt_r:>12}  {pred_r:>16}  {pred['confidence'][i]:>5.2f}"
        )
    print("=" * 95)

    if not heads_are_trained:
        print(
            "\n*** NOTE: region/confidence columns above are NOT meaningful — "
            "those heads have not been fine-tuned. Run finetune_new_heads.py. ***"
        )

    # ---- Save outputs ---------------------------------------------------------
    out_dir = args.output
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Dashboard
    plot_four_head_dashboard(
        gt_det=gt_det,
        pred_det_prob=pred["det_prob"],
        gt_sev=gt_sev,
        pred_sev=pred["sev"],
        gt_region=gt_region,
        pred_region_prob=pred["region_prob"],
        pred_confidence=pred["confidence"],
        save_path=os.path.join(out_dir, "dashboard.png"),
        title=f"{Path(args.sequence).name} — mode={args.mode}",
    )
    print(f"\nDashboard saved: {out_dir}/dashboard.png")

    # 2. Full per-frame overlays
    save_full_inference_frames(
        frames=occluded_frames,
        gt_det=gt_det,
        pred_det_prob=pred["det_prob"],
        gt_sev=gt_sev,
        pred_sev=pred["sev"],
        gt_region=gt_region,
        pred_region_prob=pred["region_prob"],
        pred_confidence=pred["confidence"],
        save_dir=os.path.join(out_dir, "full_overlay"),
    )
    print(f"Per-frame overlays saved: {out_dir}/full_overlay/")

    # 3. GIFs
    save_gif(clean_frames, os.path.join(out_dir, "clean.gif"), fps=args.fps)
    save_gif(occluded_frames, os.path.join(out_dir, "occluded.gif"), fps=args.fps)
    save_paired_gif(
        clean_frames, occluded_frames,
        os.path.join(out_dir, "paired.gif"), fps=args.fps,
    )
    print(f"GIFs saved: clean.gif, occluded.gif, paired.gif (in {out_dir}/)")

    # 4. CSV
    csv_path = os.path.join(out_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "gt_det", "pred_det_prob", "gt_sev", "pred_sev",
            "gt_upper", "gt_lower", "gt_left", "gt_right",
            "pred_upper", "pred_lower", "pred_left", "pred_right",
            "confidence",
        ])
        for i in range(N):
            writer.writerow([
                i, gt_det[i], f"{pred['det_prob'][i]:.6f}",
                f"{gt_sev[i]:.6f}", f"{pred['sev'][i]:.6f}",
                *gt_region[i],
                *[f"{v:.6f}" for v in pred["region_prob"][i]],
                f"{pred['confidence'][i]:.6f}",
            ])
    print(f"CSV saved: {csv_path}")

    # 5. Summary text
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Sequence: {args.sequence}\n")
        f.write(f"Mode: {args.mode}\n")
        if args.mode == "inject":
            f.write(f"Injected occlusion: {args.occlusion_type}, "
                    f"frames [{args.start_frame},{args.end_frame}], "
                    f"target_severity={args.target_severity}\n")
        f.write(f"Heads trained: {heads_are_trained}\n")
        f.write(f"Frames: {N}\n")
        f.write(f"GT occluded frames: {sum(gt_det)}\n")
        f.write(f"Predicted occluded frames: {sum(pred['det_binary'])}\n")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
