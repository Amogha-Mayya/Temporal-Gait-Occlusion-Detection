#!/usr/bin/env python
"""
visualize_occlusion.py
----------------------
Visualize synthetic occlusion generation on CASIA-B sequences.

Outputs:
  - Side-by-side comparison grid (PNG)
  - Clean sequence GIF
  - Occluded sequence GIF
  - Labeled annotated frames (PNG per frame)
  - Temporal label plot (PNG)

Usage:
    # Random sample from the dataset
    python visualize_occlusion.py --config configs/config.yaml

    # Specific subject / condition
    python visualize_occlusion.py \\
        --config configs/config.yaml \\
        --subject 1 --condition nm-01 \\
        --output outputs/visualizations/subj001_nm01
"""

import argparse
import os
import random
from pathlib import Path

import yaml

from data.casia_dataset import CasiaGaitDataset
from data.subject_split import build_subject_split, get_sequences_for_subjects
from data.transforms import build_transforms
from utils.seed import set_seed
from utils.visualization import (
    plot_temporal,
    save_gif,
    save_labeled_frames,
    side_by_side_frames,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize synthetic gait occlusion on CASIA-B sequences."
    )
    p.add_argument("--config",    type=str, default="configs/config.yaml")
    p.add_argument("--output",    type=str, default="outputs/visualizations/occlusion_demo",
                   help="Output directory for visualization files.")
    p.add_argument("--subject",   type=int, default=None,
                   help="Subject ID to visualize (default: random from train set).")
    p.add_argument("--condition", type=str, default=None,
                   help="Condition name, e.g. nm-01, bg-01, cl-01 (default: random).")
    p.add_argument("--index",     type=int, default=None,
                   help="Dataset index to load (overrides --subject/--condition).")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--fps",       type=int, default=6,
                   help="Frames per second for GIF output.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = args.output
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build dataset (train split, occlusion enabled) --------------------
    transform = build_transforms(cfg, split="train")
    dataset   = CasiaGaitDataset(
        dataset_root=cfg["dataset"]["path"],
        subjects=build_subject_split(
            cfg["dataset"]["path"],
            cfg["split"]["train_end"],
            cfg["split"]["val_end"],
            cfg["split"]["test_end"],
        )["train"],
        sequence_length=cfg["dataset"]["sequence_length"],
        stride=cfg["dataset"]["stride"],
        min_frames=cfg["dataset"]["min_frames"],
        occlusion_cfg=cfg.get("occlusion", {}),
        transform=None,          # We'll get raw arrays for visualization
        occlusion_enabled=True,
    )

    n = len(dataset)
    if n == 0:
        print("ERROR: Dataset index is empty. Check dataset path in config.yaml.")
        return

    # ---- Choose index -------------------------------------------------------
    if args.index is not None:
        idx = args.index % n
    else:
        # Try to find requested subject/condition
        idx = None
        if args.subject is not None:
            for i, (fp_list, _) in enumerate(dataset.index):
                # fp_list[0] is something like .../001/nm-01/0001.png
                parts = Path(fp_list[0]).parts
                subj_name = f"{args.subject:03d}"
                if subj_name in parts:
                    if args.condition is None or args.condition in parts:
                        idx = i
                        break
        if idx is None:
            idx = random.randint(0, n - 1)

    print(f"Visualizing dataset index {idx} (of {n})")
    sample_path = dataset.index[idx][0][0]
    print(f"  Sequence: {Path(sample_path).parent}")

    # ---- Load clean and occluded sequences ---------------------------------
    clean_frames = dataset.load_clean_sequence(idx)
    occ_frames, det_labels, sev_labels, region_labels = dataset.load_occluded_sequence(idx)
    T = len(clean_frames)
    print(f"  Frames: {T}  |  Occluded: {sum(det_labels)}/{T}")

    # ---- 1. Side-by-side grid ---------------------------------------------
    grid_path = os.path.join(out_dir, "comparison_grid.png")
    side_by_side_frames(
        clean_frames, occ_frames, det_labels, sev_labels,
        save_path=grid_path, n_cols=10,
    )
    print(f"Grid saved:    {grid_path}")

    # ---- 2. GIFs -----------------------------------------------------------
    clean_gif_path = os.path.join(out_dir, "clean.gif")
    occ_gif_path   = os.path.join(out_dir, "occluded.gif")
    save_gif(clean_frames, clean_gif_path, fps=args.fps)
    save_gif(occ_frames,   occ_gif_path,   fps=args.fps)
    print(f"Clean GIF:     {clean_gif_path}")
    print(f"Occluded GIF:  {occ_gif_path}")

    # ---- 3. Labeled frames -------------------------------------------------
    labeled_dir = os.path.join(out_dir, "labeled_frames")
    save_labeled_frames(
        occ_frames, det_labels, sev_labels,
        save_dir=labeled_dir, prefix="frame",
    )
    print(f"Labeled frames: {labeled_dir}/")

    # ---- 4. Temporal label plot --------------------------------------------
    plot_path = os.path.join(out_dir, "label_timeline.png")
    plot_temporal(
        gt_det=det_labels,
        pred_det=[float(d) for d in det_labels],   # GT as "pred" for ground-truth-only view
        gt_sev=sev_labels,
        pred_sev=sev_labels,
        save_path=plot_path,
        title=f"Ground-Truth Labels — Index {idx}",
    )
    print(f"Label plot:    {plot_path}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
