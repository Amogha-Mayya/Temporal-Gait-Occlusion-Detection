"""
utils/visualization.py
----------------------
Visualization helpers for gait occlusion detection:

1. side_by_side_frames  — save a grid comparing clean vs. occluded frames
2. save_gif             — save a list of frames as an animated GIF
3. save_labeled_frames  — overlay detection + severity text on frames
4. plot_temporal        — plot ground-truth vs. prediction timelines
5. annotate_frame       — low-level PIL text overlay on a single frame
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or arbitrary range array to uint8 [0,255]."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255)
    return arr.clip(0, 255).astype(np.uint8)


def _load_font(size: int = 14):
    """Try to load a truetype font; fall back to PIL default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 1. Side-by-side frame comparison grid
# ---------------------------------------------------------------------------

def side_by_side_frames(
    clean_frames: List[np.ndarray],
    occluded_frames: List[np.ndarray],
    det_labels: List[int],
    sev_labels: List[float],
    save_path: str,
    n_cols: int = 10,
) -> None:
    """
    Save a grid image with clean frames on top, occluded on bottom.

    Args:
        clean_frames:    List of (H, W) uint8 arrays.
        occluded_frames: List of (H, W) uint8 arrays.
        det_labels:      Per-frame detection labels.
        sev_labels:      Per-frame severity values.
        save_path:       Output PNG path.
        n_cols:          Number of frame columns in the grid.
    """
    T = len(clean_frames)
    n_rows = 2   # clean / occluded

    fig, axes = plt.subplots(
        n_rows, min(T, n_cols),
        figsize=(min(T, n_cols) * 1.5, n_rows * 2.5 + 0.5),
    )
    if min(T, n_cols) == 1:
        axes = axes.reshape(n_rows, 1)

    for col_idx in range(min(T, n_cols)):
        # Row 0: clean
        ax = axes[0, col_idx]
        ax.imshow(_to_uint8(clean_frames[col_idx]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"F{col_idx}", fontsize=7)
        ax.axis("off")

        # Row 1: occluded with label
        ax = axes[1, col_idx]
        ax.imshow(_to_uint8(occluded_frames[col_idx]), cmap="gray", vmin=0, vmax=255)
        lbl = det_labels[col_idx]
        sev = sev_labels[col_idx]
        color = "red" if lbl == 1 else "green"
        ax.set_title(f"D={lbl}\nS={sev:.2f}", fontsize=6, color=color)
        ax.axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=8, rotation=90, va="center")
    axes[1, 0].set_ylabel("Occluded", fontsize=8, rotation=90, va="center")

    plt.suptitle("Synthetic Occlusion Comparison", fontsize=10, y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 2. GIF export
# ---------------------------------------------------------------------------

def save_gif(
    frames: List[np.ndarray],
    save_path: str,
    fps: int = 8,
) -> None:
    """
    Save a list of (H, W) uint8 frames as an animated GIF.

    Args:
        frames:    List of (H, W) grayscale numpy arrays.
        save_path: Output .gif path.
        fps:       Frames per second.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [
        np.stack([_to_uint8(f)] * 3, axis=-1)  # grayscale → RGB
        for f in frames
    ]
    imageio.mimsave(save_path, rgb_frames, fps=fps)


# ---------------------------------------------------------------------------
# 3. Annotated frames (text overlay)
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    text: str,
    color: str = "red",
    font_size: int = 12,
) -> np.ndarray:
    """
    Draw text onto a grayscale frame and return an RGB uint8 array.

    Args:
        frame:     (H, W) uint8 or float array.
        text:      Multi-line text to overlay.
        color:     PIL colour string.
        font_size: Font size in pixels.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    img_u8 = _to_uint8(frame)
    rgb = np.stack([img_u8] * 3, axis=-1)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)
    draw.text((4, 4), text, fill=color, font=font)
    return np.array(pil)


def save_labeled_frames(
    frames: List[np.ndarray],
    det_labels: List[int],
    sev_labels: List[float],
    save_dir: str,
    prefix: str = "frame",
) -> None:
    """
    Save each frame with detection + severity text overlay.

    Args:
        frames:     List of (H, W) uint8 arrays.
        det_labels: Per-frame detection labels.
        sev_labels: Per-frame severity values.
        save_dir:   Directory to write frames into.
        prefix:     Filename prefix.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, (f, d, s) in enumerate(zip(frames, det_labels, sev_labels)):
        status = "Occluded" if d == 1 else "Clean"
        text   = f"Frame {i:03d}\n{status}\nSeverity={s:.2f}"
        color  = "red" if d == 1 else "lime"
        annotated = annotate_frame(f, text, color=color, font_size=11)
        out_path  = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        Image.fromarray(annotated).save(out_path)


# ---------------------------------------------------------------------------
# 4. Temporal prediction plot
# ---------------------------------------------------------------------------

def plot_temporal(
    gt_det: List[int],
    pred_det: List[float],    # probabilities or binary
    gt_sev: List[float],
    pred_sev: List[float],
    save_path: str,
    title: str = "Temporal Occlusion Detection",
) -> None:
    """
    Plot ground-truth vs. predicted detection and severity over time.

    Args:
        gt_det:    Ground-truth detection labels (0/1) per frame.
        pred_det:  Predicted detection probabilities per frame.
        gt_sev:    Ground-truth severity per frame.
        pred_sev:  Predicted severity per frame.
        save_path: Output PNG path.
        title:     Figure title.
    """
    T = len(gt_det)
    frames = list(range(T))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # ---- Detection --------------------------------------------------------
    ax = axes[0]
    ax.fill_between(frames, gt_det, alpha=0.25, color="steelblue", label="GT Detection")
    ax.plot(frames, gt_det,  "b-",  lw=1.5,  label="GT")
    ax.plot(frames, pred_det, "r--", lw=1.5,  label="Pred (prob)")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Detection")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # ---- Severity ---------------------------------------------------------
    ax = axes[1]
    ax.fill_between(frames, gt_sev, alpha=0.25, color="darkorange", label="GT Severity")
    ax.plot(frames, gt_sev,  "orange", lw=1.5, label="GT")
    ax.plot(frames, pred_sev, "r--",  lw=1.5,  label="Pred")
    ax.set_ylabel("Severity")
    ax.set_xlabel("Frame Index")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Inference result overlay saver
# ---------------------------------------------------------------------------

def save_inference_frames(
    frames: List[np.ndarray],
    det_probs: List[float],
    det_binary: List[int],
    sev_preds: List[float],
    save_dir: str,
) -> None:
    """
    Save inference-annotated frames to disk.

    Args:
        frames:     List of raw (H, W) uint8 silhouette frames.
        det_probs:  Per-frame detection probabilities.
        det_binary: Per-frame binary prediction.
        sev_preds:  Per-frame severity predictions.
        save_dir:   Output directory.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, (f, p, d, s) in enumerate(
        zip(frames, det_probs, det_binary, sev_preds)
    ):
        status = "Occluded" if d == 1 else "Clean"
        text   = f"Frame {i:03d}\n{status} ({p:.2f})\nSeverity={s:.2f}"
        color  = "red" if d == 1 else "lime"
        annotated = annotate_frame(f, text, color=color, font_size=11)
        out_path  = os.path.join(save_dir, f"pred_{i:03d}.png")
        Image.fromarray(annotated).save(out_path)
