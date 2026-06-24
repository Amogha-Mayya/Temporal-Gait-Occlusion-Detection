"""
utils/visualization.py
----------------------
Visualization helpers for gait occlusion detection:

1. side_by_side_frames     — save a grid comparing clean vs. occluded frames
2. save_gif                — save a list of frames as an animated GIF
3. save_labeled_frames     — overlay detection + severity text on frames
4. plot_temporal           — plot ground-truth vs. prediction timelines (2-head)
5. annotate_frame          — low-level PIL text overlay on a single frame
6. plot_four_head_dashboard— GT vs predicted dashboard for all 4 heads
7. annotate_frame_full     — rich per-frame overlay: GT vs pred, all 4 heads,
                             plus quadrant-region visual guide drawn on image
8. save_paired_gif         — single GIF with clean | occluded side by side
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
# 6. Full 4-head dashboard plot (GT vs Predicted for all heads)
# ---------------------------------------------------------------------------

def plot_four_head_dashboard(
    gt_det: List[int],
    pred_det_prob: List[float],
    gt_sev: List[float],
    pred_sev: List[float],
    gt_region: List[List[int]],          # T x 4  [upper, lower, left, right]
    pred_region_prob: List[List[float]], # T x 4
    pred_confidence: List[float],
    save_path: str,
    title: str = "4-Head Occlusion Analysis",
) -> None:
    """
    Render a single dashboard figure comparing ground truth vs. predictions
    for ALL FOUR heads: Detection, Severity, Region (4 quadrants), Confidence.

    This is the central visualization for the "inject occlusion, then see
    how every head responds" workflow: since the occlusion was injected by
    us, gt_* arrays are KNOWN exactly, so every head's prediction can be
    checked against ground truth (except confidence, which has no fixed
    ground truth — see the dedicated panel note below).

    Args:
        gt_det:            (T,) ground-truth detection labels.
        pred_det_prob:      (T,) predicted detection probabilities.
        gt_sev:            (T,) ground-truth severity values.
        pred_sev:          (T,) predicted severity values.
        gt_region:         (T, 4) ground-truth region labels.
        pred_region_prob:  (T, 4) predicted region probabilities.
        pred_confidence:   (T,) predicted confidence values.
        save_path:         Output PNG path.
        title:             Figure title.
    """
    T = len(gt_det)
    frames = list(range(T))
    region_names = ["Upper", "Lower", "Left", "Right"]
    region_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # ---- Panel 1: Detection -------------------------------------------------
    ax = axes[0]
    ax.fill_between(frames, gt_det, alpha=0.20, color="steelblue", step="mid", label="GT (occluded)")
    ax.plot(frames, pred_det_prob, "r-", lw=1.6, label="Predicted probability")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Detection")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Severity ---------------------------------------------------
    ax = axes[1]
    ax.fill_between(frames, gt_sev, alpha=0.20, color="darkorange", step="mid", label="GT severity")
    ax.plot(frames, pred_sev, "r-", lw=1.6, label="Predicted severity")
    ax.set_ylabel("Severity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Region (4 quadrants) ---------------------------------------
    ax = axes[2]
    gt_region_arr   = np.array(gt_region)         # (T, 4)
    pred_region_arr = np.array(pred_region_prob)  # (T, 4)
    for i, (name, color) in enumerate(zip(region_names, region_colors)):
        ax.plot(frames, pred_region_arr[:, i], color=color, lw=1.3,
                label=f"{name} (pred)")
        # Mark GT-positive frames with a small tick at the top of that region's band
        gt_positive_frames = [f for f in frames if gt_region_arr[f, i] == 1]
        if gt_positive_frames:
            ax.scatter(gt_positive_frames, [1.0 + 0.03 * i] * len(gt_positive_frames),
                       color=color, marker="|", s=40)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Region prob.")
    ax.set_ylim(-0.05, 1.20)
    ax.legend(fontsize=7, loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.text(0.01, 1.14, "Tick marks (top) = ground-truth occluded region",
            transform=ax.get_yaxis_transform(), fontsize=7, color="gray")

    # ---- Panel 4: Confidence ---------------------------------------------------
    ax = axes[3]
    ax.plot(frames, pred_confidence, color="purple", lw=1.6, label="Predicted confidence")
    ax.fill_between(frames, pred_confidence, alpha=0.15, color="purple")
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Frame Index")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.text(0.01, -0.22,
            "Note: confidence has no fixed ground truth — it is a learned\n"
            "calibration signal (low = model likely unreliable on this frame).",
            transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Rich per-frame overlay (all 4 heads + GT, for the "inject and inspect" flow)
# ---------------------------------------------------------------------------

def annotate_frame_full(
    frame: np.ndarray,
    gt_det: int,
    pred_det_prob: float,
    gt_sev: float,
    pred_sev: float,
    gt_region: List[int],          # [upper, lower, left, right]
    pred_region_prob: List[float], # [upper, lower, left, right]
    pred_confidence: float,
    frame_idx: int,
) -> np.ndarray:
    """
    Draw a rich text overlay showing GROUND TRUTH vs PREDICTED for all four
    heads on a single frame, plus a region-quadrant visual guide drawn
    directly on the image (colored quadrant borders) so you can see at a
    glance which part of the body the synthetic occlusion (or the
    prediction) is flagging.

    Args:
        frame:             (H, W) uint8 silhouette (occluded version).
        gt_det:             0/1 ground truth.
        pred_det_prob:      predicted probability.
        gt_sev:             ground truth severity.
        pred_sev:           predicted severity.
        gt_region:          4 ground-truth region labels.
        pred_region_prob:   4 predicted region probabilities.
        pred_confidence:    predicted confidence value.
        frame_idx:          frame number (for the header).

    Returns:
        (H, W, 3) uint8 RGB annotated image, upscaled 3x for legibility.
    """
    img_u8 = _to_uint8(frame)
    H, W = img_u8.shape
    scale = 3
    rgb = np.stack([img_u8] * 3, axis=-1)
    pil = Image.fromarray(rgb).resize((W * scale, H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(pil)
    font      = _load_font(13)
    small_font = _load_font(11)

    mid_h = (H * scale) // 2
    mid_w = (W * scale) // 2
    region_names = ["upper", "lower", "left", "right"]
    region_colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]

    # ---- Draw quadrant boundary guide lines (faint) -----------------------
    draw.line([(0, mid_h), (W * scale, mid_h)], fill=(90, 90, 90), width=1)
    draw.line([(mid_w, 0), (mid_w, H * scale)], fill=(90, 90, 90), width=1)

    # ---- Highlight predicted-positive regions with colored border ----------
    region_boxes = {
        "upper": (0, 0, W * scale, mid_h),
        "lower": (0, mid_h, W * scale, H * scale),
        "left":  (0, 0, mid_w, H * scale),
        "right": (mid_w, 0, W * scale, H * scale),
    }
    for name, color in zip(region_names, region_colors):
        idx = region_names.index(name)
        if pred_region_prob[idx] >= 0.5:
            box = region_boxes[name]
            draw.rectangle(box, outline=color, width=2)

    # ---- Header: detection + severity, GT vs Pred -------------------------
    det_status = "OCCLUDED" if gt_det == 1 else "CLEAN"
    det_color  = (255, 60, 60) if gt_det == 1 else (60, 220, 60)
    header = (
        f"Frame {frame_idx:03d}\n"
        f"GT: {det_status}  Sev={gt_sev:.2f}\n"
        f"Pred: P(occ)={pred_det_prob:.2f}  Sev={pred_sev:.2f}\n"
        f"Conf={pred_confidence:.2f}"
    )
    draw.rectangle([(2, 2), (W * scale - 2, 62)], fill=(0, 0, 0))
    draw.text((6, 4), header, fill=det_color, font=font)

    # ---- Footer: region GT vs Pred -----------------------------------------
    region_lines = []
    for name, gt_val, pred_val in zip(region_names, gt_region, pred_region_prob):
        flag = "Y" if gt_val == 1 else "n"
        region_lines.append(f"{name[0].upper()}:{flag}/{pred_val:.2f}")
    footer = "  ".join(region_lines)
    draw.rectangle([(2, H * scale - 20), (W * scale - 2, H * scale - 2)], fill=(0, 0, 0))
    draw.text((6, H * scale - 18), footer, fill=(220, 220, 220), font=small_font)

    return np.array(pil)


def save_full_inference_frames(
    frames: List[np.ndarray],
    gt_det: List[int],
    pred_det_prob: List[float],
    gt_sev: List[float],
    pred_sev: List[float],
    gt_region: List[List[int]],
    pred_region_prob: List[List[float]],
    pred_confidence: List[float],
    save_dir: str,
) -> None:
    """
    Save the full rich GT-vs-predicted overlay (annotate_frame_full) for
    every frame in a sequence.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    T = len(frames)
    for i in range(T):
        annotated = annotate_frame_full(
            frame=frames[i],
            gt_det=gt_det[i],
            pred_det_prob=pred_det_prob[i],
            gt_sev=gt_sev[i],
            pred_sev=pred_sev[i],
            gt_region=gt_region[i],
            pred_region_prob=pred_region_prob[i],
            pred_confidence=pred_confidence[i],
            frame_idx=i,
        )
        Image.fromarray(annotated).save(os.path.join(save_dir, f"full_{i:03d}.png"))


# ---------------------------------------------------------------------------
# 8. Side-by-side clean vs. injected-occlusion GIF (paired, for easy compare)
# ---------------------------------------------------------------------------

def save_paired_gif(
    clean_frames: List[np.ndarray],
    occluded_frames: List[np.ndarray],
    save_path: str,
    fps: int = 8,
) -> None:
    """
    Save a single GIF where each frame shows clean (left) and occluded
    (right) side by side, separated by a thin divider.

    Args:
        clean_frames:    List of (H, W) uint8 arrays.
        occluded_frames: List of (H, W) uint8 arrays (same length).
        save_path:       Output .gif path.
        fps:             Frames per second.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    paired = []
    for c, o in zip(clean_frames, occluded_frames):
        c_u8 = _to_uint8(c)
        o_u8 = _to_uint8(o)
        H, W = c_u8.shape
        divider = np.full((H, 2), 128, dtype=np.uint8)
        combined = np.concatenate([c_u8, divider, o_u8], axis=1)
        rgb = np.stack([combined] * 3, axis=-1)
        paired.append(rgb)
    imageio.mimsave(save_path, paired, fps=fps)


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
