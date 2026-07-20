"""
extract_tumiitgait.py  —  FINAL VERSION
=========================================
Pattern:  {condition_prefix}_id{NNN}_{trial}.avi

e.g.  backpack_id023_3.avi  →  subject=23, cond='bg', trial=3

Output (CASIA-B compatible):
  TUMIITGAIT-sil/
    023/
      bg-03/   0000.png 0001.png ...
      nm-01/
      hp-02/
      cl-01/
      so-01/
      do-01/
    ...
"""

import cv2
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────
VIDEO_ROOT = "/home/himanshu/tumiit/tumiitgait/uploaded_video"
OUT_ROOT   = "/home/himanshu/tumiit/tumiitgait/uploaded_video/extracted_frames"
MODEL_FILE = "yolov8n-seg.pt"
BATCH_SZ   = 32
TARGET_H   = 128
TARGET_W   = 88
# Auto-detect: use CUDA if available, otherwise CPU
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] torch.cuda.is_available()={torch.cuda.is_available()} → using device='{DEVICE}'")

# !! Update these if your find command shows different prefix words !!
PREFIX_TO_CODE = {
    "backpack"      : "bg",
    "normal"        : "nm",
    "hand_in_pocket": "hp",
    "gown"          : "cl",
    "static_occ"    : "so",
    "dynamic_occ"   : "do",
}

# ── Load model once ─────────────────────────────────────────────
print("Loading YOLOv8n-seg...")
yolo = YOLO(MODEL_FILE)
print("Ready.\n")


def parse_filename(fname: str):
    """
    Works for:
      backpack_id023_3.avi
      hand_in_pocket_id023_3.avi
      dynamic_occ_id023_3.avi
      static_occ_id023_3.avi
    """
    stem = Path(fname).stem                       # e.g. 'hand_in_pocket_id023_3'

    # Split on '_id' — everything before is prefix, everything after is NNN_trial
    parts = stem.split("_id")
    if len(parts) != 2:
        return None

    prefix   = parts[0].lower()                  # 'hand_in_pocket'
    rest     = parts[1]                          # '023_3'

    m = re.match(r'^(\d+)_(\d+)$', rest)
    if not m:
        return None

    subj_id = int(m.group(1))                    # 23
    trial   = int(m.group(2))                    # 3

    cond = PREFIX_TO_CODE.get(prefix)
    if cond is None:
        print(f"  WARNING: unknown prefix '{prefix}' in {fname}")
        return None

    return cond, subj_id, trial


def read_avi(path: str):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def segment_frames(frames):
    """
    Batch all frames → YOLOv8n-seg → binary silhouettes.
    For occlusion sequences (so/do) where two people appear,
    we pick the subject using a centredness + area score.
    """
    results = yolo.predict(
        frames,
        classes=[0],      # person only
        imgsz=640,
        conf=0.20,        # slightly low to catch partially occluded subjects
        iou=0.45,
        device=DEVICE,
        verbose=False,
        batch=BATCH_SZ,
    )

    silhouettes = []
    for res in results:
        H_orig, W_orig = res.orig_shape

        if res.masks is None or len(res.masks) == 0:
            silhouettes.append(
                np.zeros((TARGET_H, TARGET_W), dtype=np.uint8))
            continue

        masks  = res.masks.data.cpu().numpy()     # [N, H', W']
        W_half = W_orig / 2.0

        if len(masks) == 1:
            best = 0
        else:
            # Two people in frame (static/dynamic occlusion)
            # Score by area × centredness — subject walks through centre
            scores = []
            for m in masks:
                m_full  = cv2.resize(m, (W_orig, H_orig))
                area    = float(m_full.sum())
                cols    = np.where(m_full > 0.5)[1]
                cx      = float(cols.mean()) if len(cols) > 0 else 0.0
                centre  = 1.0 - abs(cx - W_half) / W_half
                scores.append(area * (0.5 + 0.5 * centre))
            best = int(np.argmax(scores))

        mask     = masks[best]
        mask_big = cv2.resize(mask, (W_orig, H_orig),
                              interpolation=cv2.INTER_LINEAR)
        mask_tgt = cv2.resize(mask_big, (TARGET_W, TARGET_H),
                              interpolation=cv2.INTER_LINEAR)
        sil      = (mask_tgt > 0.5).astype(np.uint8) * 255
        silhouettes.append(sil)

    return silhouettes


# ── Main ────────────────────────────────────────────────────────
all_avis = sorted(Path(VIDEO_ROOT).rglob("*.avi"))
print(f"Total .avi files found: {len(all_avis)}\n")

done, skipped, failed = 0, 0, []

for avi_path in tqdm(all_avis, desc="Extracting"):
    parsed = parse_filename(avi_path.name)
    if parsed is None:
        skipped += 1
        continue

    cond, subj_id, trial = parsed
    out_dir = Path(OUT_ROOT) / f"{subj_id:03d}" / f"{cond}-{trial:02d}"

    if out_dir.exists() and len(list(out_dir.glob("*.png"))) > 5:
        skipped += 1
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_avi(str(avi_path))

    if not frames:
        failed.append(avi_path.name)
        continue

    try:
        sils = segment_frames(frames)
    except Exception as e:
        print(f"\n  ERROR {avi_path.name}: {e}")
        failed.append(avi_path.name)
        continue

    for i, sil in enumerate(sils):
        cv2.imwrite(str(out_dir / f"{i:04d}.png"), sil)

    done += 1

print(f"\n{'='*50}")
print(f"  Done      : {done}")
print(f"  Skipped   : {skipped}")
print(f"  Failed    : {len(failed)}")
if failed:
    print("\nFailed files:")
    for f in failed:
        print(f"  {f}")