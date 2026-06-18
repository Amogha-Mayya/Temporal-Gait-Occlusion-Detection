"""
engine/inference_engine.py
--------------------------
Inference on arbitrary sequence folders (not necessarily from CASIA dataset).
Loads frames → applies transform → runs model → returns per-frame predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from data.transforms import SilhouetteTransform, SequenceTransform


class InferenceEngine:
    """
    Performs frame-by-frame occlusion detection on an input sequence.

    Args:
        model:     Loaded OcclusionDetectionModel (eval mode).
        cfg:       Full config dict.
        device:    torch.device.
        det_threshold: Binary detection threshold (default 0.5).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict,
        device: torch.device,
        det_threshold: float = 0.5,
    ):
        self.model         = model
        self.device        = device
        self.det_threshold = det_threshold
        self.seq_len       = cfg["dataset"]["sequence_length"]

        # Transform (no augmentation, with normalisation)
        ft = SilhouetteTransform(
            height=cfg["dataset"]["image_height"],
            width=cfg["dataset"]["image_width"],
            augment=False,
            normalize=True,
        )
        self.transform = SequenceTransform(ft)

    # ------------------------------------------------------------------
    # Load frames from a folder
    # ------------------------------------------------------------------

    @staticmethod
    def load_frames_from_folder(folder: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Return sorted list of (H, W) uint8 numpy frames from a folder.
        Also returns the list of paths for reference.
        """
        p = Path(folder)
        paths = sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        frames = []
        for fp in paths:
            img = Image.open(str(fp)).convert("L")
            frames.append(np.array(img, dtype=np.uint8))
        return frames, [str(fp) for fp in paths]

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        sequence_folder: str,
    ) -> Dict:
        """
        Run inference on all frames in a folder.  Frames are processed in
        sliding windows of length T (with stride 1 at inference time so
        every frame gets a prediction), then results are averaged over
        windows.

        Returns:
            dict with:
                'frames'      : raw numpy frames
                'det_prob'    : per-frame detection probability (averaged)
                'det_binary'  : per-frame binary detection
                'sev'         : per-frame severity estimate (averaged)
                'frame_paths' : list of input frame paths
        """
        self.model.eval()

        raw_frames, frame_paths = self.load_frames_from_folder(sequence_folder)
        N = len(raw_frames)

        if N == 0:
            raise RuntimeError(f"No image frames found in {sequence_folder}")

        T = self.seq_len

        # Accumulate per-frame predictions (multiple windows may cover same frame)
        det_sum = np.zeros(N, dtype=np.float32)
        sev_sum = np.zeros(N, dtype=np.float32)
        counts  = np.zeros(N, dtype=np.float32)

        # Sliding window with stride 1 at inference
        starts = list(range(0, max(1, N - T + 1)))
        if N < T:
            # Pad last window with repeated last frame
            starts = [0]

        for start in starts:
            end = start + T
            window_frames = raw_frames[start:end]

            # Pad if needed (sequence shorter than T)
            while len(window_frames) < T:
                window_frames.append(window_frames[-1])

            # Transform
            frames_tensor = self.transform(window_frames)  # (T, 1, H, W)
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # (1, T, 1, H, W)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                det_logits, sev_preds = self.model(frames_tensor)

            det_prob_w = torch.sigmoid(det_logits[0]).cpu().numpy()  # (T,)
            sev_pred_w = sev_preds[0].cpu().numpy()                  # (T,)

            # Write into accumulation arrays
            actual_len = min(T, N - start)
            for i in range(actual_len):
                fi = start + i
                det_sum[fi] += det_prob_w[i]
                sev_sum[fi] += sev_pred_w[i]
                counts[fi]  += 1.0

        # Average over overlapping windows
        counts = np.maximum(counts, 1.0)
        det_prob_avg = det_sum / counts
        sev_avg      = sev_sum / counts

        det_binary = (det_prob_avg >= self.det_threshold).astype(int)

        return {
            "frames":      raw_frames,
            "det_prob":    det_prob_avg.tolist(),
            "det_binary":  det_binary.tolist(),
            "sev":         sev_avg.tolist(),
            "frame_paths": frame_paths,
        }
