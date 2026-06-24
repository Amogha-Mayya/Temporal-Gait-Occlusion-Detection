"""
utils/confidence.py
-------------------
Construction of the per-frame CONFIDENCE TARGET used to train ConfidenceHead.

WHY A CONSTRUCTED TARGET IS NEEDED
-----------------------------------
"Confidence" has no ground-truth label in the dataset the way detection
(0/1) or severity ([0,1] pixel ratio) do — there is no sensor that tells us
"the model should be 73% confident here." So we must DEFINE what confidence
means in a way that is:
  (a) computable from quantities we already have at train time
      (ground-truth labels + the model's own predictions), and
  (b) meaningful at inference time, where ground truth is NOT available.

DEFINITION USED HERE
---------------------
confidence_target[t] = 1 - calibration_error[t]

where calibration_error combines two signals:

  1. DETECTION CALIBRATION ERROR
     |det_prob[t] - det_label[t]|
     If the frozen detection head outputs a probability close to the true
     label (near 0 or near 1, on the correct side), calibration error is
     low → confidence should be high. If the model is unsure (prob ~0.5)
     calibration error is high regardless of which way the true label
     falls → confidence should be low.

  2. SEVERITY CALIBRATION ERROR (only relevant on occluded frames)
     |sev_pred[t] - sev_label[t]|
     How far off the frozen severity head's estimate is from the true
     pixel-ratio severity.

  3. TEMPORAL CONSISTENCY (penalises abrupt, noisy flip-flopping)
     |det_prob[t] - det_prob[t-1]| averaged with the next-frame difference.
     A frame whose prediction wildly disagrees with its temporal neighbours
     (while ground truth changes smoothly, since occlusion events span
     multiple consecutive frames by construction) is treated as LESS
     reliable, since the model's view of that frame is internally
     inconsistent with the rest of the clip.

This is therefore a CALIBRATION target, not a free-floating uncertainty
estimate — it is grounded in (a) how correct the model's existing
detection/severity outputs were on the frame, and (b) how internally
self-consistent those outputs are across time. The new ConfidenceHead's
job is to learn to PREDICT this quantity directly from the transformer
features, so that AT INFERENCE TIME (without access to ground truth) it
can flag frames where the model is likely to be wrong — exactly the
"how much should I trust the detection/severity outputs for this frame"
signal you asked for.
"""

from __future__ import annotations

import torch


def compute_confidence_targets(
    det_prob: torch.Tensor,     # (B, T) detection probability from FROZEN det_head
    det_labels: torch.Tensor,   # (B, T) ground-truth 0/1
    sev_preds: torch.Tensor,    # (B, T) severity prediction from FROZEN sev_head
    sev_labels: torch.Tensor,   # (B, T) ground-truth severity [0,1]
    temporal_weight: float = 0.3,
    detection_weight: float = 0.4,
    severity_weight: float = 0.3,
) -> torch.Tensor:
    """
    Build the per-frame confidence training target.

    Args:
        det_prob:         (B, T) sigmoid output of the (frozen) detection head.
        det_labels:       (B, T) ground-truth binary detection labels.
        sev_preds:        (B, T) output of the (frozen) severity head.
        sev_labels:        (B, T) ground-truth severity values.
        temporal_weight:  Weight for the temporal-consistency term.
        detection_weight: Weight for the detection-calibration term.
        severity_weight:  Weight for the severity-calibration term.
        (weights should sum to 1.0 for the target to stay in [0,1])

    Returns:
        confidence_target: (B, T) float32 in [0, 1]
    """
    assert abs(temporal_weight + detection_weight + severity_weight - 1.0) < 1e-6, \
        "Confidence component weights must sum to 1.0"

    B, T = det_prob.shape

    # ---- 1. Detection calibration error --------------------------------
    det_error = (det_prob - det_labels.float()).abs()             # (B, T) in [0,1]

    # ---- 2. Severity calibration error (only meaningful where occluded) --
    sev_error = (sev_preds - sev_labels).abs().clamp(0, 1)        # (B, T) in [0,1]
    # On clean frames severity error is naturally tiny since both sides
    # are close to 0; no special masking required.

    # ---- 3. Temporal consistency error -----------------------------------
    # Compare each frame's det_prob to its neighbours; edge frames use
    # only the one neighbour they have.
    prev_diff = torch.zeros_like(det_prob)
    next_diff = torch.zeros_like(det_prob)

    prev_diff[:, 1:] = (det_prob[:, 1:] - det_prob[:, :-1]).abs()
    next_diff[:, :-1] = (det_prob[:, :-1] - det_prob[:, 1:]).abs()

    # Average available neighbour diffs (first/last frame only has 1 side)
    neighbour_count = torch.ones_like(det_prob) * 2
    neighbour_count[:, 0] = 1
    neighbour_count[:, -1] = 1
    temporal_error = (prev_diff + next_diff) / neighbour_count

    # ---- Combine into a single calibration error, then invert -----------
    calibration_error = (
        detection_weight * det_error
        + severity_weight * sev_error
        + temporal_weight * temporal_error
    ).clamp(0, 1)

    confidence_target = 1.0 - calibration_error
    return confidence_target
