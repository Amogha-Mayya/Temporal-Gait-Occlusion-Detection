# Temporal Gait Occlusion Detection on CASIA-B

> **Research-grade PyTorch project for frame-level occlusion detection and severity estimation in gait silhouette sequences.**
> Designed for M.Tech thesis and academic research use.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Environment Setup](#5-environment-setup)
6. [Configuration](#6-configuration)
7. [Training](#7-training)
8. [Validation](#8-validation)
9. [Testing](#9-testing)
10. [Inference](#10-inference)
11. [Visualization](#11-visualization)
12. [TensorBoard](#12-tensorboard)
12.5. [Upgrade: 4-Head Model (Region + Confidence)](#125-upgrade-4-head-model-region--confidence-and-occlusion-injection-inference)
13. [Expected Outputs](#13-expected-outputs)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

This project trains a deep neural network that watches a **sequence of silhouette frames** from a CASIA-B gait video and predicts, for every frame:

| Output | Type | Range | Loss |
|---|---|---|---|
| **Occlusion Detection** | Binary classification | {0, 1} | BCEWithLogitsLoss |
| **Occlusion Severity** | Regression | [0, 1] | SmoothL1Loss |

**Key features:**
- Synthetic occlusions generated **on-the-fly in memory** (original dataset never modified)
- Five occlusion types: rectangle, random blocks, moving temporal, partial body, multi-block
- **Temporal coherence**: occlusion events span multiple consecutive frames, not independent per-frame drops
- CNN backbone (ResNet-18/34/50) extracts per-frame features
- Transformer Encoder models temporal context across the sequence
- Mixed precision training (AMP), cosine LR schedule, early stopping
- Full TensorBoard logging and visualization suite

---

## 2. Architecture

```
Input Sequence: (B, T, 1, H, W)
         │
         ▼ ─────────────────────────────────────────────────────────
   CNN Backbone                   ResNet-18 / 34 / 50
   (frame-by-frame)               Modified for 1-channel input
                                  Output: (B*T, 512) or (B*T, 2048)
         │  reshape to (B, T, D_cnn)
         ▼ ─────────────────────────────────────────────────────────
   Input Projection               Linear(D_cnn → hidden_dim)
   Positional Encoding            Learnable embedding per frame position
         │
         ▼ ─────────────────────────────────────────────────────────
   Temporal Transformer           TransformerEncoder
   (temporal context)             num_layers=4, num_heads=8
                                  Pre-LN for training stability
                                  Output: (B, T, hidden_dim)
         │
         ├──────────────────────────┐
         ▼                          ▼
   Detection Head             Severity Head
   Linear(D→128)→GELU         Linear(D→128)→GELU
   Linear(128→1)              Linear(128→1)→Sigmoid
   BCEWithLogitsLoss          SmoothL1Loss
   Output: (B,T) logits       Output: (B,T) ∈ (0,1)

Total Loss = λ_det × det_loss + λ_sev × sev_loss
```

### Label Generation

```
Severity = occluded_silhouette_pixels / total_silhouette_pixels

Where silhouette pixels = pixels with value > 0 in the binary mask.
```

---

## 3. Folder Structure

```
gait_occlusion/
│
├── configs/
│   └── config.yaml             ← All hyperparameters and paths
│
├── data/
│   ├── casia_dataset.py        ← PyTorch Dataset + index builder
│   ├── occlusion_generator.py  ← In-memory synthetic occlusion engine
│   ├── subject_split.py        ← Train/val/test split by subject ID
│   └── transforms.py           ← Resize + normalize + optional flip
│
├── models/
│   ├── backbone.py             ← ResNet adapted for 1-channel input
│   ├── transformer.py          ← Temporal TransformerEncoder
│   ├── heads.py                ← Detection + Severity + Region + Confidence heads
│   └── model.py                ← Full model assembly + loss (2-head and 4-head)
│
├── engine/
│   ├── trainer.py              ← Training loop (AMP, grad clip)
│   ├── evaluator.py            ← Validation / test evaluation
│   └── inference_engine.py     ← Sliding-window inference
│
├── utils/
│   ├── metrics.py              ← Acc/P/R/F1/MAE/RMSE accumulators (+ region/conf)
│   ├── confidence.py           ← Confidence target construction (calibration)
│   ├── logger.py                ← File + console + TensorBoard logger
│   ├── visualization.py        ← GIF, grid, plot, annotation, 4-head dashboard
│   ├── seed.py                 ← Reproducible seed setter
│   └── checkpoint.py           ← Save / load training checkpoints
│
├── outputs/                    ← Logs, plots, TensorBoard events
├── checkpoints/                ← Saved model weights
│   ├── best_model.pth          ← Original 2-head checkpoint (60 epochs)
│   └── best_model_4heads.pth   ← 4-head checkpoint (produced by finetune_new_heads.py)
│
├── train.py                    ← Main training entry point
├── validate.py                 ← Validation entry point
├── test.py                     ← Test evaluation + report generator
├── inference.py                ← Per-frame inference on any folder (2-head, original)
├── inference_v2.py             ← 4-head inference with occlusion injection (NEW)
├── finetune_new_heads.py       ← Head-only fine-tuning for Region+Confidence (NEW)
├── visualize_occlusion.py      ← Occlusion visualization demo
│
├── requirements.txt
└── README.md
```

---

## 4. Dataset Preparation

### CASIA-B Standard Structure

Your dataset root must follow this layout:

```
CASIA-B/
├── 001/
│   ├── nm-01/        ← Normal walking, sequence 1
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── nm-02/
│   ├── nm-03/
│   ├── nm-04/
│   ├── nm-05/
│   ├── nm-06/
│   ├── bg-01/        ← Bag condition
│   ├── bg-02/
│   ├── cl-01/        ← Coat condition
│   └── cl-02/
├── 002/
│   └── ...
└── 124/
    └── ...
```

Each `.png` is a **binary silhouette**: white person on black background, grayscale.

### Subject Split (no identity leakage)

| Split | Subjects | Count |
|---|---|---|
| Train | 001–074 | 74 subjects |
| Validation | 075–099 | 25 subjects |
| Test | 100–124 | 25 subjects |

Configurable in `configs/config.yaml` under the `split:` key.

---

## 5. Environment Setup

### 5.1 Create Conda Environment

```bash
conda create -n gait_occ python=3.10 -y
conda activate gait_occ
```

### 5.2 Install PyTorch (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5.3 Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5.4 Verify Installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6. Configuration

All settings live in `configs/config.yaml`. Key sections:

```yaml
dataset:
  path: "/data/CASIA-B"      # ← CHANGE THIS to your dataset path
  sequence_length: 30
  image_height: 128
  image_width: 88

model:
  backbone: "resnet18"        # resnet18 | resnet34 | resnet50

training:
  epochs: 60
  batch_size: 16
  learning_rate: 1.0e-4
  amp: true                   # Mixed precision

occlusion:
  occlusion_prob: 0.6         # 60% of clips get an occlusion event
```

---

## 7. Training

### Basic

```bash
python train.py --config configs/config.yaml
```

### Resume from checkpoint

```bash
python train.py --config configs/config.yaml --resume checkpoints/last_model.pth
```

### What happens during training

1. Datasets are indexed (sliding windows over all sequences)
2. Each batch: frames loaded → synthetic occlusion applied in memory → labels generated
3. CNN encodes each frame → Transformer adds temporal context
4. Detection head: BCEWithLogitsLoss; Severity head: SmoothL1Loss
5. Gradient clipping at norm 1.0, AdamW, cosine LR decay
6. Validation every epoch; best model saved by F1 score
7. Early stopping with patience=15

### Checkpoints saved

```
checkpoints/best_model.pth    ← Best validation F1
checkpoints/last_model.pth    ← Last completed epoch
```

---

## 8. Validation

```bash
python validate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth
```

**Output:**

```
==================================================
VALIDATION RESULTS
==================================================
  Loss       : 0.123456
  Det Loss   : 0.098765
  Sev Loss   : 0.024691
--------------------------------------------------
  Accuracy   : 0.9234
  Precision  : 0.9101
  Recall     : 0.9387
  F1         : 0.9242
--------------------------------------------------
  MAE        : 0.0521
  RMSE       : 0.0734
==================================================
```

---

## 9. Testing

```bash
python test.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output outputs/test_report.txt
```

Evaluates on subjects 100–124 (never seen during training or validation).
Saves a full report to `outputs/test_report.txt`.

---

## 10. Inference

```bash
python inference.py \
    --sequence /data/CASIA-B/001/nm-01 \
    --checkpoint checkpoints/best_model.pth \
    --config configs/config.yaml \
    --output outputs/inference_results \
    --threshold 0.5
```

**Console output:**

```
 Frame  Detection   Det Prob  Severity
-------------------------------------------------------
     0  Clean        0.082       0.000
     1  Clean        0.091       0.000
    ...
    11  Occluded     0.934       0.421
    12  Occluded     0.967       0.673
    ...
    21  Clean        0.103       0.012
```

**Files saved:**

```
outputs/inference_results/
├── pred_000.png        ← Annotated frames
├── pred_001.png
├── ...
├── sequence.gif        ← Animated sequence with labels
├── temporal_plot.png   ← Probability + severity over time
└── predictions.csv     ← All predictions in tabular form
```

---

## 11. Visualization

### Occlusion Generation Demo

Visualize what synthetic occlusions look like on real CASIA-B data:

```bash
python visualize_occlusion.py \
    --config configs/config.yaml \
    --output outputs/visualizations/demo
```

Specific subject and condition:

```bash
python visualize_occlusion.py \
    --config configs/config.yaml \
    --subject 5 \
    --condition nm-01 \
    --output outputs/visualizations/subj005_nm01
```

**Output files:**

```
outputs/visualizations/demo/
├── comparison_grid.png   ← Clean (top) vs. Occluded (bottom) side by side
├── clean.gif             ← Original sequence animation
├── occluded.gif          ← Occluded sequence animation
├── label_timeline.png    ← Detection + severity labels over time
└── labeled_frames/
    ├── frame_000.png     ← Per-frame annotated images
    └── ...
```

---

## 12. TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

Open your browser at `http://localhost:6006`.

**Logged scalars:**

| Tag | Description |
|---|---|
| `train/batch_loss` | Per-step training total loss |
| `train/batch_det_loss` | Per-step detection loss |
| `train/batch_sev_loss` | Per-step severity loss |
| `train/lr` | Current learning rate |
| `train/accuracy` | Epoch train accuracy |
| `train/f1` | Epoch train F1 |
| `val/loss` | Epoch validation loss |
| `val/accuracy` | Epoch validation accuracy |
| `val/precision` | Epoch validation precision |
| `val/recall` | Epoch validation recall |
| `val/f1` | Epoch validation F1 |
| `val/mae` | Epoch severity MAE |
| `val/rmse` | Epoch severity RMSE |

---

## 12.5. Upgrade: 4-Head Model (Region + Confidence) and Occlusion-Injection Inference

This section documents two additions made AFTER the original 60-epoch training
run completed. They do not require retraining the backbone/transformer or the
original detection/severity heads.

### 12.5.1 What was added

1. **RegionHead** — per-frame quadrant-wise occlusion detection. Four
   independent binary outputs: `upper`, `lower`, `left`, `right`. These are
   NOT mutually exclusive (an upper-left occlusion sets both `upper=1` and
   `left=1`), since each is judged on an overlapping half of the frame.
2. **ConfidenceHead** — per-frame self-confidence score in `[0, 1]`. There is
   no natural ground truth for "confidence," so it is trained against a
   **constructed calibration target**: `1 - calibration_error`, where
   `calibration_error` blends detection error, severity error, and temporal
   consistency (see `utils/confidence.py` for the exact formula). Low
   confidence means the model's detection/severity output for that frame is
   likely unreliable.
3. **`inference_v2.py`** — lets you inject a synthetic occlusion (type, frame
   range, target severity) onto any CLEAN sequence on demand, then see
   ground-truth-vs-predicted for all four heads. Solves the "I can only feed
   clean silhouettes with no way to check results" limitation of the
   original `inference.py`.

### 12.5.2 Step 1: Fine-tune the two new heads (one-time, fast)

The new heads start randomly initialised — they must see *some* gradient
before their output means anything. This step freezes everything else
(backbone, transformer, original det/sev heads) and trains ONLY the two new
small MLPs for a handful of epochs:

```bash
python finetune_new_heads.py \
    --config configs/config.yaml \
    --base_checkpoint checkpoints/best_model.pth \
    --output checkpoints/best_model_4heads.pth \
    --epochs 8
```

- Your original `checkpoints/best_model.pth` is **never modified**.
- Output is a new file: `checkpoints/best_model_4heads.pth`, containing all
  four heads (backbone/transformer/det/sev copied unchanged from the
  original checkpoint, region/confidence newly trained).
- Because only ~0.04% of parameters are trainable (two small MLPs, no CNN or
  Transformer backward pass), this finishes in a small fraction of the time
  the original 60-epoch run took.
- Validation metrics per epoch include per-region accuracy/F1, macro F1
  across the four regions, and confidence MAE/RMSE — printed and logged to
  `outputs/finetune_heads.log` / TensorBoard
  (`outputs/tensorboard_finetune_heads`).

### 12.5.3 Step 2: Run the upgraded inference tool

**Mode A — inject a synthetic occlusion you control, then compare against
ground truth (the main new capability):**

```bash
/home/himanshu/CVL_VST/CASIA-B-64/050/nm-03/090

python3 inference_v2.py \
    --sequence /home/himanshu/CVL_VST/CASIA-B-64/050/nm-03/090 \
    --checkpoint checkpoints/best_model_4heads.pth \
    --mode inject \
    --occlusion_type rectangle \
    --start_frame 10 --end_frame 25 \
    --target_severity 0.5 \
    --output outputs/inference_v2/050_nm03_injected
```

`--occlusion_type` accepts the same five types used during training:
`rectangle`, `random_blocks`, `moving_temporal`, `partial_body`,
`multi_block`. `--target_severity` is optional — omit it to use default
config-driven random sizing instead of targeting a specific value.

**Mode B — run a sequence as-is, no injection (sanity check on real data):**

```bash
python inference_v2.py \
    --sequence /home/himanshu/CVL_VST/CASIA-B-64/050/nm-03/090 \
    --checkpoint checkpoints/best_model_4heads.pth \
    --mode clean \
    --output outputs/inference_v2/050_nm03_clean
```

**Output files** (in the `--output` directory):

```
dashboard.png          4-panel GT-vs-Predicted plot: Detection, Severity,
                        Region (4 quadrants), Confidence
full_overlay/*.png      per-frame rich overlay — quadrant guide drawn on
                        the silhouette, colored border on predicted-positive
                        regions, header/footer showing GT vs Pred for every head
clean.gif               the original, pre-injection sequence
occluded.gif            the sequence actually fed to the model
paired.gif              clean | occluded side-by-side in one GIF
predictions.csv         full per-frame table, GT + predicted, all 4 heads
summary.txt             run summary (sequence, mode, injected params, counts)
```

A full console table is also printed showing every frame's ground truth and
predicted values for all four heads at once.

**If you only have the original 2-head checkpoint** (haven't run
`finetune_new_heads.py` yet), you can still run `inference_v2.py` by passing
`--base_checkpoint checkpoints/best_model.pth` instead of `--checkpoint`. The
script builds the 4-head model on the fly, but prints a loud warning that
region/confidence output is not yet meaningful (random initialisation).

### 12.5.4 Backward compatibility notes

- `models/model.py`'s `OcclusionDetectionModel.forward()` is byte-for-byte
  unchanged and still returns the original `(det_logits, sev_preds)` tuple —
  `train.py`, `validate.py`, `test.py`, `trainer.py`, and `evaluator.py` all
  continue to work with zero modification.
- The 4-head behaviour only activates when `enable_extra_heads=True` is
  passed to `build_model()` / `OcclusionDetectionModel()`. Default is
  `False`, reproducing the exact original architecture.
- `data/casia_dataset.py`'s `__getitem__` now also returns a `region_labels`
  tensor `(T, 4)` alongside the original `frames`/`det_labels`/`sev_labels`
  keys — existing training code that only reads the original three keys is
  unaffected.

---

### Directory tree after a full training run

```
gait_occlusion/
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
└── outputs/
    ├── train.log
    ├── test_report.txt
    ├── tensorboard/
    │   └── events.out.tfevents.*
    ├── inference_results/
    │   ├── pred_*.png
    │   ├── sequence.gif
    │   ├── temporal_plot.png
    │   └── predictions.csv
    └── visualizations/
        ├── comparison_grid.png
        ├── clean.gif
        ├── occluded.gif
        ├── label_timeline.png
        └── labeled_frames/
```

### Typical performance on CASIA-B (ResNet-18, T=30)

| Metric | Expected Range |
|---|---|
| Detection Accuracy | 88–93% |
| Detection F1 | 86–92% |
| Severity MAE | 0.04–0.07 |
| Severity RMSE | 0.06–0.10 |

*Exact numbers depend on occlusion probability, backbone size, and random seed.*

---

## 14. Troubleshooting

### Dataset not found / empty index

```
ERROR: Dataset index is empty.
```

- Verify `dataset.path` in `config.yaml` points to the correct CASIA-B root.
- Ensure the folder structure is `CASIA-B/001/nm-01/0001.png` etc.
- Check frame files are `.png` (not `.bmp` or `.jpg`; edit `_get_frame_paths` if needed).

### CUDA out of memory

Reduce `batch_size` in `config.yaml`.  As a guide:
- batch_size=16: ~10 GB VRAM (ResNet-18, T=30, 128×88)
- batch_size=8:  ~5.5 GB VRAM
- batch_size=4:  ~3 GB VRAM

You can also reduce `sequence_length` or use `backbone: resnet18`.

### Slow training (no CUDA)

If `torch.cuda.is_available()` returns `False`:
- Check your CUDA toolkit version matches the PyTorch build (see step 5.2).
- Run `nvidia-smi` to verify the GPU is visible to the OS.

### TensorBoard shows no data

- Ensure training has run for at least one epoch.
- Pass `--logdir` pointing to `outputs/tensorboard`, not a parent directory.

### Low F1 score early in training

This is expected — the model needs several epochs to learn meaningful temporal patterns.
The cosine scheduler and warmup are designed to stabilise early training.
Check that `occlusion.occlusion_prob` is set between 0.4 and 0.7 in config.

### Reproducibility across runs

Set `training.seed` in config and ensure `torch.backends.cudnn.benchmark = False`
(edit `utils/seed.py`, set `deterministic=True` in the `set_seed` call in `train.py`).
Note: fully deterministic mode is ~20% slower.

---

## Citation

If you use this code for your thesis or research, please acknowledge the following:

- CASIA-B Gait Database: S. Yu et al., "A Framework for Evaluating the Effect of View Angle, Clothing and Carrying Condition on Gait Recognition," *ICPR 2006*.
- ResNet: He et al., "Deep Residual Learning for Image Recognition," *CVPR 2016*.
- Transformer: Vaswani et al., "Attention Is All You Need," *NeurIPS 2017*.

---

*Project authored for research-grade M.Tech thesis work. All code is modular and extensible.*
