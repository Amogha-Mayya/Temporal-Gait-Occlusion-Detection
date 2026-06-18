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
│   ├── heads.py                ← Detection + Severity MLP heads
│   └── model.py                ← Full model assembly + loss
│
├── engine/
│   ├── trainer.py              ← Training loop (AMP, grad clip)
│   ├── evaluator.py            ← Validation / test evaluation
│   └── inference_engine.py     ← Sliding-window inference
│
├── utils/
│   ├── metrics.py              ← Acc/P/R/F1/MAE/RMSE accumulators
│   ├── logger.py               ← File + console + TensorBoard logger
│   ├── visualization.py        ← GIF, grid, plot, annotation tools
│   ├── seed.py                 ← Reproducible seed setter
│   └── checkpoint.py           ← Save / load training checkpoints
│
├── outputs/                    ← Logs, plots, TensorBoard events
├── checkpoints/                ← Saved model weights
│
├── train.py                    ← Main training entry point
├── validate.py                 ← Validation entry point
├── test.py                     ← Test evaluation + report generator
├── inference.py                ← Per-frame inference on any folder
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
python3 validate.py \
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

## 13. Expected Outputs

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
