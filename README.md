# 🦿 Temporal Gait Occlusion Detection and Severity Estimation

> **A PyTorch-based deep learning framework for frame-level occlusion detection and severity estimation in gait silhouette sequences using the CASIA-B dataset.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![Transformer](https://img.shields.io/badge/Model-Transformer-orange)
![ResNet18](https://img.shields.io/badge/Backbone-ResNet18-purple)

---

## 📌 Overview

Gait recognition systems often suffer performance degradation due to **partial occlusions** caused by carried objects, clothing variations, or environmental obstacles.

This project presents a **multi-task deep learning framework** capable of simultaneously:

* Detecting whether a gait silhouette is occluded.
* Estimating the severity of the occlusion.
* Learning temporal motion information across complete gait sequences.

To avoid expensive manual annotation, realistic synthetic occlusions are generated dynamically during training, enabling the model to learn robust representations without modifying the original dataset.

---

# 🏗️ Proposed Architecture

<p align="center">
<img src="assets/architecture.png" width="950">
</p>

The complete framework consists of four major stages:

### ① Input Sequence

A sequence of binary gait silhouettes is extracted from the CASIA-B dataset.

↓

### ② Synthetic Occlusion Generation

During training, synthetic occlusions are generated on-the-fly using multiple occlusion strategies, including:

* Partial Body Occlusion
* Random Block Occlusion
* Multi-Block Occlusion
* Temporal Moving Occlusion
* Random Rectangle Occlusion

Each generated sequence is automatically assigned:

* Binary Occlusion Label
* Occlusion Severity Score (0–1)

↓

### ③ Spatial Feature Extraction

Each frame is independently processed using a pretrained **ResNet-18** backbone.

The CNN extracts high-level spatial representations while preserving silhouette information.

↓

### ④ Temporal Feature Learning

Frame embeddings are passed through a **Transformer Encoder**, allowing the network to capture long-range temporal dependencies across the gait sequence.

↓

### ⑤ Multi-Task Prediction Heads

The shared temporal representation is used by two independent prediction heads:

| Head           | Task                            |
| -------------- | ------------------------------- |
| Detection Head | Binary Occlusion Classification |
| Severity Head  | Continuous Severity Regression  |

---

# 🔄 Model Pipeline

```text
Input Gait Sequence
          │
          ▼
Synthetic Occlusion Generator
          │
          ▼
ResNet-18 Backbone
(Frame-wise Feature Extraction)
          │
          ▼
Transformer Encoder
(Temporal Feature Learning)
          │
     ┌────┴────┐
     ▼         ▼
Detection   Severity
 Head         Head
(Binary)   (Regression)
```

---

# ✨ Highlights

* 🎯 Frame-level occlusion detection
* 📈 Continuous severity estimation
* 🧠 Transformer-based temporal learning
* 🖼️ Dynamic synthetic occlusion generation
* ⚡ Mixed Precision (AMP) training
* 📊 TensorBoard integration
* 🔄 Modular PyTorch implementation
* 📂 Subject-wise dataset split to prevent identity leakage

---

# 🛠️ Technology Stack

| Category             | Tools                   |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| Deep Learning        | PyTorch                 |
| CNN Backbone         | ResNet-18               |
| Sequence Modeling    | Transformer Encoder     |
| Dataset              | CASIA-B                 |
| Visualization        | TensorBoard, Matplotlib |
| Image Processing     | OpenCV                  |
| GPU Support          | CUDA                    |

---

# 📂 Repository Structure

```text
Temporal-Gait-Occlusion-Detection/
│
├── assets/
│   └── architecture.png
│
├── configs/
├── data/
├── models/
├── utils/
├── checkpoints/
├── outputs/
│
├── train.py
├── validate.py
├── test.py
├── inference.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The framework is trained and evaluated on the **CASIA-B Gait Dataset**, containing:

* 124 Subjects
* Multiple Camera Angles
* Normal Walking
* Walking with Bag
* Walking with Coat

> **Note:** The dataset is not distributed with this repository due to licensing restrictions.

---

# 🚀 Installation

```bash
git clone https://github.com/Amogha-Mayya/Temporal-Gait-Occlusion-Detection.git

cd Temporal-Gait-Occlusion-Detection

pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train

```bash
python train.py
```

### Validate

```bash
python validate.py
```

### Test

```bash
python test.py
```

### Inference

```bash
python inference.py
```

---

# 📸 Sample Output

<p align="center">
<img src="outputs/visualizations/demo/comparison_grid.png" width="800">
</p>

---

# 🔮 Future Work

* Vision Transformer (ViT) backbone
* Real-world occlusion datasets
* Domain adaptation across gait datasets
* Edge deployment for real-time inference
* Lightweight mobile architecture
* Multi-person gait occlusion analysis
