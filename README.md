# CMU 11-685 · HW2P2: Face Classification & Verification

> **Academic Competition / Project** for **11-685: Introduction to Deep Learning (Spring 2026)**
> Carnegie Mellon University
> Kaggle Competition: [11785-hw-2-p-2-face-verification-spring-2026](https://www.kaggle.com/competitions/11785-hw-2-p-2-face-verification-spring-2026/leaderboard)

---

## Overview

This project implements a **face recognition system** from scratch using PyTorch, covering both **face classification** (closed-set identity prediction) and **face verification** (open-set similarity scoring). The model is trained on a subset of the VGGFace2 dataset with 8,631 identities and evaluated on the Kaggle leaderboard using Equal Error Rate (EER).

**Competition Result:** Rank **177 / 266** · Private Score (EER): **3.36752** · [View Leaderboard](https://www.kaggle.com/competitions/11785-hw-2-p-2-face-verification-spring-2026/leaderboard)

---

## Model Architecture

A custom **ResNet-34** backbone built from scratch using `torch.nn`, without any pretrained weights.

```
Input (3 × 112 × 112)
    │
    ▼
Stem: Conv2d(3→64, k=7, s=2) → BN → ReLU → MaxPool(k=3, s=2)
    │                                                      [56×56 → 28×28]
    ▼
Layer 1: 3 × ResidualBlock(64  → 64,  stride=1)
Layer 2: 4 × ResidualBlock(64  → 128, stride=2)
Layer 3: 6 × ResidualBlock(128 → 256, stride=2)
Layer 4: 3 × ResidualBlock(256 → 512, stride=2)
    │
    ▼
AdaptiveAvgPool2d(1×1) → Flatten
    │
    ▼
FC: Linear(512 → 512) → BatchNorm1d → Dropout(0.3)
    │                                             [Face Embedding z ∈ ℝ⁵¹²]
    ▼
Classifier: Linear(512 → 8631)
```

**Total Parameters:** ~26.1M (well within the 30M limit)

Each `ResidualBlock` follows the standard two-layer design:
```
x → Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → (+x) → ReLU
```
A 1×1 projection shortcut is used when channel dimensions or stride change.

---

## Configuration

```python
config = {
    'batch_size':  64,
    'lr':          1e-5,
    'epochs':      20,
    'num_classes': 8631,
}
```

**Loss:** `nn.CrossEntropyLoss(label_smoothing=0.1)`

**Optimizer:** AdamW (`lr=1e-5`, `weight_decay=5e-4`)

**Scheduler:** CosineAnnealingLR (`T_max=epochs`, `eta_min=1e-7`)

---

## Key Learnings

**1. Cosine Annealing outperformed ReduceLROnPlateau**

Cosine Annealing smoothly decays the learning rate to near-zero over training, avoiding the noisy, reactive drops of plateau-based schedulers. This led to more stable convergence and better final performance.


**2. Label Smoothing reduces overconfidence**

Adding `label_smoothing=0.1` to `CrossEntropyLoss` prevented the model from becoming overly confident on training identities, which hurt generalization to unseen faces in verification.

**3. Classification accuracy ≠ Verification quality**

A model can achieve high closed-set classification accuracy while still performing poorly at verification. The CE loss encourages correct class scores but does not explicitly enforce that embeddings of the same person are geometrically close. Due to time constraints, metric learning losses like **ArcFace** or **Triplet Loss** were not implemented in this submission — but adding them is the most impactful next step for improving verification EER.

**4. Cosine similarity is better than Euclidean distance for verification**

Cosine similarity measures the angle between embedding vectors rather than their magnitude, making it robust to scale differences across images. During inference, embeddings from two images are compared as:

```python
similarity = (feat1 * feat2).sum(dim=1)  # dot product of L2-normalized vectors
```

---

## Results

| Split | Metric | Score |
|---|---|---|
| Kaggle Private | EER (↓) | **3.36752** |
| Kaggle Leaderboard Rank | — | 177 / 266 |

EER (Equal Error Rate) is the threshold at which False Acceptance Rate = False Rejection Rate. Lower is better.

---

## Dataset

### Source
- **Classification training:** Subset of [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) — 8,631 identities, 112×112 images
- **Verification:** 6,000 image pairs across 5,749 identities (open-set, not seen during training)

### Structure
```
hw2p2_data/
├── cls_data/
│   ├── train/      # One subfolder per identity (used for training)
│   ├── dev/        # One subfolder per identity (used for val accuracy)
│   └── test/
├── ver_data/       # Unlabeled face images for verification
├── val_pairs.txt   # Labeled pairs for validation EER
└── test_pairs.txt  # Unlabeled pairs for Kaggle submission
```

### Preprocessing & Augmentation

**Training:**
- Resize to 112×112
- Random Horizontal Flip (p=0.5)
- ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1)
- ToTensor + ToDtype(float32)
- Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

**Validation / Test:**
- Resize to 112×112
- ToTensor + ToDtype(float32)
- Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


---

## Constraints (Per Course Policy)

- No pretrained weights (trained from scratch only)
- No external data beyond the provided competition dataset
- CNN architectures only — no RNNs, Transformers, or GNNs
- Max 30M parameters (including ensembles)
- All `torch.nn` components implemented directly
