# Brain Tumor Detection using an Ensemble of CNN, Transfer Learning and Swin Transformer Models

## Abstract

We present a complete deep-learning system for four-class brain tumor
classification from T1-weighted MRI images, combining three
architecturally diverse models — a custom convolutional neural network
trained from scratch, an ImageNet-pretrained EfficientNet-B0 fine-tuned
in two stages, and a Swin-Tiny vision transformer — into a weighted
soft-voting ensemble. On a held-out test set of 1,600 images
(400 per class) the best individual model (EfficientNet-B0) achieves
**94.94 % accuracy** and **macro AUC 0.991**, while the three-model
ensemble matches this performance (94.69 %). We additionally provide
explainability via Grad-CAM heatmaps for each base model, a Streamlit
front-end, a FastAPI REST backend, and ONNX exports verified to
preserve predictions while delivering 2–3.5× CPU inference speedup.
The most important empirical finding is that ensembling does not
always exceed the best individual model when one base learner
dominates — a result of practical and pedagogical value.

**Keywords:** brain tumor classification, MRI, convolutional neural
networks, transfer learning, Swin Transformer, ensemble learning,
Grad-CAM, medical imaging, PyTorch.

---

## 1. Introduction

### 1.1 Motivation

Brain tumors are abnormal growths of cells inside the cranium,
classified radiologically by tissue of origin and location. Manual
classification by a radiologist takes 15–30 minutes per scan and
inter-observer agreement on tumor *type* is reported around 70–85%.
A reliable AI assistant can (i) triage scans in seconds, (ii) act as a
second reader to catch missed findings, and (iii) provide consistent
results where expert radiologists are scarce.

### 1.2 Problem Statement

Given a single 2D T1-weighted brain MRI slice, predict one of four
classes — **glioma**, **meningioma**, **pituitary**, or **no tumor** —
along with a calibrated confidence score and an explainability heatmap
showing the regions the model attended to.

### 1.3 Contributions

1. A **production-grade reproducible pipeline** built on PyTorch 2.12
   with modular `src/` package, deterministic stratified splits, and
   bfloat16 mixed-precision training.
2. A **three-model ensemble** combining a custom CNN, an EfficientNet-B0
   transfer-learning model, and a Swin Transformer — chosen explicitly
   for architectural diversity.
3. A **rigorous evaluation** including val/test split discipline,
   per-class precision/recall/F1, multi-model ROC curves, and
   inference-speed profiling.
4. **Explainability** via Grad-CAM for all three models (including the
   non-trivial reshape required for Swin's channels-last activations).
5. **Four deployment channels** — Streamlit, FastAPI REST, ONNX
   Runtime, Docker — all wrapping the same trained checkpoints.
6. An **honest negative result**: the ensemble matches but does not
   beat the dominant base model on the test set, and we discuss why.

---

## 2. Related Work

CNN-based brain MRI classification has been widely studied since ~2017,
with reported accuracies in the 90–99 % range on this and similar
datasets. Recent work has explored transformers (ViT, Swin) and
ensembles. Our contribution is not state-of-the-art absolute accuracy
but a *transparent, reproducible* end-to-end engineering pipeline
suitable for educational and clinical-prototype use.

---

## 3. Dataset

We use the **Brain Tumor MRI Dataset** (Masoud Nickparvar, Kaggle), a
publicly available collection of T1-weighted brain MRI slices labelled
by tumor type.

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Training (full) | 1,400 | 1,400 | 1,400 | 1,400 | **5,600** |
| ↳ Train (80%) | 1,120 | 1,120 | 1,120 | 1,120 | 4,480 |
| ↳ Val   (20%) |   280 |   280 |   280 |   280 | 1,120 |
| **Test (held out)** | 400 | 400 | 400 | 400 | **1,600** |

The Training set is split 80/20 train/val using `sklearn.model_selection
.train_test_split(stratify=labels, random_state=42)`, with results cached
to `data/processed/{train,val,test}_split.csv` to guarantee
reproducibility. The Testing folder is never touched during training
or hyperparameter tuning.

### 3.1 Exploratory Data Analysis (STEP 4)

- All four classes are perfectly balanced across all splits.
- Image modes are mixed: 2,585 RGB, 1,892 grayscale (`mode='L'`), 3
  RGBA. We normalize all to 3-channel via `Image.convert('RGB')` to
  match ImageNet-pretrained backbones.
- Dimensions vary widely (150 × 168 px to 1,375 × 1,446 px); we
  square-resize to 224 × 224 for all models.
- No corrupt files detected (every image opens cleanly with PIL).
- Per-class pixel-intensity histograms overlap substantially —
  classes cannot be separated by intensity alone, justifying the need
  for deep feature learning.

---

## 4. Methodology

### 4.1 Preprocessing & Augmentation

The training pipeline applies, in order:
1. Resize to 224 × 224.
2. `RandomHorizontalFlip(p=0.5)` — brain is bilaterally symmetric.
3. `RandomRotation(±15°)` — natural head-tilt variation.
4. `RandomAffine(translate=±5%, scale=±5%)` — FOV variation.
5. `ColorJitter(brightness=0.2, contrast=0.2)` — scanner gain variation.
6. `ToTensor()` + ImageNet `Normalize`.
7. `RandomErasing(p=0.25, scale=0.02–0.10)` — forces global reasoning.

Vertical flips and aggressive rotations are deliberately excluded as
clinically implausible. Validation and test images use the
deterministic pipeline (resize + normalize only).

### 4.2 Model 1 — Custom CNN

A VGG-style architecture: four blocks of two stacked 3 × 3 convolutions
each (with BatchNorm + ReLU + 2 × 2 max-pool), doubling channel count
every block (32 → 64 → 128 → 256). The deepest feature map is
globally average-pooled and fed to a two-layer fully-connected
classifier with dropout (p = 0.5, then 0.3). Kaiming initialization is
used. Total parameters: **1.21 M**.

### 4.3 Model 2 — Transfer Learning (EfficientNet-B0)

We chose EfficientNet-B0 over ResNet-50, EfficientNet-B3, DenseNet-121,
and ConvNeXt-Tiny on the basis of accuracy/parameter efficiency
(77.7 % ImageNet top-1 at only 5.3 M params) and its compatibility with
our 4 GB VRAM budget. The pretrained backbone is loaded from
torchvision; its 1000-class classifier is replaced with a 4-class
linear head (Dropout 0.3 → Linear 1280→4). Training proceeds in two
stages:

- **Stage 1 (warm-up, 4 epochs, LR = 1e-3):** Backbone frozen;
  only the 5,124 head parameters receive gradients. We override the
  module's `.train()` to keep the frozen backbone in `.eval()` mode,
  preventing BatchNorm running-stat drift.
- **Stage 2 (fine-tune, 8 epochs, LR = 1e-4):** Backbone unfrozen.
  The 10× smaller learning rate prevents catastrophic forgetting of
  pretrained features.

Total parameters: **4.01 M**.

### 4.4 Model 3 — Swin Transformer

We use `swin_tiny_patch4_window7_224` from `timm` (28 M parameters,
4 hierarchical stages, window-attention + shifted-window blocks). The
classifier head is replaced identically to EfficientNet. Two-stage
training mirrors the transfer-learning recipe but with a far smaller
Stage 2 learning rate (**2 × 10⁻⁵**, 50× smaller than Stage 1) because
transformer attention weights are extremely sensitive to large updates.
Total parameters: **27.52 M**.

### 4.5 Ensemble

Each model outputs a softmax probability vector \(\mathbf{p}_i \in
\mathbb{R}^4\). We evaluate three combiners on the validation set:

- **Soft voting:** \(\mathbf{p}_\text{ens} = \frac{1}{3}\sum_i \mathbf{p}_i\)
- **Weighted average:** \(\mathbf{p}_\text{ens} = \sum_i w_i\,\mathbf{p}_i\)
  with \(w_i \ge 0\), \(\sum w_i = 1\). Optimal weights found by
  exhaustive grid search at step 0.05 (231 evaluations for M=3).
- **Stacking:** logistic-regression meta-learner on the concatenated
  12-dimensional probability vector. Evaluated honestly via 5-fold
  cross-validation on the validation set.

### 4.6 Training Engine

A single reusable `fit()` function (`src/train.py`) drives all model
training. Features:
- AdamW optimizer (β₁ = 0.9, β₂ = 0.999, wd = 1e-4)
- CosineAnnealingLR with `eta_min = lr × 1e-2`
- Bfloat16 mixed-precision via `torch.amp.autocast` (we switched from
  fp16 after diagnosing a cuDNN NaN bug on the Turing GPU)
- Early stopping based on validation loss (patience = 5–7 epochs)
- Best-checkpoint saving by validation accuracy
- TensorBoard scalar logging
- tqdm per-epoch progress bars

---

## 5. Implementation Details

| Component | Choice | Reason |
|---|---|---|
| Framework | PyTorch 2.12 | Industry standard for research; native CUDA |
| Mixed precision | bfloat16 | Avoids fp16 cuDNN NaN bug on Turing GPUs |
| Optimizer | AdamW | Strong baseline for small medical datasets |
| Loss | CrossEntropyLoss | Classes balanced; no weighting needed |
| Batch size | 32 | Largest that fits Swin-Tiny in 4 GB VRAM with AMP |
| Schedulers | CosineAnnealingLR | Smooth convergence with minimal tuning |
| Seed | 42 (all RNGs) | Reproducibility |
| Hardware | NVIDIA GTX 1650 (4 GB), Windows 10, Python 3.11.0 | |

Detailed reproducibility: every notebook is committed with embedded
outputs; the split CSVs are versioned; weights and config are saved
to `models/`.

---

## 6. Results

### 6.1 Per-model Performance

Trained for the per-model epoch budgets in §4. **Honest test-set
metrics** (touched once, after all hyperparameter decisions frozen):

| Model | Params | Val acc | **Test acc** | Macro F1 | Macro AUC | Train (min) | Latency ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 1.21 M | 0.9036 | 0.8275 | 0.8221 | 0.9513 | 32.1 | 6.8 |
| EfficientNet-B0 | 4.01 M | 0.9893 | **0.9494** | **0.9482** | **0.9908** | 14.9 | 24.1 |
| Swin-Tiny | 27.52 M | 0.9670 | 0.9194 | 0.9171 | 0.9865 | 12.5 | 25.7 |
| **Ensemble (weighted)** | 32.74 M | 0.9893 | 0.9469 | 0.9456 | 0.9907 | 59.5 | 56.6 |

### 6.2 Ensemble Analysis

Validation-set grid search produced weights **[w_cnn, w_transfer, w_swin]
= [0.00, 0.90, 0.10]** — i.e. the grid search effectively *ignored* the
weakest model entirely. On the validation set, all three combiners
performed similarly to (but did not exceed) EfficientNet-B0 alone:

| Method | Val acc |
|---|---:|
| Soft voting | 0.9750 |
| Weighted (0/0.9/0.1) | 0.9893 |
| Stacking (5-fold CV) | 0.9857 |
| EfficientNet alone | 0.9893 |

This is **the dominant-model problem**: when one base learner
substantially outperforms the others, averaging with weaker learners
injects noise. The variance reduction promised by ensembling only
materializes when the base learners are of comparable strength and
have *uncorrelated* errors.

### 6.3 Per-class Behaviour

Both EfficientNet-B0 and the ensemble achieve **100 % recall on
`notumor`** — the most clinically critical class (false negatives
matter most). The hardest pair is glioma ↔ meningioma: the custom CNN
misclassifies 17 % of gliomas as meningiomas; EfficientNet reduces
this to 1 %; Swin to 6 %.

### 6.4 Explainability

Grad-CAM heatmaps on correctly-classified samples confirm that all
three models attend to anatomically meaningful regions (tumor mass or
brain parenchyma), not scanner watermarks or skull artifacts.
EfficientNet's heatmaps are the most spatially focused; Swin's are
the most diffuse (a documented characteristic of self-attention's wide
effective receptive field).

On the most confidently misclassified examples (all gliomas), three
distinct failure modes emerge: (i) genuine class ambiguity between
glioma and meningioma (visually similar masses), (ii) small/subtle
tumors that fall below the model's detection threshold, and (iii)
out-of-distribution scan styles (e.g. CT-style intensities).

### 6.5 Inference Speed (Single-image, GTX 1650, bf16)

| Model | Batch 1 latency | Throughput @ batch 32 |
|---|---:|---:|
| Custom CNN | 6.8 ms | 187 img/s |
| EfficientNet-B0 | 24.1 ms | **296 img/s** |
| Swin-Tiny | 25.7 ms | 120 img/s |
| Ensemble | 56.6 ms | 59 img/s |

ONNX-exported models (CPU only) are 2.0–3.5× faster than the
PyTorch eager-mode counterparts for batch=1 inference, with verified
100 % argmax agreement.

---

## 7. Discussion

### 7.1 Why didn't the ensemble win?

Two reasons. **First**, EfficientNet-B0 already operates near the
data's noise ceiling — only 1.07 % validation error and 5.06 % test
error. Improving on it would require base models that produce
genuinely complementary errors. Our CNN and Swin make many of the
*same* errors as EfficientNet, so averaging cancels little. **Second**,
the dominant-model effect: weighting in the weaker models multiplies
their noise into the ensemble.

This is not a failure of the methodology — it is empirical evidence
that **ensembling is most useful when base models are diverse *and* of
comparable strength**.

### 7.2 What would push accuracy higher?

- Larger pretrained backbones (EfficientNet-B3, ConvNeXt-Base) — VRAM
  permitting.
- Test-time augmentation (TTA): average predictions over horizontal
  flips and crops at inference.
- Self-supervised pretraining on a larger MRI corpus before
  supervised fine-tuning.
- Targeted data collection on the hard glioma/meningioma pair.

### 7.3 Limitations

- 2D slice classification, not 3D volumetric reasoning.
- Single MRI sequence (T1) — clinical decisions usually combine T1, T2,
  FLAIR, contrast.
- Tested on one public dataset — clinical deployment requires
  cross-institution validation.
- No segmentation (this is classification only).

### 7.4 Engineering lessons learned

- **Mixed precision is hardware-dependent.** fp16 triggered a cuDNN
  NaN bug on our GTX 1650 (Turing, compute 7.5). Bfloat16 — with the
  same dynamic range as fp32 — sidesteps the issue.
- **Transformer fine-tuning needs much smaller LR than CNN
  fine-tuning** (2e-5 vs 1e-4). Attention weights are more sensitive
  to large gradient updates.
- **Test-set discipline matters.** Reporting only validation accuracy
  would have overstated all four models by 4–8 percentage points.
- **Grad-CAM exposed both strengths and failure modes.** Cases where
  the model attends to the right region but picks the wrong class
  point to genuine clinical ambiguity rather than spurious cue
  exploitation.

---

## 8. Conclusion

We have built and rigorously evaluated a three-model ensemble for
brain tumor MRI classification. The dominant individual model
(EfficientNet-B0, transfer learning) achieves 94.94 % test accuracy
and macro ROC-AUC 0.991. The ensemble matches but does not exceed
this performance — an honest finding consistent with the
dominant-model effect documented in the ensemble-learning literature.
Grad-CAM analysis confirms that all three models attend to
anatomically meaningful regions, and remaining errors map to
explainable failure modes. The system is deployable via Streamlit,
FastAPI, ONNX Runtime, and Docker, providing four interchangeable
interfaces that share a single set of trained checkpoints. The
complete pipeline — data loading, augmentation, training, evaluation,
explainability, and inference — is reproducible from a single
`setup.ps1` script.

---

## References

1. Masoud Nickparvar. *Brain Tumor MRI Dataset.* Kaggle, 2022.
2. Tan, M. and Le, Q. *EfficientNet: Rethinking Model Scaling for
   Convolutional Neural Networks.* ICML 2019.
3. Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer
   using Shifted Windows.* ICCV 2021.
4. Selvaraju, R. et al. *Grad-CAM: Visual Explanations from Deep
   Networks via Gradient-based Localization.* ICCV 2017.
5. Loshchilov, I. and Hutter, F. *Decoupled Weight Decay
   Regularization.* ICLR 2019.
6. Loshchilov, I. and Hutter, F. *SGDR: Stochastic Gradient Descent
   with Warm Restarts.* ICLR 2017.
7. Micikevicius, P. et al. *Mixed Precision Training.* ICLR 2018.

---

*For academic / educational use only. Not a medical device.
Not a substitute for radiologist diagnosis.*
