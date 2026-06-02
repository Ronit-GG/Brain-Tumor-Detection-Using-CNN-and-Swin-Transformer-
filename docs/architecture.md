# Architecture Reference

Detailed per-model architecture descriptions and a system-level
diagram. All diagrams use mermaid (renders inline on GitHub/GitLab)
and can be exported to PNG via the mermaid CLI if you need them for
LaTeX or slides.

---

## 1. System-level pipeline

```mermaid
flowchart TD
    A[MRI image<br/>any size, RGB or grayscale] --> B[Resize 224×224<br/>+ ImageNet Normalize]
    B --> T{Train or<br/>Inference?}
    T -- Train --> Aug[Augment:<br/>flip, rotate, affine,<br/>jitter, erasing]
    T -- Inference --> Z[no augmentation]
    Aug --> M1[Custom CNN<br/>1.21 M params]
    Aug --> M2[EfficientNet-B0<br/>4.0 M, pretrained]
    Aug --> M3[Swin-Tiny<br/>27.5 M, pretrained]
    Z   --> M1
    Z   --> M2
    Z   --> M3
    M1 --> P1[softmax<br/>p_cnn ∈ ℝ⁴]
    M2 --> P2[softmax<br/>p_tl ∈ ℝ⁴]
    M3 --> P3[softmax<br/>p_swin ∈ ℝ⁴]
    P1 --> E[Ensemble combiner<br/>weighted: 0.0·p_cnn + 0.9·p_tl + 0.1·p_swin]
    P2 --> E
    P3 --> E
    E --> R[argmax → class<br/>max → confidence]
    R --> X[Grad-CAM<br/>3 heatmaps<br/>optional]
```

---

## 2. Custom CNN (`src/cnn_model.py`)

```mermaid
flowchart TD
    I[Input<br/>3 × 224 × 224] --> B1[Block 1: Conv 3→32 + BN + ReLU<br/>Conv 32→32 + BN + ReLU<br/>MaxPool 2×2]
    B1 -->|32 × 112 × 112| B2[Block 2: Conv 32→64 + BN + ReLU<br/>Conv 64→64 + BN + ReLU<br/>MaxPool 2×2]
    B2 -->|64 × 56 × 56| B3[Block 3: Conv 64→128 + BN + ReLU<br/>Conv 128→128 + BN + ReLU<br/>MaxPool 2×2]
    B3 -->|128 × 28 × 28| B4[Block 4: Conv 128→256 + BN + ReLU<br/>Conv 256→256 + BN + ReLU<br/>MaxPool 2×2]
    B4 -->|256 × 14 × 14| GAP[Global Average Pool]
    GAP -->|256| FC1[Flatten + Dropout 0.5<br/>Linear 256 → 128 + ReLU]
    FC1 -->|128| FC2[Dropout 0.3<br/>Linear 128 → 4]
    FC2 --> O[4 logits]
```

| Stat | Value |
|---|---|
| Total parameters | 1,206,628 |
| Conv2d parameters | 1,171,296 (97 %) |
| BatchNorm parameters | 1,920 |
| Linear parameters | 33,412 (3 %) |
| Receptive field at last conv | ~100 × 100 pixels |
| Weight initialization | Kaiming normal (mode=fan_out, ReLU) |

---

## 3. Transfer model (`src/transfer_model.py`)

```mermaid
flowchart LR
    I[Input<br/>3 × 224 × 224] --> BB["EfficientNet-B0 backbone<br/>(ImageNet pretrained)<br/>frozen in Stage 1"]
    BB -->|1280-d features| H["Custom head<br/>Dropout 0.3<br/>Linear 1280 → 4"]
    H --> O[4 logits]
```

| Stat | Value |
|---|---|
| Backbone | `torchvision.models.efficientnet_b0` |
| Backbone params | 4,007,548 |
| Head params | 5,124 (1280 × 4 + 4) |
| Total params | 4,012,672 |
| Stage 1: trainable | 5,124 (head only) |
| Stage 2: trainable | 4,012,672 (everything) |
| Stage 1 LR | 1e-3 (4 epochs) |
| Stage 2 LR | 1e-4 (8 epochs, 10× smaller) |

### Why two-stage?

1. Random head → noisy gradients → would corrupt the pretrained
   backbone if not frozen.
2. After 3-4 epochs, the head produces sensible logits; gradients
   flowing back are informative.
3. Stage 2 unfreezes the backbone with a 10× smaller LR to *adjust*
   the pretrained features without overwriting them.

---

## 4. Swin Transformer (`src/swin_model.py`)

```mermaid
flowchart TD
    I[Input<br/>3 × 224 × 224] --> PE[Patch Embed<br/>Conv 4×4 stride 4<br/>→ 96 ch]
    PE -->|56×56×96| S1[Stage 1: 2 blocks<br/>W-MSA + SW-MSA, dim 96]
    S1 -->|56×56×96| PM1[Patch Merging<br/>concat 2×2, project to 192]
    PM1 -->|28×28×192| S2[Stage 2: 2 blocks<br/>W-MSA + SW-MSA, dim 192]
    S2 -->|28×28×192| PM2[Patch Merging<br/>concat 2×2, project to 384]
    PM2 -->|14×14×384| S3[Stage 3: 6 blocks<br/>W-MSA + SW-MSA, dim 384]
    S3 -->|14×14×384| PM3[Patch Merging<br/>concat 2×2, project to 768]
    PM3 -->|7×7×768| S4[Stage 4: 2 blocks<br/>W-MSA + SW-MSA, dim 768]
    S4 -->|7×7×768| N[LayerNorm + GlobalAvgPool]
    N -->|768| H[Dropout 0.3<br/>Linear 768 → 4]
    H --> O[4 logits]
```

| Stat | Value |
|---|---|
| Backbone | `timm.create_model('swin_tiny_patch4_window7_224')` |
| Patch size | 4 × 4 |
| Window size | 7 × 7 |
| Stage depths | [2, 2, 6, 2] |
| Heads per stage | [3, 6, 12, 24] |
| Total params | 27,522,430 |
| Head params | 3,076 (768 × 4 + 4) |
| Stage 1 LR | 1e-3 (3 epochs, head only) |
| Stage 2 LR | **2e-5** (4 epochs, 50× smaller — critical for transformers) |

### The shifted-window mechanism

Without shifted windows, tokens in different 7 × 7 windows can never
attend to each other. Swin alternates: layer 2L uses regular windows,
layer 2L+1 shifts by (3, 3) — so windows in the shifted layout
overlap the regular ones. After both layers, every token has
influenced every neighbour across the original window boundary.

---

## 5. Ensemble (`src/ensemble.py`)

```mermaid
flowchart LR
    P1[p_cnn ∈ ℝ⁴] --> W1[× 0.0]
    P2[p_tl  ∈ ℝ⁴] --> W2[× 0.9]
    P3[p_swin ∈ ℝ⁴] --> W3[× 0.1]
    W1 --> S[Σ → p_ensemble]
    W2 --> S
    W3 --> S
    S --> A[argmax → predicted class]
    S --> C[max → confidence]
```

Weights `(0.0, 0.9, 0.1)` were found by **exhaustive grid search**
over the simplex `{(w_cnn, w_tl, w_swin) : w_i ≥ 0, Σw = 1}` at
step 0.05 (231 candidates for M=3), optimizing validation accuracy.

The grid search effectively *discarded* the custom CNN (`w_cnn = 0`)
because its predictions overlap too much with EfficientNet's — adding
it brings no complementary signal but does add noise.

---

## 6. Training engine (`src/train.py`)

```mermaid
flowchart TD
    S[Start] --> Init[set_seed<br/>build optimizer AdamW<br/>build scheduler CosineAnnealingLR<br/>build GradScaler if fp16<br/>init TensorBoard writer]
    Init --> Loop{epoch < max?}
    Loop -- yes --> Train["train_one_epoch():<br/>for batch in train_loader<br/>  forward in autocast<br/>  loss.backward<br/>  optimizer.step"]
    Train --> Eval["evaluate():<br/>for batch in val_loader<br/>  no_grad forward<br/>  accumulate loss/acc"]
    Eval --> Sched[scheduler.step]
    Sched --> Log[append to history<br/>write TensorBoard scalars]
    Log --> Best{val_acc > best?}
    Best -- yes --> Save[save best checkpoint<br/>reset patience]
    Best -- no --> Inc[patience += 1]
    Save --> ES{patience reached?}
    Inc --> ES
    ES -- no --> Loop
    ES -- yes --> Done[restore best weights<br/>save history JSON<br/>close writer]
    Loop -- no --> Done
```

---

## 7. Inference path (`src/inference.py`)

```mermaid
flowchart LR
    U["PIL.Image<br/>or path"] --> L["_load_image()<br/>convert RGB<br/>resize 224<br/>normalize"]
    L --> T["preprocessed tensor<br/>(3, 224, 224)"]
    T --> R1[CNN forward]
    T --> R2[EffNet forward]
    T --> R3[Swin forward]
    R1 --> S1[softmax]
    R2 --> S2[softmax]
    R3 --> S3[softmax]
    S1 --> EN[ensemble combine]
    S2 --> EN
    S3 --> EN
    EN --> AC[argmax → class<br/>max → conf]
    AC --> GC{return_gradcam?}
    GC -- no --> O[PredictionResult]
    GC -- yes --> G["compute 3 Grad-CAMs<br/>for predicted class"]
    G --> O
```

`BrainTumorPredictor` loads all 3 models once (cached by Streamlit /
FastAPI). Per-image inference: ~85-100 ms without Grad-CAM,
~250-950 ms with Grad-CAM (first call has CUDA kernel warmup).

---

## 8. Deployment topology

```mermaid
flowchart TD
    DEV[Developer / Researcher] -.->|notebook| NB[Jupyter notebooks/]
    DEV -.->|script| SCR[python -m src.train]
    NB --> CKPT[models/*.pth + ensemble_config.json]
    SCR --> CKPT
    CKPT --> EXP[src/export.py<br/>ONNX export]
    EXP --> ONNX[models/*.onnx<br/>cross-platform]
    CKPT --> ST[app/app.py<br/>Streamlit UI]
    CKPT --> API[app/api.py<br/>FastAPI REST]
    ONNX --> EDGE[Edge devices<br/>mobile / C++ / JS]
    ST --> USER[End user<br/>browser]
    API --> CLIENT[Client apps<br/>scripts / mobile / web]
    ST --> DOCKER[Dockerfile]
    API --> DOCKER
    DOCKER --> CLOUD[Cloud<br/>K8s / ECS]
```

All paths share the same trained checkpoints — change a model once,
and every deployment channel picks it up.
