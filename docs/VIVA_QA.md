# Viva Q&A — Brain Tumor Detection Ensemble

This file contains anticipated examiner questions with concise model
answers, organized by topic. Read it the day before your viva.

---

## Group 1 — Project motivation & dataset

### Q1. What does your project do?
It classifies a brain MRI image into one of four classes — glioma,
meningioma, pituitary, no tumor — using a three-model ensemble of
architecturally diverse deep networks (custom CNN, EfficientNet-B0
via transfer learning, and a Swin Transformer). Best test-set
accuracy is **94.94 %**.

### Q2. Why these four classes?
They are the most common tumor types in publicly available brain
MRI datasets and represent clinically distinct categories. "No tumor"
is included as the negative class — clinically the most important
class to get right (false negatives are the worst error).

### Q3. Why MRI and not CT?
MRI offers superior soft-tissue contrast compared to CT, making it
the modality of choice for diagnosing intracranial pathology. The
public dataset we used is also MRI-based.

### Q4. What dataset did you use? How big is it?
The Brain Tumor MRI Dataset by Masoud Nickparvar on Kaggle:
**5,600 training + 1,600 testing images** across the 4 classes,
perfectly balanced. We further split Training into 4,480 train and
1,120 val (stratified 80/20).

### Q5. How did you ensure the test set wasn't seen during training?
The `Testing/` folder is touched exactly once — in STEP 15 — to
report final metrics. All hyperparameter tuning, model selection,
and ensemble-weight grid search happened on the Training/Val split
only. The splits are persisted to `data/processed/{train,val,test}_split.csv`
for reproducibility.

---

## Group 2 — Custom CNN

### Q6. Explain your CNN architecture.
VGG-style: 4 blocks of two stacked 3 × 3 convolutions (each followed
by BatchNorm and ReLU) plus a 2 × 2 max-pool. Channels double every
block (32 → 64 → 128 → 256), spatial size halves (224 → 112 → 56 →
28 → 14). The final feature map is globally average-pooled and fed
to a 2-layer FC head with dropout (0.5, 0.3). Total: 1.21 M params.

### Q7. Why VGG-style and not residual?
For 1.2 M parameters the network is shallow enough that vanishing
gradients aren't a problem, so residual connections offer little
benefit. The simplicity also makes the architecture easier to explain
and debug.

### Q8. Why two stacked 3 × 3 convs instead of one 5 × 5?
Two 3 × 3 convs have the same 5 × 5 receptive field but use
**18 vs 25 parameters** AND inject an extra ReLU non-linearity for free.

### Q9. Why BatchNorm before ReLU and bias=False in the Conv?
BN before ReLU is the classical ordering. BN learns an affine
transform (scale + shift), so the conv bias is redundant — we set
`bias=False` to save parameters.

### Q10. Why Global Average Pooling instead of flatten?
Flatten on a (256, 14, 14) feature map would produce a 50,176-d
vector → FC head with 6.4 M params and severe overfitting risk. GAP
produces just (256,) — 200 × smaller and intrinsically spatially
invariant.

### Q11. What is the receptive field?
After 4 blocks (each conv adds 2 to RF, each pool doubles it), the
RF is approximately 100 × 100 pixels — about half the input.

---

## Group 3 — Transfer Learning

### Q12. What is transfer learning?
Initializing a model with weights pretrained on a large dataset
(ImageNet, 1.4 M images) then adapting it to a smaller target
dataset. Early layers' generic features (edges, textures) transfer
well; only the late layers need fine-tuning.

### Q13. Why does it help so much on medical imaging?
Medical datasets are small (4–10 K images), while pretrained features
have been trained on millions of natural images. From-scratch ResNet50
on our 4,480 images would overfit catastrophically; pretrained
ResNet50 needs only enough data to retrain its head (~25 K params).

### Q14. Why EfficientNet-B0?
Best accuracy-per-parameter on ImageNet (77.7 % top-1 at 5.3 M),
3× faster than ResNet50 in training, and uses depthwise-separable
convolutions with squeeze-excite — giving us architectural diversity
relative to our vanilla CNN. Fits comfortably in 4 GB VRAM.

### Q15. What is two-stage training?
**Stage 1 (warm-up):** freeze the backbone, train only the new head
with a higher LR (1e-3) for a few epochs. This lets the random head
produce useful gradients before they reach the backbone.
**Stage 2 (fine-tune):** unfreeze the backbone, train the whole
network at a 10 × smaller LR (1e-4) to delicately adapt pretrained
features.

### Q16. What happens if you skip Stage 1?
The random head produces nonsense logits → huge loss → huge
gradients → backprop through the backbone overwrites the precious
pretrained weights → catastrophic forgetting. Your model ends up
worse than starting from scratch.

### Q17. Why 10× smaller LR for Stage 2?
Pretrained weights live near a good minimum of the loss landscape.
We want to *walk* toward our MRI-optimal minimum, not *leap* and
risk destroying useful features.

### Q18. How did you actually freeze the backbone?
We set `requires_grad=False` on all backbone parameters AND call
`backbone.eval()` to freeze BatchNorm running statistics. We also
override the wrapper module's `.train()` method so that when the
training loop calls `model.train()`, the frozen backbone stays in
eval mode while the head goes to train mode.

---

## Group 4 — Swin Transformer

### Q19. How does a Vision Transformer differ from a CNN?
A ViT splits the image into patches (e.g. 16 × 16), flattens each
into a token, and applies stacked Transformer blocks with
self-attention. Each token attends to every other token → global
receptive field from layer 1. But: self-attention is O(N²) in
sequence length, and ViTs have weak inductive bias so they need
much more data than CNNs.

### Q20. What problem does Swin solve?
Vanilla ViT can't scale to high resolution (quadratic compute) and
doesn't produce hierarchical features needed for dense prediction.
Swin solves both with **window attention** (compute attention only
within local 7 × 7 windows → linear cost) and **patch merging**
between stages (halving spatial size, doubling channels — like a CNN).

### Q21. What are shifted windows?
Pure window attention has no cross-window communication. Swin
alternates: layer 2L uses regular windows, layer 2L+1 shifts the
window grid by (M/2, M/2). Tokens that were in different windows
now share a window. Across a few layers, every token can influence
every other token, while compute stays linear.

### Q22. Why does Swin work well on medical imaging?
Tumors are local features whose *context* (which anatomical region
they sit in) matters globally. Swin's hierarchical features +
self-attention capture both. It's also less prone to "texture bias"
than CNNs.

### Q23. Why is your Swin's accuracy lower than EfficientNet's?
Transformers are data-hungry — they shine with 100K+ images. With
only 4,480 training images we don't fully unlock Swin's capacity.
We also kept the fine-tuning budget short (4 epochs at LR=2e-5).
With more epochs it would likely match or exceed EfficientNet, but
the trend was already clear.

---

## Group 5 — Ensemble Learning

### Q24. Why ensemble three models?
Different architectures make different kinds of mistakes. Averaging
their probability outputs reduces error variance, but **only when
the individual errors are uncorrelated**. We chose three models with
maximally different inductive biases (random-init CNN, pretrained
depthwise-sep CNN, pretrained transformer) to maximize that
decorrelation.

### Q25. Soft voting vs weighted averaging vs stacking?
- **Soft voting**: equal-weight average of probabilities. Zero
  parameters; only good when all models are equally accurate.
- **Weighted averaging**: weights tuned by grid search on validation.
  Gives stronger models more influence.
- **Stacking**: a meta-learner (we used logistic regression) trained
  on the concatenated probability vectors. Most flexible but risks
  overfitting on small validation sets.

### Q26. Why did your ensemble NOT beat the best individual model?
**The dominant-model problem.** EfficientNet-B0 dominates with 98.93 %
val acc; CNN is only 90.36 %, Swin 96.70 %. The grid search found
the optimal weights are [0.0, 0.9, 0.1] — i.e., ignore the weakest
model entirely. Averaging with weaker models adds noise that
*cancels* the variance reduction. Ensembles help most when base
models are diverse AND of comparable strength.

### Q27. Is this a failure?
**No** — it's an honest empirical finding consistent with the
literature on ensemble theory. The variance-reduction term in the
ensemble error decomposition is only large when base errors are
uncorrelated; if EfficientNet's mistakes largely overlap with the
others', there's nothing for averaging to cancel.

### Q28. How would you make the ensemble actually beat the best individual?
- Choose base models that are individually weaker but more diverse
  (e.g., 3 ResNets with different seeds + 1 ViT + 1 CNN with
  different training data subsets — bagging).
- Use test-time augmentation (TTA) — average each model's
  predictions over multiple augmentations of the same input.
- Add models trained on different folds of the data (k-fold ensemble).

---

## Group 6 — Training & Optimization

### Q29. Which optimizer and why?
AdamW with weight decay 1e-4. AdamW = Adam + decoupled weight decay,
which empirically generalizes slightly better than vanilla Adam.
Adaptive per-parameter LR makes it converge in ~⅓ the epochs of
SGD-with-momentum on small datasets like ours.

### Q30. What learning rate schedule?
Cosine annealing from `lr` to `lr × 1e-2` over `T_max=epochs`. Large
LR early to escape bad minima, tiny LR late to fine-tune the
minimum. Requires only one hyperparameter (epoch budget) — strong
default.

### Q31. What's mixed precision and why use it?
Forward pass uses lower-precision (fp16 or bf16) tensors; weights
and gradients stay in fp32. This gives ~2× speedup and ~40 % VRAM
reduction on Tensor-Core-equipped GPUs. We use **bfloat16** instead
of fp16 because we hit a known cuDNN NaN bug with fp16 on Turing GPUs.

### Q32. What is early stopping?
Stop training when validation loss stops improving for N consecutive
epochs (patience). Prevents wasting compute and avoids overfitting
beyond the optimal epoch. We saved the best checkpoint (by val_acc)
and reloaded it at the end of training.

### Q33. What data augmentations did you use?
Resize → HorizontalFlip → Rotation(±15°) → Affine(translate±5%,
scale±5%) → ColorJitter(brightness±20%, contrast±20%) → ToTensor →
Normalize → RandomErasing(p=0.25). All chosen to be *clinically
plausible* — no vertical flips (impossible head orientation), no
hue jitter (MRI is grayscale).

### Q34. Why didn't you use vertical flip?
An upside-down brain is never seen in clinical practice. Teaching
the model that "flipping the head upside down doesn't change the
diagnosis" would be teaching an impossible invariance and could hurt
real-world accuracy.

---

## Group 7 — Evaluation

### Q35. What metrics did you report and why?
- **Accuracy** — overall correctness (meaningful because classes are
  balanced).
- **Per-class precision/recall/F1** — exposes class-specific weakness.
- **Macro ROC-AUC** — ranking quality regardless of decision threshold.
- **Confusion matrices** — shows *which* classes are confused with
  which.
- **Inference latency + throughput** — deployment-relevant.

### Q36. Why is the test set untouched until the very end?
If you tune even one hyperparameter on the test set, your reported
test accuracy becomes optimistic — it's no longer measuring
generalization. This is the #1 sin in academic ML; we avoid it by
architectural separation of `build_splits()` (which only sees
`Training/`) from anything that touches `Testing/`.

### Q37. What's your val-to-test accuracy drop and why?
4–8 percentage points across all models. This is **normal** — val
accuracy reports the *best epoch* we saw during training, so it
overstates true accuracy. The test set, untouched, gives the honest
number. The CNN drops the most (−7.6 pp) because it was trained from
scratch and overfit the training distribution slightly.

### Q38. What does macro ROC-AUC mean for a 4-class problem?
We compute one-vs-rest binary AUCs per class then average. Each
binary AUC tells us how well the model *ranks* positives above
negatives for that class — independent of any decision threshold.
All our models achieve macro AUC > 0.95 even when accuracy is
lower, meaning their probability ordering is excellent.

---

## Group 8 — Explainability (Grad-CAM)

### Q39. What is Grad-CAM?
Gradient-weighted Class Activation Mapping. For a chosen class c, we
compute the gradient of c's logit with respect to a chosen conv
layer's feature map, average those gradients spatially to get
channel weights, then take a weighted sum of the feature-map
channels. The result is a low-resolution heatmap showing where
positive evidence for c was located. Upsampled and overlaid on the
input.

### Q40. Why is Grad-CAM important for medical AI?
Regulatory bodies (FDA, EU MDR) increasingly require explanations
alongside predictions. A radiologist must be able to verify the
model is reasoning about the tumor, not about a scanner watermark
or skull artifact. Grad-CAM is the simplest, most established way
to do this.

### Q41. How did you apply Grad-CAM to the Swin Transformer?
Swin doesn't have channels-first conv activations — its intermediate
tensors are (B, H, W, C) channels-last. We target the last block's
`norm2` layer (where 7 × 7 × 768 spatial structure is intact) and
apply a `reshape_transform` that permutes (B, H, W, C) → (B, C, H, W)
so Grad-CAM's spatial-pooling logic works.

### Q42. What did your Grad-CAM analysis reveal?
- On *correctly* classified samples, all models attend to
  anatomically meaningful regions — not artifacts.
- EfficientNet's heatmaps are the most spatially focused.
- Swin's heatmaps are the most diffuse — consistent with
  self-attention's wide effective receptive field.
- *Misclassified* samples fall into 3 explainable failure modes:
  genuine glioma/meningioma ambiguity (visually similar masses),
  subtle/missed tumors, and out-of-distribution scan styles (CT).

---

## Group 9 — Engineering & Deployment

### Q43. Walk me through your project structure.
- `src/` is the production package — every module has one
  responsibility (data, transforms, models, training, evaluation,
  inference, explainability).
- `notebooks/` is the teaching narrative — each notebook imports
  from `src/` and runs end-to-end.
- `app/app.py` is the Streamlit UI; `app/api.py` is FastAPI.
- `models/` holds trained checkpoints + ONNX exports +
  ensemble config.
- `outputs/` holds plots, confusion matrices, Grad-CAM, and CSVs.
- `setup.ps1` provisions the venv from scratch in one command.

### Q44. How is the project reproducible?
- Fixed seed (42) for all RNGs.
- Pinned dependency versions in `requirements.txt`.
- Deterministic stratified split persisted as CSV.
- All training notebooks committed with embedded outputs.
- `setup.ps1` recreates the entire environment from scratch.

### Q45. How does the Streamlit UI work internally?
`@st.cache_resource` loads the `BrainTumorPredictor` exactly once
per session (saves ~5 s per request). User upload → `predict()` →
returns a `PredictionResult` dataclass → UI renders headline class,
confidence, threshold warning, per-class probability bar chart,
per-model probability heat-table, and 3 Grad-CAM overlays.

### Q46. What does the FastAPI backend do?
`POST /predict` accepts a multipart image upload and returns JSON
with the predicted class, confidence, ensemble probabilities,
per-model probabilities, and inference time. Optionally returns
base64-encoded Grad-CAM PNGs.

### Q47. What's ONNX and why export to it?
ONNX (Open Neural Network Exchange) is a framework-agnostic graph
representation. Once exported, you can run inference via
`onnxruntime` — a C++ engine with Python/C++/C#/Java/JS/mobile
bindings — without PyTorch. On CPU, our ONNX models are
**2.0–3.5× faster** than PyTorch eager mode, with verified 100 %
argmax agreement.

### Q48. How would you deploy this to a hospital?
1. Containerize with our `Dockerfile`.
2. Deploy to a Kubernetes cluster behind a load balancer (multiple
   replicas of the FastAPI container).
3. Add authentication (OIDC/JWT), HTTPS termination at the ingress,
   and audit logging of every prediction (for clinical traceability).
4. Wire to the hospital's PACS via a DICOM bridge.
5. Add HIPAA-compliant data handling (encryption at rest +
   in-transit, access controls).

---

## Group 10 — Critical & forward-looking

### Q49. What are the limitations of your approach?
- 2D slice classification — doesn't use 3D volumetric information.
- Only one MRI sequence (T1) — clinical reading uses T1, T2, FLAIR,
  contrast together.
- Trained on one public dataset — cross-institution validation
  needed.
- Classification only, no segmentation.
- 4-class assumption — real radiology has many more diagnoses.

### Q50. What would you do differently with more time?
- 3D models (3D-CNN, V-Net, or Swin-UNETR) using volumetric input.
- Multi-modal fusion across T1/T2/FLAIR sequences.
- Self-supervised pretraining (e.g., MAE) on unlabelled MRIs before
  supervised fine-tuning.
- Test-time augmentation for higher robustness.
- Cross-validation instead of a single train/val split.
- A radiologist-in-the-loop annotation pipeline.

### Q51. What's your biggest engineering lesson from this project?
**Mixed-precision training is hardware-dependent.** I hit a cuDNN
NaN bug with fp16 on the GTX 1650 that took ~30 minutes to diagnose
with layer-by-layer activation tracing. Switching to bfloat16 (same
fp32 dynamic range, lower precision) fixed it. Lesson: don't trust
that "AMP just works" on every GPU — verify with a smoke test.

### Q52. If you had to defend ONE design decision, what would it be?
**Bilingual ensemble combiners (soft voting + weighted + stacking).**
Trying all three lets us report an honest "ensemble does not beat
the best individual" finding rather than cherry-picking one method.
This empirical honesty is more valuable for the report than a
fabricated 99 % accuracy claim.

---

*Read this file the night before the viva. Try to explain each answer
out loud — if you can rephrase it in your own words, you've internalized
it. Good luck!*
