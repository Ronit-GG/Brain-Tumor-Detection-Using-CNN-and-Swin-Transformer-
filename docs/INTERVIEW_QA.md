# Deep-Learning Interview Q&A — Grounded in This Project

The advantage of building a real project is that you can answer every
classical interview question with a concrete example. Each answer below
references something we actually did, so you're not reciting textbook
phrases — you're recounting your own engineering experience.

---

## A. Fundamentals

### A1. Explain the bias–variance tradeoff.
- **Bias** = error from incorrect modelling assumptions (underfitting).
- **Variance** = error from sensitivity to small fluctuations in the
  training set (overfitting).
- Increasing model capacity reduces bias but raises variance.

> *In our project:* the custom CNN (1.2 M params) had higher bias
> but lower variance; EfficientNet-B0 (4 M params, pretrained) had
> lower bias and similar variance because pretraining provides strong
> priors. The ensemble averages probability outputs, which reduces
> variance — but only when base errors are uncorrelated.

### A2. What's the difference between L1, L2, and weight decay?
- **L1** adds `λ Σ|w|` to the loss → sparsity (some weights → 0).
- **L2** adds `λ Σw²` to the loss → small weights everywhere.
- **Weight decay** is L2 applied via the optimizer step, *decoupled*
  from the gradient. In AdamW, decay isn't scaled by the adaptive LR
  — this is why AdamW often generalizes better than Adam + L2.

> *In our project:* we used AdamW with `weight_decay=1e-4`.

### A3. What is dropout and why does it work?
At training time, randomly zero `p` fraction of neurons in a layer.
At test time, scale activations by `(1-p)`. Effect: forces neurons
not to co-adapt; each subnetwork has to learn a useful representation
on its own. Geometrically: approximates an ensemble of all
`2^N` subnetworks.

> *In our project:* dropout `p=0.5` between the GAP and FC layers
> of the custom CNN; `p=0.3` before the final Linear of the
> transfer model. We did NOT use dropout inside the conv blocks
> because BatchNorm already regularizes them.

### A4. What is BatchNorm? What problem does it solve?
For each minibatch, normalize each channel to mean 0, variance 1,
then apply a learnable affine (γ, β) per channel. Solves **internal
covariate shift** (the distribution of layer inputs changing as
upstream weights update). Effects: faster convergence, allows higher
LR, mild regularization (the minibatch noise acts like dropout).

> *In our project:* after every conv layer in the custom CNN, with
> `bias=False` in the conv because BN's β makes the bias redundant.

### A5. BatchNorm vs LayerNorm vs GroupNorm — when to use which?
- **BN**: normalize across batch, per channel. Needs sufficient
  batch size (≥16). Standard for CNNs.
- **LN**: normalize across all channels, per sample. Independent of
  batch size. Standard for transformers (where token sequences
  vary).
- **GN**: normalize within groups of channels per sample. Compromise
  for small-batch CNN training (e.g. detection).

> *In our project:* BN in the custom CNN and EfficientNet; LN inside
> the Swin Transformer (between attention and MLP blocks).

### A6. Why does ReLU work? Why not sigmoid in hidden layers?
- ReLU = `max(0, x)`. Gradient is 1 for x > 0, 0 otherwise — never
  saturates for positive inputs, so no vanishing gradient in deep
  networks.
- Sigmoid saturates for |x| > 5, gradients become ~0 → vanishing
  gradient → deep networks can't learn.

> *In our project:* ReLU everywhere in our custom CNN's conv blocks;
> EfficientNet uses Swish/SiLU (smoother variant); Swin uses GELU.

---

## B. Optimization & Training

### B1. SGD vs Adam vs AdamW — when to use which?
- **SGD + momentum**: best generalization on large datasets when
  tuned well; many production training recipes use it.
- **Adam**: adaptive per-parameter LR via moving averages of first
  and second moments. Converges fast with little tuning. Slightly
  worse generalization than SGD on big datasets.
- **AdamW**: Adam with decoupled weight decay → better
  generalization than Adam, fast convergence like Adam.

> *In our project:* AdamW because our dataset is small (4,480
> images), Adam-style adaptation converges much faster than SGD,
> and decoupled WD gives the generalization edge.

### B2. Explain learning rate scheduling.
A learning-rate schedule changes the LR over training. Common ones:
- **Step decay**: divide LR by k at fixed epochs.
- **Cosine annealing**: smoothly decrease LR from `lr_max` to
  `lr_min` along a half-cosine.
- **Cyclical / OneCycle**: increase then decrease LR within one
  training run.

> *In our project:* `CosineAnnealingLR` with `eta_min = lr × 1e-2`,
> so LR smoothly decays to 1% of initial. No hand-tuning of decay
> points required.

### B3. What is mixed-precision training? What's the catch?
Forward pass in lower-precision (fp16 or bf16); weights and
optimizer states in fp32. Gives ~2× speedup and ~40% VRAM
reduction on Tensor-Core-equipped GPUs.

The **catch**: fp16 has limited dynamic range (max ~65504), so
gradients can overflow → NaN. The standard fix is a `GradScaler`
that multiplies the loss before backward, then divides it back out.

> *In our project:* we initially used fp16 and hit a cuDNN NaN bug
> on the GTX 1650 (Turing arch) with `Conv2d(64, 64, 3)`. We
> diagnosed it via layer-by-layer activation tracing, then switched
> to bfloat16 — same fp32 dynamic range, lower precision — and no
> GradScaler needed.

### B4. What is early stopping?
Halt training when a chosen metric stops improving for `patience`
consecutive epochs. Common metric: validation loss. Combine with
best-checkpoint saving so the final model is the *peak* of the val
curve, not where training happened to stop.

> *In our project:* `patience=5` for the transfer model and Swin,
> `patience=5–7` for CNN. Best checkpoint saved by val accuracy.

### B5. What is gradient clipping?
Cap the L2 norm of gradients at a threshold before the optimizer
step. Prevents exploding gradients in RNNs and (sometimes) deep
transformers. Less critical for modern CNN architectures with BN.

> *In our project:* available via `grad_clip_norm` parameter in
> `fit()` but not enabled — our models didn't need it.

---

## C. Computer Vision

### C1. Convolution vs fully-connected layer — what's the difference?
- **Conv**: weight-sharing across spatial positions, locality,
  translation equivariance. A 3 × 3 conv with 64 output channels
  on 224 × 224 RGB input has `3 × 64 × 9 + 64 = 1,792` params.
- **FC** flattening 224 × 224 × 3 → 64 needs `150,528 × 64 = ~9.6 M`
  params. 5000× more!

> *In our project:* we used GAP instead of flatten in the CNN
> specifically to avoid massive FC layers.

### C2. Receptive field — what is it and how do you grow it?
The region of the input that influences a given pixel in a feature
map. Grows by:
- Larger kernel (kxk → RF increases by k-1)
- More layers
- Stride > 1 or pooling
- Dilated convs (atrous)

> *In our project:* after 4 blocks the CNN's RF is ~100 × 100 px;
> Swin gets a much larger effective RF much sooner via self-attention.

### C3. What's the purpose of pooling?
Downsampling spatial dimensions → fewer params/compute in later
layers, mild translation invariance. Max-pool emphasizes the
strongest activation in each window; average-pool gives the average.

### C4. What is Global Average Pooling and why use it?
Average each feature map down to a single scalar → output is
`(C,)`. Used after the last conv layer of modern CNNs to replace
flatten + FC head: drastically fewer parameters, intrinsic spatial
invariance, less overfitting.

> *In our project:* GAP at the end of our CNN reduces feature
> dimensionality from 256 × 14 × 14 = 50,176 to just 256.

### C5. Explain residual connections.
Add the input of a block to its output: `y = F(x) + x`. Solves the
*degradation problem* in very deep networks: even when adding
layers should at worst do nothing (identity), without skips the
optimization is too hard. Skip connections make identity trivial,
so adding layers can only help.

### C6. What is a depthwise-separable convolution?
A standard conv applies KxK filters across all C input channels.
A depthwise-separable conv does it in 2 steps:
1. **Depthwise**: one KxK filter per input channel.
2. **Pointwise (1x1 conv)**: combines channels.

Same expressive power, ~9× fewer parameters and FLOPs (for K=3,
C=large). Used throughout MobileNet, EfficientNet.

> *In our project:* EfficientNet-B0 uses depthwise-separable convs
> in its MBConv blocks — this is why it has only 4 M params despite
> rivalling ResNet50's accuracy.

---

## D. Transfer Learning

### D1. Why does ImageNet pretraining transfer to medical images?
Early CNN layers learn generic features (edges, textures, blobs)
that are universal across image types. Only the late layers and
classifier need to be retrained for the new domain.

### D2. What's the right LR for fine-tuning a pretrained model?
Usually 10× smaller than for from-scratch training. Pretrained
weights live near a good minimum — small updates *walk* there;
large updates can leap out.

### D3. Should you fine-tune or use as a fixed feature extractor?
- Tiny dataset (< 1K) → fixed feature extractor + train a new head.
- Medium dataset (1K–10K) → two-stage: head-only first, then
  fine-tune the whole backbone at low LR.
- Large dataset (> 100K) → fine-tune everything from epoch 1.

> *In our project:* 4,480 images → two-stage protocol.

---

## E. Transformers

### E1. Explain self-attention.
For each token in a sequence, compute a weighted sum of all other
tokens. Weights come from `softmax(QK^T / √d)`. Q, K, V are learned
linear projections of the input.

### E2. Why is vanilla ViT bad for high-resolution images?
Self-attention is O(N²) in sequence length. For a 224 × 224 image
with 16 × 16 patches → 196 tokens → 38,416 pairwise comparisons.
For 4 × 4 patches → 3,136 tokens → 9.8M comparisons → infeasible.

### E3. How does Swin solve this?
**Window attention**: compute attention only within local 7 × 7
windows → cost linear in image area. **Shifted windows**: alternate
window grid position so information flows between windows. Result:
global receptive field after a few layers, linear compute.

### E4. Why do transformers need more data than CNNs?
CNNs have strong inductive biases (locality, translation
equivariance) baked in. Transformers learn these from scratch from
the data. Without enough data, they can't.

> *In our project:* Swin (27 M params, pretrained) achieved 96.7 %
> val acc, slightly below EfficientNet (4 M, pretrained, 98.9 %).
> With more training data Swin would likely surpass it.

---

## F. Evaluation

### F1. Why are accuracy alone insufficient as a metric?
On imbalanced data, a model that predicts the majority class always
can achieve high accuracy with zero usefulness. Even on balanced
data, accuracy doesn't tell you *which* classes are confused.

### F2. Explain precision, recall, F1.
- **Precision** = TP / (TP + FP) = "of predicted positives, how
  many are right?"
- **Recall** = TP / (TP + FN) = "of true positives, how many did
  we catch?"
- **F1** = harmonic mean of precision and recall.

### F3. Explain ROC-AUC.
ROC curve plots TPR vs FPR as the decision threshold sweeps from 0
to 1. AUC = area under this curve. AUC = 0.5 → random; AUC = 1 →
perfect ranking of positives above negatives. **Threshold-free**
measure of discrimination quality.

> *In our project:* macro AUC (averaged across one-vs-rest binary
> AUCs) is reported for all 4 models. EfficientNet's macro AUC =
> 0.991 means it almost perfectly ranks each class's positives.

### F4. ROC-AUC vs PR-AUC — when to prefer which?
- **ROC** is misleading on imbalanced data — TPR/FPR don't account
  for class skew.
- **PR-AUC** focuses on the positive class — better for highly
  imbalanced problems (e.g., rare-disease detection).

> *In our project:* classes are balanced, so ROC-AUC is fine.

### F5. Why do you need train/val/test (not just train/test)?
- **Train**: update weights.
- **Val**: tune hyperparameters, pick best checkpoint, early-stop.
- **Test**: report final metric. **Touched once.**

If you tune hyperparameters on test, your reported number leaks
that tuning effort and overstates true generalization.

---

## G. Ensembles

### G1. Why do ensembles often outperform individuals?
The squared error of the average satisfies
`E[||avg_err||²] = Σ w_i² E[||err_i||²] + cross_terms`.
When base errors are uncorrelated, the cross-terms are near zero
and the ensemble error is smaller than the average individual
error. **Variance reduction without bias increase.**

### G2. When do ensembles NOT help?
- When base models are *correlated* (similar architectures, same
  data, same seed) → averaging cancels nothing.
- When one model dominates → averaging with weaker models adds
  noise. (We documented this in our project.)
- When the dominant model is already near the data's noise floor.

### G3. Soft vs hard voting?
- **Hard**: each model votes for one class; majority wins. Loses
  confidence information.
- **Soft**: average probabilities → preserves confidence.
  Almost always better.

### G4. What's stacking?
Train a meta-learner (logistic regression, gradient boosting, etc.)
on the concatenated probability outputs of base models. More
flexible than weighted averaging but requires training data the
meta-learner hasn't seen — easy to overfit.

> *In our project:* logistic-regression stacking evaluated via
> 5-fold CV on the val set; result was 98.57 % — close but didn't
> beat the weighted average (98.93 %).

---

## H. Explainability

### H1. What's Grad-CAM in one sentence?
Class-discriminative heatmap formed by weighting a chosen conv
layer's feature maps by the spatially-averaged gradient of the
target class's score with respect to those feature maps, then
ReLU + upsampling.

### H2. Why ReLU on the heatmap?
Negative contributions (regions that *decrease* class confidence)
are usually not what you want to visualize for "where did the
model see this class?". ReLU keeps only positive evidence.

### H3. Grad-CAM on transformers — what changes?
Transformers don't have channels-first conv activations. You target
a layer with spatial structure (e.g. the last block's `norm2` in
Swin), apply a `reshape_transform` to convert (B, H, W, C) to
(B, C, H, W), then Grad-CAM works as usual.

### H4. What's a known failure mode of Grad-CAM?
Class-discriminative but **not** sensitive to model errors — if the
model is wrong, Grad-CAM still shows you the regions that drove the
*wrong* prediction, which can be misleading. SmoothGrad, Integrated
Gradients, and attention rollout are alternatives.

---

## I. Production / Engineering

### I1. How would you serve a PyTorch model at scale?
- Convert to ONNX or TorchScript for runtime portability.
- Wrap with FastAPI behind a load balancer.
- Containerize with Docker; orchestrate with Kubernetes.
- Use ONNX Runtime or NVIDIA Triton for inference.
- Add caching for repeat inputs (e.g., LRU on image hashes).
- Monitor latency, throughput, and prediction drift.

### I2. How do you reduce inference latency?
- Quantization (int8 → ~4× speedup, mild accuracy loss)
- Pruning (zero out small weights)
- Knowledge distillation (train a smaller "student")
- Compile to TorchScript / ONNX / TensorRT
- Batch requests
- Run on hardware with Tensor Cores

> *In our project:* ONNX exports gave us 2.0–3.5× speedup on CPU
> with verified identical predictions at batch=1.

### I3. How do you keep a model from getting stale?
- Monitor prediction distribution drift in production.
- Schedule periodic retraining with fresh data.
- Use canary deployments: route a small % of traffic to the new
  model, compare metrics before full rollout.

### I4. How do you debug a model that's not training?
1. Verify data is correctly loaded (shapes, ranges, labels).
2. Overfit a single batch — if it can't, the architecture is broken.
3. Check gradient flow — if some layers have zero gradient,
   you've detached them somewhere.
4. Lower LR — exploding loss often = LR too high.
5. Disable AMP — fp16 NaN bugs.

> *In our project:* exactly step 5 saved us — fp16 caused NaN
> in conv2d on Turing GPUs; switching to bf16 fixed it.

---

## J. The "tell me about a project" question

### J1. *"Walk me through a project you're proud of."*
Example 60-second pitch:

> "I built a brain tumor MRI classifier that combines a custom CNN
> trained from scratch, an EfficientNet-B0 with transfer learning,
> and a Swin Transformer into a weighted ensemble. The hardest part
> wasn't the modelling — it was diagnosing a fp16 NaN bug from
> cuDNN on my GTX 1650, which I traced to a specific 64x64 conv
> layer with layer-by-layer activation logging and fixed by
> switching to bfloat16. The interesting finding was empirical: my
> ensemble matched but didn't beat my best individual model (94.9 %
> test accuracy), which I now understand is the *dominant-model
> problem* — variance reduction needs comparable-strength base
> learners. I shipped four interfaces — Streamlit, FastAPI, ONNX
> Runtime, Docker — all wrapping the same checkpoints, and added
> Grad-CAM explainability for clinical trust."

That's 5 sentences, hits: problem, methods, debugging story,
honest negative finding, deployment, and explainability — exactly
what interviewers want to hear.

---

*Internalize the examples, not just the definitions. The point is that
you can speak from experience, not from a textbook.*
