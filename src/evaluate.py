"""
Evaluation utilities: classification report, confusion matrix, per-class
metrics, ROC-AUC, and inference-speed profiling.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------------------------- #
# Probability helpers                                                   #
# --------------------------------------------------------------------- #
def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """Softmax along the last axis; returns NumPy float32."""
    return torch.softmax(logits, dim=-1).numpy().astype(np.float32)


def logits_to_preds(logits: torch.Tensor) -> np.ndarray:
    """Argmax along the last axis; returns NumPy int64."""
    return logits.argmax(dim=-1).numpy().astype(np.int64)


# --------------------------------------------------------------------- #
# Classification report                                                 #
# --------------------------------------------------------------------- #
def build_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> tuple[str, dict]:
    """Return (pretty text report, dict-of-floats report)."""
    text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0,
    )
    obj  = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    return text, obj


def per_class_metrics_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """One row per class with precision / recall / F1 / support."""
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0,
    )
    return pd.DataFrame({
        "class":     list(class_names),
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "support":   sup,
    })


# --------------------------------------------------------------------- #
# Confusion matrix                                                      #
# --------------------------------------------------------------------- #
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    normalize: bool = True,
    title: str = "Confusion matrix",
    save_path: Optional[Path] = None,
):
    """Render a confusion matrix as a heatmap. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_display = cm_norm
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_display,
        annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, ax=ax, square=True, linewidths=0.4, linecolor="gray",
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


# --------------------------------------------------------------------- #
# ROC-AUC (multi-class: one-vs-rest)                                    #
# --------------------------------------------------------------------- #
def macro_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro-averaged one-vs-rest ROC-AUC."""
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


def per_class_auc(y_true: np.ndarray, probs: np.ndarray,
                  class_names: Sequence[str]) -> pd.DataFrame:
    """One-vs-rest ROC-AUC per class."""
    rows = []
    for c in range(len(class_names)):
        y_c = (y_true == c).astype(int)
        rows.append({
            "class": class_names[c],
            "auc":   float(roc_auc_score(y_c, probs[:, c])),
        })
    return pd.DataFrame(rows)


def plot_multimodel_roc(
    probs_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[Path] = None,
):
    """
    One subplot per class, one ROC curve per model. Each subplot's title
    includes the per-class AUC for every model.
    """
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4),
                             sharey=True)
    if n_classes == 1:
        axes = [axes]

    for c in range(n_classes):
        y_c = (y_true == c).astype(int)
        ax = axes[c]
        for name, probs in probs_dict.items():
            fpr, tpr, _ = roc_curve(y_c, probs[:, c])
            auc = roc_auc_score(y_c, probs[:, c])
            ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.4f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(class_names[c], fontweight="bold")
        ax.set_xlabel("False positive rate")
        if c == 0:
            ax.set_ylabel("True positive rate")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("ROC curves (one-vs-rest) per class",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------- #
# Inference-speed profiling                                             #
# --------------------------------------------------------------------- #
@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    n_warmup: int = 10,
    n_iter:   int = 100,
    image_size: int = 224,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Measure forward-pass latency. Returns a dict with batch size, mean/p50/p95
    latency in ms, and throughput in images/sec.

    Uses random inputs (since data loading is a separate concern from raw
    model compute). Synchronizes around each iteration on CUDA so timings
    aren't distorted by async kernel queueing.
    """
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    autocast_kw = dict(device_type=device.type, enabled=use_amp, dtype=amp_dtype)

    # warm-up: triggers cuDNN auto-tune and JIT compilation
    for _ in range(n_warmup):
        with torch.amp.autocast(**autocast_kw):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # timed iterations
    times_ms: list[float] = []
    for _ in range(n_iter):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(**autocast_kw):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms)
    return {
        "batch_size":              batch_size,
        "n_iter":                  n_iter,
        "mean_ms":                 float(arr.mean()),
        "std_ms":                  float(arr.std()),
        "p50_ms":                  float(np.percentile(arr, 50)),
        "p95_ms":                  float(np.percentile(arr, 95)),
        "throughput_imgs_per_sec": float(batch_size / (arr.mean() / 1000.0)),
    }


# --------------------------------------------------------------------- #
# Training-time helper                                                  #
# --------------------------------------------------------------------- #
def total_training_seconds(history_paths: Sequence[Path]) -> float:
    """Sum `epoch_time` across one or more saved history JSON files."""
    import json
    total = 0.0
    for p in history_paths:
        h = json.loads(Path(p).read_text())
        total += float(sum(h.get("epoch_time", [])))
    return total
