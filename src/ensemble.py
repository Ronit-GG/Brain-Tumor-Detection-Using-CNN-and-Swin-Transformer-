"""
Ensemble combinators for Custom CNN + Transfer (EfficientNet-B0) + Swin.

Three methods implemented:

* `soft_voting`        : equal-weight probability average
* `weighted_average`   : weighted probability average (weights found
                         by grid search on validation set)
* `StackingEnsemble`   : logistic-regression meta-learner on the
                         concatenated probability vectors

All methods operate on **probability arrays** rather than re-running
the models, so they're decoupled from inference cost.

Convention
----------
A probability array has shape (N, C) where N = #samples, C = #classes.
"prob_list" means a list/tuple of M such arrays, one per base model,
aligned along axis 0 (same sample index across models).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import evaluate


# --------------------------------------------------------------------- #
# 1. Probability collection (run each base model once per loader)       #
# --------------------------------------------------------------------- #
@torch.no_grad()
def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    desc: str = "collect",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run `model` over `loader` and return (probs, labels).

    probs : float32 array of shape (N, C)
    labels: int64   array of shape (N,)
    """
    criterion = nn.CrossEntropyLoss()
    _, _, logits, labels = evaluate(
        model, loader, criterion, device,
        use_amp=use_amp, amp_dtype=amp_dtype,
        desc=desc, return_predictions=True,
    )
    probs = torch.softmax(logits, dim=-1).numpy().astype(np.float32)
    return probs, labels.numpy().astype(np.int64)


# --------------------------------------------------------------------- #
# 2. Combiners                                                          #
# --------------------------------------------------------------------- #
def soft_voting(prob_list: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight probability average. Returns (N, C)."""
    return np.stack(prob_list, axis=0).mean(axis=0)


def weighted_average(
    prob_list: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """
    Probability-weighted average. Weights are renormalized to sum to 1.
    Returns (N, C).
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.sum() <= 0:
        raise ValueError(f"weights must sum to > 0, got {weights}")
    w = w / w.sum()
    stacked = np.stack(prob_list, axis=0)              # (M, N, C)
    return (stacked * w[:, None, None]).sum(axis=0)    # (N, C)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def find_optimal_weights(
    prob_list: Sequence[np.ndarray],
    labels: np.ndarray,
    step: float = 0.05,
) -> tuple[np.ndarray, float]:
    """
    Grid-search non-negative weights summing to 1 that maximize accuracy.

    For M=3 models with step=0.05 there are exactly 231 combinations,
    so this is instant.
    """
    M = len(prob_list)
    if M != 3:
        raise NotImplementedError("Grid search is implemented for M=3 only.")

    best_w = None
    best_acc = -1.0
    grid = np.arange(0.0, 1.0 + step / 2, step)

    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            w3 = max(0.0, w3)
            probs = weighted_average(prob_list, [w1, w2, w3])
            acc = accuracy(probs, labels)
            if acc > best_acc:
                best_acc = acc
                best_w = np.array([w1, w2, w3], dtype=np.float32)

    return best_w, best_acc


# --------------------------------------------------------------------- #
# 3. Stacking (sklearn LogisticRegression meta-learner)                 #
# --------------------------------------------------------------------- #
class StackingEnsemble:
    """
    Logistic-regression meta-learner trained on concatenated
    probability vectors [p_cnn; p_tl; p_swin] -> class.
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter
        self.clf = None

    def fit(self, prob_list: Sequence[np.ndarray], labels: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        X = np.concatenate(prob_list, axis=1)         # (N, M*C)
        self.clf = LogisticRegression(max_iter=self.max_iter).fit(X, labels)
        return self

    def predict_proba(self, prob_list: Sequence[np.ndarray]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Call .fit(...) before .predict_proba(...)")
        X = np.concatenate(prob_list, axis=1)
        return self.clf.predict_proba(X).astype(np.float32)

    def predict(self, prob_list: Sequence[np.ndarray]) -> np.ndarray:
        return self.predict_proba(prob_list).argmax(axis=1)


# --------------------------------------------------------------------- #
# 4. Config save / load                                                 #
# --------------------------------------------------------------------- #
@dataclass
class EnsembleConfig:
    """Serializable record of the chosen ensemble scheme."""
    method:       str                              # 'soft_voting' | 'weighted' | 'stacking'
    model_names:  list[str]
    weights:      Optional[list[float]] = None     # for weighted only
    metrics:      Optional[dict]        = None
    notes:        Optional[str]         = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EnsembleConfig":
        with open(path) as f:
            return cls(**json.load(f))


# --------------------------------------------------------------------- #
# 5. Convenience: load all 3 trained models                             #
# --------------------------------------------------------------------- #
def load_all_models(
    device: torch.device,
    cnn_path:      Path,
    transfer_path: Path,
    swin_path:     Path,
) -> dict[str, nn.Module]:
    """
    Re-instantiate the 3 architectures and load their saved state_dicts.

    All returned models are in eval() mode on `device`.
    """
    from .cnn_model      import build_cnn
    from .transfer_model import build_transfer_model
    from .swin_model     import build_swin_model

    cnn = build_cnn()
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn = cnn.to(device).eval()

    tl = build_transfer_model(freeze_backbone=False)
    tl.load_state_dict(torch.load(transfer_path, map_location=device))
    tl = tl.to(device).eval()

    swin = build_swin_model(freeze_backbone=False)
    swin.load_state_dict(torch.load(swin_path, map_location=device))
    swin = swin.to(device).eval()

    return {"cnn": cnn, "transfer": tl, "swin": swin}
