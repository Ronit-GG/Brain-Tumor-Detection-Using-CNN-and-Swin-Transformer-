"""
Reusable training engine for the Brain Tumor Detection project.

Designed to be MODEL-AGNOSTIC: the same `fit()` call trains the custom
CNN, the transfer-learning model, and the Swin Transformer (STEPs 7,
10, 13). Each model only needs to be a `torch.nn.Module` returning
logits of shape (B, num_classes).

Features
--------
* Automatic Mixed Precision (AMP) via torch.amp
* AdamW optimizer + CosineAnnealingLR scheduler
* Early stopping based on validation loss
* Best-checkpoint saving (by validation accuracy)
* TensorBoard logging (scalars: loss/acc/lr)
* tqdm progress bars per epoch
* Returns a JSON-serializable history dict
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import EPOCHS, LR, LOGS_DIR, MODELS_DIR, WEIGHT_DECAY
from .utils import get_device, set_seed


# --------------------------------------------------------------------- #
# History container                                                     #
# --------------------------------------------------------------------- #
@dataclass
class TrainHistory:
    """JSON-friendly container for per-epoch metrics."""
    train_loss: list[float] = field(default_factory=list)
    train_acc:  list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_acc:    list[float] = field(default_factory=list)
    lr:         list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------- #
# Single-epoch loops                                                    #
# --------------------------------------------------------------------- #
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    Optional[torch.amp.GradScaler],
    device:    torch.device,
    use_amp:   bool,
    amp_dtype: torch.dtype = torch.bfloat16,
    desc:      str = "train",
) -> tuple[float, float]:
    """Run one epoch of training; return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct      = 0
    seen         = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss   = criterion(logits, labels)

        # GradScaler is only needed for fp16 (bf16 has fp32 range, no underflow).
        need_scaler = use_amp and scaler is not None and amp_dtype == torch.float16
        if need_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs            = labels.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        seen         += bs

        pbar.set_postfix(
            loss=f"{running_loss / seen:.4f}",
            acc =f"{correct / seen:.4f}",
        )

    return running_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    desc:      str  = "val",
    return_predictions: bool = False,
):
    """Run one full evaluation pass; return (avg_loss, accuracy) plus
    optionally (logits, labels) tensors collected over the whole loader."""
    model.eval()
    running_loss   = 0.0
    correct        = 0
    seen           = 0
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss   = criterion(logits, labels)

        bs            = labels.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        seen         += bs

        if return_predictions:
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())

        pbar.set_postfix(
            loss=f"{running_loss / seen:.4f}",
            acc =f"{correct / seen:.4f}",
        )

    avg_loss = running_loss / seen
    accuracy = correct / seen

    if return_predictions:
        return avg_loss, accuracy, torch.cat(all_logits), torch.cat(all_labels)
    return avg_loss, accuracy


# --------------------------------------------------------------------- #
# Full training driver                                                  #
# --------------------------------------------------------------------- #
def fit(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    *,
    epochs:        int    = EPOCHS,
    lr:            float  = LR,
    weight_decay:  float  = WEIGHT_DECAY,
    patience:      int    = 7,
    use_amp:       bool   = True,
    amp_dtype:     torch.dtype = torch.bfloat16,
    run_name:      str    = "run",
    checkpoint_path: Optional[Path] = None,
    log_dir:         Optional[Path] = None,
    history_path:    Optional[Path] = None,
    seed:            int    = 42,
    optimizer_kwargs: Optional[dict] = None,
    grad_clip_norm:  Optional[float] = None,
) -> tuple[nn.Module, TrainHistory, float]:
    """
    Train `model` and return (model_with_best_weights_loaded, history, best_val_acc).

    Parameters
    ----------
    model           : torch.nn.Module returning (B, num_classes) logits
    train_loader    : DataLoader for training data
    val_loader      : DataLoader for validation data
    epochs          : maximum number of epochs
    lr              : initial learning rate
    weight_decay    : AdamW L2 penalty
    patience        : early-stopping patience (in epochs without val-loss improvement)
    use_amp         : enable Automatic Mixed Precision (recommended on CUDA)
    run_name        : identifier used to name checkpoints / log dirs
    checkpoint_path : where to save the best weights (default: models/{run_name}.pth)
    log_dir         : TensorBoard log directory (default: logs/{run_name})
    history_path    : where to dump per-epoch metrics as JSON
                      (default: outputs/reports/{run_name}_history.json)
    seed            : seeds Python/NumPy/PyTorch for reproducibility
    optimizer_kwargs: extra kwargs forwarded to AdamW (e.g. {"betas": (0.9, 0.99)})
    grad_clip_norm  : if set, clip gradient L2 norm to this value before optimizer.step()

    Returns
    -------
    model, history, best_val_acc
    """
    set_seed(seed)

    device  = get_device()
    model   = model.to(device)

    # ---- Resolve default paths ---------------------------------------
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else MODELS_DIR / f"{run_name}.pth"
    log_dir         = Path(log_dir)         if log_dir         else LOGS_DIR   / run_name
    history_path    = Path(history_path)    if history_path    else (
        MODELS_DIR.parent / "outputs" / "reports" / f"{run_name}_history.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training components ----------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        **(optimizer_kwargs or {}),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2,
    )
    # GradScaler only needed for fp16 (bf16 keeps fp32 dynamic range).
    scaler = (
        torch.amp.GradScaler(device=device.type)
        if (use_amp and device.type == "cuda" and amp_dtype == torch.float16)
        else None
    )
    writer = SummaryWriter(log_dir=str(log_dir))
    history = TrainHistory()

    # ---- Summary banner ---------------------------------------------
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    print(f"[fit] run_name        = {run_name}")
    print(f"[fit] device          = {device}")
    print(f"[fit] trainable params = {n_train_params:,}")
    print(f"[fit] train batches    = {n_train_batches} | val batches = {n_val_batches}")
    print(f"[fit] lr={lr:.0e}  wd={weight_decay:.0e}  amp={use_amp}({amp_dtype})  epochs={epochs}  patience={patience}")
    print(f"[fit] checkpoint -> {checkpoint_path}")
    print(f"[fit] tensorboard -> {log_dir}")
    print()

    # ---- Training loop ----------------------------------------------
    best_val_acc        = 0.0
    best_val_loss       = float("inf")
    best_epoch          = 0
    epochs_no_improve   = 0
    best_state_dict     = None
    t_total             = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
            amp_dtype=amp_dtype,
            desc=f"E{epoch:02d}/{epochs} train",
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_amp,
            amp_dtype=amp_dtype,
            desc=f"E{epoch:02d}/{epochs} val",
        )

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.lr.append(cur_lr)
        history.epoch_time.append(epoch_time)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("acc/train",  train_acc,  epoch)
        writer.add_scalar("acc/val",    val_acc,    epoch)
        writer.add_scalar("lr",         cur_lr,     epoch)

        # ---- Best-checkpoint logic ---------------------------------
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc      = val_acc
            best_val_loss     = val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            best_state_dict   = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, checkpoint_path)
            tag = "  <- best"
        else:
            epochs_no_improve += 1
            tag = ""

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"lr={cur_lr:.2e}  time={epoch_time:.1f}s{tag}"
        )

        # Optional gradient clipping
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n[fit] Early stopping at epoch {epoch} "
                  f"(no improvement in {patience} epochs).")
            break

    # ---- Wrap-up -----------------------------------------------------
    total_time = time.time() - t_total
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history.save(history_path)
    writer.close()

    print(f"\n[fit] Done in {total_time/60:.1f} min")
    print(f"[fit] Best val_acc = {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"[fit] Best val_loss= {best_val_loss:.4f}")
    print(f"[fit] Checkpoint saved -> {checkpoint_path}")
    print(f"[fit] History saved    -> {history_path}")

    return model, history, best_val_acc


# --------------------------------------------------------------------- #
# Plot helper (used by the notebooks and STEP 15)                       #
# --------------------------------------------------------------------- #
def plot_history(history: TrainHistory, save_path: Optional[Path] = None,
                 title: str = "Training history"):
    """Plot loss + accuracy curves on a 1x2 figure."""
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(epochs, history.train_loss, "-o", label="train", color="steelblue")
    axes[0].plot(epochs, history.val_loss,   "-o", label="val",   color="tomato")
    axes[0].set_title("Loss");     axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CE loss")
    axes[0].grid(alpha=0.3);       axes[0].legend()

    axes[1].plot(epochs, history.train_acc, "-o", label="train", color="steelblue")
    axes[1].plot(epochs, history.val_acc,   "-o", label="val",   color="tomato")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05);     axes[1].grid(alpha=0.3); axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
