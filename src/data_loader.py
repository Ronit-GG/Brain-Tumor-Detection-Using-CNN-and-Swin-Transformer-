"""
Dataset and DataLoader factories for the Brain Tumor Detection project.

Responsibilities
----------------
1. Scan the DATASET/ folder and build a (filepath, class, label) index.
2. Produce a STRATIFIED 80/20 train/val split of the Training/ folder
   so each class keeps its proportion in both halves.
3. Use Testing/ as a held-out test set (never seen during training).
4. Persist the split to `data/processed/{train,val,test}_split.csv`
   for full reproducibility across runs, machines, and notebooks.
5. Expose `get_dataloaders(...)` — a single factory that returns
   ready-to-use train/val/test DataLoaders.

Run this file directly to sanity-check:

    python -m src.data_loader
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMG_SIZE,
    NUM_WORKERS,
    PROCESSED_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_SPLIT,
)
from .preprocess import get_base_transform, get_train_transform

# Canonical class <-> index mapping. Pinned to CLASS_NAMES order so that
# label 0 always means "glioma" regardless of the OS's directory order.
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

# File extensions we recognise as image files.
IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------- #
# 1. Index a class-folder layout into a DataFrame                        #
# --------------------------------------------------------------------- #
def _index_folder(root: Path) -> pd.DataFrame:
    """
    Walk `root/<class>/<image>` and return a DataFrame with columns
    [filepath, class, label].

    Raises
    ------
    FileNotFoundError
        If any expected class subfolder is missing.
    RuntimeError
        If no images are found at all.
    """
    records: list[dict] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                records.append({
                    "filepath": str(p.resolve()),
                    "class":    class_name,
                    "label":    CLASS_TO_IDX[class_name],
                })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    return df


# --------------------------------------------------------------------- #
# 2. Stratified train/val split with on-disk cache                       #
# --------------------------------------------------------------------- #
def build_splits(
    val_ratio: float = VAL_SPLIT,
    seed: int = SEED,
    cache_dir: Path = PROCESSED_DIR,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build (or load cached) train / val / test splits.

    The first call writes 3 CSVs to `cache_dir`; subsequent calls just
    read those CSVs, guaranteeing identical splits across runs.

    Returns
    -------
    (train_df, val_df, test_df) with columns [filepath, class, label].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_csv = cache_dir / "train_split.csv"
    val_csv   = cache_dir / "val_split.csv"
    test_csv  = cache_dir / "test_split.csv"

    if not overwrite and train_csv.exists() and val_csv.exists() and test_csv.exists():
        return (
            pd.read_csv(train_csv),
            pd.read_csv(val_csv),
            pd.read_csv(test_csv),
        )

    full_train = _index_folder(TRAIN_DIR)
    test_df    = _index_folder(TEST_DIR)

    # Stratified split preserves the class proportions in both halves.
    train_df, val_df = train_test_split(
        full_train,
        test_size=val_ratio,
        stratify=full_train["label"],
        random_state=seed,
        shuffle=True,
    )

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)
    test_df.to_csv(test_csv,   index=False)

    return train_df, val_df, test_df


# --------------------------------------------------------------------- #
# 3. PyTorch Dataset                                                     #
# --------------------------------------------------------------------- #
class BrainTumorDataset(Dataset):
    """
    Map-style Dataset that reads (image, label) pairs from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'filepath' (str) and 'label' (int).
    transform : callable | None
        torchvision transform pipeline applied to the PIL image.
        If None, the returned image is a PIL.Image rather than a Tensor.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        # Reset to a contiguous 0..N-1 index so positional access works.
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        # convert("RGB") guarantees 3 channels even for grayscale MRIs.
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


# --------------------------------------------------------------------- #
# 4. DataLoader factory                                                  #
# --------------------------------------------------------------------- #
def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = IMG_SIZE,
    val_ratio: float = VAL_SPLIT,
    seed: int = SEED,
    pin_memory: Optional[bool] = None,
    overwrite_split: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders sharing the same on-disk split.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    if pin_memory is None:
        # pinned host memory speeds up CPU -> GPU transfer, but is
        # pointless on a CPU-only machine.
        pin_memory = torch.cuda.is_available()

    train_df, val_df, test_df = build_splits(
        val_ratio=val_ratio, seed=seed, overwrite=overwrite_split
    )

    train_tf = get_train_transform(image_size)
    eval_tf  = get_base_transform(image_size)

    train_ds = BrainTumorDataset(train_df, transform=train_tf)
    val_ds   = BrainTumorDataset(val_df,   transform=eval_tf)
    test_ds  = BrainTumorDataset(test_df,  transform=eval_tf)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # persistent_workers avoids re-spawning workers every epoch
        # (only valid when num_workers > 0).
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------- #
# 5. Convenience: per-class summary                                      #
# --------------------------------------------------------------------- #
def _summary_table(df: pd.DataFrame, name: str) -> str:
    counts = df["class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    total  = len(df)
    lines  = [f"  {name}: {total} images"]
    for cls, n in counts.items():
        pct = 100.0 * n / total if total else 0.0
        lines.append(f"    {cls:12s} {n:>5d}  ({pct:5.2f}%)")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# 6. CLI sanity check                                                    #
# --------------------------------------------------------------------- #
def main() -> None:
    print("=" * 64)
    print(" Brain Tumor Detection — Data loader sanity check")
    print("=" * 64)

    # Always overwrite the split here so re-running the check produces
    # the same split deterministically (controlled by SEED).
    train_df, val_df, test_df = build_splits(overwrite=True)
    print(_summary_table(train_df, "TRAIN"))
    print(_summary_table(val_df,   "VAL"))
    print(_summary_table(test_df,  "TEST"))

    # NOTE on Windows: num_workers > 0 spawns child processes that
    # re-import this module; the `if __name__ == "__main__"` guard
    # below prevents an infinite spawn loop. The very first iteration
    # therefore takes longer (~5-10 s) due to worker startup.
    train_loader, val_loader, test_loader = get_dataloaders()
    print(
        f"\nDataLoaders -> "
        f"train batches={len(train_loader)}, "
        f"val batches={len(val_loader)}, "
        f"test batches={len(test_loader)}"
    )

    t0 = time.time()
    images, labels = next(iter(train_loader))
    elapsed = time.time() - t0

    print(f"\nFirst batch (cold start, includes worker spawn): {elapsed:.2f}s")
    print(f"  images.shape  : {tuple(images.shape)}")
    print(f"  images.dtype  : {images.dtype}")
    print(f"  images.min    : {images.min().item():+.3f}")
    print(f"  images.max    : {images.max().item():+.3f}")
    print(f"  images.mean   : {images.mean().item():+.3f}")
    print(f"  images.std    : {images.std().item():+.3f}")
    print(f"  labels.shape  : {tuple(labels.shape)}")
    print(f"  labels[:8]    : {labels[:8].tolist()}")
    print(f"  classes[:8]   : {[IDX_TO_CLASS[int(l)] for l in labels[:8]]}")

    print("\nCSV cache written to:")
    for f in ("train_split.csv", "val_split.csv", "test_split.csv"):
        p = PROCESSED_DIR / f
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
