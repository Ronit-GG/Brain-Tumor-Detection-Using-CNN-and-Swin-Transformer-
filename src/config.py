"""
Central configuration for the Brain Tumor Detection project.

Single source of truth for paths, hyperparameters, class labels, and
device selection. Every other module imports from here so changing a
hyperparameter requires editing exactly one file.

NOTE: This file is populated more fully in STEP 3. For now it only
provides the constants needed by the environment check script.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Paths (all relative to project root for Windows-friendly portability)
# ---------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATASET_DIR:  Path = PROJECT_ROOT / "DATASET"
TRAIN_DIR:    Path = DATASET_DIR / "Training"
TEST_DIR:     Path = DATASET_DIR / "Testing"

MODELS_DIR:    Path = PROJECT_ROOT / "models"
OUTPUTS_DIR:   Path = PROJECT_ROOT / "outputs"
LOGS_DIR:      Path = PROJECT_ROOT / "logs"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------
# Classes (kept in a fixed order — index = label id)
# ---------------------------------------------------------------------
CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES: int       = len(CLASS_NAMES)

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------
# Image / training defaults (tuned for 4 GB VRAM GPUs like GTX 1650).
# These are *defaults*; individual training scripts may override.
# ---------------------------------------------------------------------
IMG_SIZE:    int   = 224          # all three models accept 224x224
BATCH_SIZE:  int   = 32           # safe on 4 GB VRAM with AMP
NUM_WORKERS: int   = 2            # Windows: keep low to avoid spawn overhead
EPOCHS:      int   = 25
LR:          float = 1e-4
WEIGHT_DECAY:float = 1e-4

# ---------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------
VAL_SPLIT: float = 0.20           # 20 % of Training/ becomes validation

# ---------------------------------------------------------------------
# Normalization — ImageNet statistics.
# Used for ALL three models so a single preprocessing pipeline works
# for ResNet/EfficientNet/Swin (which were pretrained on ImageNet) and
# our custom CNN (which is trained from scratch and is insensitive to
# the exact normalization constants).
# ---------------------------------------------------------------------
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)
