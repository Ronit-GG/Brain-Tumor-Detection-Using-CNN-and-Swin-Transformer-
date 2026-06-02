"""
Image transforms (training, validation, inference).

STEP 3 (this version): deterministic preprocessing only.

  Pipeline applied to every image:
    1. Resize to IMG_SIZE x IMG_SIZE  (square resize, no crop)
    2. ToTensor                       (uint8 [0,255] -> float32 [0,1], HWC->CHW)
    3. Normalize                      (per-channel: (x - mean) / std,
                                       using ImageNet statistics)

STEP 5 will extend `get_train_transform` with data augmentation
(random flips, rotations, affine, cutout) while leaving the val and
inference pipelines deterministic.

Why ImageNet mean/std for all three models?
-------------------------------------------
The transfer-learning model (ResNet/EfficientNet/DenseNet) and the
Swin Transformer were both pretrained on ImageNet. They expect inputs
distributed like ImageNet — feeding them differently scaled tensors
would shift their internal activations away from where the pretrained
weights are calibrated and hurt accuracy.

Our custom CNN is trained from scratch, so it doesn't care about the
exact normalization constants — it just learns around whatever offset
we apply. Sharing one normalization pipeline therefore costs nothing
and lets the same transforms be used by every model and the Streamlit
inference path.

Why grayscale -> 3-channel?
---------------------------
MRI images are intrinsically grayscale, but the pretrained backbones'
first convolution expects 3 input channels (weight shape [C, 3, K, K]).
We satisfy this by calling `Image.convert("RGB")` inside the Dataset,
which simply replicates the gray plane three times.
"""
from __future__ import annotations

from typing import Callable

from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


# --------------------------------------------------------------------- #
# Deterministic pipeline (used for VAL, TEST, and INFERENCE)            #
# --------------------------------------------------------------------- #
def get_base_transform(image_size: int = IMG_SIZE) -> Callable:
    """
    Deterministic preprocessing used for validation, testing, and the
    Streamlit inference path.

    Steps:
      1. Resize to (image_size, image_size).
      2. Convert PIL image -> torch.FloatTensor of shape [3, H, W],
         scaled to [0, 1].
      3. Normalize per-channel with ImageNet mean/std.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


# --------------------------------------------------------------------- #
# Training pipeline (STEP 5: medical-imaging aware augmentations)       #
# --------------------------------------------------------------------- #
def get_train_transform(
    image_size: int = IMG_SIZE,
    flip_prob:  float = 0.5,
    rotation:   float = 15.0,
    translate:  float = 0.05,
    scale_low:  float = 0.95,
    scale_high: float = 1.05,
    brightness: float = 0.20,
    contrast:   float = 0.20,
    erase_prob: float = 0.25,
    erase_scale_low:  float = 0.02,
    erase_scale_high: float = 0.10,
) -> Callable:
    """
    Training transform with medical-imaging-aware augmentations.

    Pipeline (in order):
      1. Resize          : standardize to image_size x image_size.
      2. HorizontalFlip  : 50% probability (brain is bilateral).
      3. Rotation        : +/- 15 degrees (natural head tilt).
      4. Affine          : small translate (5%) and scale (+/- 5%);
                           shear=0 (anatomically implausible).
      5. ColorJitter     : brightness +/- 20%, contrast +/- 20%.
                           NO hue/saturation jitter -- MRI is grayscale.
      6. ToTensor        : PIL -> float32 CHW [0, 1].
      7. Normalize       : ImageNet per-channel mean/std.
      8. RandomErasing   : 25% probability, small patches (2-10% of area)
                           -- forces the network to use global context
                           and tolerates partial occlusion.

    All defaults are tuned for medical imaging. Pass keyword overrides
    to make the augmentations milder/stronger from a training script.
    """
    fill_value: int = 0  # black -- matches MRI background

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=flip_prob),
        transforms.RandomRotation(degrees=rotation, fill=fill_value),
        transforms.RandomAffine(
            degrees=0,                          # rotation already covered above
            translate=(translate, translate),   # +/- 5% horizontal & vertical
            scale=(scale_low, scale_high),      # 0.95x to 1.05x zoom
            shear=0,
            fill=fill_value,
        ),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=0,                       # grayscale: no saturation
            hue=0,                              # grayscale: no hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        transforms.RandomErasing(
            p=erase_prob,
            scale=(erase_scale_low, erase_scale_high),
            ratio=(0.3, 3.3),
            value=0,                            # 0 in normalized space
            inplace=False,
        ),
    ])


# --------------------------------------------------------------------- #
# Inference pipeline                                                    #
# --------------------------------------------------------------------- #
def get_inference_transform(image_size: int = IMG_SIZE) -> Callable:
    """Alias of `get_base_transform`, used by the Streamlit app."""
    return get_base_transform(image_size)


# --------------------------------------------------------------------- #
# Utility: invert the normalization (for visualization)                 #
# --------------------------------------------------------------------- #
def denormalize(tensor):
    """
    Reverse the ImageNet normalization so a tensor can be displayed.

    Accepts a tensor of shape [C, H, W] or [B, C, H, W] and returns the
    same shape with pixel values back in [0, 1] (clipped).
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
