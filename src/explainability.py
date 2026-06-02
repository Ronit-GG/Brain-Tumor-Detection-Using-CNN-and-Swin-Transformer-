"""
Explainability helpers: Grad-CAM for each of the three models in the
ensemble.

API
---
make_cnn_gradcam(cnn_model)             -> GradCAM
make_efficientnet_gradcam(tl_model)     -> GradCAM
make_swin_gradcam(swin_model)           -> GradCAM   (with reshape_transform)
load_image_for_explain(path, image_size) -> (input_tensor, rgb01_array)
overlay_heatmap(rgb01, gray_cam, alpha)  -> RGB uint8 array

Notes
-----
* Grad-CAM requires gradients, so call site must NOT wrap in `torch.no_grad()`.
* All factory functions assume the model is already on the desired device
  and in `.eval()` mode (load_all_models() takes care of that).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .config import IMG_SIZE
from .preprocess import get_inference_transform


# --------------------------------------------------------------------- #
# Image loading helper                                                   #
# --------------------------------------------------------------------- #
def load_image_for_explain(
    path: str | Path,
    image_size: int = IMG_SIZE,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load an image and return two parallel views:

    * `tensor`  : (3, H, W) float32 normalized — feed to the model.
    * `rgb_01`  : (H, W, 3) float32 in [0, 1] — for visualization overlays.

    Both views are aligned to the same `image_size`, so the overlay
    naturally matches what the model saw.
    """
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((image_size, image_size))
    rgb_01 = np.asarray(img_resized).astype(np.float32) / 255.0
    tensor = get_inference_transform(image_size)(img)
    return tensor, rgb_01


# --------------------------------------------------------------------- #
# CNN: target the last 3x3 conv in the deepest block                    #
# --------------------------------------------------------------------- #
def make_cnn_gradcam(model):
    """Grad-CAM for BrainTumorCNN. Targets `block4.conv2` (256 ch, 14x14)."""
    from pytorch_grad_cam import GradCAM
    return GradCAM(
        model=model,
        target_layers=[model.block4.conv2],
    )


# --------------------------------------------------------------------- #
# EfficientNet-B0: target the final feature block                       #
# --------------------------------------------------------------------- #
def make_efficientnet_gradcam(transfer_model):
    """Grad-CAM for TransferModel(backbone='efficientnet_b0').

    `backbone.features` is a Sequential of MBConv blocks; the last entry
    is the 1280-channel Conv2dNormActivation that precedes the classifier.
    """
    from pytorch_grad_cam import GradCAM
    target = transfer_model.backbone.features[-1]
    return GradCAM(
        model=transfer_model,
        target_layers=[target],
    )


# --------------------------------------------------------------------- #
# Swin Transformer: target the last block's norm + reshape transform    #
# --------------------------------------------------------------------- #
def _swin_reshape(tensor: torch.Tensor) -> torch.Tensor:
    """
    timm's Swin emits intermediate tensors as (B, H, W, C). Grad-CAM
    expects (B, C, H, W) for its spatial pooling, so we just permute.
    For flat-token shapes (B, N, C) we reshape back to (B, H, W, C) first.
    """
    if tensor.dim() == 3:
        b, n, c = tensor.shape
        h = w = int(round(n ** 0.5))
        tensor = tensor.reshape(b, h, w, c)
    # (B, H, W, C) -> (B, C, H, W)
    return tensor.permute(0, 3, 1, 2).contiguous()


def make_swin_gradcam(swin_model):
    """
    Grad-CAM for SwinModel. Targets the second LayerNorm of the last
    block in the last stage, where the (7x7x768) spatial structure is
    still intact.
    """
    from pytorch_grad_cam import GradCAM

    # timm swin tree: model.backbone.layers[stage].blocks[block].{norm1,attn,norm2,mlp}
    target = swin_model.backbone.layers[-1].blocks[-1].norm2

    return GradCAM(
        model=swin_model,
        target_layers=[target],
        reshape_transform=_swin_reshape,
    )


# --------------------------------------------------------------------- #
# Overlay helper                                                        #
# --------------------------------------------------------------------- #
def overlay_heatmap(
    rgb_01: np.ndarray,
    gray_cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Place a [0,1] grayscale CAM heatmap on top of a [H,W,3] [0,1] image.

    `alpha` is the heatmap weight; (1 - alpha) is the image weight.
    Returns a uint8 RGB array suitable for `imshow`.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image
    return show_cam_on_image(
        rgb_01,
        gray_cam,
        use_rgb=True,
        image_weight=1.0 - alpha,
    )


# --------------------------------------------------------------------- #
# One-call convenience: get a CAM for any of the three models            #
# --------------------------------------------------------------------- #
def gradcam_for_model(
    model_name: str,
    model: nn.Module,
):
    """Dispatch helper: returns a GradCAM extractor for the named model."""
    name = model_name.lower()
    if name in {"cnn", "custom_cnn"}:
        return make_cnn_gradcam(model)
    if name in {"transfer", "efficientnet", "efficientnet_b0"}:
        return make_efficientnet_gradcam(model)
    if name in {"swin", "swin_tiny"}:
        return make_swin_gradcam(model)
    raise ValueError(f"Unknown model name: {model_name}")
