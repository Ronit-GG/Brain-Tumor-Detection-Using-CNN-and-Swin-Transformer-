"""
Transfer-learning model: pretrained backbone + custom classifier head.

Supported backbones
-------------------
    resnet50          ~25.6 M params, 4.10 GFLOPs, top-1 80.9 %
    efficientnet_b0    ~5.3 M params, 0.39 GFLOPs, top-1 77.7 %   <- default
    efficientnet_b3   ~12.2 M params, 1.83 GFLOPs, top-1 82.0 %
    densenet121        ~8.0 M params, 2.88 GFLOPs, top-1 74.4 %
    convnext_tiny     ~28.6 M params, 4.46 GFLOPs, top-1 82.1 %

Two-stage training pattern (see notebook 03):
    Stage 1 (warm-up)   : freeze_backbone() + train head at LR ~ 1e-3
    Stage 2 (fine-tune) : unfreeze_backbone()  + train all at LR ~ 1e-4

Run for a self-test:

    python -m src.transfer_model
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as tvm

from .config import NUM_CLASSES
from .utils import count_parameters, get_device, human_format


# --------------------------------------------------------------------- #
# Backbone factory                                                      #
# --------------------------------------------------------------------- #
def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """
    Returns (backbone_with_classifier_stripped, feature_dim).

    The backbone outputs a flat (B, feature_dim) tensor ready to feed
    into our custom classifier head.
    """
    name = name.lower()

    if name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
        feature_dim = net.fc.in_features          # 2048
        net.fc = nn.Identity()

    elif name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=weights)
        # `classifier` is Sequential(Dropout, Linear) -- take Linear's in_features.
        feature_dim = net.classifier[1].in_features  # 1280
        net.classifier = nn.Identity()

    elif name == "efficientnet_b3":
        weights = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b3(weights=weights)
        feature_dim = net.classifier[1].in_features  # 1536
        net.classifier = nn.Identity()

    elif name == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.densenet121(weights=weights)
        feature_dim = net.classifier.in_features    # 1024
        net.classifier = nn.Identity()

    elif name == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.convnext_tiny(weights=weights)
        # ConvNeXt classifier is LayerNorm2d, Flatten, Linear -- target index 2.
        feature_dim = net.classifier[2].in_features  # 768
        net.classifier[2] = nn.Identity()

    else:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from: "
            f"resnet50, efficientnet_b0, efficientnet_b3, densenet121, convnext_tiny."
        )

    return net, feature_dim


# --------------------------------------------------------------------- #
# Top-level wrapper                                                     #
# --------------------------------------------------------------------- #
class TransferModel(nn.Module):
    """
    Pretrained backbone followed by a custom 2-layer classifier head:

        backbone(x) -> (B, feature_dim) -> Dropout -> Linear -> (B, num_classes)
    """

    def __init__(
        self,
        backbone: str        = "efficientnet_b0",
        num_classes: int     = NUM_CLASSES,
        pretrained: bool     = True,
        freeze_backbone: bool = True,
        dropout: float       = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.backbone, self.feature_dim = _build_backbone(self.backbone_name, pretrained)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        nn.init.kaiming_normal_(self.classifier[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier[1].bias)

        # Track freeze state so train()/eval() do the right thing.
        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    # ----------------------------------------------------------------- #
    # Freeze / unfreeze                                                 #
    # ----------------------------------------------------------------- #
    def freeze_backbone(self) -> None:
        """Stage 1: only the classifier head receives gradients."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()                # also freezes BN running stats
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Stage 2: enable end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        Partial fine-tuning: unfreeze only the last `n` top-level
        children of the backbone (the deepest, most task-specific
        layers). Useful when full fine-tuning overfits.
        """
        children = list(self.backbone.children())
        for child in children[-n:]:
            for p in child.parameters():
                p.requires_grad = True
        # Backbone is now mixed (some frozen, some not); leave it in train()
        # mode but keep `_backbone_frozen=False` so train() doesn't .eval() it.
        self._backbone_frozen = False

    # ----------------------------------------------------------------- #
    # Override train() to respect frozen state                          #
    # ----------------------------------------------------------------- #
    def train(self, mode: bool = True):
        """
        When the backbone is frozen, keep it in eval() mode regardless
        of the parent mode. This is critical: PyTorch's default behaviour
        propagates train() to all submodules, which would re-enable
        BatchNorm running-stat updates in the frozen backbone and slowly
        corrupt its pretrained statistics.
        """
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    # ----------------------------------------------------------------- #
    # Forward                                                           #
    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    # ----------------------------------------------------------------- #
    # Convenience accessors                                             #
    # ----------------------------------------------------------------- #
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------- #
# Factory helper                                                        #
# --------------------------------------------------------------------- #
def build_transfer_model(
    backbone: str        = "efficientnet_b0",
    num_classes: int     = NUM_CLASSES,
    pretrained: bool     = True,
    freeze_backbone: bool = True,
    dropout: float       = 0.3,
) -> TransferModel:
    """Returns a TransferModel with the default 2-stage training setup."""
    return TransferModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


# --------------------------------------------------------------------- #
# Self-test                                                             #
# --------------------------------------------------------------------- #
def main() -> None:
    print("=" * 64)
    print(" Brain Tumor Detection — Transfer Learning self-test")
    print("=" * 64)

    device = get_device()
    print(f"Device: {device}")

    model = build_transfer_model(
        backbone="efficientnet_b0",
        freeze_backbone=True,
    ).to(device)
    total     = model.num_total_parameters()
    trainable = model.num_trainable_parameters()
    print(f"\nBackbone     : {model.backbone_name}")
    print(f"Feature dim  : {model.feature_dim}")
    print(f"Total params : {total:>12,} ({human_format(total)})")
    print(f"Stage 1 trainable: {trainable:>9,} ({human_format(trainable)})  <- head only")

    # Shape trace
    x = torch.randn(2, 3, 224, 224, device=device)
    print(f"\nForward shape trace (batch=2):")
    print(f"  input        : {tuple(x.shape)}")
    with torch.no_grad():
        feats = model.backbone(x)
        print(f"  after backbone: {tuple(feats.shape)}")
        logits = model.classifier(feats)
        print(f"  after head    : {tuple(logits.shape)}")

    # Stage 1 forward+backward (head only)
    import torch.nn.functional as F
    print(f"\nStage 1 (frozen backbone) forward+backward:")
    model.train()
    y = torch.randint(0, NUM_CLASSES, (2,), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"  loss = {loss.item():.4f}")
    print(f"  params with non-None grad: {n_with_grad}/{n_total}  (head only)")

    # Stage 2 setup
    model.unfreeze_backbone()
    trainable_after = model.num_trainable_parameters()
    print(f"\nAfter unfreeze_backbone():")
    print(f"  Stage 2 trainable: {trainable_after:,} ({human_format(trainable_after)})")
    assert trainable_after > trainable, "unfreeze did not enable any new params"

    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"\nPeak VRAM (batch=2): {mem_mb:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
