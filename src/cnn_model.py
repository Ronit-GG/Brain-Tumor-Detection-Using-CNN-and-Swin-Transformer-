"""
Custom CNN for 4-class brain-tumor MRI classification.

Architecture (VGG-style)
------------------------
    Input  (B, 3, 224, 224)
      |
      |---- Block 1: [Conv(3 -> 32) BN ReLU] x2 + MaxPool   -> (B, 32, 112, 112)
      |---- Block 2: [Conv(32 -> 64) BN ReLU] x2 + MaxPool  -> (B, 64,  56,  56)
      |---- Block 3: [Conv(64 -> 128) BN ReLU] x2 + MaxPool -> (B, 128, 28,  28)
      |---- Block 4: [Conv(128 -> 256) BN ReLU] x2 + MaxPool-> (B, 256, 14,  14)
      |
      |---- AdaptiveAvgPool2d(1)                            -> (B, 256, 1, 1)
      |---- Flatten                                         -> (B, 256)
      |
      `---- Dropout(0.5) -> Linear(256 -> 128) -> ReLU
            -> Dropout(0.3) -> Linear(128 -> 4)             -> (B, 4 logits)

Total parameters: ~1.21 M  (light enough to train on a 4 GB GPU at batch 32).

Run as a script for a self-test:

    python -m src.cnn_model
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES
from .utils import count_parameters, get_device, human_format


# --------------------------------------------------------------------- #
# Reusable conv block                                                   #
# --------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """
    [Conv -> BN -> ReLU] x 2 -> MaxPool

    Two stacked 3x3 convs give an effective 5x5 receptive field with
    fewer parameters than a single 5x5 conv and an extra non-linearity
    in between.

    bias=False in the Conv layers because BatchNorm's learnable shift
    makes the conv bias redundant -- a small but free parameter saving.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)
        return x


# --------------------------------------------------------------------- #
# Top-level model                                                       #
# --------------------------------------------------------------------- #
class BrainTumorCNN(nn.Module):
    """
    Custom CNN classifier for 4-class brain-tumor MRI.

    Parameters
    ----------
    num_classes : int
        Output classes (default 4: glioma, meningioma, notumor, pituitary).
    dropout : float
        Dropout rate applied just before the bottleneck FC layer.
        A second, lower dropout (`dropout * 0.6`) is applied before
        the final classifier layer.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.block1 = ConvBlock(3,   32)   # 224 -> 112
        self.block2 = ConvBlock(32,  64)   # 112 -> 56
        self.block3 = ConvBlock(64,  128)  #  56 -> 28
        self.block4 = ConvBlock(128, 256)  #  28 -> 14

        # Global Average Pooling collapses the 14x14 spatial grid to 1x1
        # which gives spatial invariance AND drastically reduces FC params
        # compared with a flatten (256*14*14 = 50,176 -> 256).
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # 2-layer FC head with dropout between layers.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    # ----------------------------------------------------------------- #
    # Weight initialization                                             #
    # ----------------------------------------------------------------- #
    def _init_weights(self) -> None:
        """
        Kaiming (He) initialization tuned for ReLU networks.

        For Conv: weights ~ N(0, sqrt(2 / fan_in)) keeps activation
        variance roughly constant across depth, preventing vanishing
        or exploding signals.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------- #
    # Forward pass                                                      #
    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return self.classifier(x)

    # ----------------------------------------------------------------- #
    # Convenience: a feature-extraction forward used later for Grad-CAM #
    # ----------------------------------------------------------------- #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final pre-pool feature maps, shape (B, 256, 14, 14)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# --------------------------------------------------------------------- #
# Factory helper                                                        #
# --------------------------------------------------------------------- #
def build_cnn(num_classes: int = NUM_CLASSES, dropout: float = 0.5) -> BrainTumorCNN:
    """Returns a fresh BrainTumorCNN instance with the default config."""
    return BrainTumorCNN(num_classes=num_classes, dropout=dropout)


# --------------------------------------------------------------------- #
# Sanity-check entry point                                              #
# --------------------------------------------------------------------- #
def _trace_shapes(model: BrainTumorCNN, x: torch.Tensor) -> None:
    """Print the tensor shape after each major stage."""
    print(f"  input        : {tuple(x.shape)}")
    with torch.no_grad():
        h = model.block1(x); print(f"  after block1 : {tuple(h.shape)}")
        h = model.block2(h); print(f"  after block2 : {tuple(h.shape)}")
        h = model.block3(h); print(f"  after block3 : {tuple(h.shape)}")
        h = model.block4(h); print(f"  after block4 : {tuple(h.shape)}")
        h = model.gap(h);    print(f"  after GAP    : {tuple(h.shape)}")
        h = model.classifier(h); print(f"  after head   : {tuple(h.shape)}")


def main() -> None:
    print("=" * 64)
    print(" Brain Tumor Detection — Custom CNN self-test")
    print("=" * 64)

    device = get_device()
    print(f"Device: {device}")

    model = build_cnn().to(device)
    n = count_parameters(model)
    n_bn = sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d)
               for p in m.parameters())
    n_conv = sum(p.numel() for m in model.modules() if isinstance(m, nn.Conv2d)
                 for p in m.parameters())
    n_fc = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear)
               for p in m.parameters())
    print(f"\nParameter breakdown:")
    print(f"  Conv2d     : {n_conv:>10,} ({human_format(n_conv)})")
    print(f"  BatchNorm2d: {n_bn:>10,}")
    print(f"  Linear     : {n_fc:>10,}")
    print(f"  TOTAL      : {n:>10,} ({human_format(n)})")

    print(f"\nShape trace (batch=2):")
    x = torch.randn(2, 3, 224, 224, device=device)
    _trace_shapes(model, x)

    # Forward + backward smoke test.
    print(f"\nForward + backward smoke test:")
    model.train()
    y = torch.randint(0, NUM_CLASSES, (2,), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    all_grads = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    grad_norm = sum(
        (p.grad.detach() ** 2).sum().item()
        for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  logits.shape  : {tuple(logits.shape)}")
    print(f"  loss (random) : {loss.item():.4f}")
    print(f"  gradients OK  : {all_grads}")
    print(f"  total grad L2 : {grad_norm:.4f}")

    # Memory footprint (CUDA only).
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"  peak VRAM     : {mem_mb:.1f} MB (batch=2)")

    print("\nDone.")


if __name__ == "__main__":
    main()
