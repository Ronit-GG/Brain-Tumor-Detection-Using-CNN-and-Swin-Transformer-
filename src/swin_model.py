"""
Swin Transformer wrapper (via timm) for brain-tumor classification.

Architecture (swin_tiny_patch4_window7_224)
------------------------------------------
    Input (B, 3, 224, 224)
      |
      |-- PatchEmbed (Conv2d 4x4 stride 4)  -> (B, 3136, 96)   stage 0
      |-- Stage 1: 2 W-MSA blocks, dim 96   -> (B, 56*56, 96)
      |-- PatchMerging                      -> (B, 28*28, 192)
      |-- Stage 2: 2 W-MSA blocks, dim 192  -> (B, 28*28, 192)
      |-- PatchMerging                      -> (B, 14*14, 384)
      |-- Stage 3: 6 W-MSA blocks, dim 384  -> (B, 14*14, 384)
      |-- PatchMerging                      -> (B,  7*7, 768)
      |-- Stage 4: 2 W-MSA blocks, dim 768  -> (B,  7*7, 768)
      |-- LayerNorm + GlobalAvgPool          -> (B, 768)
      |
      `-- Custom head: Dropout -> Linear(768, 4) -> (B, 4 logits)

Each stage alternates plain window-MSA and shifted window-MSA.

Two-stage training pattern (same as TransferModel):
    Stage 1 (warm-up):    freeze_backbone()   + train head only, LR ~ 1e-3
    Stage 2 (fine-tune):  unfreeze_backbone() + train all at LR ~ 2e-5

NOTE: transformers are more sensitive to LR than CNNs -- use a very
small fine-tuning LR (1e-5 to 5e-5) to avoid wrecking the pretrained
attention weights.

Self-test:
    python -m src.swin_model
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .config import NUM_CLASSES
from .utils import count_parameters, get_device, human_format


# --------------------------------------------------------------------- #
# Top-level wrapper                                                     #
# --------------------------------------------------------------------- #
class SwinModel(nn.Module):
    """
    Swin Transformer backbone (from timm) + custom 4-class head.

    Parameters
    ----------
    model_name : str
        Any timm Swin variant. Default: 'swin_tiny_patch4_window7_224'
        (28 M params, fits comfortably in 4 GB VRAM with bf16).

        Other supported sizes (will NOT fit on 4 GB at batch=32):
          - swin_small_patch4_window7_224 (50 M)
          - swin_base_patch4_window7_224  (88 M)

    num_classes : int
        Output classes (default 4).
    pretrained : bool
        Load ImageNet pretrained weights (recommended).
    freeze_backbone : bool
        Start with the backbone frozen (Stage 1 warm-up).
    dropout : float
        Dropout before the final Linear head.
    """

    def __init__(
        self,
        model_name: str  = "swin_tiny_patch4_window7_224",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float   = 0.3,
    ):
        super().__init__()
        self.model_name = model_name

        # `num_classes=0` -> timm returns the GAP-pooled features only.
        # `global_pool='avg'` -> ensures the output is (B, num_features).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim: int = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        nn.init.kaiming_normal_(self.classifier[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier[1].bias)

        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    # ----------------------------------------------------------------- #
    # Freeze / unfreeze                                                 #
    # ----------------------------------------------------------------- #
    def freeze_backbone(self) -> None:
        """Stage 1: only the classifier head gets gradients."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()                # also freezes LN running behaviour
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Stage 2: end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        """Respect frozen state when parent toggles train()."""
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    # ----------------------------------------------------------------- #
    # Forward                                                           #
    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)            # (B, feature_dim)
        return self.classifier(feats)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the pre-pool spatial features, shape (B, 7, 7, 768) for
        swin_tiny. Used later for attention rollout visualization.
        """
        return self.backbone.forward_features(x)

    # ----------------------------------------------------------------- #
    # Convenience                                                       #
    # ----------------------------------------------------------------- #
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------- #
# Factory helper                                                        #
# --------------------------------------------------------------------- #
def build_swin_model(
    model_name: str  = "swin_tiny_patch4_window7_224",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float   = 0.3,
) -> SwinModel:
    """Returns a SwinModel with the default config."""
    return SwinModel(
        model_name=model_name,
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
    print(" Brain Tumor Detection — Swin Transformer self-test")
    print("=" * 64)

    device = get_device()
    print(f"Device: {device}")

    # Stage-1 (frozen) view
    model = build_swin_model(freeze_backbone=True).to(device)
    total = model.num_total_parameters()
    trainable = model.num_trainable_parameters()
    print(f"\nModel name        : {model.model_name}")
    print(f"Feature dim       : {model.feature_dim}")
    print(f"Total params      : {total:>12,} ({human_format(total)})")
    print(f"Stage 1 trainable : {trainable:>12,} ({human_format(trainable)})  <- head only")

    # Shape trace
    x = torch.randn(2, 3, 224, 224, device=device)
    print(f"\nForward shape trace (batch=2):")
    print(f"  input          : {tuple(x.shape)}")
    with torch.no_grad():
        feats_pool = model.backbone(x)
        print(f"  after backbone : {tuple(feats_pool.shape)}  (GAP'd features)")
        feats_spatial = model.forward_features(x)
        print(f"  forward_features: {tuple(feats_spatial.shape)}  (pre-pool spatial)")
        logits = model(x)
        print(f"  after head     : {tuple(logits.shape)}")

    # Forward + backward (Stage 1)
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
    print(f"  params with grad: {n_with_grad}/{n_total}  (head only)")

    # Switch to Stage 2 view
    model.unfreeze_backbone()
    print(f"\nAfter unfreeze_backbone():")
    print(f"  Stage 2 trainable: {model.num_trainable_parameters():,} "
          f"({human_format(model.num_trainable_parameters())})")

    # VRAM check at training batch size with bf16 AMP
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        x_big = torch.randn(32, 3, 224, 224, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x_big)
            loss = F.cross_entropy(out, torch.randint(0, NUM_CLASSES, (32,), device=device))
        loss.backward()
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"\nPeak VRAM (batch=32, bf16 AMP, fwd+bwd): {mem_mb:.1f} MB  "
              f"({mem_mb / 1024:.2f} GB / 4.0 GB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
