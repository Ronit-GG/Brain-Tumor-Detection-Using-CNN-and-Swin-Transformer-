"""
Production inference API for the 3-model ensemble.

One class, one method:

    from src.inference import BrainTumorPredictor
    predictor = BrainTumorPredictor()                # loads everything
    result = predictor.predict("brain.jpg",          # PIL.Image or path
                               return_gradcam=True)
    print(result.predicted_class, result.confidence)
    print(result.ensemble_probs)                     # {'glioma': 0.95, ...}
    print(result.inference_time_ms)
    # result.gradcams is {name: HxWx3 uint8} when return_gradcam=True

This module is the *only* place Streamlit / FastAPI / CLI wrappers need
to import from. It hides every detail of model loading, AMP, ensemble
weights, and Grad-CAM target-layer plumbing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config         import CLASS_NAMES, IMG_SIZE, MODELS_DIR
from .ensemble       import (
    EnsembleConfig, load_all_models,
    weighted_average, soft_voting,
)
from .explainability import gradcam_for_model, overlay_heatmap
from .preprocess     import get_inference_transform
from .utils          import get_device


# --------------------------------------------------------------------- #
# Output container                                                      #
# --------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    """Structured output of a single `BrainTumorPredictor.predict()` call."""

    predicted_class:     str                 # e.g. 'glioma'
    predicted_class_idx: int                 # e.g. 0
    confidence:          float               # ensemble's top-class probability

    # Probability distributions, both ensemble and per-model, keyed by
    # class name for ergonomic access.
    ensemble_probs:      dict[str, float]
    model_probs:         dict[str, dict[str, float]]

    # Diagnostics
    inference_time_ms:   float

    # Visualization payload (None unless return_gradcam=True was passed)
    original_rgb:        Optional[np.ndarray] = None    # (H, W, 3) float32 in [0, 1]
    gradcams:            Optional[dict[str, np.ndarray]] = None
                                                        # name -> (H, W, 3) uint8 overlay


# --------------------------------------------------------------------- #
# Predictor                                                             #
# --------------------------------------------------------------------- #
class BrainTumorPredictor:
    """
    Loads the 3 trained models + ensemble config once. Each `.predict()`
    call runs all 3 models on the input image, combines their probability
    outputs using the saved ensemble weights, and (optionally) generates
    Grad-CAM overlays for the predicted class.

    Parameters
    ----------
    models_dir : Path
        Directory containing cnn_model.pth, transfer_model.pth,
        swin_model.pth, ensemble_config.json.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    amp_dtype : torch.dtype
        AMP precision for inference (bf16 default; fp16 also OK on
        Ampere+ GPUs; fp32 on CPU).
    use_amp : bool, optional
        Defaults to True on CUDA, False on CPU.
    image_size : int
        All models were trained at 224 (do not change).
    """

    def __init__(
        self,
        models_dir: Union[Path, str] = MODELS_DIR,
        device:     Optional[torch.device] = None,
        amp_dtype:  torch.dtype = torch.bfloat16,
        use_amp:    Optional[bool] = None,
        image_size: int  = IMG_SIZE,
    ):
        models_dir = Path(models_dir)
        self.device     = device if device is not None else get_device()
        self.amp_dtype  = amp_dtype
        self.use_amp    = use_amp if use_amp is not None else (self.device.type == "cuda")
        self.image_size = image_size

        # Load all 3 trained models (already in .eval() mode, on self.device).
        self.models = load_all_models(
            self.device,
            cnn_path      = models_dir / "cnn_model.pth",
            transfer_path = models_dir / "transfer_model.pth",
            swin_path     = models_dir / "swin_model.pth",
        )

        self.ensemble_config = EnsembleConfig.load(models_dir / "ensemble_config.json")

        self.transform = get_inference_transform(image_size)

        # Lazily build Grad-CAM extractors -- expensive to create, but
        # only needed when return_gradcam=True is requested.
        self._cams: Optional[dict] = None

    # ----------------------------------------------------------------- #
    # Internal helpers                                                  #
    # ----------------------------------------------------------------- #
    def _get_cams(self) -> dict:
        if self._cams is None:
            self._cams = {
                "cnn":      gradcam_for_model("cnn",      self.models["cnn"]),
                "transfer": gradcam_for_model("transfer", self.models["transfer"]),
                "swin":     gradcam_for_model("swin",     self.models["swin"]),
            }
        return self._cams

    def _load_image(
        self, image: Union[Image.Image, str, Path]
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Load to (preprocessed_tensor, rgb01_array_for_overlay)."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Expected PIL.Image, str, or Path. Got {type(image).__name__}.")

        img_resized = img.resize((self.image_size, self.image_size))
        rgb01  = np.asarray(img_resized, dtype=np.float32) / 255.0
        tensor = self.transform(img)
        return tensor, rgb01

    @torch.no_grad()
    def _predict_probs(self, tensor: torch.Tensor) -> dict[str, np.ndarray]:
        """Run each model once and return softmax probabilities."""
        x = tensor.unsqueeze(0).to(self.device)
        out: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                logits = model(x)
            out[name] = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
        return out

    def _combine(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Apply the saved ensemble method. Returns (C,) ensemble probs."""
        prob_list = [probs["cnn"][None, :],
                     probs["transfer"][None, :],
                     probs["swin"][None, :]]

        method = self.ensemble_config.method
        if method == "soft_voting":
            ens = soft_voting(prob_list)
        elif method == "weighted":
            ens = weighted_average(prob_list, self.ensemble_config.weights)
        else:
            # Fallback to weighted with whatever weights were saved (or equal).
            w = self.ensemble_config.weights or [1 / 3, 1 / 3, 1 / 3]
            ens = weighted_average(prob_list, w)

        return ens[0]

    def _compute_gradcams(
        self,
        tensor: torch.Tensor,
        rgb01:  np.ndarray,
        target_class_idx: int,
    ) -> dict[str, np.ndarray]:
        """Per-model Grad-CAM overlays (RGB uint8 arrays)."""
        cams = self._get_cams()
        x = tensor.unsqueeze(0).to(self.device)
        targets = [ClassifierOutputTarget(target_class_idx)]

        overlays: dict[str, np.ndarray] = {}
        for name, cam in cams.items():
            try:
                gray = cam(input_tensor=x, targets=targets)[0]
                overlays[name] = overlay_heatmap(rgb01, gray, alpha=0.45)
            except Exception:                       # noqa: BLE001
                # Graceful fallback: return the unmodified image so the
                # caller can still display *something*.
                overlays[name] = (rgb01 * 255).astype(np.uint8)
        return overlays

    # ----------------------------------------------------------------- #
    # Public API                                                        #
    # ----------------------------------------------------------------- #
    def predict(
        self,
        image: Union[Image.Image, str, Path],
        return_gradcam: bool = False,
    ) -> PredictionResult:
        """
        End-to-end single-image inference.

        Parameters
        ----------
        image           : PIL.Image, str, or Path.
        return_gradcam  : Adds ~150-300 ms for the 3 Grad-CAM passes.

        Returns
        -------
        PredictionResult
        """
        t0 = time.perf_counter()

        tensor, rgb01     = self._load_image(image)
        per_model_probs   = self._predict_probs(tensor)
        ensemble_probs    = self._combine(per_model_probs)
        pred_idx          = int(ensemble_probs.argmax())
        pred_class        = CLASS_NAMES[pred_idx]
        confidence        = float(ensemble_probs[pred_idx])

        gradcams = None
        if return_gradcam:
            gradcams = self._compute_gradcams(tensor, rgb01, pred_idx)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResult(
            predicted_class     = pred_class,
            predicted_class_idx = pred_idx,
            confidence          = confidence,
            ensemble_probs      = {CLASS_NAMES[i]: float(ensemble_probs[i])
                                   for i in range(len(CLASS_NAMES))},
            model_probs         = {
                name: {CLASS_NAMES[i]: float(p[i]) for i in range(len(CLASS_NAMES))}
                for name, p in per_model_probs.items()
            },
            inference_time_ms   = elapsed_ms,
            original_rgb        = rgb01,
            gradcams            = gradcams,
        )

    # ----------------------------------------------------------------- #
    # Convenience: batch prediction (e.g. for a folder of images)       #
    # ----------------------------------------------------------------- #
    def predict_batch(
        self,
        images: list,
        return_gradcam: bool = False,
    ) -> list[PredictionResult]:
        """Sequentially predicts a list of images. Useful for CLI batch jobs."""
        return [self.predict(img, return_gradcam=return_gradcam) for img in images]


# --------------------------------------------------------------------- #
# CLI self-test                                                         #
# --------------------------------------------------------------------- #
def main() -> None:
    """Smoke test: predict one image from each test class."""
    from .config import TEST_DIR

    print("=" * 64)
    print(" Brain Tumor Detection -- Inference API self-test")
    print("=" * 64)

    predictor = BrainTumorPredictor()
    print(f"Device         : {predictor.device}")
    print(f"Ensemble method: {predictor.ensemble_config.method}")
    print(f"Weights        : {predictor.ensemble_config.weights}\n")

    # Warm up
    sample = next((TEST_DIR / CLASS_NAMES[0]).iterdir())
    _ = predictor.predict(sample)

    # One image per class
    for cls in CLASS_NAMES:
        img_path = next((TEST_DIR / cls).iterdir())
        res = predictor.predict(img_path, return_gradcam=False)
        ok  = "OK " if res.predicted_class == cls else "WRG"
        print(f"  [{ok}] true={cls:11s}  pred={res.predicted_class:11s}  "
              f"conf={res.confidence:.4f}  time={res.inference_time_ms:6.1f}ms")

    # Demo Grad-CAM timing
    res = predictor.predict(sample, return_gradcam=True)
    print(f"\nWith Grad-CAM:")
    print(f"  predicted: {res.predicted_class} ({res.confidence:.4f})")
    print(f"  total time: {res.inference_time_ms:.1f} ms")
    print(f"  gradcam shapes: {[v.shape for v in res.gradcams.values()]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
