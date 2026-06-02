"""
FastAPI REST backend for the Brain Tumor Detection ensemble.

Endpoints
---------
GET  /              -> simple HTML landing page (browseable docs link)
GET  /docs          -> Swagger / OpenAPI interactive docs (auto-generated)
GET  /health        -> liveness probe
GET  /info          -> ensemble metadata
POST /predict       -> multipart image upload -> JSON prediction

Run locally:
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Test from PowerShell:
    Invoke-WebRequest -Uri http://localhost:8000/health
    curl.exe -F "file=@DATASET/Testing/glioma/Te-gl_1.jpg" http://localhost:8000/predict
"""
from __future__ import annotations

import base64
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Make `src` importable when run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from src.config    import CLASS_NAMES
from src.inference import BrainTumorPredictor


# --------------------------------------------------------------------- #
# Predictor lifecycle (load once at startup, reuse across requests)     #
# --------------------------------------------------------------------- #
_predictor: Optional[BrainTumorPredictor] = None


def _get_predictor() -> BrainTumorPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BrainTumorPredictor()
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: warm up the predictor so the first request is fast.
    _get_predictor()
    yield
    # On shutdown: nothing to clean up (PyTorch handles it).


# --------------------------------------------------------------------- #
# App                                                                   #
# --------------------------------------------------------------------- #
app = FastAPI(
    title="Brain Tumor Detection API",
    description=(
        "REST backend for a 3-model ensemble (Custom CNN + EfficientNet-B0 + "
        "Swin-Tiny) classifying brain MRI scans into glioma / meningioma / "
        "notumor / pituitary."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #
def _encode_image_to_base64_png(arr: np.ndarray) -> str:
    """uint8 (H, W, 3) -> base64-encoded PNG string (data URL ready)."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------- #
# Endpoints                                                             #
# --------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    """Liveness probe — does NOT load the model."""
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    """Return ensemble metadata (used by clients to enumerate classes)."""
    p = _get_predictor()
    return {
        "ensemble_method":  p.ensemble_config.method,
        "ensemble_weights": p.ensemble_config.weights,
        "ensemble_notes":   p.ensemble_config.notes,
        "model_names":      list(p.models.keys()),
        "classes":          CLASS_NAMES,
        "image_size":       p.image_size,
        "device":           str(p.device),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Brain MRI image (JPG/PNG/BMP)."),
    include_gradcam: bool = Query(
        False,
        description="If true, also returns base64-encoded Grad-CAM overlays "
                    "for the predicted class (adds ~250-700 ms).",
    ),
) -> JSONResponse:
    """
    Predict the tumor class for a single uploaded MRI image.

    Returns
    -------
    {
        "predicted_class":         "glioma" | "meningioma" | "notumor" | "pituitary",
        "predicted_class_index":   int,
        "confidence":              float in [0, 1],
        "ensemble_probabilities":  {class_name: prob, ...},
        "model_probabilities":     {model_name: {class_name: prob, ...}, ...},
        "inference_time_ms":       float,
        "gradcams_base64":         {model_name: <png-bytes-base64>}   (optional)
    }
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected image MIME type, got '{file.content_type}'.",
        )

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:                                              # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    result = _get_predictor().predict(img, return_gradcam=include_gradcam)

    response = {
        "predicted_class":        result.predicted_class,
        "predicted_class_index":  result.predicted_class_idx,
        "confidence":             result.confidence,
        "ensemble_probabilities": result.ensemble_probs,
        "model_probabilities":    result.model_probs,
        "inference_time_ms":      round(result.inference_time_ms, 2),
    }

    if include_gradcam and result.gradcams is not None:
        response["gradcams_base64"] = {
            name: _encode_image_to_base64_png(arr)
            for name, arr in result.gradcams.items()
        }

    return JSONResponse(content=response)


# --------------------------------------------------------------------- #
# HTML landing page                                                     #
# --------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Brain Tumor Detection API</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif;
           max-width: 760px; margin: 3em auto; padding: 0 1em; color: #222; }
    code { background:#f4f4f4; padding: 2px 6px; border-radius:4px; }
    pre  { background:#0d1117; color:#c9d1d9; padding: 1em; border-radius:6px;
           overflow-x:auto; font-size: 13px; }
    h1   { border-bottom: 2px solid #eee; padding-bottom: .3em; }
    a    { color: #0969da; }
    .pill { display:inline-block; padding:2px 8px; background:#e0e7ff;
            color:#3730a3; border-radius:10px; font-size:12px; font-weight:600; }
  </style>
</head>
<body>
  <h1>🧠 Brain Tumor Detection API</h1>
  <p class="pill">v1.0.0</p>
  <p>REST backend for a 3-model ensemble (Custom&nbsp;CNN + EfficientNet-B0 +
     Swin-Tiny) classifying brain MRI scans.</p>

  <h2>Try it</h2>
  <ul>
    <li><a href="/docs">Interactive Swagger UI</a> &mdash; upload an image right in the browser.</li>
    <li><a href="/health">/health</a> &mdash; liveness probe</li>
    <li><a href="/info">/info</a> &mdash; ensemble metadata</li>
  </ul>

  <h2>curl example</h2>
  <pre>curl -F "file=@brain.jpg" \\
     -F "include_gradcam=false" \\
     http://localhost:8000/predict</pre>

  <h2>JSON response shape</h2>
  <pre>{
  "predicted_class": "glioma",
  "predicted_class_index": 0,
  "confidence": 0.998,
  "ensemble_probabilities": { "glioma": 0.998, "meningioma": 0.001, ... },
  "model_probabilities":    { "cnn": {...}, "transfer": {...}, "swin": {...} },
  "inference_time_ms": 92.3
}</pre>

  <p style="color:#999; font-size:0.9em; margin-top:3em;">
    For academic / educational use only. Not a medical device.
  </p>
</body>
</html>
"""
