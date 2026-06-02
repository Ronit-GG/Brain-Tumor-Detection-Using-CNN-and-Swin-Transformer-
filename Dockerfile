# =====================================================================
#  Brain Tumor Detection -- container image
# ---------------------------------------------------------------------
#  Build:
#     docker build -t brain-tumor:1.0 .
#
#  Run (Streamlit UI - default):
#     docker run --rm -p 8501:8501 brain-tumor:1.0
#     open http://localhost:8501
#
#  Run (FastAPI REST):
#     docker run --rm -p 8000:8000 brain-tumor:1.0 \
#         uvicorn app.api:app --host 0.0.0.0 --port 8000
#     curl -F "file=@brain.jpg" http://localhost:8000/predict
#
#  Notes
#  -----
#  * Uses CPU-only PyTorch for portability (final image ~2.5 GB).
#    To enable GPU, use nvidia/cuda:12.6-runtime-ubuntu22.04 as base
#    and install the CUDA torch wheel.
#  * The 3 trained checkpoints (~127 MB total) are baked into the image.
#    Mount them as a volume in production to swap weights without
#    rebuilding.
# =====================================================================

FROM python:3.11-slim AS base

# System libs needed by opencv-python and pillow at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- 1. Install CPU PyTorch (smaller than CUDA, no NVIDIA dep) -------
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.0 \
        torchvision==0.27.0

# --- 2. Install the rest of the deps --------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- 3. Copy source code + trained weights --------------------------
COPY src/        ./src/
COPY app/        ./app/
COPY models/     ./models/

# Streamlit / FastAPI ports
EXPOSE 8501 8000

# Health check (works for both Streamlit on 8501 and FastAPI on 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health \
        || curl -fsS http://localhost:8000/health \
        || exit 1

# Default: Streamlit UI. Override CMD to run FastAPI instead.
CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
