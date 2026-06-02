"""
Streamlit frontend for the Brain Tumor Detection ensemble.

Run from the project root:

    streamlit run app/app.py

Then open the URL it prints (usually http://localhost:8501).

The app:
  * accepts a drag-and-drop MRI image OR a sample from the test set,
  * runs all 3 models (CNN + EfficientNet + Swin) + the ensemble,
  * shows the ensemble's class probabilities as a bar chart,
  * shows each model's per-class probabilities as a heat-table,
  * shows Grad-CAM overlays for each model (3-up),
  * warns when ensemble confidence is below a user-set threshold
    -- the recommended "consult a radiologist" gate for clinical use.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when launched via `streamlit run app/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config    import CLASS_NAMES, TEST_DIR
from src.inference import BrainTumorPredictor


# --------------------------------------------------------------------- #
# Page config -- MUST be the first Streamlit call                       #
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Class palette -- matches the notebook plots
CLASS_COLORS = {
    "glioma":     "#66c2a5",
    "meningioma": "#fc8d62",
    "notumor":    "#8da0cb",
    "pituitary":  "#e78ac3",
}

# Pretty display names for the models
MODEL_PRETTY = {
    "cnn":      "Custom CNN",
    "transfer": "EfficientNet-B0",
    "swin":     "Swin-Tiny",
}


# --------------------------------------------------------------------- #
# Cached predictor (loads exactly once per Streamlit session)           #
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading 3 trained models...")
def get_predictor() -> BrainTumorPredictor:
    return BrainTumorPredictor()


# --------------------------------------------------------------------- #
# Sidebar                                                               #
# --------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧠 Brain Tumor Detection")
    st.caption("CNN + Transfer Learning + Swin Transformer ensemble")

    st.markdown("### Model lineup")
    st.markdown(
        """
| Model | Params | Test acc |
|---|---:|---:|
| Custom CNN | 1.2 M | 82.75 % |
| EfficientNet-B0 | 4.0 M | **94.94 %** |
| Swin-Tiny | 27.5 M | 91.94 % |
| **Ensemble** | 32.7 M | **94.69 %** |
"""
    )

    st.markdown("### Settings")
    threshold = st.slider(
        "Low-confidence threshold",
        min_value=0.50, max_value=1.0, value=0.85, step=0.05,
        help="Predictions below this confidence trigger a 'consult radiologist' warning.",
    )
    show_gradcam = st.checkbox(
        "Generate Grad-CAM heatmaps",
        value=True,
        help="Visualizes which regions the models look at. Adds ~1 second / image.",
    )

    st.markdown("---")
    st.caption(
        "⚠ **Academic / educational use only.** "
        "This tool is NOT a medical device and is NOT a substitute for "
        "a qualified radiologist's diagnosis."
    )


# --------------------------------------------------------------------- #
# Header                                                                #
# --------------------------------------------------------------------- #
st.markdown("# Brain Tumor MRI Classification")
st.markdown(
    "Upload a brain MRI image (or choose a sample from the test set) and "
    "the 3-model ensemble will predict the tumor class with confidence "
    "and visual explanations of *where* each model is looking."
)

# --------------------------------------------------------------------- #
# Image input -- upload OR sample picker                                #
# --------------------------------------------------------------------- #
input_col1, input_col2 = st.columns([3, 2])

with input_col1:
    uploaded = st.file_uploader(
        "Drag and drop or click to upload an MRI image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Any 2D brain MRI slice. Will be resized to 224x224.",
    )

with input_col2:
    sample_class = st.selectbox(
        "...or pick a sample from the test set",
        options=["(none)"] + CLASS_NAMES,
        index=0,
    )
    if sample_class != "(none)":
        sample_files = sorted((TEST_DIR / sample_class).iterdir())[:20]
        sample_choice = st.selectbox(
            f"Pick a {sample_class} sample",
            options=[f.name for f in sample_files],
            index=0,
        )
    else:
        sample_choice = None

# Resolve which image (if any) is the active one.
img: Image.Image | None = None
img_caption: str = ""
true_label: str | None = None

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img_caption = f"Uploaded: {uploaded.name}"
elif sample_choice is not None:
    path = TEST_DIR / sample_class / sample_choice
    img = Image.open(path).convert("RGB")
    img_caption = f"Test sample: {sample_choice}"
    true_label = sample_class

if img is None:
    st.info("⬆ Upload an MRI image or pick a sample to begin.")
    st.stop()

# --------------------------------------------------------------------- #
# Run prediction                                                        #
# --------------------------------------------------------------------- #
predictor = get_predictor()
with st.spinner("Running 3 models + ensemble..."):
    result = predictor.predict(img, return_gradcam=show_gradcam)

# --------------------------------------------------------------------- #
# Top-line result: image | prediction + bar chart                       #
# --------------------------------------------------------------------- #
left, right = st.columns([1, 1.3])

with left:
    st.image(img, caption=img_caption, use_container_width=True)
    st.caption(f"Inference time: **{result.inference_time_ms:.0f} ms** "
               f"({'with' if show_gradcam else 'without'} Grad-CAM)")

with right:
    pred = result.predicted_class
    conf = result.confidence
    color = CLASS_COLORS[pred]

    # Headline prediction
    st.markdown(
        f"### Prediction: "
        f"<span style='color:{color}; font-size:1.4em'><b>{pred.upper()}</b></span>",
        unsafe_allow_html=True,
    )

    # True-label badge for test samples (helps demos and viva)
    if true_label is not None:
        is_correct = (true_label == pred)
        if is_correct:
            st.markdown(f"✅ Matches ground truth: **{true_label}**")
        else:
            st.markdown(f"❌ Ground truth was: **{true_label}** (model was wrong)")

    # Confidence + threshold gate
    m_col1, m_col2 = st.columns([1, 1])
    m_col1.metric("Ensemble confidence", f"{conf:.2%}")
    m_col2.metric("Threshold", f"{threshold:.0%}")

    if conf < threshold:
        st.warning(
            f"⚠ Confidence {conf:.2%} is below the {threshold:.0%} threshold. "
            "**Recommend radiologist review** before acting on this prediction."
        )
    else:
        st.success("✓ High-confidence prediction (above threshold).")

    # Ensemble probability bar chart
    df = pd.DataFrame({
        "class":       list(result.ensemble_probs.keys()),
        "probability": list(result.ensemble_probs.values()),
    })
    bar = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", title="Ensemble probability")),
            y=alt.Y("class:N", sort="-x", title=None),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(
                    domain=list(CLASS_COLORS.keys()),
                    range=list(CLASS_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("class:N"),
                alt.Tooltip("probability:Q", format=".4f"),
            ],
        )
        .properties(height=180)
    )
    st.altair_chart(bar, use_container_width=True)

# --------------------------------------------------------------------- #
# Per-model probability breakdown                                       #
# --------------------------------------------------------------------- #
st.markdown("---")
st.markdown("### Per-model probability breakdown")
st.caption(
    "Each base model's confidence for each class. The ensemble combines "
    "these using the weights found in STEP 14: "
    f"`weighted{predictor.ensemble_config.weights}`."
)

mp_df = pd.DataFrame(result.model_probs).T          # rows=models, cols=classes
mp_df.index = [MODEL_PRETTY[k] for k in mp_df.index]
mp_df = mp_df[CLASS_NAMES] * 100                    # to percent
styled = mp_df.style.format("{:.2f}%") \
                    .background_gradient(cmap="RdYlGn", vmin=0, vmax=100)
st.dataframe(styled, use_container_width=True)

# --------------------------------------------------------------------- #
# Grad-CAM visualizations                                               #
# --------------------------------------------------------------------- #
if show_gradcam and result.gradcams is not None:
    st.markdown("---")
    st.markdown("### Grad-CAM — *where* each model is looking")
    st.caption(
        "Red = high importance. Blue = low. Each heatmap reflects "
        f"the model's evidence for the predicted class **{pred}**."
    )

    g_cols = st.columns(3)
    for col, key in zip(g_cols, ["cnn", "transfer", "swin"]):
        with col:
            st.markdown(f"**{MODEL_PRETTY[key]}**")
            st.image(result.gradcams[key], use_container_width=True)
            st.caption(
                f"This model's confidence for {pred}: "
                f"**{result.model_probs[key][pred]:.3f}**"
            )

# --------------------------------------------------------------------- #
# Footer                                                                #
# --------------------------------------------------------------------- #
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 0.85em;'>
Brain Tumor Detection &mdash; Final Year Project<br>
Custom CNN + EfficientNet-B0 (transfer learning) + Swin Transformer (timm), combined via weighted-average ensemble.<br>
Trained on the Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle).<br>
<b>For academic and educational use only.</b>
</div>
""",
    unsafe_allow_html=True,
)
