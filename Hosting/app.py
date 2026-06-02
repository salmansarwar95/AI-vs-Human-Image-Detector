import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import os

st.set_page_config(
    page_title="AI vs Human Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stApp"]{background:#f0f4f8!important}
.block-container{padding:2rem 3rem!important;max-width:1100px!important}
header[data-testid="stHeader"]{display:none!important}
h1,h2,h3{color:#0a2540!important;font-weight:700!important}
[data-testid="stFileUploader"]{background:#fff!important;border:2px dashed #0077b6!important;border-radius:12px!important;padding:1rem!important}
div[data-testid="stButton"] > button{
    background:linear-gradient(135deg,#0077b6,#023e8a)!important;
    color:#fff!important;border:none!important;border-radius:8px!important;
    font-size:16px!important;font-weight:600!important;padding:12px 24px!important;}
div[data-testid="stButton"] > button:hover{
    background:linear-gradient(135deg,#0096c7,#0077b6)!important;
    box-shadow:0 4px 16px rgba(0,119,182,0.35)!important;}
div[data-testid="stAlert"]{border-radius:8px!important}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device('cpu')
IMG_SIZE = 224

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load HuggingFace model (PyTorch pipeline) ─────────────────────────────────
@st.cache_resource
def load_hf_model():
    try:
        from transformers import pipeline
        pipe = pipeline(
            "image-classification",
            model="haywoodsloan/ai-image-detector-deploy",
            device=-1
        )
        return pipe
    except Exception:
        return None

# ── Rebuild EfficientNetB0 in PyTorch + load weights ─────────────────────────
@st.cache_resource
def load_our_model():
    try:
        base         = os.path.dirname(__file__)
        weights_path = os.path.join(base, "models", "model_weights.weights.h5")

        # Use pretrained EfficientNet from torchvision as our model
        # Since weights are in keras format, we use HF model only for our model too
        # Fall back gracefully
        return None
    except Exception:
        return None

with st.spinner("Loading models... please wait"):
    hf_pipe   = load_hf_model()
    our_model = load_our_model()

# ── Prediction ────────────────────────────────────────────────────────────────
def hf_predict(image: Image.Image):
    if hf_pipe is None:
        return None, None
    try:
        results  = hf_pipe(image)
        scores   = {r['label'].lower(): r['score'] for r in results}
        ai_score = scores.get('artificial', scores.get('ai', 0.0))
        re_score = scores.get('human', scores.get('real', 1.0 - ai_score))
        return float(ai_score), float(re_score)
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔍 AI vs Human Image Detector")
st.markdown('<p style="color:#0077b6;font-weight:500;font-size:15px;margin-top:-0.5rem">Detect whether an image is AI Generated or captured by a Human</p>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("🔍 Analyse Image", use_container_width=True):
            with st.spinner("Analysing image..."):
                hf_ai, hf_real = hf_predict(image)
                st.session_state.result = {
                    "hf_ai": hf_ai, "hf_real": hf_real,
                }

with col2:
    st.subheader("Result")

    if "result" not in st.session_state:
        st.info("Upload an image and click Analyse to see results.")
    else:
        r = st.session_state.result

        if r['hf_ai'] is not None:
            is_ai  = r['hf_ai'] > r['hf_real']
            label  = "AI Generated" if is_ai else "Real (Human)"
            conf   = r['hf_ai'] if is_ai else r['hf_real']

            if is_ai:
                st.error(f"🤖 **{label}**")
            else:
                st.success(f"📷 **{label}**")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if is_ai:
                st.progress(float(r['hf_ai']), text=f"🤖 AI Generated — {r['hf_ai']*100:.1f}% confidence")
            else:
                st.progress(float(r['hf_real']), text=f"📷 Real (Human) — {r['hf_real']*100:.1f}% confidence")
        else:
            st.error("Model unavailable. Please try again.")

        st.divider()
        st.caption("This tool uses AI models to detect AI generated images. Results may not be 100% accurate. Always apply human judgment.")
