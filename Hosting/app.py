import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI vs Human Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
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
.metric-box{background:#fff;border-radius:8px;padding:1rem;text-align:center;
    margin-bottom:0.5rem;border:1px solid #e2e8f0}
</style>
""", unsafe_allow_html=True)

# ── Load Our Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_our_model():
    base         = os.path.dirname(__file__)
    weights_path = os.path.join(base, "models", "model_weights.weights.h5")
    config_path  = os.path.join(base, "models", "model_config.json")

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    inputs  = keras.Input(shape=(224,224,3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model   = keras.Model(inputs, outputs)
    model.load_weights(weights_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return model, config

# ── Load HuggingFace Model Locally ────────────────────────────────────────────
@st.cache_resource
def load_hf_model():
    try:
        from transformers import pipeline
        pipe = pipeline(
            "image-classification",
            model="haywoodsloan/ai-image-detector-deploy",
            device=-1   # CPU
        )
        return pipe
    except Exception:
        return None

# ── Load both models ──────────────────────────────────────────────────────────
with st.spinner("Loading models... please wait"):
    our_model, config = load_our_model()
    hf_pipe           = load_hf_model()

IMG_SIZE = config.get('img_size', 224)

# ── Predictions ───────────────────────────────────────────────────────────────
def local_predict(image: Image.Image):
    img_arr   = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_arr   = np.expand_dims(img_arr, axis=0)
    pred      = our_model.predict(img_arr, verbose=0)[0][0]
    return float(1 - pred), float(pred)   # ai_conf, real_conf

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

def combine(local_ai, local_real, hf_ai, hf_real):
    if hf_ai is not None:
        return (0.4*local_ai + 0.6*hf_ai), (0.4*local_real + 0.6*hf_real)
    return local_ai, local_real


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
                local_ai, local_real = local_predict(image)
                hf_ai, hf_real       = hf_predict(image)
                final_ai, final_real = combine(local_ai, local_real, hf_ai, hf_real)
                st.session_state.result = {
                    "local_ai": local_ai, "local_real": local_real,
                    "hf_ai": hf_ai,       "hf_real": hf_real,
                    "final_ai": final_ai, "final_real": final_real,
                }

with col2:
    st.subheader("Result")

    if "result" not in st.session_state:
        st.info("Upload an image and click Analyse to see results.")
    else:
        r          = st.session_state.result
        final_ai   = r["final_ai"]
        final_real = r["final_real"]
        is_ai      = final_ai > final_real
        label      = "AI Generated" if is_ai else "Real (Human)"

        if is_ai:
            st.error(f"🤖 **{label}**")
        else:
            st.success(f"📷 **{label}**")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if is_ai:
            st.progress(float(final_ai), text=f"🤖 AI Generated — {final_ai*100:.1f}% confidence")
        else:
            st.progress(float(final_real), text=f"📷 Real (Human) — {final_real*100:.1f}% confidence")

        st.divider()
        st.caption("This tool uses AI models to detect AI generated images. Results may not be 100% accurate. Always apply human judgment.")