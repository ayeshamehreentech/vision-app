import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.express as px
from streamlit_image_comparison import image_comparison
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="VisionAI | Precision Diagnostics",
    layout="wide",
    page_icon="👁️"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header { background: linear-gradient(90deg, #0f172a, #1e293b); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; }
    .diagnosis-card { background: white; padding: 25px; border-radius: 12px; border-left: 8px solid #3b82f6; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header"><h1>👁️ VisionAI: Advanced OCT Analysis</h1></div>',
    unsafe_allow_html=True
)

# --- LOAD MODEL (FIXED) ---
@st.cache_resource
def load_vision_model():
    model_path = "retina_model.keras"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Add 'retina_model.keras' to project folder.")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False   # 🔥 FIX for your error
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_vision_model()

# --- CLASS LABELS ---
CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'GLAUCOMA', 'MH']

# --- PREDICTION FUNCTION ---
def get_prediction(image, model):
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype('float32')

    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    return prediction

# --- FILE UPLOAD ---
file = st.file_uploader("📤 Upload Patient OCT Image", type=["png", "jpg", "jpeg"])

if file is not None and model is not None:
    img = Image.open(file)

    probs = get_prediction(img, model)
    label = CLASS_NAMES[np.argmax(probs)]
    conf = np.max(probs) * 100

    # --- IMAGE COMPARISON ---
    st.subheader("🖱️ Interactive Comparison Slider")

    healthy_path = "healthy.jpg"

    if os.path.exists(healthy_path):
        ref_img = Image.open(healthy_path)
        image_comparison(
            img1=ImageOps.fit(img, (1000, 450)),
            img2=ImageOps.fit(ref_img, (1000, 450)),
            label1=f"PATIENT: {label}",
            label2="HEALTHY REFERENCE",
            width=1000,
            in_memory=True
        )
    else:
        st.warning("⚠️ Healthy reference image not found.")
        st.image(img, width=500)

    # --- RESULTS ---
    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"""
            <div class="diagnosis-card">
                <h3 style="color:#64748b;">DETECTION RESULT</h3>
                <h1 style="color:#1e3a8a;">{label}</h1>
                <p>Confidence: <b>{conf:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Clinical Insight")
        if label in ["DME", "DR"]:
            st.info("Fluid accumulation and retinal swelling detected.")
        elif label in ["AMD", "DRUSEN"]:
            st.info("Presence of drusen deposits or retinal elevation.")
        elif label == "MH":
            st.info("Macular hole detected in central retina.")
        else:
            st.info(f"Pattern consistent with {label}.")

    with col2:
        fig = px.bar(
            x=CLASS_NAMES,
            y=probs * 100,
            title="Prediction Confidence (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for Research & Educational Purposes | VisionAI 2026")