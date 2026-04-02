import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Retina Disease Detection", layout="centered")

st.title("👁️ Retina Disease Detection System")
st.write("Upload a retina image to check if it's Healthy or Diseased.")

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_vision_model():
    model = tf.keras.models.load_model(
        "retina_model.h5",
        compile=False,
        safe_mode=False   # Important fix
    )
    return model

model = load_vision_model()

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))   # adjust if your model uses different size
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)[0][0]

    # Result
    if prediction > 0.5:
        st.error(f"⚠️ Diseased Retina (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ Healthy Retina (Confidence: {1 - prediction:.2f})")