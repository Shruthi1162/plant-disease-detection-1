import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #f4fff7;
}
.main {
    background-color: #f4fff7;
}
.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ----------------------------
# Supported classes (TOMATO ONLY)
# ----------------------------
CLASS_NAMES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy Tomato Leaf"
]

DISEASE_INFO = {
    "Tomato Bacterial Spot": "Bacterial disease causing dark, water-soaked spots. Avoid overhead irrigation.",
    "Tomato Early Blight": "Fungal disease causing brown concentric rings on older leaves.",
    "Tomato Late Blight": "Serious fungal disease with dark lesions. Remove infected plants immediately.",
    "Tomato Leaf Mold": "Yellow patches on upper leaf surface with mold below.",
    "Tomato Septoria Leaf Spot": "Small circular spots with dark borders.",
    "Tomato Spider Mites": "Tiny pests causing yellow speckles and webbing.",
    "Tomato Target Spot": "Brown lesions with concentric circles.",
    "Tomato Yellow Leaf Curl Virus": "Leaf curling and yellowing, spread by whiteflies.",
    "Tomato Mosaic Virus": "Mottled leaves and stunted growth.",
    "Healthy Tomato Leaf": "Leaf appears healthy with no visible disease."
}

# ----------------------------
# App UI
# ----------------------------
st.title("🌱 Plant Disease Detection")
st.caption("⚠️ This model is trained **only for tomato leaves**")

plant_choice = st.selectbox(
    "Select plant type",
    ["Tomato (Supported)", "Other plant (Not supported)"]
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction logic
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    # Safe label handling
    if predicted_index < len(CLASS_NAMES):
        predicted_label = CLASS_NAMES[predicted_index]
    else:
        predicted_label = "Unknown / Non-Tomato Leaf"

    description = DISEASE_INFO.get(
        predicted_label,
        "This plant is not supported by the model."
    )

    # ----------------------------
    # Results
    # ----------------------------
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.success("Prediction completed!")

    st.markdown(f"### 🌿 Disease: **{predicted_label}**")
    st.write(description)
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Warnings
    if plant_choice != "Tomato (Supported)":
        st.warning("⚠️ Model is trained only on tomato leaves. Result may be incorrect.")

    if confidence < 40:
        st.warning("⚠️ Low confidence prediction. Image may be healthy or unclear.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("🚀 Built using TensorFlow + Streamlit | Hackathon Ready")

