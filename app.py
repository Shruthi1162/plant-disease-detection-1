import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------
# CSS (ONLY UI – NO LOGIC CHANGES)
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #f3fff8;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 30px;
}

.upload-box {
    border-radius: 16px;
    padding: 20px;
    background-color: #ffffff;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

.result-card {
    background-color: #ffffff;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Soft pastel green background */
body {
    background: linear-gradient(
        135deg,
        #f0fff7 0%,
        #f7fffb 50%,
        #ffffff 100%
    );
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #14532d;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 32px;
}

/* Upload section */
.upload-box {
    border-radius: 18px;
    padding: 22px;
    background-color: #ffffff;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.06);
    margin-bottom: 24px;
}

/* Result card */
.result-card {
    background-color: #ffffff;
    padding: 26px;
    border-radius: 18px;
    box-shadow: 0 14px 35px rgba(0, 0, 0, 0.08);
    margin-top: 28px;
    border-left: 6px solid #22c55e;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-top: 42px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Title & description
# ---------------------------
st.markdown("<div class='main-title'>🌱 Plant Disease Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload a leaf image to detect whether it is healthy or affected by a plant disease. "
    "This model is trained on the PlantVillage dataset.</div>",
    unsafe_allow_html=True
)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------------------
# Class names + descriptions
# ---------------------------
CLASS_NAMES = [
    "Scab",
    "Black Rot",
    "Cedar Rust",
    "Healthy",
    " Healthy",
    " Powdery Mildew",
    " Healthy",
    " Gray Leaf Spot",
    " Common Rust",
    " Northern Leaf Blight",
    " Healthy",
    " Black Rot",
    " Esca",
    " Leaf Blight",
    " Healthy",
    " Haunglongbing",
    " Bacterial Spot",
    " Healthy",
    " Bell Bacterial Spot",
    " Bell Healthy",
    " Early Blight",
    " Late Blight",
    " Healthy",
    " Healthy",
    " Healthy",
    "Powdery Mildew",
    "Leaf Scorch",
    " Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# ---------------------------
# File uploader
# ---------------------------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Preprocess image
    # ---------------------------
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]

    st.success("✅ Prediction completed!")

    # ---------------------------
    # Display result
    # ---------------------------
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    st.subheader(f"🌿 Disease: {predicted_label}")

    description = DISEASE_INFO.get(
        predicted_label,
        "No detailed description available for this disease."
    )
    st.write(description)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    "<div class='footer'>🚀 Built with TensorFlow + Streamlit</div>",
    unsafe_allow_html=True
)
