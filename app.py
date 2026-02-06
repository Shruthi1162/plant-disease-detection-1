import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ---------------------------
# Custom CSS (Aesthetic & Clean)
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fff4, #e6f7f1);
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    text-align: center;
    color: #2f855a;
    font-weight: 700;
}

.info-box {
    background: #edfdf5;
    border-left: 5px solid #38a169;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    color: #22543d;
}

.result-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.disease-title {
    color: #2f855a;
    font-size: 22px;
    font-weight: 600;
}

.description {
    color: #4a5568;
    margin-top: 10px;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #718096;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title & description
# ---------------------------
st.title("🍅 Tomato Leaf Disease Detection")

st.markdown(
    '<div class="info-box">ℹ️ This system is trained <b>only on tomato leaves</b>. '
    'Uploading other plant leaves may give incorrect results.</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a tomato leaf image to identify whether it is healthy or affected by a disease. "
    "The model is trained using the PlantVillage tomato dataset."
)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------------------
# Tomato class names
# ---------------------------
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
    "Tomato Healthy"
]

# ---------------------------
# Tomato disease descriptions
# ---------------------------
DISEASE_INFO = {
    "Tomato Bacterial Spot":
        "Bacterial disease causing dark, water-soaked spots on leaves. Avoid overhead irrigation.",

    "Tomato Early Blight":
        "Fungal disease causing brown concentric rings on older leaves. Improve air circulation and avoid wet foliage.",

    "Tomato Late Blight":
        "Severe fungal disease causing dark, water-soaked lesions. Remove infected plants immediately.",

    "Tomato Leaf Mold":
        "Yellow spots on the upper leaf surface with mold growth underneath. Reduce humidity and improve ventilation.",

    "Tomato Septoria Leaf Spot":
        "Small circular spots with dark borders. Remove infected leaves and apply fungicide.",

    "Tomato Spider Mites":
        "Tiny pests causing yellow stippling and webbing on leaves. Use neem oil or insecticidal soap.",

    "Tomato Target Spot":
        "Brown lesions with target-like rings. Practice crop rotation and apply fungicides.",

    "Tomato Yellow Leaf Curl Virus":
        "Viral disease causing leaf curling and yellowing. Spread by whiteflies.",

    "Tomato Mosaic Virus":
        "Causes mottled leaves and stunted growth. Spread through contact with infected plants.",

    "Tomato Healthy":
        "The tomato leaf appears healthy with no visible disease symptoms."
}

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))

    if predicted_index < len(CLASS_NAMES):
        predicted_label = CLASS_NAMES[predicted_index]
    else:
        predicted_label = "Unknown Tomato Condition"

    description = DISEASE_INFO.get(
        predicted_label,
        "The model could not confidently identify this tomato leaf condition."
    )

    # Display result
    st.markdown(f"""
    <div class="result-card">
        <div class="disease-title">🌿 Disease: {predicted_label}</div>
        <div class="description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    '<div class="footer">🚀 Tomato Leaf Disease Detection | TensorFlow + Streamlit</div>',
    unsafe_allow_html=True
)
