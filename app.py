import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Title
# -------------------------------
st.title("🌱 Plant Disease Detection")

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model("plant_disease_model.keras")

# -------------------------------
# Class names (example – adjust order if needed)
CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew",
    "Cherry Healthy",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn Northern Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Leaf Blight",
    "Grape Healthy",
    "Orange Haunglongbing",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Strawberry Healthy",
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
DISEASE_INFO = {
    "Tomato Early Blight": "Caused by a fungus. Leads to brown spots on older leaves. Reduce moisture and use fungicide.",
    "Tomato Late Blight": "Serious disease causing dark patches. Remove infected plants immediately.",
    "Tomato Healthy": "The plant appears healthy with no visible disease symptoms.",
    "Potato Early Blight": "Fungal disease causing dry brown spots on leaves.",
    "Potato Late Blight": "Fast-spreading disease causing leaf decay.",
    "Apple Scab": "Fungal disease causing dark spots on leaves and fruit.",
}




# Upload image
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction logic
# -------------------------------
if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    # Show result
    st.success("✅ Prediction completed!")
 predicted_label = CLASS_NAMES[predicted_index]
description = DISEASE_INFO.get(
    predicted_label,
    "No detailed description available for this disease."
)

st.success("✅ Prediction completed!")
st.subheader(f"🦠 Disease: {predicted_label}")
st.write(description)


