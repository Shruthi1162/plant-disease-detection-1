import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App title
st.title("🌱 Plant Disease Detection")

# Load model (already uploaded to GitHub)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"Prediction done ✅")
    st.write("Predicted class index:", predicted_class)
