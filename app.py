import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fff4, #e6f7f1);
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
    color: #14532d;
}

.result-card {
    background-color: white;
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

# ---------------------------
# Title
# ---------------------------
st.markdown("<div class='main-title'>🍅 Tomato Leaf Disease Detection</div>", unsafe_allow_html=True)

st.info("ℹ
