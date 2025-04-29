# app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import string
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model('modelo_ocr.h5', compile=False)

chars = list(string.digits + string.ascii_uppercase)

st.title("Reconocimiento OCR simple")

uploaded_file = st.file_uploader("Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)

    pred = model.predict(img_array)
    predicted_char = chars[np.argmax(pred)]

    st.image(image, caption="Imagen cargada", width=150)
    st.write(f"🔤 Carácter detectado: **{predicted_char}**")
