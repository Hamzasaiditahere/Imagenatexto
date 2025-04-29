# app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import string
from PIL import Image

# Cargar modelo exportado (SavedModel)
model = tf.keras.models.load_model("modelo_ocr_saved_model")
st.write("✅ Modelo cargado correctamente")

# Definir los caracteres posibles
chars = list(string.digits + string.ascii_uppercase)

st.title("🔤 Reconocimiento OCR simple")

# Subir imagen
uploaded_file = st.file_uploader("📤 Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # blanco y negro
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Formato (1, 28, 28, 1)

    # Predicción
    pred = model.predict(img_array)
    predicted_char = chars[np.argmax(pred)]

    st.image(image, caption="🖼 Imagen cargada", width=150)
    st.write(f"🔎 Carácter detectado: **{predicted_char}**")
