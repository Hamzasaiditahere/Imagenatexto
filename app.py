import streamlit as st
import numpy as np
import tensorflow as tf
import string
from PIL import Image

# Cargar modelo (formato SavedModel, NO .h5)
model = tf.keras.models.load_model("modelo_ocr")  # carpeta sin extensión .h5
st.write("✅ Modelo cargado correctamente")

# Lista de caracteres posibles
chars = list(string.digits + string.ascii_uppercase)

st.title("🧠 OCR - Reconocimiento de un carácter")

uploaded_file = st.file_uploader("📤 Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')     # escala de grises
    image = image.resize((28, 28))                      # tamaño esperado por el modelo
    img_array = np.array(image) / 255.0                 # normalización
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 28, 28, 1)

    pred = model.predict(img_array)
    predicted_char = chars[np.argmax(pred)]

    st.image(image, caption="🖼 Imagen cargada", width=150)
    st.success(f"🔤 Carácter detectado: **{predicted_char}**")
