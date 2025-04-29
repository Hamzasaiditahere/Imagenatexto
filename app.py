import streamlit as st
import numpy as np
import tensorflow as tf
import string
from PIL import Image

# Cargar el modelo (formato SavedModel)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_ocr_saved_model")

model = load_model()
st.write("✅ Modelo cargado correctamente")

# Lista de caracteres posibles
chars = list(string.digits + string.ascii_uppercase)

st.title("🧠 OCR - Reconocimiento de un carácter")

uploaded_file = st.file_uploader("📤 Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')     # Convertir a escala de grises
    image = image.resize((28, 28))                      # Redimensionar a 28x28 píxeles
    img_array = np.array(image) / 255.0                 # Normalizar los píxeles
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Añadir dimensiones para el modelo

    pred = model.predict(img_array)
    predicted_char = chars[np.argmax(pred)]

    st.image(image, caption="🖼 Imagen cargada", width=150)
    st.success(f"🔤 Carácter detectado: **{predicted_char}**")
