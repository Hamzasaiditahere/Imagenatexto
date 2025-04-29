import streamlit as st
import numpy as np
import tensorflow as tf
import string
from PIL import Image

# Cargar modelo con caché
@st.cache_resource
def load_model():
    try:
        # Se recomienda guardar y cargar el modelo como .keras o .h5 para compatibilidad con .predict()
        model = tf.keras.models.load_model("modelo_ocr.keras")  # Cambiar según nombre real
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Lista de caracteres (0-9 y A-Z)
chars = list(string.digits + string.ascii_uppercase)

st.title("🧠 OCR - Reconocimiento de un carácter")

uploaded_file = st.file_uploader("📤 Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('L')  # Escala de grises
        image = image.resize((28, 28))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)

        # Realiza la predicción
        pred = model.predict(img_array)
        predicted_char = chars[np.argmax(pred)]

        st.image(image, caption="🖼 Imagen cargada", width=150)
        st.success(f"🔤 Carácter detectado: **{predicted_char}**")

    except Exception as e:
        st.error(f"⚠️ Error procesando la imagen: {e}")
