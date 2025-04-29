import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Configuración
st.set_page_config(page_title="OCR Universal", layout="centered")
st.title("🔤 Reconocimiento de Caracteres")

# Caracteres soportados (AJUSTA ESTO SEGÚN TU MODELO)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

@st.cache_resource
def load_model():
    MODEL_NAME = "ocr_model_compatible.tflite"  # Nombre EXACTO de tu archivo
    
    if not os.path.exists(MODEL_NAME):
        st.error(f"❌ Error: El archivo '{MODEL_NAME}' no existe")
        st.stop()
    
    try:
        # Carga universal compatible
        interpreter = tf.lite.Interpreter(model_path=MODEL_NAME)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

# Carga el modelo
model = load_model()
st.success("✅ Modelo cargado correctamente")

# Interfaz
uploaded_file = st.file_uploader("Sube una imagen con un carácter", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Procesamiento INFALIBLE
        img = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
        img = np.array(img.resize((32, 32))) / 255.0  # Normalizar
        img = np.expand_dims(img, axis=(0, -1)).astype(np.float32)  # Añadir dimensiones
        
        # Mostrar imagen
        st.image(img[0,:,:,0], caption="Carácter procesado", width=150)
        
        # Predicción
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        preds = model.get_tensor(output_details[0]['index'])
        
        # Resultados
        char_idx = np.argmax(preds[0])
        confidence = preds[0][char_idx]
        
        st.metric("Carácter detectado", value=f"{CHARS[char_idx]}")
        st.progress(float(confidence))
        st.caption(f"Precisión: {confidence*100:.1f}%")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
