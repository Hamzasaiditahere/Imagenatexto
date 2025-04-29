import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# Configuración de la app
st.set_page_config(page_title="OCR Liviano", layout="centered")
st.title("🔠 OCR para Números, Letras y Símbolos")

# Caracteres reconocidos (deben coincidir con tu entrenamiento)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

@st.cache_resource
def load_model():
    try:
        # Verificar si el archivo existe
        model_path = "ocr_model_compatible.tflite"  # Nombre exacto de tu archivo
        if not os.path.exists(model_path):
            st.error(f"❌ No se encontró el archivo: {model_path}")
            st.stop()
            
        # Cargar modelo
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Verificación rápida del modelo
        input_details = interpreter.get_input_details()
        if input_details[0]['shape'][1:] != (32, 32, 1):  # Ajusta según tu modelo
            st.warning("⚠️ El modelo espera un tamaño de entrada diferente al configurado")
        
        return interpreter
    except Exception as e:
        st.error(f"ERROR CRÍTICO: {str(e)}")
        st.stop()

# Carga el modelo con un spinner visual
with st.spinner("Cargando modelo OCR..."):
    model = load_model()

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    try:
        # Procesamiento de la imagen
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar y normalizar (ajusta según tu modelo)
        img = cv2.resize(img, (32, 32))
        img_preprocessed = np.expand_dims(img, axis=(0, -1)).astype(np.float32) / 255.0
        
        # Mostrar imagen
        st.image(img, caption="Imagen procesada (32x32 píxeles)", width=150)
        
        # Predicción
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], img_preprocessed)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        
        # Resultados
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        st.success(f"**Predicción:** `{CHARS[predicted_idx]}`")
        st.progress(float(confidence))
        st.caption(f"Confianza: {confidence*100:.1f}%")
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
