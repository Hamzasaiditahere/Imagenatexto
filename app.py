import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# Configuración de la app
st.set_page_config(page_title="OCR Liviano", layout="centered")
st.title("🔠 OCR para Números, Letras y Símbolos")

# Caracteres reconocidos (debe coincidir con el entrenamiento)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

@st.cache_resource
def load_model():
    try:
        # Verificar si el archivo existe
        if not os.path.exists("ocr_model.tflite"):
            raise FileNotFoundError("El archivo 'ocr_model.tflite' no existe en el directorio")
            
        # Cargar modelo con verificación de compatibilidad
        interpreter = tf.lite.Interpreter(model_path="ocr_model.tflite")
        interpreter.allocate_tensors()
        
        # Prueba rápida de inferencia
        input_details = interpreter.get_input_details()
        test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        return interpreter
    except Exception as e:
        st.error(f"ERROR CRÍTICO: {str(e)}")
        st.stop()  # Detener la app si no se puede cargar el modelo

# Cargar modelo con spinner
with st.spinner("Cargando modelo OCR..."):
    model = load_model()

# Interfaz principal
uploaded_file = st.file_uploader("Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Procesamiento de imagen
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img_preprocessed = np.expand_dims(img, axis=(0, -1)).astype(np.float32) / 255.0
        
        # Mostrar imagen
        st.image(img, caption="Imagen procesada (32x32 píxeles)", width=150)
        
        # Inferencia
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
