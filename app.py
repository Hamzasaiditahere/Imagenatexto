import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import sys

# Configuración
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_SIZE = 48

@st.cache_resource
def load_model():
    try:
        # Cargar modelo con verificación de compatibilidad
        interpreter = tf.lite.Interpreter(model_path="ocr_optimized.tflite")
        
        # Configuración explícita para máxima compatibilidad
        interpreter.allocate_tensors()
        
        # Prueba de funcionamiento
        input_details = interpreter.get_input_details()
        test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        return interpreter
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.write("""
        Solución requerida:
        1. Regenera el modelo con el código de Colab proporcionado
        2. Usa tensorflow==2.12.0 en Colab
        3. Asegúrate de subir el nuevo modelo 'ocr_ultra_compatible.tflite'
        """)
        st.stop()

def predict(image):
    try:
        img = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)/255.0
        
        interpreter = load_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return np.zeros(len(CHARS))

# Interfaz
st.title("🔠 OCR Ultra Compatible")
uploaded_file = st.file_uploader("Sube imagen de un carácter
