import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import cv2

# Configuración de caracteres (62 caracteres)
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

@st.cache_resource
def load_model():
    try:
        # Cargar modelo desde GitHub
        model_url = "https://github.com/Hamzasaiditahere/Imagenatexto/raw/main/ocr_model_compatible.tflite"
        model_path = tf.keras.utils.get_file("ocr_model.tflite", model_url)
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

model = load_model()
input_details = model.get_input_details()
output_details = model.get_output_details()

st.title("🔠 Reconocimiento de Caracteres")

def preprocess_image(image):
    """Preprocesamiento mejorado para OCR"""
    # Convertir a escala de grises
    img = image.convert('L')
    # Binarización adaptativa
    img = img.point(lambda x: 0 if x < 150 else 255)
    # Redimensionar y normalizar
    img = np.array(img.resize((32, 32))) / 255.0
    # Invertir colores si es necesario
    img = 1 - img
    return np.expand_dims(img, axis=(0, -1)).astype(np.float32)

uploaded_file = st.file_uploader("Sube una imagen con un solo carácter", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Procesar imagen
        img = Image.open(uploaded_file)
        processed_img = preprocess_image(img)
        
        # Mostrar imagen procesada
        st.image(processed_img[0,:,:,0], caption="Imagen procesada", width=150)
        
        # Predicción
        model.set_tensor(input_details[0]['index'], processed_img)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        
        # Mostrar resultados
        st.subheader("Resultados:")
        
        # Solo considerar las primeras 62 clases (para evitar índices inválidos)
        valid_predictions = predictions[0][:len(CHARS)]
        top5 = np.argsort(valid_predictions)[-5:][::-1]
        
        for i, idx in enumerate(top5):
            st.write(f"{i+1}. {CHARS[idx]} ({valid_predictions[idx]*100:.1f}%)")
            
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
