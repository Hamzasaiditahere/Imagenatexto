import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# Configuración de la app
st.set_page_config(page_title="OCR Liviano", layout="wide")
st.title("🔍 OCR para Números, Letras y Símbolos")

# Caracteres reconocidos
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

@st.cache_resource
def load_model():
    try:
        # Verificar si el modelo existe
        if not os.path.exists("ocr_model.tflite"):
            st.error("❌ El archivo del modelo no fue encontrado")
            return None
            
        # Cargar modelo
        interpreter = tf.lite.Interpreter(model_path="ocr_model.tflite")
        interpreter.allocate_tensors()
        st.success("✅ Modelo cargado correctamente")
        return interpreter
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

model = load_model()

# Solo mostrar la interfaz si el modelo se cargó correctamente
if model is not None:
    uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            # Leer y preprocesar imagen
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))
            img_preprocessed = np.expand_dims(img, axis=(0, -1)).astype(np.float32) / 255.0
            
            # Mostrar imagen
            st.image(img, caption="Imagen procesada", width=200)
            
            # Predicción
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            model.set_tensor(input_details[0]['index'], img_preprocessed)
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])
            
            # Resultados
            predicted_idx = np.argmax(predictions)
            confidence = predictions[0][predicted_idx]
            
            st.success(f"**Carácter detectado:** `{CHARS[predicted_idx]}`")
            st.write(f"**Confianza:** {confidence*100:.1f}%")
            
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")
