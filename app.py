import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Configuración de la app
st.set_page_config(page_title="OCR Liviano", layout="wide")
st.title("🔍 OCR para Números, Letras y Símbolos")

# Cargar modelo y caracteres
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ocr_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
col1, col2 = st.columns(2)

if uploaded_file:
    # Preprocesamiento
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img_preprocessed = np.expand_dims(img, axis=(0, -1)).astype(np.float32) / 255.0
    
    # Predicción
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    model.set_tensor(input_details[0]['index'], img_preprocessed)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])
    
    # Resultados
    predicted_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_idx]
    
    with col1:
        st.image(img, caption="Imagen procesada", width=200)
        
    with col2:
        st.subheader("Resultado")
        st.metric("Carácter detectado", value=CHARS[predicted_idx])
        st.progress(float(confidence))
        st.caption(f"Confianza: {confidence*100:.1f}%")
        
        if confidence < 0.7:
            st.warning("La confianza es baja. Intenta con una imagen más clara.")
