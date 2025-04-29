import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Configuración (DEBE COINCIDIR con el entrenamiento)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_SIZE = 48  # Debe coincidir con el entrenamiento

@st.cache_resource
def load_model():
    MODEL_PATH = "ocr_compatible.tflite"
    
    # Verificar existencia
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modelo no encontrado en: {os.path.abspath(MODEL_PATH)}")
        st.stop()
    
    try:
        # Cargar con verificación de compatibilidad
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Prueba de compatibilidad
        input_details = interpreter.get_input_details()
        test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        return interpreter
    except Exception as e:
        st.error(f"Error de compatibilidad: {str(e)}")
        st.write("""
        Solución requerida:
        1. Regenera el modelo en Colab con el código proporcionado
        2. Asegúrate de usar tensorflow==2.15.0
        """)
        st.stop()

def predict(image):
    img = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)/255.0
    
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Interfaz
st.title("🔠 OCR Optimizado")
uploaded_file = st.file_uploader("Sube imagen de un carácter", type=["png","jpg","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=150)
    
    with st.spinner("Analizando..."):
        try:
            preds = predict(img)
            st.success(f"Predicción: {CHARS[np.argmax(preds)]} ({np.max(preds)*100:.1f}%)")
        except Exception as e:
            st.error(f"Error: {str(e)}")
