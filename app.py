import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Configuración
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_SIZE = 48

@st.cache_resource
def load_model():
    # Verificar si el archivo existe
    if not os.path.exists("tiny_ocr.tflite"):
        st.error("❌ Error: Archivo 'tiny_ocr.tflite' no encontrado")
        st.stop()
    
    try:
        interpreter = tf.lite.Interpreter(model_path="tiny_ocr.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        st.stop()

def predict(image):
    try:
        # Preprocesamiento
        img = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)/255.0
        
        # Predicción
        model = load_model()
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], img_array)
        model.invoke()
        return model.get_tensor(output_details[0]['index'])[0]
    except Exception as e:
        st.error(f"❌ Error en predicción: {str(e)}")
        return np.zeros(len(CHARS))

# Interfaz mejorada
st.title("🔠 OCR Liviano")
uploaded_file = st.file_uploader("Sube una imagen con un carácter", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        st.image(img, width=150, caption="Imagen cargada")
        
        with st.spinner("Analizando..."):
            preds = predict(img)
            char = CHARS[np.argmax(preds)]
            confidence = np.max(preds)*100
            
            st.success(f"✅ Predicción: {char}")
            st.metric("Confianza", f"{confidence:.1f}%")
            
    except Exception as e:
        st.error(f"Error al procesar imagen: {str(e)}")
