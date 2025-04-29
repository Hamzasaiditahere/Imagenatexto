import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# LISTA CORREGIDA DE CARACTERES (asegúrate que coincida con tu entrenamiento)
CHARS = (
    "0123456789" +                    # Dígitos
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +    # Mayúsculas
    "abcdefghijklmnopqrstuvwxyz" +    # Minúsculas
    "!@#$%^&*()_+-=[]{};:,.<>?/"      # Símbolos
)  # Total: 10 + 26 + 26 + 16 = 78 caracteres

@st.cache_resource
def load_model():
    try:
        # Descargar modelo directamente desde GitHub
        model_url = "https://github.com/Hamzasaiditahere/Imagenatexto/raw/main/ocr_model_compatible.tflite"
        model_path = tf.keras.utils.get_file("ocr_model.tflite", model_url)
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"ERROR: {str(e)}")
        st.stop()

model = load_model()
input_details = model.get_input_details()

st.title("🔠 OCR para Mayúsculas y Minúsculas")

def preprocess_image(image):
    """Preprocesamiento mejorado para letras mayúsculas"""
    img = image.convert('L')  # Escala de grises
    
    # Binarización adaptativa (ajusta el 150 según necesites)
    img = img.point(lambda x: 0 if x < 150 else 255, '1')
    
    img = np.array(img.resize((32, 32))) / 255.0
    img = 1 - img  # Invertir colores (fondos blancos)
    return np.expand_dims(img, axis=(0, -1)).astype(np.float32)

uploaded_file = st.file_uploader("Sube una imagen con un solo carácter", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        processed_img = preprocess_image(img)
        
        st.image(processed_img[0,:,:,0], width=150, caption="Imagen procesada")
        
        # Predicción
        model.set_tensor(input_details[0]['index'], processed_img)
        model.invoke()
        preds = model.get_tensor(model.get_output_details()[0]['index'])
        
        # Resultados filtrados
        st.subheader("Resultados:")
        top5 = np.argsort(preds[0])[-5:][::-1]
        
        for i, idx in enumerate(top5):
            if idx < len(CHARS):  # Solo mostrar clases válidas
                st.write(f"{i+1}. {CHARS[idx]} ({preds[0][idx]*100:.1f}%)")
        
        # Diagnóstico
        with st.expander("🔍 Ver información técnica"):
            st.write(f"Posición de 'A' en CHARS: {CHARS.index('A')}")
            st.write(f"Posición de 'a' en CHARS: {CHARS.index('a')}")
            st.write("Shape de salida:", preds.shape)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
