import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Asegúrate que este orden sea IDÉNTICO al de entrenamiento
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ocr_model_compatible.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()
input_details = model.get_input_details()

st.title("🔠 OCR de Alta Precisión")

uploaded_file = st.file_uploader("Sube una letra/número/símbolo", type=["png","jpg","jpeg"])

if uploaded_file:
    # Preprocesamiento MEJORADO
    img = Image.open(uploaded_file).convert('L')
    img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarización
    img = np.array(img.resize((32, 32))) / 255.0
    img = 1 - img  # Inversión de colores
    img = np.expand_dims(img, axis=(0, -1)).astype(np.float32)
    
    st.image(img[0,:,:,0], width=150)
    
    # Predicción
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    preds = model.get_tensor(model.get_output_details()[0]['index'])
    
    # Resultados MEJORADOS
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    
    st.subheader("Resultados:")
    for i, idx in enumerate(top3_idx):
        st.write(f"{i+1}. {CHARS[idx]} ({(preds[0][idx]*100):.1f}%)")
