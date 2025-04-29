import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# CORREGIDO: Usa SOLO minúsculas o mayúsculas (no ambas)
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"  # 62 caracteres
# Verifica que coincida con tu entrenamiento:
st.write(f"Total caracteres: {len(CHARS)}. Ejemplo: 'a' está en posición {CHARS.index('a')}")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ocr_model_compatible.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()
input_details = model.get_input_details()

st.title("🔠 OCR Corregido")

uploaded_file = st.file_uploader("Sube una imagen CLARA de un carácter", type=["png","jpg","jpeg"])

if uploaded_file:
    # PREPROCESAMIENTO MEJORADO:
    img = Image.open(uploaded_file).convert('L')
    img = img.point(lambda x: 0 if x < 150 else 255, '1')  # Ajusta 150 para mejor binarización
    img = np.array(img.resize((32, 32))) / 255.0
    img = 1 - img  # Invierte si las letras son blancas con fondo negro
    img = np.expand_dims(img, axis=(0, -1)).astype(np.float32)
    
    st.image(img[0,:,:,0], width=150, caption="Imagen procesada", clamp=True)

    # PREDICCIÓN:
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    preds = model.get_tensor(model.get_output_details()[0]['index'])
    
    # RESULTADOS:
    top5 = np.argsort(preds[0])[-5:][::-1]  # Top 5 predicciones
    
    st.subheader("Resultados:")
    for i, idx in enumerate(top5):
        char = CHARS[idx] if idx < len(CHARS) else f"Clase {idx}"
        st.write(f"{i+1}. {char} ({preds[0][idx]*100:.1f}%)")
    
    # Diagnóstico adicional:
    with st.expander("🔍 Ver diagnóstico técnico"):
        st.write("Predicción para imagen negra (debe ser 0):", np.argmax(model.predict(np.zeros((1,32,32,1)))))
        st.write("Shape de salida:", preds.shape)
