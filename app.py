import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Configuración mínima
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_SIZE = 48

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="tiny_ocr.tflite")
    interpreter.allocate_tensors()
    return interpreter

def predict(image):
    img = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)/255.0
    model = load_model()
    model.set_tensor(model.get_input_details()[0]['index'], img_array)
    model.invoke()
    return model.get_tensor(model.get_output_details()[0]['index'])[0]

# Interfaz minimalista
st.title("🅾🅲🆁 🅻🅸🆅🅸🅰🅽🅾")
uploaded_file = st.file_uploader("Sube imagen", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    preds = predict(img)
    char = CHARS[np.argmax(preds)]
    confidence = np.max(preds)*100
    st.image(img, width=150)
    st.success(f"Predicción: {char} ({confidence:.1f}% confianza)")
