import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

st.set_page_config(
    page_title="OCR - Detectar un solo carácter",
    page_icon="🔠",
    layout="centered"
)

st.title("OCR - Detectar un solo carácter")

# Inicializamos el lector OCR
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    st.error(f"Error inicializando EasyOCR: {e}")

def detect_single_character(image):
    try:
        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        results = reader.readtext(img_np)
        for result in results:
            detected_text = result[1]
            if len(detected_text) == 1:
                return detected_text
        return "No se detectó un solo carácter."
    except Exception as e:
        return f"Error durante la detección: {e}"

uploaded_file = st.file_uploader("Sube una imagen (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    if st.button("Detectar Carácter"):
        result = detect_single_character(image)
        st.text_area("Texto Detectado", result)
