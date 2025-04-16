import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

# Inicializamos el lector OCR (EasyOCR preentrenado)
reader = easyocr.Reader(['en'])

# Función para detectar un solo carácter en la imagen
def detect_single_character(image):
    img_rgb = image.convert("RGB")  # Convertir la imagen en RGB
    img_np = np.array(img_rgb)  # Convertir a formato de numpy array
    results = reader.readtext(img_np)

    detected_text = ""
    for result in results:
        detected_text = result[1]

        # Si el texto detectado tiene una longitud de 1, es un solo carácter
        if len(detected_text) == 1:
            return detected_text
    return "No se detectó un solo carácter."

# Título de la aplicación
st.title("OCR - Detectar un solo carácter")

# Subir una imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Detectar Carácter"):
        detected_text = detect_single_character(image)
        st.text_area("Texto Detectado", detected_text)
