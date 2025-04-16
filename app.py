import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

# Configuración de la página (debe ir al principio)
st.set_page_config(
    page_title="OCR - Detectar un solo carácter",
    page_icon="🔠",
    layout="centered"
)

# Inicializamos el lector OCR (EasyOCR preentrenado)
reader = easyocr.Reader(['en'])

# Función para detectar un solo carácter en la imagen
def detect_single_character(image):
    try:
        # Convertir la imagen a RGB (si no lo está)
        img_rgb = image.convert("RGB")
        # Convertir a un array de NumPy (para OpenCV y EasyOCR)
        img_np = np.array(img_rgb)
        # Detectar el texto en la imagen usando EasyOCR
        results = reader.readtext(img_np)
        for result in results:
            detected_text = result[1]
            # Si el texto detectado tiene exactamente un carácter, lo retornamos
            if len(detected_text) == 1:
                return detected_text
        return "No se detectó un solo carácter."
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

# Interfaz de usuario
st.title("OCR - Detectar un solo carácter")
st.markdown("Sube una imagen con una letra, número o símbolo y se detectará el carácter.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    if st.button("Detectar Carácter"):
        result = detect_single_character(image)
        st.text_area("Texto Detectado", result)
