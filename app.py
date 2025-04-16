import streamlit as st
from PIL import Image
import easyocr  # Usamos easyocr en lugar de pytesseract

# Función para detectar texto usando easyocr
def detect_text_from_image(image):
    reader = easyocr.Reader(['en'])  # Puedes agregar más idiomas si lo necesitas
    result = reader.readtext(image)
    text = ""
    for detection in result:
        text += detection[1] + "\n"
    return text

# Título de la aplicación
st.title("Imagen a Texto")

# Subir una imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Detectar Texto"):
        extracted_text = detect_text_from_image(image)
        st.text_area("Texto Detectado", extracted_text)
