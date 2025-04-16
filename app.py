import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Configura el path de Tesseract (solo si es necesario, dependiendo de tu sistema operativo)
# En sistemas Linux y macOS no suele ser necesario si tesseract está correctamente instalado.
# En Windows podría ser necesario especificar la ruta:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Función para procesar la imagen
def detect_text_from_image(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Realizar el OCR (detección de texto)
    text = pytesseract.image_to_string(gray)
    return text

# Interfaz Streamlit
def main():
    st.title("Detección de Texto en Imágenes")
    st.write("Carga una imagen y detectaremos el texto en ella.")
    
    # Subir la imagen
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Leer la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada.", use_column_width=True)
        
        # Convertir imagen a formato de OpenCV para procesamiento
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convertir de RGB a BGR
        
        # Detectar texto en la imagen
        detected_text = detect_text_from_image(open_cv_image)
        
        # Mostrar el texto detectado
        if detected_text.strip():
            st.subheader("Texto Detectado:")
            st.write(detected_text)
        else:
            st.write("No se detectó texto en la imagen.")

if __name__ == "__main__":
    main()
