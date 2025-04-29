import streamlit as st
from PIL import Image
import numpy as np
import easyocr

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Confiable")
st.write("Esta versión utiliza EasyOCR, un motor de reconocimiento profesional")

# Inicializar EasyOCR (solo una vez)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en', 'es'])  # Soporte para inglés y español

reader = load_reader()

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen con texto", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Procesamiento de imagen
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen original", width=300)
        
        # Convertir a formato que EasyOCR puede procesar
        img_array = np.array(img)
        
        # Reconocimiento
        results = reader.readtext(img_array)
        
        # Mostrar resultados
        st.subheader("Resultados:")
        for (bbox, text, prob) in results:
            st.write(f"Texto: {text} | Confianza: {prob*100:.1f}%")
            
        if not results:
            st.warning("No se detectó texto en la imagen")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Notas adicionales (CORREGIDO EL TRIPLE QUOTE)
st.markdown("""
**Recomendaciones para mejores resultados:**  
1. Use imágenes con texto claro  
2. Asegúrese que el fondo contrasta con el texto  
3. Para letras individuales, use fuente grande y centrada  
""")
