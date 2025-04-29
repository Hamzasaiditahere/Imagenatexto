import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Universal")
st.write("Sistema profesional compatible con todas versiones")

# Función de redimensionamiento universal
def universal_resize(image, max_size=800):
    """Función 100% compatible con todas las versiones de Pillow"""
    try:
        # Versiones modernas (Pillow >= 9.1.0)
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    except AttributeError:
        try:
            # Versiones intermedias
            return image.resize((max_size, max_size), Image.LANCZOS)
        except AttributeError:
            # Versiones antiguas (fallback sin filtro)
            return image.resize((max_size, max_size))

@st.cache_resource
def load_reader():
    # Configuración ligera para español
    return easyocr.Reader(['es'], gpu=False)  # CPU mode for better compatibility

reader = load_reader()

# Interfaz de usuario mejorada
uploaded_file = st.file_uploader("Sube tu imagen aquí", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        with st.spinner("Procesando imagen..."):
            # Carga y procesamiento seguro
            img = Image.open(uploaded_file)
            
            # Redimensionamiento universal
            if max(img.size) > 800:
                img = universal_resize(img)
            
            # Conversión a array
            img_array = np.array(img)
            
            # Reconocimiento
            results = reader.readtext(img_array)
            
            # Resultados
            if results:
                st.success("✅ Texto reconocido con éxito!")
                for i, (_, text, prob) in enumerate(results, 1):
                    st.write(f"{i}. {text} (confianza: {prob*100:.1f}%)")
            else:
                st.warning("⚠️ No se encontró texto legible")

        st.image(img, caption="Imagen procesada", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.write("ℹ️ Detalles técnicos:", sys.exc_info()[0])

# Información de versión (para diagnóstico)
with st.expander("ℹ️ Información del sistema"):
    st.write(f"Versión de Pillow: {Image.__version__}")
    st.write(f"Versión de Python: {sys.version.split()[0]}")
