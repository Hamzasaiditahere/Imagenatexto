import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Universal")
st.write("Sistema profesional - Versión Estable")

# Función de redimensionamiento CORREGIDA
def universal_resize(image, max_size=800):
    """Función completamente compatible con todas versiones de Pillow"""
    try:
        # Para Pillow >= 9.1.0
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    except AttributeError:
        try:
            # Para versiones antiguas
            return image.resize((max_size, max_size), Image.LANCZOS)
        except AttributeError:
            # Fallback básico
            return image.resize((max_size, max_size))

@st.cache_resource 
def load_reader():
    return easyocr.Reader(['es'], gpu=False)  # Modo CPU para mejor compatibilidad

reader = load_reader()

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen con texto claro", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        with st.spinner("Analizando imagen..."):
            # Procesamiento seguro
            img = Image.open(uploaded_file)
            
            # Redimensionamiento CORREGIDO
            if max(img.size) > 800:
                img = universal_resize(img)
            
            # Conversión a array
            img_array = np.array(img)
            
            # Reconocimiento
            results = reader.readtext(img_array)
            
            # Mostrar resultados
            if results:
                st.success("✅ Texto reconocido:")
                for i, (_, text, prob) in enumerate(results, 1):
                    st.write(f"{i}. {text} (confianza: {prob*100:.1f}%)")
            else:
                st.warning("⚠️ No se detectó texto")
                
        st.image(img, caption="Imagen procesada", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        st.write("ℹ️ Detalles técnicos:", sys.exc_info()[0])

# Información del sistema
with st.expander("ℹ️ Versiones instaladas"):
    st.write(f"Pillow v{Image.__version__}")
    st.write(f"Python v{sys.version.split()[0]}")
    st.write(f"EasyOCR v{easyocr.__version__}")
