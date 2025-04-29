import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.set_page_config(page_title="OCR Profesional", layout="wide")
st.title("✍️ Reconocimiento de Texto Avanzado")

# Solución definitiva para ANTIALIAS
import PIL
if hasattr(PIL.Image, 'ANTIALIAS'):
    RESAMPLE = PIL.Image.ANTIALIAS
else:
    RESAMPLE = PIL.Image.Resampling.LANCZOS

@st.cache_resource
def load_reader():
    # Configuración optimizada para español
    return easyocr.Reader(['es'], 
                        gpu=False,
                        model_storage_directory='model',
                        download_enabled=True)

reader = load_reader()

def process_image(uploaded_file):
    """Procesamiento completo de imágenes"""
    try:
        img = Image.open(uploaded_file)
        
        # Redimensionamiento seguro
        if max(img.size) > 800:
            img = img.resize((800, 800), resample=RESAMPLE)
        
        # Conversión garantizada
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        return np.array(img), None
    except Exception as e:
        return None, str(e)

# Interfaz mejorada
uploaded_file = st.file_uploader("Sube tu imagen aquí", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(uploaded_file, use_column_width=True)
    
    with col2:
        st.subheader("Resultados")
        
        try:
            img_array, error = process_image(uploaded_file)
            if error:
                raise Exception(error)
                
            with st.spinner("Analizando texto..."):
                results = reader.readtext(img_array)
                
                if results:
                    st.success("✅ Texto reconocido:")
                    for i, (_, text, prob) in enumerate(results, 1):
                        st.metric(f"Opción {i}", 
                                 f"{text}", 
                                 f"{prob*100:.1f}% de confianza")
                else:
                    st.warning("No se encontró texto legible")
                    
        except Exception as e:
            st.error(f"Error en el procesamiento: {str(e)}")
            st.json({
                "Versión Pillow": PIL.__version__,
                "Versión Python": sys.version.split()[0],
                "Tipo de archivo": uploaded_file.type,
                "Solución aplicada": "Uso de LANCZOS en lugar de ANTIALIAS"
            })

# Panel informativo
with st.expander("ℹ️ Guía de uso avanzado"):
    st.markdown("""
    **📌 Para mejores resultados:**
    - Texto negro sobre fondo blanco
    - Resolución mínima: 300dpi
    - Formatos recomendados: PNG > JPEG > WEBP
    
    **⚙️ Configuración técnica:**
    - Pillow v10.0.0
    - EasyOCR v1.7.0
    - Procesamiento por CPU
    """)
