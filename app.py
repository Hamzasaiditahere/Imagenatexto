import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import easyocr

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Profesional")
st.write("Sistema mejorado con EasyOCR")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['es'])  # Solo español para mejor rendimiento

reader = load_reader()

# Interfaz mejorada
uploaded_file = st.file_uploader("Sube imagen con texto", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        # Procesamiento mejorado
        img = Image.open(uploaded_file)
        
        # Redimensionamiento CORREGIDO (sin ANTIALIAS)
        if img.size[0] > 800 or img.size[1] > 800:
            img = img.resize((800, 800), Image.Resampling.LANCZOS)  # ¡Corrección aquí!
        
        st.image(img, caption="Imagen procesada", use_column_width=True)
        
        # Reconocimiento
        results = reader.readtext(np.array(img))
        
        # Resultados organizados
        if results:
            st.subheader("📝 Texto reconocido:")
            for i, (bbox, text, prob) in enumerate(results, 1):
                st.success(f"{i}. {text} (Confianza: {prob*100:.1f}%)")
        else:
            st.warning("No se detectó texto")
            
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")

# Guía de uso (formato corregido)
st.markdown("""
**📌 Consejos para mejores resultados:**
1. Imágenes nítidas con texto claro
2. Fondo contrastante (oscuro para texto claro o viceversa)
3. Tamaño mínimo de 300x300 píxeles
""")
