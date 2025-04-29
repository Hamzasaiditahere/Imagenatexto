import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Universal")
st.write("Sistema profesional - Versión Final")

# Función de redimensionamiento 100% compatible
def resize_image(img, max_size=800):
    """Versión completamente compatible con Pillow 10+"""
    try:
        return img.resize((max_size, max_size), resample=Image.Resampling.LANCZOS)
    except:
        return img.resize((max_size, max_size))

@st.cache_resource 
def load_reader():
    return easyocr.Reader(['es'], gpu=False)

reader = load_reader()

# Interfaz mejorada
uploaded_file = st.file_uploader("Sube una imagen con texto", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        with st.spinner("Analizando..."):
            # Procesamiento seguro
            img = Image.open(uploaded_file)
            
            # Redimensionamiento seguro
            if max(img.size) > 800:
                img = resize_image(img)
            
            # Conversión a array
            img_array = np.array(img.convert('RGB'))
            
            # Reconocimiento
            results = reader.readtext(img_array)
            
            # Resultados
            if results:
                st.success("✅ Resultados:")
                cols = st.columns(2)
                cols[0].image(img, width=200)
                with cols[1]:
                    for i, (_, text, prob) in enumerate(results, 1):
                        st.write(f"{i}. {text} ({prob*100:.1f}%)")
            else:
                st.warning("No se detectó texto")
                
    except Exception as e:
        st.error("Error al procesar")
        st.code(f"""
        Error: {str(e)}
        Versión Pillow: {Image.__version__}
        Tipo archivo: {uploaded_file.type}
        """)

# Consejos de uso
st.info("""
💡 **Consejos profesionales:**
1. Use imágenes nítidas con buen contraste
2. Texto negro sobre fondo blanco funciona mejor
3. Tamaño mínimo recomendado: 300x300 píxeles
""")
