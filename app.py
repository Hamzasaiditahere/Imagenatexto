import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import easyocr

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Profesional")
st.write("Sistema mejorado con EasyOCR - Versión Estable")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['es'])  # Configuración optimizada para español

reader = load_reader()

def safe_resize(image, max_size=800):
    """Función compatible con todas las versiones de Pillow"""
    try:
        # Para Pillow >= 10.0.0
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    except:
        try:
            # Para Pillow < 10.0.0
            return image.resize((max_size, max_size), Image.LANCZOS)
        except:
            # Fallback final (sin filtro de resampling)
            return image.resize((max_size, max_size))

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen con texto claro", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Procesamiento mejorado
        img = Image.open(uploaded_file)
        
        # Redimensionamiento seguro
        if img.size[0] > 800 or img.size[1] > 800:
            img = safe_resize(img)
        
        st.image(img, caption="Imagen optimizada", use_column_width=True)
        
        # Conversión a array numpy
        img_array = np.array(img)
        
        # Reconocimiento de texto
        results = reader.readtext(img_array)
        
        # Mostrar resultados
        if results:
            st.subheader("✅ Texto reconocido:")
            for i, (bbox, text, prob) in enumerate(results, 1):
                st.success(f"{i}. {text} (Precisión: {prob*100:.1f}%)")
        else:
            st.warning("⚠️ No se detectó texto en la imagen")
            
    except Exception as e:
        st.error(f"🚨 Error en el procesamiento: {str(e)}")

# Guía de uso optimizada
st.markdown("""
**📌 Consejos profesionales:**
1. Texto negro sobre fondo blanco funciona mejor
2. Resolución mínima recomendada: 300x300 píxels
3. Evite ángulos inclinados (>15°)
""")
