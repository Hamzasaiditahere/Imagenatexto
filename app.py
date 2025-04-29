import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Universal")
st.write("Sistema profesional - Versión Estable")

# Función de redimensionamiento CORREGIDA
def safe_resize(image, max_size=800):
    """Versión 100% compatible con Pillow 10.0.0"""
    try:
        # Método moderno (Pillow 10+)
        return image.resize((max_size, max_size), resample=Image.LANCZOS)
    except Exception:
        # Fallback seguro
        return image.resize((max_size, max_size))

@st.cache_resource 
def load_reader():
    return easyocr.Reader(['es'], gpu=False)

reader = load_reader()

# Interfaz mejorada
uploaded_file = st.file_uploader("Sube una imagen con texto claro", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        with st.spinner("Analizando imagen..."):
            # Procesamiento seguro
            img = Image.open(uploaded_file)
            
            # Redimensionamiento seguro
            if max(img.size) > 800:
                img = safe_resize(img)
            
            # Conversión a array
            img_array = np.array(img)
            
            # Reconocimiento
            results = reader.readtext(img_array)
            
            # Resultados
            if results:
                st.success("✅ Texto reconocido:")
                for i, (_, text, prob) in enumerate(results, 1):
                    st.write(f"{i}. {text} (confianza: {prob*100:.1f}%)")
            else:
                st.warning("⚠️ No se detectó texto")
                
        st.image(img, caption="Imagen procesada", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        st.json({
            "Versión Pillow": Image.__version__,
            "Versión Python": sys.version.split()[0],
            "Tipo de archivo": uploaded_file.type,
            "Error": str(e)
        })

# Consejos optimizados
st.markdown("""
**📌 Recomendaciones profesionales:**
- Texto negro sobre fondo blanco
- Resolución mínima: 300x300 píxeles
- Fuentes claras sin decoraciones
""")
