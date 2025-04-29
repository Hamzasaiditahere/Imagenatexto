import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Universal")
st.write("Sistema profesional - Versión Estable")

# Función de redimensionamiento 100% compatible
def safe_resize(image, max_size=800):
    """Versión completamente robusta para todas las versiones de Pillow"""
    try:
        # Intenta el método moderno primero (Pillow >= 9.1.0)
        return image.resize((max_size, max_size), resample=Image.LANCZOS)
    except Exception:
        # Fallback para cualquier caso
        return image.resize((max_size, max_size))

@st.cache_resource 
def load_reader():
    return easyocr.Reader(['es'], gpu=False)  # Modo CPU para máxima compatibilidad

reader = load_reader()

# Interfaz de usuario mejorada
uploaded_file = st.file_uploader("Sube una imagen con texto claro", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        with st.spinner("Procesando imagen..."):
            # Carga segura de la imagen
            img = Image.open(uploaded_file)
            
            # Redimensionamiento seguro
            if max(img.size) > 800:
                img = safe_resize(img)
            
            # Conversión a array numpy
            img_array = np.array(img)
            
            # Reconocimiento de texto
            results = reader.readtext(img_array)
            
            # Mostrar resultados
            if results:
                st.success("✅ Texto reconocido con éxito!")
                for i, (_, text, prob) in enumerate(results, 1):
                    st.write(f"{i}. {text} (confianza: {prob*100:.1f}%)")
            else:
                st.warning("⚠️ No se detectó texto legible")
                
        # Mostrar imagen procesada
        st.image(img, caption="Imagen analizada", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        st.write("ℹ️ Para diagnóstico:", {
            "Versión Pillow": Image.__version__,
            "Versión Python": sys.version.split()[0],
            "Tipo de archivo": uploaded_file.type
        })

# Información adicional
st.markdown("""
**📌 Consejos para mejores resultados:**
- Texto negro sobre fondo blanco funciona mejor
- Letras deben tener al menos 50px de altura
- Evite imágenes borrosas o con mucho ruido
""")
