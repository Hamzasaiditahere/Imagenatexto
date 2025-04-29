import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import sys

# Configuración inicial
st.title("🔠 Reconocimiento de Texto Universal")
st.write("Sistema profesional - Versión 10.0.0 Compatible")

# Función de redimensionamiento 100% compatible
def compatible_resize(image, max_size=800):
    """Versión completamente compatible con Pillow 10.0.0+"""
    try:
        # Método moderno (Pillow 10+)
        return image.resize((max_size, max_size), resample=Image.Resampling.LANCZOS)
    except AttributeError:
        # Fallback ultra seguro
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
            
            # Redimensionamiento compatible
            if max(img.size) > 800:
                img = compatible_resize(img)
            
            # Conversión a array numpy
            img_array = np.array(img.convert('RGB'))  # Conversión explícita a RGB
            
            # Reconocimiento de texto
            results = reader.readtext(img_array)
            
            # Mostrar resultados
            if results:
                st.success("✅ Texto reconocido con éxito!")
                for i, (_, text, prob) in enumerate(results, 1):
                    st.write(f"{i}. {text} (confianza: {prob*100:.2f}%)")
            else:
                st.warning("⚠️ No se detectó texto legible")
                
        # Mostrar imagen procesada
        st.image(img, caption="Imagen analizada", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        st.json({
            "Versión Pillow": Image.__version__,
            "Versión Python": sys.version.split()[0],
            "Tipo de archivo": uploaded_file.type,
            "Error": str(e),
            "Solución": "Use Image.Resampling.LANCZOS en lugar de ANTIALIAS"
        })

# Consejos profesionales
st.markdown("""
**📌 Mejores prácticas:**
- Texto negro sobre fondo blanco
- Tamaño mínimo de 50px para caracteres
- Imágenes nítidas sin compresión
- Formatos recomendados: PNG > JPEG
""")
