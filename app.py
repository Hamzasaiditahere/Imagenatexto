import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

# Cargar el procesador y el modelo TrOCR
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
    
    # Verificar si hay GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return processor, model, device

processor, model, device = load_model()

# Función para aplicar estilo
def apply_style():
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f0f0;
            color: #333333;
        }
        
        h1, h2, h3 {
            color: #1f1f1f;
        }

        .stButton>button {
            background-color: #1f1f1f;
            color: #ffffff;
            border: 1px solid #1f1f1f;
            border-radius: 5px;
            padding: 10px 24px;
        }
        
        .stButton>button:hover {
            background-color: #ffffff;
            color: #1f1f1f;
        }
        
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #1f1f1f;
        }
    </style>
    """, unsafe_allow_html=True)

# Función para mostrar el texto detectado
def display_detected_text(text):
    st.markdown(f"""
    <div style="background-color: #e0e0e0; padding: 15px; border-radius: 10px;">
        <h3>Texto Detectado:</h3>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)

# Configuración de la página
st.set_page_config(page_title="ImageOCR", page_icon="📷", layout="centered")

# Aplicar estilo
apply_style()

# Contenedor principal
st.title("Reconocimiento de Texto de una Sola Letra, Número o Símbolo")
st.markdown("Sube una imagen con una letra, número o símbolo y lo reconoceremos automáticamente.")

# Área de carga de archivo
uploaded_file = st.file_uploader("Sube una imagen (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Cargar la imagen
    image = Image.open(uploaded_file).convert("RGB")
    
    # Mostrar la imagen cargada
    st.image(image, caption="Imagen Cargada", use_container_width=True)
    
    # Botón para procesar la imagen
    if st.button("🔍 Reconocer Texto"):
        try:
            # Preprocesar la imagen para el modelo TrOCR
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Generar la predicción del texto
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Filtrar solo letras, números y símbolos
            filtered_text = re.sub(r'[^a-zA-Z0-9!@#$%^&*()_+=-]', '', generated_text)

            # Mostrar el texto filtrado
            display_detected_text(filtered_text)
        
        except Exception as e:
            st.error(f"❌ Error en el OCR: {e}")
