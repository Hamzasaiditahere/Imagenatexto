import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

# Configuración de la página (debe ir justo aquí)
st.set_page_config(
    page_title="Imagenatexto",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cargar modelo y procesador
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Estilo visual
def apply_style():
    st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #00ffff !important;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #000000;
            color: #00ffff;
            border: 1px solid #00ffff;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #00ffff;
            color: #000000;
        }
        .detected-text {
            background-color: #111111;
            border-left: 3px solid #00ffff;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }
    </style>
    """, unsafe_allow_html=True)

apply_style()

# Interfaz
st.title("📷 Imagenatexto")
st.markdown("Reconocimiento Óptico de Caracteres simple.\n\nSube una imagen con una **letra, número o símbolo**.")

uploaded_file = st.file_uploader("Sube una imagen (JPG, JPEG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    if st.button("🔍 Reconocer Texto"):
        with st.spinner("Procesando imagen..."):
            try:
                # Preprocesar imagen
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Filtrar: solo letras, números y símbolos comunes (1 solo carácter)
                clean_text = re.findall(r"[a-zA-Z0-9.,!?@#%^&*()\-+=]", raw_text)
                result = clean_text[0] if clean_text else "❌ No se detectó texto válido."

                st.markdown(f"<div class='detected-text'><h3>📄 Resultado:</h3><p>{result}</p></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error al procesar la imagen: {e}")
