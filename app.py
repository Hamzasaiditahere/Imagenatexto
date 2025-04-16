import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Cargar el procesador y el modelo TrOCR
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Configuración de la página (DEBE ir al principio)
st.set_page_config(
    page_title="ImageOCR",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo visual
def apply_style():
    st.markdown("""
    <style>
        .stApp { background-color: #000000; color: #ffffff; }
        h1, h2, h3 { color: #00ffff; font-family: 'Arial'; letter-spacing: 1px; }
        .stButton>button {
            background-color: #000; color: #00ffff; border: 1px solid #00ffff;
            padding: 10px 24px; transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #00ffff; color: #000; box-shadow: 0 0 15px #00ffff;
        }
        .detected-text {
            background-color: #111111; border-left: 3px solid #00ffff;
            padding: 10px; border-radius: 5px; margin-top: 20px;
        }
        .processing {
            animation: glow 1.5s infinite;
            padding: 20px; border-radius: 5px; text-align: center;
        }
        @keyframes glow {
            0% { box-shadow: 0 0 5px #00ffff; }
            50% { box-shadow: 0 0 20px #00ffff; }
            100% { box-shadow: 0 0 5px #00ffff; }
        }
        .footer {
            position: fixed; bottom: 0; left: 0; width: 100%;
            background-color: #111111; color: #888888;
            text-align: center; padding: 10px; font-size: 12px;
        }
        .main-content { margin-bottom: 60px; }
    </style>
    """, unsafe_allow_html=True)

apply_style()

# Contenedor principal
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Logo y título
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 3em;">Image<span style="color: #00ffff;">OCR</span></h1>
    <p style="color: #888888;">Reconocimiento Óptico de Caracteres</p>
</div>
""", unsafe_allow_html=True)

# Descripción
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <p>Sube una imagen con texto manuscrito o impreso y nuestro sistema de IA lo reconocerá automáticamente.</p>
</div>
""", unsafe_allow_html=True)

# Uploader visual
st.markdown("""
<div style="
    border: 1px dashed #00ffff;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    background-color: rgba(0, 255, 255, 0.05);
    margin-bottom: 10px;
">
    <div style="font-size: 40px; color: #00ffff;">📤</div>
    <p style="margin: 0;">Arrastra o haz clic para subir una imagen</p>
    <p style="color: #888888; font-size: 0.8em;">Formatos soportados: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    st.markdown("""<h3 style="color: #00ffff; margin-top: 30px;">📷 Imagen Cargada:</h3>""", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    if st.button("🔍 Reconocer Texto"):
        with st.spinner("Procesando..."):
            st.markdown("""<div class="processing"><p style="color: #00ffff;">Procesando imagen...</p></div>""", unsafe_allow_html=True)
            try:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                st.markdown(f"""
                <div class="detected-text">
                    <h3 style="color: #00ffff;">📄 Texto Detectado:</h3>
                    <p style="white-space: pre-wrap;">{generated_text}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error al procesar la imagen: {e}")

# Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>© 2025 ImageOCR | Desarrollado con TrOCR y Streamlit</p>
</div>
""", unsafe_allow_html=True)
