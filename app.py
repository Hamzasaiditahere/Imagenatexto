import streamlit as st

# ✅ Esta línea debe ir lo primero
st.set_page_config(
    page_title="ImageOCR",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Cargar modelo TrOCR
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Estilo CSS
def apply_custom_style():
    st.markdown("""
    <style>
        .stApp {
            background-color: #000;
            color: white;
        }
        h1, h2, h3 {
            color: #00ffff;
            font-family: 'Arial';
        }
        .stButton>button {
            background-color: black;
            color: #00ffff;
            border: 1px solid #00ffff;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #00ffff;
            color: black;
            box-shadow: 0 0 10px #00ffff;
        }
        .detected-text {
            background-color: #111;
            border-left: 4px solid #00ffff;
            padding: 10px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

# Mostrar título y subtítulo
def show_header():
    st.markdown("""
    <div style="text-align:center;">
        <h1>Image<span style="color:#00ffff;">OCR</span></h1>
        <p>Reconocimiento Óptico de Caracteres</p>
    </div>
    """, unsafe_allow_html=True)

# Mostrar resultados
def show_detected_text(text):
    st.markdown(f"""
    <div class="detected-text">
        <h3>📄 Texto Detectado:</h3>
        <p style="white-space: pre-wrap;">{text}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
def show_footer():
    st.markdown("""
    <div class="footer">
        © 2025 Imagenatexto | Hecho con ❤️ y TrOCR
    </div>
    """, unsafe_allow_html=True)

# Aplicar estilo y mostrar encabezado
apply_custom_style()
show_header()

# Subida de archivo
uploaded_file = st.file_uploader("📤 Sube una imagen (JPG, JPEG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    st.image(image, caption="📷 Imagen cargada", use_column_width=True)

    if st.button("🔍 Reconocer Texto"):
        with st.spinner("Procesando imagen..."):
            try:
                inputs = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                show_detected_text(generated_text)
            except Exception as e:
                st.error(f"❌ Error en el OCR: {str(e)}")

show_footer()
