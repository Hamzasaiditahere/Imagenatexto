import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# ⚙️ Configuración de la página (debe ir al principio)
st.set_page_config(
    page_title="Imagen a Texto - Letras",
    page_icon="🔠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 🎨 Estilo futurista
def apply_futuristic_style():
    st.markdown("""
    <style>
        .stApp {
            background-color: #000;
            color: #fff;
        }
        h1, h2, h3 {
            color: #00ffff;
        }
        .stButton>button {
            background-color: #000;
            color: #00ffff;
            border: 1px solid #00ffff;
            border-radius: 5px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #00ffff;
            color: #000;
            box-shadow: 0 0 10px #00ffff;
        }
        .detected-text {
            background-color: #111;
            border-left: 3px solid #00ffff;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# 🧠 Cargar modelo
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()
apply_futuristic_style()

# 🧾 Título
st.markdown("<h1 style='text-align: center;'>🔠 Imagen a Letra</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sube una imagen con una sola letra, número o símbolo.</p>", unsafe_allow_html=True)

# 📤 Cargar imagen
uploaded_file = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagen subida", use_column_width=True)

    if st.button("🔍 Detectar carácter"):
        with st.spinner("Procesando..."):
            try:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values, max_length=2, num_beams=4)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                st.markdown(f"<div class='detected-text'><h3>Resultado:</h3><p>{generated_text}</p></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error en el OCR: {str(e)}")
