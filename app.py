import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Configuración de la página
st.set_page_config(
    page_title="ImageOCR",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cargar el procesador y el modelo TrOCR
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Verificar si hay GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return processor, model, device

processor, model, device = load_model()

# Función para aplicar estilo básico
def apply_basic_style():
    st.markdown("""
    <style>
        /* Estilo básico y minimalista */
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        
        h1, h2, h3 {
            color: #00ffff !important;
            font-family: 'Arial', sans-serif;
            font-weight: 300;
            letter-spacing: 2px;
        }
        
        .stButton>button {
            background-color: #000000;
            color: #00ffff;
            border: 1px solid #00ffff;
            border-radius: 5px;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #00ffff;
            color: #000000;
            box-shadow: 0 0 15px #00ffff;
        }
        
        .stTextInput>div>div>input {
            background-color: #111111;
            color: #ffffff;
            border: 1px solid #00ffff;
        }
        
        .stMarkdown {
            color: #ffffff;
        }
        
        /* Estilo para el área de carga de archivos */
        .css-1cpxqw2 {
            background-color: #111111;
            border: 1px dashed #00ffff;
            border-radius: 5px;
        }
        
        /* Estilo para el texto detectado */
        .detected-text {
            background-color: #111111;
            border-left: 3px solid #00ffff;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin-top: 20px;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
        }
        
        /* Footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #111111;
            color: #888888;
            text-align: center;
            padding: 10px;
            font-size: 12px;
        }
        
        /* Contenedor principal con margen inferior para el footer */
        .main-content {
            margin-bottom: 50px;
        }
    </style>
    """, unsafe_allow_html=True)

# Función para mostrar el logo
def display_logo():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3em; margin-bottom: 0;">Image<span style="color: #00ffff;">OCR</span></h1>
        <p style="color: #888888; margin-top: 0;">Reconocimiento Óptico de Caracteres</p>
    </div>
    """, unsafe_allow_html=True)

# Función para crear un área de carga personalizada
def custom_file_uploader():
    st.markdown("""
    <div style="
        border: 1px dashed #00ffff;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        background-color: rgba(0, 255, 255, 0.05);
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    ">
        <div style="font-size: 40px; margin-bottom: 10px; color: #00ffff;">
            📤
        </div>
        <p style="color: #ffffff; margin-bottom: 5px;">Arrastra o haz clic para subir una imagen</p>
        <p style="color: #888888; font-size: 0.8em;">Formatos soportados: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Función para mostrar el texto detectado con estilo
def display_detected_text(text):
    st.markdown(f"""
    <div class="detected-text">
        <h3 style="color: #00ffff; margin-top: 0;">📄 Texto Detectado:</h3>
        <p style="white-space: pre-wrap;">{text}</p>
    </div>
    """, unsafe_allow_html=True)

# Función para mostrar la animación de procesamiento
def show_processing_animation():
    st.markdown("""
    <div class="processing">
        <p style="color: #00ffff;">Procesando imagen...</p>
    </div>
    """, unsafe_allow_html=True)

# Función para mostrar el footer
def display_footer():
    st.markdown("""
    <div class="footer">
        <p>© 2023 ImageOCR | Desarrollado con TrOCR y Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Convertir la imagen a escala de grises y luego convertirla a formato RGB
def convert_to_grayscale(image):
    grayscale_image = image.convert("L")  # Convertir a escala de grises
    return grayscale_image.convert("RGB")  # Convertir a RGB (3 canales)

# Aplicar estilo básico
apply_basic_style()

# Contenedor principal
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Mostrar logo
display_logo()

# Descripción de la aplicación
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <p>Sube una imagen con texto manuscrito o impreso y nuestro sistema de IA lo reconocerá automáticamente.</p>
</div>
""", unsafe_allow_html=True)

# Área personalizada para subir archivos (visual)
custom_file_uploader()

# El verdadero cargador de archivos (funcional)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # Cargar la imagen
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convertir la imagen a escala de grises y luego a RGB
    image = convert_to_grayscale(image)
    
    # Mostrar la imagen con estilo
    st.markdown("""
    <h3 style="color: #00ffff; margin-top: 30px;">📷 Imagen Cargada:</h3>
    """, unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    
    # Botón para procesar la imagen
    if st.button("🔍 Reconocer Texto", key="process_button"):
        # Mostrar animación de procesamiento
        with st.spinner("Procesando..."):
            show_processing_animation()
            
            try:
                # Preprocesar la imagen para el modelo TrOCR
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                # Generar la predicción del texto
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Mostrar el texto detectado con estilo
                display_detected_text(generated_text)
                
            except Exception as e:
                st.error(f"❌ Error en el OCR: {e}")

# Cerrar el contenedor principal
st.markdown('</div>', unsafe_allow_html=True)

# Mostrar el footer
display_footer()
