import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Asegúrate que esto coincida EXACTAMENTE con tu entrenamiento
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{};:,.<>?/"  # 62 caracteres

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="ocr_model_compatible.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()
input_details = model.get_input_details()
output_details = model.get_output_details()

st.title("🔠 OCR Corregido")

def predict_image(img_array):
    """Función segura para predicciones"""
    interpreter = model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

uploaded_file = st.file_uploader("Sube una imagen CLARA de un carácter", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        # Preprocesamiento MEJORADO
        img = Image.open(uploaded_file).convert('L')
        img = img.point(lambda x: 0 if x < 150 else 255, '1')  # Binarización
        img = np.array(img.resize((32, 32))) / 255.0
        img = np.expand_dims(img, axis=(0, -1)).astype(np.float32)
        
        st.image(img[0,:,:,0], width=150, caption="Imagen procesada")

        # Predicción
        preds = predict_image(img)
        
        # Resultados FILTRADOS (solo clases válidas)
        valid_preds = [p for i, p in enumerate(preds[0]) if i < len(CHARS)]
        top5_idx = np.argsort(valid_preds)[-5:][::-1]
        
        st.subheader("Resultados válidos:")
        if len(valid_preds) == 0:
            st.error("El modelo no está produciendo clases válidas")
        else:
            for i, idx in enumerate(top5_idx):
                st.write(f"{i+1}. {CHARS[idx]} ({valid_preds[idx]*100:.1f}%)")
        
        # Diagnóstico
        with st.expander("🔍 Ver diagnóstico técnico"):
            test_input = np.zeros((1,32,32,1), dtype=np.float32)
            test_pred = predict_image(test_input)
            st.write("Predicción para imagen negra:", np.argmax(test_pred))
            st.write("Máxima clase predicha:", np.argmax(preds))
            st.write("Total de clases en modelo:", output_details[0]['shape'][1])
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
