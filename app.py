import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import sys

# Configuración
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_SIZE = 48

@st.cache_resource
def load_model():
    try:
        # Cargar modelo con verificación de compatibilidad
        print("Cargando el modelo...")
        interpreter = tf.lite.Interpreter(model_path="ocr_optimized.tflite")
        
        # Configuración explícita para máxima compatibilidad
        interpreter.allocate_tensors()
        
        # Prueba de funcionamiento
        input_details = interpreter.get_input_details()
        test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        return interpreter
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.write("""
        Solución requerida:
        1. Regenera el modelo con el código de Colab proporcionado
        2. Usa tensorflow==2.12.0 en Colab
        3. Asegúrate de subir el nuevo modelo 'ocr_ultra_compatible.tflite'
        """)
        st.stop()

def predict(image):
    try:
        # Convertir la imagen a escala de grises y redimensionarla a 48x48
        print("Convirtiendo la imagen a escala de grises y redimensionándola...")
        img = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
        
        # Normalización: valores de píxeles entre 0 y 1
        print("Normalizando los valores de píxeles...")
        img_array = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
        
        # Cargar el modelo
        print("Cargando el modelo...")
        interpreter = load_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Ejecutar la predicción
        print("Ejecutando la predicción...")
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Obtener la salida del modelo
        print("Obteniendo la salida del modelo...")
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Mostrar las probabilidades para cada carácter
        print("Probabilidades para cada carácter:", output)
        st.write("Probabilidades para cada carácter:", output)
        
        # Obtener el índice del valor máximo en la salida (la letra más probable)
        predicted_index = np.argmax(output)
        predicted_char = CHARS[predicted_index]
        
        # Mostrar el carácter predicho
        print(f"El carácter predicho es: {predicted_char}")
        st.write(f"El carácter predicho es: {predicted_char}")
        
        return predicted_char
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        st.error(f"Error en predicción: {str(e)}")
        return "Error"


# Interfaz
st.title("🔠 OCR Ultra Compatible")
uploaded_file = st.file_uploader("Sube imagen de un carácter")

if uploaded_file is not None:
    try:
        # Cargar la imagen y predecir
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Realizar predicción
        predicted_char = predict(image)
        
        # Mostrar el resultado
        st.write(f"El carácter predicho es: {predicted_char}")
    except Exception as e:
        st.error(f"Error al cargar o procesar la imagen: {str(e)}")
