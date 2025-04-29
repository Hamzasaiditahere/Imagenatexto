!pip install tensorflow==2.12.0  # Versión específica para compatibilidad
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Configuración
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_SIZE = 48

# Generar dataset (código anterior)
# ...

# Modelo con compatibilidad garantizada
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(CHARS), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Conversión SUPER compatible
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_TFLITE_BUILTINS]  # Clave
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Verificar tamaño
print(f"Tamaño del modelo: {len(tflite_model)/1024:.2f} KB")

# Descargar
with open('ocr_ultra_compatible.tflite', 'wb') as f:
    f.write(tflite_model)
from google.colab import files
files.download('ocr_ultra_compatible.tflite')
