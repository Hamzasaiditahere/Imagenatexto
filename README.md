# 🖼️ **Imagenatexto** - Reconocimiento Óptico de Caracteres Futurista

Bienvenido a **Imagenatexto**, una **aplicación web futurista** de reconocimiento óptico de caracteres (OCR) desarrollada con **Streamlit** y el modelo **TrOCR** de **Microsoft**. Esta herramienta te permite **extraer texto de imágenes** de forma sencilla y moderna, utilizando inteligencia artificial de vanguardia.

## 🌟 Características

- **Interfaz Futurista y Minimalista**: Un diseño oscuro con **efectos neón** que proporcionan una experiencia visual moderna.
- **Reconocimiento de Texto**: Utiliza el modelo **TrOCR** para realizar el reconocimiento de caracteres en imágenes.
- **Soporte de Archivos**: Suba imágenes en formatos **JPG, JPEG** y **PNG** para obtener el texto detectado.
- **Rápido y Eficiente**: El sistema de IA procesa las imágenes rápidamente, permitiendo una experiencia fluida.

## ⚙️ Requisitos

Asegúrate de tener las siguientes bibliotecas instaladas:

- `streamlit`
- `transformers`
- `torch`
- `Pillow`

Estas bibliotecas se instalarán automáticamente al crear el entorno, gracias al archivo `requirements.txt`.

## 📸 ¿Cómo usar la aplicación?

1. **Sube una Imagen**:  
   Arrastra o selecciona una imagen desde tu dispositivo (JPG, JPEG o PNG). La imagen debe contener texto **manuscrito o impreso** para que el modelo lo reconozca.
   
2. **Reconocer el Texto**:  
   Haz clic en el botón **"🔍 Reconocer Texto"** para que el sistema procese la imagen y extraiga el texto.

3. **Resultado**:  
   El texto extraído se mostrará en la pantalla dentro de una **caja estilizada**.

## 🛠️ Instalación

### Clona este repositorio:

```bash
git clone https://github.com/Hamzasaiditahere/Imagenatexto.git
Crea un entorno virtual e instala las dependencias:
bash
Copiar
Editar
cd Imagenatexto
pip install -r requirements.txt
Ejecuta la aplicación:
bash
Copiar
Editar
streamlit run streamlit_app.py
Accede a la aplicación en tu navegador en http://localhost:8501.

🌱 Contribuciones
Si deseas contribuir al proyecto, sigue estos pasos:

Haz un fork del repositorio.

Crea una rama para tu característica:
git checkout -b feature/mi-nueva-funcionalidad

Haz tus cambios y comitea:
git commit -m 'Añadí nueva funcionalidad'

Haz push a tu rama:
git push origin feature/mi-nueva-funcionalidad

Abre un pull request.

📝 Licencia
Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

💬 Contacto
Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue en el repositorio o enviarme un mensaje a @Hamzasaiditahere.
