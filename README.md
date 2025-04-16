# Imagenatexto
Modelo OCR desplegado en streamlit.io
Imagenatexto - Reconocimiento Óptico de Caracteres Futurista
Imagenatexto es una aplicación web futurista de reconocimiento óptico de caracteres (OCR) que utiliza modelos de inteligencia artificial para extraer texto de imágenes. Desarrollada con Streamlit y el modelo TrOCR de Microsoft, esta aplicación es capaz de reconocer texto manuscrito o impreso en imágenes y presentarlo de manera sencilla y moderna.

Características
Interfaz Futurista y Minimalista: Diseño oscuro con toques neón que proporcionan una experiencia visual moderna.

Reconocimiento de Texto: Utiliza el modelo TrOCR para realizar el reconocimiento de caracteres en imágenes.

Soporte de Archivos: Puedes subir imágenes en formatos JPG, JPEG y PNG para obtener el texto detectado.

Rápido y Eficiente: El sistema de IA realiza el procesamiento de manera rápida gracias a la optimización del modelo.

Requisitos
Asegúrate de tener las siguientes bibliotecas instaladas:

streamlit

transformers

torch

Pillow

Estas bibliotecas se instalarán automáticamente al crear el entorno, ya que se especifican en el archivo requirements.txt.

Cómo usar la aplicación
Sube una Imagen:

Arrastra o selecciona una imagen desde tu dispositivo (JPG, JPEG o PNG).

La imagen debe contener texto manuscrito o impreso para que el modelo lo reconozca.

Reconocer el Texto:

Haz clic en el botón "🔍 Reconocer Texto".

El sistema procesará la imagen y mostrará el texto detectado.

Resultado:

El texto extraído se mostrará en la pantalla en una caja de texto estilizada.

Instalación
Clona este repositorio:

bash
Copiar
Editar
git clone https://github.com/tu-usuario/imagenatexto.git
Crea un entorno virtual e instala las dependencias:

bash
Copiar
Editar
cd imagenatexto
pip install -r requirements.txt
Ejecuta la aplicación:

bash
Copiar
Editar
streamlit run streamlit_app.py
Abre tu navegador y ve a http://localhost:8501 para interactuar con la aplicación.

Contribuciones
Si deseas contribuir al proyecto, sigue estos pasos:

Haz un fork del repositorio.

Crea una rama para tu característica (git checkout -b feature/mi-nueva-funcionalidad).

Haz tus cambios y comitea (git commit -m 'Añadí nueva funcionalidad').

Haz push a tu rama (git push origin feature/mi-nueva-funcionalidad).

Abre un pull request.

Licencia
Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.
