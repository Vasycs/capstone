# Sistema de Reconocimiento de Patentes mediante Edge Computing

Este proyecto está enfocado en el desarrollo de un sistema de Reconocimiento Automático de Matrículas (ALPR - Automatic License Plate Recognition).
El modelo está entrenado para identificar y validar los formatos de patentes vehiculares chilenas, diseñado para ser desplegado en entornos de Edge Computing (como NVIDIA Jetson o Raspberry Pi).

El objetivo principal es automatizar el control de acceso vehicular en recintos privados (ej. condominios o estacionamientos), interceptando flujos de video de cámaras de seguridad (CCTV) para procesar la información en tiempo real sin depender de servidores en la nube.

##-  Características Principales:
  
Detección en Tiempo Real: Utiliza modelos de la familia YOLO ajustados (fine-tuned) para localizar y recortar la ubicación exacta de las matrículas en distintas condiciones ambientales (luz de día, ruido nocturno, encandilamiento).

Extracción de Texto (OCR): Pipeline de preprocesamiento de imágenes basado en OpenCV y lectura de caracteres mediante redes neuronales (EasyOCR/PyTorch).

Validación de Formato Nacional: Motor de filtrado mediante expresiones regulares (Regex) para aceptar estrictamente las nomenclaturas chilenas históricas (AA·1000) y actuales (BB·CC-12), reduciendo falsos positivos.

Para ejecutar proyecto:

- Crear el entorno 
python -m venv venv

- Activar el entorno 
.\venv\Scripts\Activate.ps1

- O si usas la consola tradicional
venv\Scripts\activate.bat

- instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

- Soporte para GPU
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

Stack Tecnológico
Lenguaje: Python 3.10+

Visión por Computadora: OpenCV (cv2), imutils

Detección de Objetos: Ultralytics (YOLOv8)

Reconocimiento Óptico: EasyOCR

Backend de Inferencia: PyTorch (Soporte CUDA para aceleración GPU)
