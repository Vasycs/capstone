import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# 1. INICIALIZACIÓN DE MODELOS
print("Cargando modelo YOLO...")
# Nota: Aquí usamos el modelo nano genérico para probar. 
# reemplazarlo por  modelo entrenado, ej: 'models/patentes_best.pt'
model_yolo = YOLO('yolov8n.pt') 

print("Inicializando EasyOCR...")
# Inicializamos el lector en inglés
# Si GPU compatible (Nvidia), easyocr la usará automáticamente.
reader = easyocr.Reader(['en'], gpu=True)

# 2. CONFIGURACIÓN DE LA FUENTE DE VIDEO
# Opciones:
# video_source = 0  # para usar tu cámara web (USB)
# video_source = "rtsp://usuario:clave@192.168.1.10:554/stream" # Para cámaras IP en condominios
video_source = 0 # Ruta de un video local

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la fuente de video {video_source}")
    exit()

print("Presiona la tecla 'q' en la ventana de video para salir.")

# 3. BUCLE DE PROCESAMIENTO EN TIEMPO REAL
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o se perdió la conexión del stream.")
        break

    # Reducimos un poco el tamaño del frame para procesar más rápido
    frame = cv2.resize(frame, (1024, 768))

    # Ejecutar YOLO en el frame actual (conf=0.5 filtra detecciones con menos del 50% de seguridad)
    resultados = model_yolo(frame, conf=0.5, verbose=False)

    # Iterar sobre las detecciones de este frame
    for r in resultados:
        cajas = r.boxes
        for caja in cajas:
            # Obtener coordenadas del rectángulo (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            
            # Recortar la patente del frame original (Region of Interest - ROI)
            # Aseguramos de no salirnos de los márgenes de la imagen
            h, w = frame.shape[:2]
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            recorte_patente = frame[y1:y2, x1:x2]

            # Validar que el recorte no esté vacío antes de enviarlo al OCR
            if recorte_patente.size > 0:
                # Opcional: Convertir a escala de grises para ayudar al OCR
                recorte_gris = cv2.cvtColor(recorte_patente, cv2.COLOR_BGR2GRAY)

                # Pasar el recorte por EasyOCR
                # detail=0 devuelve solo el texto, detail=1 devuelve cajas y precisión
                lecturas_ocr = reader.readtext(recorte_gris, detail=0)

                # Si el OCR detectó algún texto
                if lecturas_ocr:
                    # Unir las lecturas si la patente está en varias líneas
                    texto_detectado = " ".join(lecturas_ocr).upper()
                    
                    # Dibujar Bounding Box (Rectángulo verde)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Colocar el texto leído sobre el rectángulo
                    cv2.putText(frame, texto_detectado, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Imprimir en consola lo que va leyendo (útil para debug)
                    print(f"Patente detectada: {texto_detectado}")

    # 4. MOSTRAR EL RESULTADO
    cv2.imshow("Sistema ALPR - Tesis", frame)

    # Condición de salida: presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. LIMPIEZA DE RECURSOS
cap.release()
cv2.destroyAllWindows()
