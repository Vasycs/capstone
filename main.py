import cv2
import easyocr
from ultralytics import YOLO

print("Cargando modelo YOLOv8 y EasyOCR...")
model = YOLO('models/best.pt') 
reader = easyocr.Reader(['es', 'en'])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se encontró ninguna cámara conectada.")
    exit()

print("Procesando video optimizado... Presiona 'q' para salir.")

# Variables para optimizar el rendimiento
frame_count = 0
procesar_cada_x_frames = 5  # La IA actuará 1 de cada 5 fotogramas
ultimo_resultado = []       # Guardará temporalmente los rectángulos para no perderlos visualmente

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Solo ejecutamos la IA pesada cada 5 fotogramas
    if frame_count % procesar_cada_x_frames == 0:
        ultimo_resultado = [] # Limpiamos la memoria del fotograma anterior
        
        # verbose=False, stream=True y half=True (si usas GPU) optimizan la inferencia
        resultados = model(frame, stream=True, verbose=False)

        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confianza = box.conf[0]

                if confianza > 0.5:
                    recorte = frame[y1:y2, x1:x2]
                    texto_detectado = ""
                    
                    # Ejecutar OCR solo si el recorte tiene un tamaño razonable
                    if recorte.size > 0:
                        textos = reader.readtext(recorte)
                        for (bbox_ocr, txt, prob) in textos:
                            texto_detectado = f"{txt} ({prob:.2f})"
                    
                    # Guardar el cálculo para dibujarlo rápido en los fotogramas "vacíos"
                    ultimo_resultado.append((x1, y1, x2, y2, texto_detectado))

    # Dibujar los últimos datos calculados en TODOS los fotogramas para mantener la ilusión visual
    for (x1, y1, x2, y2, texto_detectado) in ultimo_resultado:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if texto_detectado:
            cv2.putText(frame, texto_detectado, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Deteccion ALPR - Optimizado', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()