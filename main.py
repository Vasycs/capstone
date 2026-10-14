import cv2
import easyocr
import re
import os
from ultralytics import YOLO

# Importamos nuestra función desde la carpeta src
from src.database import cargar_residentes

def limpiar_y_validar_patente(texto_crudo):
    texto_limpio = re.sub(r'[^A-Za-z0-9]', '', texto_crudo).upper()
    
    # Si tiene exactamente 6 caracteres, aplicamos corrección inteligente para formato actual (4 letras, 2 números)
    if len(texto_limpio) == 6:
        letras = texto_limpio[:4]
        numeros = texto_limpio[4:]
        
        # Si en la zona de letras leyó un número por error de ángulo/luz, lo pasamos a su letra similar
        letras = letras.replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
        # Si en la zona de números leyó una letra por error, la pasamos a número
        numeros = numeros.replace('O', '0').replace('Q', '0').replace('I', '1').replace('B', '8').replace('S', '5')
        
        texto_corregido = letras + numeros
    else:
        texto_corregido = texto_limpio

    patron_chileno = r'^([A-Z]{4}[0-9]{2}|[A-Z]{2}[0-9]{4})$'
    
    if re.match(patron_chileno, texto_corregido):
        return texto_corregido
    return None

def mejorar_imagen_para_ocr(recorte_bgr):
    """
    Normaliza el brillo, elimina sombras y aumenta el contraste de las letras.
    """
    # 1. Convertir a escala de grises 
    gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Agrandar la imagen al doble si está muy lejos 
    gris = cv2.resize(gris, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 3. Ecualización adaptativa (CLAHE) para neutralizar reflejos de luz y sombras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mejorada = clahe.apply(gris)
    
    return mejorada

# --- 1. INICIALIZACIÓN DEL SISTEMA ---
print("1. Cargando base de datos de residentes en memoria RAM...")
ruta_csv = os.path.join("data", "residentes.csv")
db_residentes = cargar_residentes(ruta_csv)

print("2. Cargando modelo YOLO y motor EasyOCR...")

model = YOLO('models/best.pt')  # Modelo entrenado para detectar patentes chilenas
reader = easyocr.Reader(['es', 'en'], gpu=True)

print("3. Iniciando cámara USB...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se encontró ninguna cámara conectada.")
    exit()

# Variables para mantener altos los FPS 
frame_count = 0
procesar_cada_x_frames = 3
ultimos_resultados = [] # Guardará: (x1, y1, x2, y2, mensaje, color_bgr)

print("\n--- SISTEMA ALPR ACTIVO (Presiona 'q' en la ventana para salir) ---")

# --- 2. BUCLE DE PROCESAMIENTO EN TIEMPO REAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Ejecutar la IA 
    if frame_count % procesar_cada_x_frames == 0:
        nuevos_resultados = []
        
        resultados = model(frame, stream=True, verbose=False)

        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confianza = box.conf[0]

                if confianza > 0.35:
                    alto_f, ancho_f, _ = frame.shape
                    x1_pad = max(0, x1 - 8)
                    y1_pad = max(0, y1 - 8)
                    x2_pad = min(ancho_f, x2 + 8)
                    y2_pad = min(alto_f, y2 + 8)
                    recorte = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # Valores por defecto mientras detecta el objeto pero aún no lee una patente válida
                    mensaje_pantalla = "Analizando..."
                    color_caja = (255, 165, 0) # Azul/Naranjo neutro (formato BGR)

                    if recorte.size > 0:
                        recorte_optimizado = mejorar_imagen_para_ocr(recorte)
                        lecturas_ocr = reader.readtext(recorte_optimizado)
                        
                        for (bbox_ocr, txt, prob) in lecturas_ocr:
                            patente_valida = limpiar_y_validar_patente(txt)
                            
                            # Si el texto cumple con el formato de patente chilena:
                            if patente_valida:
                                # Consultar en la memoria RAM (Diccionario del CSV)
                                if patente_valida in db_residentes:
                                    datos = db_residentes[patente_valida]
                                    nombre = datos.get('nombre_residente', 'Residente')
                                    depto = datos.get('departamento', 'S/N')
                                    
                                    mensaje_pantalla = f"AUTORIZADO: {patente_valida} | {nombre} (Depto {depto})"
                                    color_caja = (0, 255, 0) # VERDE (Acceso Permitido)
                                else:
                                    mensaje_pantalla = f"DENEGADO: {patente_valida} (No Registrado)"
                                    color_caja = (0, 0, 255) # ROJO (Acceso Denegado)
                                
                                # Si ya encontró una patente válida en este recorte, dejamos de buscar más texto
                                break 
                    
                    nuevos_resultados.append((x1, y1, x2, y2, mensaje_pantalla, color_caja))
        
        ultimos_resultados = nuevos_resultados

    # --- 3. DIBUJADO EN PANTALLA ---
    for (x1, y1, x2, y2, mensaje, color) in ultimos_resultados:
        # Dibujar el rectángulo alrededor del objeto/patente
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar un fondo oscuro detrás del texto para que se lea claramente con cualquier luz
        (ancho_texto, alto_texto), _ = cv2.getTextSize(mensaje, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - alto_texto - 10), (x1 + ancho_texto, y1), color, -1)
        
        # Escribir el texto en color blanco encima del fondo
        cv2.putText(frame, mensaje, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Control de Acceso Vehicular - ALPR Chile', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()