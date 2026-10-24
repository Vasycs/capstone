import cv2
import easyocr
import re
import os
import time
import math
import torch
from ultralytics import YOLO
from src.database import cargar_residentes

def limpiar_y_validar_patente(texto_crudo):
    texto_limpio = re.sub(r'[^A-Za-z0-9]', '', texto_crudo).upper()
    posibles = re.findall(r'[A-Z0-9]{6}', texto_limpio)
    
    for candidato in posibles:
        # Formato actual (BBCC12)
        letras_act = candidato[:4].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
        nums_act = candidato[4:].replace('O', '0').replace('Q', '0').replace('I', '1').replace('B', '8').replace('S', '5')
        candidato_actual = letras_act + nums_act
        if re.match(r'^[A-Z]{4}[0-9]{2}$', candidato_actual):
            return candidato_actual

        # Formato antiguo (AA1000)
        letras_ant = candidato[:2].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
        nums_ant = candidato[2:].replace('O', '0').replace('Q', '0').replace('I', '1').replace('B', '8').replace('S', '5')
        candidato_antiguo = letras_ant + nums_ant
        if re.match(r'^[A-Z]{2}[0-9]{4}$', candidato_antiguo):
            return candidato_antiguo

    return None

def mejorar_imagen_para_ocr(recorte_bgr):
    gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    if ancho < 150:
        gris = cv2.resize(gris, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gris)

# --- 1. INICIALIZACIÓN ---
usar_gpu = torch.cuda.is_available()
dispositivo = 0 if usar_gpu else 'cpu'
print(f"[HARDWARE] Inferencia activa en: {torch.cuda.get_device_name(0) if usar_gpu else 'CPU'}")

ruta_csv = os.path.join("data", "residentes.csv")
db_residentes = cargar_residentes(ruta_csv)

# modelo entrenado 
model = YOLO('models/best.pt') 
reader = easyocr.Reader(['es', 'en'], gpu=usar_gpu)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- VARIABLES DE CONTROL Y MEMORIA ESPACIAL ---
frame_count = 0
procesar_cada_x_frames = 4
ultimas_cajas = []
cache_patentes = {}

# Lista de objetos recordados en pantalla: [{'cx': int, 'cy': int, 'mensaje': str, 'color': tuple, 'tiempo': float}]
objetos_activos = []
RADIO_SEGUIMIENTO_PX = 130  # Distancia máxima en píxeles para considerar que es la misma patente moviéndose
TIEMPO_MEMORIA_SEG = 2.0    # Cuántos segundos recuerda la patente aunque se tape o desenfoque

prev_time = time.time()

print("\n--- SISTEMA ALPR CON MEMORIA ESPACIAL ACTIVO (Presiona 'q' para salir) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    ahora = time.time()

    # Limpiar de la memoria las patentes que ya salieron de la cámara hace más de 2 segundos
    objetos_activos = [obj for obj in objetos_activos if (ahora - obj['tiempo']) < TIEMPO_MEMORIA_SEG]

    if frame_count % procesar_cada_x_frames == 0:
        nuevas_cajas = []
        resultados = model(frame, stream=True, verbose=False, device=dispositivo, half=usar_gpu, conf=0.35)

        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calcular el punto central (X, Y) de la patente detectada
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 1. ¿ESTÁ CERCA DE UNA PATENTE QUE YA LEÍMOS HACE UN INSTANTE?
                objeto_conocido = None
                menor_distancia = RADIO_SEGUIMIENTO_PX

                for obj in objetos_activos:
                    distancia = math.hypot(cx - obj['cx'], cy - obj['cy'])
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        objeto_conocido = obj

                # 2. SI : Actualizamos su posición y MANTIENE EL TEXTO sin gastar OCR
                if objeto_conocido is not None:
                    objeto_conocido['cx'] = cx
                    objeto_conocido['cy'] = cy
                    objeto_conocido['tiempo'] = ahora
                    mensaje = objeto_conocido['mensaje']
                    color = objeto_conocido['color']

                # 3. SI ES NUEVA EN PANTALLA: Ejecutamos OCR para averiguar qué dice
                else:
                    mensaje = "Analizando..."
                    color = (255, 165, 0)

                    alto_f, ancho_f, _ = frame.shape
                    recorte = frame[max(0, y1-8):min(alto_f, y2+8), max(0, x1-8):min(ancho_f, x2+8)]

                    if recorte.size > 0:
                        recorte_opt = mejorar_imagen_para_ocr(recorte)
                        textos_leidos = reader.readtext(recorte_opt, detail=0)
                        
                        if textos_leidos:
                            texto_unido = "".join(textos_leidos)
                            patente_valida = limpiar_y_validar_patente(texto_unido)

                            if patente_valida:
                                if patente_valida in cache_patentes:
                                    mensaje, color = cache_patentes[patente_valida]
                                else:
                                    if patente_valida in db_residentes:
                                        datos = db_residentes[patente_valida]
                                        mensaje = f"AUTORIZADO: {patente_valida} | {datos.get('nombre_residente')} (Depto {datos.get('departamento')})"
                                        color = (0, 255, 0)
                                    else:
                                        mensaje = f"DENEGADO: {patente_valida} (No Registrado)"
                                        color = (0, 0, 255)
                                    cache_patentes[patente_valida] = (mensaje, color)

                                # Guardamos este objeto en la memoria espacial para seguirlo cuando se mueva
                                objetos_activos.append({
                                    'cx': cx,
                                    'cy': cy,
                                    'mensaje': mensaje,
                                    'color': color,
                                    'tiempo': ahora
                                })

                nuevas_cajas.append((x1, y1, x2, y2, mensaje, color))
        
        ultimas_cajas = nuevas_cajas

    # Dibujar resultados
    for (x1, y1, x2, y2, mensaje, color) in ultimas_cajas:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w_txt, h_txt), _ = cv2.getTextSize(mensaje, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - h_txt - 10), (x1 + w_txt, y1), color, -1)
        cv2.putText(frame, mensaje, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Mostrar FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Control de Acceso Vehicular - ALPR Chile', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()