import csv
import os
from datetime import datetime

def cargar_residentes(ruta_archivo):
    """
    Lee un archivo CSV de forma robusta (soporta comas, punto y coma, y formato Excel)
    y lo carga en la memoria RAM como un diccionario.
    """
    base_datos_ram = {}
    
    # Verificar si el archivo realmente existe en esa ruta
    if not os.path.exists(ruta_archivo):
        print(f"[ERROR CRÍTICO] No existe el archivo en: {os.path.abspath(ruta_archivo)}")
        return {}

    try:
        # 'utf-8-sig' elimina los caracteres invisibles que agrega Excel en Windows
        with open(ruta_archivo, mode='r', encoding='utf-8-sig') as archivo:
            contenido_muestra = archivo.read(1024)
            archivo.seek(0) # Volver al inicio del archivo
            
            # Detectar automáticamente si está separado por ';' (Excel en español) o ','
            delimitador = ';' if ';' in contenido_muestra else ','
            print(f"[INFO] Leyendo CSV con delimitador detectado: '{delimitador}'")
            
            lector = csv.DictReader(archivo, delimiter=delimitador)
            
            # Limpiar posibles espacios en los nombres de las columnas (ej: " patente " -> "patente")
            if lector.fieldnames:
                lector.fieldnames = [nombre.strip().lower() for nombre in lector.fieldnames]
                print(f"[INFO] Columnas detectadas en el archivo: {lector.fieldnames}")
            
            for fila in lector:
                # 1. Extraer la patente 
                patente_cruda = fila.get('patente', '')
                
                if patente_cruda:
                    # 2. Limpiar la patente: mayúsculas, sin espacios ni guiones
                    patente_limpia = patente_cruda.replace(" ", "").replace("-", "").replace("·", "").strip().upper()
                    
                    # 3. Guardar en el diccionario
                    if patente_limpia:
                        base_datos_ram[patente_limpia] = fila
                        
        print(f"[EXITO] Base de datos cargada. {len(base_datos_ram)} residentes encontrados.")
        print(f"[DEBUG] Patentes cargadas en memoria: {list(base_datos_ram.keys())}")
        return base_datos_ram

    except Exception as e:
        print(f"[ERROR] Problema al leer la base de datos: {e}")
        return {}

def registrar_acceso(ruta_historial, patente, datos_residente=None):
    """
    Registra la entrada de un vehículo con fecha y hora en un archivo CSV
    listo para abrirse como tabla en Excel.
    """
    archivo_existe = os.path.exists(ruta_historial)
    
    ahora = datetime.now()
    fecha_str = ahora.strftime("%Y-%m-%d")
    hora_str = ahora.strftime("%H:%M:%S")

    if datos_residente:
        estado = "AUTORIZADO"
        nombre = datos_residente.get('nombre_residente', 'Residente')
        depto = datos_residente.get('departamento', 'S/N')
        estacionamiento = datos_residente.get('estacionamiento', 'S/N')
    else:
        estado = "DENEGADO"
        nombre = "Visita / No Registrado"
        depto = "-"
        estacionamiento = "-"

    try:
        # 'a' agrega una fila al final sin borrar el historial previo
        with open(ruta_historial, mode='a', newline='', encoding='utf-8-sig') as archivo:
            escritor = csv.writer(archivo, delimiter=';')
            
            # Si el archivo es nuevo, escribimos los títulos de las columnas primero
            if not archivo_existe:
                escritor.writerow([
                    'Fecha', 'Hora', 'Patente', 'Estado Acceso', 
                    'Nombre Residente', 'Departamento', 'Estacionamiento'
                ])
            
            escritor.writerow([fecha_str, hora_str, patente, estado, nombre, depto, estacionamiento])
            
        print(f"[HISTORIAL] Registrado: {fecha_str} {hora_str} | {patente} ({estado})")

    except PermissionError:
        print(f"[AVISO] No se pudo guardar en el historial porque el archivo está abierto en Excel.")
    except Exception as e:
        print(f"[ERROR] Problema al guardar en el historial: {e}")

