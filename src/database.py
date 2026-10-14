import csv
import os

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
                # 1. Extraer la patente (ahora seguro en minúscula y sin espacios)
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

# --- BLOQUE DE PRUEBA INDIVIDUAL ---
if __name__ == "__main__":
    # Calcula la ruta exacta hacia data/residentes.csv sin importar desde dónde ejecutes el script
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(directorio_actual, "..", "data", "residentes.csv")
    
    db = cargar_residentes(ruta_csv)
    
    # Simulemos que la cámara acaba de leer una patente:
    patente_leida_por_camara = "BB-CC-12"
    patente_limpia = patente_leida_por_camara.replace("-", "").strip().upper()
    
    if patente_limpia in db:
        datos = db[patente_limpia]
        print(f"\n¡ACCESO PERMITIDO! Residente: {datos.get('nombre_residente')}, Depto: {datos.get('departamento')}")
    else:
        print(f"\nACCESO DENEGADO. La patente '{patente_limpia}' no está en la lista: {list(db.keys())}")