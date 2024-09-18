import os
import shutil

# Ruta base donde están los directorios originales
ruta_base = "data_multiple/colores"

# Ruta destino donde quieres copiar los archivos
ruta_destino = "productive_dataset/colores"

# Nombres de los familiares
acciones = ["amarillo", "negro", "rojo", "verde"]

# Niveles que quieres revisar
niveles = ["0", "1", "2", "3", "4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29"]

for accion in acciones:
    for nivel in niveles:
        ruta_origen = os.path.join(ruta_base, accion, nivel, "4")
        
        # Verifica si la ruta origen es un directorio
        if os.path.isdir(ruta_origen):
            print(f"El directorio existe: {ruta_origen}")
            try:
                # Crear la ruta destino para los archivos
                destino_directorio = os.path.join(ruta_destino, f"{accion}/{nivel}")
                os.makedirs(destino_directorio, exist_ok=True)

                # Copiar todos los archivos del directorio origen al destino
                for archivo in os.listdir(ruta_origen):
                    archivo_origen = os.path.join(ruta_origen, archivo)
                    archivo_destino = os.path.join(destino_directorio, archivo)
                    
                    # Copia cada archivo individualmente
                    shutil.copy2(archivo_origen, archivo_destino)
                    print(f"Archivo copiado desde {archivo_origen} a {archivo_destino}")
            except Exception as e:
                print(f"Error al copiar archivos: {e}")
        else:
            print(f"El directorio no se encontró: {ruta_origen}")