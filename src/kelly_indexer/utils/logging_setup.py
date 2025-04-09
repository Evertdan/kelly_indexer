# src/kelly_indexer/utils/logging_setup.py
# -*- coding: utf-8 -*-

"""
Módulo para configurar el sistema de logging de la aplicación Kelly Indexer
de forma centralizada.
"""

import logging
import sys
from pathlib import Path
# CORRECCIÓN: Importar List para el type hint de 'handlers'
from typing import Optional, List

# Mapeo de nombres de nivel de log a constantes de logging
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# Formato por defecto para los logs
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s [%(levelname)s] - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Obtener un logger para este módulo (útil para logs internos del setup)
logger = logging.getLogger(__name__)
# Poner un handler básico temporal por si setup_logging falla muy temprano
# o si se usa este logger antes de llamar a setup_logging
if not logger.hasHandlers():
     logger.addHandler(logging.StreamHandler(sys.stderr))
     logger.setLevel(logging.WARNING) # Nivel default bajo para no ser verboso

def setup_logging(
    log_level_str: str = 'INFO',
    log_file: Optional[Path] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT
) -> None:
    """
    Configura el logging raíz para la aplicación Kelly Indexer.

    Aplica un nivel, formato, y opcionalmente añade un handler para escribir
    logs a un archivo además de la consola (stderr).

    Args:
        log_level_str: Nivel mínimo de log a registrar (ej. 'DEBUG', 'INFO').
                       No sensible a mayúsculas/minúsculas.
        log_file: Ruta (objeto Path) opcional a un archivo donde guardar los logs.
                  Si es None, solo se logueará a la consola.
        log_format: Formato a usar para los mensajes de log.
        date_format: Formato a usar para la fecha/hora en los logs.
    """
    # Validar y obtener el nivel de logging numérico
    numeric_log_level = LOG_LEVEL_MAP.get(log_level_str.upper())
    if numeric_log_level is None:
         # Usar print aquí porque el logger raíz no está configurado aún
         print(f"[ADVERTENCIA SetupLogging] Nivel de log '{log_level_str}' no reconocido. Usando INFO por defecto.")
         numeric_log_level = logging.INFO # Usar INFO como fallback

    # CORRECCIÓN: Especificar tipo explícito para la lista de handlers
    handlers: List[logging.Handler] = []

    # --- Handler para la Consola ---
    try:
        console_handler = logging.StreamHandler(sys.stderr)
        # Se podría configurar un formatter específico si se quisiera diferente al de archivo
        # console_formatter = logging.Formatter(log_format, datefmt=date_format)
        # console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    except Exception as e:
         # Muy raro que falle StreamHandler, pero por si acaso
         print(f"[ERROR SetupLogging] No se pudo crear el handler de consola: {e}", file=sys.stderr)


    # --- Handler para Archivo (Opcional) ---
    file_handler: Optional[logging.FileHandler] = None # Definir fuera del if para claridad
    if log_file:
        # Convertir a Path si no lo es
        if not isinstance(log_file, Path):
             try: log_file = Path(log_file)
             except TypeError:
                 print(f"[ERROR SetupLogging] 'log_file' debe ser una ruta válida (Path), se recibió {type(log_file)}. Logueando solo a consola.", file=sys.stderr)
                 log_file = None # Anular

        if log_file: # Proceder si la ruta es válida
            try:
                # Asegurar que el directorio padre exista
                log_file.parent.mkdir(parents=True, exist_ok=True)
                # Crear el handler de archivo en modo 'append'
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                # file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format)) # Si se usa formatter específico
                handlers.append(file_handler) # Añadir a la lista (type-safe ahora)
                # Usar print para este mensaje inicial porque el logger aún no está 100% configurado
                print(f"[INFO SetupLogging] Logging configurado también para archivo: {log_file.resolve()}")
            except PermissionError:
                print(f"[ERROR SetupLogging] Permiso denegado al intentar crear/abrir archivo de log: {log_file}. Logueando solo a consola.", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR SetupLogging] No se pudo configurar el log a archivo {log_file}: {e}. Logueando solo a consola.", file=sys.stderr)

    # --- Configurar el Logger Raíz ---
    if not handlers:
         print("[ERROR SetupLogging] No se pudo configurar ningún handler. Usando config básica de emergencia.", file=sys.stderr)
         logging.basicConfig(level=numeric_log_level) # Config básica sin formato ni handlers específicos
         logging.error("Fallo al configurar handlers, usando configuración muy básica.")
         return

    try:
        # logging.basicConfig configura el logger raíz.
        # force=True (Python 3.8+) elimina handlers previos.
        logging.basicConfig(
            level=numeric_log_level,
            format=log_format,
            datefmt=date_format,
            handlers=handlers, # Pasar la lista de handlers construida
            force=True
        )
        # Loguear usando el logger raíz recién configurado
        logging.info(f"Sistema de Logging inicializado a nivel: {logging.getLevelName(logging.getLogger().level)}")
        if file_handler: # Loguear que el archivo se está usando
             logging.info(f"Logs siendo escritos también en: {log_file}")

    except ValueError as e:
         # Error común si el formato es inválido
         print(f"[ERROR CRÍTICO SetupLogging] Error en la configuración de formato de logging: {e}")
         # Intentar configurar básico para que al menos algo funcione
         logging.basicConfig(level=logging.WARNING)
         logging.critical(f"Fallo en formato de log, usando config básica: {e}")
    except Exception as e:
         print(f"[ERROR CRÍTICO SetupLogging] Error inesperado al configurar logging con basicConfig: {e}")
         logging.basicConfig(level=logging.WARNING)
         logging.critical(f"Fallo inesperado en setup logging, usando config básica: {e}")


# --- Bloque para pruebas rápidas de este módulo ---
if __name__ == "__main__":
    print("--- Probando setup_logging ---")

    # Ejemplo 1: Solo consola, nivel INFO (default implícito)
    print("\nConfigurando logging a INFO (solo consola)...")
    setup_logging()
    logging.getLogger("prueba1.debug").debug("Este mensaje DEBUG no debería aparecer.")
    logging.getLogger("prueba1.info").info("Este mensaje INFO sí debería aparecer.")
    logging.getLogger("prueba1.warning").warning("Este mensaje WARNING también.")

    # Ejemplo 2: Consola y Archivo, nivel DEBUG
    test_log_file = Path("./_test_indexer_logging.log")
    # Eliminar archivo de log anterior si existe para prueba limpia
    if test_log_file.exists(): test_log_file.unlink()
    print(f"\nConfigurando logging a DEBUG (consola y archivo: {test_log_file})...")
    setup_logging(log_level_str='DEBUG', log_file=test_log_file)
    logging.getLogger("prueba2.debug").debug("Mensaje DEBUG para consola y archivo.")
    logging.getLogger("prueba2.info").info("Mensaje INFO para consola y archivo.")
    logging.getLogger("otro_modulo").warning("Mensaje WARNING desde otro logger.")

    print(f"\nVerifica el contenido del archivo: {test_log_file.resolve()}")

    # Ejemplo 3: Nivel inválido
    print("\nConfigurando con nivel inválido...")
    setup_logging(log_level_str='INVALIDO')
    # El logger raíz debería haberse quedado en INFO (el default de fallback)
    logging.getLogger("prueba3.info").info("Este mensaje INFO debería aparecer (fallback a INFO).")
    logging.getLogger("prueba3.debug").debug("Este mensaje DEBUG NO debería aparecer.")

    # Limpieza final (opcional)
    # if test_log_file.exists():
    #     test_log_file.unlink()
    #     print("Archivo de log de prueba eliminado.")