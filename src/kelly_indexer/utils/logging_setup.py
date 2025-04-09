# src/kelly_indexer/utils/logging_setup.py
# -*- coding: utf-8 -*-

"""
Módulo para configurar el sistema de logging de la aplicación de forma centralizada.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

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

# Obtener un logger para este módulo de utilidad (opcional, pero bueno para debug)
logger = logging.getLogger(__name__)

def setup_logging(
    log_level_str: str = 'INFO',
    log_file: Optional[Path] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT
) -> None:
    """
    Configura el logging raíz para la aplicación Kelly Indexer.

    Aplica un nivel, formato, y opcionalmente añade un handler para escribir
    logs a un archivo además de la consola.

    Args:
        log_level_str: Nivel mínimo de log a registrar (ej. 'DEBUG', 'INFO').
                       No sensible a mayúsculas/minúsculas.
        log_file: Ruta (objeto Path) opcional a un archivo donde guardar los logs.
                  Si es None, solo se logueará a la consola (stderr).
        log_format: Formato a usar para los mensajes de log.
        date_format: Formato a usar para la fecha/hora en los logs.
    """
    # Validar y obtener el nivel de logging numérico
    numeric_log_level = LOG_LEVEL_MAP.get(log_level_str.upper(), logging.INFO)
    if log_level_str.upper() not in LOG_LEVEL_MAP:
         # Usar print aquí porque el logger raíz aún no está configurado con el nivel deseado
         print(f"[ADVERTENCIA SetupLogging] Nivel de log '{log_level_str}' no reconocido. Usando INFO por defecto.")

    # Crear lista de handlers (destinos de los logs)
    handlers = []

    # --- Handler para la Consola ---
    # Siempre añadir handler de consola (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    # (Opcional) Podríamos ponerle un formato diferente a la consola vs archivo si quisiéramos
    # console_formatter = logging.Formatter(log_format, datefmt=date_format)
    # console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # --- Handler para Archivo (Opcional) ---
    if log_file:
        # Convertir a Path si no lo es (aunque el type hint ya lo pide)
        if not isinstance(log_file, Path):
             try: log_file = Path(log_file)
             except TypeError:
                 print(f"[ERROR SetupLogging] 'log_file' debe ser una ruta válida (Path), se recibió {type(log_file)}. Logueando solo a consola.", file=sys.stderr)
                 log_file = None # Anular para no intentar usarlo

        if log_file: # Proceder si la ruta es válida
            try:
                # Asegurar que el directorio padre del archivo de log exista
                log_file.parent.mkdir(parents=True, exist_ok=True)

                # Crear el handler de archivo
                # Usar 'a' para modo append (añadir al final), 'w' para sobrescribir
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')

                # (Opcional) Establecer un formato específico para el archivo si es necesario
                # file_formatter = logging.Formatter(log_format, datefmt=date_format)
                # file_handler.setFormatter(file_formatter)

                handlers.append(file_handler)
                # Usar print aquí porque el logger raíz aún no está configurado
                print(f"[INFO SetupLogging] Logging configurado también para archivo: {log_file.resolve()}")

            except PermissionError:
                print(f"[ERROR SetupLogging] Permiso denegado al intentar crear/abrir archivo de log: {log_file}. Logueando solo a consola.", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR SetupLogging] No se pudo configurar el log a archivo {log_file}: {e}. Logueando solo a consola.", file=sys.stderr)

    # --- Configurar el Logger Raíz ---
    # logging.basicConfig configura el logger raíz. Es la forma más simple.
    # Si se llama varias veces, puede añadir handlers duplicados a menos que usemos force=True.
    # force=True (Python 3.8+) elimina handlers existentes antes de añadir los nuevos.
    if not handlers:
         print("[ERROR SetupLogging] No se pudo configurar ningún handler de logging.", file=sys.stderr)
         # Configurar un handler básico de emergencia para que al menos se vea algo
         logging.basicConfig(level=numeric_log_level)
         logging.error("Fallo al configurar handlers, usando configuración básica.")
         return

    try:
        logging.basicConfig(
            level=numeric_log_level,
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True # Importante para reconfigurar si es necesario
        )
        # Loguear un mensaje de prueba usando el logger raíz recién configurado
        logging.info(f"Sistema de Logging inicializado a nivel: {logging.getLevelName(logging.getLogger().level)}")

    except ValueError as e:
         # Puede ocurrir si el formato es inválido
         print(f"[ERROR CRÍTICO SetupLogging] Error en la configuración de formato de logging: {e}")
    except Exception as e:
         print(f"[ERROR CRÍTICO SetupLogging] Error inesperado al configurar logging con basicConfig: {e}")


# --- Bloque para pruebas rápidas de este módulo ---
if __name__ == "__main__":
    print("--- Probando setup_logging ---")

    # Ejemplo 1: Solo consola, nivel INFO (default implícito)
    print("\nConfigurando logging a INFO (solo consola)...")
    setup_logging() # Llama con defaults
    logging.getLogger("prueba1").debug("Este mensaje DEBUG no debería aparecer.")
    logging.getLogger("prueba1").info("Este mensaje INFO sí debería aparecer.")
    logging.getLogger("prueba1").warning("Este mensaje WARNING también.")

    # Ejemplo 2: Consola y Archivo, nivel DEBUG
    test_log_file = Path("./_test_indexer_logging.log")
    print(f"\nConfigurando logging a DEBUG (consola y archivo: {test_log_file})...")
    setup_logging(log_level_str='DEBUG', log_file=test_log_file)
    # Usar diferentes nombres de logger para verlos en el formato
    logging.getLogger("prueba2.debug").debug("Mensaje DEBUG para consola y archivo.")
    logging.getLogger("prueba2.info").info("Mensaje INFO para consola y archivo.")
    logging.getLogger("otro_modulo").warning("Mensaje WARNING desde otro logger.")

    print(f"\nVerifica el contenido del archivo: {test_log_file.resolve()}")
    # Opcional: Limpiar archivo de prueba
    # try:
    #     test_log_file.unlink()
    #     print("Archivo de log de prueba eliminado.")
    # except OSError:
    #     pass

    # Ejemplo 3: Nivel inválido
    print("\nConfigurando con nivel inválido...")
    setup_logging(log_level_str='INVALIDO')
    # El logger raíz debería haberse quedado en INFO (el default de fallback)
    logging.getLogger("prueba3").info("Este mensaje INFO debería aparecer (fallback).")
    logging.getLogger("prueba3").debug("Este mensaje DEBUG NO debería aparecer.")