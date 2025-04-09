# src/kelly_indexer/data_loader.py
# -*- coding: utf-8 -*-

"""
Módulo encargado de descubrir, cargar y validar los datos Q&A
desde los archivos JSON de entrada.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Obtener logger para este módulo
logger = logging.getLogger(__name__)

# CORRECCIÓN: Definir tqdm como dummy primero, luego intentar importar el real
tqdm_available = False
tqdm = lambda x, **kwargs: x # Dummy inicial que solo devuelve el iterable
try:
    from tqdm import tqdm as tqdm_real # Importar el real
    tqdm = tqdm_real # Sobrescribir si la importación fue exitosa
    tqdm_available = True
    # logger.debug("Librería tqdm encontrada y cargada.") # Opcional
except ImportError:
    # El dummy ya está definido, solo loguear si se desea (run_processing ya advierte)
    # logger.warning("Librería 'tqdm' no instalada. No se mostrará barra de progreso.")
    pass # Mantener el dummy lambda


# Definir las claves esperadas en cada objeto Q&A dentro del JSON
EXPECTED_QA_KEYS = {'q', 'a', 'product', 'keywords'}

def load_single_json_file(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Carga y valida el contenido de un único archivo JSON.

    Espera que el archivo contenga una lista de objetos JSON, donde cada
    objeto tiene las claves definidas en EXPECTED_QA_KEYS.

    Args:
        file_path: Ruta (objeto Path) al archivo JSON.

    Returns:
        Una lista de diccionarios Q&A válidos si el archivo es correcto,
        una lista vacía si el archivo JSON contiene una lista vacía,
        o None si ocurre un error de lectura, parseo o validación estructural grave.
    """
    if not isinstance(file_path, Path):
         logger.error(f"Tipo inválido para ruta de archivo: {type(file_path)}")
         return None
    if not file_path.is_file():
        # Loguear como warning porque el scanner podría encontrar archivos temporales, etc.
        logger.warning(f"Ruta no es un archivo válido o no encontrado: {file_path}")
        return None

    logger.debug(f"Intentando cargar y validar JSON desde: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validar estructura principal: debe ser una lista
        if not isinstance(data, list):
            logger.error(f"Error de formato en {file_path.name}: Se esperaba una lista JSON ([...]), se obtuvo {type(data).__name__}.")
            return None

        if not data: # Si la lista está vacía
             logger.info(f"Archivo JSON {file_path.name} contiene una lista vacía [].")
             return [] # Devolver lista vacía, es válido

        # Validar cada item en la lista
        valid_qa_list = []
        invalid_items_count = 0
        for i, item in enumerate(data):
            if isinstance(item, dict) and EXPECTED_QA_KEYS.issubset(item.keys()):
                 # Verificación básica de tipos y contenido no vacío para q/a
                 if (isinstance(item.get('q'), str) and item['q'].strip() and
                     isinstance(item.get('a'), str) and item['a'].strip() and
                     isinstance(item.get('product'), str) and
                     isinstance(item.get('keywords'), list)):
                     # Opcional: Validar que keywords contenga solo strings
                     # if all(isinstance(kw, str) for kw in item['keywords']):
                     #     valid_qa_list.append(item)
                     # else: ... log warning ...
                     valid_qa_list.append(item) # Añadir si tipos básicos son correctos
                 else:
                      logger.warning(f"Item {i+1} en {file_path.name} tiene claves correctas pero tipos/contenido inválido (ej. q/a vacíos o keywords no es lista). Item descartado: {str(item)[:150]}...")
                      invalid_items_count += 1
            else:
                logger.warning(f"Item {i+1} en {file_path.name} no tiene la estructura Q&A esperada (claves: {EXPECTED_QA_KEYS}). Item descartado: {str(item)[:150]}...")
                invalid_items_count += 1

        if invalid_items_count > 0:
             logger.warning(f"Se descartaron {invalid_items_count} items inválidos del archivo {file_path.name}.")

        # Decidir si devolver lista parcial o None si hubo errores
        # Devolver la lista de válidos encontrados es más flexible
        # if not valid_qa_list:
        #      logger.error(f"Ningún item válido encontrado en {file_path.name} después de filtrar.")
        #      return None # O devolver [] si se prefiere

        logger.debug(f"Cargados {len(valid_qa_list)} Q&A válidos desde {file_path.name}.")
        return valid_qa_list # Devuelve lista (potencialmente vacía si todos fallaron validación)

    except FileNotFoundError:
        logger.error(f"Archivo JSON no encontrado (inesperado después de check inicial): {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON en archivo {file_path.name}: {e}")
        return None
    except PermissionError:
         logger.error(f"Permiso denegado al leer archivo JSON: {file_path}")
         return None
    except Exception as e:
        logger.exception(f"Error inesperado al cargar o validar archivo JSON {file_path.name}: {e}")
        return None

def load_all_qas_from_directory(base_directory: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Escanea recursivamente un directorio base, carga y valida todos los
    archivos JSON encontrados, devolviendo un diccionario con los Q&A válidos.

    Args:
        base_directory: Ruta (objeto Path) al directorio raíz donde buscar
                        archivos JSON (ej. data/input/json/SOAP_TXT).

    Returns:
        Un diccionario donde las claves son las rutas relativas (como string)
        de los archivos JSON procesados (respecto a base_directory) que contenían
        al menos un Q&A válido, y los valores son las listas de esos Q&A válidos.
    """
    if not base_directory.is_dir():
        logger.error(f"El directorio de entrada especificado no existe o no es un directorio: {base_directory}")
        return {}

    all_qas: Dict[str, List[Dict[str, Any]]] = {}
    json_files_found = []
    try:
        logger.info(f"Buscando archivos .json recursivamente en: {base_directory}...")
        json_files_found = list(base_directory.rglob("*.json"))
        file_count = len(json_files_found)
        logger.info(f"Encontrados {file_count} archivos .json.")
    except Exception as e:
         logger.exception(f"Error al buscar archivos .json en {base_directory}: {e}")
         return {} # Devolver vacío si falla la búsqueda

    processed_files_count = 0
    error_files_count = 0
    total_valid_qas = 0

    if not json_files_found:
        logger.warning(f"No se encontraron archivos .json en el directorio: {base_directory}")
        return {}

    # Usar tqdm si está disponible
    iterable_files = tqdm(json_files_found, desc="Cargando archivos JSON Q&A", unit="archivo") if tqdm_available else json_files_found

    for json_file_path in iterable_files:
         if hasattr(iterable_files, 'set_postfix_str'): # Actualizar barra si tqdm está activo
              iterable_files.set_postfix_str(f"{json_file_path.name[:30]}...", refresh=True)

         logger.debug(f"Procesando archivo JSON: {json_file_path}")
         # Calcular ruta relativa para usarla como clave y para logs
         try:
              # Usar relative_to(base_directory) para obtener la ruta desde la base
              relative_path_str = str(json_file_path.relative_to(base_directory))
         except ValueError:
              logger.warning(f"No se pudo calcular ruta relativa para {json_file_path} respecto a {base_directory}. Usando nombre de archivo.")
              relative_path_str = json_file_path.name # Fallback a solo nombre de archivo

         # Cargar y validar el archivo individual
         qa_list = load_single_json_file(json_file_path)

         if qa_list is not None: # Si la carga no dio error fatal (puede ser lista vacía [])
             processed_files_count += 1
             if qa_list: # Solo añadir al resultado si la lista contiene Q&As válidos
                 all_qas[relative_path_str] = qa_list
                 total_valid_qas += len(qa_list)
                 logger.info(f"Cargado: '{relative_path_str}' ({len(qa_list)} Q&As válidos)")
             else:
                 # Si devolvió [], fue un archivo válido pero vacío o sin items válidos
                 logger.info(f"Archivo '{relative_path_str}' procesado pero no contenía Q&As válidos o estaba vacío.")
         else:
             # Si devolvió None, hubo un error durante la carga/validación
             logger.error(f"Fallo completo al cargar/validar archivo: '{relative_path_str}'. Excluido del resultado.")
             error_files_count += 1

    logger.info(f"Carga de datos finalizada.")
    logger.info(f"  Archivos JSON encontrados: {file_count}")
    logger.info(f"  Archivos JSON procesados (intentados): {processed_files_count}")
    logger.info(f"  Archivos con errores de carga/formato: {error_files_count}")
    logger.info(f"  Total Q&As válidos cargados: {total_valid_qas}")
    if error_files_count > 0:
         logger.warning("Algunos archivos JSON tuvieron errores y fueron omitidos.")

    return all_qas


# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    # Configurar logging básico para ver salida de prueba
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s: %(message)s')

    print("\n--- Probando data_loader (creando archivos temporales) ---")
    # Crear estructura de prueba temporal
    test_base_dir = Path("./_test_data_loader")
    # Limpiar directorio de prueba anterior si existe
    if test_base_dir.exists():
         shutil.rmtree(test_base_dir) # Importar shutil
    test_input_dir = test_base_dir / "input" / "json" / "SOAP_TXT"
    test_input_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de prueba creado en: {test_input_dir.resolve()}")


    # --- Crear Archivos de Prueba ---
    # Archivo válido
    valid_data = [{"q": "P1", "a": "R1", "product": "A", "keywords": ["k1"]}]
    valid_file = test_input_dir / "valid.json"
    try: valid_file.write_text(json.dumps(valid_data), encoding='utf-8')
    except Exception as e: print(f"Error creando valid.json: {e}")

    # Archivo con lista vacía
    empty_list_file = test_input_dir / "empty_list.json"
    try: empty_list_file.write_text(json.dumps([]), encoding='utf-8')
    except Exception as e: print(f"Error creando empty_list.json: {e}")

    # Archivo con estructura inválida (no lista)
    invalid_structure_file = test_input_dir / "invalid_structure.json"
    try: invalid_structure_file.write_text(json.dumps({"not": "a list"}), encoding='utf-8')
    except Exception as e: print(f"Error creando invalid_structure.json: {e}")

    # Archivo con items inválidos (1 válido, 2 inválidos)
    invalid_items_data = [
        {"q": "P3", "a": "R3", "product": "B", "keywords": ["k3"]},
        {"q": "P4", "a": "R4"}, # Faltan keys
        {"q": "", "a": "R5", "product": "B", "keywords": ["k5"]} # q vacía
    ]
    invalid_items_file = test_input_dir / "subdir" / "invalid_items.json"
    invalid_items_file.parent.mkdir(exist_ok=True)
    try: invalid_items_file.write_text(json.dumps(invalid_items_data), encoding='utf-8')
    except Exception as e: print(f"Error creando invalid_items.json: {e}")

    # Archivo JSON mal formado
    bad_json_file = test_input_dir / "bad.json"
    try: bad_json_file.write_text("[{'bad': 'json'}", encoding='utf-8')
    except Exception as e: print(f"Error creando bad.json: {e}")

    print(f"\nArchivos de prueba creados. Cargando desde: {test_input_dir}")
    # Llamar a la función principal para cargar los datos
    loaded_data = load_all_qas_from_directory(test_input_dir)

    print("\n--- Resultados de la Carga de Prueba ---")
    print(f"Diccionario de Q&As cargados (claves = rutas relativas): {list(loaded_data.keys())}")
    print(f"Total de archivos con Q&A válidos: {len(loaded_data)}")

    # Verificar resultados esperados
    assert "valid.json" in loaded_data, "Debería haber cargado valid.json"
    assert len(loaded_data["valid.json"]) == 1, "valid.json debería tener 1 Q&A"

    rel_path_invalid = Path("subdir") / "invalid_items.json" # Path relativo esperado
    assert str(rel_path_invalid) in loaded_data, "Debería haber cargado invalid_items.json"
    assert len(loaded_data[str(rel_path_invalid)]) == 1, "invalid_items.json debería tener 1 Q&A válido"
    assert loaded_data[str(rel_path_invalid)][0]['q'] == "P3"

    assert "empty_list.json" not in loaded_data, "Archivo con lista vacía no debería estar en el resultado final"
    assert "invalid_structure.json" not in loaded_data, "Archivo con estructura inválida no debería estar"
    assert "bad.json" not in loaded_data, "Archivo JSON mal formado no debería estar"

    print("\nPruebas básicas de data_loader completadas.")

    # Limpieza (opcional)
    # print(f"\nLimpiando directorio de prueba: {test_base_dir}")
    # import shutil # Importar si se usa
    # shutil.rmtree(test_base_dir, ignore_errors=True)