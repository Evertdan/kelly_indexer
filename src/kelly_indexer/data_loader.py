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
    if not file_path.is_file():
        logger.error(f"Intento de cargar JSON desde una ruta inválida o inexistente: {file_path}")
        return None

    try:
        logger.debug(f"Leyendo archivo JSON: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # json.load lee directamente desde el archivo

        # Validar estructura principal: debe ser una lista
        if not isinstance(data, list):
            logger.error(f"Error de formato en {file_path.name}: Se esperaba una lista JSON, se obtuvo {type(data).__name__}.")
            return None

        if not data: # Si la lista está vacía
             logger.info(f"Archivo JSON {file_path.name} contiene una lista vacía.")
             return [] # Devolver lista vacía, es válido

        # Validar cada item en la lista
        valid_qa_list = []
        invalid_items_count = 0
        for i, item in enumerate(data):
            if isinstance(item, dict) and EXPECTED_QA_KEYS.issubset(item.keys()):
                 # Verificación básica de tipos (podría ser más estricta)
                 if (isinstance(item.get('q'), str) and item['q'].strip() and # Pregunta no vacía
                     isinstance(item.get('a'), str) and item['a'].strip() and # Respuesta no vacía
                     isinstance(item.get('product'), str) and
                     isinstance(item.get('keywords'), list)):
                     # Podríamos añadir validación de que keywords contiene strings
                     valid_qa_list.append(item)
                 else:
                      logger.warning(f"Item {i} en {file_path.name} tiene claves correctas pero tipos/contenido inválido (ej. q/a vacíos). Item descartado: {str(item)[:100]}...")
                      invalid_items_count += 1
            else:
                logger.warning(f"Item {i} en {file_path.name} no tiene la estructura Q/A esperada ({EXPECTED_QA_KEYS}). Item descartado: {str(item)[:100]}...")
                invalid_items_count += 1

        if invalid_items_count > 0:
             logger.warning(f"Se descartaron {invalid_items_count} items inválidos del archivo {file_path.name}.")

        if not valid_qa_list:
             logger.error(f"Ningún item válido encontrado en {file_path.name} después de filtrar.")
             return None # O devolver [] si se prefiere no tratar esto como error fatal del archivo

        logger.debug(f"Cargados {len(valid_qa_list)} Q&A válidos desde {file_path.name}.")
        return valid_qa_list

    except FileNotFoundError:
        # Este caso es manejado por la verificación is_file() inicial, pero por si acaso.
        logger.error(f"Archivo JSON no encontrado: {file_path}")
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
        de los archivos JSON procesados (respecto a base_directory) y los
        valores son las listas de Q&A válidos extraídos de cada archivo.
        Los archivos con errores o sin Q&A válidos no se incluirán en el resultado.
    """
    if not base_directory.is_dir():
        logger.error(f"El directorio de entrada especificado no existe o no es un directorio: {base_directory}")
        return {}

    all_qas: Dict[str, List[Dict[str, Any]]] = {}
    json_files_found = list(base_directory.rglob("*.json"))
    file_count = len(json_files_found)
    processed_count = 0
    error_files_count = 0

    logger.info(f"Encontrados {file_count} archivos .json en '{base_directory}' y subdirectorios.")

    # Usar tqdm si está disponible
    iterable_files = tqdm(json_files_found, desc="Cargando archivos JSON", unit="archivo") if tqdm else json_files_found

    for json_file_path in iterable_files:
         if hasattr(iterable_files, 'set_postfix_str'):
              iterable_files.set_postfix_str(f"{json_file_path.name[:30]}...", refresh=True)

         logger.debug(f"Procesando archivo: {json_file_path}")
         # Calcular ruta relativa para usarla como clave y para logs
         try:
              relative_path_str = str(json_file_path.relative_to(base_directory))
         except ValueError:
              logger.warning(f"No se pudo calcular ruta relativa para {json_file_path} respecto a {base_directory}. Usando ruta absoluta.")
              relative_path_str = str(json_file_path)


         qa_list = load_single_json_file(json_file_path)

         if qa_list is not None: # Incluye el caso de lista vacía []
             if qa_list: # Solo añadir si la lista no está vacía
                 all_qas[relative_path_str] = qa_list
                 logger.info(f"Cargado exitoso: {len(qa_list)} Q&As desde '{relative_path_str}'")
             else:
                 logger.info(f"Archivo '{relative_path_str}' contenía una lista JSON vacía, omitido del resultado final.")
             processed_count += 1
         else:
             logger.error(f"Fallo al cargar o validar el archivo '{relative_path_str}'. Ver logs anteriores para detalles.")
             error_files_count += 1

    logger.info(f"Carga de datos finalizada. Archivos procesados: {processed_count}. Archivos con errores: {error_files_count}.")
    if error_files_count > 0:
         logger.warning("Algunos archivos JSON no pudieron ser cargados o validados correctamente.")

    return all_qas


# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    print("--- Probando data_loader ---")
    # Crear estructura de prueba temporal
    test_base_dir = Path("./_test_data_loader")
    test_input_dir = test_base_dir / "input" / "json" / "SOAP_TXT"
    test_input_dir.mkdir(parents=True, exist_ok=True)

    # Archivo válido
    valid_data = [
        {"q": "P1", "a": "R1", "product": "ProdA", "keywords": ["k1"]},
        {"q": "P2", "a": "R2", "product": "ProdA", "keywords": ["k2"]},
    ]
    valid_file = test_input_dir / "valid.json"
    with open(valid_file, 'w', encoding='utf-8') as f: json.dump(valid_data, f)

    # Archivo con lista vacía
    empty_list_file = test_input_dir / "empty_list.json"
    with open(empty_list_file, 'w', encoding='utf-8') as f: json.dump([], f)

    # Archivo con estructura inválida (no es lista)
    invalid_structure_file = test_input_dir / "invalid_structure.json"
    with open(invalid_structure_file, 'w', encoding='utf-8') as f: json.dump({"not": "a list"}, f)

    # Archivo con items inválidos
    invalid_items_data = [
        {"q": "P3", "a": "R3", "product": "ProdB", "keywords": ["k3"]}, # Válido
        {"q": "P4", "a": "R4"}, # Inválido (faltan keys)
        {"q": "", "a": "R5", "product": "ProdB", "keywords": ["k5"]} # Inválido (q vacía)
    ]
    invalid_items_file = test_input_dir / "subdir" / "invalid_items.json"
    invalid_items_file.parent.mkdir(exist_ok=True)
    with open(invalid_items_file, 'w', encoding='utf-8') as f: json.dump(invalid_items_data, f)

    # Archivo JSON mal formado
    bad_json_file = test_input_dir / "bad.json"
    bad_json_file.write_text("[{'bad': 'json'}", encoding='utf-8')

    # Configurar logging para ver salida de prueba
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print(f"\nCargando desde directorio base: {test_input_dir}")
    loaded_data = load_all_qas_from_directory(test_input_dir)

    print("\n--- Resultados de la Carga ---")
    print(f"Número de archivos con Q&A válidos cargados: {len(loaded_data)}")

    for file_rel_path, qas in loaded_data.items():
        print(f"\nArchivo: {file_rel_path}")
        print(f"  Número de Q&As: {len(qas)}")
        # print(f"  Primer Q&A: {qas[0]}") # Imprimir ejemplo

    assert "valid.json" in loaded_data
    assert len(loaded_data["valid.json"]) == 2
    assert "subdir/invalid_items.json" in loaded_data # Se carga porque tiene 1 item válido
    assert len(loaded_data["subdir/invalid_items.json"]) == 1
    assert loaded_data["subdir/invalid_items.json"][0]['q'] == "P3"
    assert "empty_list.json" not in loaded_data # Archivo con lista vacía no se añade al dict final
    assert "invalid_structure.json" not in loaded_data
    assert "bad.json" not in loaded_data

    print("\nPruebas básicas completadas.")

    # Limpieza
    # print(f"\nLimpiando directorio de prueba: {test_base_dir}")
    # shutil.rmtree(test_base_dir, ignore_errors=True) # Importar shutil si se usa