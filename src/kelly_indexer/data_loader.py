# src/kelly_indexer/data_loader.py
# -*- coding: utf-8 -*-

"""
Módulo encargado de descubrir, cargar y validar los datos Q&A
desde los archivos JSON de entrada.
CORREGIDO para validar correctamente faq_id, categoria, texto_para_vectorizar.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union # Añadido Union
import shutil # Importar shutil para el bloque __main__

# Obtener logger para este módulo
logger = logging.getLogger(__name__)

# Configuración de TQDM (sin cambios)
tqdm_available = False
tqdm = lambda x, **kwargs: x
try:
    from tqdm import tqdm as tqdm_real
    tqdm = tqdm_real
    tqdm_available = True
except ImportError:
    pass

# --- INICIO CORRECCIÓN ---
# MODIFICADO: Definir las claves esperadas incluyendo los nuevos campos
EXPECTED_QA_KEYS = {'q', 'a', 'product', 'keywords', 'faq_id', 'categoria', 'texto_para_vectorizar'}
# --- FIN CORRECCIÓN ---

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
            # --- INICIO BLOQUE VALIDACIÓN CORREGIDO ---
            # 1. Verificar si es diccionario y si contiene TODAS las claves esperadas
            if isinstance(item, dict) and EXPECTED_QA_KEYS.issubset(item.keys()):
                # 2. Verificar tipos y si los campos REQUERIDOS NO están vacíos
                if (isinstance(item.get('q'), str) and item['q'].strip() and
                    isinstance(item.get('a'), str) and item['a'].strip() and
                    isinstance(item.get('product'), str) and # Permitir producto vacío
                    isinstance(item.get('categoria'), str) and # Permitir categoría vacía
                    isinstance(item.get('keywords'), list) and
                    isinstance(item.get('faq_id'), str) and item['faq_id'].strip() and # Req no vacío
                    isinstance(item.get('texto_para_vectorizar'), str) and item['texto_para_vectorizar'].strip()): # Req no vacío

                    # Si pasa todas las validaciones, añadirlo
                    valid_qa_list.append(item)
                else:
                    # Tiene las claves pero el tipo o contenido requerido es inválido
                    logger.warning(f"Item {i+1} en {file_path.name} tiene claves pero tipos/contenido inválido (q, a, faq_id, texto_para_vectorizar no deben estar vacíos; keywords debe ser lista). Item descartado: {str(item)[:150]}...")
                    invalid_items_count += 1
            else:
                # No es diccionario o le faltan claves requeridas
                missing_keys = EXPECTED_QA_KEYS - set(item.keys()) if isinstance(item, dict) else EXPECTED_QA_KEYS
                logger.warning(f"Item {i+1} en {file_path.name} no tiene la estructura Q&A esperada (faltan claves: {missing_keys}). Item descartado: {str(item)[:150]}...")
                invalid_items_count += 1
            # --- FIN BLOQUE VALIDACIÓN CORREGIDO ---

        if invalid_items_count > 0:
            logger.warning(f"Se descartaron {invalid_items_count} items inválidos del archivo {file_path.name}.")

        logger.debug(f"Cargados {len(valid_qa_list)} Q&As válidos desde {file_path.name}.")
        return valid_qa_list # Devuelve lista (puede estar vacía si todos fallaron)

    except FileNotFoundError:
        logger.error(f"Archivo JSON no encontrado (inesperado): {file_path}")
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
    """
    # (Sin cambios en esta función)
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
        return {}

    processed_files_count = 0
    error_files_count = 0
    total_valid_qas = 0

    if not json_files_found:
        logger.warning(f"No se encontraron archivos .json en el directorio: {base_directory}")
        return {}

    iterable_files = tqdm(json_files_found, desc="Cargando archivos JSON Q&A", unit="archivo") if tqdm_available else json_files_found

    for json_file_path in iterable_files:
        if hasattr(iterable_files, 'set_postfix_str'):
            iterable_files.set_postfix_str(f"{json_file_path.name[:30]}...", refresh=True)

        logger.debug(f"Procesando archivo JSON: {json_file_path}")
        try:
            relative_path_str = str(json_file_path.relative_to(base_directory))
        except ValueError:
            logger.warning(f"No se pudo calcular ruta relativa para {json_file_path} respecto a {base_directory}. Usando nombre de archivo.")
            relative_path_str = json_file_path.name

        # Llama a la función corregida load_single_json_file
        qa_list = load_single_json_file(json_file_path)

        if qa_list is not None: # Carga sin error fatal
            processed_files_count += 1
            if qa_list: # Lista no vacía
                all_qas[relative_path_str] = qa_list
                total_valid_qas += len(qa_list)
            else: # Lista vacía válida
                logger.info(f"Archivo '{relative_path_str}' procesado pero no contenía Q&As válidos o estaba vacío.")
        else: # Error fatal durante carga/validación
            logger.error(f"Fallo completo al cargar/validar archivo: '{relative_path_str}'. Excluido.")
            error_files_count += 1

    logger.info(f"Carga de datos finalizada.")
    logger.info(f"  Archivos JSON encontrados: {file_count}")
    logger.info(f"  Archivos JSON procesados (intentados): {processed_files_count}")
    logger.info(f"  Archivos con errores de carga/formato: {error_files_count}")
    logger.info(f"  Total Q&As válidos cargados: {total_valid_qas}")
    if error_files_count > 0:
        logger.warning("Algunos archivos JSON tuvieron errores y fueron omitidos.")

    return all_qas


# --- Bloque para pruebas rápidas (Necesitaría actualizarse para probar nueva estructura) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s: %(message)s')
    print("\n--- Probando data_loader (con estructura NUEVA - Bloque main necesita actualización) ---")
    # (El código de prueba necesitaría ser actualizado para crear JSONs
    # con la nueva estructura, incluyendo faq_id, categoria, texto_para_vectorizar,
    # y para verificar que se cargan correctamente)
    print("\n(Bloque de prueba __main__ necesitaría actualizarse para reflejar nuevos campos)")
    # --- Crear estructura de prueba temporal ---
    test_base_dir = Path("./_test_data_loader")
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)
    test_input_dir = test_base_dir / "input" / "json" / "SOAP_TXT"
    test_input_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de prueba creado en: {test_input_dir.resolve()}")
    # --- Crear Archivos de Prueba (EJEMPLO - ¡Adaptar con nueva estructura!) ---
    valid_data_new = [{"q": "P1", "a": "R1", "product": "A", "keywords": ["k1"], "faq_id": "v1_q0", "categoria": "C1", "texto_para_vectorizar":"P1R1"}]
    invalid_data_new = [{"q": "P2", "a": "R2", "product": "A", "keywords": ["k2"], "faq_id": "", "categoria": "C2", "texto_para_vectorizar":"P2R2"}] # faq_id vacío
    (test_input_dir / "valid_new.json").write_text(json.dumps(valid_data_new), encoding='utf-8')
    (test_input_dir / "invalid_new.json").write_text(json.dumps(invalid_data_new), encoding='utf-8')
    print("Archivos de prueba creados (ejemplo básico).")
    # Llamar a la función principal
    loaded_data = load_all_qas_from_directory(test_input_dir)
    print("\n--- Resultados de la Carga de Prueba ---")
    print(f"Diccionario de Q&As cargados: {loaded_data}")
    # Añadir aserciones básicas
    assert len(loaded_data) == 1, "Solo el archivo válido debería cargarse"
    assert "valid_new.json" in loaded_data
    assert "invalid_new.json" not in loaded_data # Porque su único item falló validación
    print("Pruebas básicas (ejemplo) completadas.")
    # --- Limpieza (opcional) ---
    # shutil.rmtree(test_base_dir, ignore_errors=True)