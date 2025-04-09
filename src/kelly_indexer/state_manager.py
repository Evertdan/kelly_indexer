# src/kelly_indexer/state_manager.py
# -*- coding: utf-8 -*-

"""
Módulo para gestionar el estado de la indexación en Qdrant.
MODIFICADO para usar faq_id como ID principal y hash de texto_para_vectorizar.
Incluye correcciones para MyPy.
"""

import json
import logging
# import uuid # Ya no se necesita para el ID principal
import hashlib
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set # Asegurar que Set esté importado

logger = logging.getLogger(__name__)

# --- Constantes ---
# Namespace UUID ya no es necesario para el ID principal
# QDRANT_POINT_NAMESPACE = uuid.UUID('f8a7c9a1-e45f-4e6d-8f3c-1b7a2b9e8d0f') # ¡CAMBIA ESTO!

STATE_FILE_VERSION = "1.1" # Versión incrementada por cambio de lógica/hash

# --- Funciones de Hashing e IDs ---

def generate_content_hash(content: str) -> str:
    """Genera un hash SHA-256 para una cadena de texto."""
    # (Sin cambios en esta función)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# La función generate_qa_uuid ya no se usa para el ID principal
# def generate_qa_uuid(question: str, source_file_rel_path: str) -> str:
#     (...)

# --- Funciones de Carga y Guardado de Estado ---

def load_state(state_file_path: Path) -> Dict[str, Any]:
    """
    Carga el estado de indexación desde un archivo JSON.
    """
    # (Sin cambios funcionales necesarios en load_state)
    initial_state: Dict[str, Any] = {
        "version": STATE_FILE_VERSION,
        "last_run_utc": None,
        "indexed_points": {} # Claves son faq_id
    }
    if not state_file_path.is_file():
        logger.warning(f"Archivo de estado no encontrado en {state_file_path}. Se asumirá estado inicial.")
        return initial_state

    logger.info(f"Cargando estado desde: {state_file_path}")
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        if not isinstance(state, dict):
            raise ValueError("El archivo de estado no contiene un objeto JSON.")
        if "indexed_points" not in state or not isinstance(state.get("indexed_points"), dict):
            logger.warning("Clave 'indexed_points' no encontrada o inválida. Reiniciando puntos.")
            state["indexed_points"] = {}
        if "version" not in state:
            logger.warning("Clave 'version' no encontrada. Añadiendo versión por defecto.")
            state["version"] = STATE_FILE_VERSION
        elif state.get("version") != STATE_FILE_VERSION:
            logger.warning(f"Versión del archivo de estado ({state.get('version')}) NO coincide con la esperada ({STATE_FILE_VERSION}). ¡Se forzará reproceso!")
        if "last_run_utc" not in state:
            state["last_run_utc"] = None

        logger.info(f"Estado cargado. {len(state['indexed_points'])} puntos registrados previamente (claves son faq_id).")
        return state

    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON en estado {state_file_path}: {e}. Usando estado inicial.")
        return initial_state.copy()
    except Exception as e:
        logger.exception(f"Error inesperado al cargar estado {state_file_path}: {e}. Usando estado inicial.")
        return initial_state.copy()

# MODIFICADO: Aceptar directamente current_points_details
def save_state(state_file_path: Path, current_points_details: Dict[str, Dict]) -> bool:
    """
    Guarda el diccionario de detalles de puntos actuales en un archivo JSON atómico.
    """
    # (Sin cambios funcionales necesarios en save_state respecto a la versión modificada anterior)
    logger.info(f"Guardando estado actualizado en: {state_file_path}...")
    if not isinstance(current_points_details, dict):
        logger.error("Intento de guardar estado inválido: No se proporcionó un diccionario de detalles.")
        return False

    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"No se pudo crear el directorio padre '{state_file_path.parent}': {e}")
        return False

    state_to_save: Dict[str, Any] = {
        "version": STATE_FILE_VERSION,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "indexed_points": current_points_details
    }

    temp_path_obj: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, dir=state_file_path.parent, encoding='utf-8', suffix='.tmp') as temp_f:
            json.dump(state_to_save, temp_f, indent=2, ensure_ascii=False)
            temp_path = temp_f.name
            temp_path_obj = Path(temp_path)

        os.replace(temp_path, state_file_path)
        logger.info(f"Estado guardado exitosamente. {len(state_to_save['indexed_points'])} puntos registrados.")
        return True

    except Exception as e:
        logger.exception(f"Error fatal al guardar el archivo de estado {state_file_path}: {e}")
        if temp_path_obj and temp_path_obj.exists():
            try:
                temp_path_obj.unlink()
                logger.info(f"Archivo temporal de estado eliminado: {temp_path_obj}")
            except OSError as unlink_e:
                logger.error(f"No se pudo eliminar archivo temporal {temp_path_obj}: {unlink_e}")
        return False

# --- Lógica de Comparación y Actualización de Estado (MODIFICADA) ---

def calculate_diff(
    current_qas_map: Dict[str, List[Dict]], # {rel_path -> [qa_dict con faq_id, etc.]}
    previous_indexed_points: Dict[str, Dict] # {faq_id -> {details con content_hash}}
) -> Tuple[List[Dict], List[str], Dict[str, Dict]]:
    """
    Compara los Q&A actuales (con faq_id) con el estado anterior
    para determinar qué procesar (upsert) o eliminar.

    MODIFICADO: Usa 'faq_id' como clave y hash de 'texto_para_vectorizar'.
                Añade '_source_file' a los items a procesar para uso posterior.
                Incluye corrección para MyPy [union-attr].

    Args:
        current_qas_map: Diccionario {ruta_relativa -> lista_de_Q&As_actuales}.
                         Se espera que cada Q&A dict contenga 'faq_id' y 'texto_para_vectorizar'.
        previous_indexed_points: Diccionario 'indexed_points' del estado anterior.

    Returns:
        Una tupla:
        - qas_to_process (List[Dict]): Lista de diccionarios Q&A originales (nuevos/modificados)
                                       enriquecidos SOLAMENTE con '_source_file'.
        - ids_to_delete (List[str]): Lista de faq_ids que ya no están en la fuente.
        - current_points_details (Dict[str, Dict]): Mapa faq_id -> detalles
                                                      para TODOS los Q&A válidos actuales.
                                                      (Formato: {'source_file': str, 'content_hash': str})
    """
    logger.info("Calculando diferencias usando faq_id y hash de contenido...")
    qas_to_process: List[Dict] = []
    current_points_details: Dict[str, Dict] = {}
    current_point_ids: Set[str] = set() # Conjunto de faq_ids encontrados ahora

    total_qas_evaluated = 0

    # 1. Iterar sobre los Q&A actuales cargados por data_loader
    for file_rel_path, qa_list in current_qas_map.items():
        if not isinstance(qa_list, list):
            logger.warning(f"Valor no esperado para archivo '{file_rel_path}'. Saltando.")
            continue

        for qa_item in qa_list:
            total_qas_evaluated += 1

            # --- Usar faq_id y texto_para_vectorizar ---
            point_id = qa_item.get('faq_id')
            text_to_hash = qa_item.get('texto_para_vectorizar')

            # Validar campos necesarios
            if not point_id or not isinstance(point_id, str) or not point_id.strip():
                logger.warning(f"Q&A item sin 'faq_id' válido en archivo {file_rel_path}. Saltando: {str(qa_item)[:100]}...")
                continue
            if not text_to_hash or not isinstance(text_to_hash, str) or not text_to_hash.strip():
                logger.warning(f"Q&A item ({point_id}) sin 'texto_para_vectorizar' válido en archivo {file_rel_path}. Saltando.")
                continue

            # Generar hash del contenido relevante
            content_hash = generate_content_hash(text_to_hash)
            current_point_ids.add(point_id)

            # Guardar detalles actuales para el estado final
            current_points_details[point_id] = {
                "source_file": file_rel_path,
                "content_hash": content_hash
            }

            # Comparar con estado previo usando faq_id
            previous_entry = previous_indexed_points.get(point_id)

            # Determinar si es nuevo o modificado
            is_new = not previous_entry
            # CORRECCIÓN MYPY: Añadir chequeo explícito 'is not None' aunque lógicamente ya está cubierto.
            is_modified = not is_new and previous_entry is not None and previous_entry.get('content_hash') != content_hash

            if is_new:
                logger.debug(f"Nuevo Q&A: ID={point_id} (Archivo={file_rel_path})")
                # NUEVO: Añadir _source_file para que index_qdrant.py lo use
                qa_item_copy = qa_item.copy() # Trabajar con copia para no modificar original en map
                qa_item_copy['_source_file'] = file_rel_path
                qas_to_process.append(qa_item_copy)
            elif is_modified:
                logger.info(f"Q&A Modificado: ID={point_id} (Archivo={file_rel_path}) (Hash: {previous_entry.get('content_hash', 'N/A')[:8]}... -> {content_hash[:8]}...)")
                # NUEVO: Añadir _source_file para que index_qdrant.py lo use
                qa_item_copy = qa_item.copy()
                qa_item_copy['_source_file'] = file_rel_path
                qas_to_process.append(qa_item_copy)
            else:
                # Sin cambios
                logger.debug(f"Q&A sin cambios: ID={point_id} (Archivo={file_rel_path})")
                pass # No añadir

    # 2. Identificar IDs a eliminar
    previous_ids = set(previous_indexed_points.keys())
    ids_to_delete = list(previous_ids - current_point_ids)

    logger.info(f"Evaluados {total_qas_evaluated} Q&As actuales.")
    logger.info(f"Diff calculado: {len(qas_to_process)} a procesar (nuevos/modificados), {len(ids_to_delete)} a eliminar.")

    return qas_to_process, ids_to_delete, current_points_details


# --- Bloque para pruebas rápidas (Necesita actualizarse significativamente) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- Probando State Manager (MODIFICADO - Bloque de prueba necesita actualización) ---")
    print("\n(El bloque de prueba __main__ necesitaría ser reescrito para usar 'faq_id'")
    print(" y 'texto_para_vectorizar' en los datos de prueba y verificar 'content_hash')")
    # (Código de prueba original comentado/eliminado)