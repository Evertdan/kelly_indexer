# src/kelly_indexer/state_manager.py
# -*- coding: utf-8 -*-

"""
Módulo para gestionar el estado de la indexación en Qdrant.

Mantiene un registro de qué pares Q&A han sido indexados para
evitar duplicados y manejar actualizaciones/eliminaciones.
Genera IDs deterministas y hashes de contenido.
"""

import json
import logging
import uuid
import hashlib
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# --- Constantes ---
# Namespace UUID para generar IDs deterministas v5.
# ¡Genera uno propio y mantenlo constante para tu proyecto! Puedes usar:
# python -c "import uuid; print(uuid.uuid4())" para generar un namespace aleatorio.
# O usa un UUID conocido si prefieres. Ejemplo:
QDRANT_POINT_NAMESPACE = uuid.UUID('f8a7c9a1-e45f-4e6d-8f3c-1b7a2b9e8d0f') # ¡CAMBIA ESTO POR UNO PROPIO!

STATE_FILE_VERSION = "1.0" # Versión del formato del archivo de estado

# --- Funciones de Hashing e IDs ---

def generate_content_hash(content: str) -> str:
    """Genera un hash SHA-256 para una cadena de texto."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def generate_qa_uuid(question: str, source_file_rel_path: str) -> str:
    """
    Genera un UUID v5 determinista para un par Q&A.

    Se basa en el contenido de la pregunta y su archivo de origen
    para asegurar unicidad incluso si la misma pregunta aparece en
    múltiples archivos. Usa un namespace fijo.

    Args:
        question: El texto de la pregunta.
        source_file_rel_path: La ruta relativa del archivo JSON de origen.

    Returns:
        El UUID v5 como string.
    """
    # Crear una cadena única combinando la pregunta y la ruta normalizada
    # Normalizar separadores de ruta para consistencia entre OS
    normalized_path = source_file_rel_path.replace("\\", "/")
    unique_string = f"question:{question}|source:{normalized_path}"
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, unique_string))

# --- Funciones de Carga y Guardado de Estado ---

def load_state(state_file_path: Path) -> Dict[str, Any]:
    """
    Carga el estado de indexación desde un archivo JSON.

    Si el archivo no existe o está corrupto, devuelve un estado vacío inicial.

    Args:
        state_file_path: Ruta al archivo JSON de estado.

    Returns:
        Un diccionario representando el estado cargado o un estado inicial vacío.
        Formato esperado: {"version": "...", "last_run_utc": "...", "indexed_points": {}}
    """
    initial_state = {"version": STATE_FILE_VERSION, "last_run_utc": None, "indexed_points": {}}
    if not state_file_path.is_file():
        logger.warning(f"Archivo de estado no encontrado en {state_file_path}. Se asumirá estado inicial.")
        return initial_state

    logger.info(f"Cargando estado desde: {state_file_path}")
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # Validaciones básicas de estructura
        if not isinstance(state, dict):
             raise ValueError("El archivo de estado no contiene un objeto JSON.")
        if "indexed_points" not in state or not isinstance(state["indexed_points"], dict):
             logger.warning("Clave 'indexed_points' no encontrada o inválida en el estado. Reiniciando puntos indexados.")
             state["indexed_points"] = {}
        if "version" not in state or state.get("version") != STATE_FILE_VERSION:
             logger.warning(f"Versión del archivo de estado ({state.get('version')}) no coincide con la esperada ({STATE_FILE_VERSION}). Puede haber incompatibilidades.")
             # Podrías añadir lógica de migración aquí si fuera necesario

        logger.info(f"Estado cargado. {len(state['indexed_points'])} puntos indexados previamente.")
        return state

    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON en el archivo de estado {state_file_path}: {e}. Se usará estado inicial.")
        return initial_state
    except Exception as e:
        logger.exception(f"Error inesperado al cargar el archivo de estado {state_file_path}: {e}. Se usará estado inicial.")
        return initial_state

def save_state(state_file_path: Path, current_state: Dict[str, Any]) -> bool:
    """
    Guarda el diccionario de estado actual en un archivo JSON de forma atómica.

    Actualiza 'last_run_utc' y 'version' antes de guardar.
    Utiliza escritura a archivo temporal y renombrado para atomicidad.

    Args:
        state_file_path: Ruta al archivo JSON donde guardar el estado.
        current_state: El diccionario de estado a guardar.

    Returns:
        True si se guardó exitosamente, False en caso contrario.
    """
    logger.info(f"Guardando estado actualizado en: {state_file_path}...")
    try:
        # Asegurar directorio padre
        state_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Actualizar metadata del estado
        state_to_save = current_state.copy() # Trabajar con una copia
        state_to_save["last_run_utc"] = datetime.now(timezone.utc).isoformat()
        state_to_save["version"] = STATE_FILE_VERSION

        # Escritura atómica: Escribir a un archivo temporal en el mismo directorio
        # y luego renombrarlo al nombre final.
        # tempfile.NamedTemporaryFile crea el archivo de forma segura
        with tempfile.NamedTemporaryFile('w', delete=False, dir=state_file_path.parent, encoding='utf-8', suffix='.tmp') as temp_f:
            json.dump(state_to_save, temp_f, indent=2, ensure_ascii=False)
            temp_path = temp_f.name # Guardar nombre temporal

        # Renombrar atómicamente el archivo temporal al archivo final
        # os.replace es atómico en la mayoría de los sistemas POSIX y Windows modernos
        os.replace(temp_path, state_file_path) # Reemplaza el archivo si ya existe

        logger.info(f"Estado guardado exitosamente. {len(state_to_save.get('indexed_points', {}))} puntos registrados.")
        return True

    except Exception as e:
        logger.exception(f"Error fatal al guardar el archivo de estado {state_file_path}: {e}")
        # Intentar limpiar el archivo temporal si aún existe y falló el renombrado/escritura
        if 'temp_path' in locals() and Path(temp_path).exists():
             try: Path(temp_path).unlink()
             except OSError: pass
        return False

# --- Lógica de Comparación y Actualización de Estado ---

def calculate_diff(
    current_qas_map: Dict[str, List[Dict]],
    previous_indexed_points: Dict[str, Dict]
) -> Tuple[List[Dict], List[str], Dict[str, Dict]]:
    """
    Compara los Q&A actuales encontrados en los archivos fuente con el estado
    anteriormente indexado para determinar qué necesita ser procesado o eliminado.

    Args:
        current_qas_map: Diccionario mapeando ruta_relativa -> lista_de_Q&As_actuales.
                         (Generado por data_loader.load_all_qas_from_directory)
        previous_indexed_points: Diccionario mapeando point_id -> detalles_punto
                                  (Obtenido de previous_state['indexed_points'])

    Returns:
        Una tupla:
        - qas_to_process (List[Dict]): Lista de diccionarios Q&A que son nuevos
                                       o modificados, enriquecidos con '_id',
                                       '_question_hash' y '_source_file'.
        - ids_to_delete (List[str]): Lista de point_ids que estaban en el estado
                                     anterior pero ya no se encuentran en la fuente.
        - current_points_details (Dict[str, Dict]): Mapa de point_id -> detalles
                                                   para TODOS los Q&A válidos
                                                   encontrados en la ejecución actual.
                                                   (Formato: {'source_file': str, 'question_hash': str})
    """
    logger.info("Calculando diferencias entre estado actual y anterior...")
    qas_to_process: List[Dict] = []
    ids_to_delete: List[str] = []
    current_points_details: Dict[str, Dict] = {} # Mapa id -> {source_file, question_hash}
    current_point_ids: Set[str] = set() # Conjunto de IDs encontrados ahora

    total_qas_evaluated = 0

    # 1. Iterar sobre los Q&A encontrados en los archivos actuales
    for file_rel_path, qa_list in current_qas_map.items():
        for qa_item in qa_list:
            total_qas_evaluated += 1
            if not qa_item.get('q'): # Saltar si falta la pregunta
                logger.warning(f"Q&A item sin pregunta 'q' en archivo {file_rel_path}. Saltando item.")
                continue

            point_id = generate_qa_uuid(qa_item['q'], file_rel_path)
            q_hash = generate_content_hash(qa_item['q'])
            current_point_ids.add(point_id)

            # Guardar detalles actuales
            current_points_details[point_id] = {
                "source_file": file_rel_path,
                "question_hash": q_hash
            }

            # Comparar con estado previo
            previous_entry = previous_indexed_points.get(point_id)
            if not previous_entry:
                # Es nuevo
                logger.debug(f"Nuevo Q&A: ID={point_id}, Archivo={file_rel_path}")
                qa_item['_id'] = point_id
                qa_item['_question_hash'] = q_hash
                qa_item['_source_file'] = file_rel_path
                qas_to_process.append(qa_item)
            elif previous_entry.get('question_hash') != q_hash:
                # Ha cambiado (el hash de la pregunta es diferente)
                logger.info(f"Q&A Modificado: ID={point_id}, Archivo={file_rel_path}")
                qa_item['_id'] = point_id
                qa_item['_question_hash'] = q_hash
                qa_item['_source_file'] = file_rel_path
                qas_to_process.append(qa_item)
            else:
                # Sin cambios
                logger.debug(f"Q&A sin cambios: ID={point_id}, Archivo={file_rel_path}")
                pass

    # 2. Identificar IDs a eliminar (estaban antes, no ahora)
    previous_ids = set(previous_indexed_points.keys())
    ids_to_delete = list(previous_ids - current_point_ids)

    logger.info(f"Evaluados {total_qas_evaluated} Q&As actuales.")
    logger.info(f"Diff calculado: {len(qas_to_process)} a procesar (nuevos/modificados), {len(ids_to_delete)} a eliminar.")

    return qas_to_process, ids_to_delete, current_points_details


# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    print("--- Probando State Manager ---")

    # Prueba de generación de ID y Hash
    q1 = "¿Cómo funciona?"
    f1 = "docs/faq.json"
    id1 = generate_qa_uuid(q1, f1)
    h1 = generate_content_hash(q1)
    print(f"Pregunta: '{q1}' en '{f1}'")
    print(f"  ID UUIDv5: {id1}")
    print(f"  Hash SHA256: {h1[:12]}...") # Mostrar solo el inicio

    q2 = "¿Cómo funciona?" # Misma pregunta
    f2 = "docs/manual.json" # Diferente archivo
    id2 = generate_qa_uuid(q2, f2)
    h2 = generate_content_hash(q2)
    print(f"Pregunta: '{q2}' en '{f2}'")
    print(f"  ID UUIDv5: {id2}") # ID debe ser diferente de id1
    print(f"  Hash SHA256: {h2[:12]}...") # Hash debe ser igual a h1
    assert id1 != id2
    assert h1 == h2

    q3 = "¿Cómo funciona esto?" # Pregunta diferente
    f3 = "docs/faq.json" # Mismo archivo que f1
    id3 = generate_qa_uuid(q3, f3)
    h3 = generate_content_hash(q3)
    print(f"Pregunta: '{q3}' en '{f3}'")
    print(f"  ID UUIDv5: {id3}") # ID debe ser diferente de id1 e id2
    print(f"  Hash SHA256: {h3[:12]}...") # Hash debe ser diferente de h1/h2
    assert id3 != id1 and id3 != id2
    assert h3 != h1

    # Prueba de carga/guardado (crea archivo temporal)
    print("\nProbando carga/guardado de estado...")
    temp_dir = Path("./_test_state_manager")
    temp_dir.mkdir(exist_ok=True)
    state_path = temp_dir / "test_state.json"
    if state_path.exists(): state_path.unlink()

    # Cargar estado inicial (archivo no existe)
    initial_s = load_state(state_path)
    print(f"Estado inicial cargado: {initial_s}")
    assert initial_s["indexed_points"] == {}
    assert initial_s["last_run_utc"] is None

    # Crear estado de prueba y guardarlo
    current_s = {
        "version": "old", # Será actualizado por save_state
        "last_run_utc": "ayer", # Será actualizado
        "indexed_points": {
            id1: {"source_file": f1, "question_hash": h1},
            id2: {"source_file": f2, "question_hash": h2}
        }
    }
    save_ok = save_state(state_path, current_s)
    print(f"Guardado de estado: {'OK' if save_ok else 'FALLÓ'}")
    assert save_ok

    # Cargar estado guardado
    loaded_s = load_state(state_path)
    print(f"Estado guardado y recargado: {loaded_s}")
    assert loaded_s["version"] == STATE_FILE_VERSION
    assert loaded_s["last_run_utc"] is not None
    assert id1 in loaded_s["indexed_points"]
    assert loaded_s["indexed_points"][id1]["question_hash"] == h1
    assert len(loaded_s["indexed_points"]) == 2

    # Prueba de diff (simulado)
    print("\nProbando cálculo de diff...")
    current_data = {
        f1: [{"q": q1, "a":"...", "product":"P", "keywords":[]}], # Sin cambios
        f2: [{"q": "Pregunta modificada?", "a":"...", "product":"P", "keywords":[]}], # Pregunta cambió -> hash cambia
        "docs/nuevo.json": [{"q":"Nueva pregunta", "a":"...", "product":"P", "keywords":[]}] # Nuevo archivo/pregunta
    }
    # El estado previo tiene id1 (q1, f1) y id2 (q2, f2)
    prev_points = loaded_s['indexed_points']

    upsert_list, delete_list, current_details = calculate_diff(current_data, prev_points)

    print(f"  Q&As a procesar (upsert): {len(upsert_list)}")
    print(f"  IDs a eliminar: {len(delete_list)}")
    print(f"  Detalles actuales: {len(current_details)} puntos")

    # IDs esperados:
    id1_current = generate_qa_uuid(q1, f1) # Debería ser igual a id1
    id2_mod = generate_qa_uuid("Pregunta modificada?", f2) # Diferente a id2
    id_nuevo = generate_qa_uuid("Nueva pregunta", "docs/nuevo.json")

    # Verificar upsert: deben estar el modificado y el nuevo
    upsert_ids = {item['_id'] for item in upsert_list}
    assert id2_mod in upsert_ids
    assert id_nuevo in upsert_ids
    assert id1_current not in upsert_ids # No cambió

    # Verificar delete: debe estar el id2 original (q2, f2) que ya no existe
    assert id2 in delete_list
    assert id1 not in delete_list # id1 (q1, f1) todavía existe

    # Verificar detalles actuales: deben estar todos los puntos actuales
    assert id1_current in current_details
    assert id2_mod in current_details
    assert id_nuevo in current_details
    assert current_details[id1_current]['question_hash'] == h1
    assert current_details[id2_mod]['question_hash'] != h2


    print("\nPruebas básicas de state_manager completadas.")
    # Limpieza
    # shutil.rmtree(temp_dir, ignore_errors=True) # Importar shutil