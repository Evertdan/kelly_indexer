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
# CORRECCIÓN: Añadir importaciones de typing necesarias
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# --- Constantes ---
# Namespace UUID para generar IDs deterministas v5.
# ¡Genera uno propio y mantenlo constante para tu proyecto!
# Ejemplo: python -c "import uuid; print(uuid.uuid4())"
QDRANT_POINT_NAMESPACE = uuid.UUID('f8a7c9a1-e45f-4e6d-8f3c-1b7a2b9e8d0f') # ¡CAMBIA ESTO!

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
    # Normalizar separadores de ruta para consistencia entre OS
    normalized_path = source_file_rel_path.replace("\\", "/")
    unique_string = f"question:{question}|source:{normalized_path}"
    # Usar el namespace definido para generar el UUID v5
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
    # CORRECCIÓN: Añadir tipo explícito a initial_state
    initial_state: Dict[str, Any] = {
        "version": STATE_FILE_VERSION,
        "last_run_utc": None,
        "indexed_points": {}
    }
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
        # Asegurar que las claves principales existan, aunque estén vacías/None
        if "indexed_points" not in state or not isinstance(state.get("indexed_points"), dict):
             logger.warning("Clave 'indexed_points' no encontrada o inválida en el estado. Reiniciando puntos indexados.")
             state["indexed_points"] = {}
        if "version" not in state:
             logger.warning("Clave 'version' no encontrada en el estado. Añadiendo versión por defecto.")
             state["version"] = STATE_FILE_VERSION
        elif state.get("version") != STATE_FILE_VERSION:
             logger.warning(f"Versión del archivo de estado ({state.get('version')}) no coincide con la esperada ({STATE_FILE_VERSION}). Puede haber incompatibilidades.")
        if "last_run_utc" not in state:
             state["last_run_utc"] = None # Asegurar que exista

        logger.info(f"Estado cargado. {len(state['indexed_points'])} puntos indexados previamente.")
        return state

    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON en el archivo de estado {state_file_path}: {e}. Se usará estado inicial.")
        return initial_state.copy() # Devolver copia del inicial
    except Exception as e:
        logger.exception(f"Error inesperado al cargar el archivo de estado {state_file_path}: {e}. Se usará estado inicial.")
        return initial_state.copy() # Devolver copia del inicial

def save_state(state_file_path: Path, current_state: Dict[str, Any]) -> bool:
    """
    Guarda el diccionario de estado actual en un archivo JSON de forma atómica.

    Actualiza 'last_run_utc' y 'version' antes de guardar.
    Utiliza escritura a archivo temporal y renombrado para atomicidad.

    Args:
        state_file_path: Ruta al archivo JSON donde guardar el estado.
        current_state: El diccionario de estado a guardar (se espera que contenga al menos 'indexed_points').

    Returns:
        True si se guardó exitosamente, False en caso contrario.
    """
    logger.info(f"Guardando estado actualizado en: {state_file_path}...")
    if not isinstance(current_state.get("indexed_points"), dict):
         logger.error("Intento de guardar estado inválido: 'indexed_points' no es un diccionario.")
         return False

    # Crear directorio padre si no existe
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
         logger.error(f"No se pudo crear el directorio padre para el archivo de estado '{state_file_path.parent}': {e}")
         return False

    # Preparar estado a guardar
    state_to_save: Dict[str, Any] = {
        "version": STATE_FILE_VERSION,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        # Asegurarse de guardar solo los puntos indexados actuales
        "indexed_points": current_state.get("indexed_points", {})
    }

    # Escritura atómica
    temp_path_obj: Optional[Path] = None
    try:
        # Crear archivo temporal en el mismo directorio que el final
        with tempfile.NamedTemporaryFile('w', delete=False, dir=state_file_path.parent, encoding='utf-8', suffix='.tmp') as temp_f:
            json.dump(state_to_save, temp_f, indent=2, ensure_ascii=False)
            temp_path = temp_f.name # Guardar nombre temporal como string
            temp_path_obj = Path(temp_path) # Y como objeto Path

        # Renombrar atómicamente (reemplaza si existe)
        os.replace(temp_path, state_file_path)
        logger.info(f"Estado guardado exitosamente. {len(state_to_save['indexed_points'])} puntos registrados.")
        return True

    except Exception as e:
        logger.exception(f"Error fatal al guardar el archivo de estado {state_file_path}: {e}")
        # Intentar limpiar el archivo temporal si aún existe y falló el renombrado/escritura
        if temp_path_obj and temp_path_obj.exists():
             try:
                 temp_path_obj.unlink()
                 logger.info(f"Archivo temporal de estado eliminado: {temp_path_obj}")
             except OSError as unlink_e:
                 logger.error(f"No se pudo eliminar el archivo temporal de estado {temp_path_obj}: {unlink_e}")
        return False

# --- Lógica de Comparación y Actualización de Estado ---

def calculate_diff(
    current_qas_map: Dict[str, List[Dict]], # {rel_path -> [qa_dict]}
    previous_indexed_points: Dict[str, Dict] # {point_id -> {details}}
) -> Tuple[List[Dict], List[str], Dict[str, Dict]]:
    """
    Compara los Q&A actuales encontrados en los archivos fuente con el estado
    anteriormente indexado para determinar qué necesita ser procesado o eliminado.

    Args:
        current_qas_map: Diccionario mapeando ruta_relativa -> lista_de_Q&As_actuales.
        previous_indexed_points: Diccionario 'indexed_points' del estado anterior.

    Returns:
        Una tupla:
        - qas_to_process (List[Dict]): Lista de diccionarios Q&A (nuevos/modificados),
                                       enriquecidos con '_id', '_question_hash', '_source_file'.
        - ids_to_delete (List[str]): Lista de point_ids que ya no están en la fuente.
        - current_points_details (Dict[str, Dict]): Mapa point_id -> detalles
                                                   para TODOS los Q&A válidos actuales.
                                                   (Formato: {'source_file': str, 'question_hash': str})
    """
    logger.info("Calculando diferencias entre estado actual y anterior...")
    qas_to_process: List[Dict] = []
    current_points_details: Dict[str, Dict] = {} # Mapa id -> {source_file, question_hash}
    current_point_ids: Set[str] = set() # Conjunto de IDs encontrados ahora

    total_qas_evaluated = 0

    # 1. Iterar sobre los Q&A actuales
    for file_rel_path, qa_list in current_qas_map.items():
        if not isinstance(qa_list, list): # Chequeo extra por si data_loader falla
             logger.warning(f"Se encontró un valor no esperado (no lista) para el archivo '{file_rel_path}' en current_qas_map. Saltando archivo.")
             continue

        for qa_item in qa_list:
            total_qas_evaluated += 1
            question = qa_item.get('q')
            if not question or not isinstance(question, str): # Saltar si falta pregunta o no es string
                logger.warning(f"Q&A item sin pregunta 'q' válida en archivo {file_rel_path}. Saltando item: {str(qa_item)[:100]}...")
                continue

            point_id = generate_qa_uuid(question, file_rel_path)
            q_hash = generate_content_hash(question)
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
                # Enriquecer el dict original con info necesaria para procesamiento
                qa_item['_id'] = point_id
                qa_item['_question_hash'] = q_hash
                qa_item['_source_file'] = file_rel_path
                qas_to_process.append(qa_item)
            elif previous_entry.get('question_hash') != q_hash:
                # Ha cambiado
                logger.info(f"Q&A Modificado detectado: ID={point_id}, Archivo={file_rel_path} (Hash anterior: {previous_entry.get('question_hash', 'N/A')[:8]}... vs Nuevo: {q_hash[:8]}...)")
                qa_item['_id'] = point_id
                qa_item['_question_hash'] = q_hash
                qa_item['_source_file'] = file_rel_path
                qas_to_process.append(qa_item)
            else:
                # Sin cambios
                logger.debug(f"Q&A sin cambios: ID={point_id}, Archivo={file_rel_path}")
                pass # No añadir a qas_to_process

    # 2. Identificar IDs a eliminar
    previous_ids = set(previous_indexed_points.keys())
    ids_to_delete = list(previous_ids - current_point_ids)

    logger.info(f"Evaluados {total_qas_evaluated} Q&As actuales.")
    logger.info(f"Diff calculado: {len(qas_to_process)} a procesar (nuevos/modificados), {len(ids_to_delete)} a eliminar.")

    return qas_to_process, ids_to_delete, current_points_details


# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    # Configurar logging básico para la prueba
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    print("--- Probando State Manager ---")

    # Prueba de generación de ID y Hash
    q1 = "¿Cómo funciona?"
    f1 = "docs/faq.json"
    id1 = generate_qa_uuid(q1, f1)
    h1 = generate_content_hash(q1)
    print(f"Pregunta: '{q1}' en '{f1}'")
    print(f"  ID UUIDv5: {id1}")
    print(f"  Hash SHA256: {h1[:12]}...")

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

    # Prueba de carga/guardado
    print("\nProbando carga/guardado de estado...")
    temp_dir = Path("./_test_state_manager")
    temp_dir.mkdir(exist_ok=True)
    state_path = temp_dir / "test_state.json"
    if state_path.exists(): state_path.unlink() # Limpiar antes de probar

    # Cargar estado inicial
    initial_s = load_state(state_path)
    print(f"Estado inicial cargado: {initial_s}")
    assert initial_s["indexed_points"] == {}

    # Crear estado de prueba y guardarlo
    test_state_to_save = {
        # No necesitamos poner version/last_run aquí, save_state los añade/actualiza
        "indexed_points": {
            id1: {"source_file": f1, "question_hash": h1},
            id2: {"source_file": f2, "question_hash": h2}
        }
    }
    save_ok = save_state(state_path, test_state_to_save)
    print(f"Guardado de estado: {'OK' if save_ok else 'FALLÓ'}")
    assert save_ok
    assert state_path.is_file()

    # Cargar estado guardado
    loaded_s = load_state(state_path)
    print(f"Estado guardado y recargado: {loaded_s}")
    assert loaded_s["version"] == STATE_FILE_VERSION
    assert loaded_s["last_run_utc"] is not None
    assert id1 in loaded_s["indexed_points"]
    assert loaded_s["indexed_points"][id1]["question_hash"] == h1
    assert len(loaded_s["indexed_points"]) == 2

    # Prueba de diff
    print("\nProbando cálculo de diff...")
    # Simular datos actuales: q1 sin cambios, q2 se elimina, q3 (nuevo hash) en f1, q4 nuevo
    q4 = "Nueva pregunta"
    f4 = "docs/nuevo.json"
    id4 = generate_qa_uuid(q4, f4)
    h4 = generate_content_hash(q4)

    current_data_map = {
        f1: [ # Archivo f1 ahora tiene q1 (sin cambios) y q3 (modificada/nueva respecto a estado previo)
            {"q": q1, "a":"R1", "product":"P", "keywords":[]},
            {"q": q3, "a":"R3", "product":"P", "keywords":[]}
            ],
        f4: [ # Archivo f4 es nuevo
            {"q": q4, "a":"R4", "product":"P", "keywords":[]}
            ]
        # Archivo f2 ya no existe en los datos actuales
    }
    prev_points_dict = loaded_s['indexed_points'] # Estado previo tenía id1 y id2

    upsert_list, delete_list, current_details_map = calculate_diff(current_data_map, prev_points_dict)

    print(f"  Q&As a procesar (upsert): {len(upsert_list)} -> {[item['_id'][:8] for item in upsert_list]}")
    print(f"  IDs a eliminar: {len(delete_list)} -> {[pid[:8] for pid in delete_list]}")
    print(f"  Detalles actuales: {len(current_details_map)} puntos")

    # IDs esperados:
    id1_current = generate_qa_uuid(q1, f1) # Mismo que id1
    id3_current = generate_qa_uuid(q3, f1) # Mismo que id3
    id4_current = generate_qa_uuid(q4, f4) # Mismo que id4

    # Verificar upsert: deben estar q3 y q4
    upsert_ids_set = {item['_id'] for item in upsert_list}
    assert id3_current in upsert_ids_set
    assert id4_current in upsert_ids_set
    assert id1_current not in upsert_ids_set # q1 no cambió

    # Verificar delete: debe estar id2 (pregunta q2 del archivo f2) que ya no está
    assert id2 in delete_list
    assert id1 not in delete_list # id1 (pregunta q1 del archivo f1) sigue presente y sin cambios

    # Verificar detalles actuales: deben estar todos los Q&As actuales (q1, q3, q4)
    assert id1_current in current_details_map
    assert id3_current in current_details_map
    assert id4_current in current_details_map
    assert len(current_details_map) == 3
    assert current_details_map[id1_current]['question_hash'] == h1
    assert current_details_map[id3_current]['question_hash'] == h3

    print("\nPruebas básicas de state_manager completadas.")
    # Limpieza opcional
    # try: shutil.rmtree(temp_dir)
    # except OSError: pass # Importar shutil