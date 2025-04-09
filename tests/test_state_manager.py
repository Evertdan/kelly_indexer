# tests/test_state_manager.py
# -*- coding: utf-8 -*-

"""
Pruebas unitarias para el módulo state_manager de Kelly Indexer.
MODIFICADO para probar la nueva lógica basada en faq_id y content_hash.
"""

import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime

# Importar el módulo bajo prueba
# Asumiendo que pytest corre desde la raíz y src está en PYTHONPATH
try:
    from kelly_indexer import state_manager
except ImportError:
     pytest.fail("No se pudo importar 'state_manager' desde 'kelly_indexer'. Asegúrate de la estructura y PYTHONPATH.", pytrace=False)

# --- Datos de Prueba ---

# Simular datos como los cargaría data_loader (lista de dicts por archivo)
# Asegúrate de que estos datos incluyan los campos nuevos
SAMPLE_QA_ITEM_1_V1 = {
    "q": "Pregunta 1", "a": "Respuesta 1", "product": "P1", "categoria": "C1",
    "keywords": ["k1"], "faq_id": "DOC1_q0",
    "texto_para_vectorizar": "Pregunta: Pregunta 1 Respuesta: Respuesta 1"
}
SAMPLE_QA_ITEM_1_V2 = { # Misma faq_id, pero contenido cambiado
    "q": "Pregunta 1 Modificada", "a": "Respuesta 1 Modificada", "product": "P1", "categoria": "C1",
    "keywords": ["k1", "modificado"], "faq_id": "DOC1_q0",
    "texto_para_vectorizar": "Pregunta: Pregunta 1 Modificada Respuesta: Respuesta 1 Modificada"
}
SAMPLE_QA_ITEM_2 = {
    "q": "Pregunta 2", "a": "Respuesta 2", "product": "P2", "categoria": "C2",
    "keywords": ["k2"], "faq_id": "DOC1_q1",
    "texto_para_vectorizar": "Pregunta: Pregunta 2 Respuesta: Respuesta 2"
}
SAMPLE_QA_ITEM_3 = {
    "q": "Pregunta 3", "a": "Respuesta 3", "product": "P1", "categoria": "C3",
    "keywords": ["k3"], "faq_id": "DOC2_q0",
    "texto_para_vectorizar": "Pregunta: Pregunta 3 Respuesta: Respuesta 3"
}

# Calcular hashes para usar en el estado
HASH_ITEM_1_V1 = state_manager.generate_content_hash(SAMPLE_QA_ITEM_1_V1["texto_para_vectorizar"])
HASH_ITEM_1_V2 = state_manager.generate_content_hash(SAMPLE_QA_ITEM_1_V2["texto_para_vectorizar"])
HASH_ITEM_2 = state_manager.generate_content_hash(SAMPLE_QA_ITEM_2["texto_para_vectorizar"])
HASH_ITEM_3 = state_manager.generate_content_hash(SAMPLE_QA_ITEM_3["texto_para_vectorizar"])

# Estado previo simulado (claves son faq_id)
PREVIOUS_STATE_POINTS = {
    "DOC1_q0": {"source_file": "file1.json", "content_hash": HASH_ITEM_1_V1}, # Versión 1 de Item 1
    "DOC1_q1": {"source_file": "file1.json", "content_hash": HASH_ITEM_2},    # Item 2
    "OLD_ID_q0": {"source_file": "old_file.json", "content_hash": "some_old_hash"} # Item que será eliminado
}

# Datos actuales simulados (como los devolvería data_loader)
CURRENT_QAS_MAP_EXAMPLE = {
    "file1.json": [SAMPLE_QA_ITEM_1_V2], # Item 1 modificado (DOC1_q0), Item 2 eliminado de este archivo
    "file_new.json": [SAMPLE_QA_ITEM_3]     # Item 3 nuevo (DOC2_q0) en archivo nuevo
    # Falta item con faq_id = OLD_ID_q0
}


# --- Pruebas ---

def test_generate_content_hash_consistency():
    """Verifica que el hash sea consistente para el mismo contenido."""
    hash1 = state_manager.generate_content_hash("Mismo texto")
    hash2 = state_manager.generate_content_hash("Mismo texto")
    hash3 = state_manager.generate_content_hash("Otro texto")
    assert hash1 == hash2
    assert hash1 != hash3

# Nota: La prueba de generate_qa_uuid se omite ya que no se usa para el ID principal

def test_load_state_file_not_found(tmp_path):
    """Prueba cargar estado cuando el archivo no existe."""
    state_path = tmp_path / "non_existent_state.json"
    state = state_manager.load_state(state_path)
    assert state is not None
    assert state["version"] == state_manager.STATE_FILE_VERSION # Debe tener nueva versión
    assert state["last_run_utc"] is None
    assert state["indexed_points"] == {}

def test_load_save_state(tmp_path):
    """Prueba guardar y luego cargar un estado válido."""
    state_path = tmp_path / "test_state.json"
    # Estado actual simulado (detalles de puntos indexados)
    current_details = {
        "DOC1_q0": {"source_file": "f1.json", "content_hash": HASH_ITEM_1_V1},
        "DOC2_q0": {"source_file": "f2.json", "content_hash": HASH_ITEM_3}
    }

    # Guardar usando la nueva firma de save_state
    save_ok = state_manager.save_state(state_path, current_details)
    assert save_ok is True
    assert state_path.is_file()

    # Cargar y verificar
    loaded_state = state_manager.load_state(state_path)
    assert loaded_state["version"] == state_manager.STATE_FILE_VERSION # Verificar versión
    assert loaded_state["last_run_utc"] is not None # Debe tener timestamp
    assert loaded_state["indexed_points"] == current_details # Verificar que guardó los detalles correctos

def test_load_state_invalid_json(tmp_path):
    """Prueba cargar un archivo de estado con JSON inválido."""
    state_path = tmp_path / "invalid_state.json"
    state_path.write_text("{not json")
    state = state_manager.load_state(state_path)
    # Debe devolver estado inicial
    assert state["indexed_points"] == {}
    assert state["version"] == state_manager.STATE_FILE_VERSION

def test_load_state_wrong_structure(tmp_path):
    """Prueba cargar un archivo de estado con estructura incorrecta."""
    state_path = tmp_path / "wrong_structure.json"
    state_path.write_text(json.dumps({"data": "instead_of_indexed_points"}))
    state = state_manager.load_state(state_path)
    # Debe devolver estado inicial o corregido
    assert "indexed_points" in state and state["indexed_points"] == {}
    assert state["version"] == state_manager.STATE_FILE_VERSION

def test_calculate_diff_logic():
    """Prueba la lógica principal de cálculo de diferencias con faq_id y content_hash."""
    upsert_list, delete_list, current_details = state_manager.calculate_diff(
        CURRENT_QAS_MAP_EXAMPLE, PREVIOUS_STATE_POINTS
    )

    # 1. Verificar current_points_details (debe tener todos los items actuales)
    assert isinstance(current_details, dict)
    assert len(current_details) == 2 # Item 1 modificado + Item 3 nuevo
    assert "DOC1_q0" in current_details
    assert current_details["DOC1_q0"]["content_hash"] == HASH_ITEM_1_V2 # Hash nuevo
    assert current_details["DOC1_q0"]["source_file"] == "file1.json"
    assert "DOC2_q0" in current_details
    assert current_details["DOC2_q0"]["content_hash"] == HASH_ITEM_3
    assert current_details["DOC2_q0"]["source_file"] == "file_new.json"

    # 2. Verificar ids_to_delete (debe tener los faq_ids viejos que ya no están)
    assert isinstance(delete_list, list)
    # Esperamos OLD_ID_q0 y DOC1_q1 (porque ya no aparece en los datos actuales)
    assert len(delete_list) == 2
    assert "OLD_ID_q0" in delete_list
    assert "DOC1_q1" in delete_list

    # 3. Verificar qas_to_process (debe tener los items nuevos y modificados)
    assert isinstance(upsert_list, list)
    # Esperamos Item 1 (versión 2, modificado) y Item 3 (nuevo)
    assert len(upsert_list) == 2
    upsert_ids = {item['faq_id'] for item in upsert_list}
    assert "DOC1_q0" in upsert_ids # Item modificado
    assert "DOC2_q0" in upsert_ids # Item nuevo

    # Verificar que los items en upsert_list tengan el campo _source_file (enriquecido)
    # y NO los campos temporales _id, _question_hash
    for item in upsert_list:
        assert "_source_file" in item
        assert "_id" not in item
        assert "_question_hash" not in item

    # Chequear contenido específico de un item procesado
    item1_processed = next(item for item in upsert_list if item['faq_id'] == "DOC1_q0")
    assert item1_processed['q'] == SAMPLE_QA_ITEM_1_V2['q'] # Debe ser el item original modificado
    assert item1_processed['_source_file'] == "file1.json"


def test_calculate_diff_initial_run():
    """Prueba el diff cuando no hay estado previo."""
    empty_previous_state = {}
    upsert_list, delete_list, current_details = state_manager.calculate_diff(
        CURRENT_QAS_MAP_EXAMPLE, empty_previous_state
    )

    assert len(delete_list) == 0 # Nada que borrar
    # Todos los items actuales deben estar para procesar
    assert len(upsert_list) == 2 # Item 1 (v2) + Item 3
    assert len(current_details) == 2 # Estado actual debe tener 2 items
    assert "DOC1_q0" in current_details
    assert "DOC2_q0" in current_details

def test_calculate_diff_no_changes():
    """Prueba el diff cuando los datos actuales coinciden con el estado previo."""
    # Estado previo con Item 1 V2 y Item 3 (usando content_hash)
    previous_state_no_change = {
        "DOC1_q0": {"source_file": "file1.json", "content_hash": HASH_ITEM_1_V2},
        "DOC2_q0": {"source_file": "file_new.json", "content_hash": HASH_ITEM_3}
    }
    # Datos actuales que coinciden con ese estado
    current_data_no_change = {
        "file1.json": [SAMPLE_QA_ITEM_1_V2],
        "file_new.json": [SAMPLE_QA_ITEM_3]
    }

    upsert_list, delete_list, current_details = state_manager.calculate_diff(
        current_data_no_change, previous_state_no_change
    )

    assert len(upsert_list) == 0 # Nada nuevo ni modificado
    assert len(delete_list) == 0 # Nada que borrar
    # El estado actual debe ser igual al previo en términos de claves y hashes
    assert len(current_details) == 2
    assert current_details == previous_state_no_change


def test_calculate_diff_item_missing_required_fields():
    """Prueba que el diff ignore items si faltan faq_id o texto_para_vectorizar."""
    # Datos actuales con un item inválido
    current_data_invalid = {
        "file_invalid.json": [
            {"q": "Inválido", "a": "Falta ID", "categoria": "C", "product": "P", "keywords": [], "texto_para_vectorizar": "Inválido"}, # Falta faq_id
            {"q": "Inválido 2", "a": "Falta Texto", "categoria": "C", "product": "P", "keywords": [], "faq_id": "INV_q1"} # Falta texto_para_vectorizar
        ]
    }
    previous_state = {} # Estado inicial

    upsert_list, delete_list, current_details = state_manager.calculate_diff(
        current_data_invalid, previous_state
    )

    assert len(upsert_list) == 0 # Ningún item válido para procesar
    assert len(delete_list) == 0 # Nada que borrar
    assert len(current_details) == 0 # Ningún item válido actual