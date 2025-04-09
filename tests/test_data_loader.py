# tests/test_data_loader.py
# -*- coding: utf-8 -*-

"""
Pruebas unitarias para el módulo data_loader de Kelly Indexer.
MODIFICADO para probar la carga de la nueva estructura JSON con
faq_id, categoria y texto_para_vectorizar.
"""

import pytest
import json
import shutil
from pathlib import Path

# Importar el módulo bajo prueba
from kelly_indexer import data_loader

# --- Datos de Prueba con Nueva Estructura ---

# Ejemplo de contenido válido para un archivo JSON
VALID_JSON_CONTENT_NEW = [
    {
        "q": "Pregunta válida 1",
        "a": "Respuesta válida 1.",
        "product": "Prod_A",
        "categoria": "Cat_1", # Nuevo campo
        "keywords": ["k1", "k2"],
        "faq_id": "DOC001_q0", # Nuevo campo
        "texto_para_vectorizar": "Pregunta: Pregunta válida 1 Respuesta: Respuesta válida 1." # Nuevo campo
    },
    {
        "q": "Pregunta válida 2",
        "a": "Respuesta válida 2 con acentos.",
        "product": "Prod_A",
        "categoria": "Cat_2",
        "keywords": ["k3", "k4", "acentos"],
        "faq_id": "DOC001_q1",
        "texto_para_vectorizar": "Pregunta: Pregunta válida 2 Respuesta: Respuesta válida 2 con acentos."
    }
]

# Ejemplo con algunos items inválidos mezclados
MIXED_VALIDITY_JSON_CONTENT = [
    { # Válido
        "q": "P Válida 1", "a": "R Válida 1", "product": "P", "categoria": "C1",
        "keywords": ["k1"], "faq_id": "MIX_q0", "texto_para_vectorizar": "PV1 RV1"
    },
    { # Inválido: falta faq_id
        "q": "P Inválida 1", "a": "R Inválida 1", "product": "P", "categoria": "C2",
        "keywords": ["k2"], "texto_para_vectorizar": "PI1 RI1"
    },
     { # Inválido: texto_para_vectorizar vacío
        "q": "P Inválida 2", "a": "R Inválida 2", "product": "P", "categoria": "C1",
        "keywords": ["k3"], "faq_id": "MIX_q2", "texto_para_vectorizar": "  "
    },
    { # Válido
        "q": "P Válida 2", "a": "R Válida 2", "product": "P", "categoria": "C2",
        "keywords": ["k4"], "faq_id": "MIX_q3", "texto_para_vectorizar": "PV2 RV2"
    },
    { # Inválido: No es diccionario
        "not_a_dict": True
    }
]

EMPTY_LIST_JSON_CONTENT = []

INVALID_JSON_STRING = "[{'bad': 'json'}" # JSON mal formado

NOT_A_LIST_JSON_CONTENT = {"key": "value"} # JSON válido, pero no es una lista

# --- Fixture para Directorios Temporales ---

@pytest.fixture
def temp_json_dir(tmp_path):
    """Crea un directorio temporal para archivos JSON de prueba."""
    json_dir = tmp_path / "test_json_input"
    json_dir.mkdir()
    return json_dir

# --- Pruebas para load_single_json_file ---

def test_load_single_valid_file_new_format(temp_json_dir):
    """Prueba cargar un archivo JSON con la estructura nueva y válida."""
    file_path = temp_json_dir / "valid_new.json"
    file_path.write_text(json.dumps(VALID_JSON_CONTENT_NEW, ensure_ascii=False), encoding='utf-8')

    result = data_loader.load_single_json_file(file_path)

    assert result is not None, "El resultado no debería ser None para un archivo válido"
    assert isinstance(result, list), "El resultado debería ser una lista"
    assert len(result) == len(VALID_JSON_CONTENT_NEW), "Debería cargar todos los items válidos"
    # Verificar campos nuevos en el primer item
    assert result[0]['faq_id'] == "DOC001_q0"
    assert result[0]['categoria'] == "Cat_1"
    assert result[0]['texto_para_vectorizar'] == "Pregunta: Pregunta válida 1 Respuesta: Respuesta válida 1."
    assert result[0]['q'] == "Pregunta válida 1"

def test_load_single_mixed_validity_file(temp_json_dir):
    """Prueba cargar un archivo con items válidos e inválidos (nueva estructura)."""
    file_path = temp_json_dir / "mixed.json"
    file_path.write_text(json.dumps(MIXED_VALIDITY_JSON_CONTENT, ensure_ascii=False), encoding='utf-8')

    result = data_loader.load_single_json_file(file_path)

    assert result is not None, "El resultado no debería ser None"
    assert isinstance(result, list)
    # Esperamos solo 2 items válidos de los 5 originales
    assert len(result) == 2, "Solo debería cargar los items con estructura y contenido válidos"
    assert result[0]['faq_id'] == "MIX_q0"
    assert result[1]['faq_id'] == "MIX_q3"

def test_load_single_empty_list_file(temp_json_dir):
    """Prueba cargar un archivo JSON que contiene una lista vacía."""
    file_path = temp_json_dir / "empty.json"
    file_path.write_text(json.dumps(EMPTY_LIST_JSON_CONTENT), encoding='utf-8')

    result = data_loader.load_single_json_file(file_path)

    assert result == [], "Debería devolver una lista vacía para un JSON con []"

def test_load_single_invalid_json_string(temp_json_dir):
    """Prueba cargar un archivo con JSON mal formado."""
    file_path = temp_json_dir / "bad_json.json"
    file_path.write_text(INVALID_JSON_STRING, encoding='utf-8')

    result = data_loader.load_single_json_file(file_path)

    assert result is None, "Debería devolver None para JSON mal formado"

def test_load_single_not_a_list_file(temp_json_dir):
    """Prueba cargar un archivo JSON válido pero que no es una lista."""
    file_path = temp_json_dir / "not_list.json"
    file_path.write_text(json.dumps(NOT_A_LIST_JSON_CONTENT), encoding='utf-8')

    result = data_loader.load_single_json_file(file_path)

    assert result is None, "Debería devolver None si el JSON no es una lista"

def test_load_single_file_not_found(temp_json_dir):
    """Prueba cargar un archivo que no existe."""
    file_path = temp_json_dir / "non_existent.json"

    result = data_loader.load_single_json_file(file_path)

    assert result is None, "Debería devolver None si el archivo no existe"

# --- Pruebas para load_all_qas_from_directory ---

def test_load_all_from_directory(temp_json_dir):
    """Prueba escanear un directorio con varios archivos JSON."""
    # Crear archivos de prueba
    (temp_json_dir / "file1.json").write_text(json.dumps(VALID_JSON_CONTENT_NEW[:1]), encoding='utf-8') # 1 item válido
    (temp_json_dir / "subdir").mkdir()
    (temp_json_dir / "subdir" / "file2.json").write_text(json.dumps(MIXED_VALIDITY_JSON_CONTENT), encoding='utf-8') # 2 items válidos
    (temp_json_dir / "empty.json").write_text(json.dumps([]), encoding='utf-8')
    (temp_json_dir / "bad.json").write_text("{bad", encoding='utf-8')
    (temp_json_dir / "not_a_list.json").write_text(json.dumps({"a":1}), encoding='utf-8')
    (temp_json_dir / "other.txt").write_text("ignore me", encoding='utf-8') # Archivo no JSON

    result_map = data_loader.load_all_qas_from_directory(temp_json_dir)

    assert isinstance(result_map, dict)
    # Esperamos entradas solo para file1.json y subdir/file2.json
    assert len(result_map) == 2, "Solo los archivos con Q&As válidos deben estar en el resultado"

    # Verificar contenido de file1.json
    key1 = "file1.json"
    assert key1 in result_map
    assert len(result_map[key1]) == 1
    assert result_map[key1][0]['faq_id'] == "DOC001_q0"

    # Verificar contenido de subdir/file2.json
    # CORRECCIÓN: Usar os.path.join o Path para crear ruta relativa consistente
    key2 = str(Path("subdir") / "file2.json") # Construir ruta relativa correcta
    assert key2 in result_map
    assert len(result_map[key2]) == 2 # Solo 2 items eran válidos
    assert result_map[key2][0]['faq_id'] == "MIX_q0"
    assert result_map[key2][1]['faq_id'] == "MIX_q3"

    # Asegurarse que los archivos vacíos o inválidos no están
    assert "empty.json" not in result_map
    assert "bad.json" not in result_map
    assert "not_a_list.json" not in result_map

def test_load_all_from_empty_directory(temp_json_dir):
    """Prueba escanear un directorio vacío."""
    result_map = data_loader.load_all_qas_from_directory(temp_json_dir)
    assert result_map == {}, "Debería devolver un diccionario vacío para un directorio sin JSONs"

def test_load_all_from_nonexistent_directory(tmp_path):
    """Prueba escanear un directorio que no existe."""
    non_existent_dir = tmp_path / "non_existent"
    result_map = data_loader.load_all_qas_from_directory(non_existent_dir)
    assert result_map == {}, "Debería devolver un diccionario vacío si el directorio base no existe"