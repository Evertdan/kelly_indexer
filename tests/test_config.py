# tests/test_config.py
# -*- coding: utf-8 -*-

"""
Pruebas unitarias para el módulo de configuración de Kelly Indexer.
Verifica la carga desde el entorno, valores por defecto y validaciones.
(Este archivo NO requiere modificaciones para los cambios de faq_id/categoria/etc.)
"""

import os
import pytest
import shutil # Import necesario para la limpieza en una prueba
from pathlib import Path
from pydantic import ValidationError, SecretStr, HttpUrl # Importar tipos y error
# Importar SettingsConfigDict para la prueba de .env
from pydantic_settings import SettingsConfigDict
# Importar tipos necesarios para validadores
from typing import get_args, Optional, Any
from pydantic_core.core_schema import ValidationInfo # Necesario para validador custom

# Importar la CLASE Settings del módulo config de kelly_indexer
# Asumiendo que pytest corre desde la raíz y src está en PYTHONPATH
try:
    # Importar también QdrantDistance si se usa en validadores
    from kelly_indexer.config import Settings, PROJECT_ROOT, QdrantDistance
except ImportError:
    pytest.fail("No se pudo importar 'Settings' desde 'kelly_indexer.config'. Asegúrate de la estructura y PYTHONPATH.", pytrace=False)


# --- Fixtures (si fueran necesarios, ej. para crear archivos/dirs complejos) ---
# Por ahora usamos tmp_path directamente en las pruebas

# --- Pruebas de Configuración ---

def test_project_root_detection():
    """Verifica que PROJECT_ROOT apunte al directorio raíz correcto."""
    assert (PROJECT_ROOT / "pyproject.toml").is_file(), f"PROJECT_ROOT ({PROJECT_ROOT}) no parece ser la raíz correcta."
    assert (PROJECT_ROOT / "src" / "kelly_indexer").is_dir(), f"No se encontró src/kelly_indexer en PROJECT_ROOT ({PROJECT_ROOT})."

def test_load_from_environment_variables(monkeypatch, tmp_path):
    """Verifica que las variables de entorno sobrescriban los defaults."""
    # Valores simulados para variables de entorno
    qdrant_url_val = "http://test-qdrant:6333"
    qdrant_api_key_val = "test-api-key-xyz"
    qdrant_collection_val = "test-collection-env"
    distance_val = "Euclid" # Probar métrica diferente al default
    embedding_model_val = "test-embedding-model"
    vector_dim_val = "128" # Probar como string, Pydantic debe convertir a int
    input_json_dir_val = str(tmp_path / "env_input_json") # Usar tmp_path
    state_file_val = str(tmp_path / "env_state.json")
    chunk_size_val = "512"
    batch_size_val = "32"
    log_level_val = "DEBUG"

    # Establecer variables de entorno simuladas
    monkeypatch.setenv("QDRANT_URL", qdrant_url_val)
    monkeypatch.setenv("QDRANT_API_KEY", qdrant_api_key_val)
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", qdrant_collection_val)
    monkeypatch.setenv("DISTANCE_METRIC", distance_val)
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", embedding_model_val)
    monkeypatch.setenv("VECTOR_DIMENSION", vector_dim_val)
    monkeypatch.setenv("INPUT_JSON_DIR", input_json_dir_val)
    monkeypatch.setenv("STATE_FILE_PATH", state_file_val)
    monkeypatch.setenv("CHUNK_SIZE", chunk_size_val)
    monkeypatch.setenv("QDRANT_BATCH_SIZE", batch_size_val)
    monkeypatch.setenv("LOG_LEVEL", log_level_val)

    settings_instance = None # Definir antes del try/finally
    input_path_obj = Path(input_json_dir_val) # Definir para usar en finally
    try:
        # Crear directorio de entrada temporalmente
        input_path_obj.mkdir(parents=True, exist_ok=True)

        # Instanciar Settings *después* de establecer el entorno
        settings_instance = Settings(_env_file=None)

        # Verificar los valores cargados
        assert str(settings_instance.qdrant_url) == qdrant_url_val + "/" # HttpUrl añade /
        assert settings_instance.qdrant_api_key is not None
        assert settings_instance.qdrant_api_key.get_secret_value() == qdrant_api_key_val
        assert settings_instance.qdrant_collection_name == qdrant_collection_val
        # MODIFICADO: Ajustar comparación para validador que puede cambiar (Euclid queda Euclid)
        assert settings_instance.distance_metric == "Euclid" # Validador normaliza "Euclid"
        assert settings_instance.embedding_model_name == embedding_model_val
        assert settings_instance.vector_dimension == int(vector_dim_val) # Convertido a int
        assert settings_instance.input_json_dir == input_path_obj
        assert settings_instance.state_file_path == Path(state_file_val)
        assert settings_instance.chunk_size == int(chunk_size_val)
        assert settings_instance.qdrant_batch_size == int(batch_size_val)
        assert settings_instance.log_level == log_level_val.upper() # Validador capitaliza

    finally:
        # Limpiar directorio creado
        if input_path_obj.exists():
            try:
                shutil.rmtree(input_path_obj)
            except Exception as e:
                print(f"[Advertencia Test] No se pudo limpiar el directorio de prueba {input_path_obj}: {e}")


def test_missing_required_env_variable(monkeypatch):
    """Verifica que falle si falta una variable requerida (QDRANT_URL)."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False) # Asegurar que no interfiera
    assert os.getenv("QDRANT_URL") is None # Doble check

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None) # Evitar leer .env real

    assert "QDRANT_URL" in str(excinfo.value)
    assert "Field required" in str(excinfo.value) or "validation error" in str(excinfo.value).lower()


def test_qdrant_api_key_optional(monkeypatch):
    """Verifica que QDRANT_API_KEY sea opcional."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    try:
        settings_instance = Settings(_env_file=None) # Evitar leer .env real
        assert settings_instance.qdrant_api_key is None
    except ValidationError as e:
        pytest.fail(f"ValidationError inesperada cuando QDRANT_API_KEY es opcional: {e}")


def test_api_key_is_secretstr_when_present(monkeypatch):
    """Verifica que QDRANT_API_KEY sea SecretStr si se proporciona."""
    api_key_value = "secret-qdrant-key"
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", api_key_value)

    settings_instance = Settings(_env_file=None) # Evitar leer .env real

    assert isinstance(settings_instance.qdrant_api_key, SecretStr)
    assert api_key_value not in str(settings_instance.qdrant_api_key)
    assert api_key_value not in repr(settings_instance.qdrant_api_key)
    assert "**********" in repr(settings_instance.qdrant_api_key)
    assert settings_instance.qdrant_api_key.get_secret_value() == api_key_value


def test_log_level_validation(monkeypatch):
    """Verifica la validación del nivel de log."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333") # Necesario porque es requerido

    monkeypatch.setenv("LOG_LEVEL", "warning")
    settings_instance_warn = Settings(_env_file=None)
    assert settings_instance_warn.log_level == "WARNING"

    monkeypatch.setenv("LOG_LEVEL", "TRACE") # Nivel inválido
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "Nivel de log inválido: 'TRACE'" in str(excinfo.value)


def test_distance_metric_validation(monkeypatch):
    """Verifica la validación y normalización de la métrica de distancia."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333") # Necesario

    valid_inputs = ["Cosine", "cosine", "DOT", "dot", "Euclid", "euclid", "Euclidean"]
    expected_outputs = ["Cosine", "Cosine", "Dot", "Dot", "Euclid", "Euclid", "Euclid"] # Normalizado a Euclid

    for test_input, expected in zip(valid_inputs, expected_outputs):
        monkeypatch.setenv("DISTANCE_METRIC", test_input)
        settings_instance = Settings(_env_file=None)
        assert settings_instance.distance_metric == expected, f"Input: {test_input}"

    monkeypatch.setenv("DISTANCE_METRIC", "Manhattan") # Inválido
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "Métrica de distancia inválida: 'Manhattan'" in str(excinfo.value)
    assert "Cosine" in str(excinfo.value) and "Dot" in str(excinfo.value)


def test_chunk_overlap_validation(monkeypatch):
    """Verifica que chunk_overlap sea menor que chunk_size."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333") # Necesario

    # Caso válido
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    settings_instance_valid = Settings(_env_file=None)
    assert settings_instance_valid.chunk_overlap == 200

    # Caso inválido (igual)
    monkeypatch.setenv("CHUNK_OVERLAP", "1000")
    with pytest.raises(ValidationError) as excinfo_eq:
        Settings(_env_file=None)
    assert "chunk_overlap (1000) debe ser menor que chunk_size (1000)" in str(excinfo_eq.value)

    # Caso inválido (mayor)
    monkeypatch.setenv("CHUNK_OVERLAP", "1200")
    with pytest.raises(ValidationError) as excinfo_gt:
        Settings(_env_file=None)
    assert "chunk_overlap (1200) debe ser menor que chunk_size (1000)" in str(excinfo_gt.value)

    # Caso inválido (negativo)
    monkeypatch.setenv("CHUNK_OVERLAP", "-50")
    with pytest.raises(ValidationError) as excinfo_neg:
        Settings(_env_file=None)
    # Pydantic v2 puede dar mensajes ligeramente diferentes
    assert "Input should be greater than or equal to 0" in str(excinfo_neg.value) or \
           "greater than or equal to 0" in str(excinfo_neg.value)


def test_default_paths_are_correctly_formed(monkeypatch):
    """Verifica que las rutas por defecto se construyan como se espera."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333") # Necesario

    settings_instance = Settings(_env_file=None) # Usar defaults

    assert settings_instance.input_json_dir.is_absolute()
    assert settings_instance.input_json_dir == PROJECT_ROOT / "data" / "input" / "json" / "SOAP_TXT"
    assert settings_instance.state_file_path.is_absolute()
    assert settings_instance.state_file_path == PROJECT_ROOT / "scripts" / "indexer" / "index_state_qdrant.json"
    # Estas rutas son de config pero pueden no ser usadas activamente por indexer, solo referenciadas
    assert settings_instance.input_dir_processed == PROJECT_ROOT / "data" / "input" / "processed"
    assert settings_instance.output_dir_reports == PROJECT_ROOT / "data" / "output" / "reports"


def test_load_from_dotenv_file(tmp_path):
    """Verifica la carga desde un archivo .env de prueba."""
    # Contenido del .env temporal
    env_content = """
    QDRANT_URL = "http://dotenv-qdrant:1234" # Sobrescribe default
    QDRANT_COLLECTION_NAME = "dotenv-collection" # Sobrescribe default
    VECTOR_DIMENSION = 128 # Sobrescribe default
    CHUNK_SIZE = 256 # Sobrescribe default
    EMBEDDING_MODEL_NAME="override-model-name" # Sobrescribe default
    DISTANCE_METRIC = DOT # Sobrescribe default
    EXTRA_VAR = ignore_me
    """
    test_env_file = tmp_path / ".env_test_config_indexer"
    test_env_file.write_text(env_content, encoding='utf-8')

    # Clase temporal para apuntar al .env de prueba
    class TestLoadSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=test_env_file,
            env_file_encoding='utf-8',
            extra='ignore'
        )

    settings_instance = TestLoadSettings()

    # Verificar valores cargados desde el .env temporal
    assert str(settings_instance.qdrant_url) == "http://dotenv-qdrant:1234/"
    assert settings_instance.qdrant_api_key is None # No estaba en .env, usa default None
    assert settings_instance.qdrant_collection_name == "dotenv-collection"
    assert settings_instance.vector_dimension == 128
    assert settings_instance.chunk_size == 256
    assert settings_instance.distance_metric == "Dot" # Normalizado por validador
    assert settings_instance.embedding_model_name == "override-model-name"
    # Verificar que otros campos usan default porque no estaban en .env
    assert settings_instance.log_level == "INFO"
    assert settings_instance.chunk_overlap == 150