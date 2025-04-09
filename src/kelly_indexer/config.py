# src/kelly_indexer/config.py
# -*- coding: utf-8 -*-

"""
Módulo de configuración para Kelly Indexer.

Carga la configuración desde variables de entorno y/o un archivo .env
ubicado en la raíz del proyecto usando Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl, SecretStr, FilePath, DirectoryPath, field_validator, PositiveInt
from pathlib import Path
import os
from typing import Optional, Any, Literal # Añadir Literal para DISTANCE_METRIC

# Definir la ruta base del proyecto (config.py está en src/kelly_indexer)
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
except NameError:
    PROJECT_ROOT = Path.cwd()
    print(f"[ADVERTENCIA Config] No se pudo determinar PROJECT_ROOT desde __file__, usando CWD: {PROJECT_ROOT}")

# Tipos permitidos para la métrica de distancia en Qdrant
QdrantDistance = Literal["Cosine", "Dot", "Euclid"]

class Settings(BaseSettings):
    """
    Configuraciones para Kelly Indexer.
    """

    # --- Configuración de Qdrant ---
    qdrant_url: HttpUrl = Field(
        ..., # Requerido, sin default aquí, debe estar en .env o env var
        alias='QDRANT_URL',
        description="URL completa de la instancia de Qdrant (ej. Cloud o local)."
    )
    qdrant_api_key: Optional[SecretStr] = Field( # Hacerla opcional si Qdrant es local/sin auth
        default=None,
        alias='QDRANT_API_KEY',
        description="Clave API para Qdrant Cloud u otra instancia protegida (opcional)."
    )
    qdrant_collection_name: str = Field(
        default="kellybot-docs-v1",
        alias='QDRANT_COLLECTION_NAME',
        description="Nombre de la colección en Qdrant donde se indexarán los datos."
    )
    # Distancia - Usar tipo Literal validado
    distance_metric: QdrantDistance = Field(
        default="Cosine",
        alias='DISTANCE_METRIC',
        description="Métrica de distancia para la colección Qdrant (Cosine, Dot, Euclid)."
    )

    # --- Configuración del Modelo de Embeddings ---
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        alias='EMBEDDING_MODEL_NAME',
        description="Nombre del modelo SentenceTransformer a usar para generar embeddings."
    )
    # Usar PositiveInt para asegurar que sea > 0
    vector_dimension: PositiveInt = Field(
        default=384,
        alias='VECTOR_DIMENSION',
        description="Dimensión del vector de embeddings (DEBE coincidir con el modelo)."
    )

    # --- Configuración de Rutas ---
    input_json_dir: Path = Field( # No DirectoryPath para poder crearlo si es necesario o si está vacío al inicio
        default=PROJECT_ROOT / "data" / "input" / "json" / "SOAP_TXT",
        alias='INPUT_JSON_DIR',
        description="Directorio raíz que contiene los archivos .json Q&A a indexar."
    )
    state_file_path: Path = Field(
        # Usar FilePath aquí podría validar existencia, pero es mejor Path normal
        # y que el state_manager maneje la creación/lectura inicial.
        default=PROJECT_ROOT / "scripts" / "indexer" / "index_state_qdrant.json",
        alias='STATE_FILE_PATH',
        description="Ruta al archivo JSON que guarda el estado de la indexación."
    )

    # --- Configuración de Procesamiento ---
    chunk_size: PositiveInt = Field(
        default=1000,
        alias='CHUNK_SIZE',
        description="Tamaño máximo (caracteres) de los chunks de respuesta ('a')."
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0, # Solapamiento no puede ser negativo
        alias='CHUNK_OVERLAP',
        description="Solapamiento de caracteres entre chunks de respuesta."
    )
    qdrant_batch_size: PositiveInt = Field(
        default=128,
        alias='QDRANT_BATCH_SIZE',
        description="Número de puntos a enviar a Qdrant en cada lote de upsert/delete."
    )

    # --- Configuración General ---
    log_level: str = Field(
        default='INFO',
        alias='LOG_LEVEL',
        description="Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    )
    log_file: Optional[Path] = Field(
         default=None,
         alias='LOG_FILE',
         description="Ruta completa al archivo donde guardar logs (opcional)."
     )

    # Configuración del modelo Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
        validate_assignment=True,
    )

    # --- Validadores ---
    @field_validator('log_level', mode='before')
    @classmethod
    def validate_log_level(cls, value: Any) -> str:
        """Valida y normaliza el nivel de log."""
        if not isinstance(value, str):
            raise ValueError("LOG_LEVEL debe ser una cadena de texto.")
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        upper_value = value.upper()
        if upper_value not in valid_levels:
            raise ValueError(f"Nivel de log inválido: '{value}'. Debe ser uno de {valid_levels}")
        return upper_value

    @field_validator('distance_metric', mode='before')
    @classmethod
    def validate_distance_metric(cls, value: Any) -> str:
        """Valida y normaliza la métrica de distancia."""
        if not isinstance(value, str):
             raise ValueError("DISTANCE_METRIC debe ser una cadena de texto.")
        # Comparación insensible a mayúsculas/minúsculas y normalización
        normalized_value = value.capitalize()
        if normalized_value not in get_args(QdrantDistance): # get_args de typing para obtener literales
             # Reintentar con 'Euclidean' si 'Euclid' falla (común)
             if normalized_value == "Euclidean":
                 normalized_value = "Euclid"
             elif normalized_value == "Dotproduct":
                 normalized_value = "Dot"

        if normalized_value not in get_args(QdrantDistance):
            raise ValueError(f"Métrica de distancia inválida: '{value}'. Debe ser una de {get_args(QdrantDistance)}")
        return normalized_value # Devolver valor validado y capitalizado

    @field_validator('chunk_overlap')
    @classmethod
    def check_overlap_less_than_size(cls, v: int, info: ValidationInfo):
        """Valida que el solapamiento sea menor que el tamaño del chunk."""
        # info.data contiene los otros valores ya validados del modelo
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError(f"chunk_overlap ({v}) debe ser menor que chunk_size ({info.data['chunk_size']})")
        return v

# --- Importar get_args y ValidationInfo para validadores ---
from typing import get_args
from pydantic_core.core_schema import ValidationInfo


# --- NO crear instancia global 'settings = Settings()' aquí ---
# La instancia debe crearse en el punto de entrada (scripts/indexer/index_qdrant.py)

# Bloque para probar la carga directamente (python src/kelly_indexer/config.py)
if __name__ == "__main__":
    print(f"Intentando cargar configuración desde: {PROJECT_ROOT / '.env'}")
    try:
        # Instanciar aquí SÓLO para probar la carga
        test_settings = Settings()
        print("\n--- Configuración Cargada Exitosamente (Valores Efectivos) ---")
        # Imprimir de forma segura (SecretStr se ofusca)
        print(test_settings.model_dump_json(indent=2)) # model_dump_json es útil
        print(f"\n(Ruta Raíz del Proyecto detectada: {PROJECT_ROOT})")

        # Verificar si el directorio de entrada existe
        print(f"\nVerificando directorio de entrada: {test_settings.input_json_dir}")
        if test_settings.input_json_dir.is_dir():
             print("  -> Encontrado.")
        else:
             print("  -> NO Encontrado (El script principal debería manejar esto o fallar).")

    except ValidationError as e:
        print("\n--- Error de Validación al Cargar Configuración ---")
        print(e)
    except Exception as e:
        print(f"\n--- Error Inesperado al Cargar Configuración ---")
        print(e)