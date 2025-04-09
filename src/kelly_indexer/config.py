# src/kelly_indexer/config.py
# -*- coding: utf-8 -*-

"""
Módulo de configuración para Kelly Indexer.

Carga la configuración desde variables de entorno y/o un archivo .env
ubicado en la raíz del proyecto usando Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
# CORRECCIÓN: Añadir importación de ValidationError y tipos necesarios
from pydantic import Field, HttpUrl, SecretStr, FilePath, DirectoryPath, field_validator, ValidationError, PositiveInt
from pathlib import Path
import os
# CORRECCIÓN: Añadir importaciones de typing necesarias
from typing import Optional, Any, Literal, get_args
from pydantic_core.core_schema import ValidationInfo


# Define la ruta base del proyecto (config.py está en src/kelly_indexer)
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
except NameError:
    PROJECT_ROOT = Path.cwd()
    # Usar print aquí porque el logger aún no está configurado
    print(f"[ADVERTENCIA Config] No se pudo determinar PROJECT_ROOT desde __file__, usando CWD: {PROJECT_ROOT}")

# Tipos permitidos para la métrica de distancia en Qdrant
QdrantDistance = Literal["Cosine", "Dot", "Euclid"]

class Settings(BaseSettings):
    """
    Configuraciones para Kelly Indexer.
    """

    # --- Configuración de Qdrant ---
    qdrant_url: HttpUrl = Field(
        ..., # Requerido
        alias='QDRANT_URL',
        description="URL completa de la instancia de Qdrant (ej. Cloud o local)."
    )
    qdrant_api_key: Optional[SecretStr] = Field(
        default=None,
        alias='QDRANT_API_KEY',
        description="Clave API para Qdrant Cloud u otra instancia protegida (opcional)."
    )
    qdrant_collection_name: str = Field(
        default="kellybot-docs-v1",
        alias='QDRANT_COLLECTION_NAME',
        description="Nombre de la colección en Qdrant donde se indexarán los datos."
    )
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
    vector_dimension: PositiveInt = Field(
        default=384,
        alias='VECTOR_DIMENSION',
        description="Dimensión del vector de embeddings (DEBE coincidir con el modelo)."
    )

    # --- Configuración de Rutas ---
    input_json_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "input" / "json" / "SOAP_TXT",
        alias='INPUT_JSON_DIR',
        description="Directorio raíz que contiene los archivos .json Q&A a indexar."
    )
    state_file_path: Path = Field(
        default=PROJECT_ROOT / "scripts" / "indexer" / "index_state_qdrant.json",
        alias='STATE_FILE_PATH',
        description="Ruta al archivo JSON que guarda el estado de la indexación."
    )
    # Rutas para directorios que podrían necesitar ser creados (usar Path normal)
    input_dir_processed: Path = Field(
        default=PROJECT_ROOT / "data" / "input" / "processed",
        alias='INPUT_DIR_PROCESSED',
        description="Directorio donde mover los .txt después de procesarlos exitosamente (usado por kelly_soap, referenciado aquí si es útil)."
    )
    output_dir_reports: Path = Field(
        default=PROJECT_ROOT / "data" / "output" / "reports",
        alias='OUTPUT_DIR_REPORTS',
        description="Directorio para guardar logs de errores o reportes (opcional)."
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
        # Obtener los literales permitidos del tipo QdrantDistance
        allowed_distances = get_args(QdrantDistance)
        # Manejar alias comunes
        if normalized_value == "Euclidean":
            normalized_value = "Euclid"
        elif normalized_value == "Dotproduct":
            normalized_value = "Dot"

        if normalized_value not in allowed_distances:
            raise ValueError(f"Métrica de distancia inválida: '{value}'. Debe ser una de {allowed_distances}")
        return normalized_value # Devolver valor validado y capitalizado/normalizado

    @field_validator('chunk_overlap')
    @classmethod
    def check_overlap_less_than_size(cls, v: int, info: ValidationInfo):
        """Valida que el solapamiento sea menor que el tamaño del chunk."""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError(f"chunk_overlap ({v}) debe ser menor que chunk_size ({info.data['chunk_size']})")
        return v

# --- NO crear instancia global 'settings = Settings()' aquí ---
# La instancia debe crearse explícitamente donde se necesite.

# Bloque para probar la carga directamente (ejecutando python src/kelly_indexer/config.py)
if __name__ == "__main__":
    print(f"Intentando cargar configuración. Buscando .env en: {PROJECT_ROOT / '.env'}")
    # CORRECCIÓN: Envolver en try...except para manejar ValidationError si faltan variables
    try:
        # Instanciar aquí SÓLO para probar la carga desde .env/entorno
        test_settings = Settings()

        # Crear directorios de salida/procesados si no existen (para que el test no falle si no existen)
        # Es mejor que el código principal (run_processing) se encargue de esto al inicio.
        # Aquí solo los creamos para la demostración/prueba de este archivo.
        print("Asegurando existencia de directorios por defecto (para prueba)...")
        test_settings.output_dir_json.mkdir(parents=True, exist_ok=True)
        test_settings.input_dir_processed.mkdir(parents=True, exist_ok=True)
        test_settings.output_dir_reports.mkdir(parents=True, exist_ok=True)
        if test_settings.log_file:
             test_settings.log_file.parent.mkdir(parents=True, exist_ok=True)


        print("\n--- Configuración Cargada Exitosamente (Valores Efectivos) ---")
        # Usar model_dump() para ver los valores (excluye secretos por defecto)
        # print(test_settings.model_dump_json(indent=2)) # Alternativa JSON
        for field_name, value in test_settings.model_dump().items():
             # Mostrar SecretStr de forma segura
             if isinstance(getattr(test_settings, field_name), SecretStr):
                 print(f"  {field_name}: {'******' if value else 'None'}")
             else:
                 print(f"  {field_name}: {value}")

        print(f"\n(Ruta Raíz del Proyecto detectada: {PROJECT_ROOT})")

        # Verificar si el directorio de entrada existe (ahora es solo Path, no DirectoryPath)
        print(f"\nVerificando directorio de entrada: {test_settings.input_json_dir}")
        if test_settings.input_json_dir.is_dir():
             print("  -> Encontrado.")
        else:
             print("  -> NO Encontrado (El script principal debería manejar esto o fallar).")

    except ValidationError as e: # Capturar error si faltan variables (ej. QDRANT_URL)
        print("\n--- Error de Validación al Cargar Configuración ---")
        print(e)
        print("\nAsegúrate de que tu archivo .env en la raíz del proyecto existe y contiene todas las variables requeridas (como QDRANT_URL).")
    except Exception as e:
        print(f"\n--- Error Inesperado al Cargar Configuración ---")
        print(e)