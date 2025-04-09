# src/kelly_indexer/embeddings.py
# -*- coding: utf-8 -*-

"""
Módulo para manejar la generación de embeddings usando Sentence Transformers.

Funciones:
- get_embedding_model: Carga (y cachea) un modelo SentenceTransformer.
- generate_embeddings: Genera embeddings para una lista de textos usando un modelo cargado.
"""

import logging
from typing import List, Optional, Union, Sequence # Añadir Sequence
from functools import lru_cache # Para cachear el modelo cargado

# Importar dependencias de terceros
try:
    # Especificar numpy explícitamente ya que SentenceTransformer lo devuelve
    import numpy as np
    # Tipado específico para numpy si se usa
    NumpyArray = np.ndarray[Any, Any] # Definir un tipo más específico si es posible
except ImportError:
    print("[ADVERTENCIA EMBEDDINGS] Librería 'numpy' no instalada. Funciones podrían fallar. Ejecuta: pip install numpy")
    np = None # Definir dummy
    NumpyArray = List[List[float]] # Usar lista de listas como fallback para type hints

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR CRÍTICO] Librería 'sentence-transformers' no instalada. Este módulo es esencial. Ejecuta: pip install sentence-transformers")
    # Definir clase dummy para evitar errores de importación, pero el código fallará en runtime
    SentenceTransformer = type("SentenceTransformer", (object,), {}) # type: ignore
    # Salir podría ser apropiado si esta librería es crítica en el momento de importar
    # import sys
    # sys.exit(1)


# Obtener logger para este módulo
logger = logging.getLogger(__name__)

# Variable global para cachear el modelo (alternativa a lru_cache si se prefiere control manual)
# _model_cache: Dict[str, SentenceTransformer] = {}

@lru_cache(maxsize=2) # Cachear los últimos 2 modelos cargados (normalmente solo usarás 1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", device: str = 'cpu') -> Optional[SentenceTransformer]:
    """
    Carga y devuelve una instancia de un modelo SentenceTransformer.

    Utiliza caché para evitar recargar el mismo modelo múltiples veces.
    Fuerza el uso de CPU por defecto según requerimientos.

    Args:
        model_name: El nombre del modelo a cargar (ej. 'all-MiniLM-L6-v2').
        device: El dispositivo a usar ('cpu', 'cuda', etc.). Default 'cpu'.

    Returns:
        Una instancia del modelo SentenceTransformer si se carga correctamente,
        o None si ocurre un error.
    """
    if SentenceTransformer is None or np is None:
         logger.critical("Dependencias 'sentence-transformers' o 'numpy' no disponibles.")
         return None

    try:
        logger.info(f"Cargando modelo SentenceTransformer: '{model_name}' en dispositivo '{device}'...")
        # Forzar CPU si se especificó 'cpu' o si la librería no maneja bien 'auto'
        # sentence-transformers maneja bien device=None (auto-detect), pero forzamos 'cpu'
        # para cumplir el requisito de no dependencia de GPU.
        model = SentenceTransformer(model_name_or_path=model_name, device=device)
        logger.info(f"Modelo '{model_name}' cargado exitosamente.")
        return model
    except ImportError:
        logger.critical("Dependencia 'sentence-transformers' no encontrada. Por favor, instálala.")
        return None
    except OSError as e:
         # Errores comunes: modelo no encontrado localmente y sin conexión para descargar,
         # problemas de disco, etc.
         logger.error(f"Error de OS al cargar el modelo '{model_name}': {e}. ¿Nombre correcto? ¿Conexión a internet?")
         return None
    except Exception as e:
        logger.exception(f"Error inesperado al cargar el modelo SentenceTransformer '{model_name}': {e}")
        return None

def generate_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64, # Un batch size razonable para CPU
    show_progress_bar: bool = False # Desactivar barra interna por defecto
) -> Optional[NumpyArray]:
    """
    Genera embeddings para una lista de textos usando un modelo SentenceTransformer cargado.

    Args:
        model: La instancia del modelo SentenceTransformer cargado.
        texts: Una lista de strings (textos) para generar embeddings.
        batch_size: Tamaño del lote para procesar los textos.
        show_progress_bar: Si se muestra la barra de progreso interna de sentence-transformers.

    Returns:
        Un array de NumPy con los embeddings generados (cada fila es un vector),
        o None si ocurre un error.
    """
    if not isinstance(model, SentenceTransformer):
         logger.error("Se proporcionó un objeto inválido como modelo.")
         return None
    if not texts:
        logger.warning("Se recibió una lista de textos vacía para generar embeddings.")
        # Devolver un array vacío de la forma correcta si numpy está disponible
        return np.empty((0, model.get_sentence_embedding_dimension())) if np else []

    logger.info(f"Generando embeddings para {len(texts)} textos con batch_size={batch_size}...")
    try:
        # Usar model.encode() para generar los embeddings
        embeddings_array = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            # Podríamos especificar normalize_embeddings=True si la métrica de distancia (Coseno) lo requiere,
            # pero Qdrant también puede normalizar. Consultar documentación de Qdrant/modelo.
            # Por defecto, all-MiniLM-L6-v2 produce vectores normalizados.
            # normalize_embeddings=False (default)
        )
        logger.info(f"Embeddings generados exitosamente. Shape: {embeddings_array.shape}")
        # Asegurarse de que sea un numpy array (aunque encode usualmente lo devuelve)
        if np and isinstance(embeddings_array, np.ndarray):
            return embeddings_array
        elif isinstance(embeddings_array, list): # Fallback si no se usa numpy
             logger.warning("La salida de embeddings fue una lista, no un array NumPy.")
             return embeddings_array # Devolver lista de listas
        else:
             logger.error(f"Tipo inesperado devuelto por model.encode: {type(embeddings_array)}")
             return None

    except AttributeError:
        # Si el objeto 'model' no es realmente un SentenceTransformer válido
        logger.error("El objeto 'model' proporcionado no tiene el método 'encode'. ¿Es un modelo válido?")
        return None
    except Exception as e:
        logger.exception(f"Error inesperado durante la generación de embeddings: {e}")
        return None

# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Configurar logging para prueba
    print("--- Probando Módulo de Embeddings ---")

    # 1. Cargar modelo
    model_name_test = "all-MiniLM-L6-v2" # Modelo pequeño y rápido
    loaded_model = get_embedding_model(model_name_test)

    if loaded_model:
        print(f"\nModelo '{model_name_test}' cargado.")
        dimension = loaded_model.get_sentence_embedding_dimension()
        print(f"Dimensión del vector: {dimension}")

        # 2. Generar embeddings para textos de ejemplo
        sample_texts = [
            "¿Cómo funciona el indexador Kelly?",
            "Esta es una prueba de generación de embeddings.",
            "El cielo es azul.",
            "all-MiniLM-L6-v2 es un modelo de sentence-transformers."
        ]
        print(f"\nGenerando embeddings para {len(sample_texts)} textos de ejemplo...")
        embeddings_result = generate_embeddings(loaded_model, sample_texts)

        if embeddings_result is not None:
            print(f"Embeddings generados. Tipo: {type(embeddings_result)}")
            if np and isinstance(embeddings_result, np.ndarray):
                print(f"Shape del array: {embeddings_result.shape}")
                # Verificar shape: (num_textos, dimension_modelo)
                assert embeddings_result.shape == (len(sample_texts), dimension)
            elif isinstance(embeddings_result, list):
                 print(f"Número de vectores: {len(embeddings_result)}")
                 if embeddings_result: print(f"Dimensión del primer vector: {len(embeddings_result[0])}")
                 assert len(embeddings_result) == len(sample_texts)
                 if embeddings_result: assert len(embeddings_result[0]) == dimension


            print("\nPrueba de embeddings completada exitosamente.")
        else:
            print("\nFallo al generar embeddings.")
    else:
        print("\nFallo al cargar el modelo.")