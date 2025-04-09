# src/kelly_indexer/embeddings.py
# -*- coding: utf-8 -*-

"""
Módulo para manejar la generación de embeddings usando Sentence Transformers.

Funciones:
- get_embedding_model: Carga (y cachea) un modelo SentenceTransformer.
- generate_embeddings: Genera embeddings para una lista de textos usando un modelo cargado.
"""

import logging
# CORRECCIÓN: Añadir Any e importar siempre Optional/List/Sequence
from typing import List, Optional, Any, Sequence
from functools import lru_cache

# CORRECCIÓN: Hacer numpy una dependencia obligatoria para simplificar tipos
try:
    import numpy as np
    # CORRECCIÓN: Definir NumpyArray consistentemente
    NumpyArray = np.ndarray # Opcional: np.ndarray[Any, Any] para más detalle
except ImportError:
    print("[ERROR CRÍTICO] Librería 'numpy' no instalada. Es REQUERIDA. Ejecuta: pip install numpy")
    np = None # type: ignore # Definir para que el resto del código no falle en importación
    NumpyArray = None # type: ignore # El tipado fallará, pero evita NameError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR CRÍTICO] Librería 'sentence-transformers' no instalada. Ejecuta: pip install sentence-transformers")
    SentenceTransformer = None # type: ignore

# Obtener logger para este módulo
logger = logging.getLogger(__name__)


@lru_cache(maxsize=2) # Cachear los últimos 2 modelos cargados
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", device: str = 'cpu') -> Optional[SentenceTransformer]:
    """
    Carga y devuelve una instancia de un modelo SentenceTransformer.

    Utiliza caché para evitar recargar el mismo modelo múltiples veces.
    Fuerza el uso de CPU por defecto.

    Args:
        model_name: El nombre del modelo a cargar (ej. 'all-MiniLM-L6-v2').
        device: El dispositivo a usar ('cpu', 'cuda', etc.). Default 'cpu'.

    Returns:
        Una instancia del modelo SentenceTransformer si se carga correctamente,
        o None si ocurre un error o faltan dependencias.
    """
    # CORRECCIÓN: Verificar dependencias críticas al inicio
    if SentenceTransformer is None or np is None:
         logger.critical("Dependencias 'sentence-transformers' o 'numpy' no disponibles.")
         return None

    try:
        logger.info(f"Cargando modelo SentenceTransformer: '{model_name}' en dispositivo '{device}'...")
        # Forzar CPU para cumplir requisito de no dependencia GPU explícita
        model = SentenceTransformer(model_name_or_path=model_name, device=device)
        logger.info(f"Modelo '{model_name}' cargado exitosamente.")
        return model
    except ImportError: # Aunque ya chequeamos, por si acaso
        logger.critical("Dependencia 'sentence-transformers' no encontrada.")
        return None
    except OSError as e:
         logger.error(f"Error de OS al cargar el modelo '{model_name}': {e}. ¿Nombre correcto? ¿Conexión?")
         return None
    except Exception as e:
        logger.exception(f"Error inesperado al cargar modelo '{model_name}': {e}")
        return None

def generate_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    show_progress_bar: bool = False
) -> Optional[NumpyArray]: # CORRECCIÓN: El tipo de retorno es NumpyArray o None
    """
    Genera embeddings para una lista de textos usando un modelo cargado.

    Args:
        model: La instancia del modelo SentenceTransformer cargado.
        texts: Lista de strings (textos) para generar embeddings.
        batch_size: Tamaño del lote para procesar los textos.
        show_progress_bar: Si se muestra la barra de progreso interna.

    Returns:
        Un array de NumPy con los embeddings generados, o None si ocurre un error.
    """
    # CORRECCIÓN: Verificar dependencias críticas y modelo válido
    if np is None or SentenceTransformer is None:
         logger.critical("Dependencias (numpy/sentence-transformers) no disponibles para generar embeddings.")
         return None
    if not isinstance(model, SentenceTransformer):
         logger.error("Se proporcionó un objeto inválido como modelo.")
         return None

    if not texts:
        logger.warning("Se recibió una lista de textos vacía para generar embeddings.")
        # CORRECCIÓN: Devolver un array numpy vacío con la forma correcta
        try:
            dim = model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)
        except Exception as e:
             logger.error(f"No se pudo obtener la dimensión del modelo para crear array vacío: {e}")
             # Devolver array 2D vacío genérico como último recurso
             return np.empty((0, 0), dtype=np.float32)


    logger.info(f"Generando embeddings para {len(texts)} textos con batch_size={batch_size}...")
    try:
        embeddings_array = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            # normalize_embeddings=False (all-MiniLM-L6-v2 ya está normalizado)
        )
        logger.info(f"Embeddings generados exitosamente. Shape: {getattr(embeddings_array, 'shape', 'N/A')}")

        # CORRECCIÓN: Asegurarse de que sea ndarray y devolver None si no
        if isinstance(embeddings_array, np.ndarray):
            # Convertir a float32 si no lo es (Qdrant a menudo prefiere float32)
            if embeddings_array.dtype != np.float32:
                logger.debug(f"Convirtiendo embeddings de {embeddings_array.dtype} a float32.")
                return embeddings_array.astype(np.float32)
            return embeddings_array
        else:
             # Esto no debería pasar si sentence-transformers funciona bien
             logger.error(f"model.encode() devolvió un tipo inesperado: {type(embeddings_array)}")
             return None

    except AttributeError:
        logger.error("El objeto 'model' proporcionado no tiene el método 'encode'. ¿Es un modelo válido?")
        return None
    except Exception as e:
        logger.exception(f"Error inesperado durante la generación de embeddings: {e}")
        return None

# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Probando Módulo de Embeddings ---")

    # Verificar dependencias antes de continuar
    if SentenceTransformer is None or np is None:
        print("Dependencias críticas (numpy o sentence-transformers) no encontradas. Saliendo de la prueba.")
    else:
        # 1. Cargar modelo
        model_name_test = "all-MiniLM-L6-v2" # Modelo pequeño y rápido
        loaded_model = get_embedding_model(model_name_test, device='cpu') # Forzar CPU en prueba también

        if loaded_model:
            print(f"\nModelo '{model_name_test}' cargado.")
            try:
                dimension = loaded_model.get_sentence_embedding_dimension()
                print(f"Dimensión del vector: {dimension}")
            except Exception as e:
                print(f"Error obteniendo dimensión del modelo: {e}")
                dimension = None # Marcar como desconocido

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
                # Verificar si es array numpy y tiene shape correcto
                if isinstance(embeddings_result, np.ndarray):
                    print(f"Shape del array: {embeddings_result.shape}")
                    if dimension is not None:
                         assert embeddings_result.shape == (len(sample_texts), dimension), "Shape incorrecto"
                else: # Si por alguna razón no es ndarray (no debería pasar)
                     print(f"Resultado no es un array NumPy. Longitud: {len(embeddings_result)}")
                     assert len(embeddings_result) == len(sample_texts)


                print("\nPrueba de embeddings completada exitosamente.")
            else:
                print("\nFallo al generar embeddings.")
        else:
            print("\nFallo al cargar el modelo.")