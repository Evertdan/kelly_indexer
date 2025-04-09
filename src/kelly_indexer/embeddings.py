# src/kelly_indexer/embeddings.py
# -*- coding: utf-8 -*-

"""
Módulo para manejar la generación de embeddings usando Sentence Transformers.
(Este módulo NO requiere cambios funcionales para las modificaciones de faq_id/categoria/texto_para_vectorizar).

Funciones:
- get_embedding_model: Carga (y cachea) un modelo SentenceTransformer.
- generate_embeddings: Genera embeddings para una lista de textos usando un modelo cargado.
"""

import logging
from typing import List, Optional, Any, Sequence, Dict, Tuple # Añadido Dict, Tuple
from functools import lru_cache

# Hacer numpy una dependencia obligatoria
try:
    import numpy as np
    NumpyArray = np.ndarray # Definir tipo consistentemente
except ImportError:
    print("[ERROR CRÍTICO] Librería 'numpy' no instalada. Es REQUERIDA. Ejecuta: pip install numpy")
    np = None # type: ignore
    NumpyArray = Any # Fallback type

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
    Utiliza caché y fuerza CPU por defecto.
    """
    if SentenceTransformer is None or np is None:
        logger.critical("Dependencias 'sentence-transformers' o 'numpy' no disponibles.")
        return None

    try:
        logger.info(f"Cargando modelo SentenceTransformer: '{model_name}' en dispositivo '{device}'...")
        # Nota: 'device' permite configurar CPU/CUDA si se quisiera, aunque default es 'cpu'.
        model = SentenceTransformer(model_name_or_path=model_name, device=device)
        logger.info(f"Modelo '{model_name}' cargado exitosamente.")
        # Verificar que el objeto cargado sea del tipo esperado
        if isinstance(model, SentenceTransformer):
             return model
        else:
             logger.error(f"El objeto cargado para '{model_name}' no es una instancia de SentenceTransformer.")
             return None
    except ImportError:
        logger.critical("Dependencia 'sentence-transformers' no encontrada (inesperado).")
        return None
    except OSError as e:
        logger.error(f"Error de OS al cargar el modelo '{model_name}': {e}. ¿Nombre correcto? ¿Descargado? ¿Conexión?")
        return None
    except Exception as e:
        logger.exception(f"Error inesperado al cargar modelo '{model_name}': {e}")
        return None

def generate_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    show_progress_bar: bool = False
) -> Optional[NumpyArray]:
    """
    Genera embeddings para una lista de textos usando un modelo cargado.
    """
    if np is None or SentenceTransformer is None:
        logger.critical("Dependencias (numpy/sentence-transformers) no disponibles.")
        return None
    if not isinstance(model, SentenceTransformer):
        logger.error("Se proporcionó un objeto inválido como modelo.")
        return None

    if not texts:
        logger.warning("Lista de textos vacía para generar embeddings.")
        # Devolver array numpy vacío con la forma correcta (0, dimension)
        try:
            dim = model.get_sentence_embedding_dimension()
            # Asegurarse de que dim sea un entero positivo
            if isinstance(dim, int) and dim > 0:
                 return np.empty((0, dim), dtype=np.float32)
            else:
                 logger.error(f"Dimensión inválida obtenida del modelo: {dim}. Devolviendo array vacío genérico.")
                 return np.empty((0, 0), dtype=np.float32) # Fallback genérico
        except Exception as e:
            logger.error(f"No se pudo obtener la dimensión del modelo: {e}. Devolviendo array vacío genérico.")
            return np.empty((0, 0), dtype=np.float32) # Fallback genérico

    logger.info(f"Generando embeddings para {len(texts)} textos con batch_size={batch_size}...")
    try:
        embeddings_array = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            # normalize_embeddings=False # Dejar que el modelo maneje esto; E5 lo necesita, otros ya están norm.
            convert_to_numpy=True # Asegurar salida NumPy
        )
        logger.info(f"Embeddings generados exitosamente. Shape: {getattr(embeddings_array, 'shape', 'N/A')}")

        # Validar salida y tipo
        if isinstance(embeddings_array, np.ndarray):
            # Convertir a float32 si no lo es (preferido por Qdrant)
            if embeddings_array.dtype != np.float32:
                logger.debug(f"Convirtiendo embeddings de {embeddings_array.dtype} a float32.")
                return embeddings_array.astype(np.float32)
            return embeddings_array
        else:
            logger.error(f"model.encode() devolvió un tipo inesperado: {type(embeddings_array)}")
            return None

    except AttributeError:
        logger.error("El objeto 'model' proporcionado no tiene el método 'encode'.")
        return None
    except Exception as e:
        logger.exception(f"Error inesperado durante la generación de embeddings: {e}")
        return None

# --- Bloque para pruebas rápidas (sin cambios funcionales) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Probando Módulo de Embeddings ---")

    if SentenceTransformer is None or np is None:
        print("Dependencias críticas no encontradas. Saliendo de la prueba.")
    else:
        model_name_test = "all-MiniLM-L6-v2" # Modelo pequeño para prueba rápida
        loaded_model = get_embedding_model(model_name_test, device='cpu')

        if loaded_model:
            print(f"\nModelo '{model_name_test}' cargado.")
            dimension = None
            try:
                dimension = loaded_model.get_sentence_embedding_dimension()
                if dimension: print(f"Dimensión del vector: {dimension}")
                else: print("Error obteniendo dimensión.")
            except Exception as e: print(f"Error obteniendo dimensión: {e}")

            sample_texts = [
                "¿Cómo funciona el indexador Kelly?",
                "Prueba de generación de embeddings.",
                "El cielo es azul.",
            ]
            print(f"\nGenerando embeddings para {len(sample_texts)} textos...")
            embeddings_result = generate_embeddings(loaded_model, sample_texts)

            if embeddings_result is not None:
                print(f"Embeddings generados. Tipo: {type(embeddings_result)}, Dtype: {embeddings_result.dtype}")
                if isinstance(embeddings_result, np.ndarray):
                    print(f"Shape del array: {embeddings_result.shape}")
                    if dimension is not None:
                        assert embeddings_result.shape == (len(sample_texts), dimension), "Shape incorrecto"
                print("\nPrueba de embeddings completada exitosamente.")
            else:
                print("\nFallo al generar embeddings.")
        else:
            print("\nFallo al cargar el modelo.")