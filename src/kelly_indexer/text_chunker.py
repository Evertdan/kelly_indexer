# src/kelly_indexer/text_chunker.py
# -*- coding: utf-8 -*-

"""
Módulo para dividir textos largos (específicamente las respuestas 'a')
en fragmentos (chunks) más pequeños con posible solapamiento.
Utiliza RecursiveCharacterTextSplitter de langchain_text_splitters.
(Este módulo NO requiere cambios para las modificaciones de faq_id/categoria/texto_para_vectorizar).
"""

import logging
from typing import List, Optional, Any, Dict, Tuple # Añadido Dict, Tuple
from functools import lru_cache

# Importar dependencia de Langchain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("[ERROR CRÍTICO] Librería 'langchain-text-splitters' no instalada. Ejecuta: pip install langchain-text-splitters")
    RecursiveCharacterTextSplitter = None # type: ignore

logger = logging.getLogger(__name__)

# Caché para reutilizar instancias de chunker con la misma configuración
# (Se usa lru_cache en get_answer_chunker)

@lru_cache(maxsize=4) # Usar lru_cache es generalmente más simple
def get_answer_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> Optional[RecursiveCharacterTextSplitter]:
    """
    Obtiene una instancia configurada de RecursiveCharacterTextSplitter usando caché.
    """
    if RecursiveCharacterTextSplitter is None:
        logger.critical("Dependencia 'langchain-text-splitters' no disponible.")
        return None

    # Validación de parámetros
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.error(f"Configuración inválida: chunk_size ({chunk_size}) debe ser entero positivo.")
        return None
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        logger.error(f"Configuración inválida: chunk_overlap ({chunk_overlap}) debe ser entero no negativo.")
        return None
    if chunk_overlap >= chunk_size:
        logger.error(f"Configuración inválida: chunk_overlap ({chunk_overlap}) debe ser menor que chunk_size ({chunk_size}).")
        return None

    logger.info(f"Obteniendo/Creando instancia de RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})...")
    try:
        # Usar separadores por defecto de Langchain
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        if isinstance(chunker, RecursiveCharacterTextSplitter):
            logger.debug("Instancia de chunker obtenida/creada.")
            return chunker
        else:
            logger.error("Creación de RecursiveCharacterTextSplitter devolvió tipo inesperado.")
            return None
    except Exception as e:
        logger.exception(f"Error inesperado al crear RecursiveCharacterTextSplitter: {e}")
        return None

def chunk_text(
    chunker: Optional[RecursiveCharacterTextSplitter],
    text: Optional[str]
) -> List[str]:
    """
    Divide un texto dado en fragmentos usando un TextSplitter configurado.
    """
    if not isinstance(chunker, RecursiveCharacterTextSplitter):
        logger.error("Se proporcionó un objeto chunker inválido o None a chunk_text.")
        return [] # Devolver lista vacía en caso de error
    if not text: # Manejar None o string vacío
        logger.debug("Texto de entrada para chunking está vacío o es None.")
        return [] # Devolver lista vacía consistentemente

    try:
        logger.debug(f"Dividiendo texto (longitud: {len(text)}) en chunks...")
        chunks = chunker.split_text(text)
        logger.debug(f"Texto dividido en {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.exception(f"Error inesperado durante la división del texto: {e}")
        return [] # Devolver lista vacía para indicar fallo


# --- Bloque para pruebas rápidas (sin cambios) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("--- Probando Text Chunker ---")
    # (Código de prueba original aquí...)
    texto_largo = """Este es un párrafo largo de ejemplo. Contiene varias frases y palabras repetidas para probar la funcionalidad del RecursiveCharacterTextSplitter. La idea es que este texto exceda el tamaño del chunk configurado.
    Este es el segundo párrafo. También es parte del texto largo y añade más contenido. Debería haber un solapamiento entre el final del chunk anterior y el inicio de este.
    Una frase corta final."""
    texto_corto = "Este texto es corto y no necesita división."
    test_chunk_size = 100
    test_overlap = 20
    mi_chunker = get_answer_chunker(chunk_size=test_chunk_size, chunk_overlap=test_overlap)
    if mi_chunker:
        print(f"\nProbando texto largo (len={len(texto_largo)}):")
        chunks_largos = chunk_text(mi_chunker, texto_largo)
        print(f"  Número de chunks generados: {len(chunks_largos)}")
        # ... (resto del código de prueba) ...
    else:
        print("Fallo al obtener la instancia inicial del chunker.")