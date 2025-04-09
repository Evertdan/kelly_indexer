# src/kelly_indexer/text_chunker.py
# -*- coding: utf-8 -*-

"""
Módulo para dividir textos largos (específicamente las respuestas 'a')
en fragmentos (chunks) más pequeños con posible solapamiento.

Utiliza RecursiveCharacterTextSplitter de langchain_text_splitters.
"""

import logging
# CORRECCIÓN: Añadir Dict y Tuple a las importaciones de typing
from typing import List, Optional, Any, Dict, Tuple
from functools import lru_cache

# Importar dependencia de Langchain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("[ERROR CRÍTICO] Librería 'langchain-text-splitters' no instalada. Ejecuta: pip install langchain-text-splitters")
    RecursiveCharacterTextSplitter = None # type: ignore

logger = logging.getLogger(__name__)

# Caché para reutilizar instancias de chunker con la misma configuración
# CORRECCIÓN: Añadir tipo correcto al caché
_chunker_cache: Dict[Tuple[int, int], RecursiveCharacterTextSplitter] = {}

@lru_cache(maxsize=4) # Usar lru_cache es generalmente más simple y eficiente que el dict manual
def get_answer_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> Optional[RecursiveCharacterTextSplitter]:
    """
    Obtiene una instancia configurada de RecursiveCharacterTextSplitter usando caché.

    Args:
        chunk_size: Tamaño máximo de cada fragmento (en caracteres).
        chunk_overlap: Número de caracteres de solapamiento entre fragmentos.

    Returns:
        Una instancia de RecursiveCharacterTextSplitter o None si la librería
        no está disponible o hay un error de configuración.
    """
    if RecursiveCharacterTextSplitter is None:
        logger.critical("Dependencia 'langchain-text-splitters' no disponible.")
        return None

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.error(f"Configuración inválida: chunk_size ({chunk_size}) debe ser un entero positivo.")
        return None
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
         logger.error(f"Configuración inválida: chunk_overlap ({chunk_overlap}) debe ser un entero no negativo.")
         return None
    if chunk_overlap >= chunk_size:
        logger.error(f"Configuración inválida: chunk_overlap ({chunk_overlap}) debe ser menor que chunk_size ({chunk_size}).")
        return None

    # lru_cache maneja el cacheo basado en los argumentos de la función
    logger.info(f"Obteniendo/Creando instancia de RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})...")
    try:
        # Usar separadores por defecto de Langchain: ["\n\n", "\n", " ", ""]
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        # CORRECCIÓN: Verificar el tipo devuelto por si acaso (aunque debería ser correcto)
        # y para satisfacer a mypy si lru_cache ofusca el tipo.
        if isinstance(chunker, RecursiveCharacterTextSplitter):
             logger.debug("Instancia de chunker obtenida/creada.")
             return chunker
        else:
             logger.error("La creación de RecursiveCharacterTextSplitter devolvió un tipo inesperado.")
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

    Si el texto es None, vacío, o más corto que chunk_size, generalmente
    devolverá una lista con el texto original como único elemento.

    Args:
        chunker: La instancia de RecursiveCharacterTextSplitter a usar.
        text: El texto a dividir.

    Returns:
        Una lista de strings (fragmentos). Lista vacía si la entrada es inválida o hay error.
    """
    if not isinstance(chunker, RecursiveCharacterTextSplitter):
        logger.error("Se proporcionó un objeto chunker inválido o None a chunk_text.")
        return [] # Devolver lista vacía en caso de error
    if not text: # Manejar None o string vacío
        logger.debug("Texto de entrada para chunking está vacío o es None.")
        return [] # Devolver lista vacía consistentemente

    # El chunker de Langchain maneja textos más cortos que chunk_size devolviendo [texto]
    try:
        logger.debug(f"Dividiendo texto (longitud: {len(text)}) en chunks (size={chunker._chunk_size}, overlap={chunker._chunk_overlap})...")
        chunks = chunker.split_text(text)
        logger.debug(f"Texto dividido en {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.exception(f"Error inesperado durante la división del texto: {e}")
        # Considerar devolver [text] como fallback si falló la división? O vacía?
        return [] # Devolver lista vacía para indicar fallo


# --- Bloque para pruebas rápidas ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("--- Probando Text Chunker ---")

    texto_largo = """Este es un párrafo largo de ejemplo. Contiene varias frases y palabras repetidas para probar la funcionalidad del RecursiveCharacterTextSplitter. La idea es que este texto exceda el tamaño del chunk configurado.
    Este es el segundo párrafo. También es parte del texto largo y añade más contenido. Debería haber un solapamiento entre el final del chunk anterior y el inicio de este.
    Una frase corta final."""

    texto_corto = "Este texto es corto y no necesita división."

    # Obtener chunker con configuración de prueba
    test_chunk_size = 100
    test_overlap = 20
    mi_chunker = get_answer_chunker(chunk_size=test_chunk_size, chunk_overlap=test_overlap)

    if mi_chunker:
        print(f"\nProbando texto largo (len={len(texto_largo)}):")
        chunks_largos = chunk_text(mi_chunker, texto_largo)
        print(f"  Número de chunks generados: {len(chunks_largos)}")
        if chunks_largos:
             for i, chunk in enumerate(chunks_largos):
                 print(f"  Chunk {i+1} (len={len(chunk)}): '{chunk[:40]}...{chunk[-20:]}'" if len(chunk)>60 else f"  Chunk {i+1} (len={len(chunk)}): '{chunk}'")

        print(f"\nProbando texto corto (len={len(texto_corto)}):")
        chunks_cortos = chunk_text(mi_chunker, texto_corto)
        print(f"  Número de chunks generados: {len(chunks_cortos)}")
        if chunks_cortos:
            print(f"  Chunk 1 (len={len(chunks_cortos[0])}): '{chunks_cortos[0]}'")
            assert len(chunks_cortos) == 1
            assert chunks_cortos[0] == texto_corto

        print("\nProbando texto vacío:")
        chunks_vacios = chunk_text(mi_chunker, "")
        print(f"  Número de chunks generados: {len(chunks_vacios)}")
        assert chunks_vacios == []

        print("\nProbando texto None:")
        chunks_none = chunk_text(mi_chunker, None)
        print(f"  Número de chunks generados: {len(chunks_none)}")
        assert chunks_none == []

        print("\nProbando chunker cacheado:")
        mi_chunker_2 = get_answer_chunker(chunk_size=test_chunk_size, chunk_overlap=test_overlap)
        # lru_cache devuelve la misma instancia para los mismos argumentos
        assert mi_chunker is mi_chunker_2

        print("\nProbando configuración inválida (overlap >= size):")
        chunker_invalido = get_answer_chunker(chunk_size=50, chunk_overlap=60)
        assert chunker_invalido is None

    else:
        print("Fallo al obtener la instancia inicial del chunker.")