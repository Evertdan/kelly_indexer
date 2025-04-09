# src/kelly_indexer/text_chunker.py
# -*- coding: utf-8 -*-

"""
Módulo para dividir textos largos (específicamente las respuestas 'a')
en fragmentos (chunks) más pequeños con posible solapamiento.

Utiliza RecursiveCharacterTextSplitter de langchain_text_splitters.
"""

import logging
from typing import List, Optional, Any

# Importar dependencia de Langchain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("[ERROR CRÍTICO] Librería 'langchain-text-splitters' no instalada. Este módulo es esencial para dividir textos largos. Ejecuta: pip install langchain-text-splitters")
    # Definir dummy para evitar errores de importación, pero el código fallará
    RecursiveCharacterTextSplitter = None # type: ignore

logger = logging.getLogger(__name__)

# Variable para cachear el chunker si se usa la misma configuración repetidamente
# (Alternativa a crear una instancia nueva cada vez en get_answer_chunker)
_chunker_cache: Dict[Tuple[int, int], RecursiveCharacterTextSplitter] = {}

def get_answer_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> Optional[RecursiveCharacterTextSplitter]:
    """
    Obtiene una instancia configurada de RecursiveCharacterTextSplitter.

    Utiliza un caché simple para reutilizar instancias con la misma configuración.

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

    if chunk_overlap >= chunk_size:
        logger.error(f"Configuración inválida: chunk_overlap ({chunk_overlap}) debe ser menor que chunk_size ({chunk_size}).")
        # Devolver None o lanzar error, dependiendo de cómo se quiera manejar
        return None

    config_tuple = (chunk_size, chunk_overlap)
    if config_tuple in _chunker_cache:
        logger.debug("Reutilizando instancia de TextSplitter cacheada.")
        return _chunker_cache[config_tuple]

    logger.info(f"Creando nueva instancia de RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})...")
    try:
        # Usar separadores por defecto de Langchain: ["\n\n", "\n", " ", ""]
        # length_function=len es el default para contar caracteres
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Usar separadores literales
        )
        _chunker_cache[config_tuple] = chunker
        return chunker
    except Exception as e:
        logger.exception(f"Error inesperado al crear RecursiveCharacterTextSplitter: {e}")
        return None

def chunk_text(
    chunker: Optional[RecursiveCharacterTextSplitter],
    text: Optional[str]
) -> List[str]:
    """
    Divide un texto dado en fragmentos usando un TextSplitter configurado.

    Si el texto es None, vacío, o más corto que chunk_size (considerando overlap),
    generalmente devolverá una lista con el texto original como único elemento
    (dependiendo del comportamiento exacto del splitter).

    Args:
        chunker: La instancia de RecursiveCharacterTextSplitter a usar.
        text: El texto a dividir.

    Returns:
        Una lista de strings, donde cada string es un fragmento del texto original.
        Devuelve una lista vacía si el texto de entrada es None o vacío, o si
        el chunker no es válido.
    """
    if not isinstance(chunker, RecursiveCharacterTextSplitter):
        logger.error("Se proporcionó un objeto chunker inválido o None.")
        return []
    if not text: # Manejar None o string vacío
        logger.debug("Texto de entrada para chunking está vacío o es None. Devolviendo lista vacía.")
        return []

    try:
        logger.debug(f"Dividiendo texto (longitud: {len(text)}) en chunks...")
        # split_text devuelve la lista de fragmentos
        chunks = chunker.split_text(text)
        logger.debug(f"Texto dividido en {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.exception(f"Error inesperado durante la división del texto: {e}")
        # Devolver el texto original en una lista como fallback seguro en caso de error?
        # O devolver lista vacía para indicar fallo? Optamos por lista vacía.
        return []


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
        for i, chunk in enumerate(chunks_largos):
            print(f"  Chunk {i+1} (len={len(chunk)}): '{chunk[:50]}...{chunk[-20:]}'") # Mostrar inicio y fin

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

        print("\nProbando chunker cacheado:")
        mi_chunker_2 = get_answer_chunker(chunk_size=test_chunk_size, chunk_overlap=test_overlap)
        assert mi_chunker is mi_chunker_2 # Debería ser la misma instancia gracias al caché

    else:
        print("Fallo al obtener la instancia del chunker.")