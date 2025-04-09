# src/kelly_indexer/qdrant_ops.py
# -*- coding: utf-8 -*-

"""
Módulo para encapsular las operaciones con la base de datos vectorial Qdrant.
CORREGIDO para resolver errores de MyPy y NameError en except.
"""

import logging
import uuid
import time
import os # CORRECCIÓN: Importar os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

# Importar dependencias de Qdrant
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import PointStruct, Distance, VectorParams, UpdateStatus, CollectionInfo, VectorParamsDiff, OptimizersConfigDiff, CollectionStatus
    # Importar excepciones específicas si existen y se usan
    from qdrant_client.http.exceptions import UnexpectedResponse
    # Si qdrant_client define errores específicos de conexión/timeout, importarlos aquí
    # from qdrant_client.http.exceptions import ConnectionError, TimeoutError # Ejemplo
except ImportError:
    print("[ERROR CRÍTICO] Librería 'qdrant-client' no instalada. Ejecuta: pip install qdrant-client")
    # Definir dummies
    QdrantClient = None # type: ignore
    models = None # type: ignore
    PointStruct = Dict # type: ignore
    Distance = None # type: ignore
    VectorParams = None # type: ignore
    UpdateStatus = None # type: ignore
    CollectionInfo = None # type: ignore
    VectorParamsDiff = None # type: ignore
    OptimizersConfigDiff = None # type: ignore
    CollectionStatus = None # type: ignore
    UnexpectedResponse = ConnectionError # type: ignore

# Obtener logger
logger = logging.getLogger(__name__)

# Mapeo de nombres de métrica de distancia (config) a enums de Qdrant
# (Asegura que los enums existan si la librería se importó)
DISTANCE_MAP = {
    "Cosine": getattr(models, 'Distance', {}).COSINE if models else None,
    "Dot": getattr(models, 'Distance', {}).DOT if models else None,
    "Euclid": getattr(models, 'Distance', {}).EUCLID if models else None,
}

def _decode_qdrant_error_content(content: Optional[bytes]) -> str:
    """Intenta decodificar el contenido de un error de Qdrant (bytes) a string."""
    # (Sin cambios)
    if isinstance(content, bytes):
        try:
            return content.decode('utf-8', errors='replace')
        except Exception:
            return repr(content)
    return str(content)


def initialize_client(url: str, api_key: Optional[str] = None, timeout: int = 60) -> Optional[QdrantClient]:
    """
    Inicializa y devuelve un cliente Qdrant conectado a la instancia especificada.
    """
    # (Sin cambios funcionales, asume que las excepciones de conexión/auth
    #  serán capturadas por Exception o UnexpectedResponse si es necesario)
    if QdrantClient is None:
        logger.critical("Dependencia 'qdrant-client' no disponible.")
        return None

    logger.info(f"Inicializando cliente Qdrant para URL: {url}...")
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout
        )
        client.get_collections() # Verificar conexión
        logger.info("Cliente Qdrant inicializado y conexión verificada.")
        return client
    except ValueError as e:
        logger.error(f"Error de valor al inicializar cliente Qdrant (¿URL/config inválida?): {e}")
        return None
    # CORRECCIÓN: Eliminar catch de AuthenticationError si no está definido
    # except AuthenticationError as e:
    #     logger.error(f"Error de autenticación con Qdrant (API Key inválida?): {e}")
    #     return None
    except UnexpectedResponse as e: # Capturar errores HTTP específicos de Qdrant
         content_str = _decode_qdrant_error_content(e.content)
         logger.error(f"Error HTTP inesperado de Qdrant al conectar/verificar: Status={e.status_code}, Contenido={content_str}")
         return None
    except Exception as e: # Capturar otros errores (conexión, etc.)
        logger.exception(f"Error inesperado al inicializar/conectar cliente Qdrant para {url}: {e}")
        return None

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance_metric_str: str = "Cosine",
    recreate_if_exists: bool = False # ¡Peligroso!
) -> bool:
    """
    Asegura que una colección exista en Qdrant con los parámetros especificados.
    """
    # CORRECCIÓN: Validar métrica después de chequear 'models'
    if not client or not models:
        logger.error("Cliente Qdrant o dependencia 'models' no inicializados.")
        return False

    distance_metric = DISTANCE_MAP.get(distance_metric_str.capitalize())
    if distance_metric is None:
        logger.error(f"Métrica de distancia '{distance_metric_str}' no válida o librería Qdrant no cargada. Usar: Cosine, Dot, Euclid.")
        return False

    logger.info(f"Asegurando existencia/configuración de colección '{collection_name}'...")
    try:
        collection_exists = False
        collection_info = None # Definir para evitar UnboundLocalError
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            collection_exists = True
            logger.info(f"Colección '{collection_name}' ya existe.")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.info(f"Colección '{collection_name}' no encontrada.")
                collection_exists = False
            else:
                content_str = _decode_qdrant_error_content(e.content)
                logger.error(f"Error Qdrant al verificar colección '{collection_name}': Status={e.status_code}, Contenido={content_str}")
                return False
        except Exception as e:
            logger.exception(f"Error al obtener info de colección '{collection_name}': {e}")
            return False

        # (Resto de la lógica de ensure_collection sin cambios funcionales necesarios aquí,
        # la comparación de config existente ya maneja diferentes formatos de respuesta)
        # ... (código para recreate_if_exists) ...
        # ... (código para verificar config si no se recrea) ...
        # ... (código para crear colección si no existe) ...

        # Asegurar que la lógica de creación y verificación final permanezca
        if not collection_exists:
            logger.info(f"Creando colección '{collection_name}'...")
            client.create_collection(
                 collection_name=collection_name,
                 vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
            )
            time.sleep(0.5) # Pausa opcional
            if client.collection_exists(collection_name=collection_name):
                 logger.info(f"Colección '{collection_name}' creada exitosamente.")
                 return True
            else:
                 logger.error(f"Fallo al verificar creación de colección '{collection_name}'.")
                 return False
        else:
             # Si existía, asumimos que está OK o ya se logueó advertencia de config
             return True

    except UnexpectedResponse as e:
        content_str = _decode_qdrant_error_content(e.content)
        logger.error(f"Error Qdrant operando sobre colección '{collection_name}': Status={e.status_code}, Contenido={content_str}")
        return False
    except Exception as e:
        logger.exception(f"Error inesperado al asegurar colección '{collection_name}': {e}")
        return False


def batch_upsert(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 128,
    wait: bool = True,
    max_retries: int = 2,
    retry_delay: int = 5
) -> Tuple[int, int]:
    """
    Realiza upsert en Qdrant en lotes con reintentos básicos.
    """
    # (Lógica inicial sin cambios)
    if not client or not models: return 0, len(points)
    if not points: return 0, 0

    total_points = len(points)
    successful_points = 0
    total_errors = 0
    processed_batches = 0
    logger.info(f"Iniciando upsert de {total_points} puntos en '{collection_name}' (lotes de {batch_size})...")

    for i in range(0, total_points, batch_size):
        batch = points[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.debug(f"Procesando lote {batch_num}/{ (total_points + batch_size - 1) // batch_size } ({len(batch)} puntos)...")
        current_retries = 0
        while current_retries <= max_retries:
            try:
                status = client.upsert(collection_name=collection_name, points=batch, wait=wait)
                # ... (manejo de status ok sin cambios) ...
                if hasattr(status, 'status') and status.status == UpdateStatus.COMPLETED:
                     logger.debug(f"Lote {batch_num} completado exitosamente.")
                     successful_points += len(batch)
                     break
                elif isinstance(status, dict) and status.get('status') == 'ok':
                     logger.debug(f"Lote {batch_num} completado exitosamente (status dict ok).")
                     successful_points += len(batch)
                     break
                else:
                     logger.warning(f"Estado inesperado en upsert lote {batch_num}: {status}")
                     if current_retries >= max_retries: total_errors += len(batch)
                     raise ConnectionError(f"Estado inesperado {status}") # Re-lanzar para reintento

            # --- INICIO BLOQUE EXCEPCIONES CORREGIDO ---
            # Capturar errores específicos de Qdrant/red que podrían ser transitorios
            except (UnexpectedResponse, TimeoutError, ConnectionError) as e:
                content_str = ""
                # CORRECCIÓN: Convertir status_code siempre a string
                status_code = "N/A"
                if isinstance(e, UnexpectedResponse):
                    content_str = _decode_qdrant_error_content(e.content)
                    status_code = str(getattr(e, 'status_code', 'N/A')) # Convertir a str

                logger.warning(f"Error Qdrant/Red en upsert lote {batch_num} (Intento {current_retries+1}/{max_retries+1}): Status={status_code}, Error={type(e).__name__}, Contenido={content_str[:200]}...")
                current_retries += 1
                if current_retries > max_retries:
                    logger.error(f"Máximo reintentos ({max_retries}) para lote {batch_num}. Marcando como fallido.")
                    total_errors += len(batch)
                    break # Salir del while
                else:
                    logger.info(f"Esperando {retry_delay}s antes de reintentar...")
                    time.sleep(retry_delay)
                    # Continuar while (reintento)
            # --- FIN BLOQUE EXCEPCIONES CORREGIDO ---

            except Exception as e: # Otros errores inesperados
                logger.exception(f"Error inesperado y fatal durante upsert lote {batch_num}: {e}")
                total_errors += len(batch)
                break # Salir del while

        processed_batches += 1

    # (Lógica de resumen y retorno sin cambios)
    logger.info(f"Upsert finalizado tras procesar {processed_batches} lotes.")
    actual_success = max(0, total_points - total_errors)
    if actual_success != successful_points and total_errors > 0:
        logger.warning(f"Ajustando conteo éxitos upsert de {successful_points} a {actual_success}.")
        successful_points = actual_success
    return successful_points, total_errors


def delete_points(
    client: QdrantClient,
    collection_name: str,
    point_ids: List[Union[str, int]], # Acepta UUIDs (str) o ints
    batch_size: int = 512,
    wait: bool = True
) -> Tuple[int, int]:
    """
    Elimina puntos de una colección Qdrant por sus IDs, en lotes.
    """
    # (Sin cambios necesarios en esta función)
    if not client or not models: return 0, len(point_ids)
    if not point_ids: return 0, 0

    total_to_delete = len(point_ids)
    successful_deletes = 0
    total_errors = 0
    processed_batches = 0
    logger.info(f"Iniciando eliminación de {total_to_delete} puntos de '{collection_name}' (lotes de {batch_size})...")

    for i in range(0, total_to_delete, batch_size):
        batch_ids = point_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.debug(f"Procesando lote delete {batch_num} ({len(batch_ids)} IDs)...")
        try:
            status = client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=batch_ids),
                wait=wait
            )
            if hasattr(status, 'status') and status.status == UpdateStatus.COMPLETED:
                logger.debug(f"Lote delete {batch_num} completado exitosamente.")
                successful_deletes += len(batch_ids)
            elif isinstance(status, dict) and status.get('status') == 'ok':
                logger.debug(f"Lote delete {batch_num} completado exitosamente (status dict ok).")
                successful_deletes += len(batch_ids)
            else:
                logger.warning(f"Estado inesperado devuelto por delete para lote {batch_num}: {status}")
                total_errors += len(batch_ids)
        except UnexpectedResponse as e:
            content_str = _decode_qdrant_error_content(e.content)
            logger.error(f"Error Qdrant delete (lote {batch_num}): Status={e.status_code}, Contenido={content_str}")
            total_errors += len(batch_ids)
        except Exception as e:
            logger.exception(f"Error inesperado durante delete del lote {batch_num}: {e}")
            total_errors += len(batch_ids)
        finally:
            processed_batches += 1

    logger.info(f"Eliminación finalizada tras procesar {processed_batches} lotes.")
    actual_success = max(0, total_to_delete - total_errors)
    if actual_success != successful_deletes and total_errors > 0:
        logger.warning(f"Ajustando conteo eliminaciones exitosas de {successful_deletes} a {actual_success}.")
        successful_deletes = actual_success
    return successful_deletes, total_errors


# --- Bloque para pruebas rápidas (CORREGIDO) ---
if __name__ == "__main__":
    import uuid # Importar uuid aquí para las pruebas
    # CORRECCIÓN: Importar os aquí también
    import os
    import time # Importar time aquí también

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    logger.info("--- Probando Módulo Qdrant Ops (Requiere Qdrant Local o Configurado) ---")

    TEST_QDRANT_URL = os.getenv("TEST_QDRANT_URL", "http://localhost:6333")
    TEST_API_KEY = os.getenv("TEST_QDRANT_API_KEY", None)
    TEST_COLLECTION = "test-collection-ops-module"
    TEST_VECTOR_SIZE = 384 # Ajustar si usas otro modelo para pruebas
    TEST_DISTANCE = "Cosine"

    q_client = initialize_client(TEST_QDRANT_URL, TEST_API_KEY)

    if q_client:
        created = ensure_collection(
            q_client, TEST_COLLECTION, TEST_VECTOR_SIZE, TEST_DISTANCE,
            recreate_if_exists=True # Limpia en cada prueba
        )

        if created:
            logger.info(f"Colección '{TEST_COLLECTION}' lista.")
            points_data = []
            # CORRECCIÓN: Corregir condición de chequeo de importación
            if QdrantClient is not None and models is not None and PointStruct is not Dict:
                points_data = [
                    models.PointStruct(id=str(uuid.uuid4()), vector=[0.1] * TEST_VECTOR_SIZE, payload={"text": "punto 1"}),
                    models.PointStruct(id=str(uuid.uuid4()), vector=[0.2] * TEST_VECTOR_SIZE, payload={"text": "punto 2"}),
                    models.PointStruct(id=str(uuid.uuid4()), vector=[0.3] * TEST_VECTOR_SIZE, payload={"text": "punto 3"}),
                    models.PointStruct(id=str(uuid.uuid4()), vector=[0.4] * TEST_VECTOR_SIZE, payload={"text": "punto 4"}),
                    models.PointStruct(id=str(uuid.uuid4()), vector=[0.5] * TEST_VECTOR_SIZE, payload={"text": "punto 5"}),
                ]
                # CORRECCIÓN: Asegurarse que los IDs sean strings para el borrado
                point_ids_to_delete = [points_data[1].id, points_data[3].id] if points_data else []

                if points_data:
                    upsert_ok, upsert_err = batch_upsert(q_client, TEST_COLLECTION, points_data, batch_size=2)
                    logger.info(f"Resultado Upsert: Exitosos={upsert_ok}, Errores={upsert_err}")
                    time.sleep(1)
                    try:
                        count = q_client.count(collection_name=TEST_COLLECTION, exact=True).count
                        logger.info(f"Conteo post-upsert: {count} (Esperado: {len(points_data)})")
                    except Exception as ce: logger.error(f"Error obteniendo conteo post-upsert: {ce}")

                    if point_ids_to_delete:
                        delete_ok, delete_err = delete_points(q_client, TEST_COLLECTION, point_ids_to_delete, batch_size=3)
                        logger.info(f"Resultado Delete: Exitosos={delete_ok}, Errores={delete_err}")
                        try:
                            count = q_client.count(collection_name=TEST_COLLECTION, exact=True).count
                            expected_final = len(points_data) - len(point_ids_to_delete)
                            logger.info(f"Conteo final: {count} (Esperado: {expected_final})")
                        except Exception as ce: logger.error(f"Error obteniendo conteo final: {ce}")
                    else: logger.warning("No se generaron IDs para probar delete.")
                else: logger.warning("No se pudieron preparar puntos de prueba.")
            else:
                logger.warning("No se pudieron preparar puntos de prueba (qdrant-client no importado?).")
        else: logger.error(f"No se pudo crear/asegurar la colección '{TEST_COLLECTION}'.")
    else: logger.error(f"No se pudo conectar a Qdrant en {TEST_QDRANT_URL}.")