# src/kelly_indexer/qdrant_ops.py
# -*- coding: utf-8 -*-

"""
Módulo para encapsular las operaciones con la base de datos vectorial Qdrant.

Funciones:
- initialize_client: Crea y devuelve un cliente Qdrant.
- ensure_collection: Verifica si una colección existe y la crea si no.
- batch_upsert: Sube/actualiza puntos (vectores + payload) a Qdrant en lotes.
- delete_points: Elimina puntos de Qdrant por sus IDs.
"""

import logging
# CORRECCIÓN: Añadir Union, uuid, time
import uuid
import time
from typing import List, Optional, Dict, Any, Tuple, Union

# Importar dependencias de Qdrant
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import PointStruct, Distance, VectorParams, UpdateStatus, CollectionInfo, VectorParamsDiff, OptimizersConfigDiff, CollectionStatus
    # Importar excepciones específicas
    from qdrant_client.http.exceptions import UnexpectedResponse
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
# Asegurarse que los enums existan si la librería se importó
DISTANCE_MAP = {
    "Cosine": getattr(models, 'Distance', {}).COSINE if models else None,
    "Dot": getattr(models, 'Distance', {}).DOT if models else None,
    "Euclid": getattr(models, 'Distance', {}).EUCLID if models else None,
}

def _decode_qdrant_error_content(content: Optional[bytes]) -> str:
    """Intenta decodificar el contenido de un error de Qdrant (bytes) a string."""
    if isinstance(content, bytes):
        try:
            return content.decode('utf-8', errors='replace')
        except Exception:
            return repr(content) # Devolver repr si falla la decodificación
    return str(content) # Devolver como string si no eran bytes


def initialize_client(url: str, api_key: Optional[str] = None, timeout: int = 60) -> Optional[QdrantClient]:
    """
    Inicializa y devuelve un cliente Qdrant conectado a la instancia especificada.
    """
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
        # Verificar conexión intentando obtener clusters (más robusto que list_collections)
        client.get_collections() # Lanza excepción en caso de fallo de conexión/auth
        logger.info("Cliente Qdrant inicializado y conexión verificada.")
        return client
    except ValueError as e:
         logger.error(f"Error de valor al inicializar cliente Qdrant (¿URL/config inválida?): {e}")
         return None
    except AuthenticationError as e: # Específico si qdrant_client lo lanza
         logger.error(f"Error de autenticación con Qdrant (API Key inválida?): {e}")
         return None
    except Exception as e:
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
    La crea si no existe. NO la recrea por defecto si ya existe.
    """
    # CORRECCIÓN: Quitar Distance/VectorParams del check booleano inicial
    if not client or not models:
        logger.error("Cliente Qdrant o modelos no inicializados correctamente.")
        return False

    # Validar y obtener la métrica de distancia enum
    # Usar get() con default None para evitar KeyError si DISTANCE_MAP no está bien inicializado
    distance_metric = DISTANCE_MAP.get(distance_metric_str.capitalize())
    if distance_metric is None:
        logger.error(f"Métrica de distancia '{distance_metric_str}' no válida o librería Qdrant no cargada. Usar: Cosine, Dot, Euclid.")
        return False

    logger.info(f"Asegurando existencia/configuración de colección '{collection_name}'...")
    try:
        collection_exists = False
        try:
            # Usar get_collection para verificar existencia y obtener info a la vez
            collection_info = client.get_collection(collection_name=collection_name)
            collection_exists = True
            logger.info(f"Colección '{collection_name}' ya existe.")
        except UnexpectedResponse as e:
            # Un error 404 indica que no existe, otros errores son problemas reales
            if e.status_code == 404:
                logger.info(f"Colección '{collection_name}' no encontrada.")
                collection_exists = False
            else:
                # Otro error inesperado al intentar obtener la colección
                content_str = _decode_qdrant_error_content(e.content)
                logger.error(f"Error inesperado de Qdrant al verificar colección '{collection_name}': Status={e.status_code}, Contenido={content_str}")
                return False # Fallar si no podemos verificar
        except Exception as e: # Capturar otros errores (ej. conexión)
             logger.exception(f"Error al intentar obtener información de la colección '{collection_name}': {e}")
             return False


        if collection_exists:
            if recreate_if_exists:
                logger.warning(f"Colección '{collection_name}' existe y recreate_if_exists=True. ¡RECREANDO (borrando datos)!")
                # Usar timeout más largo para operaciones potencialmente lentas
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=distance_metric),
                    timeout=120 # Ejemplo de timeout más largo para recrear
                )
                logger.info(f"Colección '{collection_name}' recreada exitosamente.")
                return True
            else:
                # Verificar configuración existente (Lógica refinada)
                vectors_config = getattr(collection_info, 'vectors_config', None) # Usar getattr por seguridad
                vector_params = None
                config_ok = False

                if isinstance(vectors_config, dict): # Qdrant >= 1.9 con múltiples vectores nombrados
                    default_vector_name = '' # Nombre por defecto (o el único si solo hay uno)
                    if default_vector_name in vectors_config:
                        vector_params = vectors_config[default_vector_name]
                    elif len(vectors_config) == 1: # Si solo hay un vector (nombrado o no)
                        vector_params = list(vectors_config.values())[0]
                    else:
                        logger.warning(f"Colección '{collection_name}' tiene múltiples vectores nombrados. Verificando si existe uno con config esperada...")
                        # Intentar encontrar uno que coincida (esto es heurístico)
                        for params in vectors_config.values():
                            params_dict = params if isinstance(params, dict) else params.dict()
                            if params_dict.get('size') == vector_size and \
                               DISTANCE_MAP.get(params_dict.get('distance', '').capitalize()) == distance_metric:
                                logger.info(f"Encontrado vector compatible en colección existente.")
                                config_ok = True
                                break
                        if not config_ok:
                             logger.warning("No se encontró un vector compatible con la configuración esperada.")

                elif isinstance(vectors_config, models.VectorParams): # Cliente/Servidor más antiguo?
                     vector_params = vectors_config

                # Extraer y comparar si encontramos una config de vector relevante
                if not config_ok and vector_params: # Solo comparar si no encontramos ya uno compatible
                    existing_size = None
                    existing_distance = None
                    try:
                         if isinstance(vector_params, models.VectorParams):
                             existing_size = vector_params.size
                             existing_distance = vector_params.distance
                         elif isinstance(vector_params, dict): # Cliente puede devolver dict
                              existing_size = vector_params.get('size')
                              existing_distance_str = vector_params.get('distance', '').upper() # Comparar en mayúsculas
                              existing_distance = getattr(models.Distance, existing_distance_str, None)

                         if existing_size == vector_size and existing_distance == distance_metric:
                              logger.debug("Configuración de vector de colección existente coincide.")
                              config_ok = True
                         else:
                              logger.warning(f"¡Discrepancia de Configuración! Colección '{collection_name}' existe pero con parámetros diferentes.")
                              logger.warning(f"  -> Esperado: size={vector_size}, distance={distance_metric}")
                              logger.warning(f"  -> Encontrado: size={existing_size}, distance={existing_distance}")
                              # Decidir si continuar o fallar
                              # return False # Descomentar para fallar si la config no coincide
                    except Exception as e:
                         logger.warning(f"Error al extraer/comparar config de vector existente: {e}")

                if not config_ok and vector_params is None and not isinstance(vectors_config, dict):
                    logger.warning(f"No se pudo determinar la configuración de vectores para la colección existente '{collection_name}'.")

                return True # La colección existe, continuamos (incluso si hay warning de config)

        else: # Si la colección no existe, crearla
            logger.info(f"Colección '{collection_name}' no encontrada. Creando...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
                # Añadir aquí otras opciones si son necesarias (on_disk, hnsw_config, etc.)
            )
            # Esperar un poco para asegurar que la colección esté lista (opcional)
            time.sleep(0.5)
            # Verificar que realmente se creó
            if client.collection_exists(collection_name=collection_name):
                 logger.info(f"Colección '{collection_name}' creada exitosamente.")
                 return True
            else:
                 logger.error(f"Fallo al verificar la creación de la colección '{collection_name}'.")
                 return False

    except UnexpectedResponse as e:
        content_str = _decode_qdrant_error_content(e.content)
        logger.error(f"Error inesperado de Qdrant al operar sobre colección '{collection_name}': Status={e.status_code}, Contenido={content_str}")
        return False
    except Exception as e:
        logger.exception(f"Error inesperado al asegurar la colección '{collection_name}': {e}")
        return False


def batch_upsert(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 128,
    wait: bool = True,
    max_retries: int = 2, # Añadir reintentos básicos para upsert
    retry_delay: int = 5 # Segundos
) -> Tuple[int, int]:
    """
    Realiza upsert (insertar o actualizar) de puntos en Qdrant en lotes,
    con reintentos básicos para errores transitorios.
    """
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

                if hasattr(status, 'status') and status.status == UpdateStatus.COMPLETED:
                     logger.debug(f"Lote {batch_num} completado exitosamente.")
                     successful_points += len(batch)
                     break # Salir del bucle while de reintentos
                elif isinstance(status, dict) and status.get('status') == 'ok':
                     logger.debug(f"Lote {batch_num} completado exitosamente (status dict ok).")
                     successful_points += len(batch)
                     break # Salir del bucle while de reintentos
                else:
                     # Status inesperado, tratar como error del lote
                     logger.warning(f"Estado inesperado en upsert lote {batch_num}: {status}")
                     if current_retries >= max_retries: total_errors += len(batch)
                     # Intentar reintentar si no es el último intento
                     raise ConnectionError(f"Estado inesperado {status}") # Re-lanzar para reintento

            except (UnexpectedResponse, APIError, APIConnectionError, TimeoutError, ConnectionError) as e: # Errores que podrían ser transitorios
                content_str = ""
                status_code = "N/A"
                if isinstance(e, UnexpectedResponse):
                     content_str = _decode_qdrant_error_content(e.content)
                     status_code = e.status_code
                elif isinstance(e, APIError):
                     status_code = getattr(e, 'status_code', 'N/A')

                logger.warning(f"Error de Qdrant/Red en upsert lote {batch_num} (Intento {current_retries+1}/{max_retries+1}): Status={status_code}, Error={e}, Contenido={content_str[:200]}...")
                current_retries += 1
                if current_retries > max_retries:
                    logger.error(f"Máximo de reintentos alcanzado para lote {batch_num}. Marcando lote como fallido.")
                    total_errors += len(batch)
                    break # Salir del bucle while
                else:
                    logger.info(f"Esperando {retry_delay}s antes de reintentar...")
                    time.sleep(retry_delay)
                    # Continuar con la siguiente iteración del while (reintento)

            except Exception as e: # Otros errores inesperados (ej. datos inválidos)
                logger.exception(f"Error inesperado y fatal durante upsert del lote {batch_num}: {e}")
                total_errors += len(batch)
                break # Salir del bucle while, no reintentar estos errores

        processed_batches += 1 # Incrementar batches procesados (incluso si falló)

    logger.info(f"Upsert finalizado tras procesar {processed_batches} lotes.")
    # El conteo de errores ya se maneja dentro del bucle
    actual_success = max(0, total_points - total_errors)
    if actual_success != successful_points and total_errors > 0: # Corregir conteo si hubo fallos
         logger.warning(f"Ajustando conteo de éxitos upsert de {successful_points} a {actual_success} debido a errores.")
         successful_points = actual_success

    return successful_points, total_errors

def delete_points(
    client: QdrantClient,
    collection_name: str,
    point_ids: List[Union[str, int]], # IDs pueden ser UUIDs (str) o ints
    batch_size: int = 512, # El borrado puede usar lotes más grandes
    wait: bool = True
) -> Tuple[int, int]:
    """
    Elimina puntos de una colección Qdrant por sus IDs, en lotes.
    """
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
         logger.warning(f"Ajustando conteo de eliminaciones exitosas de {successful_deletes} a {actual_success}.")
         successful_deletes = actual_success

    return successful_deletes, total_errors


# --- Bloque para pruebas rápidas (requiere instancia Qdrant corriendo) ---
if __name__ == "__main__":
    # CORRECCIÓN: Añadir imports necesarios para este bloque
    import uuid
    import time
    # Configurar logging básico si se ejecuta directamente
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')

    logger.info("--- Probando Módulo Qdrant Ops (Requiere Qdrant Local o Configurado) ---")

    # --- Configuración Manual para Prueba ---
    TEST_QDRANT_URL = os.getenv("TEST_QDRANT_URL", "http://localhost:6333")
    TEST_API_KEY = os.getenv("TEST_QDRANT_API_KEY", None) # Lee de env si existe
    TEST_COLLECTION = "test-collection-ops-module"
    TEST_VECTOR_SIZE = 384 # Ajustar a tu modelo real si es diferente
    TEST_DISTANCE = "Cosine"

    # 1. Inicializar Cliente
    q_client = initialize_client(TEST_QDRANT_URL, TEST_API_KEY)

    if q_client:
        # 2. Asegurar Colección (con recreate=True para prueba limpia)
        created = ensure_collection(
            q_client, TEST_COLLECTION, TEST_VECTOR_SIZE, TEST_DISTANCE,
            recreate_if_exists=True # ¡CUIDADO! Limpia la colección en cada prueba
        )

        if created:
            logger.info(f"Colección '{TEST_COLLECTION}' lista.")

            # 3. Preparar Puntos de Prueba
            # CORRECCIÓN: Asegurar que el tipo PointStruct esté disponible o usar Dict
            points_data = []
            if PointStruct and models: # Verificar que se importaron
                 points_data = [
                     models.PointStruct(id=str(uuid.uuid4()), vector=[0.1] * TEST_VECTOR_SIZE, payload={"text": "punto 1", "num": 1}),
                     models.PointStruct(id=str(uuid.uuid4()), vector=[0.2] * TEST_VECTOR_SIZE, payload={"text": "punto 2", "num": 2}),
                     models.PointStruct(id=str(uuid.uuid4()), vector=[0.3] * TEST_VECTOR_SIZE, payload={"text": "punto 3", "num": 3}),
                     models.PointStruct(id=str(uuid.uuid4()), vector=[0.4] * TEST_VECTOR_SIZE, payload={"text": "punto 4", "num": 4}),
                     models.PointStruct(id=str(uuid.uuid4()), vector=[0.5] * TEST_VECTOR_SIZE, payload={"text": "punto 5", "num": 5}),
                 ]
            point_ids_to_delete = [points_data[1].id, points_data[3].id] if points_data else []

            if points_data:
                # 4. Probar Upsert en Lotes
                upsert_ok, upsert_err = batch_upsert(q_client, TEST_COLLECTION, points_data, batch_size=2)
                logger.info(f"Resultado Upsert: Exitosos={upsert_ok}, Errores={upsert_err}")

                time.sleep(1) # Pausa

                # Verificar conteo
                try:
                     count = q_client.count(collection_name=TEST_COLLECTION, exact=True).count
                     logger.info(f"Conteo post-upsert en '{TEST_COLLECTION}': {count} (Esperado: {len(points_data)})")
                except Exception as ce: logger.error(f"Error obteniendo conteo post-upsert: {ce}")

                # 5. Probar Delete
                if point_ids_to_delete:
                     delete_ok, delete_err = delete_points(q_client, TEST_COLLECTION, point_ids_to_delete, batch_size=3) # Probar otro batch size
                     logger.info(f"Resultado Delete: Exitosos={delete_ok}, Errores={delete_err}")

                     # Verificar conteo final
                     try:
                          count = q_client.count(collection_name=TEST_COLLECTION, exact=True).count
                          expected_final = len(points_data) - len(point_ids_to_delete)
                          logger.info(f"Conteo final en '{TEST_COLLECTION}': {count} (Esperado: {expected_final})")
                     except Exception as ce: logger.error(f"Error obteniendo conteo final: {ce}")
                else:
                     logger.warning("No se generaron IDs para probar delete.")
            else:
                logger.warning("No se pudieron preparar puntos de prueba (¿qdrant-client no importado?).")
        else:
            logger.error(f"No se pudo crear/asegurar la colección '{TEST_COLLECTION}'.")

    else:
        logger.error(f"No se pudo conectar a Qdrant en {TEST_QDRANT_URL}. Asegúrate de que esté corriendo.")