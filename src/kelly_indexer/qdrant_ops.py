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
from typing import List, Optional, Dict, Any, Tuple

# Importar dependencias de Qdrant
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import PointStruct, Distance, VectorParams, UpdateStatus
    # Importar excepciones específicas si se usan para manejo más fino
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    print("[ERROR CRÍTICO] Librería 'qdrant-client' no instalada. Este módulo es esencial. Ejecuta: pip install qdrant-client")
    # Definir dummies para evitar errores de importación, pero el código fallará
    QdrantClient = None # type: ignore
    models = None # type: ignore
    PointStruct = Dict # type: ignore # Usar Dict como fallback para type hints
    Distance = None # type: ignore
    VectorParams = None # type: ignore
    UpdateStatus = None # type: ignore
    UnexpectedResponse = ConnectionError # type: ignore

# Obtener logger
logger = logging.getLogger(__name__)

# Mapeo de nombres de métrica de distancia (config) a enums de Qdrant
DISTANCE_MAP = {
    "Cosine": models.Distance.COSINE if models else None,
    "Dot": models.Distance.DOT if models else None,
    "Euclid": models.Distance.EUCLID if models else None,
}

def initialize_client(url: str, api_key: Optional[str] = None, timeout: int = 60) -> Optional[QdrantClient]:
    """
    Inicializa y devuelve un cliente Qdrant conectado a la instancia especificada.

    Args:
        url: La URL completa de la instancia Qdrant (ej. "http://localhost:6333" o URL de Cloud).
        api_key: La clave API si la instancia está protegida (opcional).
        timeout: Timeout en segundos para las operaciones del cliente.

    Returns:
        Una instancia de QdrantClient si la conexión es exitosa, None en caso contrario.
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
            # Podrían añadirse otros parámetros como prefer_grpc=True si se desea
        )
        # Realizar una operación simple para verificar la conexión (opcional pero recomendado)
        # client.list_collections() # Esto lanzará excepción si no conecta
        logger.info("Cliente Qdrant inicializado exitosamente.")
        return client
    except ValueError as e:
         # Errores comunes: URL inválida
         logger.error(f"Error de valor al inicializar cliente Qdrant (¿URL inválida?): {e}")
         return None
    except Exception as e:
        # Capturar otras excepciones (conexión, autenticación si la clave es inválida, etc.)
        logger.exception(f"Error inesperado al inicializar cliente Qdrant para {url}: {e}")
        return None

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance_metric_str: str = "Cosine",
    recreate_if_exists: bool = False # Peligroso, usar con cuidado
) -> bool:
    """
    Asegura que una colección exista en Qdrant con los parámetros especificados.
    La crea si no existe. Por defecto, NO la recrea si ya existe.

    Args:
        client: Instancia del cliente Qdrant.
        collection_name: Nombre de la colección.
        vector_size: Dimensión de los vectores que almacenará la colección.
        distance_metric_str: Nombre de la métrica de distancia ("Cosine", "Dot", "Euclid").
        recreate_if_exists: Si es True, borra y recrea la colección si ya existe.
                            ¡PRECAUCIÓN: ESTO BORRA TODOS LOS DATOS EXISTENTES!

    Returns:
        True si la colección existe o fue creada exitosamente, False en caso contrario.
    """
    if not client or not models or not Distance or not VectorParams:
        logger.error("Cliente Qdrant o modelos no inicializados correctamente.")
        return False

    # Validar y obtener la métrica de distancia enum
    distance_metric = DISTANCE_MAP.get(distance_metric_str.capitalize())
    if distance_metric is None:
        logger.error(f"Métrica de distancia '{distance_metric_str}' no válida. Usar: Cosine, Dot, Euclid.")
        return False

    logger.info(f"Asegurando existencia de colección '{collection_name}'...")
    try:
        collection_exists = client.collection_exists(collection_name=collection_name)

        if collection_exists:
            if recreate_if_exists:
                logger.warning(f"Colección '{collection_name}' ya existe y recreate_if_exists=True. ¡RECREANDO (borrando datos)!")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
                    # Podrían añadirse más configuraciones aquí (quantization, hnsw_config, etc.)
                )
                logger.info(f"Colección '{collection_name}' recreada exitosamente.")
                return True
            else:
                logger.info(f"Colección '{collection_name}' ya existe. No se recreará.")
                # Opcional: Verificar si la configuración existente coincide
                try:
                     collection_info = client.get_collection(collection_name=collection_name)
                     existing_config = collection_info.vectors_config
                     # Qdrant > 1.9 devuelve un dict, antes un objeto VectorParams/dict anidado
                     if isinstance(existing_config, dict): # Para Qdrant >= 1.9
                         # Puede haber múltiples vectores, buscar el default o el único
                         params_key = 'params'
                         if len(existing_config) == 1: # Si solo hay config default ''
                             params_key = list(existing_config.keys())[0] # Nombre del vector, '' si es default

                         if isinstance(existing_config.get(params_key), models.VectorParams): # Cliente aún puede devolver objeto
                            existing_size = existing_config[params_key].size
                            existing_distance = existing_config[params_key].distance
                         elif isinstance(existing_config.get(params_key), dict): # Cliente puede devolver dict
                            existing_size = existing_config[params_key].get('size')
                            existing_distance_str = existing_config[params_key].get('distance', '').upper()
                            existing_distance = getattr(models.Distance, existing_distance_str, None)
                         else: # Manejar caso inesperado
                             existing_size = None
                             existing_distance = None
                     elif isinstance(existing_config, models.VectorParams): # Versiones anteriores cliente/servidor
                         existing_size = existing_config.size
                         existing_distance = existing_config.distance
                     else:
                         existing_size = None
                         existing_distance = None


                     if existing_size != vector_size or existing_distance != distance_metric:
                          logger.warning(f"¡Discrepancia de Configuración! La colección '{collection_name}' existe pero con parámetros diferentes.")
                          logger.warning(f"  -> Esperado: size={vector_size}, distance={distance_metric}")
                          logger.warning(f"  -> Encontrado: size={existing_size}, distance={existing_distance}")
                          logger.warning("El indexador podría fallar o comportarse de forma inesperada.")
                          # Considerar devolver False o lanzar un error si la discrepancia es crítica
                          # return False
                     else:
                          logger.debug("Configuración de vector de colección existente coincide.")

                except Exception as info_err:
                     logger.warning(f"No se pudo verificar la configuración de la colección existente '{collection_name}': {info_err}")

                return True # La colección existe
        else:
            logger.info(f"Colección '{collection_name}' no encontrada. Creando...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
            )
            logger.info(f"Colección '{collection_name}' creada exitosamente.")
            return True

    except UnexpectedResponse as e:
        logger.error(f"Error inesperado de Qdrant al operar sobre la colección '{collection_name}': Status={e.status_code}, Contenido={e.content}")
        return False
    except Exception as e:
        logger.exception(f"Error inesperado al asegurar la colección '{collection_name}': {e}")
        return False


def batch_upsert(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 128,
    wait: bool = True
) -> Tuple[int, int]:
    """
    Realiza upsert (insertar o actualizar) de puntos en Qdrant en lotes.

    Args:
        client: Instancia del cliente Qdrant.
        collection_name: Nombre de la colección destino.
        points: Lista de objetos PointStruct a subir.
        batch_size: Tamaño de cada lote enviado a Qdrant.
        wait: Si es True, espera a que la operación se complete en el servidor.

    Returns:
        Una tupla (puntos_exitosos, errores_totales).
        Nota: La API de Qdrant no siempre informa de errores por punto en upsert,
              un error general podría afectar a todo el lote.
    """
    if not client or not models: return 0, len(points) # Devuelve 0 éxitos, todos como error si no hay cliente
    if not points: return 0, 0 # Nada que hacer

    total_points = len(points)
    successful_points = 0
    total_errors = 0
    processed_batches = 0

    logger.info(f"Iniciando upsert de {total_points} puntos en colección '{collection_name}' (lotes de {batch_size})...")

    for i in range(0, total_points, batch_size):
        batch = points[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.debug(f"Procesando lote {batch_num} ({len(batch)} puntos)...")
        try:
            # La operación upsert puede lanzar excepciones o devolver un status
            status = client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=wait # Esperar confirmación del servidor
            )
            # Verificar el status devuelto (puede variar según versión del cliente/servidor)
            if hasattr(status, 'status') and status.status == UpdateStatus.COMPLETED:
                 logger.debug(f"Lote {batch_num} completado exitosamente.")
                 successful_points += len(batch) # Asumir éxito para todo el lote si no hay excepción
            elif isinstance(status, dict) and status.get('status') == 'ok': # Otro formato posible
                 logger.debug(f"Lote {batch_num} completado exitosamente (status dict ok).")
                 successful_points += len(batch)
            else:
                 # Si no hay excepción pero el status no es completado/ok (raro para upsert con wait=True)
                 logger.warning(f"Estado inesperado devuelto por upsert para lote {batch_num}: {status}")
                 # Considerar si contar esto como error o éxito parcial si es posible
                 # Por seguridad, lo contamos como error para el lote.
                 total_errors += len(batch)

        except UnexpectedResponse as e:
            logger.error(f"Error inesperado de Qdrant en upsert (lote {batch_num}): Status={e.status_code}, Contenido={e.content}")
            total_errors += len(batch) # Asumir que todo el lote falló
        except Exception as e:
            logger.exception(f"Error inesperado durante upsert del lote {batch_num}: {e}")
            total_errors += len(batch) # Asumir fallo del lote completo
        finally:
            processed_batches += 1

    logger.info(f"Upsert finalizado. Procesados {processed_batches} lotes.")
    # Ajustar el conteo de éxitos si hubo errores
    actual_success = max(0, total_points - total_errors)
    if actual_success != successful_points:
         logger.warning(f"Ajustando conteo de éxitos de {successful_points} a {actual_success} debido a errores detectados.")
         successful_points = actual_success


    return successful_points, total_errors

def delete_points(
    client: QdrantClient,
    collection_name: str,
    point_ids: List[Union[str, int]], # IDs pueden ser UUIDs (str) o ints
    wait: bool = True
) -> Tuple[int, int]:
    """
    Elimina puntos de una colección Qdrant por sus IDs.

    Args:
        client: Instancia del cliente Qdrant.
        collection_name: Nombre de la colección.
        point_ids: Lista de IDs de los puntos a eliminar.
        wait: Si es True, espera a que la operación se complete en el servidor.

    Returns:
        Una tupla (puntos_exitosos, errores_totales).
        Similar a upsert, un error general puede afectar a toda la operación.
    """
    if not client or not models: return 0, len(point_ids)
    if not point_ids: return 0, 0

    total_to_delete = len(point_ids)
    successful_deletes = 0
    total_errors = 0

    logger.info(f"Iniciando eliminación de {total_to_delete} puntos de colección '{collection_name}'...")

    try:
        # La operación delete puede manejar una lista de IDs directamente
        status = client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=wait
        )
        # Verificar status (similar a upsert)
        if hasattr(status, 'status') and status.status == UpdateStatus.COMPLETED:
            logger.info(f"Eliminación completada exitosamente para {total_to_delete} IDs.")
            successful_deletes = total_to_delete
        elif isinstance(status, dict) and status.get('status') == 'ok':
             logger.info(f"Eliminación completada exitosamente (status dict ok) para {total_to_delete} IDs.")
             successful_deletes = total_to_delete
        else:
            logger.warning(f"Estado inesperado devuelto por delete: {status}")
            total_errors = total_to_delete # Asumir fallo total si el status no es claro

    except UnexpectedResponse as e:
        # Puede ocurrir si la colección no existe, o por otros errores 4xx/5xx
        logger.error(f"Error inesperado de Qdrant en delete: Status={e.status_code}, Contenido={e.content}")
        total_errors = total_to_delete
    except Exception as e:
        logger.exception(f"Error inesperado durante la eliminación de puntos: {e}")
        total_errors = total_to_delete

    # Ajustar conteo
    actual_success = max(0, total_to_delete - total_errors)
    if actual_success != successful_deletes:
         # Este caso es menos probable en delete si falla todo o nada
         logger.warning(f"Ajustando conteo de eliminaciones exitosas de {successful_deletes} a {actual_success}.")
         successful_deletes = actual_success

    return successful_deletes, total_errors


# --- Bloque para pruebas rápidas (requiere instancia Qdrant corriendo) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Probando Módulo Qdrant Ops (Requiere Qdrant) ---")

    # --- Configuración Manual para Prueba (¡Usa .env en la práctica!) ---
    TEST_QDRANT_URL = "http://localhost:6333" # O tu URL Cloud
    TEST_API_KEY = None # Pon tu clave aquí si es necesaria
    TEST_COLLECTION = "test-collection-ops"
    TEST_VECTOR_SIZE = 384 # Debe coincidir con tu modelo
    TEST_DISTANCE = "Cosine"

    # 1. Inicializar Cliente
    q_client = initialize_client(TEST_QDRANT_URL, TEST_API_KEY)

    if q_client:
        # 2. Asegurar Colección
        created = ensure_collection(
            q_client, TEST_COLLECTION, TEST_VECTOR_SIZE, TEST_DISTANCE,
            recreate_if_exists=True # ¡CUIDADO! Borra datos si ya existe
        )

        if created:
            logger.info(f"Colección '{TEST_COLLECTION}' lista.")

            # 3. Preparar Puntos de Prueba (UUIDs como strings)
            points_data = [
                models.PointStruct(id=str(uuid.uuid4()), vector=[0.1] * TEST_VECTOR_SIZE, payload={"text": "punto 1"}),
                models.PointStruct(id=str(uuid.uuid4()), vector=[0.2] * TEST_VECTOR_SIZE, payload={"text": "punto 2"}),
                models.PointStruct(id=str(uuid.uuid4()), vector=[0.3] * TEST_VECTOR_SIZE, payload={"text": "punto 3"}),
                # Añadir más puntos para probar lotes
                models.PointStruct(id=str(uuid.uuid4()), vector=[0.4] * TEST_VECTOR_SIZE, payload={"text": "punto 4"}),
                models.PointStruct(id=str(uuid.uuid4()), vector=[0.5] * TEST_VECTOR_SIZE, payload={"text": "punto 5"}),
            ]
            point_ids_to_delete = [points_data[1].id, points_data[3].id] # IDs a borrar después

            # 4. Probar Upsert en Lotes
            upsert_ok, upsert_err = batch_upsert(q_client, TEST_COLLECTION, points_data, batch_size=2)
            logger.info(f"Resultado Upsert: Exitosos={upsert_ok}, Errores={upsert_err}")

            # Pequeña pausa
            time.sleep(1)

            # Verificar conteo (opcional)
            try:
                 count = q_client.count(collection_name=TEST_COLLECTION, exact=True).count
                 logger.info(f"Conteo actual en colección '{TEST_COLLECTION}': {count}")
            except Exception as ce:
                 logger.error(f"Error obteniendo conteo: {ce}")


            # 5. Probar Delete
            delete_ok, delete_err = delete_points(q_client, TEST_COLLECTION, point_ids_to_delete)
            logger.info(f"Resultado Delete: Exitosos={delete_ok}, Errores={delete_err}")

             # Verificar conteo final (opcional)
            try:
                 count = q_client.count(collection_name=TEST_COLLECTION, exact=True).count
                 logger.info(f"Conteo final en colección '{TEST_COLLECTION}': {count}")
                 # Debería ser 5 - 2 = 3 si todo funcionó
            except Exception as ce:
                 logger.error(f"Error obteniendo conteo final: {ce}")


        else:
            logger.error(f"No se pudo crear/asegurar la colección '{TEST_COLLECTION}'.")

        # Cerrar cliente (si es necesario o buena práctica, aunque a menudo no es requerido)
        # q_client.close()

    else:
        logger.error(f"No se pudo conectar a Qdrant en {TEST_QDRANT_URL}.")