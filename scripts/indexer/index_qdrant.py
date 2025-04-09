# kelly_indexer/scripts/indexer/index_qdrant.py
# -*- coding: utf-8 -*-

"""
Script principal para indexar pares Q&A desde archivos JSON a Qdrant Cloud.

Flujo:
1. Carga configuración (.env).
2. Carga estado de indexación anterior (si existe).
3. Escanea directorio de entrada recursivamente buscando archivos JSON Q&A.
4. Compara Q&As encontradas con el estado anterior para identificar nuevas, modificadas o eliminadas.
5. Genera embeddings para preguntas nuevas/modificadas usando SentenceTransformer.
6. (Opcional) Divide respuestas largas en chunks.
7. Prepara lotes de puntos para Qdrant (upsert) y lotes de IDs (delete).
8. Interactúa con Qdrant Cloud para realizar upsert/delete.
9. Guarda el nuevo estado de indexación.
10. Reporta un resumen.
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

# --- Ajuste de PYTHONPATH e Importaciones ---
# Asegurar que podamos importar desde src/kelly_indexer
try:
    project_root = Path(__file__).parent.parent.parent.resolve() # Sube dos niveles desde scripts/indexer/
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Importar módulos del proyecto
    from kelly_indexer.config import Settings # Importar la CLASE
    from kelly_indexer import data_loader
    from kelly_indexer import embeddings
    from kelly_indexer import qdrant_ops
    from kelly_indexer import state_manager
    from kelly_indexer import text_chunker
    # Opcional: importar setup de logging si existe
    # from kelly_indexer.utils.logging_setup import setup_logging

    # Importar dependencias de terceros
    from pydantic import ValidationError
    from qdrant_client.http.models import PointStruct # Para type hints

except ImportError as e:
    print(f"[ERROR CRÍTICO] No se pudieron importar módulos necesarios: {e}")
    print("Verifica la estructura del proyecto, que 'src' esté accesible y que las dependencias estén instaladas ('pip install -e .').")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR CRÍTICO] Error inesperado durante la importación: {e}")
    sys.exit(1)

# Configurar logger básico inicial (será reconfigurado después de cargar Settings)
logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("index_qdrant")

def main():
    """Función principal que orquesta el proceso de indexación."""
    start_time = time.time()

    # --- Cargar Configuración ---
    settings: Optional[Settings] = None
    try:
        settings = Settings()
        # Reconfigurar logging con valores de settings
        log_level_str = settings.log_level.upper()
        log_file_path = settings.log_file
        log_handlers = [logging.StreamHandler(sys.stderr)]
        if log_file_path:
             try:
                 log_file_path.parent.mkdir(parents=True, exist_ok=True)
                 log_handlers.append(logging.FileHandler(log_file_path, mode='a', encoding='utf-8'))
                 print(f"[INFO] Logueando también a: {log_file_path}")
             except Exception as log_e: logger.error(f"No se pudo configurar log a archivo {log_file_path}: {log_e}")
        logging.basicConfig(level=log_level_str, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=log_handlers, force=True)
        logger.info("Configuración cargada y logging reconfigurado.")

    except ValidationError as e:
        logger.critical(f"Error CRÍTICO de validación al cargar configuración: {e}")
        logger.critical("Verifica tu archivo .env y las variables de entorno.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error CRÍTICO inesperado al cargar configuración: {e}")
        sys.exit(1)

    # --- Parsear Argumentos CLI ---
    parser = argparse.ArgumentParser(
        description="Indexa Q&A desde archivos JSON a Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, default=settings.input_json_dir,
        help="Directorio raíz que contiene los archivos JSON Q&A a indexar."
    )
    parser.add_argument(
        "--state-file", type=Path, default=settings.state_file_path,
        help="Archivo JSON para guardar/leer el estado de indexación."
    )
    parser.add_argument(
        "--batch-size", type=int, default=settings.qdrant_batch_size,
        help="Tamaño del lote para subir puntos a Qdrant."
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Forzar la reindexación de todos los Q&A encontrados, ignorando el estado anterior."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecutar el proceso (leer, comparar, preparar) pero sin modificar Qdrant ni el archivo de estado."
    )
    # Se podrían añadir más overrides: --collection-name, --model-name, etc.

    args = parser.parse_args()

    logger.info("--- Iniciando Proceso de Indexación Qdrant ---")
    logger.info(f"Directorio Fuente JSON: {args.source.resolve()}")
    logger.info(f"Archivo de Estado: {args.state_file.resolve()}")
    logger.info(f"Tamaño de Lote Qdrant: {args.batch_size}")
    logger.info(f"Modelo Embeddings: {settings.embedding_model_name} (Dim: {settings.vector_dimension})")
    logger.info(f"Colección Qdrant: {settings.qdrant_collection_name}")
    logger.info(f"Forzar Reindexación: {'Sí' if args.force_reindex else 'No'}")
    logger.info(f"Dry Run (Simulación): {'Sí' if args.dry_run else 'No'}")

    # --- Inicializar Componentes ---
    try:
        logger.info("Inicializando componentes...")
        # Modelo de Embeddings
        embed_model = embeddings.get_embedding_model(settings.embedding_model_name)
        if embed_model is None: raise RuntimeError("No se pudo cargar el modelo de embeddings.")
        logger.info(f"Modelo de embeddings '{settings.embedding_model_name}' cargado.")

        # Cliente Qdrant
        qdrant_client = qdrant_ops.initialize_client(
            url=str(settings.qdrant_url), # Convertir HttpUrl a string
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )
        if qdrant_client is None: raise ConnectionError("No se pudo inicializar el cliente Qdrant.")
        logger.info(f"Cliente Qdrant conectado a: {settings.qdrant_url.host}")

        # Text Chunker (si se usa)
        chunker = text_chunker.get_answer_chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        logger.info("Chunker de texto inicializado.")

        # State Manager
        state_manager_instance = state_manager.StateManager(args.state_file)
        logger.info("Gestor de estado inicializado.")

    except Exception as e:
        logger.critical(f"Error CRÍTICO durante la inicialización de componentes: {e}", exc_info=True)
        sys.exit(1)

    # --- Lógica Principal de Indexación ---
    total_processed_files = 0
    total_qas_found = 0
    qas_to_upsert: List[Dict] = [] # Lista de dicts Q&A para procesar
    ids_in_source = set() # IDs de Q&As encontrados en esta ejecución
    error_count = 0

    try:
        # 1. Cargar estado anterior
        logger.info(f"Cargando estado anterior desde {args.state_file}...")
        previous_state = state_manager_instance.load_state()
        logger.info(f"Encontrados {len(previous_state.get('indexed_points', {}))} puntos en el estado anterior.")

        # 2. Cargar todos los Q&A actuales desde los archivos JSON fuente
        logger.info(f"Escaneando y cargando Q&As desde {args.source}...")
        all_current_qas = data_loader.load_all_qas_from_directory(args.source)
        total_processed_files = len(all_current_qas) # Número de archivos leídos
        logger.info(f"Lectura completada. Se procesaron {total_processed_files} archivos JSON fuente.")

        # 3. Comparar y determinar qué indexar/eliminar
        logger.info("Comparando Q&As actuales con estado anterior...")
        qas_for_processing = [] # Lista de Q&A dicts que son nuevos o modificados
        all_current_point_ids = set()

        for file_path, qa_list in all_current_qas.items():
            if not qa_list: continue # Saltar archivos JSON vacíos
            total_qas_found += len(qa_list)
            for qa_item in qa_list:
                 # Generar ID y Hash para este Q&A
                 point_id = state_manager.generate_qa_uuid(qa_item['q'], file_path) # Usar q y ruta para ID
                 question_hash = state_manager.generate_content_hash(qa_item['q']) # Hash solo de la pregunta
                 all_current_point_ids.add(point_id) # Guardar todos los IDs encontrados

                 # Verificar contra estado anterior
                 previous_entry = previous_state.get('indexed_points', {}).get(point_id)
                 should_process = True
                 if args.force_reindex:
                     logger.debug(f"Forzando reindexación para Q/A ID: {point_id} (Archivo: {file_path})")
                 elif previous_entry and previous_entry.get('question_hash') == question_hash:
                     logger.debug(f"Q/A sin cambios (ID: {point_id}, Hash: {question_hash[:8]}...). Saltando.")
                     should_process = False # Existe y no ha cambiado
                 elif previous_entry:
                      logger.info(f"Q/A modificado detectado (ID: {point_id}, Hash anterior != {question_hash[:8]}...). Marcado para reindexar.")
                 else:
                      logger.info(f"Nuevo Q/A detectado (ID: {point_id}). Marcado para indexar.")

                 if should_process:
                     # Añadir metadatos necesarios para el procesamiento posterior
                     qa_item['_id'] = point_id
                     qa_item['_question_hash'] = question_hash
                     qa_item['_source_file'] = file_path # Guardar ruta origen relativa
                     qas_for_processing.append(qa_item)

        # Determinar IDs a eliminar (estaban en el estado pero no se encontraron ahora)
        previous_ids = set(previous_state.get('indexed_points', {}).keys())
        ids_to_delete = list(previous_ids - all_current_point_ids)
        if args.force_reindex:
             # Si forzamos reindex, todos los IDs previos se consideran obsoletos si no están en la fuente actual
             logger.warning(f"Force-reindex: Marcados {len(ids_to_delete)} puntos del estado anterior para eliminación.")
        elif ids_to_delete:
             logger.info(f"Detectados {len(ids_to_delete)} Q&As para eliminar de Qdrant (ya no existen en fuente).")


        logger.info(f"Comparación completada. Total Q&As encontrados: {total_qas_found}. A procesar/reindexar: {len(qas_for_processing)}. A eliminar: {len(ids_to_delete)}.")

        # --- Preparación y Ejecución de Lotes (si hay algo que hacer) ---
        points_to_upsert: List[PointStruct] = []
        processed_state_updates: Dict[str, Dict] = {} # Para guardar el estado de los puntos procesados

        if qas_for_processing:
            logger.info("Preparando puntos para Qdrant (generando embeddings y chunking)...")
            questions_to_embed = [qa['_q'] for qa in qas_for_processing] # Extraer preguntas

            # Generar embeddings en lote
            logger.info(f"Generando {len(questions_to_embed)} embeddings...")
            question_vectors = embeddings.generate_embeddings(embed_model, questions_to_embed)
            logger.info("Embeddings generados.")

            if len(question_vectors) != len(qas_for_processing):
                 logger.error("Discrepancia entre número de Q&As y vectores generados. Abortando upsert.")
                 raise RuntimeError("Fallo en la generación de embeddings.")

            # Construir PointStructs
            for i, qa_item in enumerate(tqdm(qas_for_processing, desc="Construyendo puntos Qdrant", unit="q&a")):
                try:
                    answer_content = qa_item['a']
                    # Chunkear respuesta si es necesario
                    answer_chunks = text_chunker.chunk_text(chunker, answer_content)

                    payload = {
                        "question": qa_item['q'],
                        "answer": answer_chunks, # Guardar como lista de chunks (o lista con un solo chunk)
                        "product": qa_item.get('product', 'General'), # Usar 'General' si falta
                        "keywords": qa_item.get('keywords', []), # Usar lista vacía si falta
                        "source": qa_item['_source_file'] # Incluir ruta relativa del JSON original
                        # Podrías añadir más metadatos si son útiles
                    }
                    point = PointStruct(
                        id=qa_item['_id'],
                        vector=question_vectors[i].tolist(), # Convertir numpy array a lista
                        payload=payload
                    )
                    points_to_upsert.append(point)
                    # Guardar info para actualizar estado si el upsert es exitoso
                    processed_state_updates[qa_item['_id']] = {
                         "source_file": qa_item['_source_file'],
                         "question_hash": qa_item['_question_hash']
                     }
                except Exception as e:
                     logger.error(f"Error preparando punto para Qdrant (ID: {qa_item.get('_id', 'N/A')}, Archivo: {qa_item.get('_source_file', 'N/A')}): {e}")
                     error_count += 1


        # --- Ejecutar Operaciones Qdrant (si no es dry run) ---
        if args.dry_run:
            logger.warning("DRY RUN activado. No se realizarán cambios en Qdrant ni en el archivo de estado.")
            # Simular éxito para el reporte final si no hubo errores de preparación
            upserted_count = len(points_to_upsert) - error_count # Asumir que los preparados se habrían subido
            deleted_count = len(ids_to_delete)
            final_state = previous_state # Mantener estado anterior en dry run
        else:
            logger.info("Ejecutando operaciones en Qdrant...")
            # 1. Asegurar que la colección exista
            qdrant_ops.ensure_collection(
                client=qdrant_client,
                collection_name=settings.qdrant_collection_name,
                vector_size=settings.vector_dimension,
                distance_metric=settings.distance_metric
            )

            # 2. Realizar Upsert en lotes
            upserted_count = 0
            if points_to_upsert:
                logger.info(f"Realizando upsert de {len(points_to_upsert)} puntos en lotes de {args.batch_size}...")
                # La función batch_upsert debería manejar errores internos y devolver el número de éxitos
                upserted_count, upsert_errors = qdrant_ops.batch_upsert(
                    client=qdrant_client,
                    collection_name=settings.qdrant_collection_name,
                    points=points_to_upsert,
                    batch_size=args.batch_size
                )
                error_count += upsert_errors # Sumar errores de Qdrant
                logger.info(f"Upsert completado. Puntos exitosos: {upserted_count}. Errores: {upsert_errors}.")
                # Actualizar el estado solo con los puntos que realmente se procesaron Y se subieron
                # (Asumiendo que batch_upsert no devuelve IDs fallidos, una simplificación)
                # Si batch_upsert fallara por completo, upserted_count sería 0.
                # Si queremos ser precisos, necesitaríamos saber qué IDs fallaron en Qdrant.
                # Por ahora, actualizaremos el estado basado en los preparados si upserted_count > 0.
                if upserted_count == 0 and len(points_to_upsert) > 0:
                     logger.error("Falló el upsert de todos los puntos preparados.")
                     processed_state_updates.clear() # No actualizar estado si todo falló
                elif upsert_errors > 0:
                     logger.warning("Upsert completado con errores. El estado podría no reflejar todos los puntos fallidos.")
                     # Idealmente, aquí filtraríamos `processed_state_updates` para quitar los IDs fallidos si los supiéramos.


            # 3. Realizar Delete en lotes
            deleted_count = 0
            if ids_to_delete:
                logger.info(f"Eliminando {len(ids_to_delete)} puntos obsoletos...")
                # La función delete_points debería manejar errores internos y devolver el número de éxitos
                deleted_count, delete_errors = qdrant_ops.delete_points(
                     client=qdrant_client,
                     collection_name=settings.qdrant_collection_name,
                     point_ids=ids_to_delete
                 )
                error_count += delete_errors
                logger.info(f"Eliminación completada. Puntos eliminados: {deleted_count}. Errores: {delete_errors}.")

            # 4. Actualizar y Guardar Estado
            logger.info("Actualizando y guardando estado...")
            final_state = state_manager_instance.update_state(previous_state, processed_state_updates, set(ids_to_delete))
            save_success = state_manager_instance.save_state(final_state)
            if not save_success:
                 logger.error("¡FALLO AL GUARDAR EL ARCHIVO DE ESTADO! La próxima ejecución podría reprocesar datos.")
                 error_count += 1
            else:
                 logger.info(f"Nuevo estado guardado en {args.state_file}")


    except Exception as e:
        logger.exception(f"Error CRÍTICO inesperado durante el proceso principal de indexación: {e}")
        error_count += 1
        # Asegurar reporte de valores calculados hasta el momento si es posible
        upserted_count = locals().get("upserted_count", 0)
        deleted_count = locals().get("deleted_count", 0)


    # --- Reporte Final ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info("--- Resumen Final de Indexación ---")
    logger.info(f"Tiempo Total: {duration:.2f} segundos")
    logger.info(f"Archivos JSON Fuente Procesados: {total_processed_files}")
    logger.info(f"Total Q&As Encontrados en Fuente: {total_qas_found}")
    logger.info(f"Q&As Nuevos/Modificados para Upsert (Preparados): {len(qas_for_processing)}")
    logger.info(f"Q&As Exitosamente Subidos/Actualizados (Upserted): {upserted_count if not args.dry_run else 'N/A (Dry Run)'}")
    logger.info(f"Q&As Marcados para Eliminación: {len(ids_to_delete)}")
    logger.info(f"Q&As Exitosamente Eliminados: {deleted_count if not args.dry_run else 'N/A (Dry Run)'}")
    logger.info(f"Errores Totales Durante el Proceso: {error_count}")
    if args.dry_run: logger.warning("Recordatorio: La ejecución fue en modo --dry-run. No se hicieron cambios reales.")


if __name__ == "__main__":
    main()