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
import time # Necesario para medir duración y para pausas
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Set

# --- Importación de TQDM ---
tqdm_available = False
tqdm = lambda x, **kwargs: x # Dummy inicial que solo devuelve el iterable
try:
    from tqdm import tqdm as tqdm_real
    tqdm = tqdm_real # type: ignore # Sobrescribir si la importación fue exitosa
    tqdm_available = True
except ImportError:
    # No es necesario imprimir advertencia aquí, el logger lo hará si es relevante
    pass
# ---------------------------

# --- Ajuste de PYTHONPATH e Importaciones ---
try:
    project_root = Path(__file__).parent.parent.parent.resolve()
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
    try:
         from kelly_indexer.utils.logging_setup import setup_logging
    except ImportError:
         setup_logging = None # Marcar como no disponible

    from pydantic import ValidationError
    try:
        from qdrant_client.http.models import PointStruct
    except ImportError:
        PointStruct = Dict # Fallback

except ImportError as e:
    print(f"[ERROR CRÍTICO] No se pudieron importar módulos necesarios: {e}")
    print("Verifica la estructura del proyecto, que 'src' esté accesible y que las dependencias estén instaladas ('pip install -e .').")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR CRÍTICO] Error inesperado durante la importación: {e}")
    sys.exit(1)

# --- Configuración Inicial de Logging ---
logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
logger = logging.getLogger("index_qdrant")

def main():
    """Función principal que orquesta el proceso de indexación."""
    start_time = time.time()
    logger.info("Iniciando script de indexación Kelly Indexer...")

    # --- Cargar Configuración ---
    settings: Optional[Settings] = None
    try:
        settings = Settings() # Instanciar aquí

        # Reconfigurar logging usando valores de settings
        if setup_logging:
             setup_logging(log_level_str=settings.log_level, log_file=settings.log_file)
             logger.info("Logging reconfigurado usando utils.logging_setup.")
        else:
             log_level_str = settings.log_level.upper()
             log_file_path = settings.log_file
             log_handlers = [logging.StreamHandler(sys.stderr)]
             if log_file_path:
                  try:
                      log_file_path.parent.mkdir(parents=True, exist_ok=True)
                      log_handlers.append(logging.FileHandler(log_file_path, mode='a', encoding='utf-8'))
                      print(f"[INFO] Logueando también a: {log_file_path}")
                  except Exception as log_e: logger.error(f"No se pudo configurar log a archivo {log_file_path}: {log_e}")
             logging.basicConfig(level=log_level_str, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s', handlers=log_handlers, force=True)
             logger.info(f"Logging reconfigurado manualmente a nivel {log_level_str}.")

        logger.info("Configuración cargada exitosamente.")

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
    parser.add_argument("--source", type=Path, default=settings.input_json_dir, help=f"Directorio raíz JSON Q&A.")
    parser.add_argument("--state-file", type=Path, default=settings.state_file_path, help=f"Archivo JSON de estado.")
    parser.add_argument("--batch-size", type=int, default=settings.qdrant_batch_size, help="Tamaño de lote Qdrant.")
    parser.add_argument("--force-reindex", action="store_true", help="Forzar reindexación total ignorando estado.")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin modificar Qdrant ni estado.")
    args = parser.parse_args()

    # --- Validaciones Iniciales ---
    if not args.source.is_dir():
        logger.critical(f"CRÍTICO: El directorio fuente '{args.source}' no existe. Saliendo.")
        sys.exit(1)
    try:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        settings.input_dir_processed.mkdir(parents=True, exist_ok=True) # Usar de settings
        settings.output_dir_reports.mkdir(parents=True, exist_ok=True) # Usar de settings
    except Exception as e:
         logger.critical(f"CRÍTICO: No se pudieron crear directorios necesarios: {e}. Saliendo.")
         sys.exit(1)

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
        embed_model = embeddings.get_embedding_model(settings.embedding_model_name)
        if embed_model is None: raise RuntimeError("No se pudo cargar modelo embeddings.")
        logger.info(f"Modelo embeddings '{settings.embedding_model_name}' cargado.")

        qdrant_client = qdrant_ops.initialize_client(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )
        if qdrant_client is None: raise ConnectionError("No se pudo inicializar cliente Qdrant.")
        logger.info(f"Cliente Qdrant conectado a: {settings.qdrant_url.host}")

        chunker = text_chunker.get_answer_chunker(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        if chunker is None: raise RuntimeError("No se pudo inicializar text chunker.")
        logger.info("Chunker de texto inicializado.")
        logger.info("Gestor de estado (basado en funciones) listo.")

    except Exception as e:
        logger.critical(f"Error CRÍTICO durante inicialización: {e}", exc_info=True)
        sys.exit(1)

    # --- Lógica Principal de Indexación ---
    total_processed_files = 0
    total_qas_found = 0
    error_count = 0
    upserted_count = 0
    deleted_count = 0
    qas_for_processing: List[Dict] = []
    ids_to_delete: List[str] = []
    current_points_details: Dict[str, Dict] = {}

    try:
        # 1. Cargar estado anterior
        logger.info(f"Cargando estado anterior desde {args.state_file}...")
        previous_state = state_manager.load_state(args.state_file)
        previous_indexed_points = previous_state.get('indexed_points', {})
        logger.info(f"Encontrados {len(previous_indexed_points)} puntos en estado anterior.")

        # 2. Cargar Q&As actuales
        logger.info(f"Escaneando y cargando Q&As desde {args.source}...")
        all_current_qas_map = data_loader.load_all_qas_from_directory(args.source)
        total_processed_files = len(all_current_qas_map)
        logger.info(f"Lectura completada. {total_processed_files} archivos JSON procesados.")

        # 3. Calcular Diff
        logger.info("Comparando Q&As actuales con estado anterior...")
        qas_for_processing, ids_to_delete, current_points_details = state_manager.calculate_diff(
            all_current_qas_map, previous_indexed_points
        )
        total_qas_found = sum(len(v) for v in all_current_qas_map.values())

        # 4. Aplicar Force Reindex si es necesario
        if args.force_reindex and not args.dry_run: # No recalcular en dry run, solo mostrar intención
             logger.warning("Forzando reindexación (--force-reindex)...")
             qas_for_processing = []
             current_points_details = {}
             all_current_point_ids: Set[str] = set()
             total_qas_found = 0
             for file_rel_path, qa_list in all_current_qas_map.items():
                 total_qas_found += len(qa_list)
                 for qa_item in qa_list:
                     question = qa_item.get('q')
                     if not question or not isinstance(question, str): continue
                     point_id = state_manager.generate_qa_uuid(question, file_rel_path)
                     q_hash = state_manager.generate_content_hash(question)
                     all_current_point_ids.add(point_id)
                     current_points_details[point_id] = {"source_file": file_rel_path, "question_hash": q_hash}
                     qa_item['_id'] = point_id
                     qa_item['_question_hash'] = q_hash
                     qa_item['_source_file'] = file_rel_path
                     qas_for_processing.append(qa_item)
             previous_ids = set(previous_state.get('indexed_points', {}).keys())
             ids_to_delete = list(previous_ids - all_current_point_ids)
             logger.warning(f"Force-reindex: {len(qas_for_processing)} Q&As marcados para upsert.")
             logger.warning(f"Force-reindex: {len(ids_to_delete)} puntos del estado anterior marcados para eliminación.")
        elif args.force_reindex and args.dry_run:
             # Simular el cálculo para el reporte dry run
             _qfp, _idt, _cpd = state_manager.calculate_diff(all_current_qas_map, {}) # Simular sin estado previo
             qas_for_processing_simulated = len(_qfp)
             ids_to_delete_simulated = len(previous_indexed_points) # Todos los anteriores serían borrados
             logger.warning(f"Dry Run con Force-reindex: Se intentarían {qas_for_processing_simulated} upserts y {ids_to_delete_simulated} deletes.")
             # Usar los calculados sin force para el resto del dry run (preparación)
        else:
             # Log normal del diff
             logger.info(f"Comparación completada. Total Q&As evaluados: {total_qas_found}. A procesar: {len(qas_for_processing)}. A eliminar: {len(ids_to_delete)}.")


        # 5. Preparar Lotes para Qdrant
        points_to_upsert: List[PointStruct] = []
        if qas_for_processing:
            logger.info("Preparando puntos para Qdrant (generando embeddings y chunking)...")
            questions_to_embed = [qa['q'] for qa in qas_for_processing]

            # Generar embeddings en lote
            logger.info(f"Generando {len(questions_to_embed)} embeddings...")
            question_vectors = embeddings.generate_embeddings(embed_model, questions_to_embed, batch_size=args.batch_size // 2 or 16)
            logger.info("Embeddings generados.")

            if question_vectors is None or len(question_vectors) != len(qas_for_processing):
                 logger.error("Fallo en embeddings o discrepancia de número. Abortando.")
                 raise RuntimeError("Fallo crítico en embeddings.")

            # Construir PointStructs
            # CORRECCIÓN: Usar tqdm condicionalmente
            iterable_qas = qas_for_processing
            disable_tqdm = not tqdm_available or (logging.getLogger().getEffectiveLevel() > logging.INFO)
            if not disable_tqdm:
                iterable_qas = tqdm(qas_for_processing, desc="Construyendo puntos Qdrant", unit="q&a")

            logger.info("Construyendo puntos Qdrant (PointStructs)...")
            for i, qa_item in enumerate(iterable_qas):
                try:
                    answer_content = qa_item['a']
                    answer_chunks = text_chunker.chunk_text(chunker, answer_content)
                    if not answer_chunks:
                         logger.warning(f"No se generaron chunks para Q/A ID: {qa_item['_id']}. Usando respuesta original.")
                         answer_chunks = [answer_content]

                    payload = {
                        "question": qa_item['q'],
                        "answer": answer_chunks,
                        "product": qa_item.get('product', 'General'),
                        "keywords": qa_item.get('keywords', []),
                        "source": qa_item['_source_file']
                    }
                    if PointStruct is Dict: # Fallback si qdrant no importó PointStruct
                        point = {"id": qa_item['_id'], "vector": question_vectors[i].tolist(), "payload": payload}
                    else:
                         point = PointStruct(id=qa_item['_id'], vector=question_vectors[i].tolist(), payload=payload)
                    points_to_upsert.append(point) # type: ignore

                except Exception as e:
                     logger.error(f"Error preparando punto Qdrant (ID: {qa_item.get('_id', 'N/A')}, Archivo: {qa_item.get('_source_file', 'N/A')}): {e}", exc_info=False) # exc_info=False para no llenar log
                     error_count += 1
            logger.info(f"Preparados {len(points_to_upsert)} puntos para upsert (errores preparación: {error_count}).")


        # 6. Ejecutar Operaciones Qdrant (si no es dry run)
        final_state_to_save: Dict[str, Any] = {"indexed_points": current_points_details} # Estado deseado final

        if args.dry_run:
            logger.warning("--- DRY RUN ACTIVADO ---")
            upserted_count = len(points_to_upsert) # Simular éxito para reporte
            deleted_count = len(ids_to_delete)
            logger.warning(f"Simulación: Se realizaría upsert de {upserted_count} puntos.")
            logger.warning(f"Simulación: Se eliminarían {deleted_count} puntos.")
            logger.warning("No se harán cambios en Qdrant ni en el archivo de estado.")
        else:
            logger.info("Ejecutando operaciones en Qdrant...")
            # Asegurar colección
            collection_ok = qdrant_ops.ensure_collection(
                client=qdrant_client, collection_name=settings.qdrant_collection_name,
                vector_size=settings.vector_dimension, distance_metric_str=settings.distance_metric
            )
            if not collection_ok:
                 logger.critical(f"No se pudo asegurar colección '{settings.qdrant_collection_name}'. Abortando.")
                 raise RuntimeError("Fallo colección Qdrant.")

            # Realizar Upsert
            upsert_errors = 0
            if points_to_upsert:
                upserted_count, upsert_errors = qdrant_ops.batch_upsert(
                    client=qdrant_client, collection_name=settings.qdrant_collection_name,
                    points=points_to_upsert, batch_size=args.batch_size
                )
                error_count += upsert_errors
            else: logger.info("No hay puntos nuevos/modificados para upsert.")

            # Realizar Delete
            delete_errors = 0
            if ids_to_delete:
                deleted_count, delete_errors = qdrant_ops.delete_points(
                    client=qdrant_client, collection_name=settings.qdrant_collection_name,
                    point_ids=ids_to_delete, batch_size=args.batch_size
                )
                error_count += delete_errors
            else: logger.info("No hay puntos obsoletos para eliminar.")

            # 7. Guardar Estado Final (solo si no es dry run)
            logger.info("Guardando estado final...")
            save_success = state_manager.save_state(args.state_file, final_state_to_save)
            if not save_success:
                 logger.error("¡FALLO AL GUARDAR EL ARCHIVO DE ESTADO!")
                 error_count += 1
            else:
                 logger.info(f"Nuevo estado guardado en {args.state_file}")


    except Exception as e:
        logger.exception(f"Error CRÍTICO inesperado durante el proceso principal: {e}")
        error_count += 1
        # Intentar asignar valores para el reporte final si existen
        upserted_count = locals().get("upserted_count", 0)
        deleted_count = locals().get("deleted_count", 0)

    # --- Reporte Final ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info("--- Resumen Final de Indexación ---")
    logger.info(f"Tiempo Total: {duration:.2f} segundos")
    logger.info(f"Archivos JSON Fuente Procesados: {total_processed_files}")
    logger.info(f"Total Q&As Encontrados en Fuente: {total_qas_found}")
    logger.info(f"Q&As Nuevos/Modificados para Upsert (Intentados): {len(qas_for_processing)}") # Preparados
    logger.info(f"Q&As Exitosamente Subidos/Actualizados (Upserted): {upserted_count if not args.dry_run else 'N/A (Dry Run)'}")
    logger.info(f"Q&As Marcados para Eliminación: {len(ids_to_delete)}")
    logger.info(f"Q&As Exitosamente Eliminados: {deleted_count if not args.dry_run else 'N/A (Dry Run)'}")
    logger.info(f"Errores Totales Durante el Proceso: {error_count}")
    if args.dry_run: logger.warning("La ejecución fue en modo --dry-run. No se hicieron cambios reales.")
    if error_count > 0: logger.error("El proceso finalizó con errores.")
    else: logger.info("El proceso finalizó exitosamente.")


if __name__ == "__main__":
    main()