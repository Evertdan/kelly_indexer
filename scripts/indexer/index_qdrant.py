# scripts/indexer/index_qdrant.py
# -*- coding: utf-8 -*-

"""
Script principal para indexar pares Q&A desde archivos JSON a Qdrant Cloud.
MODIFICADO para usar faq_id, texto_para_vectorizar, categoria, generar UUIDs para Qdrant y usar GPU (CUDA).

Flujo:
1. Carga configuración (.env).
2. Carga estado de indexación anterior (basado en faq_id y content_hash).
3. Escanea directorio de entrada buscando JSON Q&A (nueva estructura).
4. Compara Q&As actuales con estado anterior (usando faq_id y content_hash).
5. Detecta dispositivo (CUDA/CPU).
6. Genera embeddings para texto_para_vectorizar (nuevos/modificados) usando SentenceTransformer en dispositivo detectado.
7. Divide respuestas largas ('a') en chunks para payload.
8. Prepara lotes de puntos para Qdrant (con UUIDs como ID) y lotes de IDs (UUIDs) para delete.
9. Interactúa con Qdrant Cloud para realizar upsert/delete.
10. Guarda el nuevo estado de indexación (con faq_id y content_hash).
11. Reporta un resumen.
"""

import argparse
import sys
import os
import time # Necesario para medir duración y para pausas
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Set
import uuid # NUEVO: Para generar UUIDs para Qdrant

# --- Importación de TQDM ---
tqdm_available = False
tqdm = lambda x, **kwargs: x
try:
    from tqdm import tqdm as tqdm_real
    tqdm = tqdm_real
    tqdm_available = True
except ImportError:
    pass
# ---------------------------

# --- NUEVO: Importar torch para detección de GPU ---
try:
    import torch
except ImportError:
    print("[ADVERTENCIA] PyTorch no instalado. La detección/uso de GPU no funcionará. Ejecuta: pip install torch")
    torch = None # Marcar como no disponible
# --- FIN NUEVO ---

# --- Ajuste de PYTHONPATH e Importaciones ---
try:
    project_root = Path(__file__).parent.parent.parent.resolve()
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Importar módulos del proyecto (espera versiones modificadas)
    from kelly_indexer.config import Settings
    from kelly_indexer import data_loader # Espera versión modificada
    from kelly_indexer import embeddings
    from kelly_indexer import qdrant_ops # Espera versión con except corregido
    from kelly_indexer import state_manager # Espera versión modificada
    from kelly_indexer import text_chunker
    try:
        from kelly_indexer.utils.logging_setup import setup_logging
    except ImportError:
        setup_logging = None

    from pydantic import ValidationError
    try:
        from qdrant_client.http.models import PointStruct
    except ImportError:
        PointStruct = Dict # type: ignore

except ImportError as e:
    print(f"[ERROR CRÍTICO] No se pudieron importar módulos necesarios: {e}")
    print("Verifica estructura, PYTHONPATH y dependencias ('pip install -e .').")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR CRÍTICO] Error inesperado durante la importación: {e}")
    sys.exit(1)

# --- Constantes ---
# NUEVO: Namespace UUID para generar IDs deterministas v5 para Qdrant.
# ¡Debe ser el mismo cada vez! Usa el que tenías en state_manager o genera uno nuevo.
QDRANT_POINT_NAMESPACE = uuid.UUID('f8a7c9a1-e45f-4e6d-8f3c-1b7a2b9e8d0f') # ¡CAMBIA ESTO si quieres!

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
        settings = Settings()
        # Reconfigurar logging
        if setup_logging:
             setup_logging(log_level_str=settings.log_level, log_file=settings.log_file)
             logger.info("Logging reconfigurado usando utils.logging_setup.")
        else: # Fallback manual
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
        logger.critical(f"Error CRÍTICO de validación al cargar config: {e}")
        logger.critical("Verifica .env (QDRANT_URL, EMBEDDING_MODEL_NAME, VECTOR_DIMENSION, etc).")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error CRÍTICO inesperado al cargar config: {e}")
        sys.exit(1)

    # --- Parsear Argumentos CLI ---
    parser = argparse.ArgumentParser(description="Indexa Q&A JSON a Qdrant.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=Path, default=settings.input_json_dir, help="Directorio raíz JSON Q&A.")
    parser.add_argument("--state-file", type=Path, default=settings.state_file_path, help="Archivo JSON de estado.")
    parser.add_argument("--batch-size", type=int, default=settings.qdrant_batch_size, help="Tamaño de lote Qdrant.")
    parser.add_argument("--force-reindex", action="store_true", help="Forzar reindexación total.")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin modificar Qdrant/estado.")
    args = parser.parse_args()

    # --- Validaciones Iniciales ---
    if not args.source.is_dir():
        logger.critical(f"CRÍTICO: El directorio fuente '{args.source}' no existe.")
        sys.exit(1)
    try:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.critical(f"CRÍTICO: No se pudo crear dir para state file: {e}.")
        sys.exit(1)

    # --- Detectar Dispositivo (CPU o CUDA) ---
    device = "cpu" # Default
    if torch and torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA detectado. Se usará GPU para embeddings.")
    else:
        if torch is None: logger.warning("PyTorch no instalado. Usando CPU.")
        else: logger.warning("CUDA no disponible o PyTorch no lo detecta. Usando CPU.")
    # --- Fin Detección Dispositivo ---


    logger.info("--- Iniciando Proceso de Indexación Qdrant ---")
    logger.info(f"Fuente JSON: {args.source.resolve()}")
    logger.info(f"Estado: {args.state_file.resolve()}")
    logger.info(f"Lote Qdrant: {args.batch_size}")
    logger.info(f"Modelo Embeddings: {settings.embedding_model_name} (Dim: {settings.vector_dimension}) en {device.upper()}")
    logger.info(f"Métrica: {settings.distance_metric}")
    logger.info(f"Colección: {settings.qdrant_collection_name}")
    logger.info(f"Forzar Reindex: {'Sí' if args.force_reindex else 'No'}")
    logger.info(f"Dry Run: {'Sí' if args.dry_run else 'No'}")

    # --- Inicializar Componentes ---
    try:
        logger.info("Inicializando componentes...")
        # MODIFICADO: Pasar device detectado
        embed_model = embeddings.get_embedding_model(settings.embedding_model_name, device=device)
        if embed_model is None: raise RuntimeError("No se pudo cargar modelo embeddings.")
        # Validar dimensión
        loaded_dim = embed_model.get_sentence_embedding_dimension()
        if loaded_dim != settings.vector_dimension:
             logger.critical(f"¡Discrepancia Dimensión! Modelo '{settings.embedding_model_name}' dim={loaded_dim}, config dice={settings.vector_dimension}.")
             raise ValueError("Dimensión de vector en config no coincide con modelo.")
        logger.info(f"Modelo embeddings '{settings.embedding_model_name}' (Dim: {loaded_dim}) cargado en {device.upper()}.")

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
        logger.info("Gestor de estado listo.")

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
    ids_to_delete: List[str] = [] # Contendrá faq_ids
    current_points_details_map: Dict[str, Dict] = {} # Claves son faq_id

    try:
        # 1. Cargar estado anterior
        logger.info(f"Cargando estado anterior desde {args.state_file}...")
        previous_state = state_manager.load_state(args.state_file)
        previous_indexed_points = previous_state.get('indexed_points', {}) # {faq_id -> {details}}
        logger.info(f"Encontrados {len(previous_indexed_points)} puntos (faq_id) en estado anterior.")

        # 2. Cargar Q&As actuales (Usa data_loader modificado)
        logger.info(f"Escaneando y cargando Q&As desde {args.source}...")
        all_current_qas_map = data_loader.load_all_qas_from_directory(args.source)
        total_processed_files = len(all_current_qas_map)
        logger.info(f"Lectura completada. {total_processed_files} archivos JSON procesados.")

        # 3. Calcular Diff (Usa state_manager modificado)
        logger.info("Comparando Q&As actuales con estado anterior (usando faq_id y hash de contenido)...")
        # qas_to_process contiene dicts originales + _source_file
        # ids_to_delete contiene faq_ids a borrar
        # current_points_details_map es {faq_id -> {source_file, content_hash}}
        qas_for_processing, ids_to_delete, current_points_details_map = state_manager.calculate_diff(
            all_current_qas_map, previous_indexed_points
        )
        total_qas_found = sum(len(v) for v in all_current_qas_map.values())

        # 4. Aplicar Force Reindex si es necesario (Lógica MODIFICADA para nueva estructura)
        if args.force_reindex:
            logger.warning("Forzando reindexación (--force-reindex)...")
            # Marcar todos los items actuales válidos para procesar y reconstruir detalles
            qas_for_processing = [] # Resetear
            current_points_details_map = {} # Resetear
            current_point_ids: Set[str] = set() # Resetear
            total_qas_found = 0

            for file_rel_path, qa_list in all_current_qas_map.items():
                total_qas_found += len(qa_list)
                for qa_item in qa_list:
                    point_id = qa_item.get('faq_id')
                    text_to_hash = qa_item.get('texto_para_vectorizar')
                    if not point_id or not text_to_hash or not isinstance(point_id, str) or not isinstance(text_to_hash, str):
                        continue # Saltar inválidos

                    content_hash = state_manager.generate_content_hash(text_to_hash)
                    current_point_ids.add(point_id)
                    current_points_details_map[point_id] = {"source_file": file_rel_path, "content_hash": content_hash}

                    # Enriquecer con _source_file para el bucle de procesamiento posterior
                    qa_item_copy = qa_item.copy()
                    qa_item_copy['_source_file'] = file_rel_path
                    qas_for_processing.append(qa_item_copy)

            # Marcar todos los IDs del estado *anterior* para eliminar
            ids_to_delete = list(previous_indexed_points.keys()) # Todos los faq_id anteriores

            logger.warning(f"Force-reindex: {len(qas_for_processing)} Q&As actuales marcados para upsert.")
            logger.warning(f"Force-reindex: {len(ids_to_delete)} puntos del estado anterior (faq_id) marcados para eliminación.")
            if args.dry_run:
                 logger.warning("Dry Run con Force-reindex: No se harán cambios reales.")

        else: # Si no es force-reindex, log normal del diff
            logger.info(f"Comparación completada. Total Q&As evaluados: {total_qas_found}. A procesar: {len(qas_for_processing)}. A eliminar: {len(ids_to_delete)}.")


        # 5. Preparar Lotes para Qdrant
        points_to_upsert: List[PointStruct] = []
        if qas_for_processing:
            logger.info("Preparando puntos para Qdrant (generando embeddings y chunking)...")

            # --- Usar texto_para_vectorizar ---
            texts_to_embed = [qa['texto_para_vectorizar'] for qa in qas_for_processing]
            logger.info(f"Generando {len(texts_to_embed)} embeddings en {device.upper()}...")
            item_vectors = embeddings.generate_embeddings(
                embed_model, texts_to_embed,
                batch_size=settings.qdrant_batch_size // 2 or 16,
                show_progress_bar=tqdm_available
            )
            logger.info("Embeddings generados.")

            if item_vectors is None or len(item_vectors) != len(qas_for_processing):
                logger.error("Fallo en embeddings o discrepancia de número. Abortando.")
                raise RuntimeError("Fallo crítico en embeddings.")

            # Construir PointStructs
            iterable_qas = qas_for_processing
            disable_tqdm = not tqdm_available or (logging.getLogger().getEffectiveLevel() > logging.INFO)
            if not disable_tqdm:
                iterable_qas = tqdm(qas_for_processing, desc="Construyendo puntos Qdrant", unit="q&a")

            logger.info("Construyendo puntos Qdrant (PointStructs)...")
            for i, qa_item in enumerate(iterable_qas):
                try:
                    # --- Usar faq_id y generar UUID ---
                    faq_id_str = qa_item.get('faq_id')
                    source_file_rel_path = qa_item.get('_source_file') # Recuperar de la data enriquecida

                    if not faq_id_str: continue # Saltar si falta (ya validado antes, pero por si acaso)
                    if not source_file_rel_path: source_file_rel_path = "desconocido"

                    # --- NUEVO: Generar UUID para Qdrant ID ---
                    point_uuid = str(uuid.uuid5(QDRANT_POINT_NAMESPACE, faq_id_str))
                    # --- FIN NUEVO ---

                    # Chunkear respuesta 'a'
                    answer_content = qa_item.get('a', '')
                    answer_chunks = text_chunker.chunk_text(chunker, answer_content)
                    if not answer_chunks and answer_content:
                        answer_chunks = [answer_content]
                    elif not answer_chunks and not answer_content:
                        answer_chunks = []

                    # --- Construir Payload (incluye categoria, etc.) ---
                    payload = {
                        "question": qa_item.get('q', ''),
                        "answer_chunks": answer_chunks,
                        "answer_full": qa_item.get('a', ''),
                        "product": qa_item.get('product', 'General'),
                        "categoria": qa_item.get('categoria', 'General'), # Añadido
                        "keywords": qa_item.get('keywords', []),
                        "source_doc_id": Path(source_file_rel_path).stem, # Añadido
                        "source_file_path": source_file_rel_path,
                        "original_faq_id": faq_id_str # Añadido (opcional, para referencia)
                    }

                    # --- Crear PointStruct usando UUID ---
                    vector_list = item_vectors[i].tolist()
                    if PointStruct is Dict:
                         point = {"id": point_uuid, "vector": vector_list, "payload": payload}
                    else:
                         point = PointStruct(id=point_uuid, vector=vector_list, payload=payload)

                    points_to_upsert.append(point) # type: ignore

                except Exception as e:
                    logger.error(f"Error preparando punto Qdrant (FAQ_ID: {qa_item.get('faq_id', 'N/A')}): {e}", exc_info=False)
                    error_count += 1
            logger.info(f"Preparados {len(points_to_upsert)} puntos para upsert (errores preparación: {error_count}).")


        # 6. Ejecutar Operaciones Qdrant (si no es dry run)
        if args.dry_run:
            logger.warning("--- DRY RUN ACTIVADO ---")
            upserted_count = len(points_to_upsert)
            deleted_count = len(ids_to_delete) # ids_to_delete contiene faq_ids
            logger.warning(f"Simulación: Se realizaría upsert de {upserted_count} puntos (con UUIDs).")
            logger.warning(f"Simulación: Se eliminarían {deleted_count} puntos (basado en faq_ids convertidos a UUIDs).")
            logger.warning("No se harán cambios en Qdrant ni en el archivo de estado.")
        else:
            logger.info("Ejecutando operaciones en Qdrant...")
            # Asegurar colección
            collection_ok = qdrant_ops.ensure_collection(
                client=qdrant_client, collection_name=settings.qdrant_collection_name,
                vector_size=settings.vector_dimension, distance_metric_str=settings.distance_metric
            )
            if not collection_ok: raise RuntimeError("Fallo colección Qdrant.")

            # Realizar Upsert (usa points_to_upsert con UUIDs)
            upsert_errors = 0
            if points_to_upsert:
                upserted_count, upsert_errors = qdrant_ops.batch_upsert(
                    client=qdrant_client, collection_name=settings.qdrant_collection_name,
                    points=points_to_upsert, batch_size=args.batch_size
                )
                error_count += upsert_errors
            else: logger.info("No hay puntos nuevos/modificados para upsert.")

            # Realizar Delete (Convertir faq_id a UUIDs antes)
            delete_errors = 0
            if ids_to_delete: # Esta lista contiene faq_ids
                # --- NUEVO: Convertir faq_ids a UUIDs para Qdrant ---
                uuids_to_delete = [str(uuid.uuid5(QDRANT_POINT_NAMESPACE, faq_id)) for faq_id in ids_to_delete]
                logger.info(f"Convirtiendo {len(ids_to_delete)} faq_ids a {len(uuids_to_delete)} UUIDs para eliminación.")
                # --- FIN NUEVO ---

                deleted_count, delete_errors = qdrant_ops.delete_points(
                    client=qdrant_client, collection_name=settings.qdrant_collection_name,
                    point_ids=uuids_to_delete, # Pasar lista de UUIDs
                    batch_size=args.batch_size
                )
                error_count += delete_errors
            else: logger.info("No hay puntos obsoletos para eliminar.")

            # 7. Guardar Estado Final (usa current_points_details_map con faq_id como clave)
            logger.info("Guardando estado final...")
            # La función save_state ya espera el diccionario {faq_id -> details}
            save_success = state_manager.save_state(args.state_file, current_points_details_map)
            if not save_success:
                logger.error("¡FALLO AL GUARDAR EL ARCHIVO DE ESTADO!")
                error_count += 1
            else:
                logger.info(f"Nuevo estado guardado en {args.state_file}")

    except Exception as e:
        logger.exception(f"Error CRÍTICO inesperado durante el proceso principal: {e}")
        error_count += 1
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
    logger.info(f"Q&As Marcados para Eliminación (por faq_id): {len(ids_to_delete)}")
    logger.info(f"Q&As Exitosamente Eliminados (por UUID): {deleted_count if not args.dry_run else 'N/A (Dry Run)'}")
    logger.info(f"Errores Totales Durante el Proceso: {error_count}")
    if args.dry_run: logger.warning("La ejecución fue en modo --dry-run. No se hicieron cambios reales.")
    if error_count > 0: logger.error("El proceso finalizó con errores.")
    else: logger.info("El proceso finalizó exitosamente.")


if __name__ == "__main__":
    main()