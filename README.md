# 🔎 Kelly Indexer — Indexador Inteligente de Q&A para Qdrant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Kelly Indexer** es una herramienta robusta y autónoma diseñada para transformar archivos JSON con pares Pregunta/Respuesta (Q&A) —generados por [`kelly_soap`](https://github.com/Evertdan/kelly_soap)— en vectores listos para ser indexados en **Qdrant Cloud**. Genera embeddings de preguntas usando `sentence-transformers`, fragmenta respuestas largas con `langchain-text-splitters`, mantiene un estado de idempotencia y facilita integraciones RAG con sistemas como KellyBot.

---

## 1. 🧠 Introducción

### 🧩 El desafío

Los sistemas RAG necesitan información vectorizada, actualizada y sin duplicados. Los datos en formato Q&A son ideales, pero deben convertirse en vectores con metadatos enriquecidos, controlando cambios y eliminaciones.

### 🛠 La solución: Kelly Indexer

Automatiza todo el pipeline:

1. Lee archivos `.json` de entrada.
2. Compara con el estado previo.
3. Genera embeddings (preguntas).
4. Fragmenta respuestas.
5. Construye vectores con payloads.
6. Indexa en Qdrant (`upsert`, `delete`).
7. Actualiza el estado.
8. Reporta cambios en consola.

---

## 2. ⚙️ Características Principales

- 🧪 Entrada desde carpetas recursivas (`data/input/json/SOAP_TXT/` por defecto).
- 🔁 Idempotencia con UUIDv5 y hashes SHA256.
- 🧬 Embeddings vía `sentence-transformers` (modelo local, sin GPU).
- ✂️ Fragmentación de respuestas largas (`langchain-text-splitters`).
- ☁️ Compatible con Qdrant Cloud o local.
- 🧾 Payload completo: pregunta, respuesta, producto, keywords, origen.
- 🛡️ Gestión robusta del estado (`index_state_qdrant.json`).
- 🔧 Configuración por `.env` y validación con `pydantic-settings`.
- 📊 Barra de progreso (`tqdm`) + logging centralizado.
- 🧪 CLI con soporte para dry-run y reindexado forzado.
- 🧩 Estructura modular con pruebas automatizadas (`pytest`).
- 📜 Licencia MIT.

---

## 3. 🔁 Flujo de Trabajo

```mermaid
graph TD
    A[Inicio: Ejecutar index_qdrant.py] --> B[Cargar configuración (.env)]
    B --> C[Cargar estado anterior (JSON)]
    C --> D[Leer archivos JSON con Q&As]
    D --> E[Comparar con estado (identificar cambios)]
    E --> F[Embeddings para preguntas nuevas o modificadas]
    F --> G[Dividir respuestas largas (chunking)]
    G --> H[Preparar lote para Upsert]
    E --> I[Preparar lote para Delete]
    H --> J[Enviar Upsert a Qdrant]
    I --> K[Enviar Delete a Qdrant]
    J --> L[Actualizar archivo de estado]
    K --> L
    L --> M[Mostrar resumen final]
```

---

## 4. 🧰 Instalación (Ubuntu 24 + Conda)

```bash
# Clona el repositorio
git clone git@github.com:Evertdan/kelly_indexer.git
cd kelly_indexer

# Crea entorno
conda create -n kelly_indexer_env python=3.10 -y
conda activate kelly_indexer_env

# Instala dependencias
pip install -e .[dev,test]

# Copia y edita configuración
cp .env.sample .env
nano .env  # Ajusta claves de Qdrant, rutas, modelo, etc.
```

---

## 5. ⚙️ Variables `.env` importantes

| Variable | Descripción |
|----------|-------------|
| `QDRANT_URL` | URL de tu instancia de Qdrant |
| `QDRANT_API_KEY` | Clave API (si usas Qdrant Cloud) |
| `INPUT_JSON_DIR` | Carpeta donde están los `.json` |
| `STATE_FILE_PATH` | Ruta al archivo de estado |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Parámetros de fragmentación |
| `QDRANT_COLLECTION_NAME` | Nombre de colección a usar |
| `EMBEDDING_MODEL_NAME` | Modelo local a usar |
| `LOG_LEVEL` / `LOG_FILE` | Logging personalizado |

---

## 6. 🚀 Ejecutar el Indexador

```bash
# Simula (no escribe en Qdrant ni guarda estado)
python scripts/indexer/index_qdrant.py --dry-run

# Ejecuta en modo real
python scripts/indexer/index_qdrant.py
```

### 🧾 Argumentos útiles

- `--source`: ruta personalizada de entrada
- `--state-file`: ruta personalizada para estado
- `--force-reindex`: reindexa todo ignorando estado
- `--batch-size`: cambia lote para Qdrant
- `--dry-run`: simula sin efectos colaterales

---

## 7. 📦 Estado y salida esperada

- Qdrant tendrá vectores con payload completo (`question`, `answer`, `product`, `keywords`, `source`).
- Archivo `index_state_qdrant.json` se actualizará.
- Consola mostrará resumen del proceso y errores (si los hubo).

---

## 8. 🧯 Troubleshooting

- ❌ **Error de conexión Qdrant**: revisa `QDRANT_URL` y API Key.
- ❌ **Problemas con modelo**: verifica `EMBEDDING_MODEL_NAME`, red e instalación.
- ❌ **Permisos de archivos**: asegúrate de tener lectura/escritura en los directorios involucrados.
- 🐌 **Proceso lento**: embeddings en CPU + latencia red. Ajusta `QDRANT_BATCH_SIZE`.

---

## 9. 🧩 Estructura del Código

```text
src/kelly_indexer/
├── config.py           # Configuración global
├── data_loader.py      # Lectura de archivos
├── embeddings.py       # Generación de vectores
├── qdrant_ops.py       # Conexión con Qdrant
├── state_manager.py    # Cálculo de diferencias y control de estado
├── text_chunker.py     # Fragmentación de respuestas
└── utils/
    └── logging_setup.py
```

---

## 10. 🧪 Desarrollo

```bash
conda activate kelly_indexer_env
pip install -e .[dev,test]

# Ejecuta pruebas
pytest tests/

# Verifica calidad de código
ruff check src tests scripts
ruff format src tests scripts
mypy src
```

---

## 11. 📄 Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

