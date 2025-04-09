# 🔎 Kelly Indexer — Indexador Inteligente de Q&A para Qdrant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Kelly Indexer** es una herramienta robusta y autónoma para convertir archivos `.json` con pares Pregunta/Respuesta (Q&A) —generados por [`kelly_soap`](https://github.com/Evertdan/kelly_soap)— en vectores que se indexan eficientemente en una base de datos vectorial **Qdrant Cloud**. Está optimizada para sistemas **RAG (Retrieval-Augmented Generation)** y asistentes inteligentes como **KellyBot**.

---

## 🧠 1. Introducción

### 🎯 El Desafío

Los sistemas RAG modernos requieren bases de datos vectoriales con información estructurada y actualizada. Aunque el formato Q&A es ideal, necesita ser vectorizado, enriquecido con metadatos y correctamente indexado.

### 💡 La Solución: Kelly Indexer

Automatiza el pipeline completo para convertir Q&As en vectores listos para Qdrant:

1. 📂 **Carga y descubrimiento de archivos** `.json` (incluyendo subdirectorios).
2. 🧠 **Generación de embeddings** para preguntas utilizando `sentence-transformers`.
3. ✂️ **Fragmentación de respuestas** largas en chunks configurables.
4. 📦 **Construcción de payloads ricos**: incluye respuestas, producto, keywords, archivo fuente.
5. 🚀 **Indexación eficiente en Qdrant**, usando `upsert` y `delete` en lotes.
6. 🔁 **Idempotencia segura** mediante un archivo de estado inteligente.
7. 📊 **Reporte final de cambios** (agregados, modificados, eliminados).

---

## ⚙️ 2. Características Principales

- 🗂️ **Entrada recursiva** desde un directorio configurable (`.json` con listas de Q&A).
- 🧬 **Vectorización de preguntas** con `all-MiniLM-L6-v2` por defecto.
- 🧾 **Metadatos completos**: respuesta, producto, keywords, chunks, fuente.
- 🧠 **UUIDs deterministas + SHA256** para detectar cambios reales en Q&As.
- ✂️ **Chunking automático** de respuestas largas (tamaño y solapamiento configurables).
- ☁️ **Integración con Qdrant Cloud** o local (métrica por defecto: `Cosine`).
- 🧾 **Archivo de estado** para evitar duplicados o reprocesamientos innecesarios.
- 🧪 **CLI potente y flexible** con soporte para dry-run, forzado, control de batch, etc.
- 📈 **Progreso visual** con `tqdm` y logging configurable (nivel y archivo).
- 💥 **Manejo de errores inteligente**: no se detiene por archivos defectuosos.
- 🧪 **Entorno reproducible** con Conda y pruebas unitarias (`pytest`).

---

## 🔁 3. Flujo de Trabajo

```mermaid
%%{init: {'theme': 'neutral'}}%%
graph TD
    A[Inicio: Ejecutar index_qdrant.py] --> B{Cargar Config (.env)};
    B --> C{Cargar Estado Anterior (.json)};
    C --> D{Escanear y Cargar Q&As actuales};
    D --> E{Detectar Cambios<br>(nuevos, modificados, eliminados)};
    E -- Nuevos o modificados --> F[Generar Embeddings (Q)];
    F --> G[Chunkear Respuestas (A)];
    G --> H[Preparar lote Upsert con Vector + Payload];
    E -- Eliminados --> I[Preparar lote Delete];
    H --> J[Ejecutar Upsert en Qdrant];
    I --> K[Ejecutar Delete en Qdrant];
    J & K --> L[Actualizar y Guardar Estado];
    L --> M[Mostrar resumen final];
    E -- Sin cambios --> M;
```

---

## 🛠 4. Instalación y Configuración (Ubuntu 24 + Conda)

### 🐍 Paso 1: Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO_KELLY_INDEXER>
cd kelly_indexer
```

### 🧪 Paso 2: Crear entorno Conda

```bash
conda create --name kelly_indexer_env python=3.10 -y
conda activate kelly_indexer_env
```

### 📦 Paso 3: Instalar dependencias

```bash
# Para desarrollo y pruebas
pip install -e .[dev,test]

# Solo para ejecución
# pip install -e .
```

### ⚙️ Paso 4: Configurar `.env`

```bash
cp .env.sample .env
nano .env
```

Edita los siguientes valores mínimos:

- `QDRANT_URL`  
- `QDRANT_API_KEY`

Y ajusta los demás valores según necesidad (`CHUNK_SIZE`, `LOG_LEVEL`, etc.).

---

## ⚙️ 5. Variables Importantes del `.env`

| Variable | Descripción |
|----------|-------------|
| `QDRANT_URL` | 🌐 URL de tu instancia Qdrant Cloud |
| `QDRANT_API_KEY` | 🔑 Token de autenticación |
| `QDRANT_COLLECTION_NAME` | 🗃 Nombre de colección en Qdrant (default: `kellybot-docs-v1`) |
| `EMBEDDING_MODEL_NAME` | 🧬 Modelo de embeddings (default: `all-MiniLM-L6-v2`) |
| `VECTOR_DIMENSION` | 🔢 Dimensión del vector (ej: 384) |
| `INPUT_JSON_DIR` | 📂 Ruta a los `.json` generados por `kelly_soap` |
| `STATE_FILE_PATH` | 📄 Archivo de estado para evitar duplicados |
| `CHUNK_SIZE` | 📏 Longitud máxima de fragmento de respuesta |
| `CHUNK_OVERLAP` | 🔄 Solapamiento entre fragmentos |
| `QDRANT_BATCH_SIZE` | 📦 Tamaño de lote para upserts |
| `LOG_LEVEL` | 📣 Nivel de log (`INFO`, `DEBUG`, etc.) |
| `LOG_FILE` | 📝 Archivo para logs persistentes |

---

## 🚀 6. Ejecutar el Indexador

```bash
# Asegúrate de tener el entorno activo:
conda activate kelly_indexer_env

# Ejecución estándar:
python scripts/indexer/index_qdrant.py
```

### 🧾 Opciones útiles:

| Flag | Descripción |
|------|-------------|
| `--source` | Ruta de entrada personalizada |
| `--state-file` | Ruta del archivo de estado |
| `--batch-size` | Número de puntos por lote |
| `--force-reindex` | Ignora el estado previo y reindexa todo |
| `--dry-run` | Simula todo sin escribir ni eliminar nada |

---

## 📤 7. Resultado Esperado

- En **Qdrant** se almacenan vectores de preguntas con este payload:

```json
{
  "question": "¿Qué es MiAdminXML?",
  "answer": ["Respuesta fragmentada...", "Segundo chunk..."],
  "product": "MiAdminXML",
  "keywords": ["xml", "sat", "estado"],
  "source": "SOAP_TXT/archivo.json"
}
```

- En `scripts/indexer/index_state_qdrant.json` queda guardado el estado actual:

```json
{
  "version": "1.0",
  "last_run_utc": "2025-04-08T21:45:00Z",
  "indexed_points": {
    "uuid-v5": {
      "source_file": "subfolder/file1.json",
      "question_hash": "9c1e..."
    }
  }
}
```

---

## 🧯 8. Solución de Problemas

| Problema | Solución |
|---------|----------|
| ❌ Falla `.env` | Asegúrate de copiar y configurar correctamente `.env` |
| 🔐 Qdrant 401 | Verifica `QDRANT_API_KEY` |
| 📡 Timeout / Conexión | Verifica tu red o URL de Qdrant |
| 🧠 Modelo no encontrado | Revisa `EMBEDDING_MODEL_NAME`, requiere conexión para descarga inicial |
| 🧮 Errores de memoria | Reduce `QDRANT_BATCH_SIZE` si hay muchos Q&A |
| 📁 Permisos | Asegura permisos de lectura/escritura sobre input y output |

---

## 🧪 9. Desarrollo

```bash
conda activate kelly_indexer_env
pip install -e .[dev,test]
pytest tests/
ruff format src tests scripts
mypy src
```

---

## 📄 10. Licencia

Licencia [MIT](./LICENSE).

