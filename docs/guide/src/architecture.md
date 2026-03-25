# Architecture

## Crate dependency graph

```
jammi-engine (foundation)
    |
    |-- Config, Catalog, Sources, SQL execution
    |-- Parquet storage, ANN indexes, crash recovery
    |
    v
jammi-ai (intelligence)
    |
    |-- Model resolution, loading, caching
    |-- InferenceExec, AnnSearchExec operators
    |-- Embedding pipeline, result persistence
    |-- SearchBuilder, evidence provenance
    |-- Fine-tuning, evaluation
    |-- GPU scheduling
    |-- InferenceSession (wraps JammiSession)
    |
    +-------+-------+
    |               |
    v               v
jammi-server    jammi-python
    |               |
    |-- Flight SQL  |-- PyO3 bindings
    '-- Health API   '-- pyarrow interop

jammi-cli
    |
    '-- Clap CLI wrapping InferenceSession
```

`jammi-engine` has no dependency on `jammi-ai`. The intelligence layer is an optional addition — you can use `jammi-engine` standalone for SQL queries over local data.

## Key types and their responsibilities

### Engine layer (`jammi-engine`)

| Type | Responsibility |
|------|---------------|
| `JammiConfig` | TOML + env config loading with defaults |
| `Catalog` | SQLite-backed persistence for sources, models, result tables, eval runs, evidence channels |
| `JammiSession` | DataFusion session + source registration + SQL execution |
| `SourceCatalog` / `JammiSchemaProvider` | DataFusion catalog integration |
| `ResultStore` | Parquet storage coordinator: create, finalize, recover, register |
| `ParquetResultWriter` | ZSTD-compressed Parquet file writer (64K row groups) |
| `VectorIndex` / `SidecarIndex` | ANN index trait + USearch implementation with row_id mapping |

### AI layer (`jammi-ai`)

| Type | Responsibility |
|------|---------------|
| `InferenceSession` | Wraps `JammiSession` + `ModelCache` + `ResultStore`. Entry point for all operations |
| `ModelResolver` | Resolves model ID to file paths + backend. Chain: catalog -> local -> HF Hub |
| `ModelCache` | LRU cache with single-flight loading, ref-counted guards |
| `CandleBackend` / `OrtBackend` | Model backends: Candle (safetensors), ONNX Runtime |
| `VllmBackend` / `HttpBackend` | Remote backends: vLLM server, generic HTTP |
| `InferenceExec` | DataFusion `ExecutionPlan` operator for inference with backpressure |
| `AnnSearchExec` | DataFusion `ExecutionPlan` leaf node for ANN vector search |
| `EmbeddingPipeline` | Orchestrates generate_embeddings: model -> InferenceExec -> ResultSink -> index |
| `ResultSink` | Streams inference output to Parquet + sidecar index, filters failed rows |
| `SearchBuilder` | Fluent API: join, annotate, filter, sort, limit, select, run |
| `EvidenceRow` / `RowProvenance` | Evidence model types for provenance tracking |
| `OutputAdapter` | Trait that converts raw model output to Arrow arrays per task |
| `GpuScheduler` | GPU memory permit system with budget-based admission control |
| `FineTuneJob` | LoRA fine-tuning with contrastive loss, checkpointing, early stopping |
| `EvalRunner` | Retrieval, classification, and summarization evaluation |

### Server layer (`jammi-server`)

| Type | Responsibility |
|------|---------------|
| `AppState` | Shared state: `Arc<InferenceSession>` + ANN cache |
| `FlightSqlService` | Arrow Flight SQL server backed by DataFusion |
| Health endpoint | HTTP `/health` for container liveness probes |

### Python layer (`jammi-python`)

| Type | Responsibility |
|------|---------------|
| `Database` | PyO3 class wrapping `Arc<InferenceSession>` with shared tokio runtime |
| `SearchBuilder` | PyO3 class with imperative-style search composition |
| `FineTuneJob` | PyO3 class for monitoring fine-tuning jobs |
| `connect()` | Module-level function to create a `Database` |

## Data flow

### SQL query path

```
JammiSession::sql("SELECT ...")
    -> DataFusion parses SQL
    -> Resolves table from SourceCatalog/JammiSchemaProvider
    -> Creates ListingTable scan from Parquet/CSV/JSON or federated source
    -> Executes plan
    -> Returns Vec<RecordBatch>
```

### Embedding generation path

```
InferenceSession::generate_embeddings(source, model, columns, key)
    -> EmbeddingPipeline::run()
    -> Register result_table (status = "building")
    -> Build plan: SourceScan -> InferenceExec(embedding)
    -> Execute, stream batches through ResultSink
    |   |-- Filter _status = "ok"
    |   |-- Transform to embedding schema
    |   |-- Write to Parquet via ParquetResultWriter
    |   '-- Feed vectors to SidecarIndex::add()
    -> Close writer, build ANN index, save sidecar bundle
    -> Register as DataFusion table, update catalog to "ready"
    -> Return ResultTableRecord
```

### Vector search path

```
InferenceSession::search(source, query_vec, k)
    -> Resolve embedding table from catalog
    -> AnnSearchExec: SidecarIndex (ANN) or exact_vector_search (fallback)
    -> Hydration: join ANN results back to source table
    -> SearchBuilder: .join() .annotate() .filter() .sort() .limit() .select()
    -> .run(): execute DataFusion plan, add provenance columns
    -> Returns Vec<RecordBatch> with similarity + original columns + evidence
```

## Module layout

```
crates/jammi-engine/src/
|-- config.rs           # Configuration loading
|-- error.rs            # Unified error type
|-- session.rs          # JammiSession (DataFusion wrapper)
|-- catalog/            # SQLite-backed catalog
|-- source/             # Source types, registry, schema provider
|-- store/              # ResultStore, Parquet writer/reader
'-- index/              # VectorIndex trait, sidecar, exact search

crates/jammi-ai/src/
|-- session.rs          # InferenceSession
|-- model/              # ModelResolver, ModelCache, backends
|-- operator/           # InferenceExec, AnnSearchExec
|-- inference/          # Runner, observer, output adapters
|-- pipeline/           # EmbeddingPipeline, ResultSink
|-- evidence/           # Provenance types and columns
|-- search/             # SearchBuilder
|-- fine_tune/          # LoRA training, config, jobs
|-- eval/               # Retrieval, classification, summarization eval
'-- concurrency/        # GpuScheduler, permits

crates/jammi-server/src/
|-- lib.rs              # Health server startup, signal handling
|-- routes/health.rs    # GET /health liveness probe
|-- error.rs            # 404 fallback
'-- flight.rs           # Arrow Flight SQL service

crates/jammi-cli/src/
|-- main.rs             # Clap CLI entry point
'-- commands/           # serve, query, sources, models, explain

crates/jammi-python/src/
|-- lib.rs              # PyO3 module, connect()
|-- database.rs         # Database class
|-- search.rs           # SearchBuilder class
|-- job.rs              # FineTuneJob class
|-- convert.rs          # Arrow <-> PyArrow conversion
'-- error.rs            # Error conversion
```
