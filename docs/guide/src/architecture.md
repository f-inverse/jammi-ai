# Architecture

## Crate dependency graph

```
jammi-engine (foundation)
    │
    ├── Config, Catalog, Sources, SQL execution
    ├── Parquet storage, ANN indexes, crash recovery
    │
    ▼
jammi-ai (intelligence)
    │
    ├── Model resolution, loading, caching
    ├── InferenceExec, AnnSearchExec operators
    ├── Embedding pipeline, result persistence
    ├── SearchBuilder, evidence provenance
    ├── InferenceSession (wraps JammiSession)
    │
    ▼
jammi-server (future)          jammi-python (future)
    │                              │
    ├── REST API                   ├── PyO3 bindings
    └── Arrow Flight SQL           └── Python-native API
```

`jammi-engine` has no dependency on `jammi-ai`. The intelligence layer is an optional addition — you can use `jammi-engine` standalone for SQL queries over local data.

## Key types and their responsibilities

### Engine layer (`jammi-engine`)

| Type | Responsibility |
|------|---------------|
| `JammiConfig` | TOML + env config loading with defaults |
| `Catalog` | SQLite-backed persistence for sources, models, result tables, evidence channels |
| `JammiSession` | DataFusion session + source registration + SQL execution |
| `SourceCatalog` / `JammiSchemaProvider` | DataFusion catalog integration |
| `ResultStore` | Parquet storage coordinator: create, finalize, recover, register |
| `ParquetResultWriter` | ZSTD-compressed Parquet file writer (64K row groups) |
| `VectorIndex` / `SidecarIndex` | ANN index trait + USearch implementation with row_id mapping |

### AI layer (`jammi-ai`)

| Type | Responsibility |
|------|---------------|
| `InferenceSession` | Wraps `JammiSession` + `ModelCache` + `ResultStore`. Entry point for all operations |
| `ModelResolver` | Resolves model ID → file paths + backend. Chain: catalog → local → HF Hub |
| `ModelCache` | LRU cache with single-flight loading, ref-counted guards |
| `CandleBackend` | Loads safetensors into candle's BertModel |
| `InferenceExec` | DataFusion `ExecutionPlan` operator for inference with backpressure |
| `AnnSearchExec` | DataFusion `ExecutionPlan` leaf node for ANN vector search |
| `EmbeddingPipeline` | Orchestrates generate_embeddings: model → InferenceExec → ResultSink → index |
| `ResultSink` | Streams inference output to Parquet + sidecar index, filters failed rows |
| `SearchBuilder` | Fluent API: join, annotate, filter, sort, limit, select, run |
| `EvidenceRow` / `RowProvenance` | Evidence model types for provenance tracking |
| `OutputAdapter` | Trait that converts raw model output to Arrow arrays per task |
| `GpuScheduler` | GPU memory permit system (unlimited in v1, budgeted in future) |

## Data flow

### SQL query path

```
JammiSession::sql("SELECT ...")
    → DataFusion parses SQL
    → Resolves table from SourceCatalog/JammiSchemaProvider
    → Creates ListingTable scan from Parquet/CSV
    → Executes plan
    → Returns Vec<RecordBatch>
```

### Embedding generation path

```
InferenceSession::generate_embeddings(source, model, columns, key)
    → EmbeddingPipeline::run()
    → Register result_table (status = "building")
    → Build plan: SourceScan → InferenceExec(embedding)
    → Execute, stream batches through ResultSink
    │   ├── Filter _status = "ok"
    │   ├── Transform to embedding schema (_source_id, _model_id)
    │   ├── Write to Parquet via ParquetResultWriter
    │   └── Feed vectors to SidecarIndex::add()
    → Close writer, build ANN index, save sidecar bundle
    → Register as DataFusion table, update catalog to "ready"
    → Return ResultTableRecord
```

### Vector search path

```
InferenceSession::search(source, query_vec, k)
    → Resolve embedding table from catalog
    → AnnSearchExec: SidecarIndex (ANN) or exact_vector_search (fallback)
    → Hydration: join ANN results back to source table
    → SearchBuilder: .join() .annotate() .filter() .sort() .limit() .select()
    → .run(): execute DataFusion plan, add provenance columns
    → Returns Vec<RecordBatch> with similarity + original columns + evidence
```

## Module layout

```
crates/jammi-engine/src/
├── config.rs           # Configuration loading
├── error.rs            # Unified error type
├── session.rs          # JammiSession (DataFusion wrapper)
├── catalog/
│   ├── mod.rs          # Catalog (SQLite pool), EvidenceChannelRecord
│   ├── schema.rs       # Migration SQL (MIGRATION_001, MIGRATION_002)
│   ├── migrations.rs   # Migration runner
│   ├── model_repo.rs   # Model CRUD
│   ├── source_repo.rs  # Source CRUD
│   └── result_repo.rs  # Result table CRUD, resolve_embedding_table
├── source/
│   ├── mod.rs           # SourceType, FileFormat, SourceConnection
│   ├── registry.rs      # DataFusion CatalogProvider
│   ├── schema_provider.rs # DataFusion SchemaProvider
│   └── local.rs         # ListingTable creation for local files
├── store/
│   ├── mod.rs           # ResultStore (coordinator), ResultTableInfo
│   ├── writer.rs        # ParquetResultWriter (ZSTD, 64K row groups)
│   ├── reader.rs        # Parquet validation, row counting, DataFusion registration
│   └── schema.rs        # Embedding table Arrow schema
└── index/
    ├── mod.rs           # VectorIndex trait, cosine_distance
    ├── sidecar.rs       # SidecarIndex (USearch + rowmap + manifest)
    └── exact.rs         # Brute-force vector search via DataFusion

crates/jammi-ai/src/
├── session.rs          # InferenceSession (search, generate_embeddings, encode_query, infer)
├── model/
│   ├── mod.rs          # ModelId, BackendType, ModelTask, LoadedModel, ModelGuard
│   ├── cache.rs        # ModelCache (LRU, single-flight, ref-counted)
│   ├── resolver.rs     # ModelResolver (catalog → local → HF Hub)
│   ├── tokenizer.rs    # TokenizerWrapper (batched encoding)
│   └── backend/
│       ├── mod.rs      # ModelBackend trait, DeviceConfig
│       ├── candle.rs   # CandleBackend, CandleModel, forward pass
│       └── ort.rs      # OrtBackend (future)
├── operator/
│   ├── inference_exec.rs  # InferenceExec + InferenceExecBuilder
│   └── ann_search_exec.rs # AnnSearchExec (DataFusion ExecutionPlan)
├── inference/
│   ├── mod.rs          # Arrow helpers
│   ├── runner.rs       # InferenceRunner (batch loop, OOM recovery)
│   ├── schema.rs       # Output schema construction, prefix columns
│   ├── observer.rs     # InferenceObserver trait
│   └── adapter/        # OutputAdapter implementations per task
├── pipeline/
│   ├── embedding.rs    # EmbeddingPipeline (orchestrator)
│   └── result_sink.rs  # ResultSink (Parquet + index feed, null filtering)
├── evidence/
│   ├── mod.rs          # EvidenceRow, EvidenceRowId, RowProvenance types
│   ├── provenance.rs   # add_provenance() — retrieved_by, annotated_by columns
│   └── schema.rs       # evidence_schema() — dynamic schema from catalog channels
├── search/
│   ├── mod.rs          # Re-exports
│   └── builder.rs      # SearchBuilder (join, annotate, filter, sort, limit, select, run)
└── concurrency/
    └── gpu_scheduler.rs  # GpuScheduler, GpuPermit (RAII)
```
