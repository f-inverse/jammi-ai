# Architecture

## Crate dependency graph

```
jammi-engine (foundation)
    │
    ├── Config, Catalog, Sources, SQL execution
    │
    ▼
jammi-ai (intelligence)
    │
    ├── Model resolution, loading, caching
    ├── InferenceExec operator
    ├── Output adapters
    ├── InferenceSession (wraps JammiSession)
    │
    ▼
jammi-server (Phase 11)     jammi-python (Phase 10)
    │                           │
    ├── REST API                ├── PyO3 bindings
    └── Arrow Flight SQL        └── Python-native API
```

`jammi-engine` has no dependency on `jammi-ai`. The intelligence layer is an optional addition — you can use `jammi-engine` standalone for SQL queries over local data.

## Key types and their responsibilities

### Engine layer (`jammi-engine`)

| Type | Responsibility |
|------|---------------|
| `JammiConfig` | TOML + env config loading with defaults |
| `Catalog` | SQLite-backed persistence for sources, models, eval runs |
| `JammiSession` | DataFusion session + source registration + SQL execution |
| `SourceCatalog` / `JammiSchemaProvider` | DataFusion catalog integration |

### AI layer (`jammi-ai`)

| Type | Responsibility |
|------|---------------|
| `InferenceSession` | Wraps `JammiSession` + `ModelCache` + observer. Entry point for inference |
| `ModelResolver` | Resolves model ID → file paths + backend. Chain: catalog → local → HF Hub |
| `ModelCache` | LRU cache with single-flight loading, ref-counted guards, GPU eviction |
| `CandleBackend` | Loads safetensors into candle's BertModel |
| `InferenceExec` | DataFusion `ExecutionPlan` operator for inference with backpressure |
| `InferenceRunner` | Batch processing loop with error recovery and dynamic sizing |
| `OutputAdapter` | Trait that converts raw model output to Arrow arrays per task |
| `GpuScheduler` | GPU memory permit system (unlimited in v1, budgeted in Phase 09) |

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

### Inference path

```
InferenceSession::infer(source, model, task, columns, key)
    → Validates columns
    → Builds SQL scan plan via DataFusion
    → Pre-loads model (ModelCache::get_or_load)
    → Creates InferenceExec wrapping the scan plan
    → InferenceExec::execute() spawns InferenceRunner
    │   └── Runner reads input stream
    │   └── Extracts text, tokenizes, runs forward pass
    │   └── Builds prefix + task columns
    │   └── Sends through bounded channel (backpressure)
    → Collects output batches
    → Returns Vec<RecordBatch>
```

## Module layout

```
crates/jammi-engine/src/
├── config.rs           # Configuration loading
├── error.rs            # Unified error type
├── session.rs          # JammiSession (DataFusion wrapper)
├── catalog/
│   ├── mod.rs          # Catalog (SQLite connection pool)
│   ├── schema.rs       # Migration SQL
│   ├── migrations.rs   # Migration runner
│   ├── model_repo.rs   # Model CRUD
│   └── source_repo.rs  # Source CRUD
└── source/
    ├── mod.rs           # SourceType, FileFormat, SourceConnection
    ├── registry.rs      # DataFusion CatalogProvider
    ├── schema_provider.rs # DataFusion SchemaProvider
    └── local.rs         # ListingTable creation for local files

crates/jammi-ai/src/
├── session.rs          # InferenceSession
├── model/
│   ├── mod.rs          # ModelId, BackendType, ModelTask, LoadedModel, ModelGuard
│   ├── cache.rs        # ModelCache (LRU, single-flight, ref-counted)
│   ├── resolver.rs     # ModelResolver (catalog → local → HF Hub)
│   ├── tokenizer.rs    # TokenizerWrapper (batched encoding)
│   └── backend/
│       ├── mod.rs      # ModelBackend trait, DeviceConfig
│       ├── candle.rs   # CandleBackend, CandleModel, forward pass
│       └── ort.rs      # OrtBackend (Phase 12a)
├── operator/
│   └── inference_exec.rs  # InferenceExec (DataFusion ExecutionPlan)
├── inference/
│   ├── mod.rs          # Arrow helpers (arrow_to_texts, extract_columns)
│   ├── runner.rs       # InferenceRunner (batch loop, OOM recovery)
│   ├── schema.rs       # Output schema construction, prefix columns
│   ├── observer.rs     # InferenceObserver trait
│   └── adapter/
│       ├── mod.rs      # OutputAdapter trait, BackendOutput, factories
│       ├── embedding.rs       # FixedSizeList(Float32, N)
│       ├── classification.rs  # label, confidence, all_scores_json
│       ├── summarization.rs   # summary (LargeUtf8)
│       ├── text_generation.rs # generated_text, finish_reason
│       ├── ner.rs             # entities (JSON)
│       └── object_detection.rs # detections (JSON)
└── concurrency/
    └── gpu_scheduler.rs  # GpuScheduler, GpuPermit (RAII)
```
