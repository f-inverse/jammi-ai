# Embedding Inference

Jammi generates embeddings by running BERT-family models over text columns from registered sources. The pipeline handles tokenization, batching, error recovery, and output construction automatically.

## Basic usage

```rust
use jammi_ai::model::{ModelSource, ModelTask};

let model = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
let results = session.infer(
    "patents",                     // source id
    &model,                        // ModelSource (HF Hub or local path)
    ModelTask::Embedding,          // task type
    &["abstract".to_string()],    // text column(s) to embed
    "id",                          // key column for _row_id
).await?;
```

> **Note:** For embedding generation with persistence (Parquet + ANN index), use `generate_embeddings()` instead — see [Embedding Generation](./embedding-generation.md).

## Pipeline architecture

```
Source (Parquet/CSV)
    │
    ▼  DataFusion scan
    │
InferenceExec operator
    ├── Loads model (or cache hit)
    ├── Bounded channel (capacity=2, backpressure)
    ├── InferenceRunner (async task)
    │     ├── Reads input batches
    │     ├── Extracts text from content columns
    │     ├── Tokenizes with model's tokenizer
    │     ├── BERT forward pass
    │     ├── Mean pooling + L2 normalization
    │     ├── Constructs prefix + vector columns
    │     └── Sends to output channel
    │
    ▼  RecordBatch stream
    │
Results (Vec<RecordBatch>)
```

## Error handling

Inference never panics on bad input. Errors are tracked per-row:

| Condition | `_status` | `_error` | `vector` |
|-----------|-----------|----------|----------|
| Valid text | `"ok"` | null | 384-dim float vector |
| Null text | `"error"` | `"Empty or null text input"` | null |
| Empty text | `"error"` | `"Empty or null text input"` | null |

The batch continues processing even when individual rows fail. Every row in the output has a valid `_status` — no rows are dropped.

## Multiple text columns

Pass multiple column names to concatenate them (space-separated) before embedding:

```rust
let model = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
let results = session.infer(
    "papers",
    &model,
    ModelTask::Embedding,
    &["title".to_string(), "abstract".to_string()],  // concatenated
    "doi",
).await?;
```

## Model caching

Models are loaded once and cached with LRU eviction:

- **First load**: downloads from HF Hub (or reads from local path), loads weights into memory
- **Subsequent calls**: cache hit, returns immediately
- **Ref counting**: model stays in memory while any inference is running
- **Eviction**: when GPU memory pressure requires it, the least-recently-used model with no active references is evicted

## Dynamic batch sizing

The runner starts with the configured `inference.batch_size` (default: 32). If an out-of-memory error occurs:

1. Halve the batch size
2. Retry (up to 3 times)
3. If OOM persists at batch size 1, mark the row as error and continue

The reduced batch size is sticky for the remainder of the stream.

## Observing batches

Attach an `InferenceObserver` to inspect every output batch:

```rust
use jammi_ai::inference::observer::InferenceObserver;

struct MetricsCollector;
impl InferenceObserver for MetricsCollector {
    fn on_batch(&self, batch: &RecordBatch, model_id: &str, latency: Duration) {
        // Record metrics, check quality, log progress
    }
}
```

The observer is called once per output batch with near-zero overhead when not attached (single `Option` branch).
