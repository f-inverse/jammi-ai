# Monitor Inference

Attach an observer to inspect every output batch during inference. Use this for logging, metrics collection, quality checks, or progress tracking.

## Attach an observer

### Rust

```rust
use jammi_ai::inference::observer::InferenceObserver;
use std::sync::Arc;

struct MetricsCollector;

impl InferenceObserver for MetricsCollector {
    fn on_batch(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        model_id: &str,
        latency: std::time::Duration,
    ) {
        println!(
            "Batch: {} rows from {model_id} in {latency:?}",
            batch.num_rows()
        );
    }
}

let session = InferenceSession::with_observer(
    config,
    Some(Arc::new(MetricsCollector) as Arc<dyn InferenceObserver>),
).await?;
```

The observer is called once per output batch. When no observer is attached, the overhead is a single `Option` branch — effectively zero.

## Use cases

### Progress logging

```rust
struct ProgressLogger { total: AtomicUsize }

impl InferenceObserver for ProgressLogger {
    fn on_batch(&self, batch: &RecordBatch, _model_id: &str, _latency: Duration) {
        let count = self.total.fetch_add(batch.num_rows(), Ordering::Relaxed) + batch.num_rows();
        eprintln!("Processed {count} rows...");
    }
}
```

### Quality checks

```rust
struct QualityChecker;

impl InferenceObserver for QualityChecker {
    fn on_batch(&self, batch: &RecordBatch, model_id: &str, latency: Duration) {
        // Check for high error rates
        let status = batch.column_by_name("_status").unwrap();
        let errors = status.as_any().downcast_ref::<StringArray>().unwrap()
            .iter().filter(|s| s == &Some("error")).count();

        if errors > batch.num_rows() / 2 {
            eprintln!("WARNING: {model_id} batch has {errors}/{} errors", batch.num_rows());
        }
    }
}
```

### Latency tracking

```rust
struct LatencyTracker { slow_threshold: Duration }

impl InferenceObserver for LatencyTracker {
    fn on_batch(&self, batch: &RecordBatch, model_id: &str, latency: Duration) {
        if latency > self.slow_threshold {
            eprintln!(
                "SLOW: {model_id} took {latency:?} for {} rows ({:?}/row)",
                batch.num_rows(),
                latency / batch.num_rows() as u32,
            );
        }
    }
}
```

## Pipeline architecture

```
Source (Parquet/CSV/DB)
    |
    v  DataFusion scan
    |
InferenceExec operator
    |-- Loads model (or cache hit)
    |-- Bounded channel (capacity=2, backpressure)
    |-- InferenceRunner (async task)
    |     |-- Reads input batches
    |     |-- Extracts text from content columns
    |     |-- Tokenizes with model's tokenizer
    |     |-- BERT forward pass
    |     |-- Mean pooling + L2 normalization
    |     |-- Constructs prefix + vector columns
    |     |-- ** Observer called here **
    |     '-- Sends to output channel
    |
    v  RecordBatch stream
    |
Results
```

## Model caching

Models are loaded once and cached with LRU eviction:

- **First load**: downloads from HF Hub (or reads from local path), loads weights into memory
- **Subsequent calls**: cache hit, returns immediately
- **Ref counting**: model stays in memory while any inference is running
- **Eviction**: when the LRU limit is reached, the least-recently-used model with no active references is evicted
