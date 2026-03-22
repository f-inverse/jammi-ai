use arrow::array::RecordBatch;
use std::time::Duration;

/// Optional hook for observing inference output batches.
/// Near-zero cost when not attached (single Option branch per batch).
pub trait InferenceObserver: Send + Sync {
    fn on_batch(&self, batch: &RecordBatch, model_id: &str, latency: Duration);
}
