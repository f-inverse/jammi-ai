use arrow::array::RecordBatch;
use std::time::Duration;

/// Optional hook for observing inference output batches.
/// Near-zero cost when not attached (single Option branch per batch).
pub trait InferenceObserver: Send + Sync {
    /// Called once per output batch with the batch contents, model ID, and wall-clock latency.
    fn on_batch(&self, batch: &RecordBatch, model_id: &str, latency: Duration);
}
