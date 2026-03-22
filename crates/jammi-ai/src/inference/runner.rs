use crate::model::cache::ModelCache;
use crate::model::{BackendType, ModelTask};

/// Processes input RecordBatches through a model, handling batching,
/// error recovery, and dynamic batch sizing.
pub struct InferenceRunner {
    pub model_cache: std::sync::Arc<ModelCache>,
    pub model_id: String,
    pub task: ModelTask,
    pub content_columns: Vec<String>,
    pub key_column: String,
    pub source_id: String,
    pub backend: Option<BackendType>,
    pub batch_size: usize,
}
