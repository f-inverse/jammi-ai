pub mod classification;
pub mod embedding;
pub mod ner;
pub mod object_detection;
pub mod summarization;
pub mod text_generation;

use arrow::array::ArrayRef;
use arrow::datatypes::Field;
use jammi_engine::error::Result;

pub use classification::ClassificationAdapter;
pub use embedding::EmbeddingAdapter;

use crate::model::{LoadedModel, ModelTask};

/// Raw output from a model backend.
pub struct BackendOutput {
    pub float_outputs: Vec<Vec<f32>>,
    pub string_outputs: Vec<Vec<String>>,
    pub row_status: Vec<bool>,
    pub row_errors: Vec<String>,
    pub shapes: Vec<(usize, usize)>,
}

/// Converts raw backend output into Arrow arrays for a specific task.
pub trait OutputAdapter: Send + Sync {
    /// Arrow schema for this task's output columns (excluding common prefix).
    fn output_schema(&self) -> Vec<Field>;

    /// Convert raw backend output into Arrow arrays for one batch.
    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>>;
}

/// Create an adapter for a given task with model-derived dimensions.
pub fn create_adapter(task: ModelTask, model: &LoadedModel) -> Result<Box<dyn OutputAdapter>> {
    use jammi_engine::error::JammiError;
    match task {
        ModelTask::Embedding => {
            let dim = model.embedding_dim().ok_or_else(|| {
                JammiError::Inference("Model does not report embedding dim".into())
            })?;
            Ok(Box::new(EmbeddingAdapter::new(dim)))
        }
        ModelTask::Classification => Ok(Box::new(ClassificationAdapter)),
        ModelTask::Summarization => Ok(Box::new(summarization::SummarizationAdapter)),
        ModelTask::ObjectDetection => Ok(Box::new(object_detection::ObjectDetectionAdapter)),
        ModelTask::Ner => Ok(Box::new(ner::NerAdapter)),
        ModelTask::TextGeneration => Ok(Box::new(text_generation::TextGenerationAdapter)),
    }
}

/// Create an adapter for schema construction only (no model needed).
pub fn create_adapter_for_schema(task: ModelTask) -> Box<dyn OutputAdapter> {
    match task {
        ModelTask::Embedding => Box::new(EmbeddingAdapter::new(0)),
        ModelTask::Classification => Box::new(ClassificationAdapter),
        ModelTask::Summarization => Box::new(summarization::SummarizationAdapter),
        ModelTask::ObjectDetection => Box::new(object_detection::ObjectDetectionAdapter),
        ModelTask::Ner => Box::new(ner::NerAdapter),
        ModelTask::TextGeneration => Box::new(text_generation::TextGenerationAdapter),
    }
}
