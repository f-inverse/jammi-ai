pub mod classification;
pub mod embedding;
pub mod ner;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::Field;
use jammi_engine::error::Result;

pub use classification::ClassificationAdapter;
pub use embedding::EmbeddingAdapter;

use crate::model::{LoadedModel, ModelTask};

/// Raw output from a model backend, before task-specific adaptation.
pub struct BackendOutput {
    /// Numeric output tensors flattened to 1-D (one vec per output head).
    pub float_outputs: Vec<Vec<f32>>,
    /// String output tensors (one vec per output head).
    pub string_outputs: Vec<Vec<String>>,
    /// Per-row success flag (`true` = inference succeeded).
    pub row_status: Vec<bool>,
    /// Per-row error message (empty string when status is `true`).
    pub row_errors: Vec<String>,
    /// Shape metadata for each float output as `(rows, cols)`.
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
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding => {
            let dim = model.embedding_dim().ok_or_else(|| {
                JammiError::Inference("Model does not report embedding dim".into())
            })?;
            Ok(Box::new(EmbeddingAdapter::new(dim)))
        }
        ModelTask::Classification => Ok(Box::new(ClassificationAdapter)),
        ModelTask::Ner => Ok(Box::new(ner::NerAdapter)),
    }
}

/// Create an adapter for schema construction only (no model needed).
/// For embedding, pass the model's hidden_size as `embedding_dim`.
pub(crate) fn create_adapter_for_schema(
    task: ModelTask,
    embedding_dim: Option<usize>,
) -> Box<dyn OutputAdapter> {
    match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding => {
            Box::new(EmbeddingAdapter::new(embedding_dim.unwrap_or(0)))
        }
        ModelTask::Classification => Box::new(ClassificationAdapter),
        ModelTask::Ner => Box::new(ner::NerAdapter),
    }
}

// ─── Shared null-handling helpers ────────────────────────────────────────────

/// Build a nullable StringArray: rows where `row_status[i]` is false become null.
pub(crate) fn nullify_strings(
    values: Option<&Vec<String>>,
    row_status: &[bool],
    row_count: usize,
) -> StringArray {
    match values {
        Some(v) => v
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if row_status.get(i).copied().unwrap_or(false) {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect(),
        None => vec![None::<&str>; row_count].into_iter().collect(),
    }
}

/// Build a nullable Float32Array: rows where `row_status[i]` is false become null.
pub(crate) fn nullify_floats(
    values: Option<&Vec<f32>>,
    row_status: &[bool],
    row_count: usize,
) -> Float32Array {
    match values {
        Some(v) => v
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if row_status.get(i).copied().unwrap_or(false) {
                    Some(c)
                } else {
                    None
                }
            })
            .collect(),
        None => vec![None::<f32>; row_count].into_iter().collect(),
    }
}

/// Create dummy BackendOutput for an all-error batch of a given task.
pub(crate) fn create_error_output(
    task: ModelTask,
    row_count: usize,
    embedding_dim: usize,
) -> BackendOutput {
    let (float_outputs, string_outputs) = match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding => {
            (vec![vec![0.0; row_count * embedding_dim]], vec![])
        }
        ModelTask::Classification => (
            vec![vec![0.0; row_count]],
            vec![
                vec![String::new(); row_count],
                vec![String::new(); row_count],
            ],
        ),
        ModelTask::Ner => (vec![], vec![vec![String::new(); row_count]]),
    };
    BackendOutput {
        float_outputs,
        string_outputs,
        row_status: vec![false; row_count],
        row_errors: vec![String::new(); row_count],
        shapes: vec![(row_count, 0)],
    }
}
