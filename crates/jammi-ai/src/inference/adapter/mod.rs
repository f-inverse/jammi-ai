pub mod classification;
pub mod distribution;
pub mod embedding;
pub mod ner;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::Field;
use jammi_db::error::Result;

pub use classification::ClassificationAdapter;
pub use distribution::{DistributionAdapter, DistributionForm};
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

    /// A well-formed all-error [`BackendOutput`] for `row_count` rows, shaped so
    /// `self.adapt(&output, row_count)` yields this adapter's exact columns (all
    /// null). The error read path uses this so the failure batch matches the
    /// served schema — critical for a quantile head, whose width is the level
    /// count, not the Gaussian default of 2.
    fn error_output(&self, row_count: usize) -> BackendOutput;
}

/// Create an adapter for a given task with model-derived dimensions.
pub fn create_adapter(task: ModelTask, model: &LoadedModel) -> Result<Box<dyn OutputAdapter>> {
    use jammi_db::error::JammiError;
    match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding | ModelTask::AudioEmbedding => {
            let dim = model.embedding_dim().ok_or_else(|| {
                JammiError::Inference("Model does not report embedding dim".into())
            })?;
            Ok(Box::new(EmbeddingAdapter::new(dim)))
        }
        ModelTask::Classification => Ok(Box::new(ClassificationAdapter)),
        ModelTask::Ner => Ok(Box::new(ner::NerAdapter)),
        // A regression model serves the form its head was trained for, read
        // from the head's persisted `DistributionForm`: `Gaussian` →
        // `(predicted_mean, predicted_std)`, `Quantile { levels }` → one
        // `quantile_{level}` column per level. Selecting on the persisted form
        // (never on head width — a 2-level quantile head is also width 2) is
        // what stops a quantile-trained head being silently mis-decoded as a
        // Gaussian `(mean, std)` on the public `Infer` read path. A regression
        // head saved without a form (none today) falls back to Gaussian, the
        // density-bearing core form.
        ModelTask::Regression => match model.regression_form() {
            Some(DistributionForm::Quantile { levels }) => {
                Ok(Box::new(DistributionAdapter::quantile(levels.clone())?))
            }
            // A z-space-trained Gaussian head learns a z-scale σ (σ_z ≈ 1); the
            // adapter scales it back to raw units by σ_y (the persisted scaler's
            // std) on the post-softplus column. A head with no scaler (none today
            // for a trained regression model) serves σ unscaled (`std_scale = 1`).
            Some(DistributionForm::Gaussian) | None => Ok(Box::new(
                DistributionAdapter::gaussian_scaled(model.regression_std_scale().unwrap_or(1.0)),
            )),
        },
    }
}

/// Create an adapter for schema construction only (no model handle needed).
///
/// For embedding, pass the model's hidden_size as `embedding_dim`. For
/// regression, pass the head's persisted `regression_form` so the planned
/// output schema matches what the runtime adapter (built from the loaded model
/// via [`create_adapter`]) emits — a quantile head's schema is its level
/// columns, not the Gaussian default. `None` form falls back to Gaussian, and
/// a malformed persisted quantile form (it was validated at fine-tune time)
/// likewise falls back rather than failing schema planning.
pub(crate) fn create_adapter_for_schema(
    task: ModelTask,
    embedding_dim: Option<usize>,
    regression_form: Option<&DistributionForm>,
) -> Box<dyn OutputAdapter> {
    match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding | ModelTask::AudioEmbedding => {
            Box::new(EmbeddingAdapter::new(embedding_dim.unwrap_or(0)))
        }
        ModelTask::Classification => Box::new(ClassificationAdapter),
        ModelTask::Ner => Box::new(ner::NerAdapter),
        ModelTask::Regression => match regression_form {
            Some(DistributionForm::Quantile { levels }) => {
                DistributionAdapter::quantile(levels.clone()).map_or_else(
                    |_| Box::new(DistributionAdapter::gaussian()) as Box<dyn OutputAdapter>,
                    |a| Box::new(a) as Box<dyn OutputAdapter>,
                )
            }
            Some(DistributionForm::Gaussian) | None => Box::new(DistributionAdapter::gaussian()),
        },
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
