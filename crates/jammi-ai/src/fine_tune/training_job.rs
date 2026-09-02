//! Training job lifecycle management.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::status::TrainingJobStatus;
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};

/// Handle to a training job. Can be used to poll status or wait for completion.
pub struct TrainingJob {
    /// Unique job identifier.
    pub job_id: String,
    /// Current status at creation time.
    pub status: String,
    /// Model ID for the trained output (set after completion).
    pub model_id: String,
    catalog: Arc<Catalog>,
}

impl std::fmt::Debug for TrainingJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingJob")
            .field("job_id", &self.job_id)
            .field("status", &self.status)
            .field("model_id", &self.model_id)
            .finish()
    }
}

impl TrainingJob {
    /// Create a new job handle.
    pub(crate) fn new(
        job_id: String,
        status: String,
        model_id: String,
        catalog: Arc<Catalog>,
    ) -> Self {
        Self {
            job_id,
            status,
            model_id,
            catalog,
        }
    }

    /// Block until the job reaches a terminal state (completed or failed).
    pub async fn wait(&self) -> Result<()> {
        loop {
            let record = self.catalog.get_training_job(&self.job_id).await?;
            let status: TrainingJobStatus = record
                .status
                .parse()
                .map_err(|e| JammiError::FineTune(format!("{e}")))?;
            match status {
                TrainingJobStatus::Completed => return Ok(()),
                TrainingJobStatus::Failed => {
                    let msg = record.error_message.unwrap_or_else(|| "Job failed".into());
                    return Err(JammiError::FineTune(msg));
                }
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    }

    /// Get the current status from the catalog.
    pub async fn status(&self) -> Result<String> {
        let record = self.catalog.get_training_job(&self.job_id).await?;
        Ok(record.status)
    }

    /// The model ID for the trained output.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// The deterministic output model id the two LoRA fine-tune kinds
/// (`fine_tune`, `graph_fine_tune`) register their trained artifact under.
///
/// THE naming rule, in one place: every producer of this value — the session's
/// submit path, the worker's output registration, and [`resolve_model_id`]'s
/// re-derivation for a handle that never saw the submit call — calls this
/// function rather than re-spelling the format string, so no surface can mint a
/// different id for the same job.
pub fn fine_tuned_model_id(job_id: &str) -> String {
    format!("jammi:fine-tuned:{job_id}")
}

/// Resolve the output model id a job reports, from its catalog row alone — no
/// in-memory carryover from the submit call, for the callers that never had one
/// (an attach-by-id handle, and the `TrainingStatus` wire read that backs the
/// remote one).
///
/// Once the job has completed, `record.output_model_id` (stamped by the
/// worker's finalize) is authoritative and is used directly. Before that —
/// while the row is `queued` or `running`, and for a `failed` row that never
/// stamped one — the value is re-derived by the SAME rule the submit path
/// applies: the two LoRA fine-tune kinds mint the deterministic
/// [`fine_tuned_model_id`] (checked against `record.kind`, no spec decode
/// needed — the value depends only on `job_id`); the context-predictor kind
/// carries a caller-chosen id inside `predictor_spec.model_id`, which is NOT
/// derivable from `job_id` alone, so that one arm decodes the persisted
/// `training_spec`.
///
/// This is the engine's rule, and it lives here so both surfaces call it: the
/// embedded attach handle and the server's `TrainingStatus` handler report a
/// byte-identical id at every lifecycle state.
///
/// # Errors
///
/// A row whose `kind` is not one of the three engine training kinds, or a
/// context-predictor row whose persisted `training_spec` is absent or does not
/// decode to a context-predictor spec, cannot be resolved: the id is genuinely
/// unknown there, and a guessed or empty answer would misreport a naming rule
/// the engine owns.
pub fn resolve_model_id(
    job_id: &str,
    record: &jammi_db::catalog::training_repo::TrainingJobRecord,
) -> Result<String> {
    use crate::fine_tune::spec::TrainingSpec;

    if let Some(model_id) = &record.output_model_id {
        return Ok(model_id.clone());
    }
    match record.kind.as_str() {
        "fine_tune" | "graph_fine_tune" => Ok(fine_tuned_model_id(job_id)),
        "context_predictor" => {
            let raw = record.training_spec.as_deref().ok_or_else(|| {
                JammiError::Catalog(format!(
                    "training job {job_id}: missing persisted training_spec; \
                     cannot resolve model_id before completion"
                ))
            })?;
            let spec: TrainingSpec = serde_json::from_str(raw).map_err(|parse_err| {
                JammiError::Catalog(format!(
                    "training job {job_id}: training_spec failed to parse as JSON: {parse_err}"
                ))
            })?;
            match spec {
                TrainingSpec::ContextPredictor { predictor_spec, .. } => {
                    Ok(predictor_spec.model_id)
                }
                _ => Err(JammiError::Catalog(format!(
                    "training job {job_id}: persisted training_spec kind does not match \
                     catalog kind 'context_predictor'"
                ))),
            }
        }
        other => Err(JammiError::Catalog(format!(
            "training job {job_id}: unrecognised training job kind '{other}'"
        ))),
    }
}
