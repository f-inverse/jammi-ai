//! Fine-tune job lifecycle management.

use std::sync::Arc;
use std::time::Duration;

use jammi_engine::catalog::status::FineTuneJobStatus;
use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};

/// Handle to a fine-tune job. Can be used to poll status or wait for completion.
pub struct FineTuneJob {
    /// Unique job identifier.
    pub job_id: String,
    /// Current status at creation time.
    pub status: String,
    /// Model ID for the fine-tuned output (set after completion).
    pub model_id: String,
    catalog: Arc<Catalog>,
}

impl std::fmt::Debug for FineTuneJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FineTuneJob")
            .field("job_id", &self.job_id)
            .field("status", &self.status)
            .field("model_id", &self.model_id)
            .finish()
    }
}

impl FineTuneJob {
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
            let record = self.catalog.get_fine_tune_job(&self.job_id)?;
            let status: FineTuneJobStatus = record
                .status
                .parse()
                .map_err(|e| JammiError::FineTune(format!("{e}")))?;
            match status {
                FineTuneJobStatus::Completed => return Ok(()),
                FineTuneJobStatus::Failed => {
                    let msg = record.error_message.unwrap_or_else(|| "Job failed".into());
                    return Err(JammiError::FineTune(msg));
                }
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    }

    /// Get the current status from the catalog.
    pub fn status(&self) -> Result<String> {
        let record = self.catalog.get_fine_tune_job(&self.job_id)?;
        Ok(record.status)
    }

    /// The model ID for the fine-tuned output.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}
