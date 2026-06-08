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
