use serde::{Deserialize, Serialize};

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::Catalog;
use crate::error::{JammiError, Result};

/// A row from the `fine_tune_jobs` catalog table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuneJobRecord {
    pub job_id: String,
    pub base_model_id: String,
    pub output_model_id: Option<String>,
    pub training_source: String,
    pub loss_type: String,
    pub hyperparams: String,
    pub status: String,
    pub metrics: Option<String>,
    pub error_message: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

const SELECT_COLS: &str = "job_id, base_model_id, output_model_id, training_source, loss_type, \
     hyperparams, status, metrics, created_at";

fn parse_row(row: &Row<'_>) -> std::result::Result<FineTuneJobRecord, BackendError> {
    let metrics_raw: Option<String> = row.try_get("metrics")?;
    let error_message = metrics_raw.as_deref().and_then(|m| {
        serde_json::from_str::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v["error_message"].as_str().map(String::from))
    });
    let started_at = metrics_raw.as_deref().and_then(|m| {
        serde_json::from_str::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v["started_at"].as_str().map(String::from))
    });
    let completed_at = metrics_raw.as_deref().and_then(|m| {
        serde_json::from_str::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v["completed_at"].as_str().map(String::from))
    });

    Ok(FineTuneJobRecord {
        job_id: row.get("job_id")?,
        base_model_id: row.get("base_model_id")?,
        output_model_id: row.try_get("output_model_id")?,
        training_source: row.get("training_source")?,
        loss_type: row.get("loss_type")?,
        hyperparams: row.get("hyperparams")?,
        status: row.get("status")?,
        metrics: metrics_raw,
        error_message,
        created_at: row.get("created_at")?,
        started_at,
        completed_at,
    })
}

impl Catalog {
    /// Create a new fine-tune job record with status = 'queued'.
    pub async fn create_fine_tune_job(
        &self,
        job_id: &str,
        base_model_id: &str,
        training_source: &str,
        loss_type: &str,
        hyperparams: &str,
    ) -> Result<()> {
        let job_id = job_id.to_string();
        let base_model_id = base_model_id.to_string();
        let training_source = training_source.to_string();
        let loss_type = loss_type.to_string();
        let hyperparams = hyperparams.to_string();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO fine_tune_jobs \
                         (job_id, base_model_id, training_source, loss_type, hyperparams, status) \
                         VALUES ($1, $2, $3, $4, $5, 'queued')",
                        &[
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(base_model_id),
                            SqlValue::TextOwned(training_source),
                            SqlValue::TextOwned(loss_type),
                            SqlValue::TextOwned(hyperparams),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Get a fine-tune job by ID.
    pub async fn get_fine_tune_job(&self, job_id: &str) -> Result<FineTuneJobRecord> {
        let sql = format!("SELECT {SELECT_COLS} FROM fine_tune_jobs WHERE job_id = $1");
        let id = job_id.to_string();
        let id_for_err = id.clone();
        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(&sql, &[SqlValue::TextOwned(id)], parse_row)
                            .await
                    })
                },
            )
            .await?;
        found.ok_or_else(|| JammiError::Catalog(format!("Fine-tune job '{id_for_err}' not found")))
    }

    /// Update a fine-tune job's status and optional metrics JSON.
    pub async fn update_fine_tune_status(
        &self,
        job_id: &str,
        status: super::status::FineTuneJobStatus,
        metrics: Option<&str>,
    ) -> Result<()> {
        let status_str = status.to_string();
        let metrics = metrics.map(str::to_string);
        let id = job_id.to_string();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE fine_tune_jobs SET status = $1, metrics = $2, \
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE job_id = $3",
                        &[
                            SqlValue::TextOwned(status_str),
                            SqlValue::from(metrics),
                            SqlValue::TextOwned(id),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Set the output model ID for a completed fine-tune job.
    pub async fn set_fine_tune_output_model(
        &self,
        job_id: &str,
        output_model_id: &str,
    ) -> Result<()> {
        let output_model_id = output_model_id.to_string();
        let id = job_id.to_string();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE fine_tune_jobs SET output_model_id = $1, \
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE job_id = $2",
                        &[
                            SqlValue::TextOwned(output_model_id),
                            SqlValue::TextOwned(id),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Transition all fine-tune jobs with status Running to Failed.
    /// Startup-recovery shim: a Running job at startup means the process
    /// crashed mid-training.
    pub async fn cleanup_stale_fine_tune_jobs(&self) -> Result<usize> {
        let running = super::status::FineTuneJobStatus::Running.to_string();
        let failed = super::status::FineTuneJobStatus::Failed.to_string();
        let count = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE fine_tune_jobs SET status = $1, \
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE status = $2",
                        &[SqlValue::TextOwned(failed), SqlValue::TextOwned(running)],
                    )
                    .await
                })
            })
            .await?;
        Ok(count as usize)
    }

    /// List all fine-tune jobs, most recent first.
    pub async fn list_fine_tune_jobs(&self) -> Result<Vec<FineTuneJobRecord>> {
        let sql = format!("SELECT {SELECT_COLS} FROM fine_tune_jobs ORDER BY created_at DESC");
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { tx.query(&sql, &[], parse_row).await }),
            )
            .await?)
    }
}
