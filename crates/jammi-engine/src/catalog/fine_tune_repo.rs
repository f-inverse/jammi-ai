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
    /// Create a new fine-tune job record with status = 'queued'. Tenant
    /// bound + asserted (SPEC-03 §7).
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
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "fine_tune_jobs")?;
                    tx.execute(
                        "INSERT INTO fine_tune_jobs \
                         (job_id, base_model_id, training_source, loss_type, hyperparams, status, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, 'queued', $6)",
                        &[
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(base_model_id),
                            SqlValue::TextOwned(training_source),
                            SqlValue::TextOwned(loss_type),
                            SqlValue::TextOwned(hyperparams),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Get a fine-tune job by ID. Tenant-filtered.
    pub async fn get_fine_tune_job(&self, job_id: &str) -> Result<FineTuneJobRecord> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM fine_tune_jobs WHERE job_id = $1 \
               AND (tenant_id = $2 OR tenant_id IS NULL)"
        );
        let id = job_id.to_string();
        let id_for_err = id.clone();
        let tenant = self.current_tenant();
        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            &sql,
                            &[
                                SqlValue::TextOwned(id),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        found.ok_or_else(|| JammiError::Catalog(format!("Fine-tune job '{id_for_err}' not found")))
    }

    /// Update a fine-tune job's status and optional metrics JSON. Scoped.
    pub async fn update_fine_tune_status(
        &self,
        job_id: &str,
        status: super::status::FineTuneJobStatus,
        metrics: Option<&str>,
    ) -> Result<()> {
        let status_str = status.to_string();
        let metrics = metrics.map(str::to_string);
        let id = job_id.to_string();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "UPDATE fine_tune_jobs SET status = $1, metrics = $2, \
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE job_id = $3 AND (tenant_id = $4 OR tenant_id IS NULL)",
                        &[
                            SqlValue::TextOwned(status_str),
                            SqlValue::from(metrics),
                            SqlValue::TextOwned(id),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Set the output model ID for a completed fine-tune job. Scoped.
    pub async fn set_fine_tune_output_model(
        &self,
        job_id: &str,
        output_model_id: &str,
    ) -> Result<()> {
        let output_model_id = output_model_id.to_string();
        let id = job_id.to_string();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "UPDATE fine_tune_jobs SET output_model_id = $1, \
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE job_id = $2 AND (tenant_id = $3 OR tenant_id IS NULL)",
                        &[
                            SqlValue::TextOwned(output_model_id),
                            SqlValue::TextOwned(id),
                            SqlValue::from(tenant.map(|t| t.to_string())),
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
    /// crashed mid-training. Scoped to the session tenant.
    pub async fn cleanup_stale_fine_tune_jobs(&self) -> Result<usize> {
        let running = super::status::FineTuneJobStatus::Running.to_string();
        let failed = super::status::FineTuneJobStatus::Failed.to_string();
        let tenant = self.current_tenant();
        let count = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "UPDATE fine_tune_jobs SET status = $1, \
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE status = $2 AND (tenant_id = $3 OR tenant_id IS NULL)",
                        &[
                            SqlValue::TextOwned(failed),
                            SqlValue::TextOwned(running),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(count as usize)
    }

    /// List fine-tune jobs visible to the session tenant, most recent first.
    pub async fn list_fine_tune_jobs(&self) -> Result<Vec<FineTuneJobRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM fine_tune_jobs \
             WHERE tenant_id = $1 OR tenant_id IS NULL \
             ORDER BY created_at DESC"
        );
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            &sql,
                            &[SqlValue::from(tenant.map(|t| t.to_string()))],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}
