use rusqlite::params;
use serde::{Deserialize, Serialize};

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

fn parse_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<FineTuneJobRecord> {
    let metrics_raw: Option<String> = row.get(7)?;
    // Extract error_message from metrics JSON if present
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
        job_id: row.get(0)?,
        base_model_id: row.get(1)?,
        output_model_id: row.get(2)?,
        training_source: row.get(3)?,
        loss_type: row.get(4)?,
        hyperparams: row.get(5)?,
        status: row.get(6)?,
        metrics: metrics_raw,
        error_message,
        created_at: row.get(8)?,
        started_at,
        completed_at,
    })
}

impl Catalog {
    /// Create a new fine-tune job record with status = 'queued'.
    pub fn create_fine_tune_job(
        &self,
        job_id: &str,
        base_model_id: &str,
        training_source: &str,
        loss_type: &str,
        hyperparams: &str,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO fine_tune_jobs \
             (job_id, base_model_id, training_source, loss_type, hyperparams, status) \
             VALUES (?1, ?2, ?3, ?4, ?5, 'queued')",
            params![
                job_id,
                base_model_id,
                training_source,
                loss_type,
                hyperparams
            ],
        )?;
        Ok(())
    }

    /// Get a fine-tune job by ID.
    pub fn get_fine_tune_job(&self, job_id: &str) -> Result<FineTuneJobRecord> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM fine_tune_jobs WHERE job_id = ?1");
        conn.query_row(&sql, params![job_id], parse_row)
            .map_err(|e| JammiError::Catalog(format!("Fine-tune job '{job_id}': {e}")))
    }

    /// Update a fine-tune job's status and optional metrics JSON.
    pub fn update_fine_tune_status(
        &self,
        job_id: &str,
        status: super::status::FineTuneJobStatus,
        metrics: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let status_str = status.to_string();
        conn.execute(
            "UPDATE fine_tune_jobs SET status = ?1, metrics = ?2, \
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE job_id = ?3",
            params![status_str, metrics, job_id],
        )?;
        Ok(())
    }

    /// Set the output model ID for a completed fine-tune job.
    pub fn set_fine_tune_output_model(&self, job_id: &str, output_model_id: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE fine_tune_jobs SET output_model_id = ?1, \
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE job_id = ?2",
            params![output_model_id, job_id],
        )?;
        Ok(())
    }

    /// List all fine-tune jobs, most recent first.
    pub fn list_fine_tune_jobs(&self) -> Result<Vec<FineTuneJobRecord>> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM fine_tune_jobs ORDER BY created_at DESC");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], parse_row)?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }
}
