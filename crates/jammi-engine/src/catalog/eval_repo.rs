use rusqlite::params;
use serde::{Deserialize, Serialize};

use super::Catalog;
use crate::error::{JammiError, Result};

/// A row from the `eval_runs` catalog table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRunRecord {
    pub eval_run_id: String,
    pub eval_type: String,
    pub model_id: String,
    pub source_id: String,
    pub golden_source: String,
    pub k: Option<i32>,
    pub metrics_json: String,
    pub status: String,
    pub created_at: String,
}

const SELECT_COLS: &str =
    "run_id, eval_type, model_id, source_id, golden_source, k, metrics, status, created_at";

fn parse_eval_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<EvalRunRecord> {
    Ok(EvalRunRecord {
        eval_run_id: row.get(0)?,
        eval_type: row.get(1)?,
        model_id: row.get(2)?,
        source_id: row.get(3)?,
        golden_source: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        k: row.get(5)?,
        metrics_json: row.get(6)?,
        status: row
            .get::<_, Option<String>>(7)?
            .unwrap_or_else(|| "completed".into()),
        created_at: row.get(8)?,
    })
}

impl Catalog {
    /// Insert a new eval run record.
    pub fn record_eval_run(&self, record: &EvalRunRecord) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO eval_runs \
             (run_id, eval_type, model_id, source_id, golden_source, k, metrics, status, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.eval_run_id,
                record.eval_type,
                record.model_id,
                record.source_id,
                record.golden_source,
                record.k,
                record.metrics_json,
                record.status,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    /// Get a single eval run by ID.
    pub fn get_eval_run(&self, eval_run_id: &str) -> Result<Option<EvalRunRecord>> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM eval_runs WHERE run_id = ?1");
        let result = conn.query_row(&sql, params![eval_run_id], parse_eval_row);
        match result {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(JammiError::Catalog(format!(
                "Eval run '{eval_run_id}': {e}"
            ))),
        }
    }

    /// List all eval runs, most recent first.
    pub fn list_eval_runs(&self) -> Result<Vec<EvalRunRecord>> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM eval_runs ORDER BY created_at DESC");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], parse_eval_row)?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    /// Most recent eval run for a given model + eval type.
    /// Used by ExperimentRunner (Phase 13) to read the metric being optimized.
    pub fn latest_eval_run(
        &self,
        model_id: &str,
        eval_type: &str,
    ) -> Result<Option<EvalRunRecord>> {
        let conn = self.conn()?;
        let sql = format!(
            "SELECT {SELECT_COLS} FROM eval_runs \
             WHERE model_id = ?1 AND eval_type = ?2 \
             ORDER BY created_at DESC LIMIT 1"
        );
        let result = conn.query_row(&sql, params![model_id, eval_type], parse_eval_row);
        match result {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
