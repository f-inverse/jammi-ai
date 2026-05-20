use serde::{Deserialize, Serialize};

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::Catalog;
use crate::error::Result;

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

fn parse_eval_row(row: &Row<'_>) -> std::result::Result<EvalRunRecord, BackendError> {
    Ok(EvalRunRecord {
        eval_run_id: row.get("run_id")?,
        eval_type: row.get("eval_type")?,
        model_id: row.get("model_id")?,
        source_id: row.get("source_id")?,
        golden_source: row.try_get("golden_source")?.unwrap_or_default(),
        k: row.try_get::<i64>("k")?.map(|v| v as i32),
        metrics_json: row.get("metrics")?,
        status: row.try_get("status")?.unwrap_or_else(|| "completed".into()),
        created_at: row.get("created_at")?,
    })
}

impl Catalog {
    /// Insert a new eval run record.
    pub async fn record_eval_run(&self, record: &EvalRunRecord) -> Result<()> {
        let r = record.clone();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO eval_runs \
                         (run_id, eval_type, model_id, source_id, golden_source, k, metrics, status, created_at) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                        &[
                            SqlValue::TextOwned(r.eval_run_id),
                            SqlValue::TextOwned(r.eval_type),
                            SqlValue::TextOwned(r.model_id),
                            SqlValue::TextOwned(r.source_id),
                            SqlValue::TextOwned(r.golden_source),
                            SqlValue::from(r.k.map(|v| v as i64)),
                            SqlValue::TextOwned(r.metrics_json),
                            SqlValue::TextOwned(r.status),
                            SqlValue::TextOwned(r.created_at),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Get a single eval run by ID.
    pub async fn get_eval_run(&self, eval_run_id: &str) -> Result<Option<EvalRunRecord>> {
        let sql = format!("SELECT {SELECT_COLS} FROM eval_runs WHERE run_id = $1");
        let id = eval_run_id.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(&sql, &[SqlValue::TextOwned(id)], parse_eval_row)
                            .await
                    })
                },
            )
            .await?)
    }

    /// List all eval runs, most recent first.
    pub async fn list_eval_runs(&self) -> Result<Vec<EvalRunRecord>> {
        let sql = format!("SELECT {SELECT_COLS} FROM eval_runs ORDER BY created_at DESC");
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { tx.query(&sql, &[], parse_eval_row).await }),
            )
            .await?)
    }

    /// Most recent eval run for a given model + eval type.
    pub async fn latest_eval_run(
        &self,
        model_id: &str,
        eval_type: &str,
    ) -> Result<Option<EvalRunRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM eval_runs \
             WHERE model_id = $1 AND eval_type = $2 \
             ORDER BY created_at DESC LIMIT 1"
        );
        let mid = model_id.to_string();
        let et = eval_type.to_string();
        Ok(self
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
                            &[SqlValue::TextOwned(mid), SqlValue::TextOwned(et)],
                            parse_eval_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}
