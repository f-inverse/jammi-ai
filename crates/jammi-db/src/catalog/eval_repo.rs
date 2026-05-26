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
    /// Insert a new eval run record. Tenant bound + asserted (SPEC-03 §7).
    pub async fn record_eval_run(&self, record: &EvalRunRecord) -> Result<()> {
        let r = record.clone();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "eval_runs")?;
                    tx.execute(
                        "INSERT INTO eval_runs \
                         (run_id, eval_type, model_id, source_id, golden_source, k, metrics, status, created_at, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
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

    /// Get a single eval run by ID. Tenant-filtered.
    pub async fn get_eval_run(&self, eval_run_id: &str) -> Result<Option<EvalRunRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM eval_runs WHERE run_id = $1 \
               AND (tenant_id = $2 OR tenant_id IS NULL)"
        );
        let id = eval_run_id.to_string();
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
                        tx.query_opt(
                            &sql,
                            &[
                                SqlValue::TextOwned(id),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_eval_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// List eval runs visible to the session tenant, most recent first.
    pub async fn list_eval_runs(&self) -> Result<Vec<EvalRunRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM eval_runs \
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
                            parse_eval_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Most recent eval run for a given model + eval type. Tenant-filtered.
    pub async fn latest_eval_run(
        &self,
        model_id: &str,
        eval_type: &str,
    ) -> Result<Option<EvalRunRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM eval_runs \
             WHERE model_id = $1 AND eval_type = $2 \
               AND (tenant_id = $3 OR tenant_id IS NULL) \
             ORDER BY created_at DESC LIMIT 1"
        );
        let mid = model_id.to_string();
        let et = eval_type.to_string();
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
                        tx.query_opt(
                            &sql,
                            &[
                                SqlValue::TextOwned(mid),
                                SqlValue::TextOwned(et),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_eval_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}
