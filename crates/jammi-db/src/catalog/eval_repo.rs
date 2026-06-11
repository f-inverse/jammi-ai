use serde::{Deserialize, Serialize};

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::Catalog;
use crate::error::Result;

/// A row from the `eval_runs` catalog table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRunRecord {
    pub eval_run_id: String,
    pub eval_type: String,
    /// The catalog PK of the model whose output this run scored, or `None` for
    /// a run that is not model-scoped (a calibration eval scores a held-out
    /// predictive distribution, not a registered model). When present it must
    /// reference a real `models(model_id)` row — the catalog enforces the FK.
    pub model_id: Option<String>,
    pub source_id: String,
    pub golden_source: String,
    pub k: Option<i32>,
    pub metrics_json: String,
    pub status: String,
    pub created_at: String,
}

/// A single per-query eval record, persisted in `_jammi_eval_per_query` and
/// keyed by `(eval_run_id, query_id)` (spec J9).
///
/// `metrics_json` carries the per-query metric vector — Recall@{1,3,5,10}, MRR,
/// nDCG, distance — as a JSON object; `cohorts_json` carries an opaque
/// `{key: value}` segment map (`"{}"` when none supplied at eval time). The
/// substrate stores both verbatim and never interprets cohort keys/values —
/// declaration/validation is a downstream consumer's concern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryEvalRecord {
    pub eval_run_id: String,
    pub query_id: String,
    /// Opaque cohort tags as a JSON object string; `"{}"` when none.
    pub cohorts_json: String,
    /// Per-query metrics as a JSON object string.
    pub metrics_json: String,
}

const SELECT_COLS: &str =
    "run_id, eval_type, model_id, source_id, golden_source, k, metrics, status, created_at";

const PER_QUERY_SELECT_COLS: &str = "eval_run_id, query_id, cohorts, metrics";

fn parse_per_query_row(row: &Row<'_>) -> std::result::Result<PerQueryEvalRecord, BackendError> {
    Ok(PerQueryEvalRecord {
        eval_run_id: row.get("eval_run_id")?,
        query_id: row.get("query_id")?,
        cohorts_json: row.get("cohorts")?,
        metrics_json: row.get("metrics")?,
    })
}

fn parse_eval_row(row: &Row<'_>) -> std::result::Result<EvalRunRecord, BackendError> {
    Ok(EvalRunRecord {
        eval_run_id: row.get("run_id")?,
        eval_type: row.get("eval_type")?,
        model_id: row.try_get("model_id")?,
        source_id: row.get("source_id")?,
        golden_source: row.try_get("golden_source")?.unwrap_or_default(),
        k: row.try_get::<i32>("k")?,
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
                            SqlValue::from(r.model_id),
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

    /// Bulk-insert the per-query eval records for one run. Tenant bound +
    /// asserted on every row (SPEC-03 §7), mirroring `record_eval_run`.
    ///
    /// All rows for a run are written in a single batched, multi-row `INSERT`
    /// inside one tenant-bound transaction — the same batch-write shape the J2
    /// audit log uses — so a run's per-query array is persisted atomically. The
    /// records are validated against the bound tenant before any row is issued;
    /// an empty input is a no-op.
    pub async fn record_eval_per_query(&self, records: &[PerQueryEvalRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let rows = records.to_vec();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "_jammi_eval_per_query")?;

                    // Build one multi-row INSERT with positional placeholders so
                    // every row lands in a single round-trip (J2 batch shape).
                    let mut sql = String::from(
                        "INSERT INTO _jammi_eval_per_query \
                         (eval_run_id, query_id, cohorts, metrics, tenant_id) VALUES ",
                    );
                    let mut params: Vec<SqlValue<'_>> = Vec::with_capacity(rows.len() * 5);
                    let tenant_str = tenant.map(|t| t.to_string());
                    for (i, rec) in rows.iter().enumerate() {
                        let base = i * 5;
                        if i > 0 {
                            sql.push_str(", ");
                        }
                        sql.push_str(&format!(
                            "(${}, ${}, ${}, ${}, ${})",
                            base + 1,
                            base + 2,
                            base + 3,
                            base + 4,
                            base + 5
                        ));
                        params.push(SqlValue::TextOwned(rec.eval_run_id.clone()));
                        params.push(SqlValue::TextOwned(rec.query_id.clone()));
                        params.push(SqlValue::TextOwned(rec.cohorts_json.clone()));
                        params.push(SqlValue::TextOwned(rec.metrics_json.clone()));
                        params.push(SqlValue::from(tenant_str.clone()));
                    }
                    tx.execute(&sql, &params).await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Read back the per-query eval records for a run, scoped to the session
    /// tenant (a caller never sees another tenant's rows). Ordered by
    /// `query_id` for a stable read shape.
    pub async fn get_eval_per_query(&self, eval_run_id: &str) -> Result<Vec<PerQueryEvalRecord>> {
        let sql = format!(
            "SELECT {PER_QUERY_SELECT_COLS} FROM _jammi_eval_per_query \
             WHERE eval_run_id = $1 AND (tenant_id = $2 OR tenant_id IS NULL) \
             ORDER BY query_id"
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
                        tx.query(
                            &sql,
                            &[
                                SqlValue::TextOwned(id),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_per_query_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}
