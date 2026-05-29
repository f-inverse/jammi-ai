//! Read API for per-query audit records.
//!
//! All reads go through the session's tenant-scoped SQL path
//! (`mutable.public._jammi_search_audit`), so a caller only ever sees its own
//! tenant's records — the tenant-scope analyzer injects the predicate.
//!
//! The mutable-table provider exposes only the user-declared columns; the
//! implicit `tenant_id` column is not part of its Arrow schema and so cannot be
//! `SELECT`ed. Because every visible row belongs to the bound tenant (the
//! analyzer guarantees this), the read path stamps each decoded record's
//! `tenant_id` from the session binding rather than reading it back.

use arrow::array::{Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::session::JammiSession;

use super::error::AuditError;
use super::record::PerQueryAudit;
use super::table::AUDIT_TABLE_NAME;

const SELECT_COLUMNS: &str = "query_id, model_id, model_version, query_lineage, \
     top_k_result_ids, retrieval_scores, executed_at, signature";

fn audit_table_ref() -> String {
    format!("mutable.public.\"{AUDIT_TABLE_NAME}\"")
}

/// Fetch a single audit record by its query id, scoped to the session tenant.
///
/// Returns `Ok(None)` if no record with that id is visible to the tenant.
pub async fn fetch_by_query_id(
    session: &JammiSession,
    query_id: Uuid,
) -> Result<Option<PerQueryAudit>, AuditError> {
    // query_id is a UUID; its string form contains no quotes, so direct
    // interpolation is injection-safe here.
    let sql = format!(
        "SELECT {SELECT_COLUMNS} FROM {} WHERE query_id = '{query_id}'",
        audit_table_ref()
    );
    let batches = session
        .sql(&sql)
        .await
        .map_err(|e| AuditError::Storage(e.to_string()))?;
    let tenant = session.tenant().map(|t| t.to_string());
    let records = batches_to_records(&batches, tenant)?;
    Ok(records.into_iter().next())
}

/// Fetch the most recent audit records for the session tenant, newest first.
pub async fn fetch_recent(
    session: &JammiSession,
    limit: usize,
) -> Result<Vec<PerQueryAudit>, AuditError> {
    let sql = format!(
        "SELECT {SELECT_COLUMNS} FROM {} ORDER BY executed_at DESC LIMIT {limit}",
        audit_table_ref()
    );
    let batches = session
        .sql(&sql)
        .await
        .map_err(|e| AuditError::Storage(e.to_string()))?;
    let tenant = session.tenant().map(|t| t.to_string());
    batches_to_records(&batches, tenant)
}

/// Decode query-result batches into typed audit records, stamping `tenant_id`
/// from the session binding (the visible rows all belong to that tenant).
fn batches_to_records(
    batches: &[RecordBatch],
    tenant: Option<String>,
) -> Result<Vec<PerQueryAudit>, AuditError> {
    let mut out = Vec::new();
    for batch in batches {
        let query_id = str_col(batch, "query_id")?;
        let model_id = str_col(batch, "model_id")?;
        let model_version = str_col(batch, "model_version")?;
        let lineage = str_col(batch, "query_lineage")?;
        let top_k = str_col(batch, "top_k_result_ids")?;
        let scores = str_col(batch, "retrieval_scores")?;
        let executed_at = i64_col(batch, "executed_at")?;
        let signature = str_col(batch, "signature")?;

        for row in 0..batch.num_rows() {
            let executed_at = DateTime::<Utc>::from_timestamp_micros(executed_at.value(row))
                .ok_or_else(|| {
                    AuditError::Storage(format!(
                        "executed_at micros out of range: {}",
                        executed_at.value(row)
                    ))
                })?;
            out.push(PerQueryAudit {
                query_id: Uuid::parse_str(query_id.value(row))
                    .map_err(|e| AuditError::Storage(format!("invalid query_id: {e}")))?,
                tenant_id: tenant.clone(),
                model_id: model_id.value(row).to_string(),
                model_version: model_version.value(row).to_string(),
                query_lineage: serde_json::from_str(lineage.value(row))?,
                top_k_result_ids: serde_json::from_str(top_k.value(row))?,
                retrieval_scores: serde_json::from_str(scores.value(row))?,
                executed_at,
                signature: signature.value(row).to_string(),
            });
        }
    }
    Ok(out)
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, AuditError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| AuditError::Storage(format!("column '{name}' is not a non-null Utf8 array")))
}

fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, AuditError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| AuditError::Storage(format!("column '{name}' is not an Int64 array")))
}
