//! `AuditService` proto↔domain conversions.
//!
//! The [`pb::PerQueryAudit`] message mirrors the engine's [`PerQueryAudit`]
//! field for field, including its server-computed `signature`. The encode
//! ([`From`]) carries the tenant and signature the engine populated on the
//! stored record; `executed_at` rides as epoch microseconds (the audit table's
//! storage form).
//!
//! The decode is not symmetric: on `AuditLog` the caller leaves `tenant_id`,
//! `signature`, and `executed_at` empty — the engine stamps the tenant, signs,
//! and timestamps on write — so the *receive-side* decode (which also maps the
//! engine's `AuditError` to a `Status` via the server's `map_audit_error`) stays
//! in the `jammi-server` handler. Shared here is the pure query-id parse used by
//! both the decode and the `fetch_by_query_id` verb.

use tonic::Status;
use uuid::Uuid;

use crate::wire::proto::audit as pb;
use crate::PerQueryAudit;

/// Parse a wire query-id string into a [`Uuid`]. A missing or malformed id is a
/// client error. Shared by the audit-log decode and `AuditFetchByQueryId`.
pub fn parse_query_id(id: &str) -> Result<Uuid, Status> {
    if id.is_empty() {
        return Err(Status::invalid_argument("query_id is required"));
    }
    Uuid::parse_str(id).map_err(|e| Status::invalid_argument(format!("invalid query_id: {e}")))
}

/// Encode an engine [`PerQueryAudit`] onto the wire message, carrying the tenant
/// and signature the engine populated on the stored record.
impl From<PerQueryAudit> for pb::PerQueryAudit {
    fn from(r: PerQueryAudit) -> Self {
        pb::PerQueryAudit {
            query_id: r.query_id.to_string(),
            tenant_id: r.tenant_id.unwrap_or_default(),
            model_id: r.model_id,
            model_version: r.model_version,
            query_lineage: r.query_lineage.to_string(),
            top_k_result_ids: r.top_k_result_ids,
            retrieval_scores: r.retrieval_scores,
            executed_at_micros: r.executed_at.timestamp_micros(),
            signature: r.signature,
        }
    }
}
