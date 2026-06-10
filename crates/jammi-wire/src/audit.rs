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

use jammi_db::AuditError;

use crate::proto::audit as pb;
use jammi_db::PerQueryAudit;

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

/// Reconstruct a stored [`PerQueryAudit`] from the wire message a fetch verb
/// returns — the inverse of the encode above, for the the remote client
/// read side.
///
/// Unlike the receive-side `record_from_proto` in `jammi-server` (which rebuilds
/// an *unsigned* record on `AuditLog` and lets the engine stamp tenant /
/// signature / timestamp), this decodes a record the engine has already stamped:
/// every field — including `tenant_id`, `signature`, and `executed_at` — is
/// carried verbatim so the remote read is byte-for-byte the local read and the
/// caller's signature `verify()` covers the identical canonical bytes. An empty
/// `tenant_id` decodes to `None`; the decode is fallible (uuid parse, lineage
/// JSON, timestamp range) and surfaces corruption as [`AuditError::Storage`]
/// rather than fabricating a value.
pub fn record_from_wire(p: pb::PerQueryAudit) -> Result<PerQueryAudit, AuditError> {
    let query_id = Uuid::parse_str(&p.query_id)
        .map_err(|e| AuditError::Storage(format!("invalid query_id from server: {e}")))?;
    let query_lineage: serde_json::Value = if p.query_lineage.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(&p.query_lineage)?
    };
    let executed_at = chrono::DateTime::from_timestamp_micros(p.executed_at_micros)
        .ok_or_else(|| AuditError::Storage("executed_at_micros out of range".into()))?;
    let tenant_id = if p.tenant_id.is_empty() {
        None
    } else {
        Some(p.tenant_id)
    };
    Ok(PerQueryAudit {
        query_id,
        tenant_id,
        model_id: p.model_id,
        model_version: p.model_version,
        query_lineage,
        top_k_result_ids: p.top_k_result_ids,
        retrieval_scores: p.retrieval_scores,
        executed_at,
        signature: p.signature,
    })
}
