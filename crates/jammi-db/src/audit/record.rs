//! Per-query audit record — typed, canonically serializable, HMAC-signed.
//!
//! The record is the substrate's standardized answer to "record what was
//! queried, with what model, what came back, and when, in a way that's signed
//! and queryable later." Tenant-defined detail lives in `query_lineage`; the
//! substrate enforces only the invariants every audited-ML tenant needs:
//! length agreement between result ids and scores, a lineage size cap, and a
//! deterministic signature.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::AuditError;

/// A single per-query audit record.
///
/// `tenant_id` and `signature` are populated by the substrate on write
/// (`audit::log::log_records`); callers construct records via
/// [`PerQueryAudit::new`] and leave both empty.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerQueryAudit {
    /// Unique identifier for this search invocation.
    pub query_id: Uuid,

    /// Tenant scope. Auto-injected from the session's tenant binding on write;
    /// users do not supply it. Present on reads.
    ///
    /// Stored as the tenant's canonical string form; the substrate converts
    /// to/from `jammi_db::TenantId` at the boundary.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tenant_id: Option<String>,

    /// Model identifier (e.g. a HuggingFace repo id).
    pub model_id: String,

    /// Specific version/revision of the model (commit SHA, HF revision, etc.).
    pub model_version: String,

    /// Open-ended JSON capturing tenant-specific query metadata: image hashes
    /// (NOT raw bytes), examiner id, ROI bbox, classification scope, etc. The
    /// schema is tenant-defined; the substrate does not interpret it.
    pub query_lineage: serde_json::Value,

    /// Top-K result identifiers (whatever stable primary key the tenant uses).
    pub top_k_result_ids: Vec<String>,

    /// Distance/similarity scores, parallel to `top_k_result_ids`.
    pub retrieval_scores: Vec<f32>,

    /// When the search was executed (server-side timestamp; not user-supplied).
    pub executed_at: DateTime<Utc>,

    /// HMAC-SHA256 signature over the canonical serialization. Computed
    /// server-side by `audit::log::log_records`; never user-supplied.
    #[serde(default)]
    pub signature: String,
}

impl PerQueryAudit {
    /// Construct a new, unsigned record with a server-side timestamp.
    ///
    /// Returns [`AuditError::LengthMismatch`] if `top_k_result_ids` and
    /// `retrieval_scores` differ in length.
    pub fn new(
        query_id: Uuid,
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        query_lineage: serde_json::Value,
        top_k_result_ids: Vec<String>,
        retrieval_scores: Vec<f32>,
    ) -> Result<Self, AuditError> {
        if top_k_result_ids.len() != retrieval_scores.len() {
            return Err(AuditError::LengthMismatch {
                ids: top_k_result_ids.len(),
                scores: retrieval_scores.len(),
            });
        }
        Ok(Self {
            query_id,
            tenant_id: None,
            model_id: model_id.into(),
            model_version: model_version.into(),
            query_lineage,
            top_k_result_ids,
            retrieval_scores,
            executed_at: Utc::now(),
            signature: String::new(),
        })
    }

    /// The `executed_at` instant as epoch microseconds — the storage form used
    /// by the audit table's `executed_at` column (matching the trigger backing
    /// table's `Int64`-microsecond convention so both backends round-trip
    /// identically).
    pub fn executed_at_micros(&self) -> i64 {
        self.executed_at.timestamp_micros()
    }
}

/// Canonical byte serialization used as the HMAC input.
///
/// The form is deterministic and stable across restarts and platforms:
///
/// - fields appear in a fixed order: `query_id`, `tenant_id`, `model_id`,
///   `model_version`, `query_lineage`, `top_k_result_ids`, `retrieval_scores`,
///   `executed_at`;
/// - JSON object keys are sorted recursively (so `query_lineage` canonicalizes
///   regardless of how the tenant ordered its keys);
/// - no insignificant whitespace;
/// - `executed_at` is RFC3339 in UTC with second precision;
/// - the `signature` field is excluded.
///
/// `tenant_id` MUST be set before signing; signing an unscoped record is a
/// programming error and yields [`AuditError::NoTenantBinding`].
pub fn canonical_serialize(record: &PerQueryAudit) -> Result<Vec<u8>, AuditError> {
    let tenant = record
        .tenant_id
        .as_deref()
        .ok_or(AuditError::NoTenantBinding)?;

    // Emit fields one at a time so order is guaranteed without relying on a
    // serde_json feature flag.
    let mut out = Vec::new();
    out.push(b'{');
    write_field(
        &mut out,
        "query_id",
        &serde_json::json!(record.query_id),
        true,
    )?;
    write_field(&mut out, "tenant_id", &serde_json::json!(tenant), false)?;
    write_field(
        &mut out,
        "model_id",
        &serde_json::json!(record.model_id),
        false,
    )?;
    write_field(
        &mut out,
        "model_version",
        &serde_json::json!(record.model_version),
        false,
    )?;
    write_field(
        &mut out,
        "query_lineage",
        &canonicalize_value(&record.query_lineage),
        false,
    )?;
    write_field(
        &mut out,
        "top_k_result_ids",
        &serde_json::json!(record.top_k_result_ids),
        false,
    )?;
    write_field(
        &mut out,
        "retrieval_scores",
        &serde_json::json!(record.retrieval_scores),
        false,
    )?;
    let ts = record
        .executed_at
        .to_rfc3339_opts(SecondsFormat::Secs, true);
    write_field(&mut out, "executed_at", &serde_json::json!(ts), false)?;
    out.push(b'}');
    Ok(out)
}

fn write_field(
    out: &mut Vec<u8>,
    key: &str,
    value: &serde_json::Value,
    first: bool,
) -> Result<(), AuditError> {
    if !first {
        out.push(b',');
    }
    out.extend_from_slice(serde_json::to_string(key)?.as_bytes());
    out.push(b':');
    out.extend_from_slice(serde_json::to_vec(value)?.as_slice());
    Ok(())
}

/// Recursively sort object keys so equal JSON values serialize identically.
fn canonicalize_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut sorted = serde_json::Map::new();
            for (k, v) in entries {
                sorted.insert(k.clone(), canonicalize_value(v));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_value).collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scoped(lineage: serde_json::Value) -> PerQueryAudit {
        let mut r = PerQueryAudit::new(
            Uuid::nil(),
            "m",
            "v",
            lineage,
            vec!["a".into(), "b".into()],
            vec![0.1, 0.2],
        )
        .unwrap();
        r.tenant_id = Some("t1".into());
        r
    }

    #[test]
    fn new_rejects_length_mismatch() {
        let err = PerQueryAudit::new(
            Uuid::nil(),
            "m",
            "v",
            serde_json::json!({}),
            vec!["a".into()],
            vec![0.1, 0.2],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AuditError::LengthMismatch { ids: 1, scores: 2 }
        ));
    }

    #[test]
    fn canonical_requires_tenant() {
        let r = PerQueryAudit::new(Uuid::nil(), "m", "v", serde_json::json!({}), vec![], vec![])
            .unwrap();
        assert!(matches!(
            canonical_serialize(&r),
            Err(AuditError::NoTenantBinding)
        ));
    }

    #[test]
    fn canonical_is_key_order_independent() {
        let a = scoped(serde_json::json!({ "examiner_id": "42", "image_hashes": ["x"] }));
        let b = scoped(serde_json::json!({ "image_hashes": ["x"], "examiner_id": "42" }));
        assert_eq!(
            canonical_serialize(&a).unwrap(),
            canonical_serialize(&b).unwrap(),
            "canonical form must be independent of lineage key insertion order"
        );
    }

    #[test]
    fn canonical_is_deterministic() {
        let r = scoped(serde_json::json!({ "k": [1, 2, 3], "nested": { "z": 1, "a": 2 } }));
        assert_eq!(
            canonical_serialize(&r).unwrap(),
            canonical_serialize(&r).unwrap()
        );
    }

    #[test]
    fn canonical_changes_when_a_field_changes() {
        let a = scoped(serde_json::json!({ "k": 1 }));
        let mut b = a.clone();
        b.model_id = "different".into();
        assert_ne!(
            canonical_serialize(&a).unwrap(),
            canonical_serialize(&b).unwrap()
        );
    }
}
