//! Error type for the per-query audit primitive.

use uuid::Uuid;

/// Errors surfaced by the audit module.
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    /// `top_k_result_ids` and `retrieval_scores` had different lengths.
    #[error("top_k_result_ids ({ids}) and retrieval_scores ({scores}) length mismatch")]
    LengthMismatch { ids: usize, scores: usize },

    /// `query_lineage` JSON exceeded the configured size cap.
    #[error(
        "query_lineage exceeds maximum size ({actual} bytes > {max} bytes); \
         store hashes/IDs only, not raw payloads"
    )]
    LineageTooLarge { actual: usize, max: usize },

    /// The session had no tenant binding when one was required.
    #[error("no tenant binding on the current session — call with_tenant() first")]
    NoTenantBinding,

    /// A record's signature did not match the expected HMAC.
    #[error("signature verification failed for query_id {0}")]
    SignatureMismatch(Uuid),

    /// `JAMMI_AUDIT_MASTER_KEY` was unset or invalid.
    #[error(
        "audit master key not configured or invalid — set JAMMI_AUDIT_MASTER_KEY \
         to 32 bytes of hex (64 hex chars): {0}"
    )]
    MasterKey(String),

    /// JSON (de)serialization failure.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    /// Underlying storage failure (mutable-table registry / backend).
    #[error("storage: {0}")]
    Storage(String),

    /// Trigger broker failure (topic registration or publish).
    #[error("trigger broker: {0}")]
    Broker(String),
}
