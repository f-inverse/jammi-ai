//! Per-query audit record primitive.
//!
//! A standardized, tenant-scoped, HMAC-signed record of *what was queried, with
//! what model, what came back, and when*. Every audited-ML tenant needs this;
//! the substrate provides it so each tenant does not hand-roll an incompatible
//! audit schema, signature scheme, and trigger integration.
//!
//! It composes the four substrate primitives directly: mutable tables (storage
//! in `_jammi_search_audit`), tenant scope (auto-injected `tenant_id`, scoped
//! reads), the trigger stream (publication to `jammi.audit.search.v1`), and the
//! catalog (table registration).
//!
//! # Invariants
//!
//! - `top_k_result_ids.len() == retrieval_scores.len()` (checked at construction).
//! - `query_lineage` JSON is capped (default 8 KiB, see
//!   [`log::max_lineage_bytes`]) — store hashes/ids, not raw payloads.
//! - Records are signed with a per-tenant HMAC-SHA256 secret derived from
//!   `JAMMI_AUDIT_MASTER_KEY`; signatures verify deterministically across
//!   restarts.
//! - The backing table `_jammi_search_audit` is reserved; users cannot create
//!   or directly write it.

mod error;
mod log;
mod query;
mod record;
mod signature;
mod table;

pub use error::AuditError;
pub use log::{log_records, max_lineage_bytes, DEFAULT_MAX_LINEAGE_BYTES, MAX_LINEAGE_BYTES_ENV};
pub use query::{fetch_by_query_id, fetch_recent};
pub use record::{canonical_serialize, PerQueryAudit};
pub use signature::{
    derive_tenant_secret, ensure_master_key_present, hmac_sign, master_key_from_env, sign_record,
    verify, verify_with_env, MASTER_KEY_ENV,
};
pub use table::{
    audit_schema, ensure_table_exists, is_reserved_table_name, AUDIT_TABLE_NAME, AUDIT_TOPIC,
};

use crate::session::JammiSession;
use uuid::Uuid;

/// Typed audit handle bound to a session, returned by [`JammiSession::audit`].
///
/// Thin ergonomic wrapper over the free functions in this module; it borrows
/// the session it was created from.
pub struct AuditHandle<'a> {
    session: &'a JammiSession,
}

impl<'a> AuditHandle<'a> {
    /// Create a handle for `session`.
    pub fn new(session: &'a JammiSession) -> Self {
        Self { session }
    }

    /// Sign and persist a batch of records; publishes them to the audit topic.
    pub async fn log(&self, records: Vec<PerQueryAudit>) -> Result<(), AuditError> {
        log_records(self.session, records).await
    }

    /// Fetch one record by query id (tenant-scoped).
    pub async fn fetch_by_query_id(
        &self,
        query_id: Uuid,
    ) -> Result<Option<PerQueryAudit>, AuditError> {
        fetch_by_query_id(self.session, query_id).await
    }

    /// Fetch the most recent records (tenant-scoped), newest first.
    pub async fn fetch_recent(&self, limit: usize) -> Result<Vec<PerQueryAudit>, AuditError> {
        fetch_recent(self.session, limit).await
    }
}
