//! Error type for the ephemeral-session primitive.

use uuid::Uuid;

/// Errors surfaced by the ephemeral-session module.
#[derive(Debug, thiserror::Error)]
pub enum EphemeralError {
    /// The session had no tenant binding when one was required.
    ///
    /// Ephemeral sessions are always tenant-scoped: opening one without a bound
    /// tenant would create session-private tables nobody could read back under
    /// scope, so the open is refused up front.
    #[error("no tenant binding on the current session — bind a tenant first")]
    NoTenantBinding,

    /// The user-supplied logical table name does not fit the ephemeral
    /// name budget once the session prefix is applied.
    ///
    /// The physical table id is `__eph_<32-hex-session>_<name>`, capped at the
    /// mutable-table id limit of 63 characters; this leaves 24 characters for
    /// the logical name.
    #[error(
        "ephemeral table name '{name}' is too long: logical names are limited to \
         {max} characters so the session-prefixed id fits the 63-char table-id cap"
    )]
    NameTooLong { name: String, max: usize },

    /// An ephemeral table with this logical name was already created in the
    /// session. Logical names are unique within a session.
    #[error("ephemeral table '{0}' already exists in this session")]
    DuplicateTable(String),

    /// A logical table name was referenced that the session never created.
    #[error("ephemeral table '{0}' not found in this session")]
    UnknownTable(String),

    /// One or more ephemeral tables failed to drop on close. The session has
    /// emitted a `partial_deletion_failure` lifecycle event recording which
    /// physical tables remain; the caller may retry close.
    #[error(
        "ephemeral session {session_id}: {failed}/{total} table drops failed on close \
         (a partial_deletion_failure event was emitted)"
    )]
    PartialDeletionFailure {
        session_id: Uuid,
        failed: usize,
        total: usize,
    },

    /// Underlying mutable-table failure (create, insert, drop).
    #[error("storage: {0}")]
    Storage(String),

    /// Trigger broker failure (lifecycle-topic registration or publish).
    #[error("trigger broker: {0}")]
    Broker(String),

    /// JSON (de)serialization failure building a lifecycle payload.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}
