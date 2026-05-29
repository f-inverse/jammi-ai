//! Session-lifecycle events published to `jammi.audit.session_lifecycle.v1`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The lifecycle topic every session transition is published to. Registered
/// (idempotently, per tenant) through the catalog `TopicRepo` on first publish,
/// mirroring the per-query audit topic.
pub const SESSION_LIFECYCLE_TOPIC: &str = "jammi.audit.session_lifecycle.v1";

/// Which lifecycle transition occurred. Serializes to the lowercase
/// snake-case discriminator carried in the topic payload's `event` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionLifecycleEvent {
    /// The session was opened.
    Opened,
    /// The session was explicitly closed (or dropped) and its tables deleted.
    Closed,
    /// The timeout scanner force-closed a session past its deadline.
    TimedOut,
    /// One or more table drops failed during close; `details` lists the
    /// physical tables that could not be deleted.
    PartialDeletionFailure,
}

/// One published session-lifecycle record.
///
/// `ephemeral_table_count` is the number of tables the session had created at
/// the moment of the transition; `deleted_row_count` is the total rows removed
/// by the deletion the event accompanies (`0` for `opened`). `details` carries
/// event-specific context (e.g. the surviving physical table names on a
/// partial-deletion failure) and is `null` when there is nothing extra to say.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionLifecycleRecord {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub event: SessionLifecycleEvent,
    pub occurred_at: DateTime<Utc>,
    pub ephemeral_table_count: usize,
    pub deleted_row_count: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub details: Option<serde_json::Value>,
}
