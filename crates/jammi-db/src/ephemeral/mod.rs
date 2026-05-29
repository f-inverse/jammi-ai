//! Session-scoped ephemeral storage.
//!
//! An [`EphemeralSession`] is a tenant-scoped context whose mutable tables are
//! auto-deleted when the session ends — on explicit [`EphemeralSession::close`],
//! on [`Drop`] (best-effort), or when the timeout scanner force-closes it.
//! Every transition publishes a [`SessionLifecycleRecord`] to the
//! `jammi.audit.session_lifecycle.v1` trigger topic, giving downstream audit-log
//! aggregators a durable, subscribable proof of deletion.
//!
//! It composes the existing substrate primitives directly: mutable tables
//! (session-prefixed storage), tenant scope (tables are created and read under
//! the session's bound tenant), the trigger stream (lifecycle publication), and
//! the catalog (table registration).
//!
//! # Invariants
//!
//! - A session is always tenant-scoped: [`EphemeralSession::open`] refuses to
//!   open without a bound tenant.
//! - Physical table ids are `__eph_<session-uuid>_<name>`, so a session's tables
//!   are namespaced to it and never collide with another session's.
//! - `close` is the safe path and reports deletion outcomes; `Drop` is
//!   best-effort and only logs.
//! - The timeout scanner and an explicit close coordinate through
//!   [`ActiveSessions`]: whichever removes the session's snapshot first owns the
//!   deletion, so tables are never double-dropped.

mod error;
mod event;
mod scanner;
mod session;
mod topic;

pub use error::EphemeralError;
pub use event::{SessionLifecycleEvent, SessionLifecycleRecord, SESSION_LIFECYCLE_TOPIC};
pub use scanner::{scan, spawn as spawn_timeout_scanner, ActiveSessions, DEFAULT_SCAN_INTERVAL};
pub use session::{EphemeralSession, MAX_LOGICAL_NAME_LEN};
