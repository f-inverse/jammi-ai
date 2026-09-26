//! Jammi DB — vector database, SQL federation, mutable companion tables,
//! and trigger broker for the Jammi AI engine.
//!
//! Provides the foundational infrastructure: data source registration,
//! SQL query execution via DataFusion, catalog persistence (SQLite or
//! Postgres), mutable companion tables with crash-safe WAL, a trigger
//! broker for provenance channels, and configuration management.

pub mod audit;
pub mod cache;
pub mod catalog;
pub mod compute_plane;
pub mod config;
pub mod ephemeral;
pub mod error;
pub mod evidence_channel;
pub mod index;
pub mod memory_pool;
pub mod server_info;
pub mod session;
pub mod source;
pub mod sql;
pub mod storage;
pub mod store;
pub mod tenant;
pub mod tenant_scope;
pub mod trigger;

pub use audit::{AuditError, AuditHandle, PerQueryAudit};
pub use catalog::backend::{BackendError, BackendKind};
#[cfg(feature = "test-hooks")]
pub use catalog::backend::{BackendImpl, CatalogBackend, IsolationLevel, Transaction, TxOptions};
pub use ephemeral::{
    ActiveSessions, EphemeralError, EphemeralSession, SessionLifecycleEvent,
    SessionLifecycleRecord, SESSION_LIFECYCLE_TOPIC,
};
pub use evidence_channel::ChannelId;
pub use server_info::ServerInfo;
pub use session::{AdminScope, TenantScope};
pub use tenant::TenantId;
pub use trigger::TopicId;
