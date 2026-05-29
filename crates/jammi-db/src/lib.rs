//! Jammi DB — vector database, SQL federation, mutable companion tables,
//! and trigger broker for the Jammi AI platform.
//!
//! Provides the foundational infrastructure: data source registration,
//! SQL query execution via DataFusion, catalog persistence (SQLite or
//! Postgres), mutable companion tables with crash-safe WAL, a trigger
//! broker for provenance channels, and configuration management.

pub mod audit;
pub mod cache;
pub mod catalog;
pub mod config;
pub mod ephemeral;
pub mod error;
pub mod evidence_channel;
pub mod index;
pub mod model_task;
pub mod session;
pub mod source;
pub mod sql;
pub mod storage;
pub mod store;
pub mod tenant;
pub mod tenant_scope;
pub mod trigger;

use config::{LogFormat, LoggingConfig};

pub use audit::{AuditError, AuditHandle, PerQueryAudit};
pub use catalog::backend::{
    BackendError, BackendImpl, BackendKind, CatalogBackend, IsolationLevel, Transaction, TxOptions,
};
pub use ephemeral::{
    ActiveSessions, EphemeralError, EphemeralSession, SessionLifecycleEvent,
    SessionLifecycleRecord, SESSION_LIFECYCLE_TOPIC,
};
pub use evidence_channel::ChannelId;
pub use model_task::ModelTask;
pub use session::{AdminScope, TenantScope};
pub use tenant::TenantId;
pub use trigger::TopicId;

/// Initialize the tracing subscriber using the provided logging configuration.
pub fn init_tracing(config: &LoggingConfig) {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    match config.format {
        LogFormat::Json => {
            fmt().with_env_filter(filter).json().init();
        }
        LogFormat::Text => {
            fmt().with_env_filter(filter).init();
        }
    }
}
