//! Jammi Engine — query engine, configuration, catalog, and source management.
//!
//! This crate provides the foundational infrastructure for the Jammi AI platform:
//! data source registration, SQL query execution via DataFusion, SQLite-backed
//! artifact catalog, and configuration management.

pub mod cache;
pub mod catalog;
pub mod config;
pub mod error;
pub mod evidence_channel;
pub mod index;
pub mod session;
pub mod source;
pub mod sql;
pub mod storage;
pub mod store;
pub mod tenant;
pub mod tenant_scope;
pub mod trigger;

use config::{LogFormat, LoggingConfig};

pub use catalog::backend::{
    BackendError, BackendImpl, BackendKind, CatalogBackend, IsolationLevel, Transaction, TxOptions,
};
pub use evidence_channel::ChannelId;
pub use session::{AdminScope, TenantScope};
pub use tenant::TenantId;

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
