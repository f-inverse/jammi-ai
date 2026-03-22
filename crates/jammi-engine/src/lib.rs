//! Jammi Engine — query engine, configuration, catalog, and source management.
//!
//! This crate provides the foundational infrastructure for the Jammi AI platform:
//! data source registration, SQL query execution via DataFusion, SQLite-backed
//! artifact catalog, and configuration management.

pub mod catalog;
pub mod config;
pub mod error;
pub mod session;
pub mod source;

use config::LoggingConfig;

/// Initialize the tracing subscriber using the provided logging configuration.
pub fn init_tracing(config: &LoggingConfig) {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    match config.format.as_str() {
        "json" => {
            fmt().with_env_filter(filter).json().init();
        }
        _ => {
            fmt().with_env_filter(filter).init();
        }
    }
}
