use thiserror::Error;

/// Unified error type for all Jammi Engine operations.
#[derive(Debug, Error)]
pub enum JammiError {
    /// Invalid or missing configuration value.
    #[error("Configuration error: {0}")]
    Config(String),

    /// SQLite catalog read/write failure.
    #[error("Catalog error: {0}")]
    Catalog(String),

    /// Data source operation failure, scoped to a specific source.
    #[error("Source error: {source_id}: {message}")]
    Source {
        /// Identifier of the failing source.
        source_id: String,
        /// Human-readable error description.
        message: String,
    },

    /// Model lifecycle error, scoped to a specific model.
    #[error("Model error: {model_id}: {message}")]
    Model {
        /// Identifier of the failing model.
        model_id: String,
        /// Human-readable error description.
        message: String,
    },

    /// Inference execution failure.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Filesystem I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite driver error.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// TOML configuration parse error.
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// DataFusion query-engine error.
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    /// Connection pool error.
    #[error("Pool error: {0}")]
    Pool(#[from] r2d2::Error),

    /// Schema migration error.
    #[error("Migration error: {0}")]
    Migration(#[from] rusqlite_migration::Error),

    /// Catch-all for errors that don't fit another variant.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias for `std::result::Result<T, JammiError>`.
pub type Result<T> = std::result::Result<T, JammiError>;
