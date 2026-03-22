use thiserror::Error;

#[derive(Debug, Error)]
pub enum JammiError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Catalog error: {0}")]
    Catalog(String),

    #[error("Source error: {source_id}: {message}")]
    Source { source_id: String, message: String },

    #[error("Model error: {model_id}: {message}")]
    Model { model_id: String, message: String },

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    #[error("Pool error: {0}")]
    Pool(#[from] r2d2::Error),

    #[error("Migration error: {0}")]
    Migration(#[from] rusqlite_migration::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, JammiError>;
