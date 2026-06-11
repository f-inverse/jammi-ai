use thiserror::Error;

/// Unified error type for all Jammi DB operations.
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

    /// The serve/load path was asked for a model that has been retired. A
    /// retired model is still resolvable as a reference target (provenance,
    /// FK), but it cannot be loaded or served — this is a precondition failure,
    /// not a bad argument, so it maps to gRPC `FailedPrecondition`.
    #[error("Model retired: {model_id}")]
    ModelRetired {
        /// Identifier of the retired model.
        model_id: String,
    },

    /// Inference execution failure.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Fine-tuning error.
    #[error("Fine-tune error: {0}")]
    FineTune(String),

    /// Evaluation error.
    #[error("Eval error: {0}")]
    Eval(String),

    /// GPU scheduling or detection error.
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Remote backend error (vLLM, HTTP).
    #[error("Backend error: {0}")]
    Backend(String),

    /// Filesystem I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Catalog backend (SQLite / Postgres) error.
    #[error("Backend error: {0}")]
    BackendDriver(#[from] crate::catalog::backend::BackendError),

    /// Invalid tenant identifier (e.g., nil UUID, malformed string).
    #[error("Tenant error: {0}")]
    Tenant(String),

    /// TOML configuration parse error.
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// DataFusion query-engine error.
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    /// Evidence channel validation or registration error.
    #[error("Evidence channel error: {0}")]
    EvidenceChannel(String),

    /// Lexical (BM25 / tantivy) sidecar build, persistence, or query failure.
    #[error("Lexical retrieval error: {0}")]
    Lexical(String),

    /// Mutable companion table error.
    #[error("Mutable table error: {0}")]
    MutableTable(#[from] crate::store::mutable::MutableTableError),

    /// Trigger-stream error (topic registration, publish, subscribe).
    #[error("Trigger error: {0}")]
    Trigger(#[from] crate::trigger::TriggerError),

    /// Object-store / storage-layer failure (URL parse, driver init,
    /// remote I/O, on-the-wire layout corruption).
    #[error("Storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),

    /// A typed read from a Parquet table found a column whose Arrow type
    /// disagrees with what the caller asked for (missing column, wrong
    /// `DataType`, wrong inner type on a list).
    #[error("Schema error: table {table:?} column {column:?}: expected {expected}, got {actual}")]
    Schema {
        /// Table the read targeted (typically `ResultTableRecord::table_name`).
        table: String,
        /// Name of the column the read targeted.
        column: String,
        /// Expected Arrow shape (e.g. `"FixedSizeList<Float32>"`).
        expected: String,
        /// What the on-disk schema actually carried (or `"missing"` if the
        /// column wasn't present at all).
        actual: String,
    },

    /// Catch-all for errors that don't fit another variant.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias for `std::result::Result<T, JammiError>`.
pub type Result<T> = std::result::Result<T, JammiError>;
