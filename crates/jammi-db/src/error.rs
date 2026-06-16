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

    /// Model lifecycle error, scoped to a specific model. A genuine bad-argument
    /// fault (e.g. an invalid version) — distinct from [`Self::ModelNotFound`],
    /// which an absent row raises. Maps to gRPC `InvalidArgument`.
    #[error("Model error: {model_id}: {message}")]
    Model {
        /// Identifier of the failing model.
        model_id: String,
        /// Human-readable error description.
        message: String,
    },

    /// A lifecycle verb (retire / delete / promote) resolved no model row for
    /// the caller's tenant — the model does not exist, or exists only outside
    /// the caller's scope. An absent row is a NotFound, not a bad argument, so
    /// it maps to gRPC `NotFound` rather than `InvalidArgument`.
    #[error("Model not found: {model_id}")]
    ModelNotFound {
        /// Identifier of the model that resolved no row.
        model_id: String,
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

    /// A `delete_model` was refused because the model is still the target of one
    /// or more references. Deleting it would orphan those edges, so this is a
    /// precondition failure, not a bad argument — it maps to gRPC
    /// `FailedPrecondition`. `referenced_by` names the blocking edges as generic
    /// catalog edge names (e.g. `result_tables`, `training_jobs.output_model_id`).
    #[error("Model referenced: {model_id}: still referenced by {}", referenced_by.join(", "))]
    ModelReferenced {
        /// Identifier of the referenced model.
        model_id: String,
        /// Generic names of the catalog edges still pointing at the model.
        referenced_by: Vec<String>,
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

    /// A channel-catalog operation (register a channel, append columns) failed
    /// with a caller-facing condition the gRPC surface must distinguish
    /// (already-exists / not-registered / column conflict / invalid input).
    #[error("Channel catalog error: {0}")]
    ChannelCatalog(#[from] crate::catalog::channel_repo::ChannelCatalogError),

    /// Channel-assembly runtime failure: a data-shape contract violation while
    /// merging channel contributions into a result batch. These are reached only
    /// from the engine-internal search-merge path on engine-derived inputs, so
    /// they are engine-invariant failures, not caller conditions.
    #[error("Channel assembly error: {0}")]
    ChannelAssembly(String),

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
