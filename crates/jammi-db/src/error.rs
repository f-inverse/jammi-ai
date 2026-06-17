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

    /// A `delete_model` call resolved no model row for the caller's tenant — the
    /// model does not exist, or exists only outside the caller's scope. An absent
    /// row is a NotFound, not a bad argument, so it maps to gRPC `NotFound` rather
    /// than `InvalidArgument`.
    #[error("Model not found: {model_id}")]
    ModelNotFound {
        /// Identifier of the model that resolved no row.
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

    /// A persisted artifact carries a format stamp this build cannot read: the
    /// on-disk format is newer than supported (for stamps with a compatibility
    /// ordering) or simply incompatible (for backend stamps with none). One
    /// variant serves every stamped sidecar format — the `.rowmap`, the ANN
    /// `.manifest.json` version, and the USearch graph's `backend_version` — so
    /// the load paths reject an unreadable artifact as a typed error rather than
    /// risk a silent misparse. The upgrade path is to re-emit; there is no
    /// back-compat reader.
    #[error("incompatible {artifact} format: found {found}, this build supports {supported}")]
    IncompatibleFormat {
        /// The artifact whose stamp was rejected (e.g. `"rowmap"`,
        /// `"ann-manifest"`, `"usearch-index"`).
        artifact: String,
        /// The format stamp found on disk.
        found: String,
        /// What this build supports — a version for ordered stamps, the current
        /// backend version for the strict USearch check.
        supported: String,
    },

    /// A derives-from lineage walk revisited a table it was already descending
    /// through — the reverse-dependency edges form a cycle, so a transitive walk
    /// has no well-founded termination. A materialization lineage is a DAG by
    /// construction (a producer's inputs are anchored before its output exists),
    /// so a cycle is a corruption of the recorded `input_anchors_json`, not a
    /// caller condition. Carries the table at which the back-edge was detected.
    #[error("dependency cycle in derives-from lineage at table `{table}`")]
    DependencyCycle {
        /// The table whose re-entry closed the cycle.
        table: String,
    },

    /// A `recompute` was asked to re-produce a table that carries no
    /// materialization contract — its catalog `definition_hash IS NULL`, so it is
    /// a **pre-contract** table created before the materialization contract
    /// landed. Without a recorded [`ProducingDescriptor`](crate::store::manifest::ProducingDescriptor)
    /// there is nothing to dispatch a faithful replay on, and guessing a producer
    /// call from a table's columns would be a fabricated re-run — so this is a
    /// loud typed refusal, never a silent best-effort. Carries the table named.
    #[error("table `{table}` carries no materialization contract and cannot be recomputed")]
    NotRecomputable {
        /// The table that has no recorded producing descriptor to replay.
        table: String,
    },

    /// Catch-all for errors that don't fit another variant.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias for `std::result::Result<T, JammiError>`.
pub type Result<T> = std::result::Result<T, JammiError>;
