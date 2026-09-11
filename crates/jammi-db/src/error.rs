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
    /// catalog edge names (e.g. `result_tables`, `jobs.output_model_id`).
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
    ///
    /// `#[source]` (not `#[from]`): the conversion from a
    /// [`DataFusionError`](datafusion::error::DataFusionError) is the manual
    /// `impl From` at the bottom of this file — the structural classifier that
    /// restores a typed engine error a plan node raised (`External(Box<JammiError>)`,
    /// even nested under `Context`/`ArrowError`/`ParquetError`) and types a
    /// mid-scan object-store not-found as [`Self::Storage`]. Every `?` that
    /// converts a DataFusion error routes through it by construction. The
    /// attribute keeps `source()` intact (thiserror's `#[from]` implied it).
    #[error("DataFusion error: {0}")]
    DataFusion(#[source] datafusion::error::DataFusionError),

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

    /// A `recompute` was asked to re-produce a table the engine has no faithful
    /// replay for. Two tables land here, and both are a loud typed refusal rather
    /// than a silent best-effort:
    ///
    /// - a **pre-contract** table whose catalog `definition_hash IS NULL` — created
    ///   before the materialization contract landed, so there is no recorded
    ///   [`ProducingDescriptor`](crate::store::manifest::ProducingDescriptor) to
    ///   dispatch a replay on at all; and
    /// - a table produced by an [`External`](crate::store::manifest::ProducingDescriptor::External)
    ///   producer — a verb the engine does not own — which the engine cannot
    ///   reconstruct even though a descriptor is recorded.
    ///
    /// In either case guessing a producer call would be a fabricated re-run, so
    /// the engine refuses loudly. Carries the table named.
    #[error("table `{table}` has no engine-recomputable producer and cannot be recomputed")]
    NotRecomputable {
        /// The table with no engine-recomputable producer to replay.
        table: String,
    },

    /// A building-row compare-and-set matched no row because the row is gone:
    /// the `result_tables` row the writer (or the reaper) addressed no longer
    /// exists — deleted underneath it (a source removal, a manual purge).
    /// Nothing is deleted on this outcome; the caller stops.
    #[error("result table `{table}` is gone: no catalog row to transition")]
    RowGone {
        /// The table whose row is absent.
        table: String,
    },

    /// A building-row compare-and-set matched no row because the row belongs
    /// to a different tenant than the binding in force (only possible under a
    /// non-admin binding — the STRICT tenant arm refused the write). Never a
    /// licence to delete anything.
    #[error("result table `{table}` belongs to another tenant; the transition was refused")]
    TenantMismatch {
        /// The table whose row is owned elsewhere.
        table: String,
    },

    /// A building-row compare-and-set matched no row because the row is still
    /// `building` but its `writer_id` is no longer the caller's: the lease
    /// expired and recovery claimed the row, so the claimant now owns the row
    /// AND its bytes. The caller returns this error and deletes nothing.
    #[error("result table `{table}`: lease lost to another writer")]
    LeaseLost {
        /// The table whose lease was lost.
        table: String,
    },

    /// A building-row compare-and-set matched no row because the row has
    /// already left `building` — `status` names where it went (`ready` when
    /// recovery promoted an expired-lease row whose sidecar had landed;
    /// `failed` when it was reaped). The caller returns this error, never
    /// re-promotes, and deletes nothing.
    #[error("result table `{table}` is already `{status}`; the transition was superseded")]
    CasFailed {
        /// The table whose row moved on.
        table: String,
        /// The row's current status.
        status: String,
    },

    /// A `create_result_table` call's `jobs.partial_result` compare-and-set
    /// matched zero rows: the job is either not `running` (a peer
    /// reclaimed it, the caller's attempt has been superseded) or another
    /// attempt already recorded a `partial_result` for it first. Either way
    /// this attempt is not the one of record. The transaction this CAS ran
    /// inside is rolled back — no `result_tables` row and no bytes are ever
    /// committed for a superseded attempt.
    #[error("job `{job_id}`: this attempt has been superseded; no result table was created")]
    JobAttemptSuperseded {
        /// The job whose `partial_result` CAS missed.
        job_id: String,
    },

    /// A job's executor observed `jobs.cancel_requested` at a checkpoint
    /// boundary and stopped: the job is recorded `failed` with this message
    /// and no result is returned. Raised by `InferenceSession::run_now` for
    /// an inline job and by the worker's claimed-compute path; the request
    /// itself is `Catalog::cancel_request`.
    #[error("job `{job_id}`: cancelled at the executor's request checkpoint")]
    JobCancelled {
        /// The job whose cancel request was honoured.
        job_id: String,
    },

    /// A source cannot be deleted while a live writer is still materialising a
    /// result table over it: a `building` row with an unexpired lease
    /// references the source. Retry once the writer finishes or its lease
    /// expires. Maps to a precondition failure, not a bad argument.
    #[error(
        "source `{source_id}` is busy: result table `{table}` is being built under a live lease"
    )]
    SourceBusy {
        /// The source the delete targeted.
        source_id: String,
        /// The building table holding the live lease.
        table: String,
    },

    /// A `NULL` in the key column of a source scanned for embedding /
    /// inference / refresh: refused typed at the input edge (K2), never
    /// skipped and counted, never a stringly Arrow error after model calls.
    /// Raised by the `KeyCheckExec` plan node below the blocking sort, so the
    /// count is exact and the model is invoked zero times.
    #[error("key column `{column}` has {null_count} null value(s); every row needs a key")]
    InvalidKey {
        /// The source key column the caller named.
        column: String,
        /// The exact number of null keys in the scanned source.
        null_count: u64,
    },

    /// A versioned result table whose CURRENT version cannot be served: the
    /// version row is `failed`, or its `.version.json` manifest is
    /// definitively absent on an `exists()` probe. Raised by the ANN and SQL
    /// read paths alike (the SQL path through the placeholder provider and
    /// the structural error classifier). The table row itself is untouched;
    /// the remedy is `recompute` (a new table).
    #[error(
        "result table `{table}` version {version} is unavailable (its manifest cannot be resolved)"
    )]
    VersionUnavailable {
        /// The table whose current version is unavailable.
        table: String,
        /// The unavailable version.
        version: i64,
    },

    /// A refresh or compaction was asked of a table it cannot serve
    /// incrementally: not `ready`, not an embedding table, its current
    /// version row not `ready`, or its rows carry no `_content_hash` (a table
    /// produced before the hash column existed, or by a producer that writes
    /// none). The remedy is `recompute` once (a fresh table carries hashes).
    #[error("result table `{table}` is not refreshable: {reason}")]
    NotRefreshable {
        /// The table the verb targeted.
        table: String,
        /// Why.
        reason: NotRefreshableReason,
    },

    /// The definition a refresh would run under (the table's recorded
    /// embedding parameters over the model as loaded NOW, device included)
    /// no longer hashes to the table's recorded `definition_hash` — a model or
    /// environment change, which no per-row content hash can see. Refused
    /// before any version is allocated; the consumer runs `recompute`.
    #[error("result table `{table}`: definition drift (recorded {recorded}, current {current})")]
    DefinitionDrift {
        table: String,
        /// The table's recorded `definition_hash`.
        recorded: String,
        /// The definition hash the refresh computed.
        current: String,
    },

    /// A refresh found the same key more than once on a COMPLETE scan — of
    /// the source (`Source`) or of the parent version's current state
    /// (`Parent`, two physical rows under one `_row_id`). A delta over a
    /// non-unique key space is ambiguous, so it is refused before any new
    /// version is allocated; the initial embed still tolerates duplicates,
    /// and `recompute` once yields a table a refresh can proceed from.
    #[error(
        "result table `{table}`: {total} non-unique key(s) in the {scan} scan (first {}: {keys:?})",
        keys.len()
    )]
    NonUniqueKey {
        table: String,
        /// Which scan carried the duplicates.
        scan: NonUniqueScan,
        /// Up to ten offending keys with their exact counts.
        keys: Vec<(String, u64)>,
        /// The total number of non-unique keys.
        total: u64,
    },

    /// A resource the request needs could not be reached after the bounded
    /// failure ladder: a placed segment whose owner and retry candidate both
    /// failed and whose local load was not admitted (or not attempted). Names
    /// the resource (`segment {table}/{id}`) and the last failure's reason. A
    /// peer outage is visible — never masked by a silent full scan of a
    /// larger-than-memory table. Maps to gRPC `Unavailable`.
    #[error("unavailable: {resource}: {reason}")]
    Unavailable {
        /// The resource that could not be served.
        resource: String,
        /// Why the last rung of the ladder failed.
        reason: String,
    },

    /// Catch-all for errors that don't fit another variant.
    #[error("{0}")]
    Other(String),
}

/// Why a table is [`JammiError::NotRefreshable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotRefreshableReason {
    /// The rows carry no (or a NULL / malformed) `_content_hash`.
    MissingContentHash,
    /// The table's current version row is not `ready`.
    CurrentVersionUnavailable,
    /// The table row is not `ready`.
    NotReady,
    /// Not an embedding table produced by the embedding pipeline.
    NotEmbeddingTable,
}

impl std::fmt::Display for NotRefreshableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl NotRefreshableReason {
    /// The stable wire token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingContentHash => "missing_content_hash",
            Self::CurrentVersionUnavailable => "current_version_unavailable",
            Self::NotReady => "not_ready",
            Self::NotEmbeddingTable => "not_embedding_table",
        }
    }

    /// Parse the wire token; `None` for an unknown one.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "missing_content_hash" => Some(Self::MissingContentHash),
            "current_version_unavailable" => Some(Self::CurrentVersionUnavailable),
            "not_ready" => Some(Self::NotReady),
            "not_embedding_table" => Some(Self::NotEmbeddingTable),
            _ => None,
        }
    }
}

/// Which scan a [`JammiError::NonUniqueKey`] found its duplicates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonUniqueScan {
    /// The source scan.
    Source,
    /// The parent version's current-state scan.
    Parent,
}

impl std::fmt::Display for NonUniqueScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl NonUniqueScan {
    /// The stable wire token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Parent => "parent",
        }
    }

    /// Parse the wire token; `None` for an unknown one.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "source" => Some(Self::Source),
            "parent" => Some(Self::Parent),
            _ => None,
        }
    }
}

/// The structural classifier: how EVERY DataFusion error becomes a
/// [`JammiError`].
///
/// Two shapes, in order. **(a) owned passthrough** — an
/// `External(Box<JammiError>)` payload (a typed error a plan node, provider or
/// UDF raised) is destructured BY VALUE back into the inner `JammiError`, also
/// when nested under `Context`, `ArrowError(ExternalError)` or
/// `ParquetError(External)`; a miss rebuilds the original unchanged. **(b) a
/// borrowed `source()` walk** — an `object_store::Error::NotFound` found at any
/// depth (the parquet reader wraps it under `ParquetError::External`, which
/// neither a top-level `ObjectStore` arm nor DataFusion's `find_root` reaches)
/// becomes [`JammiError::Storage`] with the existing `StorageError::Io { source:
/// NotFound { .. } }` spelling every reader already matches. Everything else
/// keeps the shape [`JammiError::DataFusion`] with `source()` intact.
/// `Shared(Arc<_>)` cannot yield ownership and is a stated fidelity limit of
/// shape (a); shape (b) still walks it.
impl From<datafusion::error::DataFusionError> for JammiError {
    fn from(e: datafusion::error::DataFusionError) -> Self {
        match unwrap_jammi(e) {
            Ok(inner) => inner,
            Err(e) => match not_found_path(&e) {
                Some((path, original)) => JammiError::Storage(crate::storage::StorageError::Io {
                    path: path.clone(),
                    source: object_store::Error::NotFound {
                        path,
                        source: Box::<dyn std::error::Error + Send + Sync>::from(original),
                    },
                }),
                None => JammiError::DataFusion(e),
            },
        }
    }
}

/// Shape (a): destructure `e` by value looking for an `External(Box<JammiError>)`
/// payload, recursing through the three Box-carrying wrappers and rebuilding the
/// original on a miss.
fn unwrap_jammi(
    e: datafusion::error::DataFusionError,
) -> std::result::Result<JammiError, datafusion::error::DataFusionError> {
    use datafusion::error::DataFusionError as DF;
    match e {
        DF::External(b) => match b.downcast::<JammiError>() {
            Ok(j) => Ok(*j),
            Err(b) => Err(DF::External(b)),
        },
        DF::Context(msg, inner) => match unwrap_jammi(*inner) {
            Ok(j) => Ok(j),
            Err(back) => Err(DF::Context(msg, Box::new(back))),
        },
        DF::ArrowError(b, bt) => match *b {
            arrow::error::ArrowError::ExternalError(inner) => {
                match inner.downcast::<JammiError>() {
                    Ok(j) => Ok(*j),
                    Err(inner) => Err(DF::ArrowError(
                        Box::new(arrow::error::ArrowError::ExternalError(inner)),
                        bt,
                    )),
                }
            }
            other => Err(DF::ArrowError(Box::new(other), bt)),
        },
        DF::ParquetError(b) => match *b {
            parquet::errors::ParquetError::External(inner) => {
                match inner.downcast::<JammiError>() {
                    Ok(j) => Ok(*j),
                    Err(inner) => Err(DF::ParquetError(Box::new(
                        parquet::errors::ParquetError::External(inner),
                    ))),
                }
            }
            other => Err(DF::ParquetError(Box::new(other))),
        },
        other => Err(other),
    }
}

/// Shape (b): walk `source()` from `e` and return the first
/// `object_store::Error::NotFound` (its `path` and the original's `Display`).
fn not_found_path(e: &datafusion::error::DataFusionError) -> Option<(String, String)> {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = cur {
        if let Some(object_store::Error::NotFound { path, .. }) =
            err.downcast_ref::<object_store::Error>()
        {
            return Some((path.clone(), err.to_string()));
        }
        cur = err.source();
    }
    None
}

/// Convenience alias for `std::result::Result<T, JammiError>`.
pub type Result<T> = std::result::Result<T, JammiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::error::DataFusionError as DF;

    fn not_found(path: &str) -> object_store::Error {
        object_store::Error::NotFound {
            path: path.to_string(),
            source: Box::<dyn std::error::Error + Send + Sync>::from("gone"),
        }
    }

    /// Shape (a): a typed engine error a plan node raised, wrapped by the
    /// optimizer's `Context`, comes back as the exact variant.
    #[test]
    fn classifier_restores_a_nested_external_jammi_error() {
        let e = DF::Context(
            "opt".into(),
            Box::new(DF::External(Box::new(JammiError::InvalidKey {
                column: "id".into(),
                null_count: 3,
            }))),
        );
        match JammiError::from(e) {
            JammiError::InvalidKey { column, null_count } => {
                assert_eq!(column, "id");
                assert_eq!(null_count, 3);
            }
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    /// Shape (b): an object-store not-found nested under the parquet reader's
    /// `External` (where no top-level arm reaches) becomes the typed `Storage`
    /// not-found every reader already matches, naming the path.
    #[test]
    fn classifier_types_a_nested_object_store_not_found() {
        let e = DF::ParquetError(Box::new(parquet::errors::ParquetError::External(Box::new(
            not_found("t__v1.parquet"),
        ))));
        match JammiError::from(e) {
            JammiError::Storage(crate::storage::StorageError::Io {
                path,
                source: object_store::Error::NotFound { path: inner, .. },
            }) => {
                assert_eq!(path, "t__v1.parquet");
                assert_eq!(inner, "t__v1.parquet");
            }
            other => panic!("expected Storage(Io(NotFound)), got {other:?}"),
        }
    }

    /// Everything else keeps the `DataFusion` shape with `source()` intact —
    /// the `#[source]` attribute survived dropping `#[from]`.
    #[test]
    fn classifier_keeps_other_errors_as_datafusion_with_source() {
        match JammiError::from(DF::Plan("x".into())) {
            JammiError::DataFusion(DF::Plan(m)) => assert_eq!(m, "x"),
            other => panic!("expected DataFusion(Plan), got {other:?}"),
        }
        let j = JammiError::from(DF::ArrowError(
            Box::new(arrow::error::ArrowError::SchemaError("x".into())),
            None,
        ));
        assert!(matches!(j, JammiError::DataFusion(_)), "got {j:?}");
        assert!(
            std::error::Error::source(&j).is_some(),
            "`source()` must survive on the public type"
        );
    }
}
