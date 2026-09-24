use thiserror::Error;

/// Unified error type for all Jammi DB operations.
///
/// `Clone`: an error raised below an exchange reaches every consumer of it
/// (DataFusion hands each a `Shared(Arc<_>)`), so every one of them must be able
/// to take the typed error out. Foreign payloads that are not `Clone` are held
/// behind an `Arc`.
#[derive(Debug, Clone, Error)]
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

    /// A source resolved no `sources` row: it was never registered, or it was
    /// removed — on this process or on any other replica sharing the catalog.
    /// An absent row is a NotFound, not a bad argument, so it maps to gRPC
    /// `NotFound` rather than `InvalidArgument`. Raised wherever a source id
    /// is resolved: a SQL scan of `<source>.public.<table>`, and every verb
    /// that names a source.
    #[error("Source not found: {source_id}")]
    SourceNotFound {
        /// Identifier of the source that resolved no row.
        source_id: String,
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

    /// A remote model endpoint's refusal or failure.
    #[error("Backend error: {0}")]
    Backend(String),

    /// Filesystem I/O error.
    #[error("IO error: {0}")]
    Io(#[source] std::sync::Arc<std::io::Error>),

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
    Json(#[source] std::sync::Arc<serde_json::Error>),

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
    DataFusion(#[source] std::sync::Arc<datafusion::error::DataFusionError>),

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
    /// replay for. Three tables land here, and each is a loud typed refusal rather
    /// than a silent best-effort:
    ///
    /// - a **pre-contract** table whose catalog `definition_hash IS NULL` — created
    ///   before migration 021 added the materialization summary, so there is no recorded
    ///   [`ProducingDescriptor`](crate::store::manifest::ProducingDescriptor) to
    ///   dispatch a replay on at all;
    /// - a table produced by an [`External`](crate::store::manifest::ProducingDescriptor::External)
    ///   producer — a verb the engine does not own — which the engine cannot
    ///   reconstruct even though a descriptor is recorded; and
    /// - a [`TrainingSet`](crate::store::manifest::ProducingDescriptor::TrainingSet)
    ///   table whose recorded `order_rule` is one this build does not implement.
    ///   The producer commits exactly one rule
    ///   ([`TRAINING_SET_ORDER_RULE_V1`](crate::store::manifest::TRAINING_SET_ORDER_RULE_V1)),
    ///   and re-materializing under a rule it does not implement would write rows
    ///   in an order the recorded descriptor does not claim, so the replay is
    ///   refused rather than run under a guessed order.
    ///
    /// In each case guessing a producer call would be a fabricated re-run, so
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

    /// A parent-pinned version CAS — the allocation UPDATE or the publish
    /// table-row swap — matched no row because `result_tables.current_version`
    /// no longer equals the parent the caller's delta was derived from: a
    /// concurrent refresh or compaction published between the caller's read
    /// and this CAS. ONE classification for both misses (the allocation
    /// miss and the publish table-row miss are the same lost race and must
    /// not get two typed spellings): raised by `classify_ready_cas_miss`
    /// ahead of its `CasFailed` fallback whenever the row IS `ready` but its
    /// `current_version` disagrees with `expected`. `expected`/`found` are
    /// both `None` only for the base publish's `current_version IS NULL`
    /// CAS; a concurrent refresh's version row is left `building` (allocation
    /// miss) or rolled back to `building` (publish miss) and neither
    /// `next_version` nor `current_version` is touched by the loser. The
    /// caller re-reads the table row and retries from the new parent, or
    /// gives up; the previous version stays live either way.
    #[error(
        "result table `{table}`: parent moved (expected {expected:?}, found {found:?}); a \
         concurrent refresh or compaction published first"
    )]
    ParentMoved {
        /// The table whose parent-pinned CAS missed.
        table: String,
        /// The parent the caller's delta was derived from.
        expected: Option<i64>,
        /// The table's actual `current_version` at the CAS.
        found: Option<i64>,
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
    /// inference / refresh: refused typed at the input edge, never
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

    /// A training set's source projection yielded **zero rows**, so there is
    /// nothing to train on. Raised by
    /// [`crate::store::ResultStore::materialize_training_set`] *before* any
    /// catalog row or byte exists.
    ///
    /// An empty training set is a refusal, never a 0-row table: a run that
    /// trained on one would complete "successfully" having learned nothing,
    /// publishing a model whose emptiness is invisible downstream. The
    /// degenerate input is caught at the producer's own edge — the boundary
    /// where the row count is first known — rather than left for a consumer
    /// to notice, so no half-built artifact and no `building` row survives
    /// the refusal.
    #[error("training set over `{source_query}` is empty: the projection yielded zero rows")]
    EmptyTrainingSet {
        /// The producer's query text that yielded no rows. Named
        /// `source_query`, not `source`: `thiserror` reads a field named
        /// `source` as the error's `std::error::Error::source()`, which a
        /// `String` cannot be.
        source_query: String,
    },

    /// A DataFusion plan operator (a `SortPreservingMergeExec`'s
    /// per-partition reservation, an external sorter's spill buffer) or an
    /// engine-side consumer registered directly against the session's
    /// [`crate::session::JammiSession::memory_pool`] (a training-set
    /// stream's chunk reservation, an eager materialization's
    /// collected-batch reservation) tried to grow past `[engine]
    /// memory_limit`'s pool. Typed, never a panic, and never a silent wait:
    /// the pool's own `try_grow` fails synchronously, so this surfaces from
    /// the same public path (`sql`, `sql_stream`, a streamed loader's
    /// `next_chunk`) the query or reservation was made on.
    #[error("resources exhausted: pool limit is {limit_bytes} byte(s): {detail}")]
    ResourcesExhausted {
        /// The pool's configured byte limit. For a `DataFusionError::ResourcesExhausted`
        /// classified through `From<DataFusionError>`, this is a BEST-EFFORT
        /// value recovered by parsing the pool's own `Display` impl
        /// (`"greedy(used: …, pool_size: …)"`) out of `detail` — that text is
        /// itself human-readable and rounded to one decimal place by
        /// DataFusion, so the recovered value is approximate, and is `0`
        /// when the message carries no recognisable `pool_size: ` marker. A
        /// caller that constructs this variant directly (an engine-side
        /// consumer that already knows the configured limit) sets the exact
        /// value. `detail` is always the untouched original message, so no
        /// information is lost regardless of what this field carries.
        limit_bytes: u64,
        /// The raising operator's or consumer's own message, verbatim.
        detail: String,
    },

    /// A plan that requires a device kind reached a holder with none of it:
    /// the compute plane before submission (no live registered executor
    /// lists the kind — `held` is every kind the live executors list) or one
    /// executor at stage creation (its own device is another kind — `held`
    /// is that one kind — a stage bound past the scheduler's KIND MATCH).
    /// The plan is never silently run on another kind. The plan itself is
    /// well-formed; what must change is the plane's device inventory, so
    /// this maps to gRPC `FailedPrecondition`.
    #[error(
        "device kind {} is unheld: the plan requires it, the holder lists [{}]",
        required.wire_str(),
        wire_kinds(held)
    )]
    DeviceKindUnheld {
        /// The kind the plan's own `InferenceExec`/`TrainingExec` stamps.
        required: jammi_datafusion::ComputeDeviceKind,
        /// The kinds the holder lists, distinct and in wire order.
        held: Vec<jammi_datafusion::ComputeDeviceKind>,
    },

    /// A stage whose plan carries a placed training attempt was planned at
    /// more than one partition: one attempt is one task, never a fan-out of
    /// its body. Refused by the executor before the stage runs. An engine
    /// invariant — the submitter's plan, or the scheduler's planning of it
    /// — never a caller condition, so this maps to gRPC `Internal`.
    #[error(
        "placed attempt of job `{job_id}` was planned at {partitions} partitions; a placed \
         attempt's stage is one partition"
    )]
    PlacedAttemptFanOut {
        /// The placed attempt's own training job id.
        job_id: String,
        /// The partition count the stage was planned at.
        partitions: u64,
    },

    /// Catch-all for errors that don't fit another variant.
    /// The compute plane cannot hold a plan a caller required it to hold —
    /// a claimed training attempt's one task — right now: no live executor, none of the
    /// kind the plan requires, only the plan's own submitter, or a plan the
    /// wire cannot carry. A runtime state of the plane, never a fault in
    /// the plan or its caller; a materialization runs in-process on the
    /// same refusal and never raises it.
    #[error("compute plane: {0}")]
    Unheld(crate::compute_plane::Unheld),

    /// The compute plane lost the executor holding a placed job's task:
    /// its scheduler expired `executor_id` — a heartbeat that stopped, a
    /// launch it could not deliver — while `job_id`, the plane's own id
    /// for the placed plan, had a task running on it. Raised by the
    /// plane's scheduler at the loss and delivered to the submitter as the
    /// placed job's failure, so the submitter learns of it when the plane
    /// does, never from a relaunch. An attempt ended this way is spent and
    /// left for its job's successor; nothing about the plan or its caller
    /// must change, so this maps to gRPC `Unavailable` — the code a peer
    /// that went away carries — never `FailedPrecondition`.
    #[error("compute plane: executor `{executor_id}` holding placed job `{job_id}` was lost")]
    ExecutorLost {
        /// The executor the plane expired.
        executor_id: String,
        /// The plane's own id for the placed plan.
        job_id: String,
    },

    #[error("{0}")]
    Other(String),
}

/// `kinds` as their wire tokens, comma-separated — the `Display` of every
/// error that lists a device inventory.
fn wire_kinds(kinds: &[jammi_datafusion::ComputeDeviceKind]) -> String {
    kinds
        .iter()
        .map(|k| k.wire_str())
        .collect::<Vec<_>>()
        .join(", ")
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
/// [`JammiError`] — one borrowed walk of the `source()` chain, outermost first,
/// stopping at the first error that is one of:
///
/// 1. a [`JammiError`] payload (a typed error a plan node, provider or UDF
///    raised as `External`), cloned out — at any depth, under `Context`,
///    `ArrowError(ExternalError)`, `ParquetError(External)`, or the
///    `Shared(Arc<_>)` an exchange hands each of its consumers;
/// 2. a `ResourcesExhausted`, always typed;
/// 3. an `object_store::Error::NotFound` (the parquet reader wraps it under
///    `ParquetError::External`, which no top-level arm reaches), as
///    [`StorageError::NotFound`](crate::storage::StorageError::NotFound).
///
/// Everything else keeps the shape [`JammiError::DataFusion`] with `source()`
/// intact.
impl From<datafusion::error::DataFusionError> for JammiError {
    fn from(e: datafusion::error::DataFusionError) -> Self {
        use datafusion::error::DataFusionError as DF;
        let typed = source_chain(&e).find_map(|err| {
            if let Some(inner) = err.downcast_ref::<JammiError>() {
                Some(inner.clone())
            } else if let Some(inner) = jammi_datafusion::Error::found_in(err) {
                Some(operator_error(inner))
            } else if let Some(DF::ResourcesExhausted(msg)) = err.downcast_ref::<DF>() {
                Some(JammiError::ResourcesExhausted {
                    limit_bytes: parse_pool_size_bytes(msg).unwrap_or(0),
                    detail: msg.clone(),
                })
            } else if let Some(object_store::Error::NotFound { path, .. }) =
                err.downcast_ref::<object_store::Error>()
            {
                Some(JammiError::Storage(
                    crate::storage::StorageError::not_found(path.clone(), err.to_string()),
                ))
            } else {
                None
            }
        });
        typed.unwrap_or_else(|| JammiError::DataFusion(std::sync::Arc::new(e)))
    }
}

/// A `jammi-datafusion` operator's error as this crate's: the null-key
/// refusal keeps its typed shape, a runtime failure that was one of ours is
/// restored from its source chain, a training job reaching a process that
/// runs none is a fine-tune refusal, and the rest is an inference error
/// naming the cause.
fn operator_error(e: &jammi_datafusion::Error) -> JammiError {
    use jammi_datafusion::Error as Operator;
    match e {
        Operator::InvalidKey { column, null_count } => JammiError::InvalidKey {
            column: column.clone(),
            null_count: *null_count,
        },
        Operator::Runtime(source) => source_chain(source.as_ref())
            .find_map(|err| err.downcast_ref::<JammiError>().cloned())
            .unwrap_or_else(|| JammiError::Inference(e.to_string())),
        Operator::NoTrainingRunner => JammiError::FineTune(e.to_string()),
        Operator::Inference(_)
        | Operator::UnknownTask(_)
        | Operator::UnknownDeviceKind(_)
        | Operator::Decode(_) => JammiError::Inference(e.to_string()),
    }
}

impl From<jammi_datafusion::Error> for JammiError {
    fn from(e: jammi_datafusion::Error) -> Self {
        operator_error(&e)
    }
}

/// `e`, then every error `source()` reaches from it.
fn source_chain<'a>(
    e: &'a (dyn std::error::Error + 'static),
) -> impl Iterator<Item = &'a (dyn std::error::Error + 'static)> {
    std::iter::successors(Some(e), |err| err.source())
}

/// A refused query vector, classified by its provenance
/// ([`jammi_numerics::query::QuerySource`]): a CALLER's vector is the caller's
/// fault — the schema class every width mismatch already maps to (gRPC
/// `InvalidArgument`); a vector read back from STORAGE, or a downstream
/// ARTIFACT the query disagreed with after construction, is a corrupt
/// artifact named by its table or index (the same class an unreadable
/// sidecar maps to, gRPC `Internal`). `ArtifactMismatch` carries no
/// `QuerySource` — it is never about the query's own provenance — so it is
/// matched directly rather than through `.source()`.
impl From<jammi_numerics::query::QueryValidationError> for JammiError {
    fn from(e: jammi_numerics::query::QueryValidationError) -> Self {
        use jammi_numerics::query::{QuerySource, QueryValidationError};

        /// A query that disagreed with `expected`, blamed by its provenance.
        fn refused_query(source: QuerySource, expected: String, actual: String) -> JammiError {
            match source {
                QuerySource::Caller => JammiError::Schema {
                    table: "query".into(),
                    column: "query".into(),
                    expected,
                    actual,
                },
                QuerySource::Stored { table } => JammiError::IncompatibleFormat {
                    artifact: format!("{table}.vector"),
                    found: actual,
                    supported: expected,
                },
            }
        }

        match e {
            QueryValidationError::NonFinite {
                index,
                value,
                source,
            } => refused_query(
                source,
                "finite f32 components".to_string(),
                format!("component {index} is {value:?}"),
            ),
            QueryValidationError::Width {
                expected,
                actual,
                source,
            } => refused_query(
                source,
                format!("{expected} dimensions"),
                format!("{actual} dimensions"),
            ),
            // Carries no `QuerySource` at all — engine-fault by construction,
            // never the query's own provenance, regardless of it.
            QueryValidationError::ArtifactMismatch {
                artifact,
                expected,
                actual,
            } => JammiError::IncompatibleFormat {
                artifact: format!("{artifact}.vector"),
                found: format!("{actual} dimensions"),
                supported: format!("{expected} dimensions"),
            },
        }
    }
}

impl From<std::io::Error> for JammiError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(std::sync::Arc::new(e))
    }
}

impl From<serde_json::Error> for JammiError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(std::sync::Arc::new(e))
    }
}

/// Best-effort recovery of a bounded pool's (the session's `ActiveSpillPool`,
/// or DataFusion's `GreedyMemoryPool`/`FairSpillPool`) configured
/// byte limit from its own `Display` impl embedded in a `ResourcesExhausted`
/// message (`"…pool_size: <value> <unit>…"`, `<value>` rounded to one
/// decimal place and `<unit>` one of `B`/`KB`/`MB`/`GB`/`TB`, binary-based —
/// see `datafusion_common::display::human_readable_size`). Returns `None`
/// when the message carries no `pool_size: ` marker, or the token after it
/// does not parse as `<f64> <unit>` — never a panic on an unrecognised
/// shape.
fn parse_pool_size_bytes(message: &str) -> Option<u64> {
    const MARKER: &str = "pool_size: ";
    let start = message.find(MARKER)? + MARKER.len();
    let rest = &message[start..];
    let end = rest.find([')', ',']).unwrap_or(rest.len());
    let token = rest[..end].trim();
    let mut parts = token.split_whitespace();
    let value: f64 = parts.next()?.parse().ok()?;
    let unit = parts.next()?;
    let multiplier: f64 = match unit {
        "B" => 1.0,
        "KB" => 1024.0,
        "MB" => 1024.0 * 1024.0,
        "GB" => 1024.0 * 1024.0 * 1024.0,
        "TB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    let bytes = value * multiplier;
    if bytes.is_finite() && bytes >= 0.0 {
        Some(bytes.round() as u64)
    } else {
        None
    }
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

    /// A typed engine error a plan node raised, wrapped by the optimizer's
    /// `Context`, comes back as the exact variant.
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

    /// An exchange hands every consumer the same upstream error as
    /// `Shared(Arc<_>)`, and other clones of the `Arc` are alive while each one
    /// classifies it. Every consumer still gets the exact typed variant.
    #[test]
    fn classifier_restores_a_typed_error_from_every_clone_of_a_shared_error() {
        let shared = std::sync::Arc::new(DF::Context(
            "below the exchange".into(),
            Box::new(DF::External(Box::new(JammiError::InvalidKey {
                column: "id".into(),
                null_count: 3,
            }))),
        ));
        let consumers = [DF::Shared(shared.clone()), DF::Shared(shared.clone())];
        for e in consumers {
            assert!(
                matches!(
                    JammiError::from(e),
                    JammiError::InvalidKey { ref column, null_count: 3 } if column == "id"
                ),
                "every consumer of a shared error classifies it typed"
            );
        }
        let exhausted = std::sync::Arc::new(DF::ResourcesExhausted("pool_size: 64.0 KB)".into()));
        let _other_consumer = exhausted.clone();
        assert!(matches!(
            JammiError::from(DF::Shared(exhausted)),
            JammiError::ResourcesExhausted {
                limit_bytes: 65536,
                ..
            }
        ));
    }

    /// An object-store not-found nested under the parquet reader's `External`
    /// (where no top-level arm reaches) becomes the typed storage not-found,
    /// naming the path.
    #[test]
    fn classifier_types_a_nested_object_store_not_found() {
        let e = DF::ParquetError(Box::new(parquet::errors::ParquetError::External(Box::new(
            not_found("t__v1.parquet"),
        ))));
        match JammiError::from(e) {
            JammiError::Storage(crate::storage::StorageError::NotFound { path, .. }) => {
                assert_eq!(path, "t__v1.parquet");
            }
            other => panic!("expected Storage(NotFound), got {other:?}"),
        }
    }

    /// Everything else keeps the `DataFusion` shape with `source()` intact —
    /// the `#[source]` attribute survived dropping `#[from]`.
    #[test]
    fn classifier_keeps_other_errors_as_datafusion_with_source() {
        match JammiError::from(DF::Plan("x".into())) {
            JammiError::DataFusion(e) => assert!(matches!(&*e, DF::Plan(m) if m == "x")),
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

    /// A bare `ResourcesExhausted` becomes the typed variant,
    /// naming the raising message verbatim in `detail` and recovering the
    /// pool size from its `Display` impl in `limit_bytes`.
    #[test]
    fn classifier_types_a_bare_resources_exhausted() {
        let msg = "Failed to allocate additional 3.7 MB for SortPreservingMergeExec[0] with \
                    0.0 B already allocated for this reservation - 63.6 KB remain available \
                    for the total memory pool: greedy(used: 456.0 B, pool_size: 64.0 KB)"
            .to_string();
        match JammiError::from(DF::ResourcesExhausted(msg.clone())) {
            JammiError::ResourcesExhausted {
                limit_bytes,
                detail,
            } => {
                assert_eq!(detail, msg);
                assert_eq!(limit_bytes, 65536); // 64.0 KB, binary
            }
            other => panic!("expected ResourcesExhausted, got {other:?}"),
        }
    }

    /// A `ResourcesExhausted` wrapped in `Context` is still classified, never
    /// falling through to the generic `DataFusion` catch-all.
    #[test]
    fn classifier_types_a_resources_exhausted_nested_under_context() {
        let e = DF::Context(
            "physical_plan".into(),
            Box::new(DF::ResourcesExhausted(
                "greedy(used: 0.0 B, pool_size: 1.0 MB)".into(),
            )),
        );
        match JammiError::from(e) {
            JammiError::ResourcesExhausted { limit_bytes, .. } => {
                assert_eq!(limit_bytes, 1024 * 1024);
            }
            other => panic!("expected ResourcesExhausted, got {other:?}"),
        }
    }

    /// A `ResourcesExhausted` whose message carries no `pool_size: ` marker
    /// (a pool this parser does not recognise the shape of) still types the
    /// variant — `limit_bytes` degrades to `0`, `detail` keeps the message
    /// whole, never a panic on the unrecognised shape.
    #[test]
    fn classifier_types_an_unparseable_resources_exhausted_with_a_zero_limit() {
        match JammiError::from(DF::ResourcesExhausted("some other pool ran dry".into())) {
            JammiError::ResourcesExhausted {
                limit_bytes,
                detail,
            } => {
                assert_eq!(limit_bytes, 0);
                assert_eq!(detail, "some other pool ran dry");
            }
            other => panic!("expected ResourcesExhausted, got {other:?}"),
        }
    }

    /// [`parse_pool_size_bytes`] over every unit and a degenerate input.
    #[test]
    fn parse_pool_size_bytes_covers_every_unit_and_degenerates_to_none() {
        assert_eq!(
            parse_pool_size_bytes("greedy(used: 0.0 B, pool_size: 456.0 B)"),
            Some(456)
        );
        assert_eq!(
            parse_pool_size_bytes("greedy(used: 0.0 B, pool_size: 2.0 KB)"),
            Some(2048)
        );
        assert_eq!(
            parse_pool_size_bytes("greedy(used: 0.0 B, pool_size: 1.0 GB)"),
            Some(1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_pool_size_bytes("greedy(used: 0.0 B, pool_size: 1.0 TB)"),
            Some(1024u64 * 1024 * 1024 * 1024)
        );
        assert_eq!(parse_pool_size_bytes("no marker here"), None);
        assert_eq!(parse_pool_size_bytes("pool_size: not-a-number KB"), None);
    }
}
