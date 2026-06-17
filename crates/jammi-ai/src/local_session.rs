//! The in-process consumer surface over an inference session.
//!
//! [`Session`] is the seam an embedded SDK consumer talks to: an in-process
//! handle that drives a local [`InferenceSession`]. Every method takes owned,
//! serialisable request shapes and returns owned terminal results, so the same
//! verb is expressible over a wire — the network peer is `jammi-client`'s
//! `DataClient`, which speaks the identical request/result vocabulary over gRPC.
//! That vocabulary ([`Modality`] / [`QueryInput`] / [`SearchRequest`] /
//! [`FineTuneJobId`]) lives on the wire substrate ([`jammi_wire::request`]) and
//! is re-exported here so an embedded consumer reaches it as `jammi_ai::*`.
//!
//! The unification choices that make the surface wire-shaped:
//!
//! * **Embeddings/encode** collapse the three per-modality verbs into one
//!   [`Session::generate_embeddings`] / [`Session::encode_query`] pair keyed
//!   by a [`Modality`]. The engine keeps its three concrete methods; the
//!   dispatch lives here.
//! * **Search** is flattened: a [`SearchRequest`] in, the terminal
//!   `Vec<RecordBatch>` out. The fluent [`crate::query::QueryBuilder`] is an
//!   internal mechanism the session drives — never the return type.
//! * **Fine-tune** returns a job **id**; status is polled by id through
//!   [`Session::fine_tune_status`]. The in-process `TrainingJob` handle never
//!   escapes.
//! * **Audit** is three flat methods rather than a borrow-scoped handle.
//! * **Subscribe** returns a `Pin<Box<dyn Stream>>`, the transport-neutral
//!   streaming shape.
//!
//! The in-process affordances that cannot cross a wire (the generic-closure
//! `with_tenant_scoped`, the admin scope, the live `ephemeral_session`) stay as
//! inherent methods on [`InferenceSession`], reached through [`Session::engine`].

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use futures::Stream;

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::model_repo::ModelDescriptor;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::source_repo::SourceDescriptor;
use jammi_db::error::{JammiError, Result};
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::manifest::{DefinitionHash, MatchVerdict};
use jammi_db::store::{DerivesFromEdge, Staleness};
use jammi_db::store::mutable::{MutableTableDefinition, MutableTableId};
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TriggerError};
use jammi_db::{ModelTask, PerQueryAudit, ServerInfo, TenantId, TopicId};

use crate::eval::{CompareEvalReport, EmbeddingEvalReport, EvalTask, InferenceEvalReport};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::model::ModelSource;
use crate::session::InferenceSession;

/// The request / result vocabulary lives on the wire substrate so the gRPC
/// converters can satisfy the orphan rule; re-exported here so an embedded
/// consumer reaches it as `jammi_ai::*`, alongside the [`Session`] it drives.
pub use jammi_wire::request::{FineTuneJobId, Modality, QueryInput, SearchQuery, SearchRequest};

pub use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};

/// The in-process consumer session: a handle over an [`InferenceSession`].
///
/// Owns an `Arc<InferenceSession>` because the engine's `search` entry point is
/// keyed on `Arc<Self>` (it hands the shared session to the search plan); every
/// other verb delegates straight through.
pub struct Session {
    engine: Arc<InferenceSession>,
    /// The embedded training worker, present only on the front-door session
    /// ([`crate::Jammi::open`]) — the one drop point that owns the worker for
    /// the process's lifetime. Per-request wrappers (gRPC handlers, the Python
    /// `Database`'s internal session) construct via [`Self::new`] and carry
    /// `None`, so a worker is not spawned per call. Dropping the front-door
    /// session stops the worker (RAII). Held for its `Drop`, not read.
    _worker: Option<crate::fine_tune::worker::EmbeddedWorker>,
}

impl Session {
    /// Wrap an existing engine session without an embedded worker. Used by the
    /// per-request wrappers (gRPC handlers) and any caller that owns the training
    /// worker elsewhere (the server `train` tier; the Python `Database`). The
    /// [`Self::fine_tune`] verb still submits jobs through this — the worker that
    /// runs them just lives elsewhere.
    pub fn new(engine: Arc<InferenceSession>) -> Self {
        Self {
            engine,
            _worker: None,
        }
    }

    /// Wrap an engine session and spawn the embedded
    /// [`crate::fine_tune::worker::TrainingWorker`] the resulting session owns.
    /// This is the SDK front-door form ([`crate::Jammi::open`]): the embedded
    /// engine both submits training jobs and runs them, and the worker stops when
    /// this session drops (RAII). Must be called inside a tokio runtime context
    /// (the worker spawns a task).
    ///
    /// Returns [`jammi_db::error::JammiError::Config`] if the session's
    /// `[training]` timing violates the worker invariants; the engine's
    /// `JammiConfig::load` already validated it for the normal front-door flow.
    pub fn with_embedded_worker(engine: Arc<InferenceSession>) -> Result<Self> {
        let worker = crate::fine_tune::worker::EmbeddedWorker::spawn(&engine)?;
        Ok(Self {
            engine,
            _worker: Some(worker),
        })
    }

    /// The underlying engine session. The in-process affordances that are not
    /// on the wire-shaped surface (`with_tenant_scoped`, `with_admin_scope`,
    /// `ephemeral_session`, and the engine internals) are reached through this
    /// handle.
    pub fn engine(&self) -> &Arc<InferenceSession> {
        &self.engine
    }

    // --- sources ---------------------------------------------------------

    /// Register a data source.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        self.engine
            .add_source(source_id, source_type, connection)
            .await
    }

    /// Remove a source and all associated state.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.engine.remove_source(source_id).await
    }

    /// Describe every source registered to the session's tenant: each one's
    /// registry identity plus the embedding result tables produced from it.
    /// Registry introspection, not a SQL query.
    pub async fn list_sources(&self) -> Result<Vec<SourceDescriptor>> {
        self.engine.catalog().list_source_descriptors().await
    }

    /// Describe every model registered to the session's tenant, as the
    /// client-facing [`ModelDescriptor`] projection. Registry introspection (the
    /// peer of [`Self::list_sources`] on the model catalog), not a SQL query. The
    /// projection is the single client-facing shape: the catalog's full
    /// [`ModelRecord`](jammi_db::catalog::model_repo::ModelRecord) (with its
    /// server-internal bookkeeping) never crosses this boundary.
    pub async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        let records = self.engine.catalog().list_models().await?;
        Ok(records.iter().map(ModelDescriptor::from).collect())
    }

    /// Describe one registered model by id, or `None` when no model with that id
    /// is visible to the session's tenant. This is the introspection sense (the
    /// per-model peer of [`Self::list_models`]). Returns the client-facing
    /// [`ModelDescriptor`] projection, never the raw record.
    pub async fn describe_model(&self, model_id: &str) -> Result<Option<ModelDescriptor>> {
        let record = self.engine.catalog().get_model(model_id).await?;
        Ok(record.as_ref().map(ModelDescriptor::from))
    }

    /// Hard-delete a model row, removing it — so it is refused while any
    /// reference still points at the model, surfacing
    /// [`JammiError::ModelReferenced`].
    /// When `version` is `None` the latest version is targeted. A tenant may
    /// delete only a model it owns. When `if_exists` is set, deleting an absent
    /// model is a success no-op; otherwise it is reported as absent.
    pub async fn delete_model(
        &self,
        model_id: &str,
        version: Option<i32>,
        if_exists: bool,
    ) -> Result<()> {
        self.engine
            .catalog()
            .delete_model(model_id, version, if_exists)
            .await
    }

    /// Describe one registered source by id, or `None` when no source with that
    /// id is visible to the session's tenant.
    pub async fn describe_source(&self, source_id: &str) -> Result<Option<SourceDescriptor>> {
        self.engine.catalog().describe_source(source_id).await
    }

    /// The engine's capabilities handshake: version, compiled feature flags, and
    /// addressable storage backends. The engine's capabilities are a compile-time
    /// fact, so this reads them straight off [`ServerInfo::current`]; it is
    /// `async` only to match the wire-shaped surface (the remote peer round-trips).
    pub async fn server_info(&self) -> Result<ServerInfo> {
        Ok(ServerInfo::current())
    }

    // --- sql -------------------------------------------------------------

    /// Execute a SQL query.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.engine.sql(query).await
    }

    // --- embeddings ------------------------------------------------------

    /// Generate embeddings for `columns` of a source with the given model and
    /// modality, persisting one vector per row. `key_column` carries each row's
    /// stable key into the result table.
    pub async fn generate_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        modality: Modality,
    ) -> Result<ResultTableRecord> {
        match modality {
            Modality::Text => {
                self.engine
                    .generate_text_embeddings(source_id, model_id, columns, key_column)
                    .await
            }
            Modality::Image => {
                let image_column = single_column(columns, "image")?;
                self.engine
                    .generate_image_embeddings(source_id, model_id, image_column, key_column)
                    .await
            }
            Modality::Audio => {
                let audio_column = single_column(columns, "audio")?;
                self.engine
                    .generate_audio_embeddings(source_id, model_id, audio_column, key_column)
                    .await
            }
        }
    }

    /// Encode a single query into a vector with the given model. The `modality`
    /// selects the tower; `input` must match it (text for [`Modality::Text`],
    /// bytes for image/audio).
    pub async fn encode_query(
        &self,
        model_id: &str,
        input: QueryInput,
        modality: Modality,
    ) -> Result<Vec<f32>> {
        match (modality, input) {
            (Modality::Text, QueryInput::Text(text)) => {
                self.engine.encode_text_query(model_id, &text).await
            }
            (Modality::Image, QueryInput::Bytes(bytes)) => {
                self.engine.encode_image_query(model_id, &bytes).await
            }
            (Modality::Audio, QueryInput::Bytes(bytes)) => {
                self.engine.encode_audio_query(model_id, &bytes).await
            }
            (modality, _) => Err(jammi_db::error::JammiError::Inference(format!(
                "encode_query: {modality:?} requires {} input",
                match modality {
                    Modality::Text => "text",
                    Modality::Image | Modality::Audio => "bytes",
                }
            ))),
        }
    }

    /// Read the `vector` column of an embedding result table into one `Vec<f32>`
    /// per row.
    pub async fn read_vectors(&self, table: &ResultTableRecord) -> Result<Vec<Vec<f32>>> {
        self.engine.read_vectors(table).await
    }

    // --- search ----------------------------------------------------------

    /// Run a vector search and return the terminal hydrated batches.
    pub async fn search(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        let SearchRequest {
            source_id,
            query,
            k,
            embedding_table,
            filter,
            select,
        } = request;
        let embedding_table = embedding_table.as_deref();

        let builder = match query {
            SearchQuery::Vector(vector) => {
                self.engine
                    .search(&source_id, vector, k, embedding_table)
                    .await?
            }
            SearchQuery::RowKey(row_key) => {
                self.engine
                    .search_by_id(&source_id, &row_key, k, embedding_table)
                    .await?
            }
        };
        let builder = match filter.as_deref() {
            Some(predicate) => builder.filter(predicate)?,
            None => builder,
        };
        let builder = if select.is_empty() {
            builder
        } else {
            builder.select(&select)?
        };
        builder.run().await
    }

    // --- inference -------------------------------------------------------

    /// Run inference on a registered source using a model.
    pub async fn infer(
        &self,
        source_id: &str,
        model_id: &str,
        task: ModelTask,
        content_columns: &[String],
        key_column: &str,
    ) -> Result<Vec<RecordBatch>> {
        let source = ModelSource::parse(model_id);
        self.engine
            .infer(source_id, &source, task, content_columns, key_column)
            .await
    }

    // --- materialization contract ----------------------------------------

    /// Recompute a materialised result table's artifact digest and check it
    /// (and, if given, an expected definition hash) against its
    /// `.materialization.json` manifest. Read-only; returns a [`MatchVerdict`],
    /// never acting on one.
    ///
    /// The verdict attests the Parquet **data**, never the ANN search index (the
    /// index is a derived accelerator reconstructible from the data). A table
    /// created before the contract landed carries no manifest and verifies as
    /// [`MatchVerdict::MissingManifest`] — a truthful unknown, never a fabricated
    /// match.
    pub async fn verify_materialization(
        &self,
        table: &str,
        expected_definition: Option<DefinitionHash>,
    ) -> Result<MatchVerdict> {
        let record = self
            .engine
            .catalog()
            .get_result_table(table)
            .await?
            .ok_or_else(|| JammiError::Catalog(format!("Result table '{table}' not found")))?;
        self.engine
            .result_store()
            .verify_materialization(&record, expected_definition.as_ref())
            .await
    }

    /// Report whether a `ready` result table is still the output of its recorded
    /// definition over its recorded inputs' current state — the read-only
    /// `staleness` sensor. Reports a [`Staleness`]; it acts on nothing
    /// (recompute / accept / alarm is the reader's policy, the
    /// `verify_materialization` stance).
    ///
    /// `current_definition` is the hash of *how this table is produced now* — the
    /// caller computes it from the producer's current descriptor + environment;
    /// a divergence from the recorded hash is a `DefinitionChanged` reason. An
    /// input with no reproducible current anchor makes the verdict
    /// [`Staleness::Undecidable`] (never a confident `Fresh`). A pre-contract
    /// table (no recorded definition) is [`Staleness::MissingManifest`].
    ///
    /// Tenant-scoped: the table is resolved through the tenant-filtered
    /// `get_result_table`, so a peer cannot sense a table it cannot resolve.
    pub async fn staleness(
        &self,
        table: &str,
        current_definition: DefinitionHash,
    ) -> Result<Staleness> {
        let record = self
            .engine
            .catalog()
            .get_result_table(table)
            .await?
            .ok_or_else(|| JammiError::Catalog(format!("Result table '{table}' not found")))?;
        self.engine
            .result_store()
            .staleness(&record, &current_definition)
            .await
    }

    /// The one-hop reverse-dependency edges of a result table — every `ready`
    /// table that anchored on it. Read-only lineage data the caller walks
    /// transitively; the engine ships the edges, never the recompute loop.
    ///
    /// Tenant-scoped: the table is resolved through the tenant-filtered
    /// `get_result_table` before its dependents are gathered, so a peer cannot
    /// enumerate the lineage of a table it cannot resolve. The returned edges are
    /// likewise drawn only from the tenant's own (and GLOBAL) `ready` tables.
    pub async fn derives_from(&self, table: &str) -> Result<Vec<DerivesFromEdge>> {
        let record = self
            .engine
            .catalog()
            .get_result_table(table)
            .await?
            .ok_or_else(|| JammiError::Catalog(format!("Result table '{table}' not found")))?;
        self.engine
            .result_store()
            .derives_from(&record.table_name)
            .await
    }

    // --- fine-tune -------------------------------------------------------

    /// Start a fine-tuning job and return its id. Poll completion with
    /// [`Self::fine_tune_status`].
    pub async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<FineTuneJobId> {
        let job = self
            .engine
            .fine_tune(source, base_model, columns, method, task, config)
            .await?;
        Ok(FineTuneJobId(job.job_id))
    }

    /// Current status string for a fine-tune job, looked up by id.
    pub async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        let record = self.engine.catalog().get_training_job(&id.0).await?;
        Ok(record.status)
    }

    // --- eval ------------------------------------------------------------

    /// Evaluate embedding quality against golden relevance judgments.
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
        cohorts: &std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    ) -> Result<EmbeddingEvalReport> {
        self.engine
            .eval_embeddings(source_id, embedding_table, golden_source, k, cohorts)
            .await
    }

    /// Read back the persisted per-query eval records for a run.
    pub async fn eval_per_query(&self, eval_run_id: &str) -> Result<Vec<PerQueryEvalRecord>> {
        self.engine.eval_per_query(eval_run_id).await
    }

    /// Evaluate inference quality against golden labels.
    pub async fn eval_inference(
        &self,
        model_id: &str,
        source_id: &str,
        columns: &[String],
        task: EvalTask,
        golden_source: &str,
        label_column: &str,
    ) -> Result<InferenceEvalReport> {
        self.engine
            .eval_inference(
                model_id,
                source_id,
                columns,
                task,
                golden_source,
                label_column,
            )
            .await
    }

    /// Compare multiple embedding tables side-by-side.
    pub async fn eval_compare(
        &self,
        embedding_tables: &[String],
        source_id: &str,
        golden_source: &str,
        k: usize,
    ) -> Result<CompareEvalReport> {
        self.engine
            .eval_compare(embedding_tables, source_id, golden_source, k)
            .await
    }

    // --- mutable tables --------------------------------------------------

    /// Register a mutable companion table.
    pub async fn create_mutable_table(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        self.engine.create_mutable_table(def).await
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        self.engine.drop_mutable_table(id).await
    }

    /// List every mutable companion table registered to the session's tenant.
    /// Registry introspection, not a SQL query.
    pub async fn list_mutable_tables(&self) -> Result<Vec<MutableTableDefinition>> {
        Ok(self
            .engine
            .mutable_tables()
            .list(self.engine.tenant())
            .await?)
    }

    // --- trigger ---------------------------------------------------------

    /// Register a topic (creates its backing table) for the trigger stream.
    pub async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        // Register with *both* the broker driver and the catalog, mirroring the
        // control-plane path: the catalog row is the system of record a later
        // lookup reads, but a `publish` resolves the topic against the broker —
        // so a catalog-only registration would make a publish to this topic fail
        // with `TopicNotFound`. The session always carries a broker (defaulting
        // to the in-memory broker), so this is total.
        self.engine.trigger_broker().register_topic(topic).await?;
        self.engine.topic_repo().register_topic(topic).await
    }

    /// List every topic visible to the session's tenant.
    pub async fn list_topics(&self) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        self.engine
            .topic_repo()
            .list_topics(self.engine.tenant())
            .await
    }

    /// Drop a topic and its backing table.
    pub async fn drop_topic(&self, topic_id: TopicId) -> std::result::Result<(), TriggerError> {
        // Drop the catalog row (the system of record) first, then the broker
        // driver's view of the topic. The broker drop is best-effort: a driver
        // failure after the catalog row is gone leaks driver resources, not
        // catalog state, so it is surfaced via tracing rather than reverting the
        // successful catalog drop — mirroring the control-plane path.
        self.engine
            .topic_repo()
            .drop_topic(topic_id, self.engine.tenant())
            .await?;
        if let Err(e) = self.engine.trigger_broker().drop_topic(topic_id).await {
            tracing::warn!(
                topic_id = %topic_id,
                error = %e,
                "trigger broker driver failed to drop topic after catalog row removal",
            );
        }
        Ok(())
    }

    /// Publish one batch to a topic under the session's tenant scope, returning
    /// the assigned offset.
    pub async fn publish(
        &self,
        topic: &TopicDefinition,
        batch: RecordBatch,
    ) -> std::result::Result<Offset, TriggerError> {
        self.engine
            .publisher()
            .publish_scoped(topic, self.engine.tenant(), batch)
            .await
    }

    /// Subscribe to a topic, returning a transport-neutral stream of delivered
    /// batches. The stream replays from `from_offset` (or the live tail when
    /// `None`) and then tails live, scoped to the session's tenant.
    ///
    /// `replay_only` selects the finite-drain primitive: when set, the stream
    /// yields exactly the retained batches the predicate accepts from
    /// `from_offset` onward and then *terminates*, rather than holding open to
    /// tail live batches. This is a different engine primitive
    /// ([`jammi_db::trigger::Subscriber::replay_only`]) from the open-ended
    /// subscribe, surfaced through the same stream return so a bounded drain
    /// (`jammi trigger subscribe --no-follow`) reads it the same way it reads a
    /// live subscription — it just sees the stream end.
    pub async fn subscribe(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
        replay_only: bool,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<DeliveredBatch, TriggerError>> + Send>>,
        TriggerError,
    > {
        let tenant = self.engine.tenant();
        if replay_only {
            // The finite-drain primitive: collect the retained replay window and
            // hand it back as a stream that yields those batches and ends, so the
            // bounded `--no-follow` path reads a subscription that simply
            // terminates — never the open-ended live tail (which cannot end).
            let drained = self
                .engine
                .subscriber()
                .replay_only_scoped(topic, tenant, predicate, from_offset)
                .await?;
            return Ok(Box::pin(futures::stream::iter(drained.into_iter().map(Ok))));
        }
        let subscription = self
            .engine
            .subscriber()
            .subscribe_scoped(topic, tenant, predicate, from_offset)
            .await?;
        Ok(Box::pin(subscription))
    }

    // --- channels --------------------------------------------------------

    /// Register an evidence channel and its columns.
    pub async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        self.engine.catalog().channels().register(spec).await
    }

    /// Append columns to an already-registered channel (append-only).
    pub async fn add_channel_columns(
        &self,
        channel: &jammi_db::ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        self.engine
            .catalog()
            .channels()
            .add_columns(channel, new_columns)
            .await
    }

    /// List every evidence channel registered to the session's tenant, ordered
    /// by `(priority, channel_id)`. Registry introspection, not a SQL query.
    pub async fn list_channels(&self) -> Result<Vec<ChannelSpec>> {
        self.engine.catalog().channels().list().await
    }

    // --- tenant ----------------------------------------------------------

    /// Bind a tenant scope to this session (sticky form).
    ///
    /// `async` to match the wire-shaped surface (a remote peer's tenant trio is
    /// a round-trip); the in-process engine binding is synchronous, so this is an
    /// `async`-wrap with no inner await — infallible in-process, but the surface
    /// returns `Result` because a remote transport's bind can fail.
    pub async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.engine.bind_tenant(t);
        Ok(())
    }

    /// Clear the bound tenant.
    pub async fn unbind_tenant(&self) -> Result<()> {
        self.engine.unbind_tenant();
        Ok(())
    }

    /// The tenant currently bound, if any.
    pub async fn tenant(&self) -> Result<Option<TenantId>> {
        Ok(self.engine.tenant())
    }

    // --- audit -----------------------------------------------------------

    /// Sign and persist a batch of audit records; publishes them to the audit
    /// topic.
    pub async fn audit_log(
        &self,
        records: Vec<PerQueryAudit>,
    ) -> std::result::Result<(), jammi_db::AuditError> {
        self.engine.audit().log(records).await
    }

    /// Fetch one audit record by query id (tenant-scoped).
    pub async fn audit_fetch_by_query_id(
        &self,
        query_id: uuid::Uuid,
    ) -> std::result::Result<Option<PerQueryAudit>, jammi_db::AuditError> {
        self.engine.audit().fetch_by_query_id(query_id).await
    }

    /// Fetch the most recent audit records (tenant-scoped), newest first.
    pub async fn audit_fetch_recent(
        &self,
        limit: usize,
    ) -> std::result::Result<Vec<PerQueryAudit>, jammi_db::AuditError> {
        self.engine.audit().fetch_recent(limit).await
    }
}

/// Resolve a single-column embedding input. The image and audio engine verbs
/// each take exactly one content column; the unified surface passes a slice, so
/// reject anything but a one-element slice with a typed error naming the
/// modality rather than silently using the first column.
fn single_column<'a>(columns: &'a [String], modality: &str) -> Result<&'a str> {
    match columns {
        [single] => Ok(single.as_str()),
        _ => Err(jammi_db::error::JammiError::Inference(format!(
            "{modality} embeddings take exactly one content column, got {}",
            columns.len()
        ))),
    }
}
