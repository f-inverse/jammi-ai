//! Transport-agnostic consumer surface over an inference session.
//!
//! [`Session`] is the seam an SDK consumer talks to. It is a closed
//! `enum` over the available transports; today the only variant is
//! [`LocalSession`], which drives an in-process [`InferenceSession`]. A
//! future remote variant slots in beside it without changing this surface —
//! every method here is expressible over a wire because it takes owned,
//! serialisable request shapes and returns owned terminal results, never a
//! borrow-scoped handle or a stateful builder.
//!
//! Enum dispatch (rather than `Arc<dyn Session>`) is the right shape for this
//! boundary: the set of transports is closed, the methods carry signatures
//! that an object-safe trait cannot express (`search` needs `Arc<Self>`,
//! `audit` returns a borrow-scoped handle, `with_tenant_scoped` is generic
//! over a closure), and a `match` over a two-or-three-arm enum compiles to a
//! direct call with no `dyn` indirection. The transport-specific affordances
//! that cannot cross a wire (the generic-closure `with_tenant_scoped`, the
//! admin scope, the live `ephemeral_session`) stay on [`LocalSession`] as
//! inherent methods and are simply absent from the [`Session`] surface.
//!
//! The unification choices that make the surface wire-shaped:
//!
//! * **Embeddings/encode** collapse the three per-modality verbs into one
//!   [`Session::generate_embeddings`] / [`Session::encode_query`] pair keyed
//!   by a [`Modality`]. The engine keeps its three concrete methods; the
//!   dispatch lives here.
//! * **Search** is flattened: a [`SearchRequest`] in, the terminal
//!   `Vec<RecordBatch>` out. The fluent [`crate::search::SearchBuilder`] is an
//!   internal mechanism `LocalSession` drives — never the return type.
//! * **Fine-tune** returns a job **id**; status is polled by id through
//!   [`Session::fine_tune_status`]. The in-process [`crate::fine_tune::FineTuneJob`]
//!   handle never escapes.
//! * **Audit** is three flat methods rather than a borrow-scoped handle.
//! * **Subscribe** returns a `Pin<Box<dyn Stream>>`, the transport-neutral
//!   streaming shape.

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use futures::Stream;

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::Result;
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::mutable::{MutableTableDefinition, MutableTableId};
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TriggerError};
use jammi_db::{ModelTask, PerQueryAudit, TenantId, TopicId};

use crate::eval::{CompareEvalReport, EmbeddingEvalReport, EvalTask, InferenceEvalReport};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::model::ModelSource;
use crate::session::InferenceSession;

/// Which embedding tower an [`Session::generate_embeddings`] /
/// [`Session::encode_query`] call targets. Unifies the three per-modality
/// engine verbs (`text`/`image`/`audio`) into one parameter so the consumer
/// surface carries one embedding verb, not three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Dense vectors of input text.
    Text,
    /// Dense vectors of input images.
    Image,
    /// Dense vectors of input audio clips.
    Audio,
}

/// A single query to encode into a vector. Text is encoded by the text tower;
/// raw bytes are encoded by the image or audio tower, with the [`Modality`]
/// the caller passes to [`Session::encode_query`] selecting which.
pub enum QueryInput {
    /// A text string to encode with the text tower.
    Text(String),
    /// Encoded bytes (an image file or an audio clip) for the vision/audio
    /// tower.
    Bytes(Vec<u8>),
}

/// The query side of a flattened search: either a caller-supplied vector or a
/// row key resolved to its stored vector inside the engine.
pub enum SearchQuery {
    /// Search against a caller-supplied query vector.
    Vector(Vec<f32>),
    /// Query-by-example: rank by the vector stored for this row key.
    RowKey(String),
}

/// A flattened vector-search request. Replaces the stateful
/// [`crate::search::SearchBuilder`] on the consumer surface: every knob the
/// builder exposed for a one-shot search (`filter`, `select`) is a field here,
/// so a remote transport can serialise the whole request rather than replay a
/// chain of builder calls.
pub struct SearchRequest {
    /// Source whose embedding table is searched.
    pub source_id: String,
    /// The query vector or the row key to resolve into one.
    pub query: SearchQuery,
    /// Number of nearest neighbours to retrieve.
    pub k: usize,
    /// Optional SQL predicate applied to the hydrated results.
    pub filter: Option<String>,
    /// Columns to project. Empty keeps every hydrated column.
    pub select: Vec<String>,
}

/// Identifier of a fine-tune job started through [`Session::fine_tune`].
/// Returned in place of the in-process [`crate::fine_tune::FineTuneJob`] handle
/// so the job is addressable across a transport boundary; poll it with
/// [`Session::fine_tune_status`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FineTuneJobId(pub String);

/// The consumer-facing session abstraction: a closed `enum` over transports.
///
/// Every method delegates to the underlying transport's implementation of the
/// same verb. The surface is the contract; the variant is the wiring.
pub enum Session {
    /// An in-process session driving a local [`InferenceSession`].
    Local(LocalSession),
}

impl Session {
    // --- sources ---------------------------------------------------------

    /// Register a data source.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        match self {
            Session::Local(s) => s.add_source(source_id, source_type, connection).await,
        }
    }

    /// Remove a source and all associated state.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        match self {
            Session::Local(s) => s.remove_source(source_id).await,
        }
    }

    // --- sql -------------------------------------------------------------

    /// Execute a SQL query.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        match self {
            Session::Local(s) => s.sql(query).await,
        }
    }

    // --- embeddings ------------------------------------------------------

    /// Generate embeddings for `columns` of a source with the given model and
    /// modality, persisting one vector per row. `key_column` carries each
    /// row's stable key into the result table.
    pub async fn generate_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        modality: Modality,
    ) -> Result<ResultTableRecord> {
        match self {
            Session::Local(s) => {
                s.generate_embeddings(source_id, model_id, columns, key_column, modality)
                    .await
            }
        }
    }

    /// Encode a single query into a vector with the given model. The
    /// `modality` selects the tower; `input` must match it (text for
    /// [`Modality::Text`], bytes for image/audio).
    pub async fn encode_query(
        &self,
        model_id: &str,
        input: QueryInput,
        modality: Modality,
    ) -> Result<Vec<f32>> {
        match self {
            Session::Local(s) => s.encode_query(model_id, input, modality).await,
        }
    }

    /// Read the `vector` column of an embedding result table into one
    /// `Vec<f32>` per row.
    pub async fn read_vectors(&self, table: &ResultTableRecord) -> Result<Vec<Vec<f32>>> {
        match self {
            Session::Local(s) => s.read_vectors(table).await,
        }
    }

    // --- search ----------------------------------------------------------

    /// Run a vector search and return the terminal hydrated batches.
    pub async fn search(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        match self {
            Session::Local(s) => s.search(request).await,
        }
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
        match self {
            Session::Local(s) => {
                s.infer(source_id, model_id, task, content_columns, key_column)
                    .await
            }
        }
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
        match self {
            Session::Local(s) => {
                s.fine_tune(source, base_model, columns, method, task, config)
                    .await
            }
        }
    }

    /// Current status string for a fine-tune job, looked up by id.
    pub async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        match self {
            Session::Local(s) => s.fine_tune_status(id).await,
        }
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
        match self {
            Session::Local(s) => {
                s.eval_embeddings(source_id, embedding_table, golden_source, k, cohorts)
                    .await
            }
        }
    }

    /// Read back the persisted per-query eval records for a run.
    pub async fn eval_per_query(&self, eval_run_id: &str) -> Result<Vec<PerQueryEvalRecord>> {
        match self {
            Session::Local(s) => s.eval_per_query(eval_run_id).await,
        }
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
        match self {
            Session::Local(s) => {
                s.eval_inference(
                    model_id,
                    source_id,
                    columns,
                    task,
                    golden_source,
                    label_column,
                )
                .await
            }
        }
    }

    /// Compare multiple embedding tables side-by-side.
    pub async fn eval_compare(
        &self,
        embedding_tables: &[String],
        source_id: &str,
        golden_source: &str,
        k: usize,
    ) -> Result<CompareEvalReport> {
        match self {
            Session::Local(s) => {
                s.eval_compare(embedding_tables, source_id, golden_source, k)
                    .await
            }
        }
    }

    // --- mutable tables --------------------------------------------------

    /// Register a mutable companion table.
    pub async fn create_mutable_table(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        match self {
            Session::Local(s) => s.create_mutable_table(def).await,
        }
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        match self {
            Session::Local(s) => s.drop_mutable_table(id).await,
        }
    }

    // --- trigger ---------------------------------------------------------

    /// Register a topic (creates its backing table) for the trigger stream.
    pub async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        match self {
            Session::Local(s) => s.register_topic(topic).await,
        }
    }

    /// List every topic visible to the session's tenant.
    pub async fn list_topics(&self) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        match self {
            Session::Local(s) => s.list_topics().await,
        }
    }

    /// Drop a topic and its backing table.
    pub async fn drop_topic(&self, topic_id: TopicId) -> std::result::Result<(), TriggerError> {
        match self {
            Session::Local(s) => s.drop_topic(topic_id).await,
        }
    }

    /// Publish one batch to a topic under the session's tenant scope, returning
    /// the assigned offset.
    pub async fn publish(
        &self,
        topic: &TopicDefinition,
        batch: RecordBatch,
    ) -> std::result::Result<Offset, TriggerError> {
        match self {
            Session::Local(s) => s.publish(topic, batch).await,
        }
    }

    /// Subscribe to a topic, returning a transport-neutral stream of delivered
    /// batches. The stream replays from `from_offset` (or the live tail when
    /// `None`) and then tails live, scoped to the session's tenant.
    pub async fn subscribe(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<DeliveredBatch, TriggerError>> + Send>>,
        TriggerError,
    > {
        match self {
            Session::Local(s) => s.subscribe(topic, predicate, from_offset).await,
        }
    }

    // --- channels --------------------------------------------------------

    /// Register an evidence channel and its columns.
    pub async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        match self {
            Session::Local(s) => s.register_channel(spec).await,
        }
    }

    /// Append columns to an already-registered channel (append-only).
    pub async fn add_channel_columns(
        &self,
        channel: &jammi_db::ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        match self {
            Session::Local(s) => s.add_channel_columns(channel, new_columns).await,
        }
    }

    // --- tenant ----------------------------------------------------------

    /// Bind a tenant scope to this session (sticky form).
    pub fn bind_tenant(&self, t: TenantId) {
        match self {
            Session::Local(s) => s.bind_tenant(t),
        }
    }

    /// Clear the bound tenant.
    pub fn unbind_tenant(&self) {
        match self {
            Session::Local(s) => s.unbind_tenant(),
        }
    }

    /// The tenant currently bound, if any.
    pub fn tenant(&self) -> Option<TenantId> {
        match self {
            Session::Local(s) => s.tenant(),
        }
    }

    // --- audit -----------------------------------------------------------

    /// Sign and persist a batch of audit records; publishes them to the audit
    /// topic.
    pub async fn audit_log(
        &self,
        records: Vec<PerQueryAudit>,
    ) -> std::result::Result<(), jammi_db::AuditError> {
        match self {
            Session::Local(s) => s.audit_log(records).await,
        }
    }

    /// Fetch one audit record by query id (tenant-scoped).
    pub async fn audit_fetch_by_query_id(
        &self,
        query_id: uuid::Uuid,
    ) -> std::result::Result<Option<PerQueryAudit>, jammi_db::AuditError> {
        match self {
            Session::Local(s) => s.audit_fetch_by_query_id(query_id).await,
        }
    }

    /// Fetch the most recent audit records (tenant-scoped), newest first.
    pub async fn audit_fetch_recent(
        &self,
        limit: usize,
    ) -> std::result::Result<Vec<PerQueryAudit>, jammi_db::AuditError> {
        match self {
            Session::Local(s) => s.audit_fetch_recent(limit).await,
        }
    }
}

pub use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};

/// In-process [`Session`] transport. Owns an `Arc<InferenceSession>` because
/// the engine's `search` entry point is keyed on `Arc<Self>` (it hands the
/// shared session to the search plan); every other verb delegates straight
/// through.
pub struct LocalSession {
    engine: Arc<InferenceSession>,
}

impl LocalSession {
    /// Wrap an existing engine session.
    pub fn new(engine: Arc<InferenceSession>) -> Self {
        Self { engine }
    }

    /// The underlying engine session. The in-process affordances that are not
    /// on the transport surface (`with_tenant_scoped`, `with_admin_scope`,
    /// `ephemeral_session`, and the engine internals) are reached through this
    /// handle.
    pub fn engine(&self) -> &Arc<InferenceSession> {
        &self.engine
    }

    async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        self.engine
            .add_source(source_id, source_type, connection)
            .await
    }

    async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.engine.remove_source(source_id).await
    }

    async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.engine.sql(query).await
    }

    async fn generate_embeddings(
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

    async fn encode_query(
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

    async fn read_vectors(&self, table: &ResultTableRecord) -> Result<Vec<Vec<f32>>> {
        self.engine.read_vectors(table).await
    }

    async fn search(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        let SearchRequest {
            source_id,
            query,
            k,
            filter,
            select,
        } = request;

        let builder = match query {
            SearchQuery::Vector(vector) => self.engine.search(&source_id, vector, k).await?,
            SearchQuery::RowKey(row_key) => {
                self.engine.search_by_id(&source_id, &row_key, k).await?
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

    async fn infer(
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

    async fn fine_tune(
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

    async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        let record = self.engine.catalog().get_fine_tune_job(&id.0).await?;
        Ok(record.status)
    }

    async fn eval_embeddings(
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

    async fn eval_per_query(&self, eval_run_id: &str) -> Result<Vec<PerQueryEvalRecord>> {
        self.engine.eval_per_query(eval_run_id).await
    }

    async fn eval_inference(
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

    async fn eval_compare(
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

    async fn create_mutable_table(&self, def: MutableTableDefinition) -> Result<MutableTableId> {
        self.engine.create_mutable_table(def).await
    }

    async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        self.engine.drop_mutable_table(id).await
    }

    async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        self.engine.topic_repo().register_topic(topic).await
    }

    async fn list_topics(&self) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        self.engine
            .topic_repo()
            .list_topics(self.engine.tenant())
            .await
    }

    async fn drop_topic(&self, topic_id: TopicId) -> std::result::Result<(), TriggerError> {
        self.engine
            .topic_repo()
            .drop_topic(topic_id, self.engine.tenant())
            .await
    }

    async fn publish(
        &self,
        topic: &TopicDefinition,
        batch: RecordBatch,
    ) -> std::result::Result<Offset, TriggerError> {
        self.engine
            .publisher()
            .publish_scoped(topic, self.engine.tenant(), batch)
            .await
    }

    async fn subscribe(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<DeliveredBatch, TriggerError>> + Send>>,
        TriggerError,
    > {
        let subscription = self
            .engine
            .subscriber()
            .subscribe_scoped(topic, self.engine.tenant(), predicate, from_offset)
            .await?;
        Ok(Box::pin(subscription))
    }

    async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        self.engine.catalog().channels().register(spec).await
    }

    async fn add_channel_columns(
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

    fn bind_tenant(&self, t: TenantId) {
        self.engine.bind_tenant(t);
    }

    fn unbind_tenant(&self) {
        self.engine.unbind_tenant();
    }

    fn tenant(&self) -> Option<TenantId> {
        self.engine.tenant()
    }

    async fn audit_log(
        &self,
        records: Vec<PerQueryAudit>,
    ) -> std::result::Result<(), jammi_db::AuditError> {
        self.engine.audit().log(records).await
    }

    async fn audit_fetch_by_query_id(
        &self,
        query_id: uuid::Uuid,
    ) -> std::result::Result<Option<PerQueryAudit>, jammi_db::AuditError> {
        self.engine.audit().fetch_by_query_id(query_id).await
    }

    async fn audit_fetch_recent(
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
