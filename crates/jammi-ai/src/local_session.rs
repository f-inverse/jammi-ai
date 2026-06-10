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
//!   `Vec<RecordBatch>` out. The fluent [`crate::query::QueryBuilder`] is an
//!   internal mechanism `LocalSession` drives — never the return type.
//! * **Fine-tune** returns a job **id**; status is polled by id through
//!   [`Session::fine_tune_status`]. The in-process `TrainingJob`
//!   handle never escapes.
//! * **Audit** is three flat methods rather than a borrow-scoped handle.
//! * **Subscribe** returns a `Pin<Box<dyn Stream>>`, the transport-neutral
//!   streaming shape.

use std::pin::Pin;
#[cfg(feature = "local")]
use std::sync::Arc;

use arrow::array::RecordBatch;
use futures::Stream;

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_db::catalog::model_repo::ModelRecord;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::source_repo::SourceDescriptor;
use jammi_db::error::Result;
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::mutable::{MutableTableDefinition, MutableTableId};
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TriggerError};
use jammi_db::{ModelTask, PerQueryAudit, ServerInfo, TenantId, TopicId};

use crate::eval::{CompareEvalReport, EmbeddingEvalReport, EvalTask, InferenceEvalReport};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
#[cfg(feature = "local")]
use crate::model::ModelSource;
#[cfg(feature = "local")]
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
/// [`crate::query::QueryBuilder`] on the consumer surface: every knob the
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
/// Returned in place of the in-process `TrainingJob` handle
/// so the job is addressable across a transport boundary; poll it with
/// [`Session::fine_tune_status`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FineTuneJobId(pub String);

/// The consumer-facing session abstraction: a closed `enum` over transports.
///
/// Every method delegates to the underlying transport's implementation of the
/// same verb. The surface is the contract; the variant is the wiring.
pub enum Session {
    /// An in-process session driving a local [`InferenceSession`]. Present only
    /// under the default-on `local` feature; a thin remote-only (`wire`-only)
    /// build sees the one-arm `Remote` enum.
    #[cfg(feature = "local")]
    Local(LocalSession),
    /// A remote session driving an engine over the gRPC wire surface. Present
    /// only under the `wire` feature; an embedded build sees the one-arm enum.
    #[cfg(feature = "wire")]
    Remote(crate::remote_session::RemoteSession),
}

/// The one raw-vector verb a [`Session::Remote`] does not drive (`read_vectors`)
/// returns this — a real, propagated value, never a panic. The CLI never calls
/// it (it reads vectors through `search`), so it is honestly absent on the
/// remote transport rather than carrying a half-wired raw-vector read. Every
/// other verb on the [`Session`] surface is wire-reachable: `sql` rides the
/// Flight SQL lane ([`crate::RemoteSession::sql`]); the embeddings /
/// encode-query / search / add-source / remove-source verbs, the registry
/// listings (sources / models / channels / mutable tables), the tenant trio, the
/// `JammiError`-returning compute verbs (inference, eval, fine-tune,
/// mutable-table create/drop, channel register / add-columns), the topics +
/// subscribe-streaming surface, and the audit surface all ride the typed gRPC
/// surface. A caller that reaches the one unwired verb on a remote session gets
/// a typed error naming it — the truthful answer to "is this verb available on
/// this transport yet?", not a stand-in for a domain failure.
#[cfg(feature = "wire")]
fn remote_verb_pending(verb: &str) -> jammi_db::error::JammiError {
    jammi_db::error::JammiError::Other(format!(
        "{verb} is not yet available on the remote (gRPC) session transport"
    ))
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
            #[cfg(feature = "local")]
            Session::Local(s) => s.add_source(source_id, source_type, connection).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.add_source(source_id, source_type, connection).await,
        }
    }

    /// Remove a source and all associated state.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.remove_source(source_id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.remove_source(source_id).await,
        }
    }

    /// Describe every source registered to the session's tenant: each one's
    /// registry identity plus the embedding result tables produced from it.
    /// Registry introspection, not a SQL query.
    pub async fn list_sources(&self) -> Result<Vec<SourceDescriptor>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.list_sources().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.list_sources().await,
        }
    }

    /// Describe every model registered to the session's tenant. Registry
    /// introspection (the peer of [`Self::list_sources`] on the model catalog),
    /// not a SQL query.
    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.list_models().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.list_models().await,
        }
    }

    /// Describe one registered model by id, or `None` when no model with that
    /// id is visible to the session's tenant. Returns the latest registered
    /// version's record — the peer of [`Self::describe_source`] on the model
    /// catalog.
    pub async fn describe_model(&self, model_id: &str) -> Result<Option<ModelRecord>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.describe_model(model_id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.describe_model(model_id).await,
        }
    }

    /// Describe one registered source by id, or `None` when no source with
    /// that id is visible to the session's tenant. The embedding
    /// `status`/`row_count`/`dimensions` ride on the descriptor's result
    /// tables — the same source-of-truth `generate_embeddings` returns.
    pub async fn describe_source(&self, source_id: &str) -> Result<Option<SourceDescriptor>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.describe_source(source_id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.describe_source(source_id).await,
        }
    }

    /// The engine's capabilities handshake: version, compiled feature flags,
    /// and addressable storage backends. Tenant-agnostic; the same value for
    /// every session against a given build.
    pub async fn server_info(&self) -> Result<ServerInfo> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.server_info().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.server_info().await,
        }
    }

    // --- sql -------------------------------------------------------------

    /// Execute a SQL query.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.sql(query).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.sql(query).await,
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
            #[cfg(feature = "local")]
            Session::Local(s) => {
                s.generate_embeddings(source_id, model_id, columns, key_column, modality)
                    .await
            }
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
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
            #[cfg(feature = "local")]
            Session::Local(s) => s.encode_query(model_id, input, modality).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.encode_query(model_id, input, modality).await,
        }
    }

    /// Read the `vector` column of an embedding result table into one
    /// `Vec<f32>` per row.
    #[cfg_attr(not(feature = "local"), allow(unused_variables))]
    pub async fn read_vectors(&self, table: &ResultTableRecord) -> Result<Vec<Vec<f32>>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.read_vectors(table).await,
            #[cfg(feature = "wire")]
            Session::Remote(_) => Err(remote_verb_pending("read_vectors")),
        }
    }

    // --- search ----------------------------------------------------------

    /// Run a vector search and return the terminal hydrated batches.
    pub async fn search(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.search(request).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.search(request).await,
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
            #[cfg(feature = "local")]
            Session::Local(s) => {
                s.infer(source_id, model_id, task, content_columns, key_column)
                    .await
            }
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
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
            #[cfg(feature = "local")]
            Session::Local(s) => {
                s.fine_tune(source, base_model, columns, method, task, config)
                    .await
            }
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
                s.fine_tune(source, base_model, columns, method, task, config)
                    .await
            }
        }
    }

    /// Current status string for a fine-tune job, looked up by id.
    pub async fn fine_tune_status(&self, id: &FineTuneJobId) -> Result<String> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.fine_tune_status(id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.fine_tune_status(id).await,
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
            #[cfg(feature = "local")]
            Session::Local(s) => {
                s.eval_embeddings(source_id, embedding_table, golden_source, k, cohorts)
                    .await
            }
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
                s.eval_embeddings(source_id, embedding_table, golden_source, k, cohorts)
                    .await
            }
        }
    }

    /// Read back the persisted per-query eval records for a run.
    pub async fn eval_per_query(&self, eval_run_id: &str) -> Result<Vec<PerQueryEvalRecord>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.eval_per_query(eval_run_id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.eval_per_query(eval_run_id).await,
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
            #[cfg(feature = "local")]
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
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
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
            #[cfg(feature = "local")]
            Session::Local(s) => {
                s.eval_compare(embedding_tables, source_id, golden_source, k)
                    .await
            }
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
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
            #[cfg(feature = "local")]
            Session::Local(s) => s.create_mutable_table(def).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.create_mutable_table(def).await,
        }
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.drop_mutable_table(id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.drop_mutable_table(id).await,
        }
    }

    /// List every mutable companion table registered to the session's tenant.
    /// Registry introspection, not a SQL query.
    pub async fn list_mutable_tables(&self) -> Result<Vec<MutableTableDefinition>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.list_mutable_tables().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.list_mutable_tables().await,
        }
    }

    // --- trigger ---------------------------------------------------------

    /// Register a topic (creates its backing table) for the trigger stream.
    pub async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.register_topic(topic).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.register_topic(topic).await,
        }
    }

    /// List every topic visible to the session's tenant.
    pub async fn list_topics(&self) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.list_topics().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.list_topics().await,
        }
    }

    /// Drop a topic and its backing table.
    pub async fn drop_topic(&self, topic_id: TopicId) -> std::result::Result<(), TriggerError> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.drop_topic(topic_id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.drop_topic(topic_id).await,
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
            #[cfg(feature = "local")]
            Session::Local(s) => s.publish(topic, batch).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.publish(topic, batch).await,
        }
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
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => {
                s.subscribe(topic, predicate, from_offset, replay_only)
                    .await
            }
            #[cfg(feature = "wire")]
            Session::Remote(s) => {
                s.subscribe(topic, predicate, from_offset, replay_only)
                    .await
            }
        }
    }

    // --- channels --------------------------------------------------------

    /// Register an evidence channel and its columns.
    pub async fn register_channel(&self, spec: &ChannelSpec) -> Result<()> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.register_channel(spec).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.register_channel(spec).await,
        }
    }

    /// Append columns to an already-registered channel (append-only).
    pub async fn add_channel_columns(
        &self,
        channel: &jammi_db::ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.add_channel_columns(channel, new_columns).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.add_channel_columns(channel, new_columns).await,
        }
    }

    /// List every evidence channel registered to the session's tenant, ordered
    /// by `(priority, channel_id)`. Registry introspection, not a SQL query.
    pub async fn list_channels(&self) -> Result<Vec<ChannelSpec>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.list_channels().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.list_channels().await,
        }
    }

    // --- tenant ----------------------------------------------------------

    /// Bind a tenant scope to this session (sticky form).
    ///
    /// `async` because a network transport cannot honor a tenant round-trip
    /// without a blocking call it has no honest way to make synchronously; the
    /// in-process [`LocalSession`] simply `async`-wraps its sync engine call,
    /// while a remote variant maps it to a `SessionService.SetTenant` RPC.
    pub async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.bind_tenant(t).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.bind_tenant(t).await,
        }
    }

    /// Clear the bound tenant.
    pub async fn unbind_tenant(&self) -> Result<()> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.unbind_tenant().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.unbind_tenant().await,
        }
    }

    /// The tenant currently bound, if any.
    pub async fn tenant(&self) -> Result<Option<TenantId>> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.tenant().await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.tenant().await,
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
            #[cfg(feature = "local")]
            Session::Local(s) => s.audit_log(records).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.audit_log(records).await,
        }
    }

    /// Fetch one audit record by query id (tenant-scoped).
    pub async fn audit_fetch_by_query_id(
        &self,
        query_id: uuid::Uuid,
    ) -> std::result::Result<Option<PerQueryAudit>, jammi_db::AuditError> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.audit_fetch_by_query_id(query_id).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.audit_fetch_by_query_id(query_id).await,
        }
    }

    /// Fetch the most recent audit records (tenant-scoped), newest first.
    pub async fn audit_fetch_recent(
        &self,
        limit: usize,
    ) -> std::result::Result<Vec<PerQueryAudit>, jammi_db::AuditError> {
        match self {
            #[cfg(feature = "local")]
            Session::Local(s) => s.audit_fetch_recent(limit).await,
            #[cfg(feature = "wire")]
            Session::Remote(s) => s.audit_fetch_recent(limit).await,
        }
    }
}

pub use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};

/// In-process [`Session`] transport. Owns an `Arc<InferenceSession>` because
/// the engine's `search` entry point is keyed on `Arc<Self>` (it hands the
/// shared session to the search plan); every other verb delegates straight
/// through. Present only under the default-on `local` feature, alongside the
/// embedded engine it drives.
#[cfg(feature = "local")]
pub struct LocalSession {
    engine: Arc<InferenceSession>,
    /// The embedded training worker, present only on the front-door session
    /// (`Jammi::open`'s local arm) — the one drop point that owns the worker for
    /// the process's lifetime. Per-request wrappers (gRPC handlers, the Python
    /// `Database`'s internal `Session`) construct via [`Self::new`] and carry
    /// `None`, so a worker is not spawned per call. Dropping the front-door
    /// session stops the worker (RAII). Held for its `Drop`, not read.
    _worker: Option<crate::fine_tune::worker::EmbeddedWorker>,
}

#[cfg(feature = "local")]
impl LocalSession {
    /// Wrap an existing engine session without an embedded worker. Used by the
    /// per-request wrappers (gRPC handlers) and any caller that owns the training
    /// worker elsewhere (the server `train` tier; the Python `Database`). The
    /// transport-agnostic [`Session::fine_tune`] still submits jobs through this
    /// — the worker that runs them just lives elsewhere.
    pub fn new(engine: Arc<InferenceSession>) -> Self {
        Self {
            engine,
            _worker: None,
        }
    }

    /// Wrap an engine session and spawn the embedded
    /// [`crate::fine_tune::worker::TrainingWorker`] the resulting session owns.
    /// This is the SDK front-door form ([`crate::Jammi::open`]'s local arm): the
    /// embedded engine both submits training jobs and runs them, and the worker
    /// stops when this session drops (RAII). Must be called inside a tokio
    /// runtime context (the worker spawns a task).
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

    async fn list_sources(&self) -> Result<Vec<SourceDescriptor>> {
        self.engine.catalog().list_source_descriptors().await
    }

    async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        self.engine.catalog().list_models().await
    }

    async fn describe_model(&self, model_id: &str) -> Result<Option<ModelRecord>> {
        self.engine.catalog().get_model(model_id).await
    }

    async fn describe_source(&self, source_id: &str) -> Result<Option<SourceDescriptor>> {
        self.engine.catalog().describe_source(source_id).await
    }

    /// The engine's capabilities are a compile-time fact, so this reads them
    /// straight off [`ServerInfo::current`]; it is `async` only to match the
    /// transport-agnostic [`Session`] surface (the remote arm round-trips).
    async fn server_info(&self) -> Result<ServerInfo> {
        Ok(ServerInfo::current())
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
        let record = self.engine.catalog().get_training_job(&id.0).await?;
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

    async fn list_mutable_tables(&self) -> Result<Vec<MutableTableDefinition>> {
        Ok(self
            .engine
            .mutable_tables()
            .list(self.engine.tenant())
            .await?)
    }

    async fn register_topic(
        &self,
        topic: &TopicDefinition,
    ) -> std::result::Result<(), TriggerError> {
        // Register with *both* the broker driver and the catalog, mirroring the
        // Flight-SQL-DDL path: the catalog row is the system of record a later
        // lookup reads, but a `publish` resolves the topic against the broker —
        // so a catalog-only registration would make a publish to this topic fail
        // with `TopicNotFound`. The session always carries a broker (defaulting
        // to the in-memory broker), so this is total.
        self.engine.trigger_broker().register_topic(topic).await?;
        self.engine.topic_repo().register_topic(topic).await
    }

    async fn list_topics(&self) -> std::result::Result<Vec<TopicDefinition>, TriggerError> {
        self.engine
            .topic_repo()
            .list_topics(self.engine.tenant())
            .await
    }

    async fn drop_topic(&self, topic_id: TopicId) -> std::result::Result<(), TriggerError> {
        // Drop the catalog row (the system of record) first, then the broker
        // driver's view of the topic. The broker drop is best-effort: a driver
        // failure after the catalog row is gone leaks driver resources, not
        // catalog state, so it is surfaced via tracing rather than reverting the
        // successful catalog drop — mirroring the Flight-SQL-DDL path.
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

    async fn list_channels(&self) -> Result<Vec<ChannelSpec>> {
        self.engine.catalog().channels().list().await
    }

    /// `async` to match the [`Session`] surface (a remote transport's tenant
    /// trio is a round-trip); the in-process engine binding is synchronous, so
    /// this is an `async`-wrap with no inner await — infallible in-process, but
    /// the surface returns `Result` because a remote transport's bind can fail.
    async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.engine.bind_tenant(t);
        Ok(())
    }

    async fn unbind_tenant(&self) -> Result<()> {
        self.engine.unbind_tenant();
        Ok(())
    }

    async fn tenant(&self) -> Result<Option<TenantId>> {
        Ok(self.engine.tenant())
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
/// modality rather than silently using the first column. Used only by the
/// `local` [`LocalSession`] embedding path.
#[cfg(feature = "local")]
fn single_column<'a>(columns: &'a [String], modality: &str) -> Result<&'a str> {
    match columns {
        [single] => Ok(single.as_str()),
        _ => Err(jammi_db::error::JammiError::Inference(format!(
            "{modality} embeddings take exactly one content column, got {}",
            columns.len()
        ))),
    }
}
