use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::JammiConfig;
use jammi_db::error::{JammiError, Result};
use jammi_db::session::JammiSession;
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::{ArtifactStore, ResultStore};

use crate::concurrency::GpuScheduler;
use crate::eval::runner::EvalRunner;
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
use crate::fine_tune::training_job::TrainingJob;
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::inference::observer::InferenceObserver;
use crate::model::backend::DeviceConfig;
use crate::model::cache::ModelCache;
use crate::model::resolver::ModelResolver;
use crate::model::{ModelSource, ModelTask};
use crate::operator::inference_exec::InferenceExecBuilder;
use crate::pipeline::embedding::EmbeddingPipeline;
use crate::query::QueryBuilder;
use jammi_db::cache::ann_cache::AnnCache;

/// An inference-capable session that wraps `JammiSession` with model loading
/// and inference execution. This is the primary entry point for CP2+.
pub struct InferenceSession {
    inner: Arc<JammiSession>,
    model_cache: Arc<ModelCache>,
    result_store: Arc<ResultStore>,
    artifact_store: Arc<ArtifactStore>,
    observer: Option<Arc<dyn InferenceObserver>>,
    ann_cache: Arc<AnnCache>,
    device_config: DeviceConfig,
    /// Registry of open ephemeral sessions, shared with the timeout scanner.
    ephemeral_sessions: jammi_db::ephemeral::ActiveSessions,
}

impl InferenceSession {
    /// Create a new session with model loading and inference capabilities.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        Self::with_observer(config, None).await
    }

    /// Build a session behind an `Arc` with the compound-query SQL functions
    /// registered (`annotate`, …). This is the canonical constructor for any
    /// long-lived shared session — the embedded `Database`, the OSS server, and
    /// the `Jammi::open` local arm — so the in-process `sql` surface and the
    /// Flight SQL lane both expose the same SQL functions. Sessions that never
    /// run compound SQL (short-lived CLI commands) can still use the plain
    /// [`Self::new`].
    pub async fn open(config: JammiConfig) -> Result<Arc<Self>> {
        let session = Arc::new(Self::new(config).await?);
        session.register_query_functions();
        Ok(session)
    }

    /// Create a new session with an optional inference observer.
    pub async fn with_observer(
        config: JammiConfig,
        observer: Option<Arc<dyn InferenceObserver>>,
    ) -> Result<Self> {
        let inner = JammiSession::new(config).await?;
        Self::wrap(inner, observer).await
    }

    /// Create a session whose trigger-stream surface is bound to a
    /// caller-supplied broker. Forwarded to
    /// [`jammi_db::session::JammiSession::with_broker`]; the
    /// `InferenceSession` adds model-loading, eval, and inference layers on
    /// top. Used by tests that need a broker with controlled behaviour, e.g.
    /// an [`jammi_db::trigger::InMemoryBroker`] armed with
    /// `trigger_failure_for_next_publish` to deterministically exercise
    /// publisher-failure paths.
    pub async fn with_broker(
        config: JammiConfig,
        trigger_broker: Arc<dyn jammi_db::trigger::TriggerBroker>,
    ) -> Result<Self> {
        let inner = JammiSession::with_broker(config, trigger_broker).await?;
        Self::wrap(inner, None).await
    }

    /// Create a session whose audit sign/verify path routes through a
    /// caller-supplied [`jammi_db::audit::SigningKeyStore`]. The catalog
    /// backend and trigger broker stay config-driven (the counterpart to
    /// [`Self::with_broker`]); forwarded to
    /// [`jammi_db::session::JammiSession::with_signing_key_store`]. Deployments
    /// whose audit master key lives behind a secrets adapter inject it here so
    /// both the engine's audit *sign* path (`scope.audit().log`) and any
    /// out-of-band verify path share one store.
    pub async fn with_signing_key_store(
        config: JammiConfig,
        signing_key_store: Arc<dyn jammi_db::audit::SigningKeyStore>,
    ) -> Result<Self> {
        let inner = JammiSession::with_signing_key_store(config, signing_key_store).await?;
        Self::wrap(inner, None).await
    }

    async fn wrap(
        inner: JammiSession,
        observer: Option<Arc<dyn InferenceObserver>>,
    ) -> Result<Self> {
        let inner = Arc::new(inner);
        let catalog = Arc::clone(inner.catalog());
        // The artifact store is built first so the resolver can reload
        // fine-tuned adapters through it: a fine-tuned model's catalog
        // `artifact_path` is an object-store prefix, fetched into a local cache
        // before candle loads it, so an adapter trained on one host serves on
        // another.
        let artifact_store = Arc::new(build_artifact_store(&inner)?);
        let resolver = ModelResolver::new(catalog.clone(), Arc::clone(&artifact_store))?;
        let device_config = DeviceConfig::from_config(inner.config());
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let model_cache = Arc::new(ModelCache::new(resolver, device_config.clone(), scheduler));
        let result_store = Arc::new(build_result_store(&inner, Arc::clone(&catalog))?);
        result_store.recover().await?;
        result_store.load_existing_tables(inner.context()).await?;

        let ann_cache_size = inner.config().cache.ann_cache_max_entries as u64;
        let ann_cache = Arc::new(AnnCache::new(ann_cache_size));

        Ok(Self {
            inner,
            model_cache,
            result_store,
            artifact_store,
            observer,
            ann_cache,
            device_config,
            ephemeral_sessions: jammi_db::ephemeral::ActiveSessions::new(),
        })
    }

    /// Register the engine's compound-query SQL functions on this session's
    /// `SessionContext`, so SQL — in-process (`sql`) and over the Flight SQL
    /// lane alike — can call them.
    ///
    /// This registers the `annotate(model, task, relation, key, col…)` table
    /// function (model inference as a relation) and the vector-aggregation
    /// UDAFs (`vector_mean`/`vector_sum`/`vector_max`, element-wise reduction
    /// over a group of fixed-width vectors). It must be called once per session,
    /// after the session is behind an `Arc`, because `annotate` holds a
    /// [`std::sync::Weak`] back-reference to the session it serves — weak to
    /// avoid the cycle the strong handle would form (the session owns the
    /// context the function registers on). The Flight SQL request path clones
    /// this context's state, so registering here makes every function reachable
    /// on every Flight SQL session too.
    pub fn register_query_functions(self: &Arc<Self>) {
        self.context().register_udtf(
            crate::query::AnnotateTableFunction::NAME,
            Arc::new(crate::query::AnnotateTableFunction::new(Arc::downgrade(
                self,
            ))),
        );
        crate::query::register_vector_agg_udafs(self.context());
    }

    /// Register a data source.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        self.inner
            .add_source(source_id, source_type, connection)
            .await
    }

    /// Remove a source and all associated state (result tables, disk files,
    /// ANN cache, DataFusion registration). Eval runs are preserved.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.inner.remove_source(source_id).await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(())
    }

    /// Execute a SQL query.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.inner.sql(query).await
    }

    /// Access the catalog.
    pub fn catalog(&self) -> &jammi_db::catalog::Catalog {
        self.inner.catalog()
    }

    /// The shared catalog handle behind an `Arc` — the form a [`TrainingJob`]
    /// handle clones to poll its job after the submitting call returns.
    pub(crate) fn catalog_arc(&self) -> &Arc<jammi_db::catalog::Catalog> {
        self.inner.catalog()
    }

    /// Access the topic-catalog repo (used by trigger-stream callers that
    /// do not want to go through Flight SQL DDL).
    pub fn topic_repo(&self) -> Arc<jammi_db::catalog::topic_repo::TopicRepo> {
        self.inner.topic_repo()
    }

    /// Access the trigger-stream publisher.
    pub fn publisher(&self) -> Arc<jammi_db::trigger::Publisher> {
        self.inner.publisher()
    }

    /// Access the trigger-stream subscriber.
    pub fn subscriber(&self) -> Arc<jammi_db::trigger::Subscriber> {
        self.inner.subscriber()
    }

    /// Access the trigger broker the session was constructed with.
    pub fn trigger_broker(&self) -> Arc<dyn jammi_db::trigger::TriggerBroker> {
        self.inner.trigger_broker()
    }

    /// Register a mutable companion table. After this returns the table is
    /// queryable as `mutable.public.<id>` in the same SQL surface that
    /// federates Parquet result tables and external sources.
    pub async fn create_mutable_table(
        &self,
        def: jammi_db::store::mutable::MutableTableDefinition,
    ) -> Result<jammi_db::store::mutable::MutableTableId> {
        self.inner.create_mutable_table(def).await
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(
        &self,
        id: &jammi_db::store::mutable::MutableTableId,
    ) -> Result<()> {
        self.inner.drop_mutable_table(id).await
    }

    /// Reference to the mutable-table registry.
    pub fn mutable_tables(&self) -> &jammi_db::source::mutable::MutableTableRegistry {
        self.inner.mutable_tables()
    }

    /// List every mutable companion table registered to the session's tenant.
    /// Registry introspection, not a SQL query.
    pub async fn list_mutable_tables(
        &self,
    ) -> Result<Vec<jammi_db::store::mutable::MutableTableDefinition>> {
        Ok(self
            .inner
            .mutable_tables()
            .list(self.inner.tenant())
            .await?)
    }

    /// Bind a tenant scope to this session. Subsequent reads/writes filter
    /// to `tenant_id = t OR tenant_id IS NULL`; writes record `tenant_id = t`.
    ///
    /// This is the sticky form: it mutates session-shared state. For
    /// concurrent gRPC request handlers on a shared `Arc<InferenceSession>`,
    /// prefer [`Self::with_tenant_scoped`].
    pub fn bind_tenant(&self, t: jammi_db::TenantId) {
        self.inner.bind_tenant(t);
    }

    /// Clear the bound tenant.
    pub fn unbind_tenant(&self) {
        self.inner.unbind_tenant();
    }

    /// Return the tenant currently bound, if any.
    pub fn tenant(&self) -> Option<jammi_db::TenantId> {
        self.inner.tenant()
    }

    /// Typed handle to the per-query audit primitive, scoped to this session's
    /// tenant. Delegates to [`jammi_db::session::JammiSession::audit`].
    pub fn audit(&self) -> jammi_db::AuditHandle<'_> {
        self.inner.audit()
    }

    /// Open an ephemeral, session-scoped storage context bound to the tenant
    /// currently set on this connection.
    ///
    /// Tables created through the returned [`jammi_db::EphemeralSession`] are
    /// auto-deleted when it ends (explicit `close`, `Drop`, or timeout), and
    /// every transition publishes to `jammi.audit.session_lifecycle.v1`. The
    /// session shares this connection's `JammiSession` (tenant binding, trigger
    /// broker, catalog), and registers with the connection's timeout-scanner
    /// registry — call [`Self::spawn_ephemeral_timeout_scanner`] once to enforce
    /// timeouts in-process.
    ///
    /// Returns [`jammi_db::EphemeralError::NoTenantBinding`] if no tenant is
    /// bound.
    pub async fn ephemeral_session(
        &self,
        timeout: std::time::Duration,
    ) -> std::result::Result<jammi_db::EphemeralSession, jammi_db::EphemeralError> {
        jammi_db::EphemeralSession::open(
            Arc::clone(&self.inner),
            timeout,
            self.ephemeral_sessions.clone(),
        )
        .await
    }

    /// Shared handle to the ephemeral-session registry the timeout scanner reads.
    pub fn ephemeral_sessions(&self) -> jammi_db::ephemeral::ActiveSessions {
        self.ephemeral_sessions.clone()
    }

    /// Spawn the in-process timeout scanner that force-closes ephemeral sessions
    /// past their deadline. Returns the task handle; the scanner runs until the
    /// handle is aborted or the runtime shuts down. Call once per connection.
    pub fn spawn_ephemeral_timeout_scanner(
        &self,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        jammi_db::ephemeral::spawn_timeout_scanner(
            Arc::clone(&self.inner),
            self.ephemeral_sessions.clone(),
            interval,
        )
    }

    /// Run `f` with `tenant` bound for the duration of the closure's future.
    ///
    /// The binding is installed as a Tokio task-local that shadows the
    /// session's sticky shared binding for the executing task only.
    /// Concurrent invocations from different tasks each see their own
    /// `tenant`; no race exists on the shared
    /// `Arc<RwLock<TenantContext>>` because no shared write happens in the
    /// scoped path.
    ///
    /// Delegates to [`jammi_db::session::JammiSession::with_tenant_scoped`];
    /// see that method for the design rationale (Option β: task-local
    /// override rather than per-call session rebuild).
    pub async fn with_tenant_scoped<'a, F, Fut, T>(&'a self, tenant: jammi_db::TenantId, f: F) -> T
    where
        F: FnOnce(jammi_db::TenantScope<'a>) -> Fut,
        Fut: std::future::Future<Output = T> + 'a,
    {
        self.inner.with_tenant_scoped(tenant, f).await
    }

    /// Run `f` with the tenant analyzer rule disabled for the duration of
    /// the closure's future.
    ///
    /// Cross-tenant administrative reads (server-startup recovery scans,
    /// background audit jobs) live here. The closure receives an
    /// [`jammi_db::AdminScope`] handle whose [`jammi_db::AdminScope::sql`] returns
    /// fully materialised batches; once the closure resolves, subsequent
    /// reads on the same session are tenant-filtered again.
    ///
    /// This surface is **not** exposed on the gRPC wire — `jammi-server`
    /// must invoke it only from in-process administrative code paths, not
    /// from a request handler.
    ///
    /// Delegates to [`jammi_db::session::JammiSession::with_admin_scope`];
    /// see that method for the safety contract.
    pub async fn with_admin_scope<'a, F, Fut, T>(&'a self, f: F) -> T
    where
        F: FnOnce(jammi_db::AdminScope<'a>) -> Fut,
        Fut: std::future::Future<Output = T> + 'a,
    {
        self.inner.with_admin_scope(f).await
    }

    /// Access the model cache.
    pub fn model_cache(&self) -> &Arc<ModelCache> {
        &self.model_cache
    }

    /// Access the result store.
    pub fn result_store(&self) -> Arc<ResultStore> {
        Arc::clone(&self.result_store)
    }

    /// Access the artifact store — the object-store surface model artifacts
    /// (fine-tune adapters, context-predictor weights) are written to and
    /// reloaded through, so a cross-host worker fleet shares trained models.
    pub fn artifact_store(&self) -> Arc<ArtifactStore> {
        Arc::clone(&self.artifact_store)
    }

    /// The device configuration the session resolves candle tensors onto — the
    /// GPU ordinal / CPU fallback every in-process training path builds its
    /// `VarMap` against.
    pub(crate) fn device_config(&self) -> &DeviceConfig {
        &self.device_config
    }

    /// The [`ComputeDevice`](jammi_db::store::manifest::ComputeDevice) this
    /// session effectively runs models on — the device-identity the
    /// materialization contract folds into every result table's definition hash,
    /// so a CPU and a CUDA run of the same model are not falsely reported as a
    /// `Match`.
    pub fn compute_device(&self) -> jammi_db::store::manifest::ComputeDevice {
        crate::model::backend::candle::effective_compute_device(&self.device_config)
    }

    /// Resolve a single member row's stored `vector` from an embedding result
    /// table by key, or `None` when no row matches — the per-member read the
    /// episodic context sampler builds its tensors from, reusing the engine's
    /// typed vector-by-key SQL path rather than a raw-vector verb.
    pub(crate) async fn read_vector_by_key(
        &self,
        table: &ResultTableRecord,
        row_key: &str,
    ) -> Result<Option<Vec<f32>>> {
        match self.inner.read_vector_by_key(table, row_key).await {
            Ok(v) => Ok(Some(v)),
            Err(JammiError::Catalog(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Access the DataFusion session context.
    pub fn context(&self) -> &datafusion::prelude::SessionContext {
        self.inner.context()
    }

    /// Access the engine configuration.
    pub fn inner_config(&self) -> &jammi_db::config::JammiConfig {
        self.inner.config()
    }

    /// Shared handle to the engine's tenant binding. The OSS server's
    /// Flight SQL `TenantBoundProvider` updates this for the duration of
    /// each query so the analyzer rule scopes rows to the bound tenant.
    pub fn tenant_binding_arc(&self) -> jammi_db::tenant_scope::TenantBinding {
        self.inner.tenant_binding_arc()
    }

    /// Access the ANN cache.
    pub fn ann_cache(&self) -> &Arc<AnnCache> {
        &self.ann_cache
    }

    /// Access the inference observer.
    pub(crate) fn observer(&self) -> &Option<Arc<dyn InferenceObserver>> {
        &self.observer
    }

    /// Start a vector-search-seeded compound query over an embedding table.
    ///
    /// Returns the fluent [`QueryBuilder`]: the first node is the ANN search,
    /// onto which `join` / `annotate` / `filter` / `select` / `sort` / `limit`
    /// compose. The bounded typed `search` verb (on [`crate::Session`]) is a
    /// thin wrapper over this — vector-search then optional `filter`/`select`
    /// then `run`.
    pub async fn search(
        self: &Arc<Self>,
        source_id: &str,
        query: Vec<f32>,
        k: usize,
        embedding_table: Option<&str>,
    ) -> Result<QueryBuilder> {
        QueryBuilder::new(Arc::clone(self), source_id, query, k, embedding_table).await
    }

    /// Start a search ranked by an existing row (query-by-example).
    ///
    /// Resolves `row_key`'s stored vector from the source's embedding table
    /// **inside the engine** and delegates to [`Self::search`]. The vector
    /// never crosses the API boundary — this is consistent with the engine's
    /// "no raw-vector reads" line while exposing the standard vector-search
    /// primitive ("rows like this row").
    ///
    /// `embedding_table` selects which table both supplies the example vector
    /// and is searched — the example and its neighbours come from the same
    /// table. `None` selects the source's most-recent ready table.
    pub async fn search_by_id(
        self: &Arc<Self>,
        source_id: &str,
        row_key: &str,
        k: usize,
        embedding_table: Option<&str>,
    ) -> Result<QueryBuilder> {
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;
        let query = self.inner.read_vector_by_key(&table, row_key).await?;
        self.search(source_id, query, k, embedding_table).await
    }

    /// Run a model over `columns` of an arbitrary input plan, appending the
    /// task's inference columns (the `annotate` operation).
    ///
    /// This is the single inference-over-a-relation operator. Both the fluent
    /// [`crate::query::QueryBuilder::annotate`] and the Flight-SQL `annotate`
    /// table function descend through it, so the in-process and remote compound
    /// surfaces run the *same* plan node rather than two reimplementations of
    /// "run inference over these columns".
    ///
    /// The output schema is the inference prefix (`_row_id`, `_source`,
    /// `_model`, `_status`, `_error`, `_latency_ms`) followed by the task's
    /// columns (e.g. a `vector` FixedSizeList for an embedding task). `key_column`
    /// names the input column carried through as `_row_id`.
    pub async fn annotate_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        model: &ModelSource,
        task: ModelTask,
        columns: &[String],
        key_column: &str,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let guard = self.model_cache.get_or_load(model, task, None).await?;
        let embedding_dim = guard.model.embedding_dim();
        let regression_form = guard.model.regression_form().cloned();
        drop(guard);

        let inference = InferenceExecBuilder::new(
            input,
            model.clone(),
            task,
            columns.to_vec(),
            key_column.to_string(),
            String::new(),
            Arc::clone(&self.model_cache),
        )
        .batch_size(self.inner.config().inference.batch_size)
        .observer(self.observer.clone())
        .embedding_dim(embedding_dim)
        .regression_form(regression_form)
        .build()?;

        Ok(Arc::new(inference))
    }

    /// Encode a single text query into a vector using the given model.
    pub async fn encode_text_query(&self, model_id: &str, text: &str) -> Result<Vec<f32>> {
        let model_source = ModelSource::parse(model_id);

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::TextEmbedding, None)
            .await?;

        // Build a single-row input with the text
        let text_array = Arc::new(arrow::array::StringArray::from(vec![text.to_string()]))
            as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[text_array], ModelTask::TextEmbedding)
            .map_err(|e| JammiError::Inference(format!("encode_query forward: {e}")))?;

        // Extract the first (and only) vector from the output
        let dim = output.shapes.first().map(|(_, c)| *c).unwrap_or(0);
        if output.float_outputs.is_empty() || output.float_outputs[0].is_empty() {
            return Err(JammiError::Inference("No embedding output".into()));
        }
        Ok(output.float_outputs[0][..dim].to_vec())
    }

    /// Generate embeddings for a source and persist to Jammi DB.
    /// Invalidates the ANN cache for this source after completion.
    pub async fn generate_text_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
    ) -> Result<ResultTableRecord> {
        let result = EmbeddingPipeline::new(self, &self.result_store, ModelTask::TextEmbedding)
            .run(source_id, model_id, columns, key_column)
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(result)
    }

    /// Read the `vector` column of an embedding result table into one
    /// `Vec<f32>` per row.
    ///
    /// Resolves the table's parquet through the underlying session's storage
    /// registry (so cloud credentials registered with the session are
    /// inherited) and surfaces [`JammiError::Schema`] when the column is not
    /// shaped `FixedSizeList<Float32>`. Delegates to
    /// [`jammi_db::session::JammiSession::read_vectors`].
    pub async fn read_vectors(&self, table: &ResultTableRecord) -> Result<Vec<Vec<f32>>> {
        self.inner.read_vectors(table).await
    }

    /// Generate image embeddings for a source and persist to Jammi DB.
    pub async fn generate_image_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        image_column: &str,
        key_column: &str,
    ) -> Result<ResultTableRecord> {
        let result = EmbeddingPipeline::new(self, &self.result_store, ModelTask::ImageEmbedding)
            .run(source_id, model_id, &[image_column.to_string()], key_column)
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(result)
    }

    /// Encode a single image into a vector using the given vision model.
    pub async fn encode_image_query(&self, model_id: &str, image_bytes: &[u8]) -> Result<Vec<f32>> {
        let model_source = ModelSource::parse(model_id);

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::ImageEmbedding, None)
            .await?;

        let binary_array =
            Arc::new(arrow::array::BinaryArray::from(vec![image_bytes])) as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[binary_array], ModelTask::ImageEmbedding)
            .map_err(|e| JammiError::Inference(format!("encode_image_query forward: {e}")))?;

        let dim = output.shapes.first().map(|(_, c)| *c).unwrap_or(0);
        if output.float_outputs.is_empty() || output.float_outputs[0].is_empty() {
            return Err(JammiError::Inference("No embedding output".into()));
        }
        Ok(output.float_outputs[0][..dim].to_vec())
    }

    /// Generate audio embeddings for a source and persist to Jammi DB.
    ///
    /// Peer of [`Self::generate_image_embeddings`]: scans `audio_column` (raw
    /// encoded audio bytes or file paths), decodes → resamples → log-mel →
    /// CLAP audio tower, and writes one L2-normalized vector per row. Reuses
    /// the modality-agnostic [`EmbeddingPipeline`] unchanged.
    pub async fn generate_audio_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        audio_column: &str,
        key_column: &str,
    ) -> Result<ResultTableRecord> {
        let result = EmbeddingPipeline::new(self, &self.result_store, ModelTask::AudioEmbedding)
            .run(source_id, model_id, &[audio_column.to_string()], key_column)
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(result)
    }

    /// Encode a single audio clip into a vector using the given audio model.
    ///
    /// Peer of [`Self::encode_image_query`]: `audio_bytes` is an encoded clip
    /// (WAV/FLAC/MP3/Ogg); the backend owns decode → resample → log-mel →
    /// forward, returning the L2-normalized shared-latent embedding.
    pub async fn encode_audio_query(&self, model_id: &str, audio_bytes: &[u8]) -> Result<Vec<f32>> {
        let model_source = ModelSource::parse(model_id);

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::AudioEmbedding, None)
            .await?;

        let binary_array =
            Arc::new(arrow::array::BinaryArray::from(vec![audio_bytes])) as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[binary_array], ModelTask::AudioEmbedding)
            .map_err(|e| JammiError::Inference(format!("encode_audio_query forward: {e}")))?;

        let dim = output.shapes.first().map(|(_, c)| *c).unwrap_or(0);
        if output.float_outputs.is_empty() || output.float_outputs[0].is_empty() {
            return Err(JammiError::Inference("No embedding output".into()));
        }
        Ok(output.float_outputs[0][..dim].to_vec())
    }

    /// TEST-ONLY non-vacuity seam for the regression surface. Loads a fresh,
    /// unshared copy of `model` (off the same resolve + backend load path serving
    /// uses), optionally zeroes its trained distribution head, runs a regression
    /// forward pass over `texts`, and returns the de-standardised served value of
    /// distribution column 0 (the Gaussian mean / the lowest quantile) per row.
    ///
    /// With `zero_head = false` this returns exactly what the head learned, so a
    /// test can confirm two text groups separate. With `zero_head = true` the head
    /// is collapsed to its zero-initialised base, which emits the scaler offset
    /// `μ_y` for every input — so the SAME separation assertion must FAIL,
    /// proving the trained test is non-vacuous (it measures learning, not the
    /// scaler centring at μ). Production never calls this; it mutates only the
    /// per-call owned model.
    #[doc(hidden)]
    pub async fn served_regression_col0_for_test(
        &self,
        model: &ModelSource,
        texts: &[String],
        zero_head: bool,
    ) -> Result<Vec<f32>> {
        use arrow::array::StringArray;

        let mut loaded = self
            .model_cache
            .load_owned_for_test(model, ModelTask::Regression)
            .await?;
        if zero_head {
            loaded.zero_distribution_head_for_test();
        }
        let col: arrow::array::ArrayRef = Arc::new(StringArray::from(texts.to_vec()));
        let output = loaded.forward(&[col], ModelTask::Regression)?;
        let (num_rows, head_width) = output.shapes[0];
        let flat = &output.float_outputs[0];
        let mut col0 = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            if output.row_status[row] {
                col0.push(flat[row * head_width]);
            }
        }
        Ok(col0)
    }

    /// Run inference on a registered source using a model.
    ///
    /// Scans the source, feeds `content_columns` through the model,
    /// and returns RecordBatches with prefix + task-specific columns.
    pub async fn infer(
        &self,
        source_id: &str,
        source: &ModelSource,
        task: ModelTask,
        content_columns: &[String],
        key_column: &str,
    ) -> Result<Vec<RecordBatch>> {
        // Validate content columns are not empty
        if content_columns.is_empty() {
            return Err(JammiError::Inference(
                "At least one content column is required".into(),
            ));
        }

        let table_name = self.find_table_name(source_id)?;
        let query = self.build_source_query(source_id, &table_name, key_column, content_columns);

        let df = self.inner.context().sql(&query).await.map_err(|e| {
            JammiError::Inference(format!("Failed to scan source '{source_id}': {e}"))
        })?;

        let input_plan = df
            .create_physical_plan()
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to create scan plan: {e}")))?;

        // Pre-load the model to get embedding dimensions for schema construction.
        // This also warms the cache so execute() hits a cache hit.
        let guard = self.model_cache.get_or_load(source, task, None).await?;
        let embedding_dim = guard.model.embedding_dim();
        let regression_form = guard.model.regression_form().cloned();
        let backend_kind = guard.model.backend_kind();
        drop(guard);

        // Wrap with InferenceExec
        let inference_exec = InferenceExecBuilder::new(
            input_plan,
            source.clone(),
            task,
            content_columns.to_vec(),
            key_column.to_string(),
            source_id.to_string(),
            Arc::clone(&self.model_cache),
        )
        .batch_size(self.inner.config().inference.batch_size)
        .observer(self.observer.clone())
        .embedding_dim(embedding_dim)
        .regression_form(regression_form)
        .build()?;

        // Execute and collect results
        let task_ctx = self.inner.context().task_ctx();
        let stream = inference_exec
            .execute(0, task_ctx)
            .map_err(|e| JammiError::Inference(format!("InferenceExec failed: {e}")))?;

        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to collect results: {e}")))?;

        // Persist results to Parquet
        if !batches.is_empty() {
            let table_info = self
                .result_store
                .create_table(
                    source_id,
                    task,
                    jammi_db::catalog::result_repo::ResultTableKind::Model,
                    None,
                    &source.to_string(),
                    None,
                    None,
                    None,
                )
                .await?;
            let schema = batches[0].schema();
            let mut writer = self
                .result_store
                .open_writer(&table_info.parquet_url, schema)
                .await?;
            for batch in &batches {
                writer.write_batch(batch).await?;
            }
            let row_count = writer.close().await?;

            // The materialization contract: the inference verb + its typed
            // parameters as the producing description, the engine/device/model
            // identity as the environment, and the source's read-time anchor as
            // the sole input. A registered source exposes no as-of/version
            // surface in open-core, so it is honestly recorded as
            // `UnpinnedAtInstant` rather than a fabricated pin.
            let descriptor = jammi_db::store::manifest::ProducingDescriptor::Inference {
                model_id: source.to_string(),
                task,
                source_id: source_id.to_string(),
                content_columns: content_columns.to_vec(),
                key_column: key_column.to_string(),
            };
            let env = jammi_db::store::manifest::MaterializationEnv::new(
                self.compute_device(),
                vec![jammi_db::store::manifest::ModelIdentity {
                    model_id: source.to_string(),
                    backend: backend_kind.to_string(),
                }],
            );
            let inputs = vec![jammi_db::store::manifest::InputAnchor::unpinned_at_instant(
                source_id,
                chrono::Utc::now().to_rfc3339(),
            )];
            self.result_store
                .finalize_with_manifest(
                    self.inner.context(),
                    &table_info.table_name,
                    &table_info.parquet_url,
                    row_count,
                    jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
                )
                .await?;
        }

        Ok(batches)
    }

    /// Build a SELECT query for the key + content columns from a source table.
    pub(crate) fn build_source_query(
        &self,
        source_id: &str,
        table_name: &str,
        key_column: &str,
        content_columns: &[String],
    ) -> String {
        let all_columns: Vec<&str> = std::iter::once(key_column)
            .chain(content_columns.iter().map(|s| s.as_str()))
            .collect();
        let select_list = all_columns
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "SELECT {select_list} FROM {}",
            source_relation(source_id, table_name)
        )
    }

    /// Find the first table name registered under a source catalog.
    pub(crate) fn find_table_name(&self, source_id: &str) -> Result<String> {
        let ctx = self.inner.context();
        let catalog = ctx
            .catalog(source_id)
            .ok_or_else(|| JammiError::Inference(format!("Source '{source_id}' not found")))?;
        let schema = catalog.schema("public").ok_or_else(|| {
            JammiError::Inference(format!("Schema 'public' not found in source '{source_id}'"))
        })?;
        let tables = schema.table_names();
        tables.into_iter().next().ok_or_else(|| {
            JammiError::Inference(format!("No tables found in source '{source_id}'"))
        })
    }

    /// Materialize the k-nearest-neighbour graph of a source's embedding table
    /// as a queryable edge `result_table`.
    ///
    /// This is for *global-structure* work — clustering, near-duplicate
    /// detection, connected components, graph-aware training-data generation —
    /// where the whole edge set is consumed as a durable artifact. For
    /// "neighbours of *these* rows", compose [`Self::search`] instead.
    ///
    /// The build resolves the input embedding table through the same
    /// tenant-scoped catalog path `search` uses: when a tenant is bound it runs
    /// inside that tenant's scope, so a caller cannot point the build at another
    /// tenant's table. The returned edge table is `kind = neighbor_graph`,
    /// derived from the resolved embedding table, with `src`/`dst` endpoints
    /// that join directly to source data on the key.
    ///
    /// The default driver is index-assisted and produces an *approximate*,
    /// *non-deterministic* graph; set `BuildNeighborGraph::exact` for a
    /// deterministic, complete one (gated by a row-count ceiling).
    pub async fn build_neighbor_graph(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        params: &crate::pipeline::neighbor_graph::BuildNeighborGraph,
    ) -> Result<ResultTableRecord> {
        match self.tenant() {
            // A bound tenant runs the build inside its scope, so the catalog
            // resolves only that tenant's embedding table — a caller cannot
            // point the build at another tenant's table.
            Some(tenant) => {
                self.with_tenant_scoped(tenant, |_scope| async move {
                    crate::pipeline::neighbor_graph::NeighborGraphPipeline::new(
                        self,
                        self.result_store.as_ref(),
                    )
                    .run(source_id, embedding_table, params)
                    .await
                })
                .await
            }
            None => {
                crate::pipeline::neighbor_graph::NeighborGraphPipeline::new(
                    self,
                    self.result_store.as_ref(),
                )
                .run(source_id, embedding_table, params)
                .await
            }
        }
    }

    /// Assemble a point-in-time-correct table: for each row of `spine`, attach
    /// the `facts` row valid as-of the spine row's temporal key, within each
    /// equality group. Writes a result table (carrying the materialization
    /// manifest) and returns its record. Left rows are always preserved;
    /// unmatched fact columns are null.
    ///
    /// `spine` and `facts` are registered source ids. Both are resolved through
    /// the session's tenant-scoped catalog: when a tenant is bound the join runs
    /// inside that tenant's scope, so a caller cannot point either side at
    /// another tenant's relation. The [`AsofJoinSpec`](crate::pipeline::asof::AsofJoinSpec)
    /// carries the four pinned knobs (direction, boundary, tolerance, tie-break)
    /// and the equality/temporal key roles.
    pub async fn asof_join(
        &self,
        spine: &str,
        facts: &str,
        spec: &crate::pipeline::asof::AsofJoinSpec,
    ) -> Result<ResultTableRecord> {
        match self.tenant() {
            Some(tenant) => {
                self.with_tenant_scoped(tenant, |_scope| async move {
                    crate::pipeline::asof::verb::run(self, spine, facts, spec).await
                })
                .await
            }
            None => crate::pipeline::asof::verb::run(self, spine, facts, spec).await,
        }
    }

    // =====================================================================
    // Fine-tuning
    // =====================================================================

    /// Submit a LoRA fine-tuning job on a registered source.
    ///
    /// Persists a self-describing [`TrainingSpec::FineTune`] into a `queued`
    /// catalog job and returns a [`TrainingJob`] handle immediately — the
    /// training runs later under a [`crate::fine_tune::worker::TrainingWorker`]
    /// that claims the job under a lease, reconstructs the data loader from the
    /// persisted source + columns, and trains while heartbeating. Call
    /// `job.wait().await` to block until a worker drives the job to a terminal
    /// state.
    pub async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<TrainingJob> {
        let config = config.unwrap_or_default();
        config.validate()?;

        let loss_type = fine_tune_loss_type(&config, task);
        let spec = TrainingSpec::FineTune {
            source: source.to_string(),
            columns: columns.to_vec(),
            method,
            task,
            common: TrainingCommon {
                base_model: base_model.to_string(),
                config: config.clone(),
            },
        };
        self.submit_fine_tune_spec(source, base_model, task, &config, &loss_type, spec)
            .await
    }

    /// Submit a job carrying one of the two LoRA fine-tune specs. Shared by the
    /// column-source [`Self::fine_tune`] and the graph [`Self::fine_tune_graph`]
    /// paths — the only thing that differs upstream is which spec variant is
    /// built. No data is read and no model is loaded here; the worker does both
    /// from the persisted spec.
    async fn submit_fine_tune_spec(
        &self,
        training_source: &str,
        base_model: &str,
        task: ModelTask,
        config: &FineTuneConfig,
        loss_type: &str,
        spec: TrainingSpec,
    ) -> Result<TrainingJob> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let output_model_id = format!("jammi:fine-tuned:{job_id}");

        // Parse model source to get the canonical name (what ModelCache uses for
        // registration).
        let model_source = ModelSource::parse(base_model);
        let canonical_name = model_source.to_string();

        // Ensure the base model is registered in the catalog (FK constraint on
        // training_jobs). The worker resolves the same row when it loads weights.
        if self.catalog().get_model(&canonical_name).await?.is_none() {
            if let Err(e) = self
                .catalog()
                .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                    model_id: &canonical_name,
                    version: 1,
                    model_type: "embedding",
                    backend: "candle",
                    task,
                    base_model_id: None,
                    artifact_path: None,
                    config_json: None,
                })
                .await
            {
                tracing::error!(model_id = %canonical_name, error = %e, "Failed to register base model in catalog");
            }
        }

        let hyperparams = serde_json::to_string(config)?;
        // The base-model FK must bind to the resolved row's catalog PK, not a
        // reconstructed `name::version`: a tenant fine-tuning a global base model
        // references the global (unqualified) PK, and one fine-tuning its own
        // model references its tenant-qualified PK — the resolved record's
        // `catalog_pk` carries whichever applies.
        let base_model_pk = self
            .catalog()
            .get_model(&canonical_name)
            .await?
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Base model '{canonical_name}' not registered in catalog"
                ))
            })?
            .catalog_pk;
        let spec_json = serde_json::to_string(&spec)?;
        self.inner
            .catalog()
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: &job_id,
                base_model_id: &base_model_pk,
                training_source,
                loss_type,
                hyperparams: &hyperparams,
                kind: spec.kind(),
                training_spec: &spec_json,
            })
            .await?;

        Ok(TrainingJob::new(
            job_id,
            "queued".into(),
            output_model_id,
            Arc::clone(self.inner.catalog()),
        ))
    }

    /// Graph-supervised fine-tune (S11): learn an embedding metric that encodes
    /// a graph's structure. Reads a node-text source and an edge source, samples
    /// the graph into `(anchor, positive, [hard_negative])` text pairs via biased
    /// random walks (node2vec), and drives the existing in-batch-negative
    /// (S10/MNRL) or triplet objective — **no new loss**.
    ///
    /// `node_source` supplies the text the encoder embeds, keyed by `id_column`,
    /// with the text in `text_column`. `edge_source` supplies directed edges
    /// (`src_column` → `dst_column`); endpoints join to `id_column`. Every
    /// endpoint must resolve to a node (the text-bearing precondition) or this is
    /// a typed error.
    ///
    /// `provenance` declares whether the edges are external/declared structure or
    /// S9-similarity edges — **the load-bearing distinction**: training on
    /// similarity edges largely re-learns the base metric (a degenerate feedback
    /// loop), so genuine gain comes from declared edges. Similarity-only edges
    /// are a weak bootstrap, never the sole supervision.
    pub async fn fine_tune_graph(
        &self,
        sources: &crate::fine_tune::graph_sampler::GraphFineTuneSources,
        base_model: &str,
        sample_config: crate::fine_tune::graph_sampler::GraphSampleConfig,
        config: Option<FineTuneConfig>,
    ) -> Result<TrainingJob> {
        let config = config.unwrap_or_default();
        config.validate()?;
        sample_config.validate()?;

        // The job record's `source` field records the node source — the model is
        // fine-tuned on that source's text, the edges only supervise the pairing.
        // The graph is read and re-sampled by the worker from the persisted
        // sources + seeded sample_config (deterministic), never from in-memory
        // batches carried across the submit boundary.
        let task = ModelTask::TextEmbedding;
        let loss_type = fine_tune_loss_type(&config, task);
        let spec = TrainingSpec::GraphFineTune {
            sources: sources.clone(),
            sample_config,
            common: TrainingCommon {
                base_model: base_model.to_string(),
                config: config.clone(),
            },
        };
        self.submit_fine_tune_spec(
            &sources.node_source,
            base_model,
            task,
            &config,
            &loss_type,
            spec,
        )
        .await
    }

    /// Run a decoded [`TrainingSpec`] on this session, dispatching each variant
    /// to its training entry point.
    ///
    /// The single spec→session seam: the gRPC `StartTraining` handler and the
    /// embedded binding both decode a `StartTrainingRequest` into a
    /// [`TrainingSpec`] and submit it here, so an identical decode yields an
    /// identical job on either transport. The dispatch lives once, beside the
    /// entry points it calls, rather than being re-written per transport.
    pub async fn run_training_spec(self: &Arc<Self>, spec: TrainingSpec) -> Result<TrainingJob> {
        match spec {
            TrainingSpec::FineTune {
                source,
                columns,
                method,
                task,
                common,
            } => {
                self.fine_tune(
                    &source,
                    &common.base_model,
                    &columns,
                    method,
                    task,
                    Some(common.config),
                )
                .await
            }
            TrainingSpec::GraphFineTune {
                sources,
                sample_config,
                common,
            } => {
                self.fine_tune_graph(
                    &sources,
                    &common.base_model,
                    sample_config,
                    Some(common.config),
                )
                .await
            }
            TrainingSpec::ContextPredictor {
                source,
                predictor_spec,
            } => self.train_context_predictor(&source, &predictor_spec).await,
        }
    }

    // =====================================================================
    // Evaluation
    // =====================================================================

    /// Evaluate embedding quality against golden relevance judgments.
    ///
    /// `cohorts` maps a golden-set `query_id` to an opaque `{key: value}`
    /// segment map persisted alongside that query's per-query metrics
    /// (`_jammi_eval_per_query`, spec J9). Pass an empty map for no tags.
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
        cohorts: &std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    ) -> Result<crate::eval::EmbeddingEvalReport> {
        EvalRunner { session: self }
            .eval_embeddings(source_id, embedding_table, golden_source, k, cohorts)
            .await
    }

    /// Read back the persisted per-query eval records for a run, scoped to the
    /// session tenant (spec J9). Returns Recall@{1,3,5,10}, MRR, nDCG,
    /// distance, and any cohort tags stored at eval time.
    pub async fn eval_per_query(
        &self,
        eval_run_id: &str,
    ) -> Result<Vec<jammi_db::catalog::eval_repo::PerQueryEvalRecord>> {
        self.catalog().get_eval_per_query(eval_run_id).await
    }

    /// Evaluate inference quality against golden labels.
    pub async fn eval_inference(
        &self,
        model_id: &str,
        source_id: &str,
        columns: &[String],
        task: crate::eval::EvalTask,
        golden_source: &str,
        label_column: &str,
    ) -> Result<crate::eval::InferenceEvalReport> {
        EvalRunner { session: self }
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
    ) -> Result<crate::eval::CompareEvalReport> {
        EvalRunner { session: self }
            .eval_compare(embedding_tables, source_id, golden_source, k)
            .await
    }

    /// Evaluate whether a predictor's uncertainty is honest (spec R2).
    ///
    /// `golden_source` is a held-out set pairing a predictive distribution with
    /// its realised `outcome`; `shape` selects the predictor's output family
    /// (parametric Gaussian or ensemble) and the columns read. `cohorts` maps a
    /// `record_id` to an opaque `{key: value}` segment map persisted alongside
    /// that record's per-record scores, the same way `eval_embeddings` cohorts
    /// work. Returns a report headlining a strictly proper score (CRPS) with the
    /// PIT-calibration diagnostic, sharpness, coverage, and per-cohort slices.
    pub async fn eval_calibration(
        &self,
        source_id: &str,
        golden_source: &str,
        shape: crate::eval::EvalCalibrationShape,
        cohorts: &std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    ) -> Result<crate::eval::CalibrationEvalReport> {
        EvalRunner { session: self }
            .eval_calibration(source_id, golden_source, shape, cohorts)
            .await
    }
}

/// Construct the session's [`ResultStore`], honouring `config.storage`.
///
/// When `storage.result_root` is set, result tables (Parquet + USearch
/// sidecars) are rooted there via [`ResultStore::with_root`], sharing the
/// session's [`jammi_db::storage::StorageRegistry`] so the deploy-wide
/// `[storage.cloud]` credentials resolve the root's driver. When it is unset,
/// result tables live on local disk under `{artifact_dir}/jammi_db/` —
/// today's behaviour. The catalog backend is independent of this choice.
fn build_result_store(
    inner: &JammiSession,
    catalog: Arc<jammi_db::catalog::Catalog>,
) -> Result<ResultStore> {
    let ann = inner.config().embedding.ann;
    match inner.config().storage.result_root.as_deref() {
        Some(root) => {
            let root = jammi_db::storage::StorageUrl::parse(root)?;
            ResultStore::with_root(root, inner.storage_registry(), catalog, ann)
        }
        None => ResultStore::new(inner.config().artifact_dir.as_path(), catalog, ann),
    }
}

/// Construct the session's [`ArtifactStore`], rooting model artifacts under the
/// same storage as result tables.
///
/// Model artifacts live under a `models/` sub-prefix of the configured
/// `storage.result_root` — one storage knob serves both result tables and
/// trained models, so an `s3://` / `r2://` root gives a worker fleet a shared
/// place to write and reload models across hosts. When `result_root` is unset,
/// artifacts root at `{artifact_dir}/jammi_db/models/`, mirroring the result
/// store's local fallback. The registry is shared with the session so cloud
/// credentials are registered once. The local fetch cache (where cloud
/// artifacts are materialised for candle to mmap) lives under
/// `{artifact_dir}/jammi_db/artifact_cache`.
fn build_artifact_store(inner: &JammiSession) -> Result<ArtifactStore> {
    let models_root = match inner.config().storage.result_root.as_deref() {
        Some(root) => {
            let trimmed = root.trim_end_matches('/');
            jammi_db::storage::StorageUrl::parse(&format!("{trimmed}/models"))?
        }
        None => {
            let local = inner.config().artifact_dir.join("jammi_db").join("models");
            jammi_db::storage::StorageUrl::parse(local.to_str().ok_or_else(|| {
                JammiError::Config("Non-UTF8 artifact_dir for artifact store".into())
            })?)?
        }
    };
    let cache_root = inner
        .config()
        .artifact_dir
        .join("jammi_db")
        .join("artifact_cache");
    ArtifactStore::with_root(models_root, inner.storage_registry(), cache_root)
}

/// The `loss_type` string persisted on a fine-tune job — a human-readable tag of
/// the objective selected by the task + config, recorded in the catalog
/// alongside the spec. The task selects the family (classification / regression /
/// embedding) and the config its specific loss.
fn fine_tune_loss_type(config: &FineTuneConfig, task: ModelTask) -> String {
    if task == ModelTask::Classification {
        config
            .classification_loss
            .map(|l| format!("{l:?}"))
            .unwrap_or_else(|| "CrossEntropy".into())
    } else if task == ModelTask::Regression {
        format!("{:?}", config.regression_loss.unwrap_or_default())
    } else {
        config
            .embedding_loss
            .map(|l| format!("{l:?}"))
            .unwrap_or_else(|| "auto".into())
    }
}
