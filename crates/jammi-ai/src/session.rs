use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::JammiConfig;
use jammi_db::error::{JammiError, Result};
use jammi_db::session::JammiSession;
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::ResultStore;

use crate::concurrency::GpuScheduler;
use crate::eval::runner::EvalRunner;
use crate::fine_tune::job::FineTuneJob;
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
        let resolver = ModelResolver::new(catalog.clone())?;
        let device_config = DeviceConfig::from_config(inner.config());
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let model_cache = Arc::new(ModelCache::new(resolver, device_config.clone(), scheduler));
        let result_store = Arc::new(build_result_store(&inner, Arc::clone(&catalog))?);
        result_store.recover().await?;
        catalog.cleanup_stale_fine_tune_jobs().await?;
        result_store.load_existing_tables(inner.context()).await?;

        let ann_cache_size = inner.config().cache.ann_cache_max_entries as u64;
        let ann_cache = Arc::new(AnnCache::new(ann_cache_size));

        Ok(Self {
            inner,
            model_cache,
            result_store,
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
    /// Currently this registers the `annotate(model, task, relation, key, col…)`
    /// table function (model inference as a relation). It must be called once
    /// per session, after the session is behind an `Arc`, because the function
    /// holds a [`std::sync::Weak`] back-reference to the session it serves —
    /// weak to avoid the cycle the strong handle would form (the session owns
    /// the context the function registers on). The Flight SQL request path
    /// clones this context's state, so registering here makes `annotate`
    /// reachable on every Flight SQL session too.
    pub fn register_query_functions(self: &Arc<Self>) {
        self.context().register_udtf(
            crate::query::AnnotateTableFunction::NAME,
            Arc::new(crate::query::AnnotateTableFunction::new(Arc::downgrade(
                self,
            ))),
        );
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
    ) -> Result<QueryBuilder> {
        QueryBuilder::new(Arc::clone(self), source_id, query, k, None).await
    }

    /// Start a search ranked by an existing row (query-by-example).
    ///
    /// Resolves `row_key`'s stored vector from the source's embedding table
    /// **inside the engine** and delegates to [`Self::search`]. The vector
    /// never crosses the API boundary — this is consistent with the engine's
    /// "no raw-vector reads" line while exposing the standard vector-search
    /// primitive ("rows like this row").
    pub async fn search_by_id(
        self: &Arc<Self>,
        source_id: &str,
        row_key: &str,
        k: usize,
    ) -> Result<QueryBuilder> {
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, None)
            .await?;
        let query = self.inner.read_vector_by_key(&table, row_key).await?;
        self.search(source_id, query, k).await
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
                .create_table(source_id, task, &source.to_string(), None, None, None)
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
            self.result_store
                .finalize(
                    self.inner.context(),
                    &table_info.table_name,
                    &table_info.parquet_url,
                    row_count,
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
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", ");
        format!("SELECT {select_list} FROM {source_id}.public.\"{table_name}\"")
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

    // =====================================================================
    // Fine-tuning
    // =====================================================================

    /// Start a LoRA fine-tuning job on a registered source.
    ///
    /// Returns a [`FineTuneJob`] handle that can be used to poll or wait for
    /// completion. The job runs synchronously in a blocking task spawned on
    /// tokio's blocking pool — call `job.wait().await` to block until done.
    pub async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        _method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<FineTuneJob> {
        let config = config.unwrap_or_default();
        config.validate()?;
        let job_id = uuid::Uuid::new_v4().to_string();
        let output_model_id = format!("jammi:fine-tuned:{job_id}");

        // Parse model source to get the canonical name (what ModelCache uses for registration).
        let model_source = ModelSource::parse(base_model);
        let canonical_name = model_source.to_string();

        // Ensure base model is registered in catalog (FK constraint on fine_tune_jobs)
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

        // Persist job in catalog. FK references models.model_id PK = "{name}::{version}".
        let hyperparams = serde_json::to_string(&config)?;
        let loss_type = if task == ModelTask::Classification {
            config
                .classification_loss
                .map(|l| format!("{l:?}"))
                .unwrap_or_else(|| "CrossEntropy".into())
        } else {
            config
                .embedding_loss
                .map(|l| format!("{l:?}"))
                .unwrap_or_else(|| "auto".into())
        };

        let base_model_pk = crate::model::to_catalog_pk(&canonical_name, 1);
        self.inner
            .catalog()
            .create_fine_tune_job(&job_id, &base_model_pk, source, &loss_type, &hyperparams)
            .await?;

        // Load training data from the source
        let table_name = self.find_table_name(source)?;
        let query = format!(
            "SELECT {} FROM {source}.public.\"{table_name}\"",
            columns
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let batches = self.sql(&query).await?;

        let data_loader = build_training_data_loader(&batches, columns, task)?;

        // Load the base model under the task being fine-tuned so the right
        // tower (text vs audio) is materialised and `embedding_dim()` reports
        // the shared-latent width the projection head must match.
        let guard = self
            .model_cache
            .get_or_load(&model_source, task, None)
            .await?;
        let base_model_arc = Arc::clone(&guard.model);
        let hidden_size = guard
            .model
            .embedding_dim()
            .ok_or_else(|| JammiError::FineTune("Base model does not support embeddings".into()))?;
        drop(guard);

        // Spawn training in a blocking task. The blocking thread carries no
        // task-local tenant override (that lives only on the request task that
        // called `fine_tune`), so its catalog writes would resolve `Unscoped`
        // and miss a tenant-scoped job's rows. Pin a catalog handle to the
        // tenant the job was created under so every background status update
        // and model registration stays in scope.
        let tenant = self.inner.catalog().current_tenant();
        let catalog = Arc::new(self.inner.catalog().pinned_to_tenant(tenant));
        let artifact_dir = self.inner.config().artifact_dir.clone();
        let job_id_clone = job_id.clone();
        let output_model_id_clone = output_model_id.clone();
        let base_model_str = base_model.to_string();
        let device_config = self.device_config.clone();

        let catalog_for_err = Arc::clone(&catalog);
        let job_id_for_err = job_id_clone.clone();
        tokio::task::spawn_blocking(move || {
            // Catch panics inside the training loop so a terminal status is
            // always recorded — otherwise `FineTuneJob::wait()` polls forever.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_fine_tune_blocking(
                    catalog,
                    artifact_dir,
                    job_id_clone,
                    output_model_id_clone,
                    base_model_str,
                    task,
                    config,
                    data_loader,
                    base_model_arc,
                    hidden_size,
                    device_config,
                )
            }));
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::error!(job_id = %job_id_for_err, error = %e, "Fine-tune blocking task failed");
                    record_failed(&catalog_for_err, &job_id_for_err, e.to_string());
                }
                Err(payload) => {
                    let msg = panic_message(payload.as_ref());
                    tracing::error!(job_id = %job_id_for_err, panic = %msg, "Fine-tune blocking task panicked");
                    record_failed(&catalog_for_err, &job_id_for_err, format!("Panic: {msg}"));
                }
            }
        });

        Ok(FineTuneJob::new(
            job_id,
            "queued".into(),
            output_model_id,
            Arc::clone(self.inner.catalog()),
        ))
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
    match inner.config().storage.result_root.as_deref() {
        Some(root) => {
            let root = jammi_db::storage::StorageUrl::parse(root)?;
            ResultStore::with_root(root, inner.storage_registry(), catalog)
        }
        None => ResultStore::new(inner.config().artifact_dir.as_path(), catalog),
    }
}

// =========================================================================
// Fine-tuning helpers (outside impl block)
// =========================================================================

/// Extract all string values from an Arrow column.
///
/// DataFusion 52+ returns Parquet string columns as `Utf8View` by default;
/// older versions returned `Utf8` or `LargeUtf8`. Dictionary-encoded variants
/// are also possible. Fast paths cover the three common types; the `cast`
/// fallback handles everything else.
fn extract_string_column(col: &dyn arrow::array::Array) -> Option<Vec<String>> {
    use arrow::array::{Array, LargeStringArray, StringArray, StringViewArray};
    use arrow::datatypes::DataType;

    // Fast path: Utf8View — DataFusion 52+ default for Parquet string columns
    if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    // Fast path: Utf8
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    // Fast path: LargeUtf8
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    // General fallback: cast to Utf8 (handles Dictionary variants, etc.)
    let casted = arrow::compute::cast(col, &DataType::Utf8).ok()?;
    let a = casted.as_any().downcast_ref::<StringArray>()?;
    Some((0..a.len()).map(|i| a.value(i).to_string()).collect())
}

/// Build a TrainingDataLoader from query result batches.
///
/// `task` selects how `anchor`/`positive`/`negative` triplet columns are
/// read: an audio embedding task reads them as encoded-audio bytes; every
/// other task reads them as text. The column names are identical across
/// modalities (the triplet shape is the same) — only the cell decoding
/// differs, so the caller's chosen task is the discriminator, not a parallel
/// set of column names.
fn build_training_data_loader(
    batches: &[RecordBatch],
    columns: &[String],
    task: ModelTask,
) -> Result<crate::fine_tune::data::TrainingDataLoader> {
    // Detect format from column names
    let col_names: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    let has_contrastive = col_names.contains(&"text_a")
        && col_names.contains(&"text_b")
        && col_names.contains(&"score");
    let has_triplet = col_names.contains(&"anchor")
        && col_names.contains(&"positive")
        && col_names.contains(&"negative");
    let has_classification = col_names.contains(&"text") && col_names.contains(&"label");

    if has_triplet && task == ModelTask::AudioEmbedding {
        return build_audio_triplet_loader(batches);
    }

    if has_contrastive {
        let mut rows = Vec::new();
        for batch in batches {
            let a_col = batch
                .column_by_name("text_a")
                .ok_or_else(|| JammiError::FineTune("Missing column 'text_a'".into()))?;
            let b_col = batch
                .column_by_name("text_b")
                .ok_or_else(|| JammiError::FineTune("Missing column 'text_b'".into()))?;
            let s_col = batch
                .column_by_name("score")
                .ok_or_else(|| JammiError::FineTune("Missing column 'score'".into()))?;

            let a_vals = extract_string_column(a_col.as_ref())
                .ok_or_else(|| JammiError::FineTune("'text_a' is not a string column".into()))?;
            let b_vals = extract_string_column(b_col.as_ref())
                .ok_or_else(|| JammiError::FineTune("'text_b' is not a string column".into()))?;
            let s_arr = s_col
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|arr| {
                    (0..arr.len())
                        .map(|i| arr.value(i) as f32)
                        .collect::<Vec<_>>()
                })
                .or_else(|| {
                    s_col
                        .as_any()
                        .downcast_ref::<arrow::array::Float32Array>()
                        .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                })
                .ok_or_else(|| JammiError::FineTune("'score' is not a float column".into()))?;

            for (i, &score) in s_arr.iter().enumerate().take(batch.num_rows()) {
                rows.push((a_vals[i].clone(), b_vals[i].clone(), score));
            }
        }
        Ok(crate::fine_tune::data::TrainingDataLoader::from_contrastive(rows))
    } else if has_triplet {
        let mut rows = Vec::new();
        for batch in batches {
            let schema_info = || {
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| format!("{}:{}", f.name(), f.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let anchor_vals = batch
                .column_by_name("anchor")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column. Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let pos_vals = batch
                .column_by_name("positive")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'positive' column. Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let neg_vals = batch
                .column_by_name("negative")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'negative' column. Batch schema: [{}]",
                        schema_info()
                    ))
                })?;

            for i in 0..batch.num_rows() {
                rows.push((
                    anchor_vals[i].clone(),
                    pos_vals[i].clone(),
                    neg_vals[i].clone(),
                ));
            }
        }
        Ok(crate::fine_tune::data::TrainingDataLoader::from_triplets(
            rows,
        ))
    } else if has_classification {
        let mut label_set = std::collections::BTreeSet::new();
        let mut rows = Vec::new();
        for batch in batches {
            let text_vals = batch
                .column_by_name("text")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'text' column".into()))?;
            let label_vals = batch
                .column_by_name("label")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'label' column".into()))?;
            for i in 0..batch.num_rows() {
                label_set.insert(label_vals[i].clone());
                rows.push((text_vals[i].clone(), label_vals[i].clone()));
            }
        }
        // Build label → index mapping
        let label_to_idx: std::collections::HashMap<String, u32> = label_set
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i as u32))
            .collect();
        let num_classes = label_to_idx.len();
        let indexed_rows: Vec<(String, u32)> = rows
            .into_iter()
            .map(|(text, label)| {
                let idx = label_to_idx[&label];
                (text, idx)
            })
            .collect();
        Ok(
            crate::fine_tune::data::TrainingDataLoader::from_classification(
                indexed_rows,
                num_classes,
            ),
        )
    } else {
        Err(JammiError::FineTune(format!(
            "Cannot detect training format from columns: {col_names:?}. \
             Expected contrastive (text_a, text_b, score), triplet (anchor, positive, negative), \
             or classification (text, label). For audio triplets, use the same \
             (anchor, positive, negative) columns with task=audio_embedding."
        )))
    }
}

/// Build an audio-triplet loader: read `anchor`/`positive`/`negative` as
/// encoded-audio byte columns. Shares the triplet column shape with the text
/// path; only the cell type differs (binary clips vs strings).
fn build_audio_triplet_loader(
    batches: &[RecordBatch],
) -> Result<crate::fine_tune::data::TrainingDataLoader> {
    let mut rows = Vec::new();
    for batch in batches {
        let schema_info = || {
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{}", f.name(), f.data_type()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let anchor_vals = batch
            .column_by_name("anchor")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'anchor' column for audio triplets. Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let pos_vals = batch
            .column_by_name("positive")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'positive' column for audio triplets. Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let neg_vals = batch
            .column_by_name("negative")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'negative' column for audio triplets. Batch schema: [{}]",
                    schema_info()
                ))
            })?;

        for i in 0..batch.num_rows() {
            rows.push((
                anchor_vals[i].clone(),
                pos_vals[i].clone(),
                neg_vals[i].clone(),
            ));
        }
    }
    Ok(crate::fine_tune::data::TrainingDataLoader::from_audio_triplets(rows))
}

/// Extract a binary column into owned byte vectors, accepting the Arrow binary
/// families DataFusion produces for an audio-bytes column
/// (`Binary`/`LargeBinary`/`BinaryView`). Returns `None` for any other type so
/// the caller can surface a typed schema error.
fn extract_binary_column(col: &dyn arrow::array::Array) -> Option<Vec<Vec<u8>>> {
    use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};

    if let Some(a) = col.as_any().downcast_ref::<BinaryArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<BinaryViewArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    None
}

/// Extract a human-readable message from a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<unknown panic payload>".into()
    }
}

/// Record a terminal `Failed` status for a fine-tune job, surfacing the cause
/// via the catalog metrics blob so callers polling `FineTuneJob::wait()` see
/// the failure instead of an indefinite `Running` state.
fn record_failed(catalog: &Arc<jammi_db::catalog::Catalog>, job_id: &str, msg: String) {
    let metrics = serde_json::json!({ "error_message": msg }).to_string();
    if let Err(e) = tokio::runtime::Handle::current().block_on(catalog.update_fine_tune_status(
        job_id,
        jammi_db::catalog::status::FineTuneJobStatus::Failed,
        Some(&metrics),
    )) {
        tracing::error!(job_id = %job_id, error = %e, "Failed to record terminal status");
    }
}

/// Run fine-tuning in a blocking context.
#[allow(clippy::too_many_arguments)]
fn run_fine_tune_blocking(
    catalog: Arc<jammi_db::catalog::Catalog>,
    artifact_dir: std::path::PathBuf,
    job_id: String,
    output_model_id: String,
    base_model: String,
    task: ModelTask,
    config: FineTuneConfig,
    data_loader: crate::fine_tune::data::TrainingDataLoader,
    base_model_arc: Arc<crate::model::LoadedModel>,
    hidden_size: usize,
    device_config: DeviceConfig,
) -> Result<()> {
    use candle_core::DType;
    use candle_nn::VarMap;

    let device = crate::model::backend::candle::select_device(&device_config);
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Choose between the two LoRA training shapes exactly once, based on
    // whether the caller requested LoRA inside the encoder (non-empty
    // `target_modules`) or a projection head over a frozen base model.
    // No half-built state — only the chosen variant is materialised.
    let target = if config.target_modules.is_empty() {
        let head = if task == ModelTask::Classification {
            let num_classes = match data_loader.format() {
                crate::fine_tune::data::TrainingFormat::Classification { num_classes } => {
                    num_classes
                }
                _ => {
                    return Err(JammiError::FineTune(
                        "Classification task requires classification training data format".into(),
                    ))
                }
            };
            crate::fine_tune::lora::build_classification_head(
                hidden_size,
                num_classes,
                &config,
                &vb,
            )?
        } else {
            crate::fine_tune::lora::build_projection_head(hidden_size, &config, &vb)?
        };
        crate::fine_tune::target::TrainingTarget::ProjectionHead { head }
    } else {
        let (encoder, adapter_cfg) =
            build_encoder_adapters(&base_model, &catalog, &config, &varmap, &device)?;
        crate::fine_tune::target::TrainingTarget::EncoderAdapters(Box::new(
            crate::fine_tune::target::EncoderAdaptersTarget {
                encoder,
                adapter_cfg,
            },
        ))
    };

    let mut training_loop =
        crate::fine_tune::trainer::TrainingLoopBuilder::new(target, varmap, config)
            .base_model(base_model_arc)
            .job_id(job_id.clone())
            .catalog(Arc::clone(&catalog))
            .artifact_dir(artifact_dir.clone())
            .output_model(crate::fine_tune::trainer::OutputModelHandle {
                output_model_id,
                base_model_id: base_model,
                task,
            })
            .device(device.clone())
            .build()?;

    // The trainer is responsible for registering the output model and
    // flipping the job's status to `Completed` atomically with respect to
    // the artifact write. Propagate any failure so the job lands in
    // `Failed` (via the trainer's failure paths) and the caller's
    // `wait()` observer sees a typed error rather than a wedged
    // `Running` row.
    training_loop.run(&data_loader)?;
    Ok(())
}

/// Construct an encoder-adapters target: load the frozen backbone weights
/// from the catalog artifact path, wrap the configured target modules with
/// LoRA, and return both the resulting encoder and the persisted adapter
/// metadata that pairs with the trained tensors on disk.
fn build_encoder_adapters(
    base_model_id: &str,
    catalog: &Arc<jammi_db::catalog::Catalog>,
    config: &FineTuneConfig,
    varmap: &candle_nn::VarMap,
    device: &candle_core::Device,
) -> Result<(jammi_encoders::AnyEncoder, jammi_lora::AdapterConfig)> {
    use std::path::Path;

    // Strip URI scheme prefixes (e.g. "hf://", "local:") so the catalog
    // lookup uses the bare model ID that was stored at download time.
    let catalog_model_id = base_model_id
        .strip_prefix("hf://")
        .or_else(|| base_model_id.strip_prefix("local:"))
        .unwrap_or(base_model_id);

    // Resolve the artifact directory from the catalog.
    // The catalog entry may have artifact_path = NULL (set before the model was
    // downloaded for FK-constraint purposes) or may point to a weights file
    // rather than a directory (older behavior in do_load).  Handle all cases.
    let model_record = tokio::runtime::Handle::current()
        .block_on(catalog.get_model(catalog_model_id))?
        .ok_or_else(|| {
            JammiError::FineTune(format!("Base model '{base_model_id}' not in catalog"))
        })?;

    let artifact_dir: std::path::PathBuf = match model_record.artifact_path.as_deref() {
        Some(p) if !p.is_empty() => {
            let path = std::path::PathBuf::from(p);
            // Stored path may be a file (old do_load behavior); use its parent as dir.
            if path.is_dir() {
                path
            } else {
                path.parent()
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                            "Cannot determine model dir from artifact_path '{p}'"
                        ))
                    })?
                    .to_path_buf()
            }
        }
        // artifact_path is NULL — the model was pre-registered for the FK
        // constraint before the weights were downloaded.  Fall back to the
        // HF hub local cache, which does not re-download if files are present.
        _ => {
            let is_hf = base_model_id.starts_with("hf://")
                || (!base_model_id.starts_with('/')
                    && !std::path::Path::new(base_model_id).exists());
            if is_hf {
                let api = hf_hub::api::sync::Api::new()
                    .map_err(|e| JammiError::FineTune(format!("HF hub init: {e}")))?;
                let repo = api.model(catalog_model_id.to_string());
                // repo.get() returns the cached path without re-downloading.
                let weights = repo.get("model.safetensors").map_err(|e| {
                    JammiError::FineTune(format!(
                        "Cannot locate '{catalog_model_id}' in HF hub cache: {e}"
                    ))
                })?;
                weights
                    .parent()
                    .ok_or_else(|| {
                        JammiError::FineTune(
                            "Cannot determine model dir from HF hub cache path".into(),
                        )
                    })?
                    .to_path_buf()
            } else {
                return Err(JammiError::FineTune(format!(
                    "Base model '{base_model_id}' has no artifact_path in catalog"
                )));
            }
        }
    };

    // Read config.json
    let config_path = artifact_dir.join("config.json");
    let model_config: serde_json::Value = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .ok_or_else(|| {
            JammiError::FineTune(format!(
                "Cannot read config.json for base model at {config_path:?}"
            ))
        })?;

    let model_type = model_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("bert");

    // Collect weights paths (standard single-shard safetensors)
    let weights_path = artifact_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(JammiError::FineTune(format!(
            "model.safetensors not found at {weights_path:?}"
        )));
    }

    let lora_dropout = if config.lora_dropout > 0.0 {
        Some(config.lora_dropout as f32)
    } else {
        None
    };

    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &config.target_modules,
        layers_to_transform: &config.layers_to_transform,
        lora_rank: config.lora_rank,
        lora_alpha: config.lora_alpha,
        use_rslora: config.use_rslora,
        lora_dropout,
        rank_pattern: &config.rank_pattern,
        init_mode: config.init_lora_weights,
    };

    let backbone_dtype: candle_core::DType = config.backbone_dtype.into();
    let adapter_cfg =
        jammi_lora::AdapterConfig::from_build(model_type, &lora, config.backbone_dtype);

    let weights_paths: Vec<&Path> = vec![weights_path.as_path()];

    let encoder = match model_type {
        "distilbert" => {
            let distilbert_config: jammi_encoders::DistilBertConfig =
                serde_json::from_value(model_config.clone()).map_err(|e| {
                    JammiError::FineTune(format!("Parse DistilBert config.json: {e}"))
                })?;
            jammi_encoders::AnyEncoder::DistilBert(
                jammi_encoders::DistilBert::builder()
                    .lora(lora)
                    .backbone_dtype(backbone_dtype)
                    .build(&weights_paths, &distilbert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build DistilBert encoder: {e}")))?,
            )
        }
        "modernbert" => {
            let modernbert_config: jammi_encoders::ModernBertConfig =
                serde_json::from_value(model_config.clone()).map_err(|e| {
                    JammiError::FineTune(format!("Parse ModernBert config.json: {e}"))
                })?;
            jammi_encoders::AnyEncoder::ModernBert(
                jammi_encoders::ModernBert::builder()
                    .lora(lora)
                    .backbone_dtype(backbone_dtype)
                    .build(&weights_paths, &modernbert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build ModernBert encoder: {e}")))?,
            )
        }
        _ => {
            let bert_config: jammi_encoders::BertConfig =
                serde_json::from_value(model_config.clone())
                    .map_err(|e| JammiError::FineTune(format!("Parse Bert config.json: {e}")))?;
            jammi_encoders::AnyEncoder::Bert(
                jammi_encoders::Bert::builder()
                    .lora(lora)
                    .backbone_dtype(backbone_dtype)
                    .build(&weights_paths, &bert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build Bert encoder: {e}")))?,
            )
        }
    };

    Ok((encoder, adapter_cfg))
}
