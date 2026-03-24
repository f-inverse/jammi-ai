use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use jammi_engine::catalog::result_repo::ResultTableRecord;
use jammi_engine::config::JammiConfig;
use jammi_engine::error::{JammiError, Result};
use jammi_engine::session::JammiSession;
use jammi_engine::source::{SourceConnection, SourceType};
use jammi_engine::store::ResultStore;

use crate::concurrency::GpuScheduler;
use crate::inference::observer::InferenceObserver;
use crate::model::backend::DeviceConfig;
use crate::model::cache::ModelCache;
use crate::model::resolver::ModelResolver;
use crate::model::{ModelSource, ModelTask};
use crate::operator::inference_exec::InferenceExecBuilder;
use crate::pipeline::embedding::EmbeddingPipeline;
use crate::search::SearchBuilder;

/// An inference-capable session that wraps `JammiSession` with model loading
/// and inference execution. This is the primary entry point for CP2+.
pub struct InferenceSession {
    inner: JammiSession,
    model_cache: Arc<ModelCache>,
    result_store: Arc<ResultStore>,
    observer: Option<Arc<dyn InferenceObserver>>,
}

impl InferenceSession {
    /// Create a new session with model loading and inference capabilities.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        Self::with_observer(config, None).await
    }

    /// Create a new session with an optional inference observer.
    pub async fn with_observer(
        config: JammiConfig,
        observer: Option<Arc<dyn InferenceObserver>>,
    ) -> Result<Self> {
        let inner = JammiSession::new(config).await?;
        let catalog = Arc::clone(inner.catalog());
        let resolver = ModelResolver::new(catalog.clone())?;
        let device_config = DeviceConfig::from_config(inner.config());
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let model_cache = Arc::new(ModelCache::new(resolver, device_config, scheduler));
        let result_store = Arc::new(ResultStore::new(
            inner.config().artifact_dir.as_path(),
            catalog,
        )?);
        result_store.recover().await?;
        result_store.load_existing_tables(inner.context()).await?;

        Ok(Self {
            inner,
            model_cache,
            result_store,
            observer,
        })
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

    /// Execute a SQL query.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.inner.sql(query).await
    }

    /// Access the catalog.
    pub fn catalog(&self) -> &jammi_engine::catalog::Catalog {
        self.inner.catalog()
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
    pub fn inner_config(&self) -> &jammi_engine::config::JammiConfig {
        self.inner.config()
    }

    /// Access the inference observer.
    pub(crate) fn observer(&self) -> &Option<Arc<dyn InferenceObserver>> {
        &self.observer
    }

    /// Start a vector search query over an embedding table.
    pub async fn search(
        self: &Arc<Self>,
        source_id: &str,
        query: Vec<f32>,
        k: usize,
    ) -> Result<SearchBuilder> {
        SearchBuilder::new(Arc::clone(self), source_id, query, k, None).await
    }

    /// Encode a single text query into a vector using the given model.
    pub async fn encode_query(&self, model_id: &str, text: &str) -> Result<Vec<f32>> {
        let model_source = if let Some(path) = model_id.strip_prefix("local:") {
            ModelSource::local(std::path::PathBuf::from(path))
        } else {
            ModelSource::hf(model_id)
        };

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::Embedding, None)
            .await?;

        // Build a single-row input with the text
        let text_array = Arc::new(arrow::array::StringArray::from(vec![text.to_string()]))
            as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[text_array], ModelTask::Embedding)
            .map_err(|e| JammiError::Inference(format!("encode_query forward: {e}")))?;

        // Extract the first (and only) vector from the output
        let dim = output.shapes.first().map(|(_, c)| *c).unwrap_or(0);
        if output.float_outputs.is_empty() || output.float_outputs[0].is_empty() {
            return Err(JammiError::Inference("No embedding output".into()));
        }
        Ok(output.float_outputs[0][..dim].to_vec())
    }

    /// Generate embeddings for a source and persist to Jammi DB.
    pub async fn generate_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
    ) -> Result<ResultTableRecord> {
        EmbeddingPipeline::new(self, &self.result_store)
            .run(source_id, model_id, columns, key_column)
            .await
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
            let task_str = format!("{task:?}").to_lowercase();
            let table_info = self.result_store.create_table(
                source_id,
                &task_str,
                &source.to_string(),
                None,
                None,
                None,
            )?;
            let schema = batches[0].schema();
            let mut writer = jammi_engine::store::writer::ParquetResultWriter::new(
                &table_info.parquet_path,
                schema,
            )?;
            for batch in &batches {
                writer.write_batch(batch)?;
            }
            let row_count = writer.close()?;
            self.result_store
                .finalize(
                    self.inner.context(),
                    &table_info.table_name,
                    &table_info.parquet_path,
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
        format!("SELECT {select_list} FROM {source_id}.public.{table_name}")
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
}
