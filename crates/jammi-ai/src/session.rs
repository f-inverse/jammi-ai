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
use crate::eval::runner::EvalRunner;
use crate::fine_tune::job::FineTuneJob;
use crate::fine_tune::FineTuneConfig;
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
            Arc::clone(&catalog),
        )?);
        result_store.recover().await?;
        catalog.cleanup_stale_fine_tune_jobs()?;
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
        let model_source = ModelSource::parse(model_id);

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
        method: &str,
        _task: &str,
        config: Option<FineTuneConfig>,
    ) -> Result<FineTuneJob> {
        if method == "qlora" {
            return Err(JammiError::FineTune(
                "QLoRA is not supported. Use method='lora'.".into(),
            ));
        }
        if method != "lora" {
            return Err(JammiError::FineTune(format!(
                "Unknown fine-tuning method '{method}'. Supported: lora"
            )));
        }

        let config = config.unwrap_or_default();
        config.validate()?;
        let job_id = uuid::Uuid::new_v4().to_string();
        let output_model_id = format!("jammi:fine-tuned:{job_id}");

        // Parse model source to get the canonical name (what ModelCache uses for registration).
        let model_source = ModelSource::parse(base_model);
        let canonical_name = model_source.to_string();

        // Ensure base model is registered in catalog (FK constraint on fine_tune_jobs)
        if self.catalog().get_model(&canonical_name)?.is_none() {
            if let Err(e) = self.catalog().register_model(
                jammi_engine::catalog::model_repo::RegisterModelParams {
                    model_id: &canonical_name,
                    version: 1,
                    model_type: "embedding",
                    backend: "candle",
                    task: "embedding",
                    ..Default::default()
                },
            ) {
                tracing::error!(model_id = %canonical_name, error = %e, "Failed to register base model in catalog");
            }
        }

        // Persist job in catalog. FK references models.model_id PK = "{name}::{version}".
        let hyperparams = serde_json::to_string(&config)?;
        let loss_type = config
            .embedding_loss
            .map(|l| format!("{l:?}"))
            .unwrap_or_else(|| "auto".into());

        let base_model_pk = crate::model::to_catalog_pk(&canonical_name, 1);
        self.inner.catalog().create_fine_tune_job(
            &job_id,
            &base_model_pk,
            source,
            &loss_type,
            &hyperparams,
        )?;

        // Load training data from the source
        let table_name = self.find_table_name(source)?;
        let query = format!(
            "SELECT {} FROM {source}.public.{table_name}",
            columns
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let batches = self.sql(&query).await?;

        let data_loader = build_training_data_loader(&batches, columns)?;

        // Load base model to get hidden_size and pass to training loop
        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::Embedding, None)
            .await?;
        let base_model_arc = Arc::clone(&guard.model);
        let hidden_size = guard
            .model
            .embedding_dim()
            .ok_or_else(|| JammiError::FineTune("Base model does not support embeddings".into()))?;
        drop(guard);

        // Spawn training in a blocking task
        let catalog = Arc::clone(self.inner.catalog());
        let artifact_dir = self.inner.config().artifact_dir.clone();
        let job_id_clone = job_id.clone();
        let output_model_id_clone = output_model_id.clone();
        let base_model_str = base_model.to_string();

        tokio::task::spawn_blocking(move || {
            run_fine_tune_blocking(
                catalog,
                artifact_dir,
                job_id_clone,
                output_model_id_clone,
                base_model_str,
                config,
                data_loader,
                base_model_arc,
                hidden_size,
            )
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
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
    ) -> Result<serde_json::Value> {
        EvalRunner { session: self }
            .eval_embeddings(source_id, embedding_table, golden_source, k)
            .await
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
    ) -> Result<serde_json::Value> {
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
    ) -> Result<serde_json::Value> {
        EvalRunner { session: self }
            .eval_compare(embedding_tables, source_id, golden_source, k)
            .await
    }
}

// =========================================================================
// Fine-tuning helpers (outside impl block)
// =========================================================================

/// Build a TrainingDataLoader from query result batches.
fn build_training_data_loader(
    batches: &[RecordBatch],
    columns: &[String],
) -> Result<crate::fine_tune::data::TrainingDataLoader> {
    use arrow::array::StringArray;

    // Detect format from column names
    let col_names: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    let has_contrastive = col_names.contains(&"text_a")
        && col_names.contains(&"text_b")
        && col_names.contains(&"score");
    let has_triplet = col_names.contains(&"anchor")
        && col_names.contains(&"positive")
        && col_names.contains(&"negative");

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

            let a_arr = a_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| JammiError::FineTune("'text_a' is not a string column".into()))?;
            let b_arr = b_col
                .as_any()
                .downcast_ref::<StringArray>()
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
                rows.push((
                    a_arr.value(i).to_string(),
                    b_arr.value(i).to_string(),
                    score,
                ));
            }
        }
        Ok(crate::fine_tune::data::TrainingDataLoader::from_contrastive(rows))
    } else if has_triplet {
        let mut rows = Vec::new();
        for batch in batches {
            let anchor = batch
                .column_by_name("anchor")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'anchor' column".into()))?;
            let pos = batch
                .column_by_name("positive")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'positive' column".into()))?;
            let neg = batch
                .column_by_name("negative")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'negative' column".into()))?;

            for i in 0..batch.num_rows() {
                rows.push((
                    anchor.value(i).to_string(),
                    pos.value(i).to_string(),
                    neg.value(i).to_string(),
                ));
            }
        }
        Ok(crate::fine_tune::data::TrainingDataLoader::from_triplets(
            rows,
        ))
    } else {
        Err(JammiError::FineTune(format!(
            "Cannot detect training format from columns: {col_names:?}. \
             Expected contrastive (text_a, text_b, score) or triplet (anchor, positive, negative)."
        )))
    }
}

/// Run fine-tuning in a blocking context.
#[allow(clippy::too_many_arguments)]
fn run_fine_tune_blocking(
    catalog: Arc<jammi_engine::catalog::Catalog>,
    artifact_dir: std::path::PathBuf,
    job_id: String,
    output_model_id: String,
    base_model: String,
    config: FineTuneConfig,
    data_loader: crate::fine_tune::data::TrainingDataLoader,
    base_model_arc: Arc<crate::model::LoadedModel>,
    hidden_size: usize,
) -> Result<()> {
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = crate::fine_tune::lora::build_lora_projection(hidden_size, &config, &vb)?;

    let mut training_loop =
        crate::fine_tune::trainer::TrainingLoopBuilder::new(model, varmap, config)
            .job_id(job_id.clone())
            .catalog(Arc::clone(&catalog))
            .artifact_dir(artifact_dir.clone())
            .base_model(base_model_arc)
            .build()?;

    match training_loop.run(&data_loader) {
        Ok(_result) => {
            // Register the fine-tuned model in catalog
            if let Err(e) = catalog.set_fine_tune_output_model(&job_id, &output_model_id) {
                tracing::error!(job_id = %job_id, error = %e, "Failed to set fine-tune output model in catalog");
            }
            let adapter_dir = artifact_dir.join("models").join(&job_id);
            if let Err(e) =
                catalog.register_model(jammi_engine::catalog::model_repo::RegisterModelParams {
                    model_id: &output_model_id,
                    version: 1,
                    model_type: "fine-tuned",
                    backend: "candle",
                    task: "embedding",
                    base_model_id: Some(&base_model),
                    artifact_path: Some(adapter_dir.to_str().unwrap_or("")),
                    config_json: None,
                })
            {
                tracing::error!(job_id = %job_id, error = %e, "Failed to register fine-tuned model in catalog");
            }
        }
        Err(e) => {
            tracing::error!(job_id = %job_id, error = %e, "Fine-tune training failed");
        }
    }

    Ok(())
}
