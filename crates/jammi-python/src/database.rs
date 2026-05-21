use std::str::FromStr;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use futures::StreamExt;
use pyo3::prelude::*;
use pyo3_arrow::PyTable;

use jammi_ai::fine_tune::{EarlyStoppingMetric, FineTuneConfig, FineTuneMethod};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use jammi_engine::trigger::{Offset, Predicate};
use jammi_lora::BackboneDtype;

use crate::convert::{batches_to_pyarrow, json_to_pydict};
use crate::error::to_pyerr;
use crate::job::PyFineTuneJob;
use crate::search::PySearchBuilder;

/// Python Database wrapping `Arc<InferenceSession>` with a shared tokio runtime.
#[pyclass(name = "Database")]
pub struct PyDatabase {
    pub(crate) session: Arc<InferenceSession>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyDatabase {
    /// Bind a tenant scope to this connection.
    ///
    /// Subsequent reads return rows whose `tenant_id` matches `tenant_id`
    /// plus globally-scoped (`tenant_id IS NULL`) rows; writes carry the
    /// bound tenant on every row. Pass an empty string to clear.
    fn with_tenant(&self, tenant_id: &str) -> PyResult<()> {
        if tenant_id.is_empty() {
            self.session.unbind_tenant();
            return Ok(());
        }
        let t = jammi_engine::TenantId::from_str(tenant_id).map_err(to_pyerr)?;
        self.session.bind_tenant(t);
        Ok(())
    }

    /// The tenant currently bound to this connection, or `None`.
    fn tenant(&self) -> Option<String> {
        self.session.tenant().map(|t| t.to_string())
    }

    /// Register a local file as a data source.
    #[pyo3(signature = (name, *, path, format))]
    fn add_source(&self, name: &str, path: &str, format: &str) -> PyResult<()> {
        let file_format = parse_file_format(format)?;
        let connection = SourceConnection::from_path(path, file_format);
        self.runtime
            .block_on(self.session.add_source(name, SourceType::Local, connection))
            .map_err(to_pyerr)
    }

    /// Execute a SQL query. Returns a `pyarrow.Table`.
    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<Py<PyAny>> {
        let batches = self
            .runtime
            .block_on(self.session.sql(query))
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }

    /// Generate text embeddings for a registered source.
    #[pyo3(signature = (*, source, model, columns, key))]
    fn generate_text_embeddings(
        &self,
        source: &str,
        model: &str,
        columns: Vec<String>,
        key: &str,
    ) -> PyResult<()> {
        self.runtime
            .block_on(
                self.session
                    .generate_text_embeddings(source, model, &columns, key),
            )
            .map_err(to_pyerr)?;
        Ok(())
    }

    /// Run inference. Returns a `pyarrow.Table`.
    #[pyo3(signature = (*, source, model, columns, task, key))]
    fn infer(
        &self,
        py: Python<'_>,
        source: &str,
        model: &str,
        columns: Vec<String>,
        task: &str,
        key: &str,
    ) -> PyResult<Py<PyAny>> {
        let model_source = ModelSource::parse(model);
        let model_task = parse_model_task(task)?;
        let batches = self
            .runtime
            .block_on(
                self.session
                    .infer(source, &model_source, model_task, &columns, key),
            )
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }

    /// List registered topic names visible to the current tenant binding.
    fn list_topics(&self) -> PyResult<Vec<String>> {
        let topic_repo = self.session.topic_repo();
        let tenant = self.session.tenant();
        let topics = self
            .runtime
            .block_on(topic_repo.list_topics(tenant))
            .map_err(|e| to_pyerr(jammi_engine::error::JammiError::Catalog(e.to_string())))?;
        Ok(topics.into_iter().map(|t| t.name).collect())
    }

    /// Publish one batch of rows to a topic. `batch` is a `pyarrow.Table`
    /// (zero-copy import via the Arrow C Stream Interface) whose schema
    /// must match the topic's. Returns the engine-assigned offset.
    #[pyo3(signature = (topic, *, batch))]
    fn publish_topic(&self, topic: &str, batch: PyTable) -> PyResult<u64> {
        let topic_repo = self.session.topic_repo();
        let tenant = self.session.tenant();
        let topic_def = self
            .runtime
            .block_on(topic_repo.lookup_by_name(topic, tenant))
            .map_err(|e| to_pyerr(jammi_engine::error::JammiError::Catalog(e.to_string())))?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("topic '{topic}' not found"))
            })?;
        let (batches, _schema) = batch.into_inner();
        // The publisher accepts one RecordBatch per call; concatenate the
        // streamed chunks so a multi-chunk pyarrow.Table publishes as one
        // logical event.
        let concatenated = if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            let schema = batches[0].schema();
            arrow::compute::concat_batches(&schema, batches.iter())
                .map_err(|e| to_pyerr(jammi_engine::error::JammiError::Catalog(e.to_string())))?
        };
        let publisher = self.session.publisher();
        let offset = self
            .runtime
            .block_on(publisher.publish(&topic_def, concatenated))
            .map_err(|e| to_pyerr(jammi_engine::error::JammiError::Catalog(e.to_string())))?;
        Ok(offset.value())
    }

    /// Open a subscription, collect up to `max_batches` matching batches
    /// (replay + live tail joined), then close. Returns the concatenated
    /// payload as a `pyarrow.Table`.
    ///
    /// Synchronous collect API: streaming iteration is left to the gRPC
    /// `TriggerService.Subscribe` surface where back-pressure flows
    /// through HTTP/2 naturally. This binding is the script-friendly
    /// equivalent for one-shot Python workflows.
    #[pyo3(signature = (topic, *, predicate=None, from_offset=None, max_batches=64))]
    fn subscribe_collect(
        &self,
        py: Python<'_>,
        topic: &str,
        predicate: Option<&str>,
        from_offset: Option<u64>,
        max_batches: usize,
    ) -> PyResult<Py<PyAny>> {
        let topic_repo = self.session.topic_repo();
        let tenant = self.session.tenant();
        let topic_def = self
            .runtime
            .block_on(topic_repo.lookup_by_name(topic, tenant))
            .map_err(|e| to_pyerr(jammi_engine::error::JammiError::Catalog(e.to_string())))?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("topic '{topic}' not found"))
            })?;
        let predicate = Predicate::from_sql(
            &SessionContext::new(),
            Arc::clone(&topic_def.schema),
            predicate.unwrap_or(""),
        )
        .map_err(|e| to_pyerr(jammi_engine::error::JammiError::Catalog(e.to_string())))?;
        let from = from_offset.map(|v| Offset::new(v, chrono::Utc::now()));
        let subscriber = self.session.subscriber();
        let collected: Vec<RecordBatch> = self
            .runtime
            .block_on(async move {
                let mut stream = subscriber
                    .subscribe(&topic_def, predicate, from)
                    .await
                    .map_err(|e| jammi_engine::error::JammiError::Catalog(e.to_string()))?;
                let mut out: Vec<RecordBatch> = Vec::new();
                while out.len() < max_batches {
                    match StreamExt::next(&mut stream).await {
                        Some(Ok(d)) => out.push(d.batch),
                        Some(Err(e)) => {
                            return Err(jammi_engine::error::JammiError::Catalog(e.to_string()))
                        }
                        None => break,
                    }
                }
                Ok::<_, jammi_engine::error::JammiError>(out)
            })
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &collected)
    }

    /// Register a new evidence-provenance channel. `columns` is a list of
    /// `(name, dtype)` tuples where `dtype` is one of `"Float32"`, `"Float64"`,
    /// `"Int32"`, `"Int64"`, `"Utf8"`, `"Boolean"`. Priority orders the
    /// channel against others on the same row when several contribute the
    /// same column. The channel id is unique across the catalog: passing
    /// one that already exists raises `RuntimeError` carrying
    /// `EvidenceChannel("channel '<id>': already exists")`.
    #[pyo3(signature = (channel_id, *, priority, columns))]
    fn register_channel(
        &self,
        channel_id: &str,
        priority: i32,
        columns: Vec<(String, String)>,
    ) -> PyResult<()> {
        let id = jammi_engine::ChannelId::new(channel_id).map_err(to_pyerr)?;
        let cols = parse_channel_columns(&columns)?;
        let spec = jammi_engine::catalog::channel_repo::ChannelSpec {
            id,
            priority,
            columns: cols,
        };
        self.runtime
            .block_on(self.session.catalog().channels().register(&spec))
            .map_err(to_pyerr)
    }

    /// Append columns to an already-registered channel. The append-only
    /// invariant is enforced: redeclaring an existing column with a
    /// different dtype raises `RuntimeError` carrying
    /// `EvidenceChannel("channel '<id>': column '<name>' was declared <X>,
    /// cannot redeclare as <Y>")`.
    #[pyo3(signature = (channel_id, *, columns))]
    fn add_channel_columns(
        &self,
        channel_id: &str,
        columns: Vec<(String, String)>,
    ) -> PyResult<()> {
        let id = jammi_engine::ChannelId::new(channel_id).map_err(to_pyerr)?;
        let cols = parse_channel_columns(&columns)?;
        self.runtime
            .block_on(self.session.catalog().channels().add_columns(&id, &cols))
            .map_err(to_pyerr)
    }

    /// Start a vector search. Returns a `SearchBuilder` for fluent chaining.
    #[pyo3(signature = (source, *, query, k))]
    fn search(&self, source: &str, query: Vec<f32>, k: usize) -> PyResult<PySearchBuilder> {
        let session_arc = Arc::clone(&self.session);
        let builder = self
            .runtime
            .block_on(session_arc.search(source, query, k))
            .map_err(to_pyerr)?;
        Ok(PySearchBuilder {
            inner: Some(builder),
            runtime: Arc::clone(&self.runtime),
        })
    }

    /// Start a fine-tuning job. Returns a `FineTuneJob` handle.
    ///
    /// All config kwargs are optional — omitting them uses the Jammi defaults:
    ///   lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
    ///   learning_rate=2e-4, epochs=3, batch_size=8, max_seq_length=512,
    ///   validation_fraction=0.10, early_stopping_patience=3,
    ///   warmup_steps=100, gradient_accumulation_steps=1,
    ///   triplet_margin=0.5 (embedding_loss="triplet" must be set for custom margin),
    ///   target_modules=[] trains a projection head over the frozen base model;
    ///     a non-empty list (e.g. ["Wqkv","Wo"]) injects LoRA into the encoder's
    ///     internal attention/FFN linears at the listed sites instead.
    ///   early_stopping_metric="val_loss" | "train_loss".
    ///     "train_loss" replicates train_embedding_model.py without --val-file:
    ///     set validation_fraction=0.0 and the full dataset trains without a held-out split.
    ///   backbone_dtype="f32" | "bf16" | "f16" — dtype for frozen backbone weights.
    ///     "bf16" cuts backbone VRAM by ~half; LoRA A/B always stay in f32.
    ///   weight_decay — AdamW L2 regularization. Default: 0.01 (matches train_embedding_model.py).
    ///   max_grad_norm — global gradient clipping norm. Default: 1.0. Pass 0.0 to disable.
    #[pyo3(signature = (
        *,
        source,
        base_model,
        columns,
        method,
        task = "embedding",
        lora_rank = None,
        lora_alpha = None,
        lora_dropout = None,
        learning_rate = None,
        epochs = None,
        batch_size = None,
        max_seq_length = None,
        validation_fraction = None,
        early_stopping_patience = None,
        warmup_steps = None,
        gradient_accumulation_steps = None,
        triplet_margin = None,
        target_modules = None,
        early_stopping_metric = None,
        backbone_dtype = None,
        weight_decay = None,
        max_grad_norm = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: Vec<String>,
        method: &str,
        task: &str,
        lora_rank: Option<usize>,
        lora_alpha: Option<f64>,
        lora_dropout: Option<f64>,
        learning_rate: Option<f64>,
        epochs: Option<usize>,
        batch_size: Option<usize>,
        max_seq_length: Option<usize>,
        validation_fraction: Option<f64>,
        early_stopping_patience: Option<usize>,
        warmup_steps: Option<usize>,
        gradient_accumulation_steps: Option<usize>,
        triplet_margin: Option<f64>,
        target_modules: Option<Vec<String>>,
        early_stopping_metric: Option<&str>,
        backbone_dtype: Option<&str>,
        weight_decay: Option<f64>,
        max_grad_norm: Option<f64>,
    ) -> PyResult<PyFineTuneJob> {
        let mut cfg = FineTuneConfig::default();
        if let Some(v) = lora_rank {
            cfg.lora_rank = v;
        }
        if let Some(v) = lora_alpha {
            cfg.lora_alpha = v;
        }
        if let Some(v) = lora_dropout {
            cfg.lora_dropout = v;
        }
        if let Some(v) = learning_rate {
            cfg.learning_rate = v;
        }
        if let Some(v) = epochs {
            cfg.epochs = v;
        }
        if let Some(v) = batch_size {
            cfg.batch_size = v;
        }
        if let Some(v) = max_seq_length {
            cfg.max_seq_length = v;
        }
        if let Some(v) = validation_fraction {
            cfg.validation_fraction = v;
        }
        if let Some(v) = early_stopping_patience {
            cfg.early_stopping_patience = v;
        }
        if let Some(v) = warmup_steps {
            cfg.warmup_steps = v;
        }
        if let Some(v) = gradient_accumulation_steps {
            cfg.gradient_accumulation_steps = v;
        }
        if let Some(m) = triplet_margin {
            cfg.embedding_loss = Some(jammi_ai::fine_tune::EmbeddingLoss::Triplet { margin: m });
        }
        if let Some(v) = target_modules {
            cfg.target_modules = v;
        }
        if let Some(metric) = early_stopping_metric {
            cfg.early_stopping_metric = match metric {
                "train_loss" => EarlyStoppingMetric::TrainLoss,
                "val_loss" => EarlyStoppingMetric::ValLoss,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown early_stopping_metric '{other}'. Use 'val_loss' or 'train_loss'."
                    )))
                }
            };
        }
        if let Some(dtype_str) = backbone_dtype {
            cfg.backbone_dtype = match dtype_str {
                "f32" => BackboneDtype::F32,
                "bf16" => BackboneDtype::BF16,
                "f16" => BackboneDtype::F16,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown backbone_dtype '{other}'. Use 'f32', 'bf16', or 'f16'."
                    )))
                }
            };
        }
        if let Some(v) = weight_decay {
            cfg.weight_decay = v;
        }
        if let Some(v) = max_grad_norm {
            cfg.max_grad_norm = v;
        }

        let job = self
            .runtime
            .block_on(self.session.fine_tune(
                source,
                base_model,
                &columns,
                method.parse::<FineTuneMethod>().map_err(to_pyerr)?,
                task,
                Some(cfg),
            ))
            .map_err(to_pyerr)?;
        Ok(PyFineTuneJob::new(job, Arc::clone(&self.runtime)))
    }

    /// Evaluate embedding quality. Returns a dict with metric keys.
    #[pyo3(signature = (*, source, golden_source, model=None, k=10))]
    fn eval_embeddings(
        &self,
        py: Python<'_>,
        source: &str,
        golden_source: &str,
        model: Option<&str>,
        k: usize,
    ) -> PyResult<Py<PyAny>> {
        let json = self
            .runtime
            .block_on(
                self.session
                    .eval_embeddings(source, model, golden_source, k),
            )
            .map_err(to_pyerr)?;
        json_to_pydict(py, &json)
    }

    /// Evaluate inference quality. Returns a dict with task-specific metrics.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, source, columns, task, golden_source, label_column))]
    fn eval_inference(
        &self,
        py: Python<'_>,
        model: &str,
        source: &str,
        columns: Vec<String>,
        task: &str,
        golden_source: &str,
        label_column: &str,
    ) -> PyResult<Py<PyAny>> {
        let eval_task = parse_eval_task(task)?;
        let json = self
            .runtime
            .block_on(self.session.eval_inference(
                model,
                source,
                &columns,
                eval_task,
                golden_source,
                label_column,
            ))
            .map_err(to_pyerr)?;
        json_to_pydict(py, &json)
    }

    /// Compare multiple embedding tables side-by-side.
    #[pyo3(signature = (*, embedding_tables, source, golden_source, k=10))]
    fn eval_compare(
        &self,
        py: Python<'_>,
        embedding_tables: Vec<String>,
        source: &str,
        golden_source: &str,
        k: usize,
    ) -> PyResult<Py<PyAny>> {
        let json = self
            .runtime
            .block_on(
                self.session
                    .eval_compare(&embedding_tables, source, golden_source, k),
            )
            .map_err(to_pyerr)?;
        json_to_pydict(py, &json)
    }

    /// Encode a text query into an embedding vector using the given model.
    fn encode_text_query(&self, model_id: &str, text: &str) -> PyResult<Vec<f32>> {
        self.runtime
            .block_on(self.session.encode_text_query(model_id, text))
            .map_err(to_pyerr)
    }

    /// Preload a model into the cache without running inference.
    fn preload_model(&self, model_id: &str) -> PyResult<()> {
        let source = ModelSource::parse(model_id);
        self.runtime
            .block_on(self.session.model_cache().get_or_load(
                &source,
                ModelTask::TextEmbedding,
                None,
            ))
            .map_err(to_pyerr)?;
        Ok(())
    }

    /// Generate image embeddings for a registered source.
    #[pyo3(signature = (*, source, model, image_column, key))]
    fn generate_image_embeddings(
        &self,
        source: &str,
        model: &str,
        image_column: &str,
        key: &str,
    ) -> PyResult<()> {
        self.runtime
            .block_on(
                self.session
                    .generate_image_embeddings(source, model, image_column, key),
            )
            .map_err(to_pyerr)?;
        Ok(())
    }

    /// Encode an image into an embedding vector using the given vision model.
    fn encode_image_query(&self, model_id: &str, image_bytes: &[u8]) -> PyResult<Vec<f32>> {
        self.runtime
            .block_on(self.session.encode_image_query(model_id, image_bytes))
            .map_err(to_pyerr)
    }
}

fn parse_file_format(s: &str) -> PyResult<FileFormat> {
    s.parse().map_err(to_pyerr)
}

fn parse_model_task(s: &str) -> PyResult<ModelTask> {
    s.parse().map_err(to_pyerr)
}

fn parse_eval_task(s: &str) -> PyResult<jammi_ai::eval::EvalTask> {
    s.parse().map_err(to_pyerr)
}

fn parse_channel_columns(
    columns: &[(String, String)],
) -> PyResult<Vec<jammi_engine::catalog::channel_repo::ChannelColumn>> {
    columns
        .iter()
        .map(|(name, dtype)| {
            let data_type =
                jammi_engine::catalog::channel_repo::ChannelColumnType::from_sql_str(dtype)
                    .map_err(to_pyerr)?;
            Ok(jammi_engine::catalog::channel_repo::ChannelColumn {
                name: name.clone(),
                data_type,
            })
        })
        .collect()
}
