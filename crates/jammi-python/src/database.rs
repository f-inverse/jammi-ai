use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::fine_tune::{BackboneDtype, EarlyStoppingMetric, FineTuneConfig, FineTuneMethod};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

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
    ///   target_modules=[] (projection-only LoRA; pass e.g. ["Wqkv","Wo"] for deep LoRA).
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
        if let Some(v) = lora_rank                  { cfg.lora_rank = v; }
        if let Some(v) = lora_alpha                 { cfg.lora_alpha = v; }
        if let Some(v) = lora_dropout               { cfg.lora_dropout = v; }
        if let Some(v) = learning_rate              { cfg.learning_rate = v; }
        if let Some(v) = epochs                     { cfg.epochs = v; }
        if let Some(v) = batch_size                 { cfg.batch_size = v; }
        if let Some(v) = max_seq_length             { cfg.max_seq_length = v; }
        if let Some(v) = validation_fraction        { cfg.validation_fraction = v; }
        if let Some(v) = early_stopping_patience    { cfg.early_stopping_patience = v; }
        if let Some(v) = warmup_steps               { cfg.warmup_steps = v; }
        if let Some(v) = gradient_accumulation_steps { cfg.gradient_accumulation_steps = v; }
        if let Some(m) = triplet_margin {
            cfg.embedding_loss = Some(jammi_ai::fine_tune::EmbeddingLoss::Triplet { margin: m });
        }
        if let Some(v) = target_modules             { cfg.target_modules = v; }
        if let Some(metric) = early_stopping_metric {
            cfg.early_stopping_metric = match metric {
                "train_loss" => EarlyStoppingMetric::TrainLoss,
                "val_loss"   => EarlyStoppingMetric::ValLoss,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown early_stopping_metric '{other}'. Use 'val_loss' or 'train_loss'."
                ))),
            };
        }
        if let Some(dtype_str) = backbone_dtype {
            cfg.backbone_dtype = match dtype_str {
                "f32"  => BackboneDtype::F32,
                "bf16" => BackboneDtype::BF16,
                "f16"  => BackboneDtype::F16,
                other  => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown backbone_dtype '{other}'. Use 'f32', 'bf16', or 'f16'."
                ))),
            };
        }
        if let Some(v) = weight_decay               { cfg.weight_decay = v; }
        if let Some(v) = max_grad_norm              { cfg.max_grad_norm = v; }

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
