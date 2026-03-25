use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::fine_tune::FineTuneMethod;
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

    /// Generate embeddings for a registered source.
    #[pyo3(signature = (*, source, model, columns, key))]
    fn generate_embeddings(
        &self,
        source: &str,
        model: &str,
        columns: Vec<String>,
        key: &str,
    ) -> PyResult<()> {
        self.runtime
            .block_on(
                self.session
                    .generate_embeddings(source, model, &columns, key),
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
    #[pyo3(signature = (*, source, base_model, columns, method, task="embedding"))]
    fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: Vec<String>,
        method: &str,
        task: &str,
    ) -> PyResult<PyFineTuneJob> {
        let job = self
            .runtime
            .block_on(self.session.fine_tune(
                source,
                base_model,
                &columns,
                method.parse::<FineTuneMethod>().map_err(to_pyerr)?,
                task,
                None,
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
    fn encode_query(&self, model_id: &str, text: &str) -> PyResult<Vec<f32>> {
        self.runtime
            .block_on(self.session.encode_query(model_id, text))
            .map_err(to_pyerr)
    }

    /// Preload a model into the cache without running inference.
    fn preload_model(&self, model_id: &str) -> PyResult<()> {
        let source = ModelSource::parse(model_id);
        self.runtime
            .block_on(
                self.session
                    .model_cache()
                    .get_or_load(&source, ModelTask::Embedding, None),
            )
            .map_err(to_pyerr)?;
        Ok(())
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
