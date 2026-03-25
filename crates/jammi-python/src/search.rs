use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::search::SearchBuilder;

use crate::convert::batches_to_pyarrow;
use crate::error::to_pyerr;

/// Python SearchBuilder wrapping the Rust `SearchBuilder` for fluent chaining.
///
/// Uses `Option<SearchBuilder>` for move semantics — each method takes
/// ownership of the inner builder and replaces it with the result.
#[pyclass(name = "SearchBuilder")]
pub struct PySearchBuilder {
    pub(crate) inner: Option<SearchBuilder>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

impl PySearchBuilder {
    fn take_inner(&mut self) -> PyResult<SearchBuilder> {
        self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SearchBuilder already consumed by run()")
        })
    }
}

#[pymethods]
impl PySearchBuilder {
    /// Join with another registered source.
    /// `on` is `"left_col=right_col"`. `how` is `"inner"` or `"left"` (default).
    #[pyo3(signature = (source, *, on, how=None))]
    fn join(&mut self, source: &str, on: &str, how: Option<&str>) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(
            self.runtime
                .block_on(builder.join(source, on, how))
                .map_err(to_pyerr)?,
        );
        Ok(())
    }

    /// Annotate results with model inference.
    #[pyo3(signature = (*, model, task, columns))]
    fn annotate(&mut self, model: &str, task: &str, columns: Vec<String>) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(
            self.runtime
                .block_on(builder.annotate(model, task, &columns))
                .map_err(to_pyerr)?,
        );
        Ok(())
    }

    /// Filter results with a SQL WHERE clause predicate.
    fn filter(&mut self, predicate: &str) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.filter(predicate).map_err(to_pyerr)?);
        Ok(())
    }

    /// Sort results by a column.
    #[pyo3(signature = (column, *, descending=false))]
    fn sort(&mut self, column: &str, descending: bool) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.sort(column, descending).map_err(to_pyerr)?);
        Ok(())
    }

    /// Limit the number of results.
    fn limit(&mut self, n: usize) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.limit(n));
        Ok(())
    }

    /// Select specific columns.
    fn select(&mut self, columns: Vec<String>) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.select(&columns).map_err(to_pyerr)?);
        Ok(())
    }

    /// Execute the search and return a `pyarrow.Table`.
    fn run(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let builder = self.take_inner()?;
        let batches = self.runtime.block_on(builder.run()).map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }
}
