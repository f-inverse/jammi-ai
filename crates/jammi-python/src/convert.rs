use arrow::array::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyTable;

/// Convert `Vec<RecordBatch>` to a `pyarrow.Table` via zero-copy PyCapsule protocol.
///
/// Uses `pyo3-arrow`'s `PyTable` to export batches through the Arrow C Stream
/// Interface — no serialization, no data copies.
pub fn batches_to_pyarrow(py: Python<'_>, batches: &[RecordBatch]) -> PyResult<Py<PyAny>> {
    if batches.is_empty() {
        let pa = py.import("pyarrow")?;
        let empty = pyo3::types::PyDict::new(py);
        let table = pa.call_method1("table", (empty,))?;
        return Ok(table.unbind());
    }

    let schema = batches[0].schema();
    let table = PyTable::try_new(batches.to_vec(), schema)?;
    Ok(table.into_pyarrow(py)?.unbind())
}

/// Convert `serde_json::Value` to a Python dict via `json.loads()`.
pub fn json_to_pydict(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    let json_str = serde_json::to_string(value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize: {e}")))?;
    let json_mod = py.import("json")?;
    let dict = json_mod.call_method1("loads", (json_str,))?;
    Ok(dict.unbind())
}
