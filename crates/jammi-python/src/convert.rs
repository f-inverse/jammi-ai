use arrow::array::RecordBatch;
use arrow::ipc::writer::StreamWriter;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Convert `Vec<RecordBatch>` to a `pyarrow.Table` via IPC serialization.
///
/// Avoids pyo3-arrow version coupling. Serializes to IPC stream bytes
/// in Rust, then deserializes via `pyarrow.ipc.open_stream()` in Python.
pub fn batches_to_pyarrow(py: Python<'_>, batches: &[RecordBatch]) -> PyResult<Py<PyAny>> {
    let pa = py.import("pyarrow")?;
    let ipc_mod = py.import("pyarrow.ipc")?;

    if batches.is_empty() {
        let empty = pyo3::types::PyDict::new(py);
        let table = pa.call_method1("table", (empty,))?;
        return Ok(table.unbind());
    }

    let schema = batches[0].schema();
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("IPC write error: {e}"))
        })?;
        for batch in batches {
            writer.write(batch).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("IPC batch write: {e}"))
            })?;
        }
        writer
            .finish()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC finish: {e}")))?;
    }

    let py_bytes = PyBytes::new(py, &buf);
    let reader = ipc_mod.call_method1("open_stream", (py_bytes,))?;
    let table = reader.call_method0("read_all")?;
    Ok(table.unbind())
}

/// Convert `serde_json::Value` to a Python dict via `json.loads()`.
pub fn json_to_pydict(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    let json_str = serde_json::to_string(value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize: {e}")))?;
    let json_mod = py.import("json")?;
    let dict = json_mod.call_method1("loads", (json_str,))?;
    Ok(dict.unbind())
}
