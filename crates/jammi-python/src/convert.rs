use arrow::array::RecordBatch;
use jammi_db::source::SourceDescription;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_arrow::{PySchema, PyTable};
use serde::Serialize;

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

/// Serialize any `Serialize` value into a Python dict via JSON. The dict
/// preserves the type's nested structure (lists become Python lists, maps
/// become Python dicts, scalars become their Python equivalents).
pub fn serializable_to_pydict<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let json_str = serde_json::to_string(value)
        .map_err(|e| crate::error::client_error("BackendError", format!("JSON serialize: {e}")))?;
    let json_mod = py.import("json")?;
    let dict = json_mod.call_method1("loads", (json_str,))?;
    Ok(dict.unbind())
}

/// One source's description as the dict `describe_source` returns: the
/// descriptor's keys, plus `tables` — each table the source serves as
/// `{"schema": str, "table": str, "columns": pyarrow.Schema}`, in the
/// source's order.
pub fn source_description_to_pydict(
    py: Python<'_>,
    description: SourceDescription,
) -> PyResult<Py<PyAny>> {
    let dict = serializable_to_pydict(py, &description.descriptor)?;
    let tables = PyList::empty(py);
    for described in description.tables {
        let entry = PyDict::new(py);
        entry.set_item("schema", described.table.schema)?;
        entry.set_item("table", described.table.table)?;
        entry.set_item(
            "columns",
            PySchema::new(described.columns).into_pyarrow(py)?,
        )?;
        tables.append(entry)?;
    }
    dict.bind(py).set_item("tables", tables)?;
    Ok(dict)
}
