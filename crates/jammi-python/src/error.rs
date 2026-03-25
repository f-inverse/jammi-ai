use pyo3::PyErr;

/// Convert a `JammiError` to a Python exception.
pub fn to_pyerr(e: jammi_engine::error::JammiError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}
