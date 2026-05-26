use pyo3::PyErr;

use jammi_db::error::JammiError;

/// Convert any error that maps into `JammiError` to a Python exception.
///
/// Accepting `Into<JammiError>` lets call sites surface typed engine errors
/// (`MutableTableError`, `TriggerError`, …) without re-wrapping each one in a
/// stringly-typed `JammiError::Catalog` first — that wrapping is what would
/// force Python tests to substring-match on error messages.
pub fn to_pyerr<E: Into<JammiError>>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.into().to_string())
}
