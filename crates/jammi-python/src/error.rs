use pyo3::PyErr;
use tonic::{Code, Status};

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

/// Convert a wire-decode [`Status`] to a Python exception.
///
/// The embedded training verbs decode their request through the shared
/// `jammi_ai::wire` seam, which reports a malformed or invalid request body as a
/// gRPC [`Status`] (the same one the server returns over the wire). A caller
/// error — an absent required field, a stringly-enum that does not parse, a body
/// that is not a valid request — arrives as [`Code::InvalidArgument`] and
/// surfaces as `ValueError`, matching how the in-process kwargs validators raise.
/// Any other code is a genuine fault and surfaces as `RuntimeError`.
pub fn status_to_pyerr(status: Status) -> PyErr {
    match status.code() {
        Code::InvalidArgument => {
            pyo3::exceptions::PyValueError::new_err(status.message().to_string())
        }
        _ => pyo3::exceptions::PyRuntimeError::new_err(status.message().to_string()),
    }
}
