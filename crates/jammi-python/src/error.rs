use pyo3::prelude::*;
use pyo3::PyErr;
use tonic::{Code, Status};

use jammi_db::error::JammiError;

/// Raise a `jammi.errors` exception class by name, carrying `message`.
///
/// The `JammiError` taxonomy is defined once, in the pure-Python `jammi`
/// (`jammi.errors`); the compiled engine RAISES those classes rather than
/// owning a parallel native taxonomy — so the embedded and remote transports map
/// a failure onto one class, and the dependency arrow stays native → client (the
/// engine imports the client, never the reverse). `jammi` is a hard
/// dependency of this wheel, so the import always resolves at runtime; a failure
/// to import it is a broken environment, surfaced loudly with the same message.
pub(crate) fn client_error(class: &str, message: String) -> PyErr {
    // The error converters have no `py` token in scope (they are called via
    // `.map_err(...)` from methods that do not thread one), so re-attach to the
    // interpreter here — the GIL is always held on the calling thread when a
    // pymethod maps its result, so this is a cheap re-borrow, not a fresh acquire.
    Python::attach(|py| {
        let built = py
            .import("jammi")
            .and_then(|m| m.getattr("errors"))
            .and_then(|errors| errors.getattr(class))
            .and_then(|cls| cls.call1((message.as_str(),)));
        match built {
            Ok(instance) => PyErr::from_value(instance),
            // A failure to import/construct the taxonomy class is a miswiring
            // (a renamed/removed class, a broken install) — surface it LOUDLY,
            // naming the class that could not be raised, rather than silently
            // downgrading to a bare `RuntimeError` that would slip past an
            // `except <TaxonomyClass>` and mask the real fault.
            Err(import_err) => pyo3::exceptions::PyRuntimeError::new_err(format!(
                "jammi taxonomy miswiring: could not raise \
                 jammi.errors.{class} ({import_err}); \
                 original error was: {message}"
            )),
        }
    })
}

/// The `jammi.errors` class an engine error raises in process — the class the
/// remote client raises for the same failure, so one `except` holds on both
/// transports. A typed refusal with its own leaf class raises it; every other
/// error raises the class of the status code the server sends for it
/// ([`jammi_wire::status_code`], the one classification both transports
/// share).
fn jammi_error_class(err: &JammiError) -> &'static str {
    match err {
        JammiError::FineTune(_) => "TrainingError",
        // The end the caller asked for, distinct from a fault — the class the
        // remote client raises for a `cancelled` job (`RemoteJob.wait`).
        JammiError::JobCancelled { .. } => "JobCancelled",
        JammiError::InvalidKey { .. } => "InvalidKey",
        JammiError::NonUniqueKey { .. } => "NonUniqueKey",
        JammiError::NoQueryEncoder { .. } => "NoQueryEncoder",
        JammiError::MissingManifest { .. } => "MissingManifest",
        JammiError::VersionUnavailable { .. } => "VersionUnavailable",
        JammiError::NotRefreshable { .. } => "NotRefreshable",
        JammiError::DefinitionDrift { .. } => "DefinitionDrift",
        JammiError::ModelNotFound { .. } => "ModelNotFound",
        JammiError::ModelReferenced { .. } => "ModelReferenced",
        other => status_class(jammi_wire::status_code(other)),
    }
}

/// Convert any error that maps into `JammiError` to a Python exception in the
/// `jammi.errors` taxonomy.
///
/// Accepting `Into<JammiError>` lets call sites surface typed engine errors
/// (`MutableTableError`, `TriggerError`, …) without re-wrapping each one in a
/// stringly-typed `JammiError::Catalog` first. The `JammiError` variant selects
/// the taxonomy class (see [`jammi_error_class`]) so a caller catches the same
/// class the remote client raises for the same failure — a failed training
/// `wait()` (`JammiError::FineTune`) is a `TrainingError`, a bad argument is an
/// `InvalidArgument`, a runtime/backend fault is a `BackendError`.
pub fn to_pyerr<E: Into<JammiError>>(e: E) -> PyErr {
    let err = e.into();
    let class = jammi_error_class(&err);
    client_error(class, err.to_string())
}

/// Convert a wire-decode [`Status`] to a Python exception in the taxonomy.
///
/// The embedded verbs decode their request through the shared `jammi_ai::wire`
/// seam, which reports a malformed or invalid request body as a gRPC [`Status`].
/// A caller error — an absent required field, a stringly-enum that does not
/// parse, a body that is not a valid request — arrives as
/// [`Code::InvalidArgument`] and surfaces as `InvalidArgument`, matching how the
/// in-process validators raise. Any other code is a genuine fault and surfaces
/// as `BackendError`.
pub fn status_to_pyerr(status: Status) -> PyErr {
    client_error(status_class(status.code()), status.message().to_string())
}

/// The `jammi.errors` class a gRPC [`Code`] maps to.
///
/// The same partition the pure-Python remote client applies to a live status
/// (`clients/python/jammi/_database.py::_rpc_to_jammi`): `INVALID_ARGUMENT` is a
/// caller error, everything else is a backend fault. Named as its own function
/// so [`jammi_error_class`] can be checked AGAINST it — an engine variant the
/// server surfaces under a given code must land on the class the remote client
/// raises for that code, rather than on a class chosen independently here.
fn status_class(code: Code) -> &'static str {
    match code {
        Code::InvalidArgument => "InvalidArgument",
        Code::NotFound => "NotFound",
        Code::AlreadyExists => "AlreadyExists",
        Code::FailedPrecondition => "FailedPrecondition",
        _ => "BackendError",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedded ⇄ remote parity: the class the embedded engine raises for
    /// `EmptyTrainingSet` is the class the remote transport raises for the same
    /// failure.
    ///
    /// The remote side of the equality is derived, not restated: the server
    /// sends this variant as `Code::InvalidArgument`
    /// ([`jammi_wire::status_code`]), and [`status_class`] is the remote
    /// client's code → class partition. Were the variant to fall through to
    /// `BackendError`, a caller catching `InvalidArgument` would see the
    /// empty-training-set refusal remotely and miss it embedded.
    #[test]
    fn empty_training_set_raises_the_class_the_remote_transport_raises() {
        let err = JammiError::EmptyTrainingSet {
            source_query: "SELECT text, label FROM reviews.public.rows".to_string(),
        };
        assert_eq!(
            jammi_error_class(&err),
            status_class(Code::InvalidArgument),
            "the embedded class for an empty training set must equal the class \
             the remote client raises for the INVALID_ARGUMENT the server sends",
        );
    }

    /// The residual bucket is still the residual bucket: a fault that is not a
    /// caller error keeps mapping to `BackendError`, so the arm above narrows
    /// exactly one variant rather than widening the caller-error class.
    #[test]
    fn a_non_caller_fault_still_classifies_as_a_backend_error() {
        assert_eq!(
            jammi_error_class(&JammiError::Other("disk on fire".to_string())),
            status_class(Code::Internal),
        );
    }
}
