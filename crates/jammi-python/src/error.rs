use pyo3::prelude::*;
use pyo3::PyErr;
use tonic::{Code, Status};

use jammi_db::catalog::channel_repo::ChannelCatalogError;
use jammi_db::error::JammiError;
use jammi_db::store::mutable::MutableTableError;

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

/// Classify a [`JammiError`] onto the `jammi.errors` class name it maps to.
///
/// The partition mirrors the server's gRPC status mapping (`map_engine_error`)
/// so the embedded and remote transports agree on which failures are caller
/// errors: the variants the server surfaces as `InvalidArgument` become
/// `InvalidArgument`; a failed training job (`FineTune`) becomes `TrainingError`;
/// everything else — a transport/runtime/backend fault — becomes `BackendError`.
fn jammi_error_class(err: &JammiError) -> &'static str {
    match err {
        JammiError::FineTune(_) => "TrainingError",
        JammiError::Config(_)
        | JammiError::Source { .. }
        | JammiError::Model { .. }
        | JammiError::Tenant(_)
        | JammiError::Schema { .. }
        | JammiError::Eval(_) => "InvalidArgument",
        // The mutable-table / channel-catalog kinds carry validation-shaped
        // variants that are caller errors (`InvalidArgument`); their remaining
        // variants (NotFound / AlreadyExists / conflict / backend) are not, and
        // fall through to `BackendError`. Flattened into the outer match so the
        // caller-error variants are one arm each.
        JammiError::MutableTable(
            MutableTableError::InvalidId(_)
            | MutableTableError::Schema(_)
            | MutableTableError::MissingPrimaryKey(_)
            | MutableTableError::ReservedColumn(_)
            | MutableTableError::NoOrderColumn,
        ) => "InvalidArgument",
        JammiError::ChannelCatalog(
            ChannelCatalogError::InvalidId(_) | ChannelCatalogError::InvalidColumnType(_),
        ) => "InvalidArgument",
        _ => "BackendError",
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
    let class = match status.code() {
        Code::InvalidArgument => "InvalidArgument",
        _ => "BackendError",
    };
    client_error(class, status.message().to_string())
}
