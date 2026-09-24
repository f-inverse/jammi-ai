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
        // The end the caller asked for, distinct from a fault — the class the
        // remote client raises for a `cancelled` job (`RemoteJob.wait`).
        JammiError::JobCancelled { .. } => "JobCancelled",
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
        // A training set whose projection yields no rows is a degenerate input
        // the caller must change. The server maps it to `Code::InvalidArgument`
        // (`jammi_server::grpc::wire::map_engine_error`), which the remote
        // client raises as `jammi.errors.InvalidArgument`
        // (`clients/python/jammi/_database.py::_rpc_to_jammi`) — so the
        // embedded engine raises THAT class, not the `BackendError` the
        // fall-through below would give it. No leaf class: the remote client
        // decodes no status detail, so a refinement here would be catchable on
        // one transport only.
        JammiError::EmptyTrainingSet { .. } => "InvalidArgument",
        // The leaf classes `jammi.errors` refines from `InvalidArgument` /
        // `BackendError` for the typed refusals the embedded engine raises
        // (each subclasses the class the remote mapper produces for its gRPC
        // code, so one `except` holds on both transports).
        JammiError::InvalidKey { .. } => "InvalidKey",
        JammiError::VersionUnavailable { .. } => "VersionUnavailable",
        JammiError::MissingManifest { .. } => "MissingManifest",
        JammiError::NotRefreshable { .. } => "NotRefreshable",
        JammiError::DefinitionDrift { .. } => "DefinitionDrift",
        JammiError::NonUniqueKey { .. } => "NonUniqueKey",
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
    /// The remote side of the equality is derived, not restated: the server maps
    /// this variant to `Code::InvalidArgument`
    /// (`jammi_server::grpc::wire::map_engine_error`, pinned by that crate's
    /// `empty_training_set_round_trips_as_its_typed_variant_not_other`), and
    /// [`status_class`] is this crate's copy of the remote client's
    /// code → class partition. Were the variant to fall through to
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
