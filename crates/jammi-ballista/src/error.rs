//! This crate's own error taxonomy — role hosting (`roles.rs`) and the
//! submit client (`client.rs`) return [`Result`]. The codec (`codec.rs`) and
//! engine (`engine.rs`) implement DataFusion/Ballista traits whose own
//! methods return `datafusion::error::Result`, so codec/engine call sites
//! convert a typed refusal directly into a [`datafusion::error::DataFusionError`]
//! at the point of return (`Self::into_df_error` below) rather than routing
//! through this type — there is no ambient conversion because the trait
//! signatures are fixed by DataFusion/Ballista, not by this crate.

use datafusion::error::DataFusionError;

/// This crate's error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A plan node neither `JammiCodec` nor the delegate Ballista codec
    /// recognizes.
    #[error("jammi-ballista: unsupported plan node: {0}")]
    UnsupportedNode(String),
    /// The codec's `Weak<InferenceSession>` has no live session — the
    /// process that encoded this plan has gone away.
    #[error("jammi-ballista: no live session to decode against")]
    SessionGone,
    /// A wire buffer failed to decode (truncated, wrong magic, malformed
    /// prost message).
    #[error("jammi-ballista: malformed operator buffer: {0}")]
    Decode(String),
    /// A catalog lookup this codec performs on decode (e.g. re-reading a
    /// result table by name) failed.
    #[error("jammi-ballista: catalog error: {0}")]
    Catalog(#[from] jammi_db::error::JammiError),
    /// An invalid `[ballista]` role configuration.
    #[error("jammi-ballista: config error: {0}")]
    Config(String),
    /// No live executor can hold the plan — refused before submitting,
    /// never parked unschedulable on the scheduler.
    #[error("jammi-ballista: {0} — refused before submitting, never parked unschedulable")]
    Unheld(jammi_db::compute_plane::Unheld),
    /// A role (scheduler/executor) failed to start or stop.
    #[error("jammi-ballista: role error: {0}")]
    Role(String),
    /// A DataFusion-side failure surfaced while building or executing a plan.
    #[error(transparent)]
    DataFusion(#[from] DataFusionError),
    /// A Ballista-side failure (scheduler/executor/client).
    #[error("jammi-ballista: {0}")]
    Ballista(#[from] ballista_core::error::BallistaError),
    /// A transport failure standing up a role's listener.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A tonic transport failure.
    #[error(transparent)]
    Transport(#[from] tonic::transport::Error),
}

/// This crate's `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Convert into the `DataFusionError` the `PhysicalExtensionCodec` and
    /// `ExecutionEngine` trait methods must return — the typed-refusal shape
    /// every codec/engine call site uses (never a silent fallback).
    pub fn into_df_error(self) -> DataFusionError {
        match self {
            Error::DataFusion(e) => e,
            other => DataFusionError::External(Box::new(other)),
        }
    }
}

/// The engine error a caller of the submit client sees: the typed error
/// this crate carried (`Catalog`), the classified one a DataFusion error
/// holds (a placed task's restored failure among them), a configuration
/// refusal as the engine's own; a plan the plane cannot hold right now —
/// whichever reason — as the one typed class `JammiError::Unheld`, a
/// runtime state of the plane and never a caller fault; an I/O fault as the
/// engine's own; the rest fold to `Other` carrying their `Display`, the
/// same fold the wire codec applies to a foreign error.
impl From<Error> for jammi_db::error::JammiError {
    fn from(e: Error) -> Self {
        use jammi_db::error::JammiError;
        match e {
            Error::Catalog(e) => e,
            Error::DataFusion(e) => JammiError::from(e),
            Error::Config(m) => JammiError::Config(m),
            Error::Unheld(why) => JammiError::Unheld(why),
            Error::Io(e) => JammiError::from(e),
            other => JammiError::Other(other.to_string()),
        }
    }
}
