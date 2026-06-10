//! Jammi AI — model loading, inference execution, and output adapters.
//!
//! This crate provides the intelligence layer of the Jammi platform:
//! explicit model source resolution, Candle/ORT backends, batch inference
//! with backpressure, and typed output adapters for embedding,
//! classification, summarization, and other ML tasks.

// The embedded engine: model loading, inference execution, fine-tune training,
// and the embedding/search pipelines that run on the candle stack. Gated behind
// the default-on `local` feature. `eval` and `fine_tune` keep their candle-free
// config / report vocabulary (`EvalTask`, the eval reports,
// `FineTuneConfig`/`FineTuneMethod`) available either way — re-exported from the
// wire substrate — while only their engine-running submodules ride `local`.
#[cfg(feature = "local")]
pub mod concurrency;
pub mod eval;
#[cfg(feature = "local")]
pub mod evidence;
pub mod fine_tune;
#[cfg(feature = "local")]
pub mod index;
#[cfg(feature = "local")]
pub mod inference;
#[cfg(feature = "local")]
pub mod jammi;
#[cfg(feature = "local")]
pub mod local_session;
#[cfg(feature = "local")]
pub mod model;
#[cfg(feature = "local")]
pub mod operator;
#[cfg(feature = "local")]
pub mod pipeline;
/// Serving-time prediction wrappers (split conformal prediction). Pure math
/// over predictor outputs — no model, no candle stack — so it rides every
/// build, not just the `local` engine: a distribution-free coverage guarantee
/// is a serving output that must work with a dead license.
pub mod predict;
#[cfg(feature = "local")]
pub mod query;
#[cfg(feature = "local")]
pub mod session;

/// The engine's residual wire surface: the engine-spec proto↔domain conversions
/// the candle-free `jammi-wire` substrate cannot home (`TrainingSpec`, the
/// declared-edge gather, the served distribution, the pipeline request/response
/// structs). Built on `jammi_wire`'s `proto` + helpers, so `jammi-server`
/// consumes these conversions alongside the candle-free ones it gets straight
/// from `jammi-wire`. Engine-spec, so it rides the `local` feature.
#[cfg(feature = "local")]
pub mod wire;

/// The in-process consumer surface: a [`Session`] over the embedded engine. It
/// rides the `local` feature alongside the engine it drives; the request/result
/// vocabulary ([`Modality`], the query and search shapes) lives on the
/// `jammi-wire` substrate and is re-exported through it.
#[cfg(feature = "local")]
pub use local_session::{Modality, QueryInput, SearchQuery, SearchRequest, Session};

/// Engine introspection shapes the [`Session`] surface returns: a per-source
/// [`SourceDescriptor`] (registry identity joined with its embedding result
/// tables) and the build's capabilities [`ServerInfo`]. Both originate in the
/// `jammi-db` substrate (the catalog and the compile-time capability facts) and
/// are re-exported here so SDK consumers reach them as `jammi_ai::*`.
pub use jammi_db::catalog::source_repo::SourceDescriptor;
pub use jammi_db::ServerInfo;

/// The SDK front door: [`jammi::Jammi::open`] opens an embedded [`Session`] from
/// a [`jammi::Target`] config in one call. Rides the `local` feature with the
/// embedded engine it builds.
#[cfg(feature = "local")]
pub use jammi::{Jammi, Target};

/// The per-query audit primitive lives in the `jammi-db` substrate (it composes
/// mutable tables, tenant scope, and the trigger stream). It is re-exported here
/// so AI-layer users can `use jammi_ai::PerQueryAudit`.
pub use jammi_db::audit;
pub use jammi_db::{AuditError, AuditHandle, PerQueryAudit};

/// The ephemeral session-storage primitive lives in the `jammi-db` substrate
/// (it composes mutable tables, tenant scope, and the trigger stream). It is
/// re-exported here so AI-layer users can `use jammi_ai::EphemeralSession`.
pub use jammi_db::ephemeral;
pub use jammi_db::{
    EphemeralError, EphemeralSession, SessionLifecycleEvent, SessionLifecycleRecord,
};
