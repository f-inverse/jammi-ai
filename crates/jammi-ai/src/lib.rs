//! Jammi AI — model loading, inference execution, and output adapters.
//!
//! This crate provides the intelligence layer of the Jammi platform:
//! explicit model source resolution, Candle/ORT backends, batch inference
//! with backpressure, and typed output adapters for embedding,
//! classification, summarization, and other ML tasks.

// The embedded engine: model loading, inference execution, fine-tune training,
// and the embedding/search pipelines that run on the candle stack. Gated behind
// the default-on `local` feature so a remote-only `wire` build pulls none of it.
// `eval` and `fine_tune` keep their candle-free config / report vocabulary
// (`EvalTask`, the eval reports, `FineTuneConfig`/`FineTuneMethod`) available
// either way — only their engine-running submodules ride `local`.
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
pub mod jammi;
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

/// The gRPC wire surface: generated `jammi.v1` tonic stubs plus the
/// proto↔domain conversions. Gated behind the default-off `wire` feature so a
/// default / embedded build pulls no transport stack. `jammi-server` enables
/// it for the server stubs + conversions; a future `RemoteSession` will reuse
/// the same module's client stubs.
#[cfg(feature = "wire")]
pub mod wire;

/// The remote [`local_session::Session`] transport: a gRPC client peer of
/// [`local_session::LocalSession`]. Gated behind `wire` alongside the stubs it
/// drives; a default / embedded build has no remote transport and the
/// [`local_session::Session`] enum is the one-arm `Local` shape.
#[cfg(feature = "wire")]
pub mod remote_session;

/// The transport-agnostic consumer surface: a closed `enum` over session
/// transports. The in-process [`local_session::LocalSession`] behind the
/// `Local` arm rides the `local` feature alongside the engine it drives; the
/// request/result vocabulary ([`Modality`], the query and search shapes) is
/// transport-neutral and always present.
pub use local_session::{Modality, QueryInput, SearchQuery, SearchRequest, Session};

/// Engine introspection shapes the [`Session`] surface returns: a per-source
/// [`SourceDescriptor`] (registry identity joined with its embedding result
/// tables) and the build's capabilities [`ServerInfo`]. Both originate in the
/// `jammi-db` substrate (the catalog and the compile-time capability facts) and
/// are re-exported here so SDK consumers reach them as `jammi_ai::*`.
pub use jammi_db::catalog::source_repo::SourceDescriptor;
pub use jammi_db::ServerInfo;

/// The in-process [`Session`] transport, behind the `local` feature with the
/// embedded engine it drives.
#[cfg(feature = "local")]
pub use local_session::LocalSession;

/// The SDK front door: [`jammi::Jammi::open`] opens a [`Session`] against a
/// [`jammi::Target`], selecting the embedded (`Local`) or — under `wire` — the
/// remote (`Remote`) transport in one call.
pub use jammi::{Jammi, Target};

/// The remote transport behind the [`Session`] enum's `wire`-gated `Remote`
/// arm.
#[cfg(feature = "wire")]
pub use remote_session::RemoteSession;

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
