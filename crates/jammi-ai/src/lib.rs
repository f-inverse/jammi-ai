//! Jammi AI — model loading, inference execution, and output adapters.
//!
//! This crate provides the intelligence layer of the Jammi platform:
//! explicit model source resolution, Candle/ORT backends, batch inference
//! with backpressure, and typed output adapters for embedding,
//! classification, summarization, and other ML tasks.

pub mod concurrency;
pub mod eval;
pub mod evidence;
pub mod fine_tune;
pub mod inference;
pub mod jammi;
pub mod local_session;
pub mod model;
pub mod operator;
pub mod pipeline;
pub mod search;
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
/// transports, with the in-process [`local_session::LocalSession`] behind it.
pub use local_session::{LocalSession, Modality, QueryInput, SearchQuery, SearchRequest, Session};

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
