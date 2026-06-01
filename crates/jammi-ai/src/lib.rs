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
pub mod local_session;
pub mod model;
pub mod operator;
pub mod pipeline;
pub mod search;
pub mod session;

/// The transport-agnostic consumer surface: a closed `enum` over session
/// transports, with the in-process [`local_session::LocalSession`] behind it.
pub use local_session::{LocalSession, Modality, QueryInput, SearchQuery, SearchRequest, Session};

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
