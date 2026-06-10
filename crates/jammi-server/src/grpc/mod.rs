//! gRPC services exposed by `jammi-server`.
//!
//! The wire surface is hybrid (per ADR-01): Flight SQL retains query/result;
//! typed gRPC services split the rest into one control plane and the data-plane
//! compute/stream services. The control plane is [`catalog::CatalogServer`] —
//! one `CatalogService` holding every catalog / metadata / lifecycle /
//! observability verb, including the tenant trio. The data plane is the
//! per-capability services (embedding / inference / training / eval / pipeline /
//! trigger / audit). Both Flight SQL and the gRPC services share a
//! [`session::SessionStore`] via the [`session::TenantInterceptor`] so a tenant
//! bound through `CatalogService.SetTenant` is observable on any downstream
//! request — including SQL queries issued against Flight SQL.
//!
//! The generated `jammi.v1` stubs and the proto↔domain conversions live in
//! [`jammi_ai::wire`]; this module re-exports the stubs as [`proto`] (so the
//! service impls and the integration-test harness keep their existing paths)
//! and adds the server-receive helpers in [`wire`].

pub mod audit;
pub mod catalog;
pub mod embedding;
pub mod eval;
pub mod inference;
pub mod pipeline;
pub mod session;
pub mod training;
pub mod trigger;
pub mod wire;

/// Proto-generated types for the `jammi.v1` API surface, re-exported from
/// [`jammi_ai::wire::proto`] — the single home for the generated stubs.
pub use jammi_ai::wire::proto;
