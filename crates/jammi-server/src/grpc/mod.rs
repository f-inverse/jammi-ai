//! gRPC services exposed by `jammi-server`.
//!
//! The wire surface is hybrid (per ADR-01): Flight SQL retains query/result;
//! typed gRPC services handle session management and (later) trigger
//! streams. Both Flight SQL and gRPC services share a
//! [`session::SessionStore`] via the [`session::TenantInterceptor`] so a
//! tenant bound through `SessionService.SetTenant` is observable on any
//! downstream request — including SQL queries issued against Flight SQL.
//!
//! The generated `jammi.v1` stubs and the proto↔domain conversions live in
//! [`jammi_ai::wire`]; this module re-exports the stubs as [`proto`] (so the
//! service impls and the integration-test harness keep their existing paths)
//! and adds the server-receive helpers in [`wire`].

pub mod audit;
pub mod channel;
pub mod embedding;
pub mod eval;
pub mod fine_tune;
pub mod inference;
pub mod mutable_table;
pub mod session;
pub mod trigger;
pub mod wire;

/// Proto-generated types for the `jammi.v1` API surface, re-exported from
/// [`jammi_ai::wire::proto`] — the single home for the generated stubs.
pub use jammi_ai::wire::proto;
