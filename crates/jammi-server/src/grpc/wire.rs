//! Server-receive helpers shared by the engine-backed gRPC services
//! (`EmbeddingService`, `InferenceService`, `EvalService`, `TrainingService`,
//! `MutableTableService`, `ChannelService`, `AuditService`, and the topic-admin
//! verbs on `TriggerService`).
//!
//! These are transport concerns that belong on the receive side, not wire
//! conversions: the proto↔domain conversions and Arrow-IPC body helpers live in
//! [`jammi_ai::wire`] (shared with a future remote client). What stays here is
//! everything that touches a tonic [`Request`] extension or maps an engine error
//! to a tonic [`Status`]:
//!
//! * [`session_tenant`] — read the tenant the [`crate::grpc::session::
//!   TenantInterceptor`] attached to the request.
//! * [`scoped`] — run a session call under that tenant via the
//!   concurrency-safe `with_tenant_scoped` task-local.
//! * [`require_nonempty`] — reject a missing required string field.
//! * [`map_engine_error`] — map an engine [`JammiError`] to a gRPC [`Status`]
//!   preserving the failure kind.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::wire::attach_error_detail;
use jammi_db::error::JammiError;
use jammi_db::TenantId;
use tonic::{Code, Request, Status};

use crate::grpc::session::SessionTenant;

/// Read the bound tenant the [`crate::grpc::session::TenantInterceptor`]
/// attached to the request.
pub fn session_tenant<T>(request: &Request<T>) -> Option<TenantId> {
    request
        .extensions()
        .get::<SessionTenant>()
        .and_then(|s| s.0)
}

/// Run a session call under the request's tenant scope.
///
/// A bound tenant installs the engine's task-local tenant override for the
/// duration of the call via `with_tenant_scoped` — the concurrency-safe form
/// the gRPC handlers must use, since they share one `Arc<InferenceSession>`
/// and the sticky `bind_tenant` would race across concurrent requests. The
/// `TenantScope` handle the closure receives is the marker that the scope is
/// active on this task; `f` calls the verb on the [`jammi_ai::LocalSession`]
/// (which delegates to the same engine) and observes the same task-local. An
/// unscoped session runs the call directly.
///
/// Generic over the call's error type: most verbs return [`JammiError`], but the
/// trigger and audit verbs return their own error enums (`TriggerError`,
/// `AuditError`). The scoping mechanism is identical regardless, so the one
/// helper serves them all — only the per-verb `Status` mapping differs at the
/// call site.
pub async fn scoped<F, Fut, T, E>(
    session: &Arc<InferenceSession>,
    tenant: Option<TenantId>,
    f: F,
) -> Result<T, E>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    match tenant {
        Some(t) => session.with_tenant_scoped(t, |_scope| f()).await,
        None => f().await,
    }
}

/// Reject a missing required string field with `invalid_argument`.
pub fn require_nonempty(value: &str, field: &str) -> Result<(), Status> {
    if value.is_empty() {
        Err(Status::invalid_argument(format!("{field} is required")))
    } else {
        Ok(())
    }
}

/// Map an engine [`JammiError`] to a gRPC [`Status`], preserving the kind of
/// failure so a client can distinguish a bad request from an internal fault.
///
/// The `code` + `message` are the idiomatic gRPC surface (a client that does
/// not decode the structured detail still sees a sensible status). On top of
/// that, every status carries a faithful [`jammi_ai::wire`] error detail so a
/// remote client (`RemoteSession`) reconstructs the *exact* `JammiError` the
/// in-process [`jammi_ai::Session`] returns — the standard gRPC code set is too
/// coarse to distinguish Source / Model / Tenant / Config / Schema / Eval, all
/// of which collapse onto `invalid_argument`. The detail is built centrally
/// here so the faithful path covers the whole `JammiError` enum from one place.
pub fn map_engine_error(err: JammiError) -> Status {
    let (code, message) = match &err {
        JammiError::Source { source_id, message } => (
            Code::InvalidArgument,
            format!("source {source_id}: {message}"),
        ),
        JammiError::Model { model_id, message } => (
            Code::InvalidArgument,
            format!("model {model_id}: {message}"),
        ),
        JammiError::Tenant(detail) => (Code::InvalidArgument, format!("tenant: {detail}")),
        JammiError::Config(detail) => (Code::InvalidArgument, format!("config: {detail}")),
        JammiError::Schema { .. } => (Code::InvalidArgument, err.to_string()),
        JammiError::Eval(detail) => (Code::InvalidArgument, format!("eval: {detail}")),
        JammiError::Inference(detail) => (Code::Internal, format!("inference: {detail}")),
        other => (Code::Internal, other.to_string()),
    };
    attach_error_detail(code, message, &err)
}
