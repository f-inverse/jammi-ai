//! Server-receive helpers shared by the engine-backed gRPC services: the
//! control-plane `CatalogService` (sources / models / channels / mutable tables
//! / topic admin) and the data-plane `EmbeddingService`, `InferenceService`,
//! `EvalService`, `TrainingService`, `AuditService`, and `TriggerService`
//! publish/subscribe verbs.
//!
//! These are transport concerns that belong on the receive side, not wire
//! conversions: the proto↔domain conversions and Arrow-IPC body helpers live in
//! [`jammi_wire`] + the engine-spec converters here (shared with the client crates). What stays here is
//! everything that touches a tonic [`Request`] extension or maps an engine error
//! to a tonic [`Status`]:
//!
//! * [`session_tenant`] — read the tenant the async tenant-binding layer
//!   ([`crate::tenant_resolver_layer`]) attached to the request.
//! * [`scoped`] — run a session call under that tenant via the
//!   concurrency-safe `with_tenant_scoped` task-local.
//! * [`require_nonempty`] — reject a missing required string field.
//! * [`map_engine_error`] — map an engine [`JammiError`] to a gRPC [`Status`]
//!   preserving the failure kind.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::catalog::channel_repo::ChannelCatalogError;
use jammi_db::error::JammiError;
use jammi_db::store::mutable::MutableTableError;
use jammi_db::trigger::TriggerError;
use jammi_db::TenantId;
use jammi_wire::{attach_error_detail, attach_trigger_detail};
use tonic::{Code, Request, Status};

use crate::grpc::session::SessionTenant;

/// Read the bound tenant the async tenant-binding layer
/// ([`crate::tenant_resolver_layer`]) attached to the request.
pub fn session_tenant<T>(request: &Request<T>) -> Option<TenantId> {
    request
        .extensions()
        .get::<SessionTenant>()
        .and_then(|s| s.0)
}

/// Read the request's bound tenant and stamp it onto the current request span's
/// `tenant_id` field.
///
/// The tenant only becomes known *inside* a handler — the per-service async
/// tenant-binding layer deposits the [`SessionTenant`] extension post-routing,
/// so a pre-routing tower layer cannot see it. Each tenant-aware handler therefore
/// opens its span with `tenant_id` empty and calls this once it has the request
/// in hand to fill it in, so a trace ties the gRPC request to the tenant scope
/// the call runs under. Returns the resolved tenant for the handler to scope on.
pub fn session_tenant_traced<T>(request: &Request<T>) -> Option<TenantId> {
    let tenant = session_tenant(request);
    tracing::Span::current().record("tenant_id", tracing::field::debug(&tenant));
    tenant
}

/// Run a session call under the request's tenant scope.
///
/// A bound tenant installs the engine's task-local tenant override for the
/// duration of the call via `with_tenant_scoped` — the concurrency-safe form
/// the gRPC handlers must use, since they share one `Arc<InferenceSession>`
/// and the sticky `bind_tenant` would race across concurrent requests. The
/// `TenantScope` handle the closure receives is the marker that the scope is
/// active on this task; `f` calls the verb on the [`jammi_ai::Session`]
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
/// that, every status carries a faithful [`jammi_wire`] error detail so a
/// data-plane client reconstructs the *exact* `JammiError` the
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
        // An absent model row a lifecycle verb resolved nothing for — a NotFound,
        // not the bad-argument `Model` fault. Mirrors the `ModelReferenced` arm.
        JammiError::ModelNotFound { model_id } => {
            (Code::NotFound, format!("model {model_id} not found"))
        }
        JammiError::ModelReferenced {
            model_id,
            referenced_by,
        } => (
            Code::FailedPrecondition,
            format!(
                "model {model_id} is still referenced by {}",
                referenced_by.join(", ")
            ),
        ),
        JammiError::Tenant(detail) => (Code::InvalidArgument, format!("tenant: {detail}")),
        JammiError::Config(detail) => (Code::InvalidArgument, format!("config: {detail}")),
        JammiError::Schema { .. } => (Code::InvalidArgument, err.to_string()),
        JammiError::Eval(detail) => (Code::InvalidArgument, format!("eval: {detail}")),
        JammiError::Inference(detail) => (Code::Internal, format!("inference: {detail}")),
        // The mutable-table kind carries a typed failure the coarse gRPC code set
        // must preserve so a remote client distinguishes a missing table (the
        // `if_exists` no-op signal) and an id collision from a genuine fault — the
        // mutable-table analogue of the `TopicNotFound` → `NotFound` mapping. The
        // validation variants are caller errors (`InvalidArgument`); a backend
        // fault falls through to `Internal`.
        JammiError::MutableTable(mt) => {
            let code = match mt {
                MutableTableError::NotFound(_) => Code::NotFound,
                MutableTableError::AlreadyExists(_) => Code::AlreadyExists,
                MutableTableError::InvalidId(_)
                | MutableTableError::Schema(_)
                | MutableTableError::MissingPrimaryKey(_)
                | MutableTableError::ReservedColumn(_)
                | MutableTableError::NoOrderColumn => Code::InvalidArgument,
                MutableTableError::Backend(_) => Code::Internal,
            };
            (code, mt.to_string())
        }
        // A channel-catalog op carries a typed caller condition the coarse gRPC
        // code set must preserve so a remote client distinguishes a duplicate
        // channel, an absent channel, a column conflict, and bad input. A
        // same-type redeclare and a duplicate channel are both `AlreadyExists`
        // (the resource is already present); a different-type redeclare is a
        // `FailedPrecondition` conflict against the stored declaration; an
        // unregistered channel is `NotFound`; a bad slug or column-type token is
        // `InvalidArgument`. (The two input variants are pre-rejected at the wire
        // boundary — slugs by `parse_channel_id`, dtype tokens by the closed
        // proto enum — so they are reachable only from the embedded surfaces.)
        JammiError::ChannelCatalog(c) => {
            let code = match c {
                ChannelCatalogError::AlreadyExists(_)
                | ChannelCatalogError::ColumnAlreadyDeclared { .. } => Code::AlreadyExists,
                ChannelCatalogError::NotRegistered(_) => Code::NotFound,
                ChannelCatalogError::ColumnConflict { .. } => Code::FailedPrecondition,
                ChannelCatalogError::InvalidId(_) | ChannelCatalogError::InvalidColumnType(_) => {
                    Code::InvalidArgument
                }
            };
            (code, c.to_string())
        }
        // Channel-assembly failures are reached only from the engine-internal
        // search-merge path on engine-derived inputs — an engine invariant, not a
        // caller condition — so they fall through to `Internal`.
        JammiError::ChannelAssembly(detail) => {
            (Code::Internal, format!("channel assembly: {detail}"))
        }
        other => (Code::Internal, other.to_string()),
    };
    attach_error_detail(code, message, &err)
}

/// Map a [`TriggerError`] onto a gRPC [`Status`], preserving the failure kind so
/// a client can distinguish a bad request from an internal fault.
///
/// The `code` + `message` are the idiomatic gRPC surface (a client that does not
/// decode the structured detail still sees a sensible status). On top of that,
/// every status carries a faithful [`jammi_wire`] trigger-error detail so a
/// remote the data-plane client reconstructs the *exact* [`TriggerError`] the
/// in-process path returns — the standard gRPC code set is too coarse to
/// distinguish, e.g., `PredicateParse` from `PredicateUnsupported`, or the two
/// engine-owned `#[from]` nests. The detail is built centrally here so the
/// faithful path covers the whole `TriggerError` enum from one place — the
/// trigger analogue of [`map_engine_error`]. Shared by the topic-admin verbs on
/// [`CatalogService`](crate::grpc::catalog) and the publish/subscribe verbs on
/// [`TriggerService`](crate::grpc::trigger).
pub fn map_trigger_error(err: TriggerError) -> Status {
    let (code, message) = match &err {
        TriggerError::TopicNotFound(name) => (Code::NotFound, name.clone()),
        TriggerError::BatchSchemaMismatch(detail) => (Code::InvalidArgument, detail.clone()),
        TriggerError::SchemaConflict { topic, detail } => (
            Code::FailedPrecondition,
            format!("schema conflict on {topic}: {detail}"),
        ),
        TriggerError::UnsupportedSchemaType { column, data_type } => (
            Code::InvalidArgument,
            format!("unsupported topic schema type for '{column}': {data_type}"),
        ),
        TriggerError::PublishTenantMismatch {
            topic,
            topic_tenant,
            publish_tenant,
        } => (
            Code::PermissionDenied,
            format!(
                "publish tenant mismatch on topic '{topic}': topic_tenant={topic_tenant:?}, publish_tenant={publish_tenant:?}"
            ),
        ),
        TriggerError::PredicateParse(detail) | TriggerError::PredicateUnsupported(detail) => {
            (Code::InvalidArgument, format!("predicate: {detail}"))
        }
        TriggerError::PredicateEval(detail) => (Code::Internal, format!("predicate: {detail}")),
        TriggerError::OffsetEvicted(n) => {
            (Code::FailedPrecondition, format!("offset {n} evicted"))
        }
        TriggerError::BackingTable(e) => (Code::Internal, format!("backing table: {e}")),
        TriggerError::Backend(e) => (Code::Internal, format!("backend: {e}")),
        TriggerError::Driver(detail) => (Code::Unavailable, format!("broker: {detail}")),
        TriggerError::Catalog(detail) => (Code::Internal, format!("catalog: {detail}")),
    };
    attach_trigger_detail(code, message, &err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jammi_wire::error_from_status;

    /// An absent model from a lifecycle verb is `ModelNotFound`, which maps to
    /// gRPC `NotFound` — distinct from the bad-argument `Model` fault, which maps
    /// to `InvalidArgument`. This is the contract the catalog handlers rely on
    /// after dropping their manual `Model → not_found` interception.
    #[test]
    fn model_not_found_maps_to_not_found() {
        let status = map_engine_error(JammiError::ModelNotFound {
            model_id: "acme/embed-mini".into(),
        });
        assert_eq!(status.code(), Code::NotFound);

        // The bad-argument model fault keeps mapping to InvalidArgument.
        let bad = map_engine_error(JammiError::Model {
            model_id: "acme/embed-mini".into(),
            message: "invalid version".into(),
        });
        assert_eq!(bad.code(), Code::InvalidArgument);
    }

    /// esc-089 F3 (round-3 audit): a resolver/predictor integrity failure OR
    /// an unpublished-bundle refusal is a client-visible precondition
    /// failure on BOTH adapter-reload surfaces — `ModelResolver`'s
    /// fine-tuned arm (`crates/jammi-ai/src/model/resolver.rs`) and the
    /// context-predictor reload arm
    /// (`crates/jammi-ai/src/pipeline/context_predictor.rs`) both raise the
    /// SAME `JammiError::Model`, which this wire boundary maps to
    /// `Code::InvalidArgument` per the `model_not_found_maps_to_not_found`
    /// test above. Before the fix, the predictor surface raised its own
    /// `JammiError::Inference` for this case instead, which falls through
    /// this function's catch-all arm to `Code::Internal` — wrong for a
    /// client-visible precondition failure, and disagreeing with the
    /// resolver surface for the identical class of outcome. A genuine
    /// transport/IO fault (credential rot, a disabled cloud scheme, a
    /// network fault against S3/GCS/azure) is propagated UNCHANGED as
    /// `JammiError::Storage` by both surfaces and has no typed arm in this
    /// function, so it falls through to `Code::Internal` on both — this is
    /// the correct, deliberate asymmetry: a caller-visible precondition gets
    /// `InvalidArgument`, a backend/transport fault gets `Internal`.
    #[test]
    fn adapter_bundle_refusal_codes_agree_across_both_reload_surfaces() {
        use jammi_db::storage::{Scheme, StorageError};

        // Resolver integrity failure (a manifest-listed key absent or hash
        // mismatch) — `ModelResolver::try_catalog_lookup`'s fine-tuned arm.
        let resolver_integrity = map_engine_error(JammiError::Model {
            model_id: "jammi:fine-tuned:job-1".into(),
            message: "adapter bundle at 'file:///artifacts/job-1' failed integrity check \
                      for fine-tuned model 'jammi:fine-tuned:job-1': artifact file \
                      'adapter.safetensors' missing under this prefix (manifest lists it)"
                .into(),
        });
        assert_eq!(
            resolver_integrity.code(),
            Code::InvalidArgument,
            "a resolver integrity failure must be a client-visible precondition failure"
        );

        // Resolver transport/IO fault — propagated unchanged, never folded
        // into `JammiError::Model`. `DriverInit` stands in for the class of
        // fault named in `resolver.rs`'s own doc comment (credential rot, a
        // disabled scheme, a network fault) — never this model's own
        // content being wrong.
        let resolver_transport = map_engine_error(JammiError::Storage(StorageError::DriverInit {
            scheme: Scheme::S3,
            reason: "credential rot: token expired".into(),
        }));
        assert_eq!(
            resolver_transport.code(),
            Code::Internal,
            "a resolver transport/IO fault must stay Internal, never InvalidArgument"
        );

        // Predictor integrity failure — the context-predictor reload arm's
        // own typed refusal, unified onto the SAME `JammiError::Model`
        // variant the resolver surface raises (this is the fix: it used to
        // be `JammiError::Inference`, which mapped to `Internal` above).
        let predictor_integrity = map_engine_error(JammiError::Model {
            model_id: "cnp-1".into(),
            message: "adapter bundle at 'file:///artifacts/cnp-1' failed integrity check \
                      for context predictor 'cnp-1': artifact file 'model.safetensors' \
                      missing under this prefix (manifest lists it)"
                .into(),
        });
        assert_eq!(
            predictor_integrity.code(),
            Code::InvalidArgument,
            "a predictor integrity failure must be a client-visible precondition failure, \
             the SAME code the resolver surface gets for the same class of outcome"
        );

        // Predictor transport/IO fault — propagated unchanged, same as the
        // resolver surface.
        let predictor_transport = map_engine_error(JammiError::Storage(StorageError::DriverInit {
            scheme: Scheme::Gcs,
            reason: "network fault: connection reset".into(),
        }));
        assert_eq!(
            predictor_transport.code(),
            Code::Internal,
            "a predictor transport/IO fault must stay Internal, matching the resolver surface"
        );
    }

    /// The faithful detail attached to the Status reconstructs the exact
    /// `ModelNotFound` variant on the client side — not a coarse code guess.
    #[test]
    fn model_not_found_detail_round_trips() {
        let status = map_engine_error(JammiError::ModelNotFound {
            model_id: "acme/embed-mini".into(),
        });
        match error_from_status(&status) {
            JammiError::ModelNotFound { model_id } => assert_eq!(model_id, "acme/embed-mini"),
            other => panic!("expected ModelNotFound to round-trip, got {other:?}"),
        }
    }
}
