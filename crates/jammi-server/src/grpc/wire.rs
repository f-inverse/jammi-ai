//! Server-receive helpers shared by the engine-backed gRPC services: the
//! control-plane `CatalogService` (sources / models / channels / mutable tables
//! / topic admin) and the data-plane `EmbeddingService`, `InferenceService`,
//! `EvalService`, `JobService`, `AuditService`, and `TriggerService`
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
use jammi_db::error::JammiError;
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
    attach_error_detail(jammi_wire::status_code(&err), err.to_string(), &err)
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
    /// to `InvalidArgument`. The catalog handlers rely on this mapping; they
    /// carry no manual `Model → not_found` interception.
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

    /// The building-row CAS zero-row classification
    /// each maps to a DISTINCT gRPC code, never the catch-all arm's generic
    /// `Internal`: `RowGone` → `NotFound`,
    /// `TenantMismatch` → `PermissionDenied`, `LeaseLost` / `CasFailed` →
    /// `Aborted` (a retryable conflict, not a permanent precondition
    /// failure), `SourceBusy` → `FailedPrecondition` (mirroring
    /// `ModelReferenced`'s delete-precondition mapping). Also proves the
    /// wire round-trip: `error_from_status` reconstructs the exact variant
    /// from the attached detail, not just the coarse code.
    #[test]
    fn building_row_cas_errors_map_to_distinct_grpc_codes() {
        let row_gone = map_engine_error(JammiError::RowGone { table: "t1".into() });
        assert_eq!(row_gone.code(), Code::NotFound);
        assert!(matches!(
            error_from_status(&row_gone),
            JammiError::RowGone { table } if table == "t1"
        ));

        let tenant_mismatch = map_engine_error(JammiError::TenantMismatch { table: "t1".into() });
        assert_eq!(tenant_mismatch.code(), Code::PermissionDenied);
        assert!(matches!(
            error_from_status(&tenant_mismatch),
            JammiError::TenantMismatch { table } if table == "t1"
        ));

        let lease_lost = map_engine_error(JammiError::LeaseLost { table: "t1".into() });
        assert_eq!(lease_lost.code(), Code::Aborted);
        assert!(matches!(
            error_from_status(&lease_lost),
            JammiError::LeaseLost { table } if table == "t1"
        ));

        let cas_failed = map_engine_error(JammiError::CasFailed {
            table: "t1".into(),
            status: "ready".into(),
        });
        assert_eq!(cas_failed.code(), Code::Aborted);
        assert!(matches!(
            error_from_status(&cas_failed),
            JammiError::CasFailed { table, status } if table == "t1" && status == "ready"
        ));

        let source_busy = map_engine_error(JammiError::SourceBusy {
            source_id: "src1".into(),
            table: "t1".into(),
        });
        assert_eq!(source_busy.code(), Code::FailedPrecondition);
        assert!(matches!(
            error_from_status(&source_busy),
            JammiError::SourceBusy { source_id, table } if source_id == "src1" && table == "t1"
        ));

        // A parent-pinned version CAS lost the race the same way `CasFailed`
        // does — the same retryable `Aborted` code, but its own typed
        // variant carrying `expected`/`found` rather than `CasFailed`'s
        // `status`, which would misname the cause (the row IS `ready`).
        let parent_moved = map_engine_error(JammiError::ParentMoved {
            table: "t1".into(),
            expected: Some(3),
            found: Some(4),
        });
        assert_eq!(parent_moved.code(), Code::Aborted);
        assert!(matches!(
            error_from_status(&parent_moved),
            JammiError::ParentMoved { table, expected, found }
                if table == "t1" && expected == Some(3) && found == Some(4)
        ));
    }

    /// `JobAttemptSuperseded`/`JobCancelled` must round-trip as their typed
    /// variant across the wire, never fold into the lossy
    /// `JammiError::Other`: `map_engine_error` classifies them with the
    /// right gRPC `Code` (`Aborted`/`Cancelled`), and `attach_error_detail`'s
    /// `pb::JammiErrorDetail::from(&JammiError)` carries a dedicated arm for
    /// each, so a remote client's `error_from_status` reconstructs the typed
    /// variant with its `job_id`, never just the `Display` string.
    /// This exercises the SAME `attach_error_detail` → real `tonic::Status`
    /// (genuine `grpc-status-details-bin` metadata bytes) → `error_from_status`
    /// round trip a live gRPC call uses — the client-facing seam this
    /// invariant protects, not merely the in-memory `From` impl.
    #[test]
    fn job_attempt_superseded_and_job_cancelled_round_trip_as_their_typed_variant_not_other() {
        let superseded = map_engine_error(JammiError::JobAttemptSuperseded {
            job_id: "job-1".into(),
        });
        assert_eq!(superseded.code(), Code::Aborted);
        assert!(matches!(
            error_from_status(&superseded),
            JammiError::JobAttemptSuperseded { job_id } if job_id == "job-1"
        ));

        let cancelled = map_engine_error(JammiError::JobCancelled {
            job_id: "job-2".into(),
        });
        assert_eq!(cancelled.code(), Code::Cancelled);
        assert!(matches!(
            error_from_status(&cancelled),
            JammiError::JobCancelled { job_id } if job_id == "job-2"
        ));
    }

    /// The plane's loss of a placed job's executor reaches a remote caller
    /// as `Unavailable` — a peer that went away, retried by the job's
    /// successor — and reconstructs as its typed variant naming the
    /// executor and the placed job, never `FailedPrecondition` (nothing the
    /// caller could fix) and never a fold into `Other`.
    #[test]
    fn executor_lost_is_unavailable_and_round_trips_typed() {
        let lost = map_engine_error(JammiError::ExecutorLost {
            executor_id: "executor-1".into(),
            job_id: "7bY2".into(),
        });
        assert_eq!(lost.code(), Code::Unavailable);
        assert!(matches!(
            error_from_status(&lost),
            JammiError::ExecutorLost { executor_id, job_id }
                if executor_id == "executor-1" && job_id == "7bY2"
        ));
    }

    /// The empty-training-set refusal must reach a remote caller as the
    /// refusal it is. An empty
    /// training set is raised by the producer before any row or byte exists;
    /// folded into `Other`/`Internal` a remote caller would read a server
    /// fault where the embedded caller reads a typed, caller-fixable
    /// `EmptyTrainingSet` naming the query — the two surfaces disagreeing on
    /// exactly the degenerate input the refusal exists to catch. This pins the classified
    /// code, the typed reconstruction, and the `Display` text through the same
    /// `attach_error_detail` → real `tonic::Status` → `error_from_status` chain
    /// a live call uses.
    #[test]
    fn empty_training_set_round_trips_as_its_typed_variant_not_other() {
        let query = "SELECT \"abstract\" FROM patents WHERE 1 = 0";
        let engine = JammiError::EmptyTrainingSet {
            source_query: query.to_string(),
        };
        let status = map_engine_error(JammiError::EmptyTrainingSet {
            source_query: query.to_string(),
        });
        assert_eq!(status.code(), Code::InvalidArgument);
        let back = error_from_status(&status);
        assert!(
            matches!(&back, JammiError::EmptyTrainingSet { source_query } if source_query == query),
            "an empty training set must reconstruct as itself, got {back:?}"
        );
        assert_eq!(
            back.to_string(),
            engine.to_string(),
            "the remote text must be the embedded text"
        );
    }

    /// A memory-pool exhaustion (a training-set stream's reservation, an
    /// ordinary DataFusion plan) maps to gRPC `ResourceExhausted` — never a
    /// bare `Internal` — and reconstructs on the remote side as the exact
    /// typed variant, `limit_bytes` and `detail` both faithful, through the
    /// same `attach_error_detail` → real `tonic::Status` → `error_from_status`
    /// chain a live call uses.
    #[test]
    fn resources_exhausted_maps_to_resource_exhausted_and_round_trips() {
        let engine = JammiError::ResourcesExhausted {
            limit_bytes: 67_108_864,
            detail: "greedy(used: 10.0 MB, pool_size: 64.0 MB)".to_string(),
        };
        let status = map_engine_error(JammiError::ResourcesExhausted {
            limit_bytes: 67_108_864,
            detail: "greedy(used: 10.0 MB, pool_size: 64.0 MB)".to_string(),
        });
        assert_eq!(status.code(), Code::ResourceExhausted);
        let back = error_from_status(&status);
        assert!(
            matches!(
                &back,
                JammiError::ResourcesExhausted { limit_bytes, detail }
                    if *limit_bytes == 67_108_864
                        && detail == "greedy(used: 10.0 MB, pool_size: 64.0 MB)"
            ),
            "resources-exhausted must reconstruct as itself, got {back:?}"
        );
        assert_eq!(
            back.to_string(),
            engine.to_string(),
            "the remote text must be the embedded text"
        );
    }

    /// This wire test pins
    /// ONLY the LAST leg of the chain — a variant, once produced, maps to
    /// the right gRPC code — never a substitute for the it-tests in
    /// `crates/jammi-ai/tests/it/models.rs` /
    /// `crates/jammi-ai/tests/it/context_predictor.rs`, which pin the FIRST
    /// leg (a real reload surface, driven end-to-end, actually PRODUCES that
    /// variant). Both legs are needed and neither substitutes for the
    /// other: this test constructs the exact variants the two real reload
    /// surfaces are proven (by the it-tests above) to raise — never a
    /// stand-in — so a wire-boundary regression here composes with, rather
    /// than merely echoes, the it-tests' surface→variant proof.
    ///
    /// The three variants both surfaces raise, per `resolver.rs`'s
    /// fine-tuned arm and `context_predictor.rs`'s reload arm:
    ///   - `JammiError::Model` for a corrupted catalog pointer (unparseable
    ///     URL, no `artifact_path` recorded).
    ///   - `JammiError::Model` for a manifest-verified integrity failure
    ///     (`StorageError::Layout`, reclassified) — same variant, distinct
    ///     message.
    ///   - `JammiError::Model` for an absent bundle (`StorageError::
    ///     NotPublished`, reclassified) — "no bundle published here", not
    ///     corruption, but still the caller-visible precondition shape.
    ///   - `JammiError::Storage(StorageError::Io { .. })` for a genuine
    ///     transport/permission fault against an INTACT bundle — propagated
    ///     UNCHANGED by both surfaces (never folded into `Model`), and the
    ///     it-tests pin this EXACT variant
    ///     (`fine_tuned_adapter_bundle_permission_fault_is_not_a_typed_model_error`
    ///     / `context_predictor_reload_permission_fault_is_not_a_typed_model_error`
    ///     assert `matches!(err, JammiError::Storage(StorageError::Io { .. }))`
    ///     against a real chmod fault, not merely `!Model`) — `Io` is what a
    ///     permission-denied read on `object_store::LocalFileSystem` folds
    ///     into (`Error::Generic`, never its own `NotFound`), so that is the
    ///     variant this test constructs too, not `DriverInit` (which stands
    ///     for a cloud-driver construction fault neither reload surface's
    ///     transport path actually raises for a local permission fault).
    #[test]
    fn adapter_bundle_refusal_codes_agree_across_both_reload_surfaces() {
        use jammi_db::storage::StorageError;

        fn io_fault(path: &str) -> JammiError {
            JammiError::Storage(StorageError::io(
                path,
                object_store::Error::Generic {
                    store: "LocalFileSystem",
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "permission denied",
                    )),
                },
            ))
        }

        // Resolver: corrupted catalog pointer (unparseable artifact_path) —
        // `ModelResolver::try_catalog_lookup`'s fine-tuned arm.
        let resolver_bad_pointer = map_engine_error(JammiError::Model {
            model_id: "jammi:fine-tuned:job-1".into(),
            message: "fine-tuned model 'jammi:fine-tuned:job-1' artifact_path 'not a url' is \
                      not a valid storage URL: invalid scheme — this catalog record's \
                      pointer is corrupted"
                .into(),
        });
        assert_eq!(
            resolver_bad_pointer.code(),
            Code::InvalidArgument,
            "a corrupted catalog pointer must be a client-visible precondition failure"
        );

        // Resolver: manifest-verified integrity failure
        // (`StorageError::Layout`, reclassified to `Model` by the resolver).
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

        // Resolver: no bundle ever published (`StorageError::NotPublished`,
        // reclassified to `Model` by the resolver) — "no bundle published
        // here", not corruption, but still the SAME caller-visible code.
        let resolver_not_published = map_engine_error(JammiError::Model {
            model_id: "jammi:fine-tuned:job-1".into(),
            message: "no adapter bundle is published at 'file:///artifacts/job-1' for \
                      fine-tuned model 'jammi:fine-tuned:job-1' (manifest.json absent); the \
                      catalog pointer may be misdirected"
                .into(),
        });
        assert_eq!(
            resolver_not_published.code(),
            Code::InvalidArgument,
            "an unpublished bundle must be a client-visible precondition failure, the SAME \
             code as an integrity failure"
        );

        // Resolver: transport/permission fault against an INTACT bundle —
        // propagated UNCHANGED, never folded into `JammiError::Model`. This
        // is the SAME `StorageError::Io` variant
        // `fine_tuned_adapter_bundle_permission_fault_is_not_a_typed_model_error`
        // (models.rs) proves the resolver actually raises against a real
        // chmod fault.
        let resolver_transport = map_engine_error(io_fault("file:///artifacts/job-1"));
        assert_eq!(
            resolver_transport.code(),
            Code::Internal,
            "a resolver transport/IO fault must stay Internal, never InvalidArgument"
        );

        // Predictor: manifest-verified integrity failure — the
        // context-predictor reload arm's own typed refusal, unified onto
        // the SAME `JammiError::Model` variant the resolver surface raises.
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

        // Predictor: no bundle ever published — the predictor peer of
        // `resolver_not_published`.
        let predictor_not_published = map_engine_error(JammiError::Model {
            model_id: "cnp-1".into(),
            message: "no adapter bundle is published at 'file:///artifacts/cnp-1' for context \
                      predictor 'cnp-1' (manifest.json absent); the catalog pointer may be \
                      misdirected"
                .into(),
        });
        assert_eq!(
            predictor_not_published.code(),
            Code::InvalidArgument,
            "an unpublished predictor bundle must be a client-visible precondition failure, \
             the SAME code as an integrity failure"
        );

        // Predictor: transport/permission fault against an INTACT bundle —
        // propagated UNCHANGED, same as the resolver surface. This is the
        // SAME `StorageError::Io` variant
        // `context_predictor_reload_permission_fault_is_not_a_typed_model_error`
        // (context_predictor.rs) proves the predictor reload arm actually
        // raises against a real chmod fault.
        let predictor_transport = map_engine_error(io_fault("file:///artifacts/cnp-1"));
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

    /// W — the Caller finiteness refusal crosses the wire as
    /// `INVALID_ARGUMENT`: the same code (and the same Python exception,
    /// `InvalidArgument` / `ValueError`) as a width mismatch. Before the
    /// query was typed, a non-finite query landed on `Other` → `Internal` → a
    /// different exception class. The Stored twin is `Internal`, naming the
    /// table: a corrupt artifact is never the caller's fault.
    #[test]
    fn query_finiteness_refusal_codes_follow_provenance() {
        use jammi_db::index::{validate_query, QuerySource};
        let caller: JammiError = validate_query(vec![f32::NAN], 1, QuerySource::Caller)
            .unwrap_err()
            .into();
        let status = map_engine_error(caller);
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        // …and it round-trips as the same schema-class variant a width
        // mismatch would.
        assert!(matches!(
            error_from_status(&status),
            JammiError::Schema { .. }
        ));
        let stored: JammiError = validate_query(
            vec![f32::NAN],
            1,
            QuerySource::Stored {
                table: "docs_embeddings".into(),
            },
        )
        .unwrap_err()
        .into();
        let status = map_engine_error(stored);
        assert_eq!(status.code(), Code::Internal, "{status:?}");
        assert!(status.message().contains("docs_embeddings"), "{status:?}");
    }
}
