//! Service-tier mounting, end-to-end over the wire.
//!
//! These pin the S8 contract that capability matches deployment:
//!
//! * A **serve-only** deployment (core tier only) advertises `services =
//!   ["core"]` and mounts no eval verbs; reaching `EvalService` is a truthful
//!   `Unimplemented`, never a misleading success. Job submission is core, so
//!   the same serve-only deployment DOES answer `JobService` — whether
//!   it also runs those jobs is `[worker] enabled`, not a tier.
//! * A **core + eval** deployment advertises both and the same eval verb is
//!   reachable (it fails on its *arguments*, not because the service is
//!   unmounted) — proving the gating, not the engine, is what made the
//!   serve-only call `Unimplemented`.
//! * The `GetServerInfo.services` handshake reports exactly the mounted set, so
//!   a client can negotiate capability without probing each verb.
//!
//! Hermetic: in-process engine over a temp catalog; no live network. Each
//! probe sends a request that the *engine* would reject on its arguments (an
//! empty eval run id; an unspecified fine-tune method), so on a server that
//! mounts the verb the error is `InvalidArgument` (the verb ran), and on one
//! that does not it is `Unimplemented` (the verb is not mounted) — the two
//! codes are exactly the "ran but bad input" vs "not enabled here" distinction
//! the tier gate draws.

use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::eval::eval_service_client::EvalServiceClient;
use jammi_server::grpc::proto::eval::EvalPerQueryRequest;
use jammi_server::grpc::proto::inference::ModelTask;
use jammi_server::grpc::proto::job::job_service_client::JobServiceClient;
use jammi_server::grpc::proto::job::{submit_job_request::Spec, SubmitJobRequest};
use jammi_server::grpc::proto::training::{FineTuneMethod, FineTuneSpec};
use jammi_server::tiers::{ServiceTier, TierSet};

use super::common::grpc::{channel, start_engine_server_with_tiers};

/// An EvalPerQuery request the *engine* rejects on its arguments (empty run
/// id). On a server that mounts the eval tier this returns `InvalidArgument` —
/// the verb ran. On a serve-only server it returns `Unimplemented` — the verb
/// is not mounted at all.
fn eval_probe_request() -> EvalPerQueryRequest {
    EvalPerQueryRequest {
        eval_run_id: String::new(),
        tenant_id: String::new(),
    }
}

/// A SubmitJob request the *engine* rejects on its arguments (unspecified
/// method). `JobService` is core, so this returns `InvalidArgument` on
/// EVERY deployment — the verb ran — never `Unimplemented`.
fn training_probe_request() -> SubmitJobRequest {
    SubmitJobRequest {
        spec: Some(Spec::FineTune(FineTuneSpec {
            source: "training".into(),
            columns: vec!["text_a".into(), "text_b".into(), "score".into()],
            method: FineTuneMethod::Unspecified as i32,
            task: ModelTask::TextEmbedding as i32,
        })),
        base_model: "local:does-not-matter".into(),
        config: None,
        idempotency_key: String::new(),
    }
}

async fn server_info_services(addr: std::net::SocketAddr) -> Vec<String> {
    let mut client = CatalogServiceClient::new(channel(addr).await);
    client
        .get_server_info(())
        .await
        .expect("get_server_info")
        .into_inner()
        .services
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_only_advertises_core_and_rejects_eval_verb_as_unimplemented() {
    // Core only — no optional tiers.
    let server = start_engine_server_with_tiers(TierSet::resolve(std::iter::empty())).await;

    // The handshake advertises exactly the core tier.
    assert_eq!(
        server_info_services(server.addr).await,
        vec!["core".to_string()],
        "a serve-only deployment advertises only the core tier"
    );

    // Reaching the eval verb is a truthful "not enabled on this deployment".
    let mut eval = EvalServiceClient::new(channel(server.addr).await);
    let err = eval
        .eval_per_query(eval_probe_request())
        .await
        .expect_err("eval verb is not mounted on a serve-only deployment");
    assert_eq!(
        err.code(),
        tonic::Code::Unimplemented,
        "an unmounted tier's verb is Unimplemented, not a misleading success"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Job submission is core: a serve-only deployment (no optional tiers) still
/// mounts `JobService`, so the same probe is rejected on its ARGUMENTS
/// here — the verb ran. Whether this process runs the job it accepted is the
/// `[worker] enabled` key (`grpc_job.rs` proves that knob), not a tier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_only_still_mounts_the_job_submission_verb() {
    let server = start_engine_server_with_tiers(TierSet::resolve(std::iter::empty())).await;

    let mut client = JobServiceClient::new(channel(server.addr).await);
    let err = client
        .submit_job(training_probe_request())
        .await
        .expect_err("the engine rejects the unspecified method");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "JobService is core: the error is the engine's argument check on \
         every deployment, never an unmounted-service Unimplemented"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_plus_eval_advertises_eval_and_mounts_the_verb() {
    // Core + the eval tier.
    let server = start_engine_server_with_tiers(TierSet::resolve([ServiceTier::Eval])).await;

    let services = server_info_services(server.addr).await;
    assert_eq!(
        services,
        vec!["core".to_string(), "eval".to_string()],
        "serve+eval advertises exactly core and eval"
    );

    // The verb is now mounted: the same probe that was Unimplemented on the
    // serve-only server runs and is rejected on its *arguments* instead.
    let mut eval = EvalServiceClient::new(channel(server.addr).await);
    let err = eval
        .eval_per_query(eval_probe_request())
        .await
        .expect_err("the engine rejects the empty run id");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "the eval verb is mounted, so the error is the engine's argument check, \
         not an unmounted-service Unimplemented"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_only_advertises_core_and_event() {
    // Core + the event tier (the surface a downstream consumer builds on).
    let server = start_engine_server_with_tiers(TierSet::resolve([ServiceTier::Event])).await;

    let services = server_info_services(server.addr).await;
    assert_eq!(
        services,
        vec!["core".to_string(), "event".to_string()],
        "an event box advertises core + event"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
