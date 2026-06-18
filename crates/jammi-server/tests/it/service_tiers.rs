//! Service-tier mounting, end-to-end over the wire.
//!
//! These pin the S8 contract that capability matches deployment:
//!
//! * A **serve-only** deployment (core tier only) mounts no train verbs and
//!   advertises `services = ["core"]`; reaching `TrainingService` is a truthful
//!   `Unimplemented`, never a misleading success.
//! * An **all-in-one** deployment advertises every compiled-in tier and the
//!   same train verb is reachable (it fails on its *arguments*, not because the
//!   service is unmounted) — proving the gating, not the engine, is what made
//!   the serve-only call `Unimplemented`.
//! * The `GetServerInfo.services` handshake reports exactly the mounted set, so
//!   a client can negotiate capability without probing each verb.
//!
//! Hermetic: in-process engine over a temp catalog; no live network. The
//! train-verb probe sends a request that the *engine* would reject on its
//! arguments (unspecified method), so on the all-in-one server the error is
//! `InvalidArgument` (the verb ran), and on the serve-only server it is
//! `Unimplemented` (the verb is not mounted) — the two codes are exactly the
//! "ran but bad input" vs "not enabled here" distinction the tier gate draws.

use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::inference::ModelTask;
use jammi_server::grpc::proto::training::start_training_request::Spec;
use jammi_server::grpc::proto::training::training_service_client::TrainingServiceClient;
use jammi_server::grpc::proto::training::{FineTuneMethod, FineTuneSpec, StartTrainingRequest};
use jammi_server::tiers::{ServiceTier, TierSet};

use super::common::grpc::{channel, start_engine_server_with_tiers};

/// A StartTraining request the *engine* rejects on its arguments (unspecified
/// method). On a server that mounts the train tier this returns
/// `InvalidArgument` — the verb ran. On a serve-only server it returns
/// `Unimplemented` — the verb is not mounted at all.
fn probe_request() -> StartTrainingRequest {
    StartTrainingRequest {
        spec: Some(Spec::FineTune(FineTuneSpec {
            source: "training".into(),
            columns: vec!["text_a".into(), "text_b".into(), "score".into()],
            method: FineTuneMethod::Unspecified as i32,
            task: ModelTask::TextEmbedding as i32,
        })),
        base_model: "local:does-not-matter".into(),
        config: None,
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
async fn serve_only_advertises_core_and_rejects_train_verb_as_unimplemented() {
    // Core only — no optional tiers.
    let server =
        start_engine_server_with_tiers(TierSet::resolve(std::iter::empty()).expect("core-only"))
            .await;

    // The handshake advertises exactly the core tier.
    assert_eq!(
        server_info_services(server.addr).await,
        vec!["core".to_string()],
        "a serve-only deployment advertises only the core tier"
    );

    // Reaching the train verb is a truthful "not enabled on this deployment".
    let mut client = TrainingServiceClient::new(channel(server.addr).await);
    let err = client
        .start_training(probe_request())
        .await
        .expect_err("train verb is not mounted on a serve-only deployment");
    assert_eq!(
        err.code(),
        tonic::Code::Unimplemented,
        "an unmounted tier's verb is Unimplemented, not a misleading success"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_plus_train_advertises_train_and_mounts_the_verb() {
    // Core + the train tier.
    let server = start_engine_server_with_tiers(
        TierSet::resolve([ServiceTier::Train]).expect("train resolves when compiled"),
    )
    .await;

    let services = server_info_services(server.addr).await;
    assert!(
        services.contains(&"core".to_string()) && services.contains(&"train".to_string()),
        "serve+train advertises both core and train, got {services:?}"
    );

    // The verb is now mounted: the same probe that was Unimplemented on the
    // serve-only server runs and is rejected on its *arguments* instead.
    let mut client = TrainingServiceClient::new(channel(server.addr).await);
    let err = client
        .start_training(probe_request())
        .await
        .expect_err("the engine rejects the unspecified method");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "the train verb is mounted, so the error is the engine's argument check, \
         not an unmounted-service Unimplemented"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_only_advertises_core_and_event() {
    // Core + the event tier (the surface a downstream consumer builds on).
    let server = start_engine_server_with_tiers(
        TierSet::resolve([ServiceTier::Event]).expect("event resolves"),
    )
    .await;

    let services = server_info_services(server.addr).await;
    assert_eq!(
        services,
        vec!["core".to_string(), "event".to_string()],
        "an event box advertises core + event"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
