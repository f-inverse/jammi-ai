//! `TrainingService` end-to-end over the wire.
//!
//! An in-process Tonic server hosts the gRPC chain including the
//! `TrainingService`. A client registers the shipped `training_pairs.csv`
//! fixture as a source (through the embedding service's `AddSource`, which backs
//! onto the same engine session), starts a minimal LoRA fine-tune over its
//! contrastive `(text_a, text_b, score)` columns with the local `tiny_bert`
//! cookbook encoder via the `FineTuneSpec` arm of `StartTraining`, then polls
//! `TrainingStatus` until a terminal state and asserts the job completed with the
//! output model id `StartTraining` returned. This pins the wire adapter's
//! contract: `StartTraining` returns a `job_id` + a deterministic `model_id`,
//! `TrainingStatus` is poll-based (no progress stream) and carries the output
//! model id + the failure error.
//!
//! Hermetic: the encoder is a local fixture (no network, no download), the
//! training corpus is the shipped CSV, and the run is a 1-epoch projection-head
//! LoRA over the tiny 32-dim model. A tenant-scoped run is covered too.

use std::time::Duration;

use jammi_server::grpc::proto::inference::ModelTask;
use jammi_server::grpc::proto::training::start_training_request::Spec;
use jammi_server::grpc::proto::training::training_service_client::TrainingServiceClient;
use jammi_server::grpc::proto::training::{
    FineTuneMethod, FineTuneSpec, StartTrainingRequest, TrainingStatusRequest,
};
use jammi_test_utils::{cookbook_fixture, fixture_url};
use tonic::transport::Channel;

use super::common::grpc::{channel, start_engine_server, with_session, TENANT_A};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn training_url() -> String {
    fixture_url("training_pairs.csv")
}

/// The contrastive columns the engine detects as `(text_a, text_b, score)`
/// training data.
fn training_columns() -> Vec<String> {
    vec!["text_a".into(), "text_b".into(), "score".into()]
}

/// Register the training source through the embedding service's `AddSource`
/// (both services back onto the same engine session, so a source registered on
/// one is visible to the other). When `session` is supplied the call is bound
/// to that session's tenant.
async fn add_training_source(
    client_channel: Channel,
    session: Option<
        impl Fn(tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status> + Clone,
    >,
) {
    use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
    use jammi_server::grpc::proto::embedding::{
        AddSourceRequest, FileFormat, SourceConnection, SourceKind,
    };
    let request = AddSourceRequest {
        source_id: "training".into(),
        source_kind: SourceKind::File as i32,
        connection: Some(SourceConnection {
            url: training_url(),
            format: FileFormat::Csv as i32,
        }),
    };
    match session {
        Some(interceptor) => {
            let mut embedding =
                EmbeddingServiceClient::with_interceptor(client_channel, interceptor);
            embedding.add_source(request).await.expect("add_source");
        }
        None => {
            let mut embedding = EmbeddingServiceClient::new(client_channel);
            embedding.add_source(request).await.expect("add_source");
        }
    }
}

/// A `StartTraining` request carrying the `FineTuneSpec` arm for a minimal
/// projection-head LoRA over the training source: an absent `config` keeps the
/// engine defaults; the small fixture + tiny model keep the run short.
fn start_request() -> StartTrainingRequest {
    StartTrainingRequest {
        spec: Some(Spec::FineTune(FineTuneSpec {
            source: "training".into(),
            columns: training_columns(),
            method: FineTuneMethod::Lora as i32,
            task: ModelTask::TextEmbedding as i32,
        })),
        base_model: tiny_bert_model_id(),
        // Defaults: projection head (empty target_modules), 3 epochs. The tiny
        // fixture + 32-dim model keep this within the engine's own fine-tune
        // test runtime.
        config: None,
    }
}

/// Poll `TrainingStatus` until the job reaches a terminal state, returning the
/// full response (status + output model id + error). Bounded so a wedged job
/// fails the test instead of hanging. Generic over the client's transport so the
/// plain and tenant-intercepted clients (distinct concrete types) share one
/// poller.
async fn poll_until_terminal<T>(
    client: &mut TrainingServiceClient<T>,
    job_id: &str,
) -> jammi_server::grpc::proto::training::TrainingStatusResponse
where
    T: tonic::client::GrpcService<tonic::body::Body>,
    T::Error: Into<tonic::codegen::StdError>,
    T::ResponseBody:
        tonic::transport::Body<Data = tonic::codegen::Bytes> + std::marker::Send + 'static,
    <T::ResponseBody as tonic::transport::Body>::Error:
        Into<tonic::codegen::StdError> + std::marker::Send,
{
    for _ in 0..600 {
        let resp = client
            .training_status(TrainingStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .expect("training_status")
            .into_inner();
        if resp.status == "completed" || resp.status == "failed" {
            return resp;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("training job '{job_id}' did not reach a terminal state in time");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_training_runs_to_completion_over_the_wire() {
    let server = start_engine_server().await;
    add_training_source(
        channel(server.addr).await,
        None::<fn(tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status>>,
    )
    .await;

    let mut client = TrainingServiceClient::new(channel(server.addr).await);

    let start = client
        .start_training(start_request())
        .await
        .expect("start_training")
        .into_inner();
    assert!(!start.job_id.is_empty(), "StartTraining returns a job id");
    assert!(
        !start.model_id.is_empty(),
        "StartTraining returns the deterministic output model id"
    );

    let resp = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        resp.status, "completed",
        "the minimal LoRA fine-tune should complete, got '{}' (error: {})",
        resp.status, resp.error
    );
    // On completion the status response carries the output model id (the catalog
    // `output_model_id`), and no error.
    assert!(
        !resp.model_id.is_empty(),
        "a completed job's status carries the output model id"
    );
    assert!(resp.error.is_empty(), "a completed job carries no error");

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn training_under_a_tenant_scope_succeeds_over_the_wire() {
    use jammi_server::grpc::proto::session::session_service_client::SessionServiceClient;
    use jammi_server::grpc::proto::session::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;

    // Bind the session (keyed by the `jammi-session-id` header) to TENANT_A,
    // then register the source and train under the same session id — every call
    // carries that header through `with_session`, so the interceptor scopes them
    // all to TENANT_A (StartTraining persists the job row under the tenant;
    // TrainingStatus reads it back under the same scope).
    let session_iface = with_session("training-tenant-a");
    let mut session_client =
        SessionServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    session_client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");

    add_training_source(channel(server.addr).await, Some(session_iface.clone())).await;

    let mut client =
        TrainingServiceClient::with_interceptor(channel(server.addr).await, session_iface);

    let start = client
        .start_training(start_request())
        .await
        .expect("start_training under tenant scope")
        .into_inner();
    assert!(!start.job_id.is_empty());

    let resp = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        resp.status, "completed",
        "tenant-scoped training should complete, got '{}'",
        resp.status
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_training_rejects_unspecified_method() {
    let server = start_engine_server().await;
    let mut client = TrainingServiceClient::new(channel(server.addr).await);

    let err = client
        .start_training(StartTrainingRequest {
            spec: Some(Spec::FineTune(FineTuneSpec {
                method: FineTuneMethod::Unspecified as i32,
                source: "training".into(),
                columns: training_columns(),
                task: ModelTask::TextEmbedding as i32,
            })),
            base_model: tiny_bert_model_id(),
            config: None,
        })
        .await
        .expect_err("unspecified method must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_training_rejects_missing_columns() {
    let server = start_engine_server().await;
    let mut client = TrainingServiceClient::new(channel(server.addr).await);

    let err = client
        .start_training(StartTrainingRequest {
            spec: Some(Spec::FineTune(FineTuneSpec {
                columns: Vec::new(),
                source: "training".into(),
                method: FineTuneMethod::Lora as i32,
                task: ModelTask::TextEmbedding as i32,
            })),
            base_model: tiny_bert_model_id(),
            config: None,
        })
        .await
        .expect_err("missing columns must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
