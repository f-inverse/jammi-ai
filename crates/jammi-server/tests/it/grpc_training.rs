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
//!
//! The `context_predictor` and `graph_fine_tune` kinds are exercised end to end
//! over the wire as well, both under a tenant scope: the predictor over a
//! synthetic meta-dataset whose embedding table is stamped to the tenant (the
//! kind whose reconstruction reads are tenant-scoped), the graph kind over
//! federated node/edge CSV sources (tenant-agnostic reads that must keep
//! completing under the uniform per-job scoping).

use std::time::Duration;

use jammi_server::grpc::proto::inference::ModelTask;
use jammi_server::grpc::proto::training::start_training_request::Spec;
use jammi_server::grpc::proto::training::training_service_client::TrainingServiceClient;
use jammi_server::grpc::proto::training::{
    ContextArchitecture, ContextPredictorSpec, ContextPredictorTrainConfig, EdgeProvenance,
    FineTuneMethod, FineTuneSpec, GaussianObjective, GraphFineTuneSources, GraphFineTuneSpec,
    GraphSampleConfig, ListTrainingJobsRequest, PredictiveHead, StartTrainingRequest,
    TrainingStatusRequest,
};
use jammi_test_utils::{cookbook_fixture, fixture_url};
use tonic::transport::Channel;

#[cfg(feature = "train")]
use super::common::grpc::start_engine_server_with_run_worker;
#[cfg(feature = "train")]
use super::common::grpc::start_engine_server_worker_quiesced;
use super::common::grpc::{
    channel, start_engine_server, tenant_a, with_session, EngineServer, TENANT_A,
};

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
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{
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
            let mut catalog = CatalogServiceClient::with_interceptor(client_channel, interceptor);
            catalog.add_source(request).await.expect("add_source");
        }
        None => {
            let mut catalog = CatalogServiceClient::new(client_channel);
            catalog.add_source(request).await.expect("add_source");
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
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;

    // Bind the session (keyed by the `jammi-session-id` header) to TENANT_A,
    // then register the source and train under the same session id — every call
    // carries that header through `with_session`, so the interceptor scopes them
    // all to TENANT_A (StartTraining persists the job row under the tenant;
    // TrainingStatus reads it back under the same scope).
    let session_iface = with_session("training-tenant-a");
    let mut session_client =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
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

/// `ListTrainingJobs` returns the lifecycle projection for every job visible to
/// the session tenant, and only those: an unscoped session lists unscoped jobs
/// but never a tenant's, while a tenant-bound session lists its own jobs plus
/// the unscoped ones — the same visibility predicate `TrainingStatus` reads
/// with.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_training_jobs_is_tenant_scoped_and_carries_the_status_projection() {
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;

    // Job 1 — unscoped session.
    add_training_source(
        channel(server.addr).await,
        None::<fn(tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status>>,
    )
    .await;
    let mut unscoped = TrainingServiceClient::new(channel(server.addr).await);
    let unscoped_job = unscoped
        .start_training(start_request())
        .await
        .expect("start_training unscoped")
        .into_inner();
    poll_until_terminal(&mut unscoped, &unscoped_job.job_id).await;

    // Job 2 — a session bound to TENANT_A (its own source registration included,
    // mirroring the tenant-scope test above).
    let session_iface = with_session("training-list-tenant-a");
    let mut session_client =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    session_client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");
    // The unscoped "training" source registered above is visible to the tenant
    // session too (the same `tenant OR unscoped` visibility the job listing
    // asserts below), so no second registration is needed — registering the
    // same source id twice on one server is a conflict.
    let mut scoped =
        TrainingServiceClient::with_interceptor(channel(server.addr).await, session_iface);
    let scoped_job = scoped
        .start_training(start_request())
        .await
        .expect("start_training under tenant scope")
        .into_inner();
    poll_until_terminal(&mut scoped, &scoped_job.job_id).await;

    // The unscoped listing carries the unscoped job — with the full status
    // projection — and never the tenant's job.
    let listed = unscoped
        .list_training_jobs(ListTrainingJobsRequest {})
        .await
        .expect("list_training_jobs unscoped")
        .into_inner()
        .jobs;
    let row = listed
        .iter()
        .find(|j| j.job_id == unscoped_job.job_id)
        .expect("the unscoped listing carries the unscoped job");
    assert_eq!(row.kind, "fine_tune");
    assert_eq!(row.status, "completed");
    // `base_model_id` is the catalog's registered model id — the resolved
    // fixture path plus a version suffix — not the submit-time `local:` string.
    assert!(
        row.base_model_id
            .starts_with(&cookbook_fixture("tiny_bert").display().to_string()),
        "base_model_id is the catalog id of the submitted base model, got '{}'",
        row.base_model_id
    );
    assert_eq!(
        row.output_model_id, unscoped_job.model_id,
        "a completed row carries the output model id StartTraining returned"
    );
    assert!(!row.created_at.is_empty(), "created_at is recorded");
    assert!(row.error.is_empty(), "a completed row carries no error");
    assert!(
        listed.iter().all(|j| j.job_id != scoped_job.job_id),
        "an unscoped listing must never carry a tenant's job"
    );

    // The tenant-bound listing carries its own job AND the unscoped one (the
    // `tenant_id = $tenant OR tenant_id IS NULL` visibility), most recent first.
    let listed_a = scoped
        .list_training_jobs(ListTrainingJobsRequest {})
        .await
        .expect("list_training_jobs under tenant scope")
        .into_inner()
        .jobs;
    assert!(
        listed_a.iter().any(|j| j.job_id == scoped_job.job_id),
        "the tenant listing carries the tenant's own job"
    );
    assert!(
        listed_a.iter().any(|j| j.job_id == unscoped_job.job_id),
        "the tenant listing carries unscoped jobs too"
    );
    assert!(
        listed_a
            .windows(2)
            .all(|w| w[0].created_at >= w[1].created_at),
        "rows are ordered most recent first (created_at descending)"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

// ---------------------------------------------------------------------------
// `context_predictor` and `graph_fine_tune` end-to-end over the wire.
//
// Both kinds reconstruct from a persisted spec on the server's training worker,
// which drains every tenant's queue over the *unbound* shared session. The
// predictor's reconstruction reads its embedding table (catalog read) and the
// per-member context vectors (SQL-surface reads); those must observe the job's
// tenant, not the worker's unbound default, or a tenant-A job cannot see its
// own `tenant_id = A` embedding table and fails. The graph kind reads only
// federated sources (tenant-agnostic), so it must keep completing under a
// tenant once the per-job scoping is uniform.
// ---------------------------------------------------------------------------

const FEATURE_DIM: usize = 4;

/// splitmix64 — a deterministic generator so the synthetic meta-dataset is
/// reproducible without a test-only rng dependency.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// Stand up the predictor's source + embedding table on the server's engine
/// session **under tenant A**, mirroring the in-process
/// `session_with_meta_dataset` fixture: a linear-function meta-dataset
/// (`n_tasks` tasks, each a random weight vector over `FEATURE_DIM` features,
/// `rows_per_task` rows; outcome `y = w_task · x`), written as a CSV source
/// (`_row_id, task, y`) and a materialised embedding table whose `vector`
/// column is each row's feature `x`, keyed by `_row_id`.
///
/// Both writes run inside `with_tenant_scoped(A, …)` so the embedding-table
/// catalog row is stamped `tenant_id = A` — the row a worker draining the
/// unbound queue can only resolve if its reads are re-scoped to A. The source
/// parquet/CSV rows carry no `tenant_id` column (federated, tenant-agnostic),
/// matching production: the embedding table is the tenant-scoped read.
async fn seed_predictor_dataset_under_tenant_a(server: &EngineServer) {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let n_tasks = 8;
    let rows_per_task = 16;
    let mut rng = Rng(321);

    // Build the rows: id, task, x (feature vector), y = w_task · x.
    let mut ids = Vec::new();
    let mut tasks = Vec::new();
    let mut ys = Vec::new();
    let mut pairs: Vec<(String, Vec<f32>)> = Vec::new();
    for t in 0..n_tasks {
        let w: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
        for r in 0..rows_per_task {
            let x: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
            let y: f64 = x.iter().zip(&w).map(|(xi, wi)| (xi * wi) as f64).sum();
            let id = format!("t{t}_r{r}");
            ids.push(id.clone());
            tasks.push(format!("task_{t}"));
            ys.push(y);
            pairs.push((id, x));
        }
    }

    // Source CSV: `_row_id, task, y`. `_row_id` is the shared identity with the
    // embedding table's key column.
    let mut body = String::from("_row_id,task,y\n");
    for ((id, task), y) in ids.iter().zip(&tasks).zip(&ys) {
        body.push_str(&format!("{id},{task},{y}\n"));
    }
    let source_path = server._dir.path().join("fns_source.csv");
    std::fs::write(&source_path, body).unwrap();

    // Both the source registration and the embedding-table materialisation run
    // under tenant A so the result-table catalog row is stamped `tenant_id = A`.
    server
        .engine
        .with_tenant_scoped(tenant_a(), |_scope| async {
            server
                .engine
                .add_source(
                    "fns",
                    SourceType::File,
                    SourceConnection {
                        url: Some(format!("file://{}", source_path.display())),
                        format: Some(FileFormat::Csv),
                        ..Default::default()
                    },
                )
                .await
                .expect("add predictor source under tenant A");

            let (__d, __e, __i) =
                jammi_test_utils::synthetic_seed_contract("synthetic-embed", "fns", FEATURE_DIM);
            server
                .engine
                .result_store()
                .materialize_embedding_table(
                    server.engine.context(),
                    jammi_db::store::EmbeddingTableSpec {
                        source_id: "fns",
                        model_id: "synthetic-embed",
                        derived_from: None,
                        dimensions: FEATURE_DIM,
                        key_column: Some("_row_id"),
                        text_columns: None,
                    },
                    &pairs,
                    jammi_db::store::manifest::Materialization::new(&__d, &__e, __i),
                )
                .await
                .expect("materialize tenant-A embedding table");
        })
        .await;
}

/// A `StartTraining` request carrying the `ContextPredictorSpec` arm over the
/// `fns` source — a small CNP with a Gaussian/CRPS head, the same shape the
/// in-process predictor integration test trains.
fn predictor_start_request() -> StartTrainingRequest {
    StartTrainingRequest {
        spec: Some(Spec::ContextPredictor(ContextPredictorSpec {
            source: "fns".into(),
            predictor_spec: Some(ContextPredictorTrainConfig {
                model_id: "ctx-predictor-wire".into(),
                architecture: ContextArchitecture::Cnp as i32,
                key_column: "_row_id".into(),
                task_column: "task".into(),
                value_column: "y".into(),
                context_k: 6,
                hidden_dim: 16,
                num_heads: 2,
                num_layers: 2,
                head: Some(PredictiveHead {
                    head: Some(jammi_server::grpc::proto::training::predictive_head::Head::Gaussian(
                        jammi_server::grpc::proto::training::predictive_head::Gaussian {
                            objective: Some(GaussianObjective {
                                objective: Some(
                                    jammi_server::grpc::proto::training::gaussian_objective::Objective::Crps(
                                        jammi_server::grpc::proto::training::gaussian_objective::Crps {},
                                    ),
                                ),
                            }),
                        },
                    )),
                }),
                epochs: 20,
                learning_rate: 0.005,
                grad_clip: 1.0,
                test_task_fraction: 0.25,
                min_task_count: 4,
                seed: 7,
            }),
        })),
        // The predictor carries its full budget in `predictor_spec`; no base
        // model / LoRA config applies.
        base_model: String::new(),
        config: None,
    }
}

/// A `context_predictor` job submitted under tenant A over the wire trains to
/// completion: the source + embedding table are stamped `tenant_id = A`, the
/// server's (unbound) training worker claims the job, and — because the whole
/// claimed-job run executes in the job's tenant scope — its reconstruction
/// reads resolve the tenant-A embedding table and per-member vectors. Before
/// the per-job scoping fix the unbound reads could not see the `tenant_id = A`
/// table and the job landed `failed`; this asserts it reaches `completed` and
/// the predictor is registered under A.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_predictor_under_a_tenant_scope_completes_over_the_wire() {
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;
    seed_predictor_dataset_under_tenant_a(&server).await;

    // Bind a session id to TENANT_A so every wire call carries that scope.
    let session_iface = with_session("predictor-tenant-a");
    let mut session_client =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    session_client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");

    let mut client =
        TrainingServiceClient::with_interceptor(channel(server.addr).await, session_iface);

    let start = client
        .start_training(predictor_start_request())
        .await
        .expect("start_training(context_predictor) under tenant scope")
        .into_inner();
    assert!(!start.job_id.is_empty(), "StartTraining returns a job id");
    assert_eq!(
        start.model_id, "ctx-predictor-wire",
        "the predictor's deterministic model id is returned"
    );

    let resp = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        resp.status, "completed",
        "a tenant-scoped context_predictor job must reach `completed` (the worker's \
         reconstruction reads must observe tenant A's embedding table), got '{}' (error: {})",
        resp.status, resp.error
    );
    assert_eq!(
        resp.model_id, "ctx-predictor-wire",
        "a completed predictor job carries its registered model id"
    );

    // The predictor model row is registered under tenant A: visible inside A's
    // scope, invisible to an unscoped catalog read.
    let under_a = server
        .engine
        .with_tenant_scoped(tenant_a(), |_scope| {
            server.engine.catalog().get_model("ctx-predictor-wire")
        })
        .await
        .expect("get_model under tenant A");
    assert!(
        under_a.is_some(),
        "the trained predictor is registered under tenant A"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// A `graph_fine_tune` job submitted under a tenant runs to completion over the
/// wire. Its reconstruction reads only federated node/edge sources (tenant
/// agnostic), so the uniform per-job tenant scoping must not regress it — this
/// closes the wire-coverage gap for the graph kind alongside the predictor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_fine_tune_under_a_tenant_scope_completes_over_the_wire() {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;

    // Two small communities + a bridge, directed both ways so walks traverse.
    let node_ids = ["a0", "a1", "a2", "b0", "b1", "b2"];
    let mut node_body = String::from("id,text\n");
    for id in node_ids {
        node_body.push_str(&format!("{id},document about topic {id}\n"));
    }
    let node_path = server._dir.path().join("graph_nodes.csv");
    std::fs::write(&node_path, node_body).unwrap();

    let edge_pairs = [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ];
    let mut edge_body = String::from("src,dst\n");
    for (s, d) in edge_pairs {
        edge_body.push_str(&format!("{s},{d}\n"));
    }
    let edge_path = server._dir.path().join("graph_edges.csv");
    std::fs::write(&edge_path, edge_body).unwrap();

    // Register the sources on the engine under tenant A (federated; rows carry
    // no tenant column, so the reads are tenant-agnostic by construction).
    for (id, path) in [("nodes", &node_path), ("edges", &edge_path)] {
        server
            .engine
            .with_tenant_scoped(tenant_a(), |_scope| async {
                server
                    .engine
                    .add_source(
                        id,
                        SourceType::File,
                        SourceConnection {
                            url: Some(format!("file://{}", path.display())),
                            format: Some(FileFormat::Csv),
                            ..Default::default()
                        },
                    )
                    .await
                    .expect("add graph source under tenant A");
            })
            .await;
    }

    let session_iface = with_session("graph-tenant-a");
    let mut session_client =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    session_client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");

    let mut client =
        TrainingServiceClient::with_interceptor(channel(server.addr).await, session_iface);

    let request = StartTrainingRequest {
        spec: Some(Spec::GraphFineTune(GraphFineTuneSpec {
            sources: Some(GraphFineTuneSources {
                node_source: "nodes".into(),
                id_column: "id".into(),
                text_column: "text".into(),
                edge_source: "edges".into(),
                src_column: "src".into(),
                dst_column: "dst".into(),
                provenance: EdgeProvenance::Declared as i32,
            }),
            sample_config: Some(GraphSampleConfig {
                walk_length: 3,
                walks_per_node: 2,
                return_p: 1.0,
                in_out_q: 1.0,
                hard_negatives: 1,
                exclude_hops: 1,
                min_negatives: 1,
                seed: 11,
            }),
        })),
        base_model: tiny_bert_model_id(),
        config: None,
    };

    let start = client
        .start_training(request)
        .await
        .expect("start_training(graph_fine_tune) under tenant scope")
        .into_inner();
    assert!(!start.job_id.is_empty());

    let resp = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        resp.status, "completed",
        "a tenant-scoped graph_fine_tune job must reach `completed`, got '{}' (error: {})",
        resp.status, resp.error
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

// ---------------------------------------------------------------------------
// esc-075: `TrainingStatus.acceleration_report_json` (campaign #443).
//
// The tri-state esc-075 marker (`NULL` legacy-unknown / `{"state":"pending"}`
// / `{"state":"determined",...}`) rides the wire on
// `TrainingStatusResponse.acceleration_report_json`, appended after
// `metrics_json` (field 5) with the identical presence contract: field
// presence — never the empty string — distinguishes "no column value" from an
// already-recorded blob, mirroring the catalog's `NULL`/`NOT NULL`. The
// ai-core worker-side writer has not landed yet (a parallel wave adds it), so
// these tests seed the column directly through the catalog API rather than
// driving a real claim-time determination.
// ---------------------------------------------------------------------------

/// Directly seed a `training_jobs` row, bypassing
/// [`jammi_db::catalog::Catalog::create_training_job`] (which always lands
/// `status = 'queued'`). The row this writes never passes through `'queued'`,
/// so it is race-free against the server's own background `TrainingWorker` —
/// mounted alongside `TrainingService` by every `start_engine_server*` fixture
/// — which claims exclusively `WHERE status = 'queued'` and would otherwise
/// compete for a freshly-queued row nondeterministically.
async fn seed_training_job_row(
    catalog: &jammi_db::catalog::Catalog,
    job_id: &str,
    base_model_id: &str,
    status: &str,
    claimed_by: Option<&str>,
    attempts: i64,
    acceleration_report: Option<&str>,
) {
    use jammi_db::catalog::backend::{SqlNullType, SqlValue, TxOptions};

    let job_id = job_id.to_string();
    let base_model_id = base_model_id.to_string();
    let status = status.to_string();
    let claimed_by = claimed_by.map(str::to_string);
    let acceleration_report = acceleration_report.map(str::to_string);
    // Far enough in the future that `reclaim_expired_training_jobs`'s
    // `lease_expires_at < now` sweep (run on every worker tick) never reclaims
    // this row mid-test.
    let lease_expires_at = claimed_by
        .is_some()
        .then(|| "9999-12-31T23:59:59.000000Z".to_string());

    catalog
        .backend_arc()
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO training_jobs \
                     (job_id, base_model_id, training_source, loss_type, hyperparams, status, \
                      kind, training_spec, tenant_id, acceleration_report, claimed_by, attempts, \
                      lease_expires_at) \
                     VALUES ($1, $2, 'seed.csv', 'contrastive', '{}', $3, 'fine_tune', $4, $5, \
                             $6, $7, $8, $9)",
                    &[
                        SqlValue::TextOwned(job_id),
                        SqlValue::TextOwned(base_model_id),
                        SqlValue::TextOwned(status),
                        SqlValue::Null(SqlNullType::Text),
                        SqlValue::Null(SqlNullType::Text),
                        SqlValue::from(acceleration_report),
                        SqlValue::from(claimed_by),
                        SqlValue::Int(attempts),
                        SqlValue::from(lease_expires_at),
                    ],
                )
                .await
            })
        })
        .await
        .expect("seed training_jobs row");
}

/// Register the FK target model every seeded row's `base_model_id` points at.
async fn register_acceleration_test_model(catalog: &jammi_db::catalog::Catalog, model_id: &str) {
    use jammi_db::catalog::model_repo::RegisterModelParams;

    catalog
        .register_model(RegisterModelParams {
            model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: jammi_db::ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .expect("register acceleration-report test model");
}

/// K4 (esc-075): a submitted job's `TrainingStatus.acceleration_report_json`
/// matches the catalog's `training_jobs.acceleration_report` column
/// byte-for-byte, in the submission-time state — the explicit `{"state":
/// "pending"}` marker `create_training_job` writes, never `NULL`. This is the
/// divergence-prone tri-state field, not a single-scalar happy path.
///
/// THE READ POINT IS QUIESCED BY CONSTRUCTION (#446 finding 8): this fixture is
/// [`start_engine_server_worker_quiesced`], whose embedded training worker has
/// been stopped AND joined before the fixture returns, so between the submit and
/// the two reads below there is NO claimant that could move the marker off
/// `pending`. Reading the pre-claim state under the production fixture (worker
/// running) is a TOCTOU: the worker claims a `queued` row on its first tick with
/// no initial sleep, so on a slow runner the claim lands first — see
/// `an_eagerly_running_worker_moves_the_acceleration_marker_off_pending` below,
/// the control that demonstrates the mutation is real. The fix is the quiesced
/// read point, NOT a weaker "pending-or-terminal" assertion: the byte-exact
/// marker IS the K4 oracle and stays byte-exact.
///
/// Then the worker is released and the SAME parity assertion is re-run at the
/// second quiesced point — the job's terminal state, with the worker stopped
/// again — so the wire↔embedded byte-equality is pinned on both ends of the
/// job's life, each read where nothing can mutate the row.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn training_status_acceleration_report_pending_state_matches_the_catalog_record() {
    let server = start_engine_server_worker_quiesced().await;
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

    // Quiesced read point #1 — post-submission, pre-claim: no worker exists in
    // this process and the catalog is the fixture's own temp dir, so no other
    // claimant can exist either. The wire field and the embedded catalog read
    // must carry the identical pending marker.
    let resp = client
        .training_status(TrainingStatusRequest {
            job_id: start.job_id.clone(),
        })
        .await
        .expect("training_status")
        .into_inner();
    let embedded = server
        .engine
        .catalog()
        .get_training_job(&start.job_id)
        .await
        .expect("get_training_job");
    // The fixture's quiescence is itself asserted, so this test can never pass
    // vacuously against a job that was already claimed: a claimed row's status
    // is `running`/terminal, never `queued`.
    assert_eq!(
        resp.status, "queued",
        "the quiesced fixture must leave the submitted job unclaimed — a \
         non-`queued` status here means a claimant ran and the read below is \
         no longer a pre-claim read"
    );
    assert_eq!(
        resp.acceleration_report_json.as_deref(),
        Some(r#"{"state":"pending"}"#),
        "a freshly submitted job's wire acceleration_report_json must carry the \
         explicit pending marker, never absent or empty"
    );
    assert_eq!(
        resp.acceleration_report_json, embedded.acceleration_report,
        "TrainingStatus.acceleration_report_json must BYTE-EQUAL the embedded \
         catalog record's acceleration_report column for the SAME job"
    );

    // Release the worker: the identical `EmbeddedWorker` the `train` tier
    // spawns, over the identical engine session the server drives. The job now
    // runs, and its terminal state is awaited through the PUBLIC wire surface.
    let worker = server.spawn_training_worker();
    let terminal = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        terminal.status, "completed",
        "the released worker must run the submitted job to completion, got '{}' \
         (error: {})",
        terminal.status, terminal.error
    );

    // Quiesced read point #2 — the job is terminal AND the worker is stopped and
    // joined again, so nothing can write the row between the two reads below.
    worker
        .stop_and_join()
        .await
        .expect("stop the released training worker");
    let after = client
        .training_status(TrainingStatusRequest {
            job_id: start.job_id.clone(),
        })
        .await
        .expect("training_status")
        .into_inner();
    let embedded_after = server
        .engine
        .catalog()
        .get_training_job(&start.job_id)
        .await
        .expect("get_training_job");
    assert_eq!(
        after.acceleration_report_json, embedded_after.acceleration_report,
        "at the job's terminal state the wire acceleration_report_json must \
         still BYTE-EQUAL the embedded catalog record's column"
    );
    assert_eq!(
        acceleration_state(after.acceleration_report_json.as_deref()),
        Some("determined".to_string()),
        "a completed job's report must have moved off the pending marker to the \
         determined state, got {:?}",
        after.acceleration_report_json
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// NON-VACUITY CONTROL for the quiesced fixture above (#446 finding 8): under
/// the PRODUCTION fixture — `start_engine_server`, whose `train` tier spawns and
/// keeps an embedded worker — the submission-time `{"state":"pending"}` marker
/// is a TRANSIENT observable, not a stable one. This test submits the same job,
/// lets the running worker claim it, and shows that once the job is observed
/// claimed-and-terminal the very same field no longer reads `pending`.
///
/// That is exactly the mutation the old assertion shape raced: a byte-exact
/// `pending` assert placed after the submit, against a fixture with a live
/// worker, is asserting a value the worker is concurrently overwriting. This
/// control fails if that mutation ever stops happening — which would make the
/// quiesced read point above pointless — so the fix cannot rot into a no-op.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_eagerly_running_worker_moves_the_acceleration_marker_off_pending() {
    // The production fixture: the `train` tier's embedded worker is running and
    // claims `queued` rows on its first tick.
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

    // Observe the job past the claim, at a terminal state (the one point that IS
    // stable under a running worker).
    let terminal = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        terminal.status, "completed",
        "the eager worker must run the submitted job to completion, got '{}' \
         (error: {})",
        terminal.status, terminal.error
    );
    assert_ne!(
        terminal.acceleration_report_json.as_deref(),
        Some(r#"{"state":"pending"}"#),
        "CONTROL: a claimed-and-run job's report must NOT still read the \
         submission-time pending marker — if it did, the pending marker would \
         be stable and the quiesced fixture unnecessary"
    );
    assert_eq!(
        acceleration_state(terminal.acceleration_report_json.as_deref()),
        Some("determined".to_string()),
        "CONTROL: the claiming worker overwrites the pending marker with the \
         determined report, got {:?}",
        terminal.acceleration_report_json
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// The `state` discriminant of an `acceleration_report_json` field, or `None`
/// for the field-absent (legacy `NULL`) case. Panics on a present-but-malformed
/// payload — a report that is not valid JSON with a string `state` is its own
/// loud failure, never folded into "absent".
fn acceleration_state(field: Option<&str>) -> Option<String> {
    let raw = field?;
    let value: serde_json::Value = serde_json::from_str(raw).unwrap_or_else(|e| {
        panic!("acceleration_report_json is not valid JSON: {e} (raw={raw:?})")
    });
    Some(
        value
            .get("state")
            .and_then(|s| s.as_str())
            .unwrap_or_else(|| {
                panic!("acceleration_report_json carries no string `state`: {value}")
            })
            .to_string(),
    )
}

/// K4 (esc-075), the determined-state and legacy-NULL halves: a claiming
/// worker's `{"state":"determined",...}` report, and a pre-migration-026 row's
/// `NULL`, both round-trip over `TrainingStatus.acceleration_report_json`
/// byte-for-byte against the embedded catalog read. Absence semantics: the
/// legacy row surfaces as field-ABSENT (`None`), never the empty string.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn training_status_acceleration_report_determined_and_legacy_states_match_the_catalog_record()
{
    let server = start_engine_server().await;
    let catalog = server.engine.catalog();
    register_acceleration_test_model(catalog, "acc-test-base").await;

    // A claimed, running job carrying the explicit pending marker — the exact
    // shape `create_training_job` + a real claim produce — which this test
    // then overwrites with a determined report through the same catalog API
    // the claiming worker itself would call.
    seed_training_job_row(
        catalog,
        "acc-determined",
        "acc-test-base::1",
        "running",
        Some("acc-test-worker"),
        1,
        Some(r#"{"state":"pending"}"#),
    )
    .await;
    let determined_json = r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#;
    let wrote = catalog
        .record_acceleration_report("acc-determined", "acc-test-worker", 1, determined_json)
        .await
        .expect("record_acceleration_report");
    assert!(
        wrote,
        "the seeded row's lease/attempt must match the write guard"
    );

    // A legacy row: `acceleration_report` is SQL `NULL`, never touched by any
    // migration-026-aware writer — the pre-migration-026 shape.
    seed_training_job_row(
        catalog,
        "acc-legacy",
        "acc-test-base::1",
        "completed",
        None,
        0,
        None,
    )
    .await;

    let mut client = TrainingServiceClient::new(channel(server.addr).await);

    for (job_id, expect_absent) in [("acc-determined", false), ("acc-legacy", true)] {
        let resp = client
            .training_status(TrainingStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .expect("training_status")
            .into_inner();
        let embedded = catalog
            .get_training_job(job_id)
            .await
            .expect("get_training_job");
        assert_eq!(
            resp.acceleration_report_json, embedded.acceleration_report,
            "TrainingStatus.acceleration_report_json for '{job_id}' must BYTE-EQUAL \
             the embedded catalog record's acceleration_report column"
        );
        if expect_absent {
            assert_eq!(
                resp.acceleration_report_json, None,
                "a legacy NULL row must surface as field-ABSENT over the wire, \
                 never the empty string"
            );
        } else {
            assert_eq!(
                resp.acceleration_report_json.as_deref(),
                Some(determined_json),
                "a determined report must round-trip verbatim over the wire"
            );
        }
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// esc-075 remote-visibility control: a REMOTE caller — reading nothing but
/// `TrainingStatusResponse.acceleration_report_json` — can distinguish all
/// three tri-state values (legacy-unknown / pending / determined) purely from
/// the response. This is `esc-075-f16-silent-eager-no-per-job-signal`'s
/// closure proof: the tri-state marker must survive the wire round-trip
/// losslessly, not merely "both transports respond".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_caller_distinguishes_acceleration_report_tri_state_purely_from_the_response() {
    #[derive(Debug, PartialEq, Eq)]
    enum AccelerationState {
        LegacyUnknown,
        Pending,
        Determined,
    }

    /// Classify purely off the wire response — no catalog / engine access.
    /// Reads the `state` discriminant through the one shared
    /// [`acceleration_state`] decoder, so this control and the pending/terminal
    /// assertions above cannot drift into two different notions of "the state".
    fn classify(field: Option<&str>) -> AccelerationState {
        match acceleration_state(field).as_deref() {
            None => AccelerationState::LegacyUnknown,
            Some("pending") => AccelerationState::Pending,
            Some("determined") => AccelerationState::Determined,
            other => panic!("unrecognized acceleration_report state: {other:?}"),
        }
    }

    let server = start_engine_server().await;
    let catalog = server.engine.catalog();
    register_acceleration_test_model(catalog, "acc-vis-base").await;

    seed_training_job_row(
        catalog,
        "acc-vis-legacy",
        "acc-vis-base::1",
        "completed",
        None,
        0,
        None,
    )
    .await;
    // A claimed, running row still carrying the pending marker — never
    // overwritten, so it stays distinguishable from the determined row below.
    seed_training_job_row(
        catalog,
        "acc-vis-pending",
        "acc-vis-base::1",
        "running",
        Some("acc-vis-worker-1"),
        1,
        Some(r#"{"state":"pending"}"#),
    )
    .await;
    seed_training_job_row(
        catalog,
        "acc-vis-determined",
        "acc-vis-base::1",
        "running",
        Some("acc-vis-worker-2"),
        1,
        Some(r#"{"state":"pending"}"#),
    )
    .await;
    let determined_json = r#"{"state":"determined","fa2_f16":false}"#;
    assert!(
        catalog
            .record_acceleration_report(
                "acc-vis-determined",
                "acc-vis-worker-2",
                1,
                determined_json
            )
            .await
            .expect("record_acceleration_report"),
        "the seeded row's lease/attempt must match the write guard"
    );

    let mut client = TrainingServiceClient::new(channel(server.addr).await);
    async fn read(client: &mut TrainingServiceClient<Channel>, job_id: &str) -> Option<String> {
        client
            .training_status(TrainingStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .expect("training_status")
            .into_inner()
            .acceleration_report_json
    }

    assert_eq!(
        classify(read(&mut client, "acc-vis-legacy").await.as_deref()),
        AccelerationState::LegacyUnknown
    );
    assert_eq!(
        classify(read(&mut client, "acc-vis-pending").await.as_deref()),
        AccelerationState::Pending
    );
    assert_eq!(
        classify(read(&mut client, "acc-vis-determined").await.as_deref()),
        AccelerationState::Determined
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

// ---------------------------------------------------------------------------
// GAP-A-2 (#446): `[training] run_worker` — whether THIS process claims.
//
// The `train` tier mounts `TrainingService` unconditionally; whether the same
// process ALSO runs the claim loop is a configuration key, not a second code
// path and not a build feature. The three tests below drive that key through
// the real `jammi.toml` → `JammiConfig::load` path the binary uses (the
// `start_engine_server_with_run_worker` fixture), never through
// `ChainParts::train_worker` — the seam a downstream owns, which would prove
// only that stopping a worker stops it.
// ---------------------------------------------------------------------------

/// Read `TrainingStatus` over the wire and the SAME job's catalog record in
/// process, returning both. The K4 cross-transport pair: whatever the remote
/// surface reports for the acceleration marker must byte-equal the embedded
/// read of the identical row.
#[cfg(feature = "train")]
async fn wire_and_embedded(
    client: &mut TrainingServiceClient<Channel>,
    server: &EngineServer,
    job_id: &str,
) -> (
    jammi_server::grpc::proto::training::TrainingStatusResponse,
    jammi_db::catalog::training_repo::TrainingJobRecord,
) {
    let wire = client
        .training_status(TrainingStatusRequest {
            job_id: job_id.to_string(),
        })
        .await
        .expect("training_status")
        .into_inner();
    let embedded = server
        .engine
        .catalog()
        .get_training_job(job_id)
        .await
        .expect("get_training_job");
    (wire, embedded)
}

/// THE BINDING ORACLE for `run_worker = false`: a server whose `train` tier is
/// mounted from a `jammi.toml` carrying `run_worker = false` accepts a
/// submission over `TrainingService` and then never claims it — the job's
/// `queued` status and its byte-exact `{"state":"pending"}` acceleration marker
/// are STABLE, not merely observed once.
///
/// Stability is what makes this an oracle rather than a lucky read. The polls
/// span strictly MORE than the config's own `idle_poll_secs` (the interval at
/// which a claim loop, if one existed in this process, would tick and claim a
/// `queued` row — its first tick has no initial sleep at all), so a spawned
/// worker could not hide inside the observation window. Both fields are
/// re-asserted at EVERY poll, so a claim landing at any point in the span fails
/// the test.
///
/// Cross-transport (K4) at every poll: the remotely-read
/// `acceleration_report_json` byte-equals the embedded catalog read of the same
/// row, and the two surfaces agree on the status. A pending marker that the wire
/// reported but the embedded read did not (or vice versa) is a divergence even
/// while the value itself is "right".
///
/// The control that this is caused by the KNOB and not by a server that never
/// runs anything is
/// `train_tier_with_run_worker_true_lets_the_submitted_job_leave_queued`, which
/// runs the identical body against the identical fixture with the key flipped.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn train_tier_with_run_worker_false_leaves_the_job_queued_and_pending_stable() {
    let server = start_engine_server_with_run_worker(false).await;
    add_training_source(
        channel(server.addr).await,
        None::<fn(tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status>>,
    )
    .await;
    let mut client = TrainingServiceClient::new(channel(server.addr).await);

    // The submission surface is unaffected by the key: `TrainingService` is
    // mounted and `StartTraining` succeeds exactly as it does with a worker.
    let start = client
        .start_training(start_request())
        .await
        .expect(
            "start_training must succeed with run_worker = false — the \
                 train tier still MOUNTS TrainingService; only claiming is off",
        )
        .into_inner();
    assert!(
        !start.job_id.is_empty(),
        "StartTraining returns a job id regardless of run_worker"
    );

    // Span strictly more than one idle poll interval, read off the server's OWN
    // config so the window follows the knob's timing rather than a magic number.
    let idle_poll =
        Duration::from_secs(server.engine.inner_config().training.idle_poll_secs.max(1));
    const POLLS: u32 = 6;
    let step = (idle_poll * 2) / POLLS;
    let began = std::time::Instant::now();

    for poll in 0..POLLS {
        let (wire, embedded) = wire_and_embedded(&mut client, &server, &start.job_id).await;
        assert_eq!(
            wire.status, "queued",
            "poll {poll}: with run_worker = false NOTHING in this process may \
             claim the job — it must still read `queued`"
        );
        assert_eq!(
            embedded.status, "queued",
            "poll {poll}: the embedded catalog read must agree the job is still \
             queued"
        );
        assert_eq!(
            embedded.claimed_by, None,
            "poll {poll}: an unclaimed job carries no claimant — a `claimed_by` \
             here means a claim loop ran despite run_worker = false"
        );
        assert_eq!(
            wire.acceleration_report_json.as_deref(),
            Some(r#"{"state":"pending"}"#),
            "poll {poll}: the submission-time acceleration marker must be STABLE \
             byte-for-byte under run_worker = false"
        );
        assert_eq!(
            wire.acceleration_report_json, embedded.acceleration_report,
            "poll {poll}: K4 — the remotely-read acceleration_report_json must \
             BYTE-EQUAL the embedded catalog read of the same row"
        );
        if poll + 1 < POLLS {
            tokio::time::sleep(step).await;
        }
    }

    assert!(
        began.elapsed() > idle_poll,
        "the observation window ({:?}) must exceed one idle_poll_secs ({idle_poll:?}) — \
         otherwise a claim loop could have been hiding between the polls",
        began.elapsed()
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// THE CONTROL for the oracle above: the same fixture, the same submission, the
/// same tier set — with `run_worker = true` (the default) in the `jammi.toml`.
/// The job LEAVES `queued`.
///
/// Without this, the stable-`queued` assertion above would pass just as well
/// against a server that could never run anything at all (a broken train tier, a
/// wedged catalog), so it proves the knob is the cause. It also pins the default
/// direction: an unconfigured deployment is a whole one.
///
/// Deliberately NO per-poll wire↔embedded parity assertion here, unlike the
/// `run_worker = false` oracle: with a live claimant the row is being mutated
/// between the two reads, so a per-poll byte-equality would be asserting against
/// a moving target (exactly the TOCTOU the quiesced fixture exists to remove).
/// The cross-transport leg is asserted at the job's TERMINAL state instead —
/// `completed`/`failed` is absorbing, so a wire read of it cannot be overtaken
/// by the embedded read that follows.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn train_tier_with_run_worker_true_lets_the_submitted_job_leave_queued() {
    let server = start_engine_server_with_run_worker(true).await;
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

    // Bounded well past the window the run_worker = false test held the job
    // `queued` across, so "left queued" here is a real difference in behaviour
    // and not a difference in patience.
    let mut left_queued = None;
    for _ in 0..600 {
        let wire = client
            .training_status(TrainingStatusRequest {
                job_id: start.job_id.clone(),
            })
            .await
            .expect("training_status")
            .into_inner();
        if wire.status != "queued" {
            left_queued = Some(wire);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let observed = left_queued.expect(
        "CONTROL: with run_worker = true the train tier's worker must claim the \
         submitted job — it never left `queued`, which would make the \
         run_worker = false oracle vacuous",
    );
    assert_ne!(
        observed.status, "queued",
        "the claimed job has left the queue"
    );

    // Cross-transport at the one point that is stable under a live worker: the
    // terminal state. `completed` is absorbing, so the embedded read taken after
    // the wire read cannot have moved on to something else.
    let terminal = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        terminal.status, "completed",
        "the claimed job runs to completion, got '{}' (error: {})",
        terminal.status, terminal.error
    );
    let embedded = server
        .engine
        .catalog()
        .get_training_job(&start.job_id)
        .await
        .expect("get_training_job");
    assert_eq!(
        terminal.status, embedded.status,
        "K4: at the terminal state the remote status and the embedded catalog \
         status must agree for the same job"
    );
    assert_ne!(
        terminal.acceleration_report_json.as_deref(),
        Some(r#"{"state":"pending"}"#),
        "CONTROL: a claimed-and-run job's marker must have moved off the \
         submission-time pending value — that mutation is precisely what \
         run_worker = false suppresses"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Shutting a `run_worker = false` server down must not await a worker that was
/// never started: no hang, no panic.
///
/// `assemble_grpc_chain` leaves `AssembledChain`'s worker slot `None` in this
/// configuration, so the serve loop's teardown has nothing extra to drop or
/// join. The bound below is what makes the assertion an assertion — a teardown
/// that waited on a never-spawned worker would sit here until the timeout
/// rather than returning, and the serve task's join result is unwrapped so a
/// panic inside the drop path surfaces as a failure instead of a silent
/// `JoinError`.
///
/// A job is submitted first, so the queue is non-empty at shutdown: the
/// "nothing to wait for" property must hold with work outstanding, which is the
/// state a `run_worker = false` process is normally in.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_worker_false_shutdown_does_not_await_a_worker_that_never_started() {
    let server = start_engine_server_with_run_worker(false).await;
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
    let (wire, _) = wire_and_embedded(&mut client, &server, &start.job_id).await;
    assert_eq!(
        wire.status, "queued",
        "the job is outstanding at shutdown — that is the state this teardown \
         must handle"
    );
    // Drop the client (and its channel) so the graceful drain has no live
    // connection to wait on; what remains is the worker slot, which is `None`.
    drop(client);

    let _ = server.shutdown.send(());
    let joined = tokio::time::timeout(Duration::from_secs(30), server.handle)
        .await
        .expect(
            "shutdown with run_worker = false must not hang — nothing spawned a \
             worker, so teardown has no worker to await",
        );
    joined.expect("the serve task must end cleanly, not by panicking, when no worker was spawned");
}

// ---------------------------------------------------------------------------
// `TrainingStatus.model_id` ⇄ the embedded attach handle (K4, #446).
//
// The divergence these close: attaching to a job by id, the EMBEDDED handle
// reports the deterministic output model id re-derived from the persisted row
// (the engine's naming rule), while the remote handle read `""` because
// `TrainingStatus` relayed the catalog's `output_model_id` column, which is
// stamped only at completion. Both surfaces now resolve through the ONE engine
// function, so the pre-completion states — the divergence-prone ones, not the
// terminal happy path — carry a byte-identical id on both transports.
//
// The embedded-arm oracle in each test is
// `jammi_ai::fine_tune::training_job::resolve_model_id` applied to the embedded
// catalog read of the SAME row: exactly the call (and the value) the embedded
// attach handle reports. It is triangulated against the id `StartTraining`
// itself returned — the value the in-process submit handle carries — so the
// assertion cannot pass by both sides sharing one wrong answer.
// ---------------------------------------------------------------------------

/// The id an ATTACHED embedded handle would report for this job: the engine's
/// own resolution over the embedded catalog read of the same row.
async fn embedded_attach_model_id(server: &EngineServer, job_id: &str) -> String {
    let record = server
        .engine
        .catalog()
        .get_training_job(job_id)
        .await
        .expect("get_training_job");
    jammi_ai::fine_tune::training_job::resolve_model_id(job_id, &record)
        .expect("the embedded arm resolves this job's model id")
}

/// K4: a `fine_tune` job that has NOT completed reports the same `model_id`
/// over the wire that the embedded attach handle derives — byte-for-byte, at
/// the pre-completion state, which is precisely where the two surfaces diverged
/// (`""` remotely vs the derived id in process).
///
/// The read point is quiesced by construction (the fixture's worker is stopped
/// AND joined before it returns), so `queued` here is a state nothing can move
/// underneath the two reads — the same discipline the acceleration-marker
/// parity tests use.
///
/// Non-vacuity is asserted three ways: the wire status really is `queued` (a
/// pre-completion read, not a terminal one), the catalog column really is
/// unstamped (`output_model_id IS NULL` — so the wire value is DERIVED, not
/// relayed), and the value equals the non-empty id `StartTraining` returned.
///
/// The terminal leg is then re-asserted after the worker is released: at
/// `completed` the wire id must still equal both the catalog's now-stamped
/// column and the submit-time id, so closing the pre-completion gap does not
/// disturb the state that already agreed.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn training_status_model_id_matches_the_embedded_derived_id_before_completion() {
    let server = start_engine_server_worker_quiesced().await;
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
    assert!(
        !start.model_id.is_empty(),
        "StartTraining returns the deterministic output model id — the value the \
         embedded submit handle carries"
    );

    // Quiesced read point: post-submission, pre-claim.
    let (wire, embedded) = wire_and_embedded(&mut client, &server, &start.job_id).await;
    assert_eq!(
        wire.status, "queued",
        "the quiesced fixture must leave the job unclaimed — a non-`queued` \
         status means this is no longer a pre-completion read"
    );
    assert_eq!(
        embedded.output_model_id, None,
        "the catalog has NOT stamped output_model_id before completion — without \
         this the wire value below could be a plain column relay and the test \
         would prove nothing"
    );
    assert_eq!(
        wire.model_id,
        embedded_attach_model_id(&server, &start.job_id).await,
        "K4: TrainingStatus.model_id must BYTE-EQUAL the id the embedded attach \
         handle derives for the same job at the same pre-completion state"
    );
    assert_eq!(
        wire.model_id, start.model_id,
        "the pre-completion wire id is the SAME deterministic id StartTraining \
         returned at submit time"
    );

    // Release the worker and re-assert at the terminal state: the leg that
    // already agreed must stay green.
    let worker = server.spawn_training_worker();
    let terminal = poll_until_terminal(&mut client, &start.job_id).await;
    assert_eq!(
        terminal.status, "completed",
        "the released worker must run the job to completion, got '{}' (error: {})",
        terminal.status, terminal.error
    );
    worker
        .stop_and_join()
        .await
        .expect("stop the released training worker");

    let (after, embedded_after) = wire_and_embedded(&mut client, &server, &start.job_id).await;
    assert_eq!(
        embedded_after.output_model_id.as_deref(),
        Some(after.model_id.as_str()),
        "at completion the wire id is the catalog's stamped output_model_id"
    );
    assert_eq!(
        after.model_id, start.model_id,
        "a completed job's wire id is still the submit-time id — the terminal \
         parity is unchanged by the pre-completion fix"
    );
    assert_eq!(
        after.model_id,
        embedded_attach_model_id(&server, &start.job_id).await,
        "K4 at the terminal state: the wire id and the embedded attach id agree"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// K4 across the REST of the lifecycle: a `running` row and a `failed` row —
/// neither of which ever stamps `output_model_id` — report the embedded attach
/// handle's derived id over the wire, not the empty string.
///
/// A `failed` job is the sharp case: it is TERMINAL yet has no stamped column,
/// so a "populated once terminal" reading of the old contract would still leave
/// it empty on the wire while the embedded handle names the model the job would
/// have produced.
///
/// The rows are seeded directly (never passing through `queued`), so they are
/// race-free against the fixture's own running worker, which claims exclusively
/// `WHERE status = 'queued'`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn training_status_model_id_is_derived_for_running_and_failed_rows() {
    let server = start_engine_server().await;
    let catalog = server.engine.catalog();
    register_acceleration_test_model(catalog, "model-id-base").await;

    seed_training_job_row(
        catalog,
        "model-id-running",
        "model-id-base::1",
        "running",
        Some("model-id-worker"),
        1,
        Some(r#"{"state":"pending"}"#),
    )
    .await;
    seed_training_job_row(
        catalog,
        "model-id-failed",
        "model-id-base::1",
        "failed",
        None,
        1,
        None,
    )
    .await;

    let mut client = TrainingServiceClient::new(channel(server.addr).await);

    for (job_id, status) in [
        ("model-id-running", "running"),
        ("model-id-failed", "failed"),
    ] {
        let resp = client
            .training_status(TrainingStatusRequest {
                job_id: job_id.to_string(),
            })
            .await
            .expect("training_status")
            .into_inner();
        let embedded = catalog
            .get_training_job(job_id)
            .await
            .expect("get_training_job");
        assert_eq!(
            resp.status, status,
            "the seeded row must be read back in the state it was seeded in"
        );
        assert_eq!(
            embedded.output_model_id, None,
            "'{job_id}' has no stamped output_model_id — the wire value below is \
             a derivation, not a relay"
        );
        assert_eq!(
            resp.model_id,
            embedded_attach_model_id(&server, job_id).await,
            "K4: a '{status}' job's TrainingStatus.model_id must BYTE-EQUAL the \
             embedded attach handle's derived id"
        );
        assert!(
            !resp.model_id.is_empty(),
            "a '{status}' job names the model it produces (or would have \
             produced); the empty string was the divergence"
        );
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// K4 for the OTHER derivation arm: a `context_predictor` job's model id is
/// caller-chosen (it rides inside the persisted `training_spec`), NOT derivable
/// from the job id, so the pre-completion wire read must decode the persisted
/// spec exactly as the embedded attach does. A handler that assumed the
/// fine-tune format would pass the fine-tune tests above and be wrong here.
///
/// Tenant-scoped throughout — the predictor's source and embedding table are
/// stamped `tenant_id = A`, so both the submit and the embedded record read run
/// inside A's scope.
#[cfg(feature = "train")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn training_status_model_id_decodes_the_predictor_spec_before_completion() {
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server_worker_quiesced().await;
    seed_predictor_dataset_under_tenant_a(&server).await;

    let session_iface = with_session("predictor-model-id-tenant-a");
    let mut session_client =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    session_client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");
    let mut client =
        TrainingServiceClient::with_interceptor(channel(server.addr).await, session_iface);

    let start = client
        .start_training(predictor_start_request())
        .await
        .expect("start_training(context_predictor) under tenant scope")
        .into_inner();
    assert_eq!(
        start.model_id, "ctx-predictor-wire",
        "the predictor's model id is the caller-chosen id inside its spec"
    );

    let resp = client
        .training_status(TrainingStatusRequest {
            job_id: start.job_id.clone(),
        })
        .await
        .expect("training_status")
        .into_inner();
    assert_eq!(
        resp.status, "queued",
        "the quiesced fixture must leave the predictor job unclaimed — this is a \
         pre-completion read"
    );

    // The embedded arm's own answer, resolved inside tenant A's scope (the job
    // row is stamped to A).
    let (embedded_record, embedded_model_id) = server
        .engine
        .with_tenant_scoped(tenant_a(), |_scope| async {
            let record = server
                .engine
                .catalog()
                .get_training_job(&start.job_id)
                .await
                .expect("get_training_job under tenant A");
            let model_id =
                jammi_ai::fine_tune::training_job::resolve_model_id(&start.job_id, &record)
                    .expect("the embedded arm resolves the predictor job's model id");
            (record, model_id)
        })
        .await;
    assert_eq!(
        embedded_record.output_model_id, None,
        "the predictor row has not stamped output_model_id before completion"
    );
    assert_eq!(
        resp.model_id, embedded_model_id,
        "K4: the predictor job's wire model_id must BYTE-EQUAL the id the \
         embedded attach handle decodes out of the persisted training_spec"
    );
    assert_eq!(
        resp.model_id, "ctx-predictor-wire",
        "and that id is the caller-chosen one, never a jammi:fine-tuned: id \
         derived from the job id"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
