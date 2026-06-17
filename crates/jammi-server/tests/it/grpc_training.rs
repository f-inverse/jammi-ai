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
    GraphSampleConfig, PredictiveHead, StartTrainingRequest, TrainingStatusRequest,
};
use jammi_test_utils::{cookbook_fixture, fixture_url};
use tonic::transport::Channel;

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
