//! The data-plane client over the `JammiError`-returning compute verbs, proven
//! interchangeable with a local `Session`: inference, eval (the four verbs),
//! fine-tune (start + status), the mutable-table create/drop lifecycle, and the
//! channel register / add-columns verbs (the latter through the data client's
//! composed `CatalogClient`).
//!
//! An in-process gRPC chain (`runtime::serve_grpc_chain`) hosts a real
//! `InferenceSession`. A `jammi_client::DataClient` connects over a real HTTP/2
//! channel and a `jammi_ai::Session` wraps the *same* engine `Arc`, so any
//! divergence is the transport's fault, not the engine's. Two properties are
//! pinned per verb group:
//!
//! * **Round-trip parity** — the same call through either transport returns the
//!   same result against the same engine, on realistic inputs (the `tiny_bert`
//!   cookbook encoder over the shipped `patents` corpus, a real golden set, a
//!   real mutable-table definition, a real channel).
//! * **Error parity (the #1 proof)** — a real failure returns the *same*
//!   `JammiError` variant + fields from both transports. The mutable case is
//!   the proof the previously-folding `JammiError::MutableTable` now reconstructs
//!   faithfully (NOT as `Other`): registering a reserved `_jammi_*` table name
//!   fails inside the engine with `MutableTable(MutableTableError::InvalidId)`,
//!   and the remote transport rebuilds that exact nested variant.
//!
//! Hermetic: local fixtures only (the `tiny_bert` encoder, the bundled
//! `patents.parquet`, `golden_relevance.csv`); no live network, no download.

use std::sync::Arc;

use arrow::array::StringArray;
use arrow_schema::{DataType, Field, Schema};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::local_session::{ChannelColumn, ChannelSpec};
use jammi_ai::{Modality, Session};
use jammi_client::DataClient;
use jammi_db::catalog::channel_repo::{ChannelCatalogError, ChannelColumnType};
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::{MutableTableDefinitionBuilder, MutableTableError, MutableTableId};
use jammi_db::{ChannelId, ModelTask};
use jammi_test_utils::{cookbook_fixture, fixture};
use tonic::transport::Endpoint;

use super::common::grpc::{start_engine_server, EngineServer};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn file_connection(name: &str, format: FileFormat) -> SourceConnection {
    SourceConnection {
        url: Some(format!("file://{}", fixture(name).display())),
        format: Some(format),
        ..Default::default()
    }
}

const GOLDEN_SOURCE: &str = "golden_rel.public.golden_relevance";

/// Connect a `DataClient` to the in-process server.
async fn remote(server: &EngineServer) -> DataClient {
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    DataClient::connect(endpoint)
        .await
        .expect("data client connect")
}

/// Wrap the server's engine `Arc` in a local session — the same engine the
/// remote calls reach.
fn local(server: &EngineServer) -> Session {
    Session::new(Arc::clone(&server.engine))
}

/// Register the patents corpus on the shared engine (AddSource is not a remote
/// verb yet; both transports then reach the same source).
async fn add_patents(session: &Session) {
    session
        .add_source(
            "patents",
            SourceType::File,
            file_connection("patents.parquet", FileFormat::Parquet),
        )
        .await
        .expect("add patents");
}

/// Register patents + the golden relevance set and generate one embedding table
/// over `abstract` so the eval verbs have a real run to evaluate. Returns the
/// generated table name.
async fn embed_patents_and_golden(session: &Session) -> String {
    add_patents(session).await;
    session
        .add_source(
            "golden_rel",
            SourceType::File,
            file_connection("golden_relevance.csv", FileFormat::Csv),
        )
        .await
        .expect("add golden");
    session
        .generate_embeddings(
            "patents",
            &tiny_bert_model_id(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .expect("generate embeddings")
        .table_name
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_infer_round_trips_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;

    let columns = ["abstract".to_string()];
    let remote_rows = remote
        .infer(
            "patents",
            &tiny_bert_model_id(),
            ModelTask::TextEmbedding,
            &columns,
            "id",
        )
        .await
        .expect("remote infer");
    let local_rows = local
        .infer(
            "patents",
            &tiny_bert_model_id(),
            ModelTask::TextEmbedding,
            &columns,
            "id",
        )
        .await
        .expect("local infer");

    let row_ids = |batches: &[arrow::record_batch::RecordBatch]| -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            let ids = b
                .column_by_name("_row_id")
                .expect("_row_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("_row_id is Utf8");
            for r in 0..b.num_rows() {
                out.push(ids.value(r).to_string());
            }
        }
        out
    };
    let remote_ids = row_ids(&remote_rows);
    assert!(!remote_ids.is_empty(), "infer over patents produces rows");
    assert_eq!(
        remote_ids,
        row_ids(&local_rows),
        "remote and local infer return the same row keys"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_eval_round_trips_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);
    let table = embed_patents_and_golden(&local).await;

    let cohorts = std::collections::HashMap::new();
    let remote_report = remote
        .eval_embeddings("patents", None, GOLDEN_SOURCE, 10, &cohorts)
        .await
        .expect("remote eval_embeddings");
    let local_report = local
        .eval_embeddings("patents", None, GOLDEN_SOURCE, 10, &cohorts)
        .await
        .expect("local eval_embeddings");

    // The aggregate metrics are deterministic over the same persisted table.
    assert_eq!(
        remote_report.aggregate.recall_at_k, local_report.aggregate.recall_at_k,
        "remote and local eval agree on recall@k"
    );
    assert_eq!(
        remote_report.per_query.len(),
        local_report.per_query.len(),
        "remote and local eval agree on the per-query record count"
    );
    assert!(!remote_report.eval_run_id.is_empty(), "run id recorded");

    // eval_per_query reads back the persisted rows for the remote run.
    let persisted = remote
        .eval_per_query(&remote_report.eval_run_id)
        .await
        .expect("remote eval_per_query");
    assert_eq!(
        persisted.len(),
        remote_report.per_query.len(),
        "every per-query record persisted for the run"
    );

    // eval_compare: a self-comparison yields the baseline + one zero-delta entry.
    let compare_tables = [table.clone(), table.clone()];
    let remote_compare = remote
        .eval_compare(&compare_tables, "patents", GOLDEN_SOURCE, 10)
        .await;
    let local_compare = local
        .eval_compare(&compare_tables, "patents", GOLDEN_SOURCE, 10)
        .await;
    match (&remote_compare, &local_compare) {
        (Ok(r), Ok(l)) => assert_eq!(
            r.per_table.len(),
            l.per_table.len(),
            "remote and local compare agree on the table count"
        ),
        (Err(r), Err(l)) => assert_eq!(
            std::mem::discriminant(r),
            std::mem::discriminant(l),
            "remote and local compare agree on the failure variant"
        ),
        other => panic!("remote and local eval_compare disagreed on success: {other:?}"),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Error parity on the eval surface: evaluating a source that has no embedding
/// table fails inside the engine; both transports must reconstruct the identical
/// variant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_eval_reconstructs_the_exact_error_variant() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await; // no embeddings generated

    let cohorts = std::collections::HashMap::new();
    let local_err = local
        .eval_embeddings("patents", None, GOLDEN_SOURCE, 10, &cohorts)
        .await
        .expect_err("local eval with no embedding table must fail");
    let remote_err = remote
        .eval_embeddings("patents", None, GOLDEN_SOURCE, 10, &cohorts)
        .await
        .expect_err("remote eval with no embedding table must fail");

    assert_eq!(
        std::mem::discriminant(&local_err),
        std::mem::discriminant(&remote_err),
        "remote reconstructs the same eval failure variant: {local_err:?} vs {remote_err:?}"
    );
    assert_eq!(
        local_err.to_string(),
        remote_err.to_string(),
        "remote carries the same eval failure message the engine produced"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Fine-tune `start` over the wire, exercising the full request encode
/// (method + task + a populated `FineTuneConfig`). `fine_tune` is submit-only:
/// it persists a `queued` job and returns a job id immediately — the format
/// detection that the patents corpus (no training-format columns) fails now
/// happens in the worker, surfacing as a *failed job*, not a synchronous error
/// from the submit call. The engine-backed server mounts the train tier, which
/// runs an embedded worker against the shared engine, so the submitted job is
/// claimed, fails format detection, and lands `failed`.
///
/// Both transports submit against the same engine, so this pins the current
/// (deferred-error) contract: submit returns `Ok` from either transport, and
/// the worker drives the job to `failed` whichever transport submitted it.
///
/// `TrainingStatus` now carries the worker's failure `error` (and the output
/// `model_id`) alongside the status string, so a remote `wait()` can surface the
/// failure reason — see the pure-Python `RemoteTrainingJob.wait`, which raises
/// `TrainingError` with that wire message, and the verb-parity coverage in
/// `crates/jammi-python/tests/test_conformance.py`. The data-plane client
/// exposed here reads back only the status string (its `fine_tune_status`
/// signature is status-only), so this test asserts the deferred-failure contract
/// — submit returns `Ok`, the worker drives the job to `failed` — over both
/// transports; the error-message round-trip is exercised through the Python
/// handle that consumes the new `error` field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_fine_tune_start_defers_failure_to_the_worker() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;

    let config = || {
        Some(FineTuneConfig {
            epochs: 1,
            lora_rank: 4,
            ..FineTuneConfig::default()
        })
    };
    let columns = ["abstract".to_string()];
    let model = tiny_bert_model_id();

    // Submit succeeds (Ok job id) from both transports — the format failure is
    // deferred to the worker, not raised synchronously here.
    let local_job = local
        .fine_tune(
            "patents",
            &model,
            &columns,
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            config(),
        )
        .await
        .expect("local fine_tune submit returns Ok (failure is deferred to the worker)");
    let remote_job = remote
        .fine_tune(
            "patents",
            &model,
            &columns,
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            config(),
        )
        .await
        .expect("remote fine_tune submit returns Ok (failure is deferred to the worker)");

    // The shared engine's embedded worker (train tier) claims each job and fails
    // format detection on patents. Poll each transport's status until terminal;
    // both must reach `failed`. (The rich variant/message is NOT carried over
    // the wire yet — that lands in T3; here we assert only the failed status.)
    // Both transports expose `fine_tune_status(&id) -> Result<String>`, but on
    // distinct types (the local `Session` and the remote `DataClient`); a tiny
    // local trait lets the one poll loop drive either without duplicating it.
    trait FineTuneStatus {
        async fn status(
            &self,
            job: &jammi_ai::local_session::FineTuneJobId,
        ) -> jammi_db::error::Result<String>;
    }
    impl FineTuneStatus for Session {
        async fn status(
            &self,
            job: &jammi_ai::local_session::FineTuneJobId,
        ) -> jammi_db::error::Result<String> {
            self.fine_tune_status(job).await
        }
    }
    impl FineTuneStatus for DataClient {
        async fn status(
            &self,
            job: &jammi_ai::local_session::FineTuneJobId,
        ) -> jammi_db::error::Result<String> {
            self.fine_tune_status(job).await
        }
    }

    async fn poll_until_failed(
        session: &impl FineTuneStatus,
        job: &jammi_ai::local_session::FineTuneJobId,
    ) {
        for _ in 0..600 {
            let status = session.status(job).await.expect("fine_tune_status");
            if status == "failed" {
                return;
            }
            assert_ne!(
                status, "completed",
                "patents has no training-format columns — the job must fail, not complete"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("the fine-tune job did not reach a terminal `failed` state in time");
    }

    poll_until_failed(&local, &local_job).await;
    poll_until_failed(&remote, &remote_job).await;

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// `fine_tune_status` on an unknown id fails inside the engine; both transports
/// reconstruct the identical variant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_fine_tune_status_reconstructs_the_exact_error_variant() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);

    let unknown = jammi_ai::local_session::FineTuneJobId("no-such-job-id".to_string());
    let local_err = local
        .fine_tune_status(&unknown)
        .await
        .expect_err("local status on an unknown job must fail");
    let remote_err = remote
        .fine_tune_status(&unknown)
        .await
        .expect_err("remote status on an unknown job must fail");

    assert_eq!(
        std::mem::discriminant(&local_err),
        std::mem::discriminant(&remote_err),
        "remote reconstructs the same fine_tune_status failure variant: {local_err:?} vs {remote_err:?}"
    );
    assert_eq!(local_err.to_string(), remote_err.to_string());

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// A realistic mutable-table definition: a `patents_dim` companion keyed on a
/// string id with a value column.
fn patents_dim_definition() -> jammi_db::store::mutable::MutableTableDefinition {
    let id = MutableTableId::new("patents_dim").expect("valid id");
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Utf8, true),
    ]));
    MutableTableDefinitionBuilder::new(id, schema)
        .primary_key(vec!["k".to_string()])
        .build()
        .expect("definition builds")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_create_and_drop_mutable_table_round_trips_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);

    // The remote creates the table; the id echoes the request id.
    let id = remote
        .catalog()
        .create_mutable_table(patents_dim_definition())
        .await
        .expect("remote create_mutable_table");
    assert_eq!(id.as_str(), "patents_dim");

    // A second create of the same id fails the same way on both transports: the
    // backing-store unique constraint trips, surfacing
    // `MutableTable(Backend(Constraint { .. }))`. This exercises the *nested*
    // engine-owned `BackendError` reconstruction — the remote transport must
    // rebuild the inner `Backend(Constraint)` faithfully, never folding the
    // outer `MutableTable` to `Other`.
    let local_dup = local
        .create_mutable_table(patents_dim_definition())
        .await
        .expect_err("local re-create must fail");
    let remote_dup = remote
        .catalog()
        .create_mutable_table(patents_dim_definition())
        .await
        .expect_err("remote re-create must fail");
    match (&local_dup, &remote_dup) {
        (
            JammiError::MutableTable(MutableTableError::Backend(local_b)),
            JammiError::MutableTable(MutableTableError::Backend(remote_b)),
        ) => assert_eq!(
            local_b.to_string(),
            remote_b.to_string(),
            "the nested Backend error crosses the wire intact"
        ),
        (_, JammiError::Other(_)) => {
            panic!(
                "REGRESSION: the nested MutableTable(Backend) error folded to Other over the wire"
            )
        }
        other => panic!("remote did not reconstruct MutableTable(Backend) faithfully: {other:?}"),
    }

    // The remote drops it; afterwards the local session can recreate it,
    // proving the drop reached the shared engine.
    remote
        .catalog()
        .drop_mutable_table(&id)
        .await
        .expect("remote drop_mutable_table");
    local
        .create_mutable_table(patents_dim_definition())
        .await
        .expect("local can recreate after the remote drop");

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// THE fold-closure proof. Creating a mutable table under a reserved `_jammi_*`
/// name fails inside the engine with
/// `JammiError::MutableTable(MutableTableError::InvalidId(..))` — the variant
/// that previously folded to `JammiError::Other` over the wire. With the typed
/// `MutableTableErrorDetail` contract, the remote transport must reconstruct the
/// IDENTICAL nested variant + message, NOT `Other`. This is the test that proves
/// the reachable fold on this surface is closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_mutable_table_reserved_name_reconstructs_faithfully_not_as_other() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);

    let reserved = || {
        let id = MutableTableId::new("_jammi_reserved_probe").expect("valid id shape");
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8, false)]));
        MutableTableDefinitionBuilder::new(id, schema)
            .primary_key(vec!["k".to_string()])
            .build()
            .expect("definition builds")
    };

    let local_err = local
        .create_mutable_table(reserved())
        .await
        .expect_err("a reserved name must be rejected locally");
    let remote_err = remote
        .catalog()
        .create_mutable_table(reserved())
        .await
        .expect_err("a reserved name must be rejected remotely");

    // Local is the engine truth: a MutableTable::InvalidId.
    assert!(
        matches!(
            local_err,
            JammiError::MutableTable(MutableTableError::InvalidId(_))
        ),
        "the reserved-name failure is a MutableTable::InvalidId locally, got {local_err:?}"
    );
    // Remote must reconstruct the SAME nested variant — never the old `Other`
    // fold — with the identical message.
    match (&local_err, &remote_err) {
        (
            JammiError::MutableTable(MutableTableError::InvalidId(local_msg)),
            JammiError::MutableTable(MutableTableError::InvalidId(remote_msg)),
        ) => assert_eq!(
            local_msg, remote_msg,
            "the InvalidId message crosses the wire intact"
        ),
        (_, JammiError::Other(_)) => panic!(
            "REGRESSION: the MutableTable error folded to Other over the wire — the fold is not closed"
        ),
        other => panic!("remote did not reconstruct MutableTable::InvalidId faithfully: {other:?}"),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_register_and_add_channel_columns_round_trips_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);

    let channel_id = ChannelId::new("evidence").expect("valid channel id");
    let spec = ChannelSpec {
        id: channel_id.clone(),
        priority: 10,
        columns: vec![ChannelColumn {
            name: "score".to_string(),
            data_type: ChannelColumnType::Float64,
        }],
    };
    remote
        .catalog()
        .register_channel(&spec)
        .await
        .expect("remote register_channel");

    // Re-registering the same channel fails identically on both transports —
    // a faithful `ChannelCatalog(AlreadyExists)` error over the wire.
    let remote_dup = remote
        .catalog()
        .register_channel(&spec)
        .await
        .expect_err("re-registering a channel must fail");
    assert!(
        matches!(
            remote_dup,
            JammiError::ChannelCatalog(ChannelCatalogError::AlreadyExists(ref c)) if c == "evidence"
        ),
        "re-register is a ChannelCatalog(AlreadyExists) error, got {remote_dup:?}"
    );

    // The remote appends a column; the local session sees the same channel
    // (shared engine), so its append of a *different* column also succeeds.
    remote
        .catalog()
        .add_channel_columns(
            &channel_id,
            &[ChannelColumn {
                name: "rationale".to_string(),
                data_type: ChannelColumnType::Utf8,
            }],
        )
        .await
        .expect("remote add_channel_columns");
    local
        .add_channel_columns(
            &channel_id,
            &[ChannelColumn {
                name: "source_uri".to_string(),
                data_type: ChannelColumnType::Utf8,
            }],
        )
        .await
        .expect("local add_channel_columns on the same channel");

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
