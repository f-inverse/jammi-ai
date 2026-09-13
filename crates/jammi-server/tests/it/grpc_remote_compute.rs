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

use super::common::grpc::{start_engine_server, start_engine_server_with_devices, EngineServer};

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
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect("generate embeddings")
        .0
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
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect("remote infer")
        .0;
    let local_rows = local
        .infer(
            "patents",
            &tiny_bert_model_id(),
            ModelTask::TextEmbedding,
            &columns,
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect("local infer")
        .0;

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
    // K4 byte-parity beyond the keys: the identical schema (with
    // `_ordinal`), the identical `_ordinal` values, and every column's bytes
    // — only the per-row latency may differ between the two runs.
    assert!(
        remote_rows[0]
            .schema()
            .column_with_name("_ordinal")
            .is_some(),
        "the remote infer result carries `_ordinal`"
    );
    assert_batches_byte_identical(&remote_rows, &local_rows, &["_latency_ms"]);

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
/// from the submit call. The engine-backed server runs an embedded worker
/// (`[worker] enabled`, the test config's default) against the shared engine,
/// so the submitted job is claimed, fails format detection, and lands `failed`.
///
/// Both transports submit against the same engine, so this pins the current
/// (deferred-error) contract: submit returns `Ok` from either transport, and
/// the worker drives the job to `failed` whichever transport submitted it.
///
/// `JobStatus` now carries the worker's failure `error` (and the output
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

    // The shared engine's embedded worker claims each job and fails
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

/// The contrastive columns the engine detects as `(text_a, text_b, score)`
/// training data (mirrors `grpc_job.rs`'s fixture).
fn training_pairs_columns() -> Vec<String> {
    vec!["text_a".into(), "text_b".into(), "score".into()]
}

/// Register the shipped contrastive `training_pairs.csv` fixture as a source
/// on the shared engine (both transports then reach the same source, matching
/// `add_patents`).
async fn add_training_pairs(session: &Session) {
    session
        .add_source(
            "training",
            SourceType::File,
            file_connection("training_pairs.csv", FileFormat::Csv),
        )
        .await
        .expect("add training_pairs");
}

/// K4 (issue #441): a completed fine-tune job's run-metrics blob round-trips
/// over the wire, on the divergence-prone case — the multi-epoch
/// `train_loss_curve` / `val_loss_curve` arrays, not a single scalar. The
/// embedded surface reads the catalog's `training_jobs.metrics` column
/// directly (the same read `jammi-python`'s `TrainingJob.metrics()`
/// performs); the remote surface reads it back through the NEW
/// `JobStatus`'s result.model.metrics_json wire field.
///
/// THE byte-equality parity oracle is the SAME-JOB comparison: this test
/// submits `remote_job` once and reads its metrics back through two
/// independent paths — the data-plane `DataClient::fine_tune_metrics` and the
/// control-plane `CatalogClient::job_status` — and asserts those two
/// reads decode byte-identical `metrics_json`. Any divergence there is
/// unambiguously the wire adapter's fault, not the engine's, because both
/// reads observe the one job's one stored blob.
///
/// `local_job` and `remote_job` themselves are a SEPARATE, weaker,
/// cross-job comparison: two INDEPENDENT training runs (not two reads of one
/// job), each stamping its own `started_at`/`completed_at` wall clock. Their
/// full JSON is intentionally NOT asserted byte-identical here — that would
/// silently assume this fixture is bit-deterministic across independent
/// (and, under this test's multi-thread runtime, potentially concurrent)
/// executions, which this suite does not establish for the real BERT-backed
/// embedding path. The cross-job comparison is instead scoped to fields that
/// are deterministic BY CONSTRUCTION — the top-level key set, the
/// `early_stopping_metric` config echo, `total_steps`, and (pinned
/// absolutely, not merely compared) every curve's row COUNT, the
/// divergence-prone multi-row case this test exists to catch. See the
/// in-line comments at the comparison sites for the full derivation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_fine_tune_metrics_round_trips_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_training_pairs(&local).await;

    let columns = training_pairs_columns();
    let model = tiny_bert_model_id();

    let local_job = local
        .fine_tune(
            "training",
            &model,
            &columns,
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .expect("local fine_tune submit");
    let remote_job = remote
        .fine_tune(
            "training",
            &model,
            &columns,
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .expect("remote fine_tune submit");

    // Poll both to `completed` through the shared status-only surface (the
    // same tiny local trait `remote_fine_tune_start_defers_failure_to_the_worker`
    // uses to drive one poll loop over both transport types).
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
    async fn poll_until_completed(
        session: &impl FineTuneStatus,
        job: &jammi_ai::local_session::FineTuneJobId,
    ) {
        for _ in 0..600 {
            let status = session.status(job).await.expect("fine_tune_status");
            if status == "completed" {
                return;
            }
            assert_ne!(
                status, "failed",
                "the minimal LoRA fine-tune over training_pairs.csv must complete"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("fine-tune job did not reach `completed` in time");
    }
    poll_until_completed(&local, &local_job).await;
    poll_until_completed(&remote, &remote_job).await;

    // Embedded read: straight off the catalog record, the same field
    // `jammi-python`'s `TrainingJob.metrics()` decodes.
    let local_record = server
        .engine
        .catalog()
        .get_job(&local_job.0)
        .await
        .expect("local get_job");
    let local_metrics: serde_json::Value = {
        // The generalised `jobs.result` tagged payload
        // (`jammi_ai::jobs::JobResult::Model`) nests the raw metrics JSON as
        // a STRING field — the generalised `jobs` schema has no dedicated
        // metrics column of its own (C1b/N8).
        let result_raw = local_record
            .result
            .as_deref()
            .expect("a completed local job carries a result");
        let result_value: serde_json::Value =
            serde_json::from_str(result_raw).unwrap_or_else(|e| {
                panic!("local job's result is not valid JSON: {e} (raw={result_raw:?})")
            });
        let raw = result_value
            .get("metrics")
            .and_then(|m| m.as_str())
            .expect("a completed local job's result carries a metrics blob");
        // Distinct from the absence check above: a metrics blob that IS
        // present but fails to parse as JSON must be its own loud failure,
        // never folded into "no blob" — the `.ok()` swallow this leg used to
        // repeat (the BLOCK-4 sibling: `jammi-python`'s `job.rs` collapsed the
        // identical malformed-present state into `{}`).
        serde_json::from_str(raw).unwrap_or_else(|e| {
            panic!("local job's metrics blob is present but is not valid JSON: {e} (raw={raw:?})")
        })
    };

    // Remote read: the new `JobStatus`'s result.model.metrics_json wire field.
    let remote_metrics_json = remote
        .fine_tune_metrics(&remote_job)
        .await
        .expect("remote fine_tune_metrics")
        .expect("a completed remote job carries a metrics blob");
    let remote_metrics: serde_json::Value =
        serde_json::from_str(&remote_metrics_json).expect("remote metrics_json is valid JSON");

    // The control-plane read (`CatalogClient::job_status`, composed on
    // `DataClient`) must not silently lag the data-plane read of the SAME
    // wire field (family M lockstep): both decode the identical
    // the terminal result's metrics_json.
    let admin_status = remote
        .catalog()
        .job_status(&remote_job.0)
        .await
        .expect("catalog job_status");
    assert_eq!(
        admin_status.metrics_json.as_deref(),
        Some(remote_metrics_json.as_str()),
        "CatalogClient::job_status's metrics_json must match \
         DataClient::fine_tune_metrics's — both decode the same wire field"
    );

    // Cross-job (local vs remote) comparison below: `local_job` and
    // `remote_job` are TWO INDEPENDENT training runs, not two reads of the
    // same job — the same-job byte-equality oracle already ran above
    // (`admin_status.metrics_json` vs `remote_metrics_json`: two DIFFERENT
    // read paths over the SAME job's SAME wire field). A full-JSON equality
    // claim across two independent runs is only honest if this fixture is
    // bit-deterministic across independent executions, and this suite does
    // not establish that: both jobs share the default seed
    // (`DEFAULT_FINE_TUNE_SEED`, `crates/jammi-wire/src/fine_tune.rs:517`),
    // and `crates/jammi-ai/src/fine_tune/trainer.rs`'s
    // `same_seed_byte_identical_through_trained_forward` proves a same-seed
    // CPU run is byte-identical through the production
    // forward/backward/AdamW path — but that proof is for two SEQUENTIAL
    // calls on one task, over a simpler regression fixture with no
    // validation split. This test's two jobs are driven by independent
    // worker attempts that can execute CONCURRENTLY under this test's
    // multi-thread runtime, and nothing in this codebase establishes that
    // CPU floating-point reduction stays order-independent under concurrent
    // scheduling for the real BERT-backed embedding path these jobs actually
    // run. Rather than assert full-JSON equality on an unproven determinism
    // claim, this narrows the cross-job comparison to fields that are
    // deterministic BY CONSTRUCTION regardless of any float
    // non-associativity: the top-level key set (both runs measure/omit the
    // same fields), the `early_stopping_metric` echo (a config echo, never a
    // measurement), and `total_steps` (a pure function of epoch count /
    // batch size / row count, never of randomness). Curve LENGTHS — the
    // divergence-prone multi-row case — are pinned absolutely below; their
    // per-row float VALUES are intentionally excluded from this cross-job
    // comparison.
    fn key_set(v: &serde_json::Value) -> Vec<&str> {
        let mut keys: Vec<&str> = v
            .as_object()
            .expect("metrics blob is a JSON object")
            .keys()
            .map(String::as_str)
            .filter(|k| *k != "started_at" && *k != "completed_at")
            .collect();
        keys.sort_unstable();
        keys
    }
    assert_eq!(
        key_set(&local_metrics),
        key_set(&remote_metrics),
        "the two independent runs must measure/omit the same metrics fields \
         (ignoring the two per-run wall-clock timestamps): \
         local={local_metrics:?} remote={remote_metrics:?}"
    );
    assert_eq!(
        local_metrics.get("early_stopping_metric"),
        remote_metrics.get("early_stopping_metric"),
        "early_stopping_metric is a config echo, not a measurement — it must \
         match across independent runs of the identical config: \
         local={local_metrics:?} remote={remote_metrics:?}"
    );
    assert_eq!(
        local_metrics.get("total_steps"),
        remote_metrics.get("total_steps"),
        "total_steps is a pure function of epoch count / batch size / row \
         count — deterministic by construction, never a function of \
         randomness — so it must match across independent runs of the \
         identical config and data: local={local_metrics:?} \
         remote={remote_metrics:?}"
    );

    // The divergence-prone case itself: multi-row curve arrays, not a single
    // scalar. Default config trains 3 epochs with `early_stopping_metric:
    // ValLoss`, so both curves populate on every completed run.
    let curve_len =
        |v: &serde_json::Value, key: &str| v.get(key).and_then(|c| c.as_array()).map(|a| a.len());
    assert_eq!(
        curve_len(&local_metrics, "train_loss_curve"),
        Some(3),
        "3 epochs must leave a 3-row train_loss_curve locally, got {local_metrics:?}"
    );
    assert_eq!(
        curve_len(&remote_metrics, "train_loss_curve"),
        curve_len(&local_metrics, "train_loss_curve"),
        "the wire train_loss_curve must have the SAME row count as the embedded one \
         (the divergence-prone multi-row case, not just \"both non-empty\")"
    );
    // `val_loss_curve` is pinned ABSOLUTELY on both arms, exactly like
    // `train_loss_curve` above — not merely compared for equal-length-with-
    // remote, which would pass vacuously if both sides simply omitted the
    // key. The default `FineTuneConfig`
    // (`crates/jammi-wire/src/fine_tune.rs:486-520`) runs `epochs: 3` with
    // `early_stopping_metric: EarlyStoppingMetric::ValLoss` (`:505`), so
    // `avg_val_loss` is measured every epoch and a row is pushed onto
    // `val_loss_curve` (`trainer.rs`'s `val_loss_curve.push`, immediately
    // before the early-stopping break decision) on every epoch that runs.
    // The default `early_stopping_patience` is also `3` (`fine_tune.rs:502`),
    // but the patience counter can accumulate at most 2 non-improving epochs
    // across a 3-epoch run (epoch 1 always sets the initial `best_val_loss`
    // baseline, so `patience_counter` starts its climb from epoch 2), so
    // `patience_counter >= early_stopping_patience` (3) can never fire within
    // only 3 configured epochs — early stopping cannot truncate this
    // fixture, and the curve is unconditionally 3 rows long on both arms.
    // Corroborated in-tree by
    // `trainer.rs::metrics_json_loss_curves_match_the_epoch_complete_tracing_event`,
    // which pins the identical "ValLoss measured every epoch -> 3 rows"
    // derivation on the trainer's own unit-test fixture.
    assert_eq!(
        curve_len(&local_metrics, "val_loss_curve"),
        Some(3),
        "3 epochs with early_stopping_metric: ValLoss must leave a 3-row \
         val_loss_curve locally, got {local_metrics:?}"
    );
    assert_eq!(
        curve_len(&remote_metrics, "val_loss_curve"),
        Some(3),
        "3 epochs with early_stopping_metric: ValLoss must leave a 3-row \
         val_loss_curve over the wire, got {remote_metrics:?}"
    );

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

/// Every column of `a` and `b` — schema (names and types, after the
/// `Utf8View`→`Utf8` normalization a registered-table scan needs) and the
/// bytes of every row — must agree, except the columns named in `skip`.
fn assert_batches_byte_identical(
    a: &[arrow::record_batch::RecordBatch],
    b: &[arrow::record_batch::RecordBatch],
    skip: &[&str],
) {
    let a = normalized_single_batch(a);
    let b = normalized_single_batch(b);
    assert_eq!(
        a.schema().fields(),
        b.schema().fields(),
        "remote and local must read back the identical schema"
    );
    assert_eq!(a.num_rows(), b.num_rows());
    for (field, (col_a, col_b)) in a
        .schema()
        .fields()
        .iter()
        .zip(a.columns().iter().zip(b.columns()))
    {
        if skip.contains(&field.name().as_str()) {
            continue;
        }
        assert_eq!(
            col_a.to_data(),
            col_b.to_data(),
            "column `{}` must be byte-identical remote vs local",
            field.name()
        );
    }
}

/// Concatenate `batches` into one and cast every `Utf8View`/`BinaryView`
/// column to its plain encoding, so the Flight-decoded remote batch and the
/// local read-back compare on content, not on the encoding either side
/// happened to choose.
fn normalized_single_batch(
    batches: &[arrow::record_batch::RecordBatch],
) -> arrow::record_batch::RecordBatch {
    use arrow::datatypes::{DataType, Field, Schema};
    let first = batches.first().expect("at least one batch");
    let batch = arrow::compute::concat_batches(&first.schema(), batches).unwrap();
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
        let target = match field.data_type() {
            DataType::Utf8View => Some(DataType::Utf8),
            DataType::BinaryView => Some(DataType::Binary),
            _ => None,
        };
        match target {
            Some(ty) => {
                columns.push(arrow::compute::cast(col, &ty).unwrap());
                fields.push(std::sync::Arc::new(Field::new(
                    field.name(),
                    ty,
                    field.is_nullable(),
                )));
            }
            None => {
                columns.push(std::sync::Arc::clone(col));
                fields.push(std::sync::Arc::clone(field));
            }
        }
    }
    arrow::record_batch::RecordBatch::try_new(std::sync::Arc::new(Schema::new(fields)), columns)
        .unwrap()
}

// ---------------------------------------------------------------------------
// #500 U4a: the data-parallel rank count across the two submit surfaces.
// ---------------------------------------------------------------------------

/// The one submission both surfaces make, identical in every field — so a
/// divergence in the persisted spec can only be attributed to the surface it
/// came through.
fn two_rank_request(model: &str) -> jammi_wire::request::FineTuneRequest {
    jammi_wire::request::FineTuneRequest {
        source: "patents".to_string(),
        base_model: model.to_string(),
        columns: vec!["abstract".to_string()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        config: Some(FineTuneConfig {
            epochs: 1,
            lora_rank: 4,
            ..FineTuneConfig::default()
        }),
        world_size: std::num::NonZeroU32::new(2),
        cache: jammi_db::store::CachePolicy::Bypass,
    }
}

/// PARITY (K4). The same multi-rank submission through the remote client and
/// through the embedded session persists the BYTE-IDENTICAL `jobs.spec` — the
/// stored `TrainingSpec` JSON, not merely two `Ok` responses.
///
/// The count is exactly the field that diverges invisibly: it rides on the
/// request rather than inside the spec `oneof`, its default is silent (`0` on
/// the wire = unset), and the embedded path never touches the wire at all — so
/// a remote path that dropped, defaulted, or clamped it would still hand back
/// a job id. Comparing the persisted spec catches that; comparing the
/// responses does not. A multi-rank count is also the divergence-prone input
/// here: the single-rank case is the one both paths get right by doing
/// nothing.
///
/// The count is asserted PRESENT in the JSON as well as equal across the two
/// surfaces: two surfaces that both dropped it would otherwise agree
/// vacuously.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_and_embedded_submits_persist_the_identical_multi_rank_spec() {
    let server = start_engine_server_with_devices(2).await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;
    let model = tiny_bert_model_id();

    let remote_job = remote
        .submit_fine_tune(two_rank_request(&model))
        .await
        .expect("remote submit of a two-rank job returns a handle");
    let local_job = local
        .submit_fine_tune(two_rank_request(&model))
        .await
        .expect("embedded submit of a two-rank job returns a handle");

    let catalog = server.engine.catalog();
    let remote_spec = catalog
        .get_job(&remote_job.0)
        .await
        .expect("remote get_job")
        .spec;
    let local_spec = catalog
        .get_job(&local_job.0)
        .await
        .expect("local get_job")
        .spec;

    assert!(
        remote_spec.contains("\"world_size\":2"),
        "the persisted spec must carry the count the caller chose, so this \
         oracle cannot pass with both surfaces dropping it: {remote_spec}"
    );
    assert_eq!(
        remote_spec, local_spec,
        "a job submitted over the wire and the same job submitted in-process \
         must persist byte-identical specs"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// PARITY, the unset boundary. A submission that chooses no count persists the
/// same spec through both surfaces too, and that spec carries the engine's
/// single rank — a case the multi-rank test above does not cover, because the
/// wire's `0` is a DIFFERENT value from the engine's `1`: the remote path has a
/// resolution step the embedded path does not, and an unresolved `0` reaching
/// the spec is a zero-rank job on disk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unset_count_persists_the_identical_single_rank_spec_on_both_paths() {
    let server = start_engine_server_with_devices(2).await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;
    let model = tiny_bert_model_id();

    let request = || jammi_wire::request::FineTuneRequest {
        world_size: None,
        ..two_rank_request(&model)
    };

    let remote_job = remote
        .submit_fine_tune(request())
        .await
        .expect("remote submit with no chosen count returns a handle");
    let local_job = local
        .submit_fine_tune(request())
        .await
        .expect("embedded submit with no chosen count returns a handle");

    let catalog = server.engine.catalog();
    let remote_spec = catalog
        .get_job(&remote_job.0)
        .await
        .expect("remote get_job")
        .spec;
    let local_spec = catalog
        .get_job(&local_job.0)
        .await
        .expect("local get_job")
        .spec;

    assert!(
        remote_spec.contains("\"world_size\":1"),
        "an unset count resolves to the engine's single rank BEFORE it is \
         persisted -- the wire's 0 must never reach the spec: {remote_spec}"
    );
    assert_eq!(
        remote_spec, local_spec,
        "an unset count must persist byte-identically through both surfaces"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

// ---------------------------------------------------------------------------
// U3 wire-server follow-up: `SubmitJobRequest.cache` across the two submit
// surfaces.
// ---------------------------------------------------------------------------

/// PARITY. `CachePolicy::Use` submitted over the wire persists the
/// BYTE-IDENTICAL `jobs.spec` a `CachePolicy::Use` submitted through the
/// embedded session persists — the stored `TrainingCommon.cache`, not merely
/// two `Ok` responses. `cache` is excluded from the fine-tune definition hash
/// (a call-time dial, not identity), so nothing about a persisted-spec
/// comparison depends on the model-level reuse probe ever firing; this test
/// only proves the wire carries the caller's choice through to the row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_use_cache_policy_persists_the_identical_spec_on_both_paths() {
    let server = start_engine_server_with_devices(2).await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;
    let model = tiny_bert_model_id();

    let request = || jammi_wire::request::FineTuneRequest {
        cache: jammi_db::store::CachePolicy::Use,
        ..two_rank_request(&model)
    };

    let remote_job = remote
        .submit_fine_tune(request())
        .await
        .expect("remote submit with cache = use returns a handle");
    let local_job = local
        .submit_fine_tune(request())
        .await
        .expect("embedded submit with cache = use returns a handle");

    let catalog = server.engine.catalog();
    let remote_spec = catalog
        .get_job(&remote_job.0)
        .await
        .expect("remote get_job")
        .spec;
    let local_spec = catalog
        .get_job(&local_job.0)
        .await
        .expect("local get_job")
        .spec;

    assert!(
        remote_spec.contains("\"cache\":\"use\""),
        "the persisted spec must carry the policy the caller chose, so this \
         oracle cannot pass with both surfaces dropping it: {remote_spec}"
    );
    assert_eq!(
        remote_spec, local_spec,
        "a job submitted over the wire and the same job submitted in-process \
         must persist byte-identical specs"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// PARITY, the unset boundary. A submission that chooses no cache policy
/// persists the same spec through both surfaces too, and that spec carries
/// the engine's `Bypass` default — the wire's `UNSPECIFIED` (`0`) and the
/// explicit `BYPASS` (`2`) are two different byte strings on the wire but the
/// SAME decoded engine value, so this is the case a naive "assert the field is
/// present" check would pass vacuously while still shipping the wrong policy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unset_cache_policy_persists_the_identical_bypass_spec_on_both_paths() {
    let server = start_engine_server_with_devices(2).await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;
    let model = tiny_bert_model_id();

    let request = two_rank_request(&model);
    assert_eq!(
        request.cache,
        jammi_db::store::CachePolicy::Bypass,
        "the fixture's default must be the unset case this test means to cover"
    );

    let remote_job = remote
        .submit_fine_tune(two_rank_request(&model))
        .await
        .expect("remote submit with no chosen cache policy returns a handle");
    let local_job = local
        .submit_fine_tune(two_rank_request(&model))
        .await
        .expect("embedded submit with no chosen cache policy returns a handle");

    let catalog = server.engine.catalog();
    let remote_spec = catalog
        .get_job(&remote_job.0)
        .await
        .expect("remote get_job")
        .spec;
    let local_spec = catalog
        .get_job(&local_job.0)
        .await
        .expect("local get_job")
        .spec;

    assert!(
        remote_spec.contains("\"cache\":\"bypass\""),
        "an unset cache policy must resolve to the engine's Bypass default: {remote_spec}"
    );
    assert_eq!(
        remote_spec, local_spec,
        "an unset cache policy must persist byte-identically through both surfaces"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// REFUSAL, over the REMOTE path. A count the deployment cannot serve
/// (`world_size > devices`) is refused at the submit edge from the wire, not
/// only from the embedded API: the refusal is a property of the submit edge,
/// and the submit edge has more than one entrance.
///
/// Three assertions, because a refusal can be wrong in three independent ways:
/// the remote caller reconstructs the SAME typed `JammiError` the embedded
/// caller sees (not a coarse status, not the lossy `Other` fallback), the
/// message is the engine's own, and NOTHING was enqueued — a refusal that
/// leaves a queued row behind is a job that later runs with an unservable
/// count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_count_beyond_the_devices_is_refused_from_the_wire_and_enqueues_nothing() {
    // One device: a two-rank job is unservable on this deployment.
    let server = start_engine_server_with_devices(1).await;
    let remote = remote(&server).await;
    let local = local(&server);
    add_patents(&local).await;
    let model = tiny_bert_model_id();

    let before = server
        .engine
        .catalog()
        .list_jobs()
        .await
        .expect("list_jobs")
        .len();

    let remote_err = remote
        .submit_fine_tune(two_rank_request(&model))
        .await
        .expect_err("a two-rank job on a one-device deployment must be refused");
    let local_err = local
        .submit_fine_tune(two_rank_request(&model))
        .await
        .expect_err("the embedded surface refuses the same submission");

    assert_eq!(
        std::mem::discriminant(&remote_err),
        std::mem::discriminant(&local_err),
        "the remote caller must reconstruct the SAME typed refusal the embedded \
         caller sees: {remote_err:?} vs {local_err:?}"
    );
    assert_eq!(
        remote_err.to_string(),
        local_err.to_string(),
        "the remote caller must carry the refusal message the engine produced"
    );

    let after = server
        .engine
        .catalog()
        .list_jobs()
        .await
        .expect("list_jobs")
        .len();
    assert_eq!(
        before, after,
        "a refused submission must enqueue nothing -- on either path"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
