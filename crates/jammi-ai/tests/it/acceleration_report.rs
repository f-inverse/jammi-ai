//! esc-075 (campaign #443): the claim-time, per-job acceleration report.
//!
//! `.jammi/escapes.jsonl`'s `esc-075-f16-silent-eager-no-per-job-signal` names
//! three negative controls this file implements as tests, plus the
//! pending→determined transition and the K4 "one record, two transports"
//! parity the plan's Part 4 v3 promises:
//!
//! - **(i)** the SECOND f16 job submitted to the same process — after a prior
//!   f16 job already burned the process-wide `(op, predicate)` warn dedup —
//!   still gets ITS OWN determined report on ITS record
//!   (`second_f16_job_in_process_still_reports_its_own_eager_ops`).
//! - **(ii)** a positive-control job (f32, since this suite runs CPU-only)
//!   reports its ops as `Holds` — an always-degraded implementation cannot
//!   pass this (`f32_positive_control_reports_ops_holds`).
//! - **(iii)** a missing/empty/pending report is asserted as FAILURE, never
//!   read as "no misses" (`expect_determined_report_fails_closed_on_absence`,
//!   `expect_determined_report_fails_closed_on_pending_marker`) — the same
//!   `expect_determined_report` helper every other test in this file relies
//!   on to read a report, so a regression that silently tolerates absence
//!   would fail everywhere, not just here.
//! - The pending→determined transition itself is asserted directly
//!   (`submission_writes_pending_then_claim_overwrites_with_determined`).
//! - K4: an embedded-session-submitted job (`session.fine_tune`) and a raw
//!   catalog-submitted job (mimicking a non-embedded transport's write) carry
//!   the SAME determined-report shape — one record, two transports
//!   (`embedded_and_raw_transports_produce_the_same_report_shape`).

use std::sync::Arc;
use std::time::Duration;

use serial_test::serial;
use tempfile::TempDir;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::EmbeddedWorker;
use jammi_ai::fine_tune::{ComputePrecision, FineTuneConfig, FineTuneMethod, LrSchedule};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

use crate::common;

/// ModernBERT, not plain BERT: ModernBERT's LayerNorm/attention are bias-free
/// and gate their fused arm on TRAINING mode
/// (`crates/jammi-encoders/src/layer_norm.rs`'s `forward` doc: "Eval NEVER
/// reaches the fused arm"; `modernbert.rs`'s `self.training` branches on
/// rope/softmax/attention). Plain BERT's biased LayerNorm never reaches the
/// admission-instrumented path in EITHER mode, so it cannot exercise
/// esc-075's report at all — this suite needs an architecture the fused path
/// is actually reachable on.
fn tiny_modernbert_model() -> String {
    "local:".to_string() + common::fixture("tiny_modernbert").to_str().unwrap()
}

fn training_columns() -> Vec<String> {
    vec![
        "text_a".to_string(),
        "text_b".to_string(),
        "score".to_string(),
    ]
}

async fn session_with_training_data() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    (session, dir)
}

/// A fast encoder-adapters config (`backbone_dtype` only takes effect on this
/// arm) at the given precision — the same shape esc-075's own reproduction
/// used (an f16 fine-tune).
fn encoder_adapters_config(backbone_dtype: ComputePrecision) -> FineTuneConfig {
    FineTuneConfig {
        epochs: 1,
        batch_size: 4,
        lora_rank: 4,
        warmup_steps: 0,
        lr_schedule: LrSchedule::Constant,
        target_modules: vec!["Wqkv".to_string(), "Wo".to_string()],
        backbone_dtype,
        ..Default::default()
    }
}

/// Poll a raw (non-`TrainingJob`-handle) job to a terminal state — the same
/// polling `jammi_ai::fine_tune::training_job::TrainingJob::wait` does, for a
/// job this test submitted directly through the catalog rather than through
/// `session.fine_tune`. Expects success; panics on a `failed` terminal status.
async fn wait_for_terminal(
    catalog: &jammi_db::catalog::Catalog,
    job_id: &str,
) -> jammi_db::catalog::training_repo::TrainingJobRecord {
    let record = wait_for_any_terminal(catalog, job_id).await;
    if record.status != "completed" {
        panic!(
            "job {job_id} failed unexpectedly: {:?}",
            record.error_message
        );
    }
    record
}

/// Poll a job to EITHER terminal status (`completed` or `failed`) — used by
/// the f16 tests below, which care only that the acceleration report was
/// written (necessarily BEFORE the first training step, esc-075's own
/// ordering requirement) and not about whether f16's own numerics later
/// converge or diverge on this tiny fixture over a handful of eager-composed
/// batches — an orthogonal training-stability question this file does not
/// own.
async fn wait_for_any_terminal(
    catalog: &jammi_db::catalog::Catalog,
    job_id: &str,
) -> jammi_db::catalog::training_repo::TrainingJobRecord {
    loop {
        let record = catalog.get_training_job(job_id).await.unwrap();
        match record.status.as_str() {
            "completed" | "failed" => return record,
            _ => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}

/// esc-075 control (iii): reads a catalog record's `acceleration_report` as a
/// determined payload, PANICKING on `None` (a missing report) or on a
/// `"pending"` marker (a submission-time echo that was never overwritten by a
/// claimant) — absence is asserted as failure, never as "no misses". Every
/// other test in this file reads its report through this one helper, so a
/// regression that let absence read as success would fail broadly, not just
/// in the two dedicated negative-control tests below.
fn expect_determined_report(report_json: Option<&str>) -> serde_json::Value {
    let raw = report_json.expect(
        "acceleration_report must be Some — esc-075 control (iii): a missing report is a \
         failure, never a clean pass",
    );
    let value: serde_json::Value =
        serde_json::from_str(raw).expect("acceleration_report must be valid JSON");
    assert_eq!(
        value.get("state").and_then(|s| s.as_str()),
        Some("determined"),
        "acceleration_report must be in the \"determined\" state, got: {value}"
    );
    value
}

#[test]
fn expect_determined_report_fails_closed_on_absence() {
    let result = std::panic::catch_unwind(|| expect_determined_report(None));
    assert!(
        result.is_err(),
        "esc-075 control (iii): a missing acceleration_report must be asserted as failure"
    );
}

#[test]
fn expect_determined_report_fails_closed_on_pending_marker() {
    let result =
        std::panic::catch_unwind(|| expect_determined_report(Some(r#"{"state":"pending"}"#)));
    assert!(
        result.is_err(),
        "esc-075 control (iii): a submission-time pending marker must never read as a \
         determined report"
    );
}

/// esc-075 control (i): the SECOND f16 job in this process — run after a
/// first f16 job already fired (and dedup-suppressed) the per-process
/// `tracing::warn` for `layer_norm_fused`'s `f16` domain miss — still gets
/// its OWN, independently-attributed `holds: false` on its OWN record. If the
/// worker's report computation ever regressed to reading the process-lifetime
/// `fallback_warnings_emitted()`/dedup state as ITS signal (rather than a
/// delta scoped to this job's own probe), the second job would see no new
/// evidence and could wrongly default to "no misses".
// `jammi_kernels::admission`'s dispatch registries are process-wide, and this
// binary's other `it` test modules also drive real encoder forwards
// concurrently by default — `#[serial]` (the same idiom `inference.rs` uses
// for its own shared-resource tests) keeps THESE four tests from racing each
// other's before/after probe windows. It does not (and cannot, without
// serializing the whole binary) guard against an unrelated concurrently
// running test's forward pass also nudging the same counters; that residual
// is accepted here on the same precedent `inference.rs` already established.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn second_f16_job_in_process_still_reports_its_own_eager_ops() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");

    let job1 = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(encoder_adapters_config(ComputePrecision::F16)),
        )
        .await
        .unwrap();
    // Tolerant of EITHER terminal outcome: the acceleration report is written
    // BEFORE the first training step, so it must exist regardless of whether
    // f16's own (unrelated) numeric stability on this tiny fixture later
    // converges or diverges over the eager-composed batches that follow.
    let record1 = wait_for_any_terminal(session.catalog(), &job1.job_id).await;
    let report1 = expect_determined_report(record1.acceleration_report.as_deref());
    let ln1 = &report1["ops"]["layer_norm"];
    assert_eq!(
        ln1["holds"],
        serde_json::json!(false),
        "job 1 (f16, CPU): layer_norm_fused must decline (f16 is not in {{f32,bf16}}), got: {report1}"
    );
    assert_eq!(
        ln1["reason"],
        serde_json::json!("dtype_f32_or_bf16_matching_between_x_and_weight"),
        "job 1's miss reason must be the verbatim predicate key jammi-encoders' own \
         `admit()` call site records, got: {report1}"
    );

    // Job 2: same process, same op, same f16 dtype — after job 1 already
    // burned the process-wide warn dedup for exactly this (op, predicate).
    let job2 = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(encoder_adapters_config(ComputePrecision::F16)),
        )
        .await
        .unwrap();
    let record2 = wait_for_any_terminal(session.catalog(), &job2.job_id).await;
    let report2 = expect_determined_report(record2.acceleration_report.as_deref());
    let ln2 = &report2["ops"]["layer_norm"];
    assert_eq!(
        ln2["holds"],
        serde_json::json!(false),
        "job 2 must independently report its OWN eager outcome, not silence from the burned \
         dedup, got: {report2}"
    );
    assert_eq!(
        ln2["reason"],
        serde_json::json!("dtype_f32_or_bf16_matching_between_x_and_weight"),
        "job 2's reason must still resolve to the same verbatim predicate key, got: {report2}"
    );
    assert_ne!(
        job1.job_id, job2.job_id,
        "sanity: the two jobs must be distinct records"
    );
}

/// esc-075 control (ii): a positive control (f32 — this suite is CPU-only;
/// bf16 requires CUDA) reports its ops as `Holds`, so an always-degraded
/// implementation (one that reports `holds: false` unconditionally) fails
/// this test even though it would trivially pass control (i) alone.
// `jammi_kernels::admission`'s dispatch registries are process-wide, and this
// binary's other `it` test modules also drive real encoder forwards
// concurrently by default — `#[serial]` (the same idiom `inference.rs` uses
// for its own shared-resource tests) keeps THESE four tests from racing each
// other's before/after probe windows. It does not (and cannot, without
// serializing the whole binary) guard against an unrelated concurrently
// running test's forward pass also nudging the same counters; that residual
// is accepted here on the same precedent `inference.rs` already established.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn f32_positive_control_reports_ops_holds() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(encoder_adapters_config(ComputePrecision::F32)),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    let report = expect_determined_report(record.acceleration_report.as_deref());

    // `attention_block` is intentionally excluded from the strict `Holds`
    // loop below: its own domain check additionally requires the fixed head
    // dimension the fused attention kernel was built for, independent of
    // dtype — this tiny fixture's head_dim genuinely does not satisfy it, so
    // asserting `Holds` there would assert something false about THIS
    // fixture's shape, not about f32 dtype admission. It is still measured
    // (present in `ops`) — just not asserted `Holds` here.
    assert!(
        report["ops"].get("attention_block").is_some(),
        "attention_block must still be measured (present), got: {report}"
    );
    for op in ["layer_norm", "softmax", "geglu"] {
        let entry = report["ops"].get(op).unwrap_or_else(|| {
            panic!("f32 positive control must measure op {op:?}, got: {report}")
        });
        assert_eq!(
            entry["holds"],
            serde_json::json!(true),
            "f32 on CPU must be ACCELERATED (Holds) for {op}, got: {report}"
        );
        assert_eq!(entry["reason"], serde_json::json!("domain_ok"));
    }
    assert_eq!(report["dtype"], serde_json::json!("f32"));
    assert_eq!(
        report["cuda_compiled"],
        serde_json::json!(cfg!(feature = "cuda"))
    );
}

/// The three-state contract in the payload: `create_training_job` writes the
/// explicit `{"state":"pending"}` marker at submission (before any claimant
/// exists), and the claiming worker overwrites it with a `"determined"`
/// report — asserted here as a state TRANSITION, not just the end state.
// `jammi_kernels::admission`'s dispatch registries are process-wide, and this
// binary's other `it` test modules also drive real encoder forwards
// concurrently by default — `#[serial]` (the same idiom `inference.rs` uses
// for its own shared-resource tests) keeps THESE four tests from racing each
// other's before/after probe windows. It does not (and cannot, without
// serializing the whole binary) guard against an unrelated concurrently
// running test's forward pass also nudging the same counters; that residual
// is accepted here on the same precedent `inference.rs` already established.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn submission_writes_pending_then_claim_overwrites_with_determined() {
    let (session, _dir) = session_with_training_data().await;

    // No worker yet: the job sits `queued` with only the submission-time echo.
    let job = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(encoder_adapters_config(ComputePrecision::F32)),
        )
        .await
        .unwrap();
    let pre_claim = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    let pre_claim_report: serde_json::Value =
        serde_json::from_str(pre_claim.acceleration_report.as_deref().expect(
            "create_training_job must write the explicit pending marker, not leave the column \
             NULL, for a fresh submission",
        ))
        .unwrap();
    assert_eq!(
        pre_claim_report.get("state").and_then(|s| s.as_str()),
        Some("pending"),
        "before any claimant runs, the record must show the submission-time pending marker, \
         got: {pre_claim_report}"
    );

    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");
    job.wait().await.unwrap();

    let post_claim = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    let post_claim_report = expect_determined_report(post_claim.acceleration_report.as_deref());
    assert_eq!(
        post_claim_report["attempt"],
        serde_json::json!(1),
        "the first successful claim's attempt must be 1, got: {post_claim_report}"
    );
}

/// K4: "one record, two transports". Transport A is the embedded SDK path
/// (`session.fine_tune`, `session.rs:1131`); transport B is a raw catalog
/// write mirroring what a non-embedded (e.g. remote/gRPC) submission path
/// does at its substrate — `create_training_job` directly, with a
/// hand-assembled [`TrainingSpec`] — then relies on the SAME already-running
/// [`EmbeddedWorker`] loop to claim and run it (there is only one claim→run
/// code path regardless of how a job was submitted). Both jobs use the
/// identical fine-tune configuration, so their determined reports must carry
/// the same top-level shape.
// `jammi_kernels::admission`'s dispatch registries are process-wide, and this
// binary's other `it` test modules also drive real encoder forwards
// concurrently by default — `#[serial]` (the same idiom `inference.rs` uses
// for its own shared-resource tests) keeps THESE four tests from racing each
// other's before/after probe windows. It does not (and cannot, without
// serializing the whole binary) guard against an unrelated concurrently
// running test's forward pass also nudging the same counters; that residual
// is accepted here on the same precedent `inference.rs` already established.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn embedded_and_raw_transports_produce_the_same_report_shape() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");
    let config = encoder_adapters_config(ComputePrecision::F32);

    // Transport A: the embedded SDK path.
    let job_a = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(config.clone()),
        )
        .await
        .unwrap();
    job_a.wait().await.unwrap();
    let record_a = session
        .catalog()
        .get_training_job(&job_a.job_id)
        .await
        .unwrap();
    let report_a = expect_determined_report(record_a.acceleration_report.as_deref());

    // Transport B: a raw catalog write — job A already registered the base
    // model under its resolved catalog PK, so transport B reuses it rather
    // than re-deriving `ModelSource` resolution logic in this test.
    let base_model_pk = record_a.base_model_id.clone();
    let spec = TrainingSpec::FineTune {
        source: "training".to_string(),
        columns: training_columns(),
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_modernbert_model(),
            config: config.clone(),
        },
    };
    let spec_json = serde_json::to_string(&spec).unwrap();
    let hyperparams = serde_json::to_string(&config).unwrap();
    let job_b_id = "esc075-raw-transport-job".to_string();
    session
        .catalog()
        .create_training_job(CreateTrainingJobParams {
            job_id: &job_b_id,
            base_model_id: &base_model_pk,
            training_source: "training",
            loss_type: "cosent",
            hyperparams: &hyperparams,
            kind: spec.kind(),
            training_spec: &spec_json,
        })
        .await
        .unwrap();
    let record_b = wait_for_terminal(session.catalog(), &job_b_id).await;
    let report_b = expect_determined_report(record_b.acceleration_report.as_deref());

    let mut keys_a: Vec<&String> = report_a.as_object().unwrap().keys().collect();
    let mut keys_b: Vec<&String> = report_b.as_object().unwrap().keys().collect();
    keys_a.sort();
    keys_b.sort();
    assert_eq!(
        keys_a, keys_b,
        "the embedded and raw-transport reports must carry the same top-level shape"
    );
    let mut ops_a: Vec<&String> = report_a["ops"].as_object().unwrap().keys().collect();
    let mut ops_b: Vec<&String> = report_b["ops"].as_object().unwrap().keys().collect();
    ops_a.sort();
    ops_b.sort();
    assert_eq!(
        ops_a, ops_b,
        "both transports measure the same set of ops for the same job config"
    );
    assert_eq!(
        report_a["ops"]["layer_norm"], report_b["ops"]["layer_norm"],
        "the same config on the same device must produce the same per-op determination \
         regardless of submission transport"
    );
}
