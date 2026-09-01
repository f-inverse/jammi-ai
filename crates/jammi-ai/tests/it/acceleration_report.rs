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
/// `tracing::warn` for `attention_block_fused`'s domain miss — still gets its
/// OWN, independently-attributed `holds: false` on its OWN record. If the
/// worker's report computation ever regressed to reading the process-lifetime
/// `fallback_warnings_emitted()`/dedup state as ITS signal (rather than a
/// delta scoped to this job's own probe), the second job would see no new
/// evidence and could wrongly default to "no misses".
///
/// `attention_block` (not `layer_norm`) is the op under test here: campaign
/// #443's W2a/W2b (landed in parallel, merged into this branch after this
/// wave's base commit) widened `layer_norm`/`softmax`/`geglu`/`rope`'s dtype
/// admission to include F16 — an f16 job on the current tree legitimately
/// reports `Holds` on those four now, so "f16 ⇒ eager" is a stale premise for
/// them. `attention_block_fused`'s domain additionally requires the fixed
/// `head_dim` its kernel was built for (`jammi_kernels::ops::attention_block::
/// HEAD_DIM == 64`); `tiny_modernbert`'s `head_dim` (`hidden_size /
/// num_attention_heads`) is nowhere near 64, so this op declines
/// REGARDLESS of dtype — a shape-based domain miss, not a dtype-widening
/// candidate, so this control stays robust to further campaign #443 dtype
/// widenings.
///
/// (Revision history, kept because a prior version of this doc got the
/// PREDICATE ORDER wrong and it is worth remembering why: campaign #443 W2d
/// briefly widened `attention_block_admission_predicate`'s dtype check to
/// `F32 | BF16 | F16` with NO device split, which made CPU+F16 wrongly
/// CLEAR the dtype gate — this doc's prior revision predicted that would
/// make the head_dim check fire next, reason
/// `head_dim_is_attention_block_fixed_head_dim`. That was itself a real bug
/// (CPU's `cpu_fwd`, `jammi-kernels::ops::attention_block`, has no `BF16`/
/// `F16` match arm at all — it only ever supported `F32`), fixed by the
/// round-2 audit's device-split correction: `BF16`/`F16` are admitted ONLY
/// when `qkv.device().is_cuda()`; CPU stays `F32`-only, matching `cpu_fwd`'s
/// real domain. On this CPU-only suite, an f16 job's `qkv` therefore declines
/// at the DTYPE check itself — reason `dtype_f32_matching_between_qkv_and_
/// mask_on_cpu` — and the head_dim check below it is never reached. `holds:
/// false` still never flips; only the verbatim reason key this test reads
/// back does, and it has now round-tripped back to the dtype reason it
/// started at, for the correct underlying cause.)
///
/// Also asserts `layer_norm` (now genuinely admitted at F16) reports
/// `holds: true` on BOTH jobs — the anti-always-degraded half of this same
/// control: an implementation that reports `holds: false` unconditionally,
/// regardless of what actually dispatched, must fail here even though it
/// would trivially "pass" the attention_block assertion above.
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
    let ab1 = &report1["ops"]["attention_block"];
    assert_eq!(
        ab1["holds"],
        serde_json::json!(false),
        "job 1 (f16, CPU): attention_block_fused must decline (tiny_modernbert's head_dim is \
         nowhere near the kernel's fixed 64), got: {report1}"
    );
    assert_eq!(
        ab1["reason"],
        serde_json::json!("dtype_f32_matching_between_qkv_and_mask_on_cpu"),
        "job 1's miss reason must be the verbatim predicate key jammi-encoders' own \
         `admit()` call site records — on CPU, attention_block's dtype check admits ONLY \
         F32 (the round-2 audit's device-split fix: BF16/F16 are CUDA-only, matching \
         cpu_fwd's real domain), so an f16 qkv declines at the dtype check itself and the \
         head_dim check below it is never reached, got: {report1}"
    );
    let ln1 = &report1["ops"]["layer_norm"];
    assert_eq!(
        ln1["holds"],
        serde_json::json!(true),
        "job 1 (f16): layer_norm_fused is genuinely admitted at F16 on the current tree \
         (campaign #443 W2a/W2b widened it) — an always-degraded report would wrongly say \
         false here, got: {report1}"
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
    let ab2 = &report2["ops"]["attention_block"];
    assert_eq!(
        ab2["holds"],
        serde_json::json!(false),
        "job 2 must independently report its OWN eager outcome, not silence from the burned \
         dedup, got: {report2}"
    );
    assert_eq!(
        ab2["reason"],
        serde_json::json!("dtype_f32_matching_between_qkv_and_mask_on_cpu"),
        "job 2's reason must still resolve to the same verbatim predicate key (CPU's dtype \
         check declines f16 before the head_dim check is ever reached), got: {report2}"
    );
    let ln2 = &report2["ops"]["layer_norm"];
    assert_eq!(
        ln2["holds"],
        serde_json::json!(true),
        "job 2 must also independently report layer_norm as genuinely admitted, got: {report2}"
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
    // NOT a strict ops-key-SET equality: `"low_rank_residual_linear"`/
    // `"dropout"` read the heavily-shared `lora_linear_fused` registry key
    // (advisory 6's documented attribution precondition — every OTHER LoRA
    // test in this `cargo test -p jammi-ai` binary races the SAME counter,
    // and `crates/jammi-ai/tests/it/encoder_adapters.rs`'s plain-BERT LoRA
    // legitimately dispatches it EAGER, `has_bias`, unlike this test's
    // bias-free ModernBERT config) — a concurrently-running test can shift
    // one transport's probe window into ambiguity (`two_arm_holds`'s
    // both-moved `None`) without the other's, which is a shared-counter
    // artifact of running the full suite in parallel, not a transport-parity
    // regression. `"layer_norm"` is asserted per-op instead: uncontended in
    // practice (every concurrent CPU test in this suite trains at F32/BF16,
    // never forcing it eager), so its determination is a reliable,
    // race-free K4 parity signal.
    let ops_a = report_a["ops"].as_object().unwrap();
    let ops_b = report_b["ops"].as_object().unwrap();
    assert!(
        !ops_a.is_empty() && !ops_b.is_empty(),
        "both transports must measure at least one op, got a={ops_a:?} b={ops_b:?}"
    );
    assert_eq!(
        report_a["ops"]["layer_norm"], report_b["ops"]["layer_norm"],
        "the same config on the same device must produce the same per-op determination \
         regardless of submission transport"
    );
}

/// Phase-4 adversarial-audit finding 3 ("FABRICATED REASON"): the
/// projection-head arm (`target_modules` empty — `backbone_dtype` never
/// takes effect there) never builds an encoder to probe at all. Before the
/// fix, `worker.rs` passed `probe_ok = false` into `flash_report` for this
/// arm, fabricating `"reason": "probe_forward_failed"` for a probe that was
/// NEVER attempted. `flash_compiled_device_reason`'s compiled/device
/// short-circuits are checked FIRST regardless (`"cuda_not_compiled"` on
/// this CPU-only scoped-gate build — legitimately true, and cheaper to state
/// than "no probe" when flash could never hold here either way), so THIS
/// test's own build can only prove the fabricated value is GONE, not that
/// the honest `"no_encoder_to_probe_projection_head_arm"` reason is reached
/// (that needs `cuda` compiled AND a real CUDA device — see
/// `flash_report_no_probe_attempted`'s doc). `ops` must be empty regardless
/// (never a fabricated per-op measurement for an arm that built no
/// dtype-typed encoder).
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn projection_head_arm_reports_no_probe_attempted_not_a_fabricated_failure() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 4,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                // target_modules left empty (the default): the projection-head
                // arm, which never builds an encoder to probe.
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let record = wait_for_any_terminal(session.catalog(), &job.job_id).await;
    let report = expect_determined_report(record.acceleration_report.as_deref());

    assert_eq!(
        report["ops"],
        serde_json::json!({}),
        "the projection-head arm builds no encoder to probe — ops must be empty, got: {report}"
    );
    assert_ne!(
        report["flash"]["reason"],
        serde_json::json!("probe_forward_failed"),
        "the projection-head arm never attempts a probe at all — it must never claim the \
         probe-was-attempted-and-failed reason, got: {report}"
    );
    if cfg!(feature = "cuda") && report["device"] == serde_json::json!("cuda") {
        assert_eq!(
            report["flash"]["reason"],
            serde_json::json!("no_encoder_to_probe_projection_head_arm"),
            "on a real CUDA device the honest no-probe-attempted reason must appear, got: \
             {report}"
        );
    }
    assert_eq!(report["flash"]["holds"], serde_json::json!(false));
}

/// This test's `ComputePrecision` as the dtype class
/// `jammi_kernels::admission::PROBED_OPS` resolves registry keys against —
/// mirrors `crates/jammi-ai/src/fine_tune/worker.rs`'s own `dtype_class_of`.
fn dtype_class(p: ComputePrecision) -> jammi_kernels::admission::DtypeClass {
    match p {
        ComputePrecision::F32 => jammi_kernels::admission::DtypeClass::F32,
        ComputePrecision::BF16 => jammi_kernels::admission::DtypeClass::Bf16,
        ComputePrecision::F16 => jammi_kernels::admission::DtypeClass::F16,
    }
}

/// The report keys `jammi_kernels::admission::PROBED_OPS` says a job at this
/// dtype class CAN produce a measurement for — the report's candidate `ops`
/// key set, a pure function of the dtype class.
fn candidate_report_keys(p: ComputePrecision) -> std::collections::BTreeSet<&'static str> {
    let dtype = dtype_class(p);
    jammi_kernels::admission::PROBED_OPS
        .iter()
        .filter(|op| op.kind == jammi_kernels::admission::ProbedOpKind::TwoArm)
        .filter(|op| op.registry_keys_for(dtype).next().is_some())
        .map(|op| op.report_key)
        .collect()
}

fn ops_keys(report: &serde_json::Value) -> std::collections::BTreeSet<String> {
    report["ops"]
        .as_object()
        .expect("a determined report's `ops` is an object")
        .keys()
        .cloned()
        .collect()
}

/// Campaign #446 finding 2 — the f16 report's missing cast epilogue, and the
/// process-history-dependent key set behind it. THREE jobs in ONE process
/// (f16, f16, f32) prove all four halves the finding names:
///
/// **(c) the table binds to the REAL registry, measured live.** `cast_scale`
/// and `cast_add` are not asserted from the table's own say-so: an f16 job's
/// report carries them `holds: true`, which can only happen if the
/// before/after delta on `cast_scale_f16_f32` / `cast_add_f16` — the exact
/// registry keys the table names for `DtypeClass::F16` — actually moved,
/// i.e. if some workspace call site really does pass those literals to
/// `admit()` (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:814`,
/// `:911`, reached from `LowRankResidualLinear::bwd` during the probe's
/// backward pass). Before the fix the shipped table named only
/// `cast_add_bf16`, so an f16 job's report could not contain either key at
/// any value.
///
/// **CPU-runnable vs CUDA-only arms.** This suite is CPU-only, so the F16
/// cast-boundary keys are the ones proven live here — `low_rank_residual_
/// linear.rs`'s own `cpu_f16_bwd_dispatches_both_new_cast_boundary_kernels`
/// pins the same two keys at the kernel level. The **`Bf16` arms
/// (`cast_scale_bf16_f32` / `cast_add_bf16`) cannot execute on CPU at all**:
/// `bwd`'s bf16 branch runs `dx_base_2d = dy_base_2d.matmul(w)` in bf16, and
/// candle's CPU matmul has no BF16 implementation (pinned by this crate's own
/// `cpu_matmul_still_cannot_do_bf16`), so a bf16 backbone is refused before
/// training on CPU (`validate_backbone_precision`). Those two keys are
/// exercised only on the CUDA lane, by
/// `crates/jammi-ai/tests/gpu_capability/capability_surface.rs`'s per-dtype
/// two-arm assertions, which derive from the SAME table.
///
/// **(d) the key set is deterministic per dtype class, regardless of process
/// history.** The two f16 jobs run back to back in one process — the second
/// after the first has already burned the process-wide warn dedup and already
/// populated every registry entry — and must produce IDENTICAL `ops` key
/// sets. A key set derived from `jammi_kernels::admission::snapshot_all()`
/// (which reflects only ops looked up at least once) could not guarantee
/// this: the first job in a fresh process would see a strictly smaller table
/// than the second.
///
/// **The K4 report-vocabulary change.** No report may carry a dtype-SUFFIXED
/// key: `"cast_add_bf16"` was a REGISTRY key spelled into the report's
/// dtype-neutral vocabulary, and its removal in favour of `"cast_add"` is a
/// consumer-visible surface change asserted here explicitly, not left
/// implicit.
///
/// **The f32 negative control is real, not vacuous.** An f32 backbone takes
/// `bwd`'s `admit()`-free "nothing to fuse" branch, so `cast_scale`/`cast_add`
/// must be ABSENT from its report — not present-and-false. An implementation
/// that emitted every table key unconditionally (the obvious way to make (d)
/// pass) fails here.
// `jammi_kernels::admission`'s dispatch registries are process-wide — same
// `#[serial]` rationale as the other tests in this file.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn probed_ops_bind_to_the_real_registry_and_key_sets_are_dtype_deterministic() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");

    let mut reports = Vec::new();
    for precision in [
        ComputePrecision::F16,
        ComputePrecision::F16,
        ComputePrecision::F32,
    ] {
        let job = session
            .fine_tune(
                "training",
                &tiny_modernbert_model(),
                &training_columns(),
                FineTuneMethod::Lora,
                ModelTask::TextEmbedding,
                Some(encoder_adapters_config(precision)),
            )
            .await
            .unwrap();
        // Tolerant of either terminal outcome: the report is written before
        // the first training step, so f16's own numeric stability on this
        // tiny fixture is irrelevant to what is asserted here.
        let record = wait_for_any_terminal(session.catalog(), &job.job_id).await;
        reports.push((
            precision,
            expect_determined_report(record.acceleration_report.as_deref()),
        ));
    }

    // Every realized key must be a candidate for that job's own dtype class —
    // no fabricated key, and (the direct finding-2 guard) no dtype-suffixed
    // registry key leaking into the report vocabulary.
    for (precision, report) in &reports {
        let candidates = candidate_report_keys(*precision);
        for key in ops_keys(report) {
            assert!(
                candidates.contains(key.as_str()),
                "report key {key:?} at {precision} is not a PROBED_OPS two-arm row with a \
                 registry key for that dtype class (candidates: {candidates:?}) — the report \
                 must never name an op it cannot attribute to a real dispatch decision. \
                 Report: {report}"
            );
            for suffix in ["_bf16", "_f16", "_f32"] {
                assert!(
                    !key.ends_with(suffix),
                    "report key {key:?} carries a dtype suffix — report keys are dtype-NEUTRAL \
                     (campaign #446 finding 2's K4 vocabulary change: `cast_add_bf16` became \
                     `cast_add`, with the registry key resolved from the backbone dtype). \
                     Report: {report}"
                );
            }
        }
    }

    let (_, f16_first) = &reports[0];
    let (_, f16_second) = &reports[1];
    let (_, f32_report) = &reports[2];

    // (d): same dtype, same process, different position in process history →
    // identical key sets.
    assert_eq!(
        ops_keys(f16_first),
        ops_keys(f16_second),
        "two f16 jobs in ONE process must produce IDENTICAL `ops` key sets — a key set that \
         depends on which job ran first is a report shape derived from process history, not \
         from the job. first={f16_first} second={f16_second}"
    );

    // (c): the F16 cast-boundary keys the pre-#446 table could not name, and
    // the live proof they bind to real `admit()` sites.
    for (label, report) in [("first", f16_first), ("second", f16_second)] {
        for key in ["cast_scale", "cast_add"] {
            assert_eq!(
                report["ops"][key]["holds"],
                serde_json::json!(true),
                "the {label} f16 job's report must carry {key:?} as `holds: true` — \
                 LowRankResidualLinear::bwd dispatches its F16 cast-boundary kernels fused on \
                 CPU (low_rank_residual_linear.rs:814,911). Before campaign #446 the probed-op \
                 table named only `cast_add_bf16`, so an f16 job's report was structurally \
                 unable to contain this key at all. Report: {report}"
            );
        }
        assert!(
            !ops_keys(report).contains("cast_add_bf16"),
            "the retired dtype-suffixed report key must be gone. Report: {report}"
        );
    }

    // The f32 negative control: absent, never present-and-false.
    for key in ["cast_scale", "cast_add"] {
        assert!(
            !ops_keys(f32_report).contains(key),
            "an f32 backbone takes LowRankResidualLinear::bwd's admit()-free \"nothing to \
             fuse\" branch — there is no registry key for {key:?} at f32, so the report must \
             OMIT it rather than claim a determination. Report: {f32_report}"
        );
    }
    // Non-vacuity: the f32 job did measure something, so the absence above is
    // a real dtype-resolved absence and not an empty report.
    assert_eq!(
        f32_report["ops"]["layer_norm"]["holds"],
        serde_json::json!(true),
        "the f32 control must still report layer_norm as genuinely admitted — otherwise the \
         cast_scale/cast_add absence above would be vacuous. Report: {f32_report}"
    );
}

/// Campaign #446 finding 3 — FABRICATED MISS REASONS. Four jobs in ONE
/// process, all reading the SAME registry key (`attention_block_fused`), at
/// two DIFFERENT failing predicates:
///
/// - `f32` on CPU clears the dtype gate and declines at the head-dim check →
///   `head_dim_is_attention_block_fixed_head_dim` (`tiny_modernbert`'s head
///   dim is nowhere near the kernel's fixed 64).
/// - `f16` on CPU declines at the DTYPE check itself → `dtype_f32_matching_
///   between_qkv_and_mask_on_cpu` (the round-2 device split: BF16/F16 are
///   CUDA-only, matching `cpu_fwd`'s real domain), so the head-dim check
///   below it is never reached.
///
/// **Why the order f32, f16, f32, f32 and not just two jobs.** The pre-fix
/// `reason_for_registry_key` read the process-lifetime
/// `fallback_warnings_emitted()` list and took the most recent entry for the
/// op. That list is populated INSIDE `warn_fallback_once_with_message`'s
/// `seen.insert((op, predicate))` guard (`crates/jammi-kernels/src/
/// admission.rs`), so it records each `(op, predicate)` pair AT MOST ONCE per
/// process. Jobs 1 and 2 therefore each push a fresh pair and read back
/// correctly even pre-fix; job 3 is where it breaks: its `head_dim` pair is
/// already in `seen`, nothing is pushed, and the most recent entry for
/// `attention_block_fused` is still job 2's `dtype_...` — a different
/// predicate, for a different dtype, persisted durably as THIS job's reason.
/// Job 3 is the deterministic RED. (Verified by running this test against the
/// unmodified `worker.rs`: job 3 reported `dtype_f32_matching_between_qkv_and_
/// mask_on_cpu` for an f32 backbone.)
///
/// Job 4 is the SAME-predicate repeat (the dedupe case the naive
/// "before/after window over `fallback_warnings_emitted()`" fix cannot serve
/// either — that window is EMPTY for job 4, because the dedupe is upstream of
/// the record). It must still carry its own predicate, never
/// `reason_unavailable`.
///
/// Every job must also keep `holds: false` — the bug was never about the
/// determination, only about the verbatim key attached to it.
// `jammi_kernels::admission`'s dispatch registries and its warn-dedup set are
// process-wide — same `#[serial]` rationale as the other tests in this file.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn each_job_reports_its_own_miss_predicate_not_the_most_recent_different_one() {
    const HEAD_DIM_MISS: &str = "head_dim_is_attention_block_fixed_head_dim";
    const CPU_DTYPE_MISS: &str = "dtype_f32_matching_between_qkv_and_mask_on_cpu";

    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");

    let plan = [
        (ComputePrecision::F32, HEAD_DIM_MISS, "job 1 (f32, first)"),
        (ComputePrecision::F16, CPU_DTYPE_MISS, "job 2 (f16)"),
        (
            ComputePrecision::F32,
            HEAD_DIM_MISS,
            "job 3 (f32, after f16 burned the dedup for a DIFFERENT predicate)",
        ),
        (
            ComputePrecision::F32,
            HEAD_DIM_MISS,
            "job 4 (f32, SAME predicate as job 3 — the dedupe case)",
        ),
    ];

    for (precision, expected_reason, label) in plan {
        let job = session
            .fine_tune(
                "training",
                &tiny_modernbert_model(),
                &training_columns(),
                FineTuneMethod::Lora,
                ModelTask::TextEmbedding,
                Some(encoder_adapters_config(precision)),
            )
            .await
            .unwrap();
        let record = wait_for_any_terminal(session.catalog(), &job.job_id).await;
        let report = expect_determined_report(record.acceleration_report.as_deref());
        let ab = &report["ops"]["attention_block"];
        assert_eq!(
            ab["holds"],
            serde_json::json!(false),
            "{label}: attention_block_fused must decline on CPU at {precision}, got: {report}"
        );
        assert_eq!(
            ab["reason"],
            serde_json::json!(expected_reason),
            "{label}: the miss reason must be the verbatim predicate key THIS job's own probe \
             window recorded, never the most recent DIFFERENT predicate some earlier job left \
             in the process-lifetime warn list (campaign #446 finding 3), and never a \
             placeholder for a miss that really did record a predicate. Report: {report}"
        );
    }
}

/// Phase-4 adversarial-audit finding 4 ("PENDING-FOREVER"), `ContextPredictor`
/// half: this job kind never routes through `run_fine_tune_blocking`'s
/// measuring probe at all (`run_spec`'s `ContextPredictor` arm calls
/// `InferenceSession::run_context_predictor_training` directly). Before the
/// fix, its `acceleration_report` stayed at the submission-time
/// `{"state":"pending"}` marker FOREVER, even past this job's terminal
/// status — which the tri-state contract's own definition of "pending"
/// (no claimant has computed a determination YET) does not describe once the
/// job is done. The self-describing `{"state":"not_applicable",
/// "reason":"context_predictor"}` marker must land instead, regardless of
/// whether the predictor training itself succeeds (this test's own tiny
/// synthetic dataset is not tuned to guarantee that — `wait_for_any_terminal`
/// accepts either outcome, since the marker is written BEFORE training even
/// starts).
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn context_predictor_job_reports_not_applicable_acceleration() {
    use arrow::array::{ArrayRef, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use jammi_ai::pipeline::context_predictor::{
        ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
    };
    use jammi_encoders::ContextArchitecture;
    use parquet::arrow::ArrowWriter;

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session.register_query_functions();

    // A minimal meta-dataset-shaped source: a few rows across two tasks. This
    // test does not need the predictor to actually LEARN anything (or even
    // complete) — only that the job reaches a terminal status with the
    // acceleration marker already written.
    let ids = ["a0", "a1", "a2", "b0", "b1", "b2"];
    let tasks = ["task_a", "task_a", "task_a", "task_b", "task_b", "task_b"];
    let ys = [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6];
    let texts = [
        "task a example zero",
        "task a example one",
        "task a example two",
        "task b example zero",
        "task b example one",
        "task b example two",
    ];
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("task", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(StringArray::from(ids.to_vec())) as ArrayRef,
            std::sync::Arc::new(StringArray::from(tasks.to_vec())),
            std::sync::Arc::new(Float64Array::from(ys.to_vec())),
            std::sync::Arc::new(StringArray::from(texts.to_vec())),
        ],
    )
    .unwrap();
    let source_path = dir.path().join("ctx_source.parquet");
    {
        let file = std::fs::File::create(&source_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, std::sync::Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    session
        .add_source(
            "ctx",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", source_path.to_str().unwrap())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .generate_text_embeddings(
            "ctx",
            &tiny_modernbert_model(),
            &["text".to_string()],
            "_row_id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let predictor_spec = ContextPredictorTrainConfig {
        model_id: "esc075-ctx-predictor".to_string(),
        architecture: ContextArchitecture::Cnp,
        key_column: "_row_id".to_string(),
        task_column: "task".to_string(),
        value_column: "y".to_string(),
        context_k: 2,
        hidden_dim: 8,
        num_heads: 1,
        num_layers: 1,
        head: PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
        epochs: 1,
        learning_rate: 0.01,
        grad_clip: 1.0,
        test_task_fraction: 0.5,
        min_task_count: 2,
        seed: 1,
    };

    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");
    let job = session
        .train_context_predictor("ctx", &predictor_spec)
        .await
        .unwrap();
    let record = wait_for_any_terminal(session.catalog(), &job.job_id).await;
    let report_json = record.acceleration_report.as_deref().expect(
        "esc-075 control (iii): a missing report is a failure — a ContextPredictor job must \
         still carry a self-describing terminal marker, never the submission-time pending \
         marker forever",
    );
    let report: serde_json::Value = serde_json::from_str(report_json).unwrap();
    assert_eq!(
        report["state"],
        serde_json::json!("not_applicable"),
        "a ContextPredictor job never runs the fine-tune measuring probe — its record must \
         carry the self-describing not_applicable marker, got: {report}"
    );
    assert_eq!(report["reason"], serde_json::json!("context_predictor"));
}

/// Phase-4 adversarial-audit finding 4 ("PENDING-FOREVER"), pre-device-
/// resolution-failure half: a job that fails in `run_claimed_job` BEFORE
/// `run_spec`/`run_fine_tune_blocking` are ever reached (an undeserialisable
/// `training_spec`) never runs the measuring probe either. Before the fix,
/// its `acceleration_report` stayed at the submission-time
/// `{"state":"pending"}` marker forever past this job's terminal `failed`
/// status. The self-describing `{"state":"undetermined",
/// "reason":"failed_before_device_resolution"}` marker must land instead.
#[serial(esc075_acceleration_report)]
#[tokio::test(flavor = "multi_thread")]
async fn pre_device_resolution_failure_reports_undetermined_acceleration() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = EmbeddedWorker::spawn(&session).expect("default worker intervals are valid");

    // A real job first, purely to mint a valid `base_model_id` FK target —
    // mirrors `embedded_and_raw_transports_produce_the_same_report_shape`'s
    // own reuse pattern rather than re-deriving `ModelSource` resolution.
    let seed_job = session
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
    seed_job.wait().await.unwrap();
    let seed_record = session
        .catalog()
        .get_training_job(&seed_job.job_id)
        .await
        .unwrap();

    // A raw job carrying a deliberately undeserialisable `training_spec` —
    // fails in `run_claimed_job` before `run_spec`/`run_fine_tune_blocking`
    // (and therefore the device resolution + measuring probe) are ever
    // reached.
    let job_id = "esc075-pre-device-resolution-failure".to_string();
    session
        .catalog()
        .create_training_job(CreateTrainingJobParams {
            job_id: &job_id,
            base_model_id: &seed_record.base_model_id,
            training_source: "training",
            loss_type: "cosent",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "this is not valid JSON at all {{{",
        })
        .await
        .unwrap();

    let record = wait_for_any_terminal(session.catalog(), &job_id).await;
    assert_eq!(
        record.status, "failed",
        "an undeserialisable training_spec must land the job failed, got {}",
        record.status
    );
    let report_json = record.acceleration_report.as_deref().expect(
        "esc-075 control (iii): a missing report is a failure — a pre-device-resolution \
         failure must still carry a self-describing terminal marker, never the submission-time \
         pending marker forever",
    );
    let report: serde_json::Value = serde_json::from_str(report_json).unwrap();
    assert_eq!(
        report["state"],
        serde_json::json!("undetermined"),
        "a job that fails before the device is ever resolved never runs the measuring probe — \
         its record must carry the self-describing undetermined marker, got: {report}"
    );
    assert_eq!(
        report["reason"],
        serde_json::json!("failed_before_device_resolution")
    );
}
