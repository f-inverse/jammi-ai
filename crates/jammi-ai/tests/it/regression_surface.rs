//! End-to-end integration tests for the public `db.fine_tune(task="regression")`
//! surface (W5-PR4 — the consumer on-ramp).
//!
//! These drive the FULL public path a real consumer hits — `add_source` → the
//! worker's column→loader detector (`build_training_data_loader` →
//! `from_regression`) → train on `Device::Cpu` → publish → serve the
//! de-standardised prediction back through `Infer` — and assert the served
//! numbers, not direct loop construction. They are the deliverable's proof that
//! a regression head is reachable AND read back correctly through the only
//! served read path (`Infer`'s Arrow columns).
//!
//! ## What these tests prove (and why the bar is *separation*, not proximity-to-μ)
//!
//! A regression head's de-standardising affine emits `μ_y + σ_y·z`. With the
//! head zero-initialised, `z ≈ 0`, so the served value is *exactly* `μ_y` for
//! **every** input — independent of the text. A test that only checks the served
//! value lands near `μ_y` therefore proves the *scaler centres at μ*, not that
//! the model learned any `input → target` mapping: an untrained head passes it.
//!
//! So the fixture splits the rows into two TOPICALLY DISTINCT text groups mapped
//! to WELL-SEPARATED year bands — group **A** (physics vocabulary: `quantum`,
//! `energy`, `surface`, `structure`, `system`) → ~2003-2006, group **B**
//! (biology/ML vocabulary: `protein`, `gene`, `cell`, `neural`, `network`,
//! `learning`) → ~2042-2045. Every word is in the tiny BERT fixture's
//! 256-token vocabulary, so the frozen encoder embeds the two groups separably
//! (distinct topical tokens, not one template with a swapped number), and the
//! LoRA projection + distribution head can learn embedding → year separation.
//!
//! The tests then serve HELD-OUT items from each group (`regression_years_holdout_{a,b}.csv`,
//! text instances absent from training) and assert the served predictions
//! **separate the groups** by a margin an untrained μ-regurgitating head cannot
//! produce — it gives `μ_y` for both, i.e. ~0 separation. The
//! [`untrained_regression_head_collapses_to_mu_no_separation`] guard proves
//! exactly that collapse against the same fixture, locking these tests against a
//! future regression that drops the trained head on serve (the original Break 5).
//!
//! ## Objective choice
//!
//! These surface tests use `Crps` (a Gaussian-form head) and `Pinball` (the
//! quantile head) — the two robust objectives — to exercise the served-form and
//! quantile-vs-Gaussian dispatch on a realistic-variance target (σ_y ≈ 19.5 here,
//! vs the PR1 oracle's σ ≈ 2).
//!
//! Historically (pre-W5-PR5) the Gaussian NLL objectives (`GaussianNll`,
//! `BetaNll`) DIVERGED on this scale: the loss scored `(y-μ)²/σ²` in raw outcome
//! units, so a tens-of-years residual blew the loss past the trainer's divergence
//! guard (`> 100`) before the head's raw σ could adapt. W5-PR5 fixed that by
//! scoring the loss in standardized (z) space — `db.fine_tune(task=regression)`
//! now converges for ALL FOUR objectives on any target scale (see the
//! `standardization_contract` high-variance oracle for the per-objective proof).
//! These tests keep `Crps`/`Pinball` because the surface they pin (separation +
//! served-form dispatch) is objective-independent and they were green pre- and
//! post-fix, locking the public read path against either regression.

use std::sync::Arc;

use arrow::array::{Array, Float32Array, StringArray};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod, LrSchedule, RegressionLoss};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

/// Group-A (physics) held-out texts, absent from training, targeting ~2003-2006.
const GROUP_A: &[&str] = &[
    "quantum surface energy system structure",
    "structure energy system surface quantum",
    "system quantum structure energy surface",
    "energy surface system structure quantum",
];
/// Group-B (biology/ML) held-out texts, absent from training, targeting ~2042-2045.
const GROUP_B: &[&str] = &[
    "protein cell network gene learning neural",
    "neural gene learning protein network cell",
    "cell learning neural network protein gene",
    "network protein gene neural cell learning",
];

/// Minimum held-out separation `mean(B) - mean(A)` a TRAINED Gaussian-form head
/// must show. Measured ≈ 6.9 yr with the budget below; an untrained head gives
/// ≈ 0 (both groups collapse to μ_y). The 3.0 bar sits well above that 0 and
/// below the measured value, so it fails for a μ-regurgitating head and passes
/// for a head that learned the group split.
const GAUSSIAN_MIN_SEPARATION: f32 = 3.0;

/// Minimum held-out separation on the median quantile column for a TRAINED
/// quantile head. Measured ≈ 12.9 yr; an untrained head gives ≈ 0.
const QUANTILE_MIN_SEPARATION: f32 = 5.0;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn add_source(session: &Arc<InferenceSession>, name: &str, file: &str) {
    session
        .add_source(
            name,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url(file)),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// Session with the two-group training source and the two held-out group sources
/// registered.
async fn session_with_regression_data() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    add_source(&session, "years", "regression_years.csv").await;
    add_source(&session, "holdout_a", "regression_years_holdout_a.csv").await;
    add_source(&session, "holdout_b", "regression_years_holdout_b.csv").await;
    (session, dir)
}

fn regression_columns() -> Vec<String> {
    vec!["text".to_string(), "target".to_string()]
}

fn group_strings(group: &[&str]) -> Vec<String> {
    group.iter().map(|s| s.to_string()).collect()
}

fn mean(v: &[f32]) -> f32 {
    assert!(!v.is_empty(), "no served rows to average");
    v.iter().sum::<f32>() / v.len() as f32
}

/// Read the named served column over every OK row across all batches.
fn served_column(batches: &[arrow::record_batch::RecordBatch], name: &str) -> Vec<f32> {
    let mut out = Vec::new();
    for batch in batches {
        let status = batch
            .column_by_name("_status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let col = batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("served column {name} present"))
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            if status.value(i) == "ok" {
                out.push(col.value(i));
            }
        }
    }
    out
}

/// END-TO-END GAUSSIAN, LEARNING PROVEN BY GROUP SEPARATION: two topically
/// distinct `(text, int64-target)` groups, fine-tuned with the `Crps`
/// Gaussian-form objective through the public worker path, then served back via
/// `Infer` on HELD-OUT items of each group. The served means must SEPARATE the
/// groups (B above A) by a margin an untrained μ-regurgitating head cannot reach
/// — proving the LoRA + distribution head learned an `input → year` mapping, not
/// merely that the scaler centres at μ_y. This is the consumer on-ramp proven end
/// to end: detector → `from_regression` → train on CPU → publish → de-standardised
/// `Infer` read that TRACKS the input.
#[tokio::test(flavor = "multi_thread")]
async fn gaussian_regression_separates_groups_through_public_path() {
    let (session, _dir) = session_with_regression_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "years",
            &tiny_bert_model(),
            &regression_columns(),
            FineTuneMethod::Lora,
            ModelTask::Regression,
            Some(FineTuneConfig {
                epochs: 200,
                batch_size: 8,
                lora_rank: 4,
                learning_rate: 3e-2,
                warmup_steps: 8,
                lr_schedule: LrSchedule::Constant,
                // CRPS — a Gaussian-form objective (serves mean/std). Post-W5-PR5
                // (z-space loss) GaussianNll/BetaNll also converge on this σ≈19.5
                // target; CRPS is kept here as the robust Gaussian-form surface.
                regression_loss: Some(RegressionLoss::Crps),
                seed: 7,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    let model_source = ModelSource::parse(job.model_id());

    // The served Gaussian columns must be present (break #4: a Gaussian-form head
    // serves predicted_mean/predicted_std, not a mis-decode).
    let results_a = session
        .infer(
            "holdout_a",
            &model_source,
            ModelTask::Regression,
            &["text".to_string()],
            "target",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let schema_a = results_a[0].schema();
    let cols: Vec<&str> = schema_a
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert!(
        cols.contains(&"predicted_mean") && cols.contains(&"predicted_std"),
        "Gaussian-form regression serves predicted_mean/predicted_std, got {cols:?}"
    );

    let results_b = session
        .infer(
            "holdout_b",
            &model_source,
            ModelTask::Regression,
            &["text".to_string()],
            "target",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    let mean_a = mean(&served_column(&results_a, "predicted_mean"));
    let mean_b = mean(&served_column(&results_b, "predicted_mean"));
    let separation = mean_b - mean_a;
    assert!(
        separation >= GAUSSIAN_MIN_SEPARATION,
        "served means must SEPARATE the groups (learning, not μ-regurgitation): \
         group A (physics, ~2004) served {mean_a:.2}, group B (biology, ~2043) served \
         {mean_b:.2}, separation {separation:.2} < required {GAUSSIAN_MIN_SEPARATION}. \
         An untrained head gives μ_y for both → ~0 separation."
    );

    // Every served std must be a positive, finite floored value — never NaN.
    for std in served_column(&results_a, "predicted_std")
        .into_iter()
        .chain(served_column(&results_b, "predicted_std"))
    {
        assert!(
            std.is_finite() && std > 0.0,
            "served Gaussian std {std} must be positive and finite"
        );
    }
}

/// BREAK #4 NON-VACUITY — QUANTILE SERVED CORRECTLY *AND* LEARNING: a
/// Pinball/Quantile head fine-tuned through the public path is (a) read back via
/// `Infer` as its quantile columns (one per level, non-crossing), NOT silently
/// mis-served as a Gaussian `(mean, std)` — this FAILS on the pre-fix
/// hardcoded-Gaussian behaviour — and (b) SEPARATES the two groups: served on
/// held-out items, group A's quantiles sit below group B's by a margin an
/// untrained head (μ_y for both → ~0) cannot produce.
#[tokio::test(flavor = "multi_thread")]
async fn quantile_regression_serves_and_separates_groups() {
    let (session, _dir) = session_with_regression_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let levels = vec![0.1, 0.5, 0.9];
    let job = session
        .fine_tune(
            "years",
            &tiny_bert_model(),
            &regression_columns(),
            FineTuneMethod::Lora,
            ModelTask::Regression,
            Some(FineTuneConfig {
                epochs: 120,
                batch_size: 8,
                lora_rank: 4,
                learning_rate: 1e-1,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                regression_loss: Some(RegressionLoss::Pinball),
                quantile_levels: levels.clone(),
                seed: 7,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    let model_source = ModelSource::parse(job.model_id());
    let results_a = session
        .infer(
            "holdout_a",
            &model_source,
            ModelTask::Regression,
            &["text".to_string()],
            "target",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let results_b = session
        .infer(
            "holdout_b",
            &model_source,
            ModelTask::Regression,
            &["text".to_string()],
            "target",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    let cols: Vec<String> = results_a[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();

    // The served schema must be the quantile columns, NOT the Gaussian mis-serve.
    assert!(
        !cols
            .iter()
            .any(|c| c == "predicted_mean" || c == "predicted_std"),
        "a quantile head must NOT be served as Gaussian mean/std (break #4), got {cols:?}"
    );
    let quantile_cols: Vec<&String> = cols.iter().filter(|c| c.starts_with("quantile_")).collect();
    assert_eq!(
        quantile_cols.len(),
        levels.len(),
        "served schema must carry one column per quantile level, got {cols:?}"
    );

    // Pull the first OK row's quantile points and assert non-crossing (ascending).
    let first_ok_row = |batches: &[arrow::record_batch::RecordBatch]| -> Vec<f32> {
        for batch in batches {
            let status = batch
                .column_by_name("_status")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                if status.value(i) == "ok" {
                    return levels
                        .iter()
                        .map(|q| {
                            let name = format!("quantile_{q}");
                            batch
                                .column_by_name(&name)
                                .unwrap_or_else(|| panic!("missing {name}"))
                                .as_any()
                                .downcast_ref::<Float32Array>()
                                .unwrap()
                                .value(i)
                        })
                        .collect();
                }
            }
        }
        panic!("a served ok quantile row");
    };
    let row_a = first_ok_row(&results_a);
    for w in row_a.windows(2) {
        assert!(
            w[1] >= w[0],
            "served quantile columns must be non-crossing (ascending), got {row_a:?}"
        );
    }

    // Group separation: the median served quantile must split the groups, B above
    // A, by a margin an untrained μ-regurgitating head cannot reach.
    let med_a = mean(&served_column(&results_a, "quantile_0.5"));
    let med_b = mean(&served_column(&results_b, "quantile_0.5"));
    let separation = med_b - med_a;
    assert!(
        separation >= QUANTILE_MIN_SEPARATION,
        "served median quantile must SEPARATE the groups (learning, not μ-regurgitation): \
         group A (physics) served {med_a:.2}, group B (biology) served {med_b:.2}, \
         separation {separation:.2} < required {QUANTILE_MIN_SEPARATION}. \
         An untrained head gives μ_y for both → ~0 separation."
    );

    // The separation holds across EVERY quantile column, not just the median —
    // group A's whole predictive band sits below group B's.
    for q in &levels {
        let name = format!("quantile_{q}");
        let a = mean(&served_column(&results_a, &name));
        let b = mean(&served_column(&results_b, &name));
        assert!(
            b > a,
            "served {name} must place group B above group A (A={a:.2}, B={b:.2})"
        );
    }
}

/// PERMANENT NON-VACUITY GUARD (locks the separation bar against a future
/// head-serving regression): train the SAME two-group model, then serve each
/// group through a copy of the head whose trained `distribution.lora_b` is
/// zeroed (the in-process equivalent of an auditor destructively zeroing the
/// LoRA delta on disk — the untrained-head state). The de-standardising affine
/// then emits `μ_y + σ_y·0 = μ_y` for EVERY input, so the served value is
/// identical across both groups → ~0 separation, and the
/// `GAUSSIAN_MIN_SEPARATION` bar the trained test asserts FAILS.
///
/// This is the destructive proof that the trained tests measure LEARNING: if the
/// served head ever silently stops applying its learned distribution layer (the
/// original Break 5: serving the pooled embedding, or a head reset to base), the
/// separation collapses to what this guard pins, and the trained tests above go
/// red.
#[tokio::test(flavor = "multi_thread")]
async fn untrained_regression_head_collapses_to_mu_no_separation() {
    let (session, _dir) = session_with_regression_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "years",
            &tiny_bert_model(),
            &regression_columns(),
            FineTuneMethod::Lora,
            ModelTask::Regression,
            Some(FineTuneConfig {
                epochs: 200,
                batch_size: 8,
                lora_rank: 4,
                learning_rate: 3e-2,
                warmup_steps: 8,
                lr_schedule: LrSchedule::Constant,
                regression_loss: Some(RegressionLoss::Crps),
                seed: 7,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    let model_source = ModelSource::parse(job.model_id());
    let a = group_strings(GROUP_A);
    let b = group_strings(GROUP_B);

    // Sanity: with the TRAINED head, the groups DO separate past the trained
    // test's bar — confirming the model and budget learn the split, so the
    // collapse below is attributable to zeroing the head, not to a weak model.
    let trained_a = session
        .served_regression_col0_for_test(&model_source, &a, false)
        .await
        .unwrap();
    let trained_b = session
        .served_regression_col0_for_test(&model_source, &b, false)
        .await
        .unwrap();
    let trained_sep = mean(&trained_b) - mean(&trained_a);
    assert!(
        trained_sep >= GAUSSIAN_MIN_SEPARATION,
        "control: the TRAINED head must separate the groups (A={:.2}, B={:.2}, sep={trained_sep:.2})",
        mean(&trained_a),
        mean(&trained_b)
    );

    // Destructive: zero the trained distribution head. The served value must now
    // collapse to a single μ_y for BOTH groups → the separation bar must FAIL.
    let zeroed_a = session
        .served_regression_col0_for_test(&model_source, &a, true)
        .await
        .unwrap();
    let zeroed_b = session
        .served_regression_col0_for_test(&model_source, &b, true)
        .await
        .unwrap();
    let zeroed_sep = (mean(&zeroed_b) - mean(&zeroed_a)).abs();
    assert!(
        zeroed_sep < GAUSSIAN_MIN_SEPARATION,
        "a zeroed (untrained) head must NOT separate the groups: it emits μ_y for every \
         input, yet served A={:.2} B={:.2} (sep {zeroed_sep:.2}) cleared the trained bar \
         {GAUSSIAN_MIN_SEPARATION} — the trained tests would then be vacuous",
        mean(&zeroed_a),
        mean(&zeroed_b)
    );

    // The collapse is total: every served value is the SAME constant μ_y,
    // regardless of group — the literal μ-regurgitation the trained tests must
    // out-separate.
    let mu = zeroed_a[0];
    for v in zeroed_a.iter().chain(zeroed_b.iter()) {
        assert!(
            (v - mu).abs() < 1e-2,
            "a zeroed head must emit one constant μ_y for every input, got {v} vs {mu}"
        );
    }
}
