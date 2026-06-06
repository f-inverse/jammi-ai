//! Distributional inference (S18): the regression decoder verified end to end.
//!
//! The serving contracts live here against the public surface:
//! - `ModelTask::Regression` is a real `ModelTask::ALL` participant that
//!   resolves to a `DistributionAdapter`.
//! - the Gaussian head serves an input-dependent (heteroscedastic) `σ`;
//! - the quantile head is monotone after the non-crossing guard — zero
//!   crossings even when the raw backend output crosses;
//! - the `uncertainty` evidence channel registers and merges through the real
//!   catalog substrate (the conformal-sibling pattern, no migration);
//! - the **R2 calibration gate**: a calibrated heteroscedastic predictive
//!   Gaussian, served through the adapter, has realised coverage ≈ nominal and a
//!   strictly lower proper score (CRPS) than a misspecified constant-variance
//!   baseline — the head is *calibrated*, not merely accurate.
//!
//! The gradient-level pathology contracts (variance collapse under naive NLL vs
//! β-NLL/CRPS, heteroscedastic σ, ordered pinball quantiles) are pinned by the
//! in-crate `fine_tune::trainer::tests` where the autodiff Vars live.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use jammi_ai::evidence::merge_channels;
use jammi_ai::evidence::uncertainty::{channel_spec, contribution, UncertaintyOutput};
use jammi_ai::inference::adapter::{
    BackendOutput, DistributionAdapter, DistributionForm, OutputAdapter,
};
use jammi_ai::model::ModelTask;
use jammi_db::catalog::Catalog;
use jammi_db::ChannelId;
use jammi_numerics::calibration::{crps_gaussian, interval_coverage};
use tempfile::tempdir;

// ─── ModelTask::Regression is a real ModelTask::ALL participant ───────────────

#[test]
fn regression_is_a_real_model_task_all_participant() {
    // It is in ALL, round-trips through the db string, and is not an embedding.
    assert!(ModelTask::ALL.contains(&ModelTask::Regression));
    assert_eq!(ModelTask::Regression.as_db_str(), "regression");
    assert_eq!(
        ModelTask::try_from_db_str("regression").unwrap(),
        ModelTask::Regression
    );
    assert!(!ModelTask::Regression.is_embedding());
}

#[test]
fn regression_serves_parametric_distribution_columns() {
    // The parametric Gaussian head — the form Regression resolves to by default
    // — declares the predictive-distribution columns.
    let names: Vec<String> = DistributionAdapter::gaussian()
        .output_schema()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["predicted_mean", "predicted_std"]);
}

// ─── Gaussian head serves an input-dependent std ──────────────────────────────

fn gaussian_backend(rows: &[(f32, f32)]) -> BackendOutput {
    // Each row is (mean, raw_std); the adapter maps raw_std → softplus + floor.
    let mut flat = Vec::with_capacity(rows.len() * 2);
    for &(m, r) in rows {
        flat.push(m);
        flat.push(r);
    }
    BackendOutput {
        float_outputs: vec![flat],
        string_outputs: vec![],
        row_status: vec![true; rows.len()],
        row_errors: vec![String::new(); rows.len()],
        shapes: vec![(rows.len(), 2)],
    }
}

#[test]
fn gaussian_head_serves_heteroscedastic_std() {
    // Two rows: an "easy" row with a small raw_std and a "hard" row with a large
    // one. The served std must vary with the input (heteroscedasticity).
    let out = gaussian_backend(&[(1.0, -3.0), (1.0, 3.0)]);
    let adapter = DistributionAdapter::gaussian();
    let cols = adapter.adapt(&out, 2).unwrap();
    let std = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();
    assert!(
        std.value(1) > std.value(0) * 5.0,
        "served std must be input-dependent: easy {}, hard {}",
        std.value(0),
        std.value(1)
    );
}

// ─── Quantile head is monotone after the non-crossing guard ───────────────────

#[test]
fn quantile_head_has_zero_crossings_even_when_raw_output_crosses() {
    let adapter = DistributionAdapter::quantile(vec![0.05, 0.5, 0.95]).unwrap();
    assert!(matches!(adapter.form(), DistributionForm::Quantile { .. }));

    // Two rows, BOTH crossing in the raw head output (q05 > q95). The guard
    // sorts each row ascending before serving.
    let flat = vec![
        2.0f32, 0.0, -1.0, // row 0 crosses
        5.0, 1.0, 3.0, // row 1 crosses
    ];
    let out = BackendOutput {
        float_outputs: vec![flat],
        string_outputs: vec![],
        row_status: vec![true; 2],
        row_errors: vec![String::new(); 2],
        shapes: vec![(2, 3)],
    };
    let cols = adapter.adapt(&out, 2).unwrap();
    let q05 = cols[0].as_any().downcast_ref::<Float32Array>().unwrap();
    let q50 = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();
    let q95 = cols[2].as_any().downcast_ref::<Float32Array>().unwrap();

    // Zero crossings: every served row is q05 ≤ q50 ≤ q95.
    for row in 0..2 {
        assert!(
            q05.value(row) <= q50.value(row) && q50.value(row) <= q95.value(row),
            "row {row} crosses after the guard: {} {} {}",
            q05.value(row),
            q50.value(row),
            q95.value(row)
        );
    }
}

// ─── The uncertainty evidence channel merges through the real catalog ─────────

async fn open_catalog() -> (tempfile::TempDir, Catalog) {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    (dir, catalog)
}

fn source_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
    ]));
    let row_id = Arc::new(StringArray::from(
        (0..n).map(|i| format!("r{i}")).collect::<Vec<_>>(),
    )) as ArrayRef;
    let src = Arc::new(StringArray::from(vec!["src"; n])) as ArrayRef;
    RecordBatch::try_new(schema, vec![row_id, src]).unwrap()
}

#[tokio::test]
async fn uncertainty_channel_registers_and_merges_through_the_catalog() {
    let (_dir, catalog) = open_catalog().await;

    // The uncertainty channel is additive — registering its declared spec is the
    // substrate step (the conformal-sibling pattern, no catalog migration).
    catalog
        .channels()
        .register(&channel_spec().unwrap())
        .await
        .unwrap();
    let uncertainty = ChannelId::new("uncertainty").unwrap();

    let batch = source_batch(2);
    // One Gaussian row (with S16 context provenance) and one quantile row.
    let contrib = contribution(&[
        UncertaintyOutput::Gaussian {
            mean: 0.7,
            std: 0.2,
            context_ref: Some(vec!["ctx-a".into(), "ctx-b".into()]),
        },
        UncertaintyOutput::Quantiles {
            levels: vec![(0.05, -1.0), (0.5, 0.0), (0.95, 1.0)],
            context_ref: None,
        },
    ])
    .unwrap();

    let merged = merge_channels(
        &catalog,
        &[batch],
        &[uncertainty.clone()],
        &[],
        &[uncertainty.clone()],
        &[vec![contrib]],
    )
    .await
    .unwrap();

    assert_eq!(merged.len(), 1);
    let schema = merged[0].schema();
    let cols: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    for expected in [
        "predicted_mean",
        "predicted_std",
        "quantiles",
        "context_ref",
    ] {
        assert!(
            cols.contains(&expected),
            "merged output is missing the uncertainty column '{expected}': {cols:?}"
        );
    }
}

// ─── R2 calibration gate: calibrated, not merely accurate ─────────────────────

/// The mandatory R2 calibration gate. A *calibrated* heteroscedastic predictive
/// Gaussian — drawn so each observation comes from the very `Normal(μ, σ)` the
/// head predicts — is served through the adapter, then scored with R2's P2
/// calibration primitives. Two contracts:
///
/// 1. **Coverage ≈ nominal.** The central 90% interval `μ ± 1.645σ` covers
///    ≈90% of observations. A head that is accurate-but-overconfident (σ too
///    small) would under-cover; this is what NLL-only "done" misses.
/// 2. **Proper score.** The calibrated head's mean CRPS is strictly lower than a
///    *constant-variance* baseline that uses the right mean but the wrong
///    (mismatched) σ everywhere — heteroscedastic uncertainty earns its keep on
///    a proper score, not just on point accuracy (both share the mean, so MSE
///    cannot separate them).
#[test]
fn r2_calibration_gate_coverage_and_proper_score() {
    let mut rng = StdRng::seed_from_u64(20180625);
    let n = 4000;

    // Heteroscedastic truth: σ alternates between an easy and a hard regime; the
    // mean is a simple function of the row. The head predicts the TRUE (μ, σ)
    // per row — a calibrated head — and we draw y from it.
    let mut means = Vec::with_capacity(n);
    let mut true_sigmas = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    let mut raw_rows: Vec<(f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let mu = (i % 7) as f64 - 3.0;
        let sigma = if i % 2 == 0 { 0.5 } else { 2.5 };
        let y = Normal::new(mu, sigma).unwrap().sample(&mut rng);
        means.push(mu);
        true_sigmas.push(sigma);
        observed.push(y);
        // The head emits (mean, raw_std) such that softplus(raw_std)+floor ≈ σ.
        // Invert: raw = ln(e^{σ−floor} − 1). σ here is well above the floor.
        let raw = ((sigma - 1e-3).exp() - 1.0).ln();
        raw_rows.push((mu as f32, raw as f32));
    }

    // Serve through the real adapter so the gate scores exactly what inference
    // would emit (mean passthrough, raw_std → softplus + floor).
    let out = gaussian_backend(&raw_rows);
    let cols = DistributionAdapter::gaussian().adapt(&out, n).unwrap();
    let served_mean = cols[0].as_any().downcast_ref::<Float32Array>().unwrap();
    let served_std = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();

    // Collect the served (mean, std) as plain vecs so the calibration
    // primitives consume them as slices, aligned with `observed`.
    let served_means: Vec<f64> = (0..n).map(|i| served_mean.value(i) as f64).collect();
    let served_stds: Vec<f64> = (0..n).map(|i| served_std.value(i) as f64).collect();

    // ── Contract 1: central-90% coverage ≈ nominal via P2's interval_coverage.
    let z90 = 1.6448536269514722; // Φ⁻¹(0.95)
    let lower: Vec<f64> = served_means
        .iter()
        .zip(&served_stds)
        .map(|(m, s)| m - z90 * s)
        .collect();
    let upper: Vec<f64> = served_means
        .iter()
        .zip(&served_stds)
        .map(|(m, s)| m + z90 * s)
        .collect();
    let cov = interval_coverage(&lower, &upper, &observed).unwrap();
    assert!(
        (cov - 0.9).abs() < 0.03,
        "calibrated head's 90% interval should cover ≈0.90, got {cov}"
    );

    // ── Contract 2: the proper score (CRPS) beats a constant-variance baseline.
    // Baseline: the right mean everywhere, but a single global σ (the RMS of the
    // true σ's) — the "ignored heteroscedasticity" predictor. Same mean, so a
    // point metric (MSE on the mean) cannot tell them apart.
    let global_sigma = (true_sigmas.iter().map(|s| s * s).sum::<f64>() / n as f64).sqrt();
    let mut crps_head = 0.0;
    let mut crps_baseline = 0.0;
    for ((m, s), &y) in served_means.iter().zip(&served_stds).zip(&observed) {
        crps_head += crps_gaussian(y, *m, *s).unwrap();
        crps_baseline += crps_gaussian(y, *m, global_sigma).unwrap();
    }
    crps_head /= n as f64;
    crps_baseline /= n as f64;
    assert!(
        crps_head < crps_baseline,
        "the calibrated heteroscedastic head must score a lower proper score \
         than a constant-variance baseline with the same mean \
         (head CRPS {crps_head}, baseline CRPS {crps_baseline})"
    );
}
