//! End-to-end conformal prediction: the serving primitive verified against the
//! coverage guarantee on a synthetic exchangeable held-out set, and its
//! `conformal` evidence channel merged through the real catalog substrate.
//!
//! The marginal-coverage check is the contract: on exchangeable data the
//! realised coverage of the emitted sets is `>= 1 - alpha`, and the
//! finite-sample `(n+1)` quantile — not the naive one — is what makes it hold.
//! A three-way split is enforced by construction: the calibration draws are
//! disjoint from the test draws, and a separate seed offset stands in for the
//! (implicit) training split that produced the synthetic predictor.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use jammi_ai::evidence::conformal::{channel_spec, contribution, ConformalOutput};
use jammi_ai::evidence::merge_channels;
use jammi_ai::predict::{ClassScore, ConformalModel, IntervalScore};
use jammi_db::catalog::Catalog;
use jammi_db::ChannelId;
use jammi_numerics::calibration::coverage;
use tempfile::tempdir;

/// A noisy-softmax synthetic classifier whose realised labels are *sampled*
/// from the predicted distribution, making `(probabilities, label)` pairs
/// exchangeable and calibrated by construction.
fn synthetic_classification(
    rng: &mut StdRng,
    n: usize,
    n_classes: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut probs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let logits: Vec<f64> = (0..n_classes).map(|_| rng.gen_range(-2.5..2.5)).collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        let row: Vec<f64> = exp.iter().map(|e| e / sum).collect();

        let u: f64 = rng.gen_range(0.0..1.0);
        let mut acc = 0.0;
        let mut label = n_classes - 1;
        for (c, p) in row.iter().enumerate() {
            acc += p;
            if u <= acc {
                label = c;
                break;
            }
        }
        probs.push(row);
        labels.push(label);
    }
    (probs, labels)
}

#[test]
fn classification_coverage_meets_nominal_on_exchangeable_split() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let n_classes = 6;

    for &alpha in &[0.05, 0.1, 0.2] {
        // Three-way split: calibration and test are disjoint draws.
        let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 3000, n_classes);
        let (test_probs, test_labels) = synthetic_classification(&mut rng, 5000, n_classes);

        let model = ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, alpha)
            .unwrap();

        let hits: Vec<bool> = test_probs
            .iter()
            .zip(test_labels.iter())
            .map(|(row, &y)| model.predict_set(row, None).unwrap().contains(&y))
            .collect();
        let cov = coverage(&hits).unwrap();

        assert!(
            cov >= 1.0 - alpha - 0.03,
            "alpha={alpha}: realised coverage {cov} below the 1 - alpha guarantee"
        );
    }
}

#[test]
fn absolute_residual_interval_coverage_meets_nominal() {
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let dist = Normal::new(0.0, 1.0).unwrap();
    let alpha = 0.1;

    let cal_pred = vec![0.0; 3000];
    let cal_obs: Vec<f64> = (0..3000).map(|_| dist.sample(&mut rng)).collect();
    let model = ConformalModel::regression(
        &cal_pred,
        &[],
        &[],
        &cal_obs,
        IntervalScore::AbsoluteResidual,
        alpha,
    )
    .unwrap();

    let test_obs: Vec<f64> = (0..5000).map(|_| dist.sample(&mut rng)).collect();
    let (lower, upper): (Vec<f64>, Vec<f64>) = test_obs
        .iter()
        .map(|_| model.predict_interval(0.0, 0.0, 0.0, None).unwrap())
        .unzip();
    let cov = jammi_numerics::calibration::interval_coverage(&lower, &upper, &test_obs).unwrap();
    assert!(
        cov >= 1.0 - alpha - 0.02,
        "absolute-residual coverage {cov} below the 1 - alpha guarantee"
    );
}

/// Open a fresh catalog over a temp dir.
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
async fn conformal_channel_registers_and_merges_through_the_catalog() {
    let (_dir, catalog) = open_catalog().await;

    // The conformal channel is not a built-in; registering its declared spec
    // is the additive substrate step (no new provenance machinery).
    catalog
        .channels()
        .register(&channel_spec().unwrap())
        .await
        .unwrap();
    let conformal = ChannelId::new("conformal").unwrap();

    let batch = source_batch(3);
    let contrib = contribution(&[
        ConformalOutput::Set {
            classes: vec![0, 2],
            alpha: 0.1,
        },
        ConformalOutput::Interval {
            lower: -1.0,
            upper: 1.0,
            alpha: 0.1,
        },
        ConformalOutput::Set {
            classes: vec![1],
            alpha: 0.1,
        },
    ])
    .unwrap();

    let merged = merge_channels(
        &catalog,
        &[batch],
        &[conformal.clone()],
        &[],
        &[conformal.clone()],
        &[vec![contrib]],
    )
    .await
    .unwrap();

    assert_eq!(merged.len(), 1);
    let schema = merged[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec![
            "_row_id",
            "_source_id",
            "retrieved_by",
            "annotated_by",
            "prediction_set",
            "lower",
            "upper",
            "alpha",
        ]
    );

    let m = &merged[0];
    let set = m
        .column(schema.index_of("prediction_set").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(set.value(0), "[0,2]");
    assert!(set.is_null(1)); // the interval row writes no set
    assert_eq!(set.value(2), "[1]");

    let lower = m
        .column(schema.index_of("lower").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(lower.is_null(0));
    assert_eq!(lower.value(1), -1.0);

    let alpha = m
        .column(schema.index_of("alpha").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(alpha.value(0), 0.1);
    assert_eq!(alpha.null_count(), 0);
}

#[test]
fn sets_are_deterministic_across_independent_calibrations() {
    // The audit property at the integration boundary: two models built from the
    // same calibration set produce identical sets on the same test rows.
    let mut rng = StdRng::seed_from_u64(31337);
    let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 1500, 5);
    let (test_probs, _) = synthetic_classification(&mut rng, 300, 5);

    let a = ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, 0.1).unwrap();
    let b = ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, 0.1).unwrap();
    for row in &test_probs {
        assert_eq!(
            a.predict_set(row, None).unwrap(),
            b.predict_set(row, None).unwrap()
        );
    }
}
