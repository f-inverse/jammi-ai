//! P5: train/serve parity between the differentiable segment-aggregate
//! (`jammi_encoders::segment_aggregate`) and the data-plane vector-aggregation
//! UDAF (`vector_mean`/`vector_sum`/`vector_max`), plus the non-text parallel
//! training loop.
//!
//! The UDAF lives in `jammi-ai` and folds f64 over arbitrary SQL group keys; the
//! tensor path lives in `jammi-encoders` and folds f32 over dense segment ids.
//! There is no shared code across that boundary, so this cross-impl test is the
//! only thing holding the parity contract. It feeds identical grouped data
//! through both and asserts row-by-row agreement within an f32-vs-f64 tolerance,
//! the singleton-group case, the empty-segment divergence, and
//! permutation-invariance.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use candle_core::{Device, Tensor, Var};
use candle_nn::{Module, VarBuilder, VarMap};
use datafusion::prelude::SessionContext;

use jammi_ai::pipeline::parallel_train::{train_loop, ParallelTrainConfig, TensorBatch};
use jammi_ai::query::register_vector_agg_udafs;
use jammi_encoders::aggregate::{segment_aggregate, SegmentReduce};

/// f32-vs-f64 tolerance for the parity assertions.
///
/// The UDAF accumulates in f64 and casts to f32 only at finalisation; the tensor
/// path accumulates entirely in f32. On the small magnitudes used here (|v| < 16,
/// ≤ 3 terms per group) the two differ only by the f32 rounding of the final
/// value — one `f32::EPSILON`-scale step per lane. `1e-5` absolute is ~100×
/// f32::EPSILON, comfortably above that rounding yet far below any algorithmic
/// disagreement (a wrong divisor or a missed term would be O(1)).
const PARITY_TOL: f32 = 1e-5;

/// Three groups with distinct sizes — `a` has 3 rows, `b` is a singleton, `c`
/// has 2 — over width-2 vectors. The group label is the parity surface: SQL
/// keys `a/b/c` map to dense ids `0/1/2`.
fn grouped_rows() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("a", vec![1.0, 2.0]),
        ("b", vec![10.0, -4.0]),
        ("a", vec![3.0, 8.0]),
        ("c", vec![-1.0, 5.0]),
        ("a", vec![5.0, -6.0]),
        ("c", vec![7.0, 0.5]),
    ]
}

/// Dense id for each SQL group key, the documented key→dense-id mapping.
fn dense_id(label: &str) -> u32 {
    match label {
        "a" => 0,
        "b" => 1,
        "c" => 2,
        other => panic!("unexpected group {other}"),
    }
}

/// Run `vector_<reduce>(v) GROUP BY g ORDER BY g` over the grouped rows and
/// return one output vector per group, in `a,b,c` order.
async fn udaf_grouped(name: &str) -> Vec<Vec<f32>> {
    let rows = grouped_rows();
    let ctx = SessionContext::new();
    register_vector_agg_udafs(&ctx);

    let labels: Vec<&str> = rows.iter().map(|(g, _)| *g).collect();
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors = Arc::new(FixedSizeListArray::new(
        item,
        2,
        Arc::new(Float32Array::from(flat)),
        None,
    )) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Utf8, false),
        Field::new_fixed_size_list("v", Field::new("item", DataType::Float32, false), 2, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(labels)) as ArrayRef, vectors],
    )
    .unwrap();
    ctx.register_batch("t", batch).unwrap();

    let out = ctx
        .sql(&format!(
            "SELECT g, {name}(v) AS r FROM t GROUP BY g ORDER BY g"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let list = out[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    (0..list.len())
        .map(|i| {
            list.value(i)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

/// Run `segment_aggregate` over the same grouped rows mapped to dense ids,
/// returning one output vector per segment in `0,1,2` order.
fn tensor_grouped(reduce: SegmentReduce) -> Vec<Vec<f32>> {
    let rows = grouped_rows();
    let dev = Device::Cpu;
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let values = Tensor::from_vec(flat, (rows.len(), 2), &dev).unwrap();
    let ids: Vec<u32> = rows.iter().map(|(g, _)| dense_id(g)).collect();
    let ids = Tensor::from_vec(ids, (rows.len(),), &dev).unwrap();
    segment_aggregate(&values, &ids, 3, reduce)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
}

fn assert_parity(udaf: &[Vec<f32>], tensor: &[Vec<f32>], name: &str) {
    assert_eq!(udaf.len(), tensor.len(), "{name}: group count");
    for (g, (u, t)) in udaf.iter().zip(tensor).enumerate() {
        for (lane, (a, b)) in u.iter().zip(t).enumerate() {
            assert!(
                (a - b).abs() <= PARITY_TOL,
                "{name} group {g} lane {lane}: udaf {a} vs tensor {b}"
            );
        }
    }
}

#[tokio::test]
async fn segment_aggregate_matches_udaf_sum() {
    assert_parity(
        &udaf_grouped("vector_sum").await,
        &tensor_grouped(SegmentReduce::Sum),
        "sum",
    );
}

#[tokio::test]
async fn segment_aggregate_matches_udaf_mean() {
    // Group `b` is a singleton — its mean equals its one row, pinning the
    // divide-by-count on a count of 1.
    assert_parity(
        &udaf_grouped("vector_mean").await,
        &tensor_grouped(SegmentReduce::Mean),
        "mean",
    );
}

#[tokio::test]
async fn segment_aggregate_matches_udaf_max() {
    assert_parity(
        &udaf_grouped("vector_max").await,
        &tensor_grouped(SegmentReduce::Max),
        "max",
    );
}

/// The documented divergence: where the UDAF yields a *null* vector for a group
/// with no rows, the tensor path yields a *zero* row. The UDAF emits one null
/// row for the empty group; the tensor path emits a zero row for the empty
/// segment. Same position, different (by-design) value.
#[tokio::test]
async fn empty_group_udaf_null_vs_tensor_zero() {
    // UDAF: a WHERE that eliminates every row folds one empty group → null.
    let ctx = SessionContext::new();
    register_vector_agg_udafs(&ctx);
    let schema = Arc::new(Schema::new(vec![Field::new_fixed_size_list(
        "v",
        Field::new("item", DataType::Float32, false),
        2,
        true,
    )]));
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let v = Arc::new(FixedSizeListArray::new(
        item,
        2,
        Arc::new(Float32Array::from(vec![1.0f32, 2.0])),
        None,
    )) as ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![v]).unwrap();
    ctx.register_batch("t", batch).unwrap();
    let out = ctx
        .sql("SELECT vector_mean(v) AS r FROM t WHERE false")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let list = out[0]
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert!(list.is_null(0), "UDAF empty group is a null vector");

    // Tensor: a segment with no member is a zero row, NaN-free.
    let dev = Device::Cpu;
    let values = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap();
    let ids = Tensor::from_vec(vec![0u32], (1,), &dev).unwrap();
    // num_segments 2 leaves segment 1 empty.
    let pooled = segment_aggregate(&values, &ids, 2, SegmentReduce::Mean)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(pooled[1], vec![0.0, 0.0], "tensor empty segment is zero");
}

/// Permuting the rows (and ids) leaves the segment-aggregate output unchanged —
/// the determinism the UDAF guarantees, asserted on the tensor side.
#[test]
fn segment_aggregate_permutation_invariant() {
    for reduce in [SegmentReduce::Sum, SegmentReduce::Mean, SegmentReduce::Max] {
        let base = tensor_grouped(reduce);

        let rows = grouped_rows();
        let mut perm: Vec<usize> = (0..rows.len()).collect();
        perm.reverse();
        let dev = Device::Cpu;
        let flat: Vec<f32> = perm
            .iter()
            .flat_map(|&i| rows[i].1.iter().copied())
            .collect();
        let ids: Vec<u32> = perm.iter().map(|&i| dense_id(rows[i].0)).collect();
        let values = Tensor::from_vec(flat, (rows.len(), 2), &dev).unwrap();
        let ids = Tensor::from_vec(ids, (rows.len(),), &dev).unwrap();
        let permuted = segment_aggregate(&values, &ids, 3, reduce)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(base, permuted, "{reduce:?} not permutation-invariant");
    }
}

/// A tiny linear model trains to convergence through `train_loop` on synthetic
/// regression data — proving the parallel path runs autograd + optimizer with no
/// text machinery.
#[test]
fn train_loop_converges_on_synthetic_regression() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    // y = 2*x0 - 3*x1 + 0.5 ; a width-2 → width-1 linear must recover it.
    let linear = candle_nn::linear(2, 1, vb.pp("l")).unwrap();

    // 32 synthetic examples in one batch.
    let n = 32usize;
    let mut feats = Vec::with_capacity(n * 2);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let x0 = (i as f32 % 7.0) - 3.0;
        let x1 = (i as f32 % 5.0) - 2.0;
        feats.push(x0);
        feats.push(x1);
        targets.push(2.0 * x0 - 3.0 * x1 + 0.5);
    }
    let features = Tensor::from_vec(feats, (n, 2), &dev).unwrap();
    let targets = Tensor::from_vec(targets, (n, 1), &dev).unwrap();
    let batches = vec![TensorBatch { features, targets }];

    let config = ParallelTrainConfig {
        epochs: 400,
        learning_rate: 0.05,
        weight_decay: 0.0,
        grad_clip: 1.0,
    };

    let report = train_loop(
        &varmap,
        &batches,
        &config,
        &std::sync::atomic::AtomicBool::new(false),
        |batch: &TensorBatch| linear.forward(&batch.features).map_err(into_err),
        |preds, batch: &TensorBatch| {
            let diff = (preds - &batch.targets).map_err(into_err)?;
            diff.sqr().map_err(into_err)?.mean_all().map_err(into_err)
        },
    )
    .unwrap();

    assert_eq!(report.total_steps, 400, "one step per epoch (single batch)");
    assert!(
        report.final_loss < 1e-3,
        "linear regression did not converge: final loss {}",
        report.final_loss
    );
}

/// Structural decoupling proof: `train_loop` is callable with a model and loss
/// over plain tensors and `VarMap` — its signature references no tokenizer, no
/// `LoadedModel`, no `input_ids`. This compiles only because the parallel path
/// is text-free; it is the decoupling assertion the spec asks for (structural,
/// not a loss curve).
#[test]
fn train_loop_signature_is_text_free() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    // A single trainable scalar, driven purely as tensors.
    let w = Var::from_vec(vec![0.0f32], (1,), &dev).unwrap();
    varmap
        .data()
        .lock()
        .unwrap()
        .insert("w".to_string(), w.clone());

    let features = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();
    let targets = Tensor::from_vec(vec![3.0f32], (1, 1), &dev).unwrap();
    let batches = vec![TensorBatch { features, targets }];

    let w_for_model = w.clone();
    let report = train_loop(
        &varmap,
        &batches,
        &ParallelTrainConfig {
            epochs: 50,
            learning_rate: 0.2,
            weight_decay: 0.0,
            grad_clip: 0.0,
        },
        &std::sync::atomic::AtomicBool::new(false),
        move |batch: &TensorBatch| {
            let wt: &Tensor = &w_for_model;
            batch.features.broadcast_add(wt).map_err(into_err)
        },
        |preds, batch: &TensorBatch| {
            let diff = (preds - &batch.targets).map_err(into_err)?;
            diff.sqr().map_err(into_err)?.mean_all().map_err(into_err)
        },
    )
    .unwrap();

    // The bias should move toward 2.0 (so 1 + w ≈ 3); loss drops well below start.
    assert!(report.final_loss < 0.1, "final loss {}", report.final_loss);
}

fn into_err(e: candle_core::Error) -> jammi_db::error::JammiError {
    jammi_db::error::JammiError::FineTune(format!("{e}"))
}
