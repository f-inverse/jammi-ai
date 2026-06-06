//! Finite-difference gradient checks for the differentiable segment-aggregate.
//!
//! For each reduction the analytic gradient from `.backward()` must match the
//! numerical gradient of a scalar objective `sum(weights * segment_aggregate(v))`
//! w.r.t. each input value. The objective's weights are distinct per output cell
//! so the check pins which member row each output is differentiated against —
//! catching a Sum that scattered to the wrong segment, a Mean that forgot the
//! `1/count` factor, or a Max that routed gradient to a non-winning (or
//! non-member) row.

use candle_core::{Device, Tensor, Var};
use jammi_encoders::aggregate::{segment_aggregate, SegmentReduce};

/// Scalar objective `sum(weights ⊙ segment_aggregate(values, ids))`, returning a
/// differentiable f32 tensor. `weights` makes each output cell contribute a
/// distinct, known coefficient so the gradient w.r.t. each input is unambiguous.
fn objective(
    values: &Tensor,
    ids: &Tensor,
    num_segments: usize,
    reduce: SegmentReduce,
    weights: &Tensor,
) -> Tensor {
    let pooled = segment_aggregate(values, ids, num_segments, reduce).unwrap();
    (pooled * weights).unwrap().sum_all().unwrap()
}

/// Forward-only objective value as f64, used for the numerical difference.
fn objective_value(
    values: &Tensor,
    ids: &Tensor,
    num_segments: usize,
    reduce: SegmentReduce,
    weights: &Tensor,
) -> f64 {
    objective(values, ids, num_segments, reduce, weights)
        .to_scalar::<f32>()
        .unwrap() as f64
}

fn grad_check(reduce: SegmentReduce) {
    let dev = Device::Cpu;
    // 4 rows, 2 lanes; ids [0,0,1,2] → segment 0 has rows 0,1; segment 1 row 2;
    // segment 2 row 3 (every segment non-empty so Max has a defined winner).
    let raw = vec![1.0f32, 4.0, 3.0, 2.0, 7.0, 5.0, -1.0, 6.0];
    let var = Var::from_vec(raw.clone(), (4, 2), &dev).unwrap();
    let values: &Tensor = &var;
    let ids = Tensor::from_vec(vec![0u32, 0, 1, 2], (4,), &dev).unwrap();
    let num_segments = 3;
    // Distinct per-cell weights.
    let weights = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (num_segments, 2),
        &dev,
    )
    .unwrap();

    // Analytic gradient.
    let loss = objective(values, &ids, num_segments, reduce, &weights);
    let grads = loss.backward().unwrap();
    let analytic = grads.get(values).unwrap().to_vec2::<f32>().unwrap();

    // Numerical gradient via central differences on each input cell.
    let eps = 1e-2f64;
    for r in 0..4 {
        for c in 0..2 {
            let mut up = raw.clone();
            let mut down = raw.clone();
            up[r * 2 + c] += eps as f32;
            down[r * 2 + c] -= eps as f32;
            let v_up = Tensor::from_vec(up, (4, 2), &dev).unwrap();
            let v_down = Tensor::from_vec(down, (4, 2), &dev).unwrap();
            let f_up = objective_value(&v_up, &ids, num_segments, reduce, &weights);
            let f_down = objective_value(&v_down, &ids, num_segments, reduce, &weights);
            let numerical = (f_up - f_down) / (2.0 * eps);
            let a = analytic[r][c] as f64;
            assert!(
                (a - numerical).abs() < 1e-2,
                "{reduce:?} grad mismatch at ({r},{c}): analytic {a}, numerical {numerical}"
            );
        }
    }
}

#[test]
fn sum_gradient_matches_finite_difference() {
    grad_check(SegmentReduce::Sum);
}

#[test]
fn mean_gradient_matches_finite_difference() {
    grad_check(SegmentReduce::Mean);
}

#[test]
fn max_gradient_matches_finite_difference() {
    grad_check(SegmentReduce::Max);
}

/// Max must hand all gradient to the winning member and none to a losing member
/// or to a non-member of the segment: differentiate a single-output objective
/// and read which input rows received gradient.
#[test]
fn max_gradient_only_to_winner() {
    let dev = Device::Cpu;
    // Segment 0 = rows 0,1; segment 1 = row 2. In lane 0, row 1 (4.0) beats row
    // 0 (1.0); in lane 1, row 0 (5.0) beats row 1 (2.0).
    let raw = vec![1.0f32, 5.0, 4.0, 2.0, 9.0, 9.0];
    let var = Var::from_vec(raw, (3, 2), &dev).unwrap();
    let values: &Tensor = &var;
    let ids = Tensor::from_vec(vec![0u32, 0, 1], (3,), &dev).unwrap();
    let weights = Tensor::ones((2, 2), candle_core::DType::F32, &dev).unwrap();

    let loss = objective(values, &ids, 2, SegmentReduce::Max, &weights);
    let grads = loss.backward().unwrap();
    let g = grads.get(values).unwrap().to_vec2::<f32>().unwrap();

    // Lane 0 winner is row 1; lane 1 winner is row 0. Row 2 is the sole member
    // of segment 1, so both its lanes win.
    assert_eq!(g[0], vec![0.0, 1.0], "row 0 grad: only lane 1 wins");
    assert_eq!(g[1], vec![1.0, 0.0], "row 1 grad: only lane 0 wins");
    assert_eq!(
        g[2],
        vec![1.0, 1.0],
        "row 2 grad: singleton wins both lanes"
    );
}
