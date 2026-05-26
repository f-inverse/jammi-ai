use approx::assert_abs_diff_eq;
use jammi_numerics::histogram::{
    bin_proportions, interpolate_to, padded_range, smooth_and_renormalise, NUM_BINS, RANGE_PADDING,
};

#[test]
fn bin_proportions_sum_to_one() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
    let bins = bin_proportions(&data, 0.0, 1.0, NUM_BINS);
    let sum: f64 = bins.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
}

#[test]
fn bin_proportions_empty_is_all_zero() {
    let bins = bin_proportions(&[], 0.0, 1.0, NUM_BINS);
    assert_eq!(bins.len(), NUM_BINS);
    assert!(bins.iter().all(|&b| b == 0.0));
}

#[test]
fn smooth_preserves_normalisation() {
    let mut bins = vec![0.0, 0.0, 0.5, 0.5, 0.0];
    smooth_and_renormalise(&mut bins);
    let sum: f64 = bins.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    // Every entry should now be strictly positive (additive smoothing).
    assert!(bins.iter().all(|&b| b > 0.0));
}

#[test]
fn padded_range_widens_actual_extent() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let (lo, hi) = padded_range(&data);
    assert!(lo < 1.0, "lower bound {lo} should be < 1.0");
    assert!(hi > 3.0, "upper bound {hi} should be > 3.0");
    // RANGE_PADDING relative to span = 2.0 ⇒ pad = 0.02 each side.
    let span = hi - lo;
    assert!(
        span > 2.0 + RANGE_PADDING,
        "span {span} should include both pads"
    );
}

#[test]
fn padded_range_empty_returns_unit_interval() {
    let (lo, hi) = padded_range(&[]);
    assert_eq!((lo, hi), (0.0, 1.0));
}

#[test]
fn interpolate_to_three_points() {
    let out = interpolate_to(&[0.0, 1.0], 3);
    assert_eq!(out.len(), 3);
    assert_abs_diff_eq!(out[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(out[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(out[2], 1.0, epsilon = 1e-12);
}

#[test]
fn interpolate_to_preserves_identical_length() {
    let out = interpolate_to(&[1.0, 2.0, 3.0], 3);
    assert_eq!(out, vec![1.0, 2.0, 3.0]);
}
