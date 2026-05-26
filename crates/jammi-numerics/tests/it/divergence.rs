use approx::assert_abs_diff_eq;
use jammi_numerics::divergence::{
    build_projection_vector, cosine_similarity_shift, jensen_shannon, project_all, psi,
    wasserstein_1d, PROJECTION_SEED,
};
use jammi_numerics::histogram::NUM_BINS;

#[test]
fn jensen_shannon_self_is_zero() {
    let v: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01).collect();
    let js = jensen_shannon(&v, &v, NUM_BINS);
    assert!(js.abs() < 1e-6, "expected near-zero JS, got {js}");
}

#[test]
fn jensen_shannon_in_unit_interval() {
    let a: Vec<f32> = (0..500).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..500).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let js = jensen_shannon(&a, &b, NUM_BINS);
    assert!((0.0..=1.0).contains(&js), "JS out of range: {js}");
}

#[test]
fn psi_self_is_zero() {
    let v: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01).collect();
    assert!(psi(&v, &v, NUM_BINS).abs() < 1e-6);
}

#[test]
fn psi_finite_under_smoothing() {
    let a: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let b: Vec<f32> = (40..90).map(|i| i as f32).collect();
    let p = psi(&a, &b, NUM_BINS);
    assert!(
        p.is_finite(),
        "PSI should be finite even with sparse overlap, got {p}"
    );
}

#[test]
fn wasserstein_self_is_zero() {
    let v: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01).collect();
    assert_abs_diff_eq!(wasserstein_1d(&v, &v), 0.0, epsilon = 1e-9);
}

#[test]
fn wasserstein_is_scale_invariant() {
    let a: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01 + 0.5).collect();
    let unscaled = wasserstein_1d(&a, &b);
    let a_scaled: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
    let b_scaled: Vec<f32> = b.iter().map(|x| x * 2.0).collect();
    let scaled = wasserstein_1d(&a_scaled, &b_scaled);
    assert_abs_diff_eq!(unscaled, scaled, epsilon = 1e-6);
}

#[test]
fn projection_vector_is_seeded() {
    let p1 = build_projection_vector(16);
    let p2 = build_projection_vector(16);
    assert_eq!(
        p1, p2,
        "seed {PROJECTION_SEED:#x} must produce deterministic vector"
    );
}

#[test]
fn project_all_produces_one_scalar_per_vector() {
    let projection = build_projection_vector(4);
    let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.5, -0.5, 1.0]).collect();
    let projected = project_all(&vectors, &projection);
    assert_eq!(projected.len(), vectors.len());
}

#[test]
fn cosine_similarity_shift_zero_when_current_matches_reference_mean() {
    // Two vectors at right angles; mean within-current cosine = 0.
    let current = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let shift = cosine_similarity_shift(0.0, &current);
    assert_abs_diff_eq!(shift, 0.0, epsilon = 1e-6);
}
