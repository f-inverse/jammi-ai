use jammi_numerics::stats::{bootstrap_ci, mann_whitney_u, welch_t_test};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn sample_normal(mean: f64, std: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(mean, std).unwrap();
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

#[test]
fn welch_distinguishes_shifted_normals() {
    // 1.0σ shift at n=30 has power ~0.97 and is robust to seed variance
    // — what we assert is the kernel's correctness, not its statistical
    // power, so we deliberately pick an effect size where seed-induced
    // sample noise can't flip the verdict.
    let a = sample_normal(0.0, 1.0, 30, 42);
    let b = sample_normal(1.0, 1.0, 30, 43);
    let r = welch_t_test(&a, &b).unwrap();
    assert!(
        r.p_value < 0.05,
        "Welch should reject H0 for a 1.0σ shift at n=30, got p={}",
        r.p_value
    );
}

#[test]
fn welch_does_not_falsely_reject_identical_normals() {
    let a = sample_normal(0.0, 1.0, 30, 42);
    let b = sample_normal(0.0, 1.0, 30, 43);
    let r = welch_t_test(&a, &b).unwrap();
    assert!(
        r.p_value >= 0.01,
        "Welch should not reject H0 for two N(0,1) samples, got p={}",
        r.p_value
    );
}

#[test]
fn welch_errors_on_undersized_inputs() {
    assert!(welch_t_test(&[1.0], &[2.0, 3.0]).is_err());
}

#[test]
fn mann_whitney_distinguishes_shifted_normals() {
    let a = sample_normal(0.0, 1.0, 30, 42);
    let b = sample_normal(1.0, 1.0, 30, 43);
    let r = mann_whitney_u(&a, &b).unwrap();
    assert!(
        r.p_value < 0.05,
        "Mann-Whitney should reject H0 for a 1.0σ shift at n=30, got p={}",
        r.p_value
    );
}

#[test]
fn mann_whitney_does_not_falsely_reject_identical_normals() {
    let a = sample_normal(0.0, 1.0, 30, 42);
    let b = sample_normal(0.0, 1.0, 30, 43);
    let r = mann_whitney_u(&a, &b).unwrap();
    assert!(
        r.p_value >= 0.01,
        "Mann-Whitney should not reject H0 for two N(0,1) samples, got p={}",
        r.p_value
    );
}

#[test]
fn bootstrap_ci_contains_true_mean() {
    let samples: Vec<f64> = (1..=5).map(|x| x as f64).collect();
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let ci = bootstrap_ci(&samples, mean, 2000, 0.05, 42).unwrap();
    assert!(
        ci.lower <= 3.0 && 3.0 <= ci.upper,
        "CI [{}, {}] should contain true mean 3.0",
        ci.lower,
        ci.upper
    );
}

#[test]
fn bootstrap_ci_errors_on_empty_samples() {
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(bootstrap_ci(&[], mean, 100, 0.05, 42).is_err());
}
