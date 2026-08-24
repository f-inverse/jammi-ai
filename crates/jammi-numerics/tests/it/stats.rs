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
fn mann_whitney_refuses_nan_in_a() {
    let a = [1.0, f64::NAN, 3.0];
    let b = sample_normal(0.0, 1.0, 5, 1);
    assert!(mann_whitney_u(&a, &b).is_err());
}

#[test]
fn mann_whitney_admits_infinite_in_b() {
    // `±inf` is a well-ordered, meaningful observation for a rank-based
    // test (it is simply the most extreme rank) — empirically confirmed
    // against `scipy.stats.mannwhitneyu`, which returns a finite statistic
    // and p-value for `mannwhitneyu([1.0, inf, 3.0], [2.0, 4.0, 5.0])`
    // rather than refusing.
    let a = sample_normal(0.0, 1.0, 5, 1);
    let b = [1.0, f64::INFINITY, 3.0];
    let r = mann_whitney_u(&a, &b).unwrap();
    assert!(r.statistic.is_finite());
    assert!(r.p_value.is_finite());
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

#[test]
fn bootstrap_ci_refuses_nan_sample() {
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let samples = [1.0, f64::NAN, 3.0];
    assert!(bootstrap_ci(&samples, mean, 100, 0.05, 42).is_err());
}

#[test]
fn bootstrap_ci_refuses_nan_sample_even_when_a_lucky_seed_would_never_draw_it() {
    // The edge guard must fire on the *presence* of a NaN sample, not on
    // whether the seeded resampler happens to draw it. With samples =
    // [1.0, NaN] (n = 2) and iterations = 1, the resampler draws exactly 2
    // positions from the sorted basis [1.0, NaN] (index 0 = 1.0, index 1 =
    // NaN under `total_cmp`'s total order, since positive NaN sorts as the
    // IEEE-754 maximum); seed 2 draws index 0 both times (empirically
    // confirmed against `rand::rngs::StdRng` in this crate's `rand`
    // version), so a resample that never includes the NaN element and a
    // downstream statistic-finite check alone would silently accept this
    // input and return a clean-looking Ok(Interval { lower: 1.0, upper: 1.0
    // }) — this is exactly the corner a downstream-only guard misses.
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let samples = [1.0, f64::NAN];
    let result = bootstrap_ci(&samples, mean, 1, 0.05, 2);
    assert!(
        result.is_err(),
        "a NaN sample must be refused before any draw, independent of whether \
         this seed's draws would have missed it"
    );
}

#[test]
fn bootstrap_ci_admits_infinite_sample() {
    // Unlike NaN, `±inf` has a well-defined position under `total_cmp` and
    // is a legitimate observed value a heavy-tailed statistic can produce a
    // finite or infinite result from; it is not refused at the input edge.
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let samples = [1.0, f64::INFINITY, 3.0];
    assert!(bootstrap_ci(&samples, mean, 100, 0.05, 42).is_ok());
}

#[test]
fn bootstrap_ci_admits_infinite_statistic() {
    // `statistic_fn` is caller-supplied and can produce `±inf` purely from
    // its own arithmetic (e.g. a ratio that divides by zero on every
    // resample) even when every input sample is finite. `±inf` is a
    // legitimate percentile-selection outcome (no arithmetic is performed
    // on `stats` beyond ordering), so this is not refused: every resample
    // here is deterministically [1.0, 1.0, 1.0, 1.0], whose denom is
    // always 0.0, so `1.0 / 0.0 = +inf` on every draw and the resulting
    // interval is `(+inf, +inf)`.
    let samples: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
    let inf_statistic = |xs: &[f64]| {
        let denom = xs.iter().map(|x| x - 1.0).sum::<f64>();
        xs[0] / denom
    };
    let ci = bootstrap_ci(&samples, inf_statistic, 100, 0.05, 42).unwrap();
    assert_eq!(ci.lower, f64::INFINITY);
    assert_eq!(ci.upper, f64::INFINITY);
}

#[test]
fn bootstrap_ci_refuses_nan_statistic() {
    // Unlike the `±inf` case above, a genuine `0.0 / 0.0` NaN has no
    // numeric meaning and must still be refused, purely from
    // `statistic_fn`'s own arithmetic — every input sample here is finite.
    let samples: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
    let nan_statistic = |xs: &[f64]| {
        let denom = xs.iter().map(|x| x - 1.0).sum::<f64>();
        denom / denom
    };
    assert!(bootstrap_ci(&samples, nan_statistic, 100, 0.05, 42).is_err());
}

#[test]
fn bootstrap_ci_is_invariant_to_input_order() {
    // The bootstrap is a property of the sample multiset, not its order. The
    // resampler draws positions under a fixed seed, so before the canonical
    // basis this same multiset in two orders selected different values and
    // produced two different intervals. A permutation must now be byte-identical.
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let ordered: Vec<f64> = vec![0.1, 0.4, 0.2, 0.9, 0.3, 0.7, 0.5, 0.8, 0.6, 0.0];
    let mut shuffled = ordered.clone();
    shuffled.reverse();
    shuffled.rotate_left(3);
    assert_ne!(ordered, shuffled, "the two orders must actually differ");

    let a = bootstrap_ci(&ordered, mean, 10_000, 0.05, 0x6a616d6d695f7031).unwrap();
    let b = bootstrap_ci(&shuffled, mean, 10_000, 0.05, 0x6a616d6d695f7031).unwrap();
    assert_eq!(
        a.lower.to_bits(),
        b.lower.to_bits(),
        "ci_lower must be byte-identical across input orders"
    );
    assert_eq!(
        a.upper.to_bits(),
        b.upper.to_bits(),
        "ci_upper must be byte-identical across input orders"
    );
}
