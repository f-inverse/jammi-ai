use jammi_numerics::stats::{bootstrap_ci, mann_whitney_u, sign_test, welch_t_test};
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
fn bootstrap_ci_refuses_nan_sample_without_invoking_statistic_fn() {
    // The edge guard must fire on the *presence* of a NaN sample, before any
    // resample is drawn — not rely on a downstream, output-edge check that
    // only catches a NaN that happens to land in a particular resample. A
    // seed-keyed assertion (e.g. picking a seed empirically known to draw
    // only the non-NaN position and arguing "even this lucky seed is still
    // refused") would be unsound: rand's `StdRng` stream is explicitly NOT
    // value-stable across `rand` versions, so any "known-lucky seed" claim
    // can silently go false on a dependency bump, and the test would then
    // pass vacuously (for the wrong reason) or fail for an unrelated one.
    // This test instead proves the guard-placement claim directly and
    // seed-independently: a `Cell<bool>` records whether `statistic_fn` was
    // EVER invoked. If the input-edge guard is what causes the error,
    // `statistic_fn` is never called at all — no resample, lucky or not,
    // gets the chance to be the one that "missed" the NaN.
    use std::cell::Cell;
    let invoked = Cell::new(false);
    let mean = |xs: &[f64]| {
        invoked.set(true);
        xs.iter().sum::<f64>() / xs.len() as f64
    };
    let samples = [1.0, f64::NAN];
    let result = bootstrap_ci(&samples, mean, 1_000, 0.05, 2);
    assert!(
        result.is_err(),
        "a NaN sample must be refused before any draw"
    );
    assert!(
        !invoked.get(),
        "statistic_fn must never be invoked once a NaN sample is present — \
         the input-edge guard must fire before any resample is drawn, so no \
         seed's draw sequence can be relevant to whether this errors"
    );
}

#[test]
fn bootstrap_ci_refuses_mixed_sign_infinite_samples() {
    // samples = [+inf, -inf, 1.0]: a resample that draws both infinities
    // sums to `inf + -inf = NaN` for the mean. Whether any given resample
    // draws both is a function of `n`, `iterations`, and `seed` — a guard
    // that only refuses at the OUTPUT edge (on the resulting NaN statistic)
    // makes `Ok`/`Err` a seed lottery instead of a deterministic property of
    // the input. Refusing non-finite samples at the INPUT edge removes the
    // lottery entirely, so every seed must be refused, not just an unlucky
    // one — swept across many seeds to prove this isn't one seed's accident.
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let samples = [f64::INFINITY, f64::NEG_INFINITY, 1.0];
    for seed in 0..64u64 {
        let result = bootstrap_ci(&samples, mean, 3, 0.05, seed);
        assert!(
            result.is_err(),
            "mixed-sign infinite samples must be refused at the input edge \
             regardless of seed (seed={seed})"
        );
    }
}

#[test]
fn bootstrap_ci_refuses_infinite_sample() {
    // A single-sign `±inf` sample is refused too, not just the mixed-sign
    // case above: the contract is "refuse any non-finite sample", not
    // "refuse only combinations that happen to produce NaN for `mean`". A
    // narrower same-sign-only rule would still make the resample statistic's
    // finiteness depend on `statistic_fn` (e.g. `a - b` on two same-signed
    // infinities is NaN even though their sum is not) — refusing at the
    // input edge regardless of sign is the simpler, statistic-fn-agnostic
    // contract.
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let samples = [1.0, f64::INFINITY, 3.0];
    assert!(bootstrap_ci(&samples, mean, 100, 0.05, 42).is_err());
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

// --- sign_test -------------------------------------------------------------

#[test]
fn sign_test_golden_n12_k11_pinned_cell() {
    // Pre-registered decision cell (CONTRACT H2 / PLAN v2 delta 3): n=12,
    // k=11 positives (1 negative, 0 ties). Hand computation:
    //   t = max(11, 1) = 11
    //   tail = C(12,11) + C(12,12) = 12 + 1 = 13
    //   p = 2 * tail / 2^12 = 2 * 13 / 4096 = 26 / 4096 = 13 / 2048
    //     = 0.00634765625
    // which rounds to the pre-registered alpha2 = 0.0064 (2 s.f.) /
    // 0.006348 (6 d.p.). The exact value returned here is the unrounded
    // rational 13/2048, pinned bit-exactly.
    let mut diffs = vec![1.0; 11];
    diffs.push(-1.0);
    let r = sign_test(&diffs).unwrap();
    assert_eq!(r.n, 12);
    assert_eq!(r.n_pos, 11);
    assert_eq!(r.n_neg, 1);
    assert_eq!(r.ties, 0);
    let expected = 13.0_f64 / 2048.0_f64;
    assert_eq!(
        r.p_value.to_bits(),
        expected.to_bits(),
        "golden cell (n=12, k=11) must equal the exact rational 13/2048 = {expected} bit-for-bit, got {}",
        r.p_value
    );
    assert!(
        (r.p_value - 0.0064).abs() < 0.0005,
        "sanity: exact p_value {} should round to the pre-registered alpha2=0.0064",
        r.p_value
    );
}

#[test]
fn sign_test_exact_small_n_all_same_sign() {
    // n=5, k=5 (unanimous): t = max(5,0) = 5.
    //   tail = C(5,5) = 1
    //   p = 2 * 1 / 2^5 = 2/32 = 0.0625
    let diffs = vec![2.0, 3.0, 1.0, 0.5, 7.0];
    let r = sign_test(&diffs).unwrap();
    assert_eq!((r.n, r.n_pos, r.n_neg, r.ties), (5, 5, 0, 0));
    assert_eq!(r.p_value.to_bits(), (2.0_f64 / 32.0_f64).to_bits());
}

#[test]
fn sign_test_exact_small_n_one_dissent() {
    // n=5, k=4: t = max(4,1) = 4.
    //   tail = C(5,4) + C(5,5) = 5 + 1 = 6
    //   p = 2 * 6 / 32 = 12/32 = 0.375
    let diffs = vec![2.0, 3.0, 1.0, 0.5, -7.0];
    let r = sign_test(&diffs).unwrap();
    assert_eq!((r.n, r.n_pos, r.n_neg, r.ties), (5, 4, 1, 0));
    assert_eq!(r.p_value.to_bits(), (12.0_f64 / 32.0_f64).to_bits());
}

#[test]
fn sign_test_exact_small_n_balanced_saturates_at_one() {
    // n=4, k=2 (perfectly balanced): t = max(2,2) = 2.
    //   tail = C(4,2) + C(4,3) + C(4,4) = 6 + 4 + 1 = 11
    //   2 * 11 = 22 > 2^4 = 16, so p is capped at the honest ceiling of 1.0
    //   (not 22/16 = 1.375, which would not be a probability).
    let diffs = vec![1.0, 1.0, -1.0, -1.0];
    let r = sign_test(&diffs).unwrap();
    assert_eq!((r.n, r.n_pos, r.n_neg, r.ties), (4, 2, 2, 0));
    assert_eq!(r.p_value, 1.0);
}

#[test]
fn sign_test_two_sided_symmetry_k_and_n_minus_k() {
    // Flipping every sign swaps n_pos <-> n_neg but must not change the
    // two-sided p-value: t = max(k, n-k) = max(n-k, k) is symmetric by
    // construction. Asserted bit-exactly since both paths compute the same
    // exact-integer ratio.
    let diffs = vec![
        3.0, 1.5, -2.0, 4.0, 0.2, -0.1, 5.0, -6.0, 0.9, 2.2, -1.1, 8.0,
    ];
    let flipped: Vec<f64> = diffs.iter().map(|d| -d).collect();

    let r = sign_test(&diffs).unwrap();
    let r_flipped = sign_test(&flipped).unwrap();

    assert_eq!(r.n, r_flipped.n);
    assert_eq!(r.n_pos, r_flipped.n_neg);
    assert_eq!(r.n_neg, r_flipped.n_pos);
    assert_eq!(
        r.p_value.to_bits(),
        r_flipped.p_value.to_bits(),
        "p_value must be symmetric under a full sign flip: {} vs {}",
        r.p_value,
        r_flipped.p_value
    );
}

#[test]
fn sign_test_reports_ties_without_dropping_them() {
    // diffs = [1,1,1,0,0,-1]: n_pos=3, n_neg=1, ties=2, n=4.
    //   t = max(3,1) = 3
    //   tail = C(4,3) + C(4,4) = 4 + 1 = 5
    //   p = 2 * 5 / 16 = 10/16 = 0.625
    let diffs = vec![1.0, 1.0, 1.0, 0.0, 0.0, -1.0];
    let r = sign_test(&diffs).unwrap();
    assert_eq!(r.n, 4, "ties must be excluded from n, not folded in");
    assert_eq!(r.n_pos, 3);
    assert_eq!(r.n_neg, 1);
    assert_eq!(r.ties, 2, "the tie count must be reported, not dropped");
    assert_eq!(r.p_value.to_bits(), (10.0_f64 / 16.0_f64).to_bits());
}

#[test]
fn sign_test_refuses_empty_input() {
    let err = sign_test(&[]).unwrap_err();
    assert!(format!("{err}").contains("n=0"));
}

#[test]
fn sign_test_refuses_all_ties() {
    // Non-empty input, but every element resolves to a tie: n=0 is reached
    // via a distinct path from the empty-input case, and the two must not be
    // conflated in the error text.
    let diffs = vec![0.0, 0.0, 0.0];
    let err = sign_test(&diffs).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("tie") || msg.contains("tied"),
        "all-tie refusal should name ties specifically, got: {msg}"
    );
}

#[test]
fn sign_test_refuses_nan() {
    // NaN has no sign (NaN > 0.0, NaN < 0.0, and NaN == 0.0 are all false),
    // so it must be refused rather than silently routed into positive,
    // negative, or tie.
    let diffs = vec![1.0, f64::NAN, -1.0];
    assert!(sign_test(&diffs).is_err());
}

#[test]
fn sign_test_admits_infinite_as_a_well_defined_sign() {
    // Unlike a tie, `±inf` has a well-defined, non-zero sign and must
    // classify like any other non-zero finite value, not be refused and not
    // be counted as a tie.
    let diffs = vec![f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0, 3.0];
    let r = sign_test(&diffs).unwrap();
    assert_eq!(r.n, 5);
    assert_eq!(r.n_pos, 4);
    assert_eq!(r.n_neg, 1);
    assert_eq!(r.ties, 0);
    assert!(r.p_value.is_finite());
}

#[test]
fn sign_test_negative_control_balanced_is_not_significant() {
    // Non-vacuous negative control: 6 positive / 6 negative out of n=12
    // (perfectly balanced) must NOT report significance under any
    // conventional alpha — a naive or buggy implementation that always
    // reports a small p-value would fail this.
    let mut diffs = vec![1.0; 6];
    diffs.extend(vec![-1.0; 6]);
    let r = sign_test(&diffs).unwrap();
    assert_eq!(r.n, 12);
    assert!(
        r.p_value > 0.5,
        "a perfectly balanced 6/6 split at n=12 must not look significant, got p={}",
        r.p_value
    );
}

#[test]
fn sign_test_is_invariant_to_input_order() {
    // The sign test is a property of the multiset of signs, not the order
    // paired differences arrive in. A permutation must produce a
    // bit-identical result.
    let ordered = vec![
        1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
    ];
    let mut shuffled = ordered.clone();
    shuffled.reverse();
    shuffled.rotate_left(3);
    assert_ne!(ordered, shuffled, "the two orders must actually differ");

    let a = sign_test(&ordered).unwrap();
    let b = sign_test(&shuffled).unwrap();
    assert_eq!(a.n, b.n);
    assert_eq!(a.n_pos, b.n_pos);
    assert_eq!(a.n_neg, b.n_neg);
    assert_eq!(a.ties, b.ties);
    assert_eq!(a.p_value.to_bits(), b.p_value.to_bits());
}
