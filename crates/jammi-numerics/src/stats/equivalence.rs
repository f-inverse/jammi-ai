//! Paired equivalence test: is a mean paired difference small enough to call
//! the two sides the same?
//!
//! A difference test that finds nothing has not shown equivalence — it may
//! only have been underpowered. Equivalence is its own claim with its own
//! null: *the true mean difference is at least `delta` away from zero*. That
//! null is rejected at level `alpha` exactly when the `1 - 2 * alpha`
//! confidence interval of the mean difference lies inside `(-delta, +delta)`
//! — the interval form of the two one-sided tests, each one-sided test
//! spending `alpha` on its own tail.
//!
//! `delta` is an input, never an output: it is the smallest difference that
//! would matter, fixed before the differences are seen. Nothing here can
//! check *when* a caller chose it; what this function does guarantee is that
//! it is refused unless it is a finite positive number, so "no margin was
//! set" can never be mistaken for a margin of zero or infinity.

use crate::error::{NumericsError, Result};
use crate::stats::bootstrap::bootstrap_ci;
use crate::stats::summary::mean;
use crate::stats::types::EquivalenceResult;

/// Equivalence of paired differences `diffs[i] = a_i - b_i` within `±delta`
/// at level `alpha`, by the percentile bootstrap interval of their mean.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] when `diffs` has fewer than two values or
/// a non-finite one, when `delta` is not finite and strictly positive, when
/// `alpha` is outside `(0, 0.5)` (the interval's level `1 - 2 * alpha` must
/// be positive), or when `iterations == 0`.
pub fn paired_equivalence(
    diffs: &[f64],
    delta: f64,
    iterations: usize,
    alpha: f64,
    seed: u64,
) -> Result<EquivalenceResult> {
    if diffs.len() < 2 {
        return Err(NumericsError::InvalidInput(format!(
            "equivalence requires at least 2 paired differences, got {}",
            diffs.len()
        )));
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(NumericsError::InvalidInput(format!(
            "equivalence margin must be finite and > 0, got {delta}"
        )));
    }
    if !(0.0 < alpha && alpha < 0.5) {
        return Err(NumericsError::InvalidInput(format!(
            "equivalence alpha out of range (must be in (0, 0.5)): {alpha}"
        )));
    }
    let interval = bootstrap_ci(
        diffs,
        |s| s.iter().sum::<f64>() / s.len() as f64,
        iterations,
        2.0 * alpha,
        seed,
    )?;
    Ok(EquivalenceResult {
        mean: mean(diffs)?,
        interval,
        delta,
        equivalent: -delta < interval.lower && interval.upper < delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIFFS: [f64; 12] = [
        0.004, -0.006, 0.002, -0.001, 0.005, -0.003, 0.001, -0.004, 0.003, -0.002, 0.006, -0.005,
    ];

    #[test]
    fn small_differences_inside_the_margin_are_equivalent() {
        let r = paired_equivalence(&DIFFS, 0.01, 4000, 0.05, 1).unwrap();
        assert!(r.equivalent, "{r:?}");
        assert!(r.interval.lower <= r.mean && r.mean <= r.interval.upper);
    }

    #[test]
    fn the_same_differences_are_not_equivalent_under_a_tighter_margin() {
        let r = paired_equivalence(&DIFFS, 0.001, 4000, 0.05, 1).unwrap();
        assert!(!r.equivalent, "{r:?}");
    }

    #[test]
    fn a_shift_past_the_margin_is_not_equivalent() {
        let shifted: Vec<f64> = DIFFS.iter().map(|d| d + 0.02).collect();
        let r = paired_equivalence(&shifted, 0.01, 4000, 0.05, 1).unwrap();
        assert!(!r.equivalent, "{r:?}");
        assert!(r.interval.lower > 0.01);
    }

    /// No detected difference is not equivalence: a wide, centred scatter
    /// has a mean near zero and still fails the margin.
    #[test]
    fn a_noisy_zero_mean_sample_is_not_equivalent() {
        let noisy: Vec<f64> = DIFFS.iter().map(|d| d * 40.0).collect();
        let r = paired_equivalence(&noisy, 0.05, 4000, 0.05, 1).unwrap();
        assert!(r.mean.abs() < 0.05);
        assert!(!r.equivalent, "{r:?}");
    }

    #[test]
    fn refuses_an_unset_or_nonsensical_margin() {
        for delta in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(paired_equivalence(&DIFFS, delta, 100, 0.05, 1).is_err());
        }
        assert!(paired_equivalence(&DIFFS[..1], 0.1, 100, 0.05, 1).is_err());
        assert!(paired_equivalence(&DIFFS, 0.1, 100, 0.5, 1).is_err());
        assert!(paired_equivalence(&DIFFS, 0.1, 0, 0.05, 1).is_err());
    }
}
