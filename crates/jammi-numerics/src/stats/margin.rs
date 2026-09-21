//! Paired margin tests: is a mean paired difference small enough to call the
//! two sides the same — or, one-sidedly, to call one side no worse?
//!
//! A difference test that finds nothing has not shown equivalence — it may
//! only have been underpowered. A margin claim is its own claim with its own
//! null. There are two one-sided nulls, each rejected at level `alpha` from
//! one bound of the `1 - 2 * alpha` confidence interval of the mean
//! difference:
//!
//! - *the true mean is at least `+delta`* — rejected when the interval's
//!   upper bound is below `+delta`;
//! - *the true mean is at most `-delta`* — rejected when the interval's lower
//!   bound is above `-delta`.
//!
//! **Non-inferiority** is one of them, chosen by which direction is better:
//! where lower is better, `a` is no worse than `b` by `delta` when the first
//! null is rejected, and a difference far on the *favourable* side is no
//! evidence against the claim. **Equivalence** is both at once (the two
//! one-sided tests), each spending `alpha` on its own tail. One interval
//! therefore answers all three questions, and [`MarginTestResult`] carries
//! the two one-sided outcomes rather than only their conjunction.
//!
//! `delta` is an input, never an output: it is the smallest difference that
//! would matter, fixed before the differences are seen. Nothing here can
//! check *when* a caller chose it; what this function does guarantee is that
//! it is refused unless it is a finite positive number, so "no margin was
//! set" can never be mistaken for a margin of zero or infinity.

use crate::error::{NumericsError, Result};
use crate::stats::bootstrap::bootstrap_ci;
use crate::stats::summary::mean;
use crate::stats::types::MarginTestResult;

/// Both one-sided margin tests of paired differences `diffs[i] = a_i - b_i`
/// against `±delta`, each at level `alpha`, by the percentile bootstrap
/// interval of their mean.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] when `diffs` has fewer than two values or
/// a non-finite one, when `delta` is not finite and strictly positive, when
/// `alpha` is outside `(0, 0.5)` (the interval's level `1 - 2 * alpha` must
/// be positive), or when `iterations == 0`.
pub fn paired_margin_test(
    diffs: &[f64],
    delta: f64,
    iterations: usize,
    alpha: f64,
    seed: u64,
) -> Result<MarginTestResult> {
    if diffs.len() < 2 {
        return Err(NumericsError::InvalidInput(format!(
            "a margin test requires at least 2 paired differences, got {}",
            diffs.len()
        )));
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(NumericsError::InvalidInput(format!(
            "the margin must be finite and > 0, got {delta}"
        )));
    }
    if !(0.0 < alpha && alpha < 0.5) {
        return Err(NumericsError::InvalidInput(format!(
            "margin-test alpha out of range (must be in (0, 0.5)): {alpha}"
        )));
    }
    let interval = bootstrap_ci(
        diffs,
        |s| s.iter().sum::<f64>() / s.len() as f64,
        iterations,
        2.0 * alpha,
        seed,
    )?;
    Ok(MarginTestResult {
        mean: mean(diffs)?,
        interval,
        delta,
        below_upper_margin: interval.upper < delta,
        above_lower_margin: -delta < interval.lower,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::types::Better;

    const DIFFS: [f64; 12] = [
        0.004, -0.006, 0.002, -0.001, 0.005, -0.003, 0.001, -0.004, 0.003, -0.002, 0.006, -0.005,
    ];

    #[test]
    fn small_differences_inside_the_margin_are_equivalent() {
        let r = paired_margin_test(&DIFFS, 0.01, 4000, 0.05, 1).unwrap();
        assert!(r.equivalent(), "{r:?}");
        assert!(r.interval.lower <= r.mean && r.mean <= r.interval.upper);
    }

    #[test]
    fn the_same_differences_are_not_equivalent_under_a_tighter_margin() {
        let r = paired_margin_test(&DIFFS, 0.001, 4000, 0.05, 1).unwrap();
        assert!(!r.equivalent(), "{r:?}");
    }

    #[test]
    fn a_shift_past_the_margin_is_not_equivalent() {
        let shifted: Vec<f64> = DIFFS.iter().map(|d| d + 0.02).collect();
        let r = paired_margin_test(&shifted, 0.01, 4000, 0.05, 1).unwrap();
        assert!(!r.equivalent(), "{r:?}");
        assert!(r.interval.lower > 0.01);
    }

    /// No detected difference is not equivalence: a wide, centred scatter
    /// has a mean near zero and still fails the margin.
    #[test]
    fn a_noisy_zero_mean_sample_is_not_equivalent() {
        let noisy: Vec<f64> = DIFFS.iter().map(|d| d * 40.0).collect();
        let r = paired_margin_test(&noisy, 0.05, 4000, 0.05, 1).unwrap();
        assert!(r.mean.abs() < 0.05);
        assert!(!r.equivalent(), "{r:?}");
    }

    /// Better is not a failure of "no worse": a mean far on the favourable
    /// side fails equivalence and passes non-inferiority, and the mirror
    /// image fails it.
    #[test]
    fn a_favourable_shift_past_the_margin_is_non_inferior_and_not_equivalent() {
        let better: Vec<f64> = DIFFS.iter().map(|d| d - 0.02).collect();
        let r = paired_margin_test(&better, 0.01, 4000, 0.05, 1).unwrap();
        assert!(!r.equivalent(), "{r:?}");
        assert!(r.non_inferior(Better::Lower), "{r:?}");
        assert!(!r.non_inferior(Better::Higher), "{r:?}");

        let worse: Vec<f64> = DIFFS.iter().map(|d| d + 0.02).collect();
        let r = paired_margin_test(&worse, 0.01, 4000, 0.05, 1).unwrap();
        assert!(!r.non_inferior(Better::Lower), "{r:?}");
        assert!(r.non_inferior(Better::Higher), "{r:?}");
    }

    /// Non-inferiority is still a claim the data must support: a wide scatter
    /// whose interval reaches past the unfavourable margin does not make it.
    #[test]
    fn a_noisy_sample_is_not_non_inferior_either_way() {
        let noisy: Vec<f64> = DIFFS.iter().map(|d| d * 40.0).collect();
        let r = paired_margin_test(&noisy, 0.05, 4000, 0.05, 1).unwrap();
        assert!(!r.non_inferior(Better::Lower) && !r.non_inferior(Better::Higher), "{r:?}");
    }

    #[test]
    fn refuses_an_unset_or_nonsensical_margin() {
        for delta in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(paired_margin_test(&DIFFS, delta, 100, 0.05, 1).is_err());
        }
        assert!(paired_margin_test(&DIFFS[..1], 0.1, 100, 0.05, 1).is_err());
        assert!(paired_margin_test(&DIFFS, 0.1, 100, 0.5, 1).is_err());
        assert!(paired_margin_test(&DIFFS, 0.1, 0, 0.05, 1).is_err());
    }
}
