//! Multi-series circular block bootstrap confidence intervals.
//!
//! [`crate::stats::bootstrap::bootstrap_ci`] resamples one exchangeable
//! sample. A statistic of several *time series* — the ratio of two runs'
//! typical iteration time, the geometric mean of such ratios over a sweep —
//! needs two things that function does not give:
//!
//! * **Serial dependence is kept.** Consecutive iterations of a run are not
//!   independent (a cache that is warm stays warm; a throttled clock stays
//!   throttled). Resampling single observations destroys that dependence and
//!   reports an interval that is too narrow. Resampling *blocks* of
//!   consecutive observations carries the short-range dependence into every
//!   resample. Blocks are circular — a block that runs off the end wraps to
//!   the start — so every observation is equally likely to be drawn.
//! * **Each series is resampled on its own.** The series are separate runs;
//!   a resample of the whole comparison is one independent resample of each.
//!
//! The block length is `ceil(n^(1/3))` per series, the rate at which the
//! block bootstrap's variance estimate is consistent for a stationary
//! series. At `n = 1` that is a block of one, which is the ordinary
//! bootstrap.
//!
//! # The statistic must be a smooth functional
//!
//! The percentile interval is only valid for a statistic whose bootstrap
//! distribution tracks its sampling distribution: a mean, a median, a ratio
//! or geometric mean of those. It is **not** valid for an extreme order
//! statistic. A resample's minimum equals the sample's minimum with
//! probability `1 - (1 - 1/n)^n -> 1 - 1/e`, and can never fall below it, so
//! the bootstrap distribution of a minimum is a spike at the observed value
//! with all remaining mass above it; its percentile interval's lower bound
//! is the observed minimum itself, whatever the true floor is. A caller who
//! reports a minimum as a point estimate takes the interval from a median.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{NumericsError, Result};
use crate::stats::types::Interval;

/// `ceil(n^(1/3))`, at least 1.
pub fn block_length(n: usize) -> usize {
    ((n as f64).cbrt().ceil() as usize).max(1)
}

/// Percentile interval at level `1 - alpha` for `statistic_fn` over
/// independently block-resampled `series`.
///
/// - `series` — one slice per run, each in its measured (time) order;
///   non-empty, and every slice non-empty and finite.
/// - `statistic_fn` — receives one resample per input series, in the same
///   order, each the same length as its original. See the module doc for
///   which statistics the interval is valid for.
/// - `iterations` — number of resamples; ≥ 1.
/// - `alpha` — two-tailed level in `(0, 1)`.
/// - `seed` — RNG seed; the interval is a deterministic function of
///   `(series, iterations, alpha, seed)`.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] for an empty `series`, an empty or
/// non-finite member, `iterations == 0`, `alpha` outside `(0, 1)`, or a
/// `statistic_fn` that returns `NaN` on a resample of finite data.
pub fn block_bootstrap_ci<F>(
    series: &[&[f64]],
    statistic_fn: F,
    iterations: usize,
    alpha: f64,
    seed: u64,
) -> Result<Interval>
where
    F: Fn(&[Vec<f64>]) -> f64,
{
    if series.is_empty() || series.iter().any(|s| s.is_empty()) {
        return Err(NumericsError::InvalidInput(
            "block bootstrap requires at least one series, each non-empty".into(),
        ));
    }
    if series.iter().any(|s| s.iter().any(|x| !x.is_finite())) {
        return Err(NumericsError::InvalidInput(
            "block bootstrap requires finite observations".into(),
        ));
    }
    if iterations == 0 {
        return Err(NumericsError::InvalidInput(
            "block bootstrap requires at least 1 iteration".into(),
        ));
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(NumericsError::InvalidInput(format!(
            "alpha out of range (must be in (0, 1)): {alpha}"
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut resamples: Vec<Vec<f64>> = series.iter().map(|s| vec![0.0; s.len()]).collect();
    let mut stats = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        for (original, resample) in series.iter().zip(resamples.iter_mut()) {
            let n = original.len();
            for block in resample.chunks_mut(block_length(n)) {
                let start = rng.gen_range(0..n);
                for (offset, slot) in block.iter_mut().enumerate() {
                    *slot = original[(start + offset) % n];
                }
            }
        }
        stats.push(statistic_fn(&resamples));
    }
    if stats.iter().any(|x| x.is_nan()) {
        return Err(NumericsError::InvalidInput(
            "block bootstrap statistic_fn produced a NaN value from finite resampled inputs".into(),
        ));
    }
    stats.sort_by(f64::total_cmp);
    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = (((1.0 - alpha / 2.0) * iterations as f64).ceil() as usize).saturating_sub(1);
    Ok(Interval {
        lower: stats[lower_idx.min(iterations - 1)],
        upper: stats[upper_idx.min(iterations - 1)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::summary::{median_of_finite, minimum};

    fn noisy(level: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| level * (1.0 + rng.gen_range(0.0..0.04)))
            .collect()
    }

    fn median_ratio(resamples: &[Vec<f64>]) -> f64 {
        median_of_finite(&resamples[0]) / median_of_finite(&resamples[1])
    }

    #[test]
    fn block_length_is_the_cube_root_rounded_up() {
        assert_eq!(block_length(1), 1);
        assert_eq!(block_length(8), 2);
        assert_eq!(block_length(9), 3);
        assert_eq!(block_length(1000), 10);
    }

    #[test]
    fn the_interval_of_a_median_ratio_covers_the_true_ratio() {
        let (a, b) = (noisy(1.30, 300, 1), noisy(1.00, 300, 2));
        let ci = block_bootstrap_ci(&[&a, &b], median_ratio, 2000, 0.05, 7).unwrap();
        assert!(ci.lower < 1.30 && 1.30 < ci.upper, "{ci:?}");
        assert!(ci.upper - ci.lower < 0.02, "{ci:?}");
    }

    #[test]
    fn same_inputs_and_seed_give_the_same_interval() {
        let (a, b) = (noisy(2.0, 64, 3), noisy(1.0, 64, 4));
        let x = block_bootstrap_ci(&[&a, &b], median_ratio, 500, 0.05, 9).unwrap();
        let y = block_bootstrap_ci(&[&a, &b], median_ratio, 500, 0.05, 9).unwrap();
        assert_eq!((x.lower, x.upper), (y.lower, y.upper));
    }

    /// The degeneracy the module doc names, observed: the percentile
    /// interval of a minimum has the sample minimum as its lower bound.
    #[test]
    fn the_interval_of_a_minimum_cannot_reach_below_the_observed_minimum() {
        let a = noisy(1.0, 200, 5);
        let observed = minimum(&a).unwrap();
        let ci = block_bootstrap_ci(
            &[&a],
            |r| r[0].iter().copied().fold(f64::INFINITY, f64::min),
            2000,
            0.05,
            1,
        )
        .unwrap();
        assert_eq!(ci.lower, observed);
    }

    /// Blocks keep serial dependence: on a slowly wandering series the
    /// block interval of the mean is wider than a block-of-one interval.
    #[test]
    fn blocks_widen_the_interval_of_an_autocorrelated_series() {
        let mut rng = StdRng::seed_from_u64(21);
        let mut level = 0.0_f64;
        let series: Vec<f64> = (0..1000)
            .map(|_| {
                level = 0.95 * level + rng.gen_range(-1.0..1.0);
                level
            })
            .collect();
        let mean_of = |r: &[Vec<f64>]| r[0].iter().sum::<f64>() / r[0].len() as f64;
        let blocked = block_bootstrap_ci(&[&series], mean_of, 2000, 0.05, 3).unwrap();
        // The same data split into single-observation series has block length
        // 1 everywhere: the dependence-free bootstrap of the same mean.
        let mut shuffled = series.clone();
        shuffled.sort_by(f64::total_cmp);
        let iid = crate::stats::bootstrap_ci(
            &shuffled,
            |s| s.iter().sum::<f64>() / s.len() as f64,
            2000,
            0.05,
            3,
        )
        .unwrap();
        assert!(
            blocked.upper - blocked.lower > 1.5 * (iid.upper - iid.lower),
            "blocked {blocked:?} iid {iid:?}"
        );
    }

    #[test]
    fn refuses_degenerate_input() {
        let a = [1.0, 2.0];
        let stat = |r: &[Vec<f64>]| r[0][0];
        assert!(block_bootstrap_ci(&[], stat, 10, 0.05, 1).is_err());
        assert!(block_bootstrap_ci(&[&a, &[]], stat, 10, 0.05, 1).is_err());
        assert!(block_bootstrap_ci(&[&[f64::NAN]], stat, 10, 0.05, 1).is_err());
        assert!(block_bootstrap_ci(&[&a], stat, 0, 0.05, 1).is_err());
        assert!(block_bootstrap_ci(&[&a], stat, 10, 1.0, 1).is_err());
        assert!(block_bootstrap_ci(&[&a], |_| f64::NAN, 10, 0.05, 1).is_err());
    }
}
