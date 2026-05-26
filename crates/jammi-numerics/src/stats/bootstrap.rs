//! Percentile bootstrap confidence intervals.
//!
//! Resamples `samples` with replacement `iterations` times, applies the
//! caller's `statistic_fn` to each resample, and returns the
//! `alpha/2` and `1 - alpha/2` percentiles of the resulting sampling
//! distribution.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{NumericsError, Result};
use crate::stats::types::Interval;

/// Percentile bootstrap CI at level `1 - alpha`.
///
/// - `samples` — observed data; non-empty.
/// - `statistic_fn` — function computing the statistic from a resample.
/// - `iterations` — number of bootstrap resamples; ≥ 1.
/// - `alpha` — two-tailed significance level in `(0, 1)`. For a 95% CI,
///   pass `alpha = 0.05`.
/// - `seed` — RNG seed for reproducibility.
pub fn bootstrap_ci<F>(
    samples: &[f64],
    statistic_fn: F,
    iterations: usize,
    alpha: f64,
    seed: u64,
) -> Result<Interval>
where
    F: Fn(&[f64]) -> f64,
{
    if samples.is_empty() {
        return Err(NumericsError::InvalidInput(
            "bootstrap requires non-empty samples".into(),
        ));
    }
    if iterations == 0 {
        return Err(NumericsError::InvalidInput(
            "bootstrap requires at least 1 iteration".into(),
        ));
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(NumericsError::InvalidInput(format!(
            "alpha out of range (must be in (0, 1)): {alpha}"
        )));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let n = samples.len();
    let mut stats = Vec::with_capacity(iterations);
    let mut buf = vec![0.0_f64; n];
    for _ in 0..iterations {
        for slot in buf.iter_mut() {
            *slot = samples[rng.gen_range(0..n)];
        }
        stats.push(statistic_fn(&buf));
    }
    stats.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).ceil() as usize - 1;
    let lower_idx = lower_idx.min(stats.len() - 1);
    let upper_idx = upper_idx.min(stats.len() - 1);
    Ok(Interval {
        lower: stats[lower_idx],
        upper: stats[upper_idx],
    })
}
