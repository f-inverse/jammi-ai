//! Percentile bootstrap confidence intervals.
//!
//! Resamples `samples` with replacement `iterations` times, applies the
//! caller's `statistic_fn` to each resample, and returns the
//! `alpha/2` and `1 - alpha/2` percentiles of the resulting sampling
//! distribution.
//!
//! The resample is drawn positionally under a fixed [`StdRng`] seed, so without
//! a canonical basis the same multiset of `samples` in a different order would
//! select different values and yield a different interval. The bootstrap is a
//! property of the sample *multiset*, not its order, so the input is sorted
//! into a canonical order before resampling — see [`bootstrap_ci`].

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{NumericsError, Result};
use crate::stats::types::Interval;

/// Percentile bootstrap CI at level `1 - alpha`.
///
/// - `samples` — observed data; non-empty.
/// - `statistic_fn` — a function of the resample *multiset*; it must be
///   invariant to the order of its argument (e.g. a mean, a quantile). The
///   resample is drawn positionally under a fixed seed, so an order-sensitive
///   statistic would make the interval depend on the order in which `samples`
///   were appended — see the order-invariance note below.
/// - `iterations` — number of bootstrap resamples; ≥ 1.
/// - `alpha` — two-tailed significance level in `(0, 1)`. For a 95% CI,
///   pass `alpha = 0.05`.
/// - `seed` — RNG seed for reproducibility.
///
/// The interval is a deterministic function of the `samples` *multiset*, not
/// of their order: `samples` is sorted into a canonical order before the seeded
/// resampling, so a permutation of the same values yields a byte-identical
/// interval. This is the property the seeded resampler needs to be reproducible
/// — the seed alone fixes which *positions* are drawn, and only a canonical
/// basis fixes which *values* those positions hold.
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
    // Canonicalize the resample basis: the seeded RNG draws positions, so the
    // interval is only a function of the sample multiset once the values those
    // positions hold are in a fixed order.
    let mut basis = samples.to_vec();
    basis.sort_by(f64::total_cmp);
    let mut rng = StdRng::seed_from_u64(seed);
    let n = basis.len();
    let mut stats = Vec::with_capacity(iterations);
    let mut buf = vec![0.0_f64; n];
    for _ in 0..iterations {
        for slot in buf.iter_mut() {
            *slot = basis[rng.gen_range(0..n)];
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
