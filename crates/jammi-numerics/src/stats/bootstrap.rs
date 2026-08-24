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
///
/// # Errors
///
/// Returns [`NumericsError::InvalidInput`] if `samples` is empty, if
/// `iterations` is `0`, if `alpha` is outside `(0, 1)`, if `samples`
/// contains a non-finite (`NaN` or `±inf`) value, or if `statistic_fn`
/// produces a `NaN` on any resample. `±inf` IS accepted as a `statistic_fn`
/// output (but never as a `samples` input) — see the non-finite-handling
/// notes on the two edge checks in the implementation.
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
    if samples.iter().any(|x| !x.is_finite()) {
        // The canonicalizing `sort_by(f64::total_cmp)` below is a genuine
        // total order even over a NaN- or ±inf-containing slice (that is the
        // whole point of `total_cmp`), so a non-finite sample would NOT
        // corrupt the sort. The real hazard is downstream and probabilistic:
        // the seeded resampler draws POSITIONS, and `statistic_fn` is
        // caller-supplied arithmetic over whatever values land at those
        // positions. Whether a NaN sample's position gets drawn at all — and
        // whether a drawn combination of samples is even well-defined
        // arithmetically (e.g. `+inf` and `-inf` both drawn into a `mean`
        // sum to NaN even though neither individual sample is a NaN) —
        // depends on `n`, `iterations`, and the seed. At small `n` / few
        // iterations a resample can dodge the offending combination
        // entirely, in which case the output-edge NaN check below (which
        // only catches a NaN that DID get produced) never fires either, and
        // the returned interval looks like a clean, deterministic answer
        // while actually being contingent on RNG luck rather than a
        // property of the input multiset alone. Refusing every non-finite
        // sample here — not just NaN, and not only mixed-sign infinities —
        // makes that contingency impossible for ANY `statistic_fn` the
        // caller might supply, not just `mean`: a same-sign pair of
        // infinities is fine under `+` but still NaN under `-`, so the
        // blanket rule is the simplest one that doesn't require reasoning
        // about which arithmetic `statistic_fn` performs.
        return Err(NumericsError::InvalidInput(
            "bootstrap requires finite (non-NaN, non-infinite) samples".into(),
        ));
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
    if stats.iter().any(|x| x.is_nan()) {
        // Every element of `buf` fed into `statistic_fn` above is guaranteed
        // finite — the input-edge check refused any non-finite `samples`
        // element before any resample was drawn, so a NaN reaching this
        // point cannot be attributed to a non-finite input. It can only be
        // `statistic_fn`'s OWN degenerate arithmetic over finite inputs
        // (e.g. a `0.0 / 0.0` ratio on a degenerate resample), so this error
        // message is now accurate rather than a misattribution. Refuse for
        // the same lack-of-numeric-meaning reason the input-edge check above
        // refuses a non-finite sample: a NaN statistic doesn't represent a
        // real value in the sampling distribution, so a percentile landing
        // on it would silently hand back a NaN-valued interval bound. `±inf`
        // is NOT refused here: the only operation performed on `stats` past
        // this point is `total_cmp`-ordered percentile selection (no
        // arithmetic combination of its elements), which `±inf` participates
        // in correctly — an interval bound of `±inf` is a legitimate, if
        // extreme, answer for a heavy-tailed `statistic_fn` (e.g. a ratio
        // dividing by an exact-zero denominator on every resample).
        return Err(NumericsError::InvalidInput(
            "bootstrap statistic_fn produced a NaN value from finite resampled inputs".into(),
        ));
    }
    // Every element is non-NaN at this point (it may be `±inf`), so
    // `total_cmp`'s total order and `partial_cmp` agree here; `total_cmp` is
    // used regardless to keep the fold order pinned rather than depend on
    // that precondition.
    stats.sort_by(f64::total_cmp);
    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).ceil() as usize - 1;
    let lower_idx = lower_idx.min(stats.len() - 1);
    let upper_idx = upper_idx.min(stats.len() - 1);
    Ok(Interval {
        lower: stats[lower_idx],
        upper: stats[upper_idx],
    })
}
