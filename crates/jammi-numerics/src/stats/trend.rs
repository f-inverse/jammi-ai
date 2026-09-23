//! Mann-Kendall monotonic-trend test with the Theil-Sen slope.
//!
//! Asks of an ordered series: do later observations tend to be larger (or
//! smaller) than earlier ones? Kendall's `S` counts concordant minus
//! discordant pairs over the time order; under the no-trend null it is
//! approximately normal with a variance that depends only on `n` and the tie
//! structure. The test is rank-based, so a handful of one-sided spikes — the
//! ordinary shape of timing noise — moves it far less than it would move a
//! least-squares slope.
//!
//! The Theil-Sen slope (the median of all pairwise slopes) is reported beside
//! the test because significance alone is not a finding: over a long enough
//! series an arbitrarily small drift becomes significant. A caller deciding
//! whether a series is stationary needs both halves — *is there* a trend
//! (the p-value) and *how large* is it (the slope against the series' own
//! level).
//!
//! The null assumes independent observations. Positive autocorrelation, which
//! timing series usually carry, inflates the false-positive rate of the
//! p-value; the slope's magnitude is unaffected, which is the second reason a
//! caller should require both.

use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{NumericsError, Result};
use crate::stats::summary::median_of_finite;
use crate::stats::types::TrendResult;

/// The smallest series the normal approximation of `S` is used for.
pub const MIN_TREND_SAMPLES: usize = 8;

/// Mann-Kendall test over `series` in its given (time) order.
///
/// Time and memory are `O(n^2)` in the series length: every pair is visited
/// once for `S` and its slope is held for the Theil-Sen median.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] when `series` has fewer than
/// [`MIN_TREND_SAMPLES`] observations or contains a non-finite value.
pub fn mann_kendall(series: &[f64]) -> Result<TrendResult> {
    let n = series.len();
    if n < MIN_TREND_SAMPLES {
        return Err(NumericsError::InvalidInput(format!(
            "Mann-Kendall requires at least {MIN_TREND_SAMPLES} observations, got {n}"
        )));
    }
    if series.iter().any(|x| !x.is_finite()) {
        return Err(NumericsError::InvalidInput(
            "Mann-Kendall requires finite observations".into(),
        ));
    }

    let mut s: i64 = 0;
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = series[j] - series[i];
            s += match diff.partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => 1,
                Some(std::cmp::Ordering::Less) => -1,
                _ => 0,
            };
            slopes.push(diff / (j - i) as f64);
        }
    }

    let variance =
        (pair_term(n) - tie_groups(series).into_iter().map(pair_term).sum::<f64>()) / 18.0;
    let z = if variance <= 0.0 || s == 0 {
        0.0
    } else {
        // Continuity correction: `S` moves in steps of 2 on tie-free data, so
        // the normal approximation is evaluated one unit toward zero.
        (s as f64 - (s.signum() as f64)) / variance.sqrt()
    };
    let standard_normal = Normal::new(0.0, 1.0)
        .map_err(|e| NumericsError::InvalidInput(format!("standard normal: {e}")))?;
    let p_value = (2.0 * (1.0 - standard_normal.cdf(z.abs()))).clamp(0.0, 1.0);

    Ok(TrendResult {
        s,
        z,
        p_value,
        sen_slope: median_of_finite(&slopes),
    })
}

/// `t (t - 1) (2t + 5)`: the variance term for a group of `t` equal values
/// (and, at `t = n`, for the whole series).
fn pair_term(t: usize) -> f64 {
    let t = t as f64;
    t * (t - 1.0) * (2.0 * t + 5.0)
}

/// Sizes of the groups of equal values, singletons included (a singleton's
/// [`pair_term`] is zero).
fn tie_groups(series: &[f64]) -> Vec<usize> {
    let mut sorted = series.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.chunk_by(|a, b| a == b).map(<[f64]>::len).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn a_strictly_increasing_series_has_maximal_s_and_the_exact_slope() {
        let series: Vec<f64> = (0..20).map(|i| 1.0 + 0.5 * i as f64).collect();
        let r = mann_kendall(&series).unwrap();
        assert_eq!(r.s, 20 * 19 / 2);
        assert_relative_eq!(r.sen_slope, 0.5, epsilon = 1e-12);
        assert!(r.p_value < 1e-6);
    }

    #[test]
    fn reversing_the_series_flips_the_sign_and_keeps_the_p_value() {
        let series: Vec<f64> = (0..30)
            .map(|i| ((i * 7919) % 31) as f64 + 0.1 * i as f64)
            .collect();
        let reversed: Vec<f64> = series.iter().rev().copied().collect();
        let a = mann_kendall(&series).unwrap();
        let b = mann_kendall(&reversed).unwrap();
        assert_eq!(a.s, -b.s);
        assert_relative_eq!(a.p_value, b.p_value, epsilon = 1e-12);
        assert_relative_eq!(a.sen_slope, -b.sen_slope, epsilon = 1e-12);
    }

    /// Pinned against the textbook worked value: for n = 10 tie-free
    /// observations `Var(S) = 10 * 9 * 25 / 18 = 125`.
    #[test]
    fn tie_free_variance_matches_the_closed_form() {
        let series = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5];
        let r = mann_kendall(&series).unwrap();
        let expected_z = (r.s as f64 - r.s.signum() as f64) / 125.0_f64.sqrt();
        assert_relative_eq!(r.z, expected_z, epsilon = 1e-12);
    }

    /// Calibration under the null: over many independent trend-free series
    /// the test rejects at `0.05` about one time in twenty, and the slope it
    /// reports stays negligible against the series' level every time.
    #[test]
    fn trend_free_noise_is_rejected_at_about_the_nominal_rate() {
        let rejections = (0..200_u64)
            .filter(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let series: Vec<f64> = (0..100).map(|_| 1.0 + rng.gen_range(0.0..0.05)).collect();
                let r = mann_kendall(&series).unwrap();
                assert!(
                    r.sen_slope.abs() * 100.0 < 0.02,
                    "drift {}",
                    r.sen_slope * 100.0
                );
                r.p_value < 0.05
            })
            .count();
        assert!(
            (2..=20).contains(&rejections),
            "{rejections} of 200 rejected"
        );
    }

    #[test]
    fn one_sided_spikes_do_not_read_as_a_trend() {
        let mut series = vec![1.0; 100];
        for i in [7, 31, 58, 90] {
            series[i] = 25.0;
        }
        let r = mann_kendall(&series).unwrap();
        assert!(r.p_value > 0.5, "p = {}", r.p_value);
        assert_eq!(r.sen_slope, 0.0);
    }

    #[test]
    fn a_constant_series_is_no_evidence_either_way() {
        let r = mann_kendall(&[2.0; 12]).unwrap();
        assert_eq!((r.s, r.z, r.p_value, r.sen_slope), (0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn refuses_short_and_non_finite_series() {
        assert!(mann_kendall(&[1.0; MIN_TREND_SAMPLES - 1]).is_err());
        let mut series = vec![1.0; 10];
        series[3] = f64::NAN;
        assert!(mann_kendall(&series).is_err());
    }
}
