//! Calibration and proper-scoring metrics for probabilistic predictions.
//!
//! Every function here is a pure, deterministic primitive over plain slices:
//! no model, no I/O, no randomness, no policy. They evaluate how well a set
//! of predictions matches observed outcomes — coverage of prediction sets and
//! intervals, expected calibration error, the continuous ranked probability
//! score (CRPS), negative log-likelihood (NLL), sharpness, and the
//! probability integral transform (PIT).
//!
//! Two scoring families are offered for the same quantity. The `_gaussian`
//! variants take a parametric predictive mean and standard deviation and use
//! the closed-form Gaussian expression; the `_sample` / sample variants take
//! an empirical ensemble of predictive draws and use the empirical CDF. A
//! caller picks the family that matches the shape of its predictions.

use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::error::{NumericsError, Result};

/// Reciprocal of the square root of pi, the constant term in the closed-form
/// Gaussian CRPS.
const INV_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI / 2.0;

/// Fraction of outcomes that fall inside their prediction set.
///
/// `hits[i]` is `true` when the `i`-th prediction set contained the observed
/// label. The result is the empirical coverage in `[0, 1]`. For a calibrated
/// set predictor built at miscoverage level `alpha`, this approaches
/// `1 - alpha`.
///
/// Returns `InvalidInput` when `hits` is empty (coverage is undefined over no
/// trials).
pub fn coverage(hits: &[bool]) -> Result<f64> {
    if hits.is_empty() {
        return Err(NumericsError::InvalidInput(
            "coverage requires at least one prediction".into(),
        ));
    }
    let covered = hits.iter().filter(|&&h| h).count();
    Ok(covered as f64 / hits.len() as f64)
}

/// Mean cardinality of the prediction sets.
///
/// `sizes[i]` is the number of labels in the `i`-th prediction set. Together
/// with [`coverage`] this is the efficiency side of the coverage/efficiency
/// trade-off: a predictor can always cover by predicting every label, so a
/// small mean set size at a target coverage is the goal.
///
/// Returns `InvalidInput` when `sizes` is empty.
pub fn mean_set_size(sizes: &[usize]) -> Result<f64> {
    if sizes.is_empty() {
        return Err(NumericsError::InvalidInput(
            "mean_set_size requires at least one prediction".into(),
        ));
    }
    let total: usize = sizes.iter().sum();
    Ok(total as f64 / sizes.len() as f64)
}

/// Expected calibration error over equal-width confidence bins.
///
/// `confidences[i]` is the predicted probability of correctness for prediction
/// `i` (the model's confidence in its top label), and `correct[i]` is whether
/// that prediction was correct. Predictions are partitioned into `n_bins`
/// equal-width bins over `[0, 1]` by confidence; within each bin the absolute
/// gap between mean confidence and observed accuracy is taken, and the bins
/// are averaged weighted by their occupancy.
///
/// A perfectly calibrated predictor scores `0.0`.
///
/// Returns `DimensionMismatch` when the slices differ in length, and
/// `InvalidInput` when they are empty, `n_bins` is zero, or any confidence
/// lies outside `[0, 1]`.
pub fn expected_calibration_error(
    confidences: &[f64],
    correct: &[bool],
    n_bins: usize,
) -> Result<f64> {
    if confidences.len() != correct.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: confidences.len(),
            got: correct.len(),
        });
    }
    if confidences.is_empty() {
        return Err(NumericsError::InvalidInput(
            "expected_calibration_error requires at least one prediction".into(),
        ));
    }
    if n_bins == 0 {
        return Err(NumericsError::InvalidInput(
            "expected_calibration_error requires at least one bin".into(),
        ));
    }
    if confidences.iter().any(|&c| !(0.0..=1.0).contains(&c)) {
        return Err(NumericsError::InvalidInput(
            "confidences must lie in [0, 1]".into(),
        ));
    }

    let mut bin_conf_sum = vec![0.0_f64; n_bins];
    let mut bin_correct = vec![0_usize; n_bins];
    let mut bin_count = vec![0_usize; n_bins];

    for (&c, &ok) in confidences.iter().zip(correct.iter()) {
        // Confidence 1.0 lands in the last bin rather than overflowing.
        let idx = ((c * n_bins as f64) as usize).min(n_bins - 1);
        bin_conf_sum[idx] += c;
        bin_correct[idx] += usize::from(ok);
        bin_count[idx] += 1;
    }

    let n = confidences.len() as f64;
    let mut ece = 0.0;
    for b in 0..n_bins {
        if bin_count[b] == 0 {
            continue;
        }
        let count = bin_count[b] as f64;
        let mean_conf = bin_conf_sum[b] / count;
        let accuracy = bin_correct[b] as f64 / count;
        ece += (count / n) * (mean_conf - accuracy).abs();
    }
    Ok(ece)
}

/// Empirical coverage of prediction intervals.
///
/// `lower[i]` and `upper[i]` bound the `i`-th interval and `observed[i]` is the
/// realised outcome. An interval counts as covering when
/// `lower[i] <= observed[i] <= upper[i]`. The result is the fraction of
/// intervals that cover, in `[0, 1]`; for an interval predictor built at level
/// `1 - alpha` it approaches `1 - alpha`.
///
/// Returns `DimensionMismatch` when the slices differ in length and
/// `InvalidInput` when they are empty.
pub fn interval_coverage(lower: &[f64], upper: &[f64], observed: &[f64]) -> Result<f64> {
    if lower.len() != upper.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: lower.len(),
            got: upper.len(),
        });
    }
    if lower.len() != observed.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: lower.len(),
            got: observed.len(),
        });
    }
    if lower.is_empty() {
        return Err(NumericsError::InvalidInput(
            "interval_coverage requires at least one interval".into(),
        ));
    }
    let covered = lower
        .iter()
        .zip(upper.iter())
        .zip(observed.iter())
        .filter(|((&lo, &hi), &y)| lo <= y && y <= hi)
        .count();
    Ok(covered as f64 / lower.len() as f64)
}

/// Mean width of prediction intervals — the standard sharpness measure.
///
/// `lower[i]` and `upper[i]` bound the `i`-th interval. Sharpness is the mean
/// of `upper[i] - lower[i]`: lower is sharper. Read alongside
/// [`interval_coverage`], since an interval can always be made sharper at the
/// cost of coverage.
///
/// Returns `DimensionMismatch` when the slices differ in length,
/// `InvalidInput` when they are empty, and when any interval is inverted
/// (`upper < lower`).
pub fn sharpness(lower: &[f64], upper: &[f64]) -> Result<f64> {
    if lower.len() != upper.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: lower.len(),
            got: upper.len(),
        });
    }
    if lower.is_empty() {
        return Err(NumericsError::InvalidInput(
            "sharpness requires at least one interval".into(),
        ));
    }
    let mut total = 0.0;
    for (&lo, &hi) in lower.iter().zip(upper.iter()) {
        if hi < lo {
            return Err(NumericsError::InvalidInput(
                "interval upper bound is below its lower bound".into(),
            ));
        }
        total += hi - lo;
    }
    Ok(total / lower.len() as f64)
}

/// Closed-form continuous ranked probability score for a Gaussian forecast.
///
/// For an observation `y` under a predictive `Normal(mean, sd)`, the CRPS has
/// the closed form
///
/// ```text
/// CRPS = sd * ( z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ),   z = (y - mean)/sd
/// ```
///
/// where `Phi` and `phi` are the standard-normal CDF and PDF. CRPS is a proper
/// score in the units of `y`; lower is better, and it reduces to the absolute
/// error as `sd -> 0`.
///
/// Returns `InvalidInput` when `sd` is non-positive or any argument is
/// non-finite.
pub fn crps_gaussian(y: f64, mean: f64, sd: f64) -> Result<f64> {
    if !y.is_finite() || !mean.is_finite() || !sd.is_finite() {
        return Err(NumericsError::InvalidInput(
            "crps_gaussian requires finite arguments".into(),
        ));
    }
    if sd <= 0.0 {
        return Err(NumericsError::InvalidInput(
            "crps_gaussian requires a positive standard deviation".into(),
        ));
    }
    let z = (y - mean) / sd;
    let n = Normal::standard();
    let cdf = n.cdf(z);
    let pdf = n.pdf(z);
    Ok(sd * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - INV_SQRT_PI))
}

/// Empirical continuous ranked probability score from an ensemble of draws.
///
/// Given predictive draws `forecast` and an observation `y`, the CRPS is
/// estimated from the empirical CDF as
///
/// ```text
/// CRPS = mean_i |x_i - y| - 1/2 * mean_{i,j} |x_i - x_j|
/// ```
///
/// The second term is the mean absolute pairwise spread of the draws and is
/// computed in `O(m log m)` from the sorted draws rather than the naive
/// `O(m^2)`: for sorted `x` (0-based `i`),
/// `sum_{i,j} |x_i - x_j| = 2 sum_i (2 i - (m - 1)) x_i`.
///
/// Returns `InvalidInput` when `forecast` is empty or any draw or `y` is
/// non-finite.
pub fn crps_sample(forecast: &[f64], y: f64) -> Result<f64> {
    if forecast.is_empty() {
        return Err(NumericsError::InvalidInput(
            "crps_sample requires at least one draw".into(),
        ));
    }
    if !y.is_finite() {
        return Err(NumericsError::InvalidInput(
            "crps_sample requires a finite observation".into(),
        ));
    }
    if forecast.iter().any(|x| !x.is_finite()) {
        return Err(NumericsError::InvalidInput(
            "crps_sample requires finite draws".into(),
        ));
    }
    let m = forecast.len();
    let mean_abs_err = forecast.iter().map(|&x| (x - y).abs()).sum::<f64>() / m as f64;

    let mut sorted = forecast.to_vec();
    // Draws are finite (checked above), so `total_cmp` orders them exactly.
    sorted.sort_by(f64::total_cmp);
    let mut weighted = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        // Coefficient (2 i - (m - 1)) over a signed-integer domain.
        let coeff = 2 * i as isize - (m as isize - 1);
        weighted += coeff as f64 * x;
    }
    // sum_{i,j} |x_i - x_j| = 2 * weighted, so its mean over m^2 ordered pairs
    // is 2 * weighted / m^2; the spread term is half that mean.
    let spread = weighted / (m * m) as f64;
    Ok(mean_abs_err - spread)
}

/// Negative log-likelihood of an observation under a Gaussian forecast.
///
/// For `y` under `Normal(mean, sd)`,
///
/// ```text
/// NLL = 1/2 ln(2 pi sd^2) + (y - mean)^2 / (2 sd^2).
/// ```
///
/// NLL is a proper local score in nats; lower is better.
///
/// Returns `InvalidInput` when `sd` is non-positive or any argument is
/// non-finite.
pub fn gaussian_nll(y: f64, mean: f64, sd: f64) -> Result<f64> {
    if !y.is_finite() || !mean.is_finite() || !sd.is_finite() {
        return Err(NumericsError::InvalidInput(
            "gaussian_nll requires finite arguments".into(),
        ));
    }
    if sd <= 0.0 {
        return Err(NumericsError::InvalidInput(
            "gaussian_nll requires a positive standard deviation".into(),
        ));
    }
    let var = sd * sd;
    let resid = y - mean;
    Ok(0.5 * (2.0 * std::f64::consts::PI * var).ln() + resid * resid / (2.0 * var))
}

/// Negative log-likelihood of an observation under a Gaussian fit to draws.
///
/// The ensemble `forecast` is summarised by its sample mean and sample
/// standard deviation (Bessel-corrected, dividing by `m - 1`), and the
/// observation is scored against that Gaussian via [`gaussian_nll`].
///
/// Returns `InvalidInput` when `forecast` has fewer than two draws (sample
/// standard deviation is undefined), the draws are all equal (zero spread), or
/// `y` is non-finite.
pub fn sample_nll(forecast: &[f64], y: f64) -> Result<f64> {
    if forecast.len() < 2 {
        return Err(NumericsError::InvalidInput(
            "sample_nll requires at least two draws".into(),
        ));
    }
    if !y.is_finite() {
        return Err(NumericsError::InvalidInput(
            "sample_nll requires a finite observation".into(),
        ));
    }
    let m = forecast.len() as f64;
    let mean = forecast.iter().sum::<f64>() / m;
    let var = forecast.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (m - 1.0);
    if var <= 0.0 {
        return Err(NumericsError::InvalidInput(
            "sample_nll requires draws with non-zero spread".into(),
        ));
    }
    gaussian_nll(y, mean, var.sqrt())
}

/// Probability integral transform of observations under Gaussian forecasts.
///
/// `means[i]` and `sds[i]` parameterise the `i`-th predictive `Normal`, and
/// `observed[i]` is the realised outcome. The PIT value is the predictive CDF
/// evaluated at the observation, `Phi((observed - mean)/sd)`. When the
/// forecasts are calibrated the returned values are uniform on `[0, 1]`;
/// systematic departures from uniformity diagnose miscalibration.
///
/// Returns `DimensionMismatch` when the slices differ in length,
/// `InvalidInput` when they are empty or any `sd` is non-positive.
pub fn pit_values(means: &[f64], sds: &[f64], observed: &[f64]) -> Result<Vec<f64>> {
    if means.len() != sds.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: means.len(),
            got: sds.len(),
        });
    }
    if means.len() != observed.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: means.len(),
            got: observed.len(),
        });
    }
    if means.is_empty() {
        return Err(NumericsError::InvalidInput(
            "pit_values requires at least one forecast".into(),
        ));
    }
    let n = Normal::standard();
    let mut out = Vec::with_capacity(means.len());
    for ((&mean, &sd), &y) in means.iter().zip(sds.iter()).zip(observed.iter()) {
        if sd <= 0.0 {
            return Err(NumericsError::InvalidInput(
                "pit_values requires positive standard deviations".into(),
            ));
        }
        out.push(n.cdf((y - mean) / sd));
    }
    Ok(out)
}

/// Adaptive (equal-mass), debiased calibration error of a probabilistic
/// regression forecast from its PIT values — the regression analogue of the
/// classification reliability gap.
///
/// Under calibration the probability-integral-transform values are uniform on
/// `[0, 1]`: the empirical CDF of the PIT lies on the diagonal. This sorts the
/// PIT values and partitions them into `n_bins` bins of (nearly) equal
/// occupancy — adaptive binning, which adapts to where the PIT mass lies and
/// avoids the bin-count sensitivity of fixed-width histograms ([Kumar et al.
/// 2019]). Within each bin the reliability gap is
///
/// ```text
/// | mean_PIT_in_bin - empirical_uniform_position_of_bin |,
/// ```
///
/// where the empirical uniform position is the bin's central rank fraction
/// `(start + end) / (2 n)` — the cumulative level a calibrated (uniform) PIT
/// would sit at. Each gap is debiased by subtracting the finite-sample expected
/// deviation `sqrt(p (1 - p) / m)` of a calibrated bin (clamped at zero), so a
/// genuinely calibrated forecast is not penalised for sampling noise. Bins are
/// averaged weighted by occupancy. A calibrated forecast scores `≈ 0.0`; a
/// systematic over- or under-dispersion bows the PIT off the diagonal and the
/// gap grows.
///
/// Returns `InvalidInput` when `pit` is empty, `n_bins` is zero, or any PIT
/// value lies outside `[0, 1]`.
pub fn pit_calibration_error(pit: &[f64], n_bins: usize) -> Result<f64> {
    if pit.is_empty() {
        return Err(NumericsError::InvalidInput(
            "pit_calibration_error requires at least one PIT value".into(),
        ));
    }
    if n_bins == 0 {
        return Err(NumericsError::InvalidInput(
            "pit_calibration_error requires at least one bin".into(),
        ));
    }
    if pit.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err(NumericsError::InvalidInput(
            "PIT values must lie in [0, 1]".into(),
        ));
    }
    let mut sorted = pit.to_vec();
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len();
    let n_f = n as f64;
    // Cap bins at the sample size so no bin is empty.
    let effective_bins = n_bins.min(n);
    let mut ece = 0.0;
    for bin in 0..effective_bins {
        let start = bin * n / effective_bins;
        let end = (bin + 1) * n / effective_bins;
        let slice = &sorted[start..end];
        let m = slice.len();
        if m == 0 {
            continue;
        }
        let m_f = m as f64;
        let mean_pit = slice.iter().sum::<f64>() / m_f;
        // The uniform position a calibrated PIT would occupy at this bin's
        // centre: the mean rank fraction over the bin's indices.
        let expected = (start as f64 + end as f64) / (2.0 * n_f);
        let gap = (mean_pit - expected).abs();
        // Finite-sample deviation of a calibrated bin's empirical position from
        // its mean PIT, subtracted to debias; never push a gap below zero.
        let bias = (mean_pit * (1.0 - mean_pit) / m_f).sqrt();
        ece += (m_f / n_f) * (gap - bias).max(0.0);
    }
    Ok(ece)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as NormalDraw};

    #[test]
    fn coverage_of_exchangeable_set_approaches_one_minus_alpha() {
        // A calibrated set predictor at alpha = 0.1: 9 of every 10 sets cover.
        let mut hits = Vec::new();
        for i in 0..1000 {
            hits.push(i % 10 != 0);
        }
        let cov = coverage(&hits).unwrap();
        assert_abs_diff_eq!(cov, 0.9, epsilon = 1e-12);
    }

    #[test]
    fn coverage_empty_is_error() {
        assert!(coverage(&[]).is_err());
    }

    #[test]
    fn mean_set_size_is_arithmetic_mean() {
        let sizes = [1, 2, 3, 4];
        assert_abs_diff_eq!(mean_set_size(&sizes).unwrap(), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn ece_is_zero_for_perfectly_calibrated() {
        // In every confidence band the accuracy equals the mean confidence:
        // the 0.0-bin is always wrong, the 1.0-bin always right, and the
        // 0.5-bin is right exactly half the time.
        let confidences = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
        let correct = [false, false, true, false, true, true];
        let ece = expected_calibration_error(&confidences, &correct, 10).unwrap();
        assert_abs_diff_eq!(ece, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ece_detects_overconfidence() {
        // Confidence 1.0 but only half correct: gap of 0.5 in one bin.
        let confidences = [1.0, 1.0, 1.0, 1.0];
        let correct = [true, false, true, false];
        let ece = expected_calibration_error(&confidences, &correct, 10).unwrap();
        assert_abs_diff_eq!(ece, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn ece_rejects_out_of_range_confidence() {
        assert!(expected_calibration_error(&[1.5], &[true], 10).is_err());
        assert!(expected_calibration_error(&[0.5], &[true], 0).is_err());
        assert!(expected_calibration_error(&[0.5, 0.5], &[true], 10).is_err());
    }

    #[test]
    fn interval_coverage_counts_inclusion() {
        let lower = [0.0, 0.0, 0.0];
        let upper = [1.0, 1.0, 1.0];
        let observed = [0.5, 2.0, 1.0]; // inside, outside, on boundary (counts)
        let cov = interval_coverage(&lower, &upper, &observed).unwrap();
        assert_abs_diff_eq!(cov, 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn sharpness_is_mean_width() {
        let lower = [0.0, 1.0];
        let upper = [2.0, 5.0];
        assert_abs_diff_eq!(sharpness(&lower, &upper).unwrap(), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn sharpness_rejects_inverted_interval() {
        assert!(sharpness(&[1.0], &[0.0]).is_err());
    }

    #[test]
    fn crps_gaussian_at_the_mean_matches_analytic() {
        // At y = mean, z = 0: CRPS = sd * (2 phi(0) - 1/sqrt(pi))
        //                            = sd * (2/sqrt(2 pi) - 1/sqrt(pi)).
        let sd = 2.0;
        let expected =
            sd * (2.0 / (2.0 * std::f64::consts::PI).sqrt() - 1.0 / std::f64::consts::PI.sqrt());
        let got = crps_gaussian(3.0, 3.0, sd).unwrap();
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }

    #[test]
    fn crps_gaussian_reduces_to_absolute_error_as_sd_shrinks() {
        // As sd -> 0 the predictive mass concentrates and CRPS -> |y - mean|.
        let got = crps_gaussian(5.0, 3.0, 1e-6).unwrap();
        assert_abs_diff_eq!(got, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn crps_gaussian_rejects_nonpositive_sd() {
        assert!(crps_gaussian(0.0, 0.0, 0.0).is_err());
        assert!(crps_gaussian(0.0, 0.0, -1.0).is_err());
    }

    #[test]
    fn crps_sample_rejects_non_finite_inputs() {
        assert!(crps_sample(&[], 0.0).is_err());
        assert!(crps_sample(&[1.0, 2.0], f64::NAN).is_err());
        assert!(crps_sample(&[1.0, f64::INFINITY], 0.0).is_err());
        assert!(crps_sample(&[1.0, f64::NAN], 0.0).is_err());
    }

    #[test]
    fn crps_sample_matches_two_point_closed_form() {
        // Two draws {a, b}, observation y. The empirical CRPS averages the
        // pairwise spread over all m^2 = 4 ordered pairs (including the two
        // zero-distance diagonal pairs), so the spread term is
        // (1/2) * (0 + |a-b| + |b-a| + 0) / 4 = |a-b| / 4.
        let forecast = [1.0, 3.0];
        let y = 0.0;
        let mae = (1.0 + 3.0) / 2.0;
        let spread = (3.0 - 1.0_f64).abs() / 4.0;
        let expected = mae - spread;
        assert_abs_diff_eq!(
            crps_sample(&forecast, y).unwrap(),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn crps_sample_approaches_gaussian_crps_for_large_ensemble() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = NormalDraw::new(0.0, 1.0).unwrap();
        let draws: Vec<f64> = (0..20_000).map(|_| dist.sample(&mut rng)).collect();
        let y = 0.5;
        let empirical = crps_sample(&draws, y).unwrap();
        let analytic = crps_gaussian(y, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(empirical, analytic, epsilon = 2e-2);
    }

    #[test]
    fn gaussian_nll_matches_closed_form() {
        // Standard normal at y = 0: NLL = 1/2 ln(2 pi).
        let got = gaussian_nll(0.0, 0.0, 1.0).unwrap();
        let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }

    #[test]
    fn gaussian_nll_adds_quadratic_residual() {
        // At one sd from the mean the residual term contributes 1/2.
        let base = gaussian_nll(0.0, 0.0, 1.0).unwrap();
        let offset = gaussian_nll(1.0, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(offset - base, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn sample_nll_fits_gaussian_to_draws() {
        // Draws {-1, 1}: mean 0, Bessel variance 2, sd = sqrt(2).
        let forecast = [-1.0, 1.0];
        let got = sample_nll(&forecast, 0.0).unwrap();
        let expected = gaussian_nll(0.0, 0.0, 2.0_f64.sqrt()).unwrap();
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }

    #[test]
    fn sample_nll_rejects_degenerate_inputs() {
        assert!(sample_nll(&[1.0], 0.0).is_err());
        assert!(sample_nll(&[2.0, 2.0], 0.0).is_err());
    }

    #[test]
    fn pit_values_are_uniform_for_calibrated_forecasts() {
        // Draw observations from the same Normal that defines each forecast;
        // the PIT values should be approximately uniform on [0, 1].
        let mut rng = StdRng::seed_from_u64(7);
        let dist = NormalDraw::new(2.0, 3.0).unwrap();
        let n = 20_000;
        let observed: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let means = vec![2.0; n];
        let sds = vec![3.0; n];
        let pits = pit_values(&means, &sds, &observed).unwrap();

        // Uniform mean is 1/2 and variance is 1/12.
        let mean = pits.iter().sum::<f64>() / n as f64;
        let var = pits.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / n as f64;
        assert_abs_diff_eq!(mean, 0.5, epsilon = 1e-2);
        assert_abs_diff_eq!(var, 1.0 / 12.0, epsilon = 1e-2);
    }

    #[test]
    fn pit_values_evaluate_the_predictive_cdf() {
        // Observation at the mean maps to the median of the PIT, 0.5.
        let pits = pit_values(&[0.0], &[1.0], &[0.0]).unwrap();
        assert_abs_diff_eq!(pits[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn pit_values_reject_nonpositive_sd() {
        assert!(pit_values(&[0.0], &[0.0], &[0.0]).is_err());
        assert!(pit_values(&[0.0], &[1.0, 1.0], &[0.0]).is_err());
    }

    #[test]
    fn pit_calibration_error_zero_for_uniform_pit() {
        // A perfectly uniform PIT grid lies exactly on the diagonal: zero gap.
        let n = 1000;
        let pit: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let err = pit_calibration_error(&pit, 10).unwrap();
        assert!(err < 1e-2, "uniform PIT should score near zero: {err}");
    }

    #[test]
    fn pit_calibration_error_detects_overdispersion() {
        // Severely overdispersed forecasts collapse the PIT toward the centre
        // (~0.5). The first equal-mass bins then carry mean PIT well above their
        // low expected rank position (and the last bins well below), so the
        // adaptive, debiased gap is large.
        let n = 1000;
        let pit: Vec<f64> = (0..n)
            .map(|i| 0.45 + 0.1 * (i as f64 + 0.5) / n as f64)
            .collect();
        let err = pit_calibration_error(&pit, 10).unwrap();
        assert!(err > 0.1, "clustered PIT should score a large gap: {err}");
    }

    #[test]
    fn pit_calibration_error_rejects_bad_inputs() {
        assert!(pit_calibration_error(&[], 10).is_err());
        assert!(pit_calibration_error(&[0.5], 0).is_err());
        assert!(pit_calibration_error(&[1.5], 10).is_err());
    }
}
