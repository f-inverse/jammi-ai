//! Location and spread summaries over a finite sample: mean, median,
//! minimum, population standard deviation, and the geometric mean.
//!
//! Every function refuses an empty or non-finite sample rather than
//! returning a sentinel: a summary of nothing, or of a `NaN`, is not a number
//! a caller can compare against anything.

use crate::error::{NumericsError, Result};

fn require_finite(name: &str, samples: &[f64]) -> Result<()> {
    if samples.is_empty() {
        return Err(NumericsError::InvalidInput(format!(
            "{name} requires a non-empty sample"
        )));
    }
    if samples.iter().any(|x| !x.is_finite()) {
        return Err(NumericsError::InvalidInput(format!(
            "{name} requires finite (non-NaN, non-infinite) samples"
        )));
    }
    Ok(())
}

/// Arithmetic mean, summed left to right.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] on an empty or non-finite sample.
pub fn mean(samples: &[f64]) -> Result<f64> {
    require_finite("mean", samples)?;
    Ok(samples.iter().sum::<f64>() / samples.len() as f64)
}

/// The middle order statistic; the midpoint of the two middle ones for an
/// even count.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] on an empty or non-finite sample.
pub fn median(samples: &[f64]) -> Result<f64> {
    require_finite("median", samples)?;
    Ok(median_of_finite(samples))
}

/// [`median`] without the input check, for a caller that has already
/// established the sample is non-empty and finite (a bootstrap statistic over
/// a resample of validated data).
pub(crate) fn median_of_finite(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// The smallest observation.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] on an empty or non-finite sample.
pub fn minimum(samples: &[f64]) -> Result<f64> {
    require_finite("minimum", samples)?;
    Ok(samples.iter().copied().fold(f64::INFINITY, f64::min))
}

/// Population standard deviation (divisor `n`, not `n - 1`): the spread of
/// exactly these observations, not an estimate for a population they were
/// drawn from.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] on an empty or non-finite sample.
pub fn population_std_dev(samples: &[f64]) -> Result<f64> {
    let m = mean(samples)?;
    let variance = samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / samples.len() as f64;
    Ok(variance.sqrt())
}

/// Geometric mean, `exp(mean(ln x))`.
///
/// The summary for a set of ratios: it is the only mean for which
/// `gm(a_i / b_i) == gm(a_i) / gm(b_i)`, so the summary of a set of speed
/// ratios does not depend on which side was chosen as the denominator.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] on an empty or non-finite sample, or on
/// any value `<= 0` (a ratio that is not positive has no logarithm).
pub fn geometric_mean(samples: &[f64]) -> Result<f64> {
    require_finite("geometric mean", samples)?;
    if samples.iter().any(|&x| x <= 0.0) {
        return Err(NumericsError::InvalidInput(
            "geometric mean requires strictly positive samples".into(),
        ));
    }
    Ok(geometric_mean_of_positive(samples))
}

/// [`geometric_mean`] without the input check; see [`median_of_finite`].
pub(crate) fn geometric_mean_of_positive(samples: &[f64]) -> f64 {
    (samples.iter().map(|x| x.ln()).sum::<f64>() / samples.len() as f64).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn median_of_odd_and_even_counts() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn minimum_and_mean() {
        assert_eq!(minimum(&[3.0, -1.0, 2.0]).unwrap(), -1.0);
        assert_relative_eq!(mean(&[1.0, 2.0, 6.0]).unwrap(), 3.0);
    }

    #[test]
    fn population_std_dev_uses_divisor_n() {
        // {2, 4, 4, 4, 5, 5, 7, 9}: mean 5, population variance 4.
        let s = population_std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
        assert_relative_eq!(s, 2.0);
    }

    #[test]
    fn geometric_mean_is_symmetric_in_the_choice_of_denominator() {
        let ratios = [2.0, 0.5, 4.0, 1.25];
        let inverted: Vec<f64> = ratios.iter().map(|r| 1.0 / r).collect();
        let forward = geometric_mean(&ratios).unwrap();
        let backward = geometric_mean(&inverted).unwrap();
        assert_relative_eq!(forward * backward, 1.0, epsilon = 1e-12);
        assert_relative_eq!(forward, (2.0_f64 * 0.5 * 4.0 * 1.25).powf(0.25));
    }

    #[test]
    fn every_summary_refuses_empty_and_non_finite() {
        for f in [mean, median, minimum, population_std_dev, geometric_mean] {
            assert!(f(&[]).is_err());
            assert!(f(&[1.0, f64::NAN]).is_err());
            assert!(f(&[1.0, f64::INFINITY]).is_err());
        }
    }

    #[test]
    fn geometric_mean_refuses_non_positive() {
        assert!(geometric_mean(&[1.0, 0.0]).is_err());
        assert!(geometric_mean(&[1.0, -2.0]).is_err());
    }
}
