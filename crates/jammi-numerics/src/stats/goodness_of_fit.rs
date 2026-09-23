//! Goodness of fit of observed counts to a fully specified multinomial law:
//! Pearson's chi-square and the likelihood-ratio G statistic.
//!
//! A *cell* is one multinomial experiment: `k` categories with known
//! probabilities and the count observed in each. Under the law, each
//! statistic is asymptotically chi-square with `k - 1` degrees of freedom —
//! exactly `k - 1`, because the probabilities are given, not fitted: no
//! parameter is estimated from the counts, so none is subtracted. Independent
//! cells add: the pooled statistic is the sum over cells and its degrees of
//! freedom the sum of theirs.
//!
//! The chi-square approximation is only as good as the expected counts are
//! large. Every category must expect at least [`MIN_EXPECTED_COUNT`]
//! observations; a cell that does not is refused rather than tested, because
//! a p-value from a sparse cell says nothing about the sampler it came from.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::error::{NumericsError, Result};

/// The smallest expected count a category may have (Cochran's rule).
pub const MIN_EXPECTED_COUNT: f64 = 5.0;

/// How far each category's probabilities may sum from 1.
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitStatistic {
    /// `sum (O - E)^2 / E`.
    PearsonChiSquare,
    /// `2 sum O ln(O / E)`, a zero count contributing zero.
    LikelihoodRatioG,
}

/// One multinomial experiment: counts observed and the law they are tested
/// against, category by category.
#[derive(Debug, Clone, Copy)]
pub struct FitCell<'a> {
    pub observed: &'a [u64],
    pub probabilities: &'a [f64],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GoodnessOfFit {
    pub statistic: f64,
    /// `sum over cells of (categories - 1)`.
    pub degrees_of_freedom: usize,
    /// Upper tail of the chi-square distribution at `statistic`.
    pub p_value: f64,
    /// Total observations over all cells.
    pub observations: u64,
}

/// Pooled goodness of fit of independent `cells` to their laws.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] when there are no cells; when a cell has
/// fewer than two categories, a probability outside `(0, 1]`, probabilities
/// that do not sum to 1, or no observations; or when a category's expected
/// count is below [`MIN_EXPECTED_COUNT`].
/// [`NumericsError::DimensionMismatch`] when a cell's counts and
/// probabilities differ in length.
pub fn goodness_of_fit(cells: &[FitCell<'_>], statistic: FitStatistic) -> Result<GoodnessOfFit> {
    if cells.is_empty() {
        return Err(NumericsError::InvalidInput(
            "goodness of fit requires at least one cell".into(),
        ));
    }
    let mut total = 0.0;
    let mut degrees_of_freedom = 0;
    let mut observations = 0_u64;
    for (index, cell) in cells.iter().enumerate() {
        let k = cell.probabilities.len();
        if cell.observed.len() != k {
            return Err(NumericsError::DimensionMismatch {
                expected: k,
                got: cell.observed.len(),
            });
        }
        let invalid =
            |reason: String| NumericsError::InvalidInput(format!("cell {index}: {reason}"));
        if k < 2 {
            return Err(invalid(format!(
                "{k} categor(y/ies); a law over fewer than 2 has nothing to fit"
            )));
        }
        if cell
            .probabilities
            .iter()
            .any(|p| !(p.is_finite() && *p > 0.0 && *p <= 1.0))
        {
            return Err(invalid("every probability must lie in (0, 1]".into()));
        }
        let sum: f64 = cell.probabilities.iter().sum();
        if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(invalid(format!("probabilities sum to {sum}, not 1")));
        }
        let n: u64 = cell.observed.iter().sum();
        if n == 0 {
            return Err(invalid("no observations".into()));
        }
        let least = cell
            .probabilities
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            * n as f64;
        if least < MIN_EXPECTED_COUNT {
            return Err(invalid(format!(
                "smallest expected count {least:.3} is below {MIN_EXPECTED_COUNT}: too sparse for the chi-square approximation"
            )));
        }
        total += cell
            .observed
            .iter()
            .zip(cell.probabilities)
            .map(|(&o, &p)| {
                let (o, e) = (o as f64, p * n as f64);
                match statistic {
                    FitStatistic::PearsonChiSquare => (o - e).powi(2) / e,
                    FitStatistic::LikelihoodRatioG if o == 0.0 => 0.0,
                    FitStatistic::LikelihoodRatioG => 2.0 * o * (o / e).ln(),
                }
            })
            .sum::<f64>();
        degrees_of_freedom += k - 1;
        observations += n;
    }
    // Rounding can leave G a hair below zero when the counts match the law
    // exactly; the statistic is non-negative by definition.
    let statistic = total.max(0.0);
    let chi_square = ChiSquared::new(degrees_of_freedom as f64)
        .map_err(|e| NumericsError::InvalidInput(format!("chi-square distribution: {e}")))?;
    Ok(GoodnessOfFit {
        statistic,
        degrees_of_freedom,
        p_value: (1.0 - chi_square.cdf(statistic)).clamp(0.0, 1.0),
        observations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const BOTH: [FitStatistic; 2] = [
        FitStatistic::PearsonChiSquare,
        FitStatistic::LikelihoodRatioG,
    ];

    fn cell<'a>(observed: &'a [u64], probabilities: &'a [f64]) -> FitCell<'a> {
        FitCell {
            observed,
            probabilities,
        }
    }

    /// Hand-computed: O = (30, 50, 20), E = (25, 50, 25):
    /// X^2 = 25/25 + 0 + 25/25 = 2; G = 2 (30 ln 1.2 + 20 ln 0.8).
    #[test]
    fn the_statistics_match_their_closed_forms() {
        let (observed, law) = ([30, 50, 20], [0.25, 0.5, 0.25]);
        let x2 = goodness_of_fit(&[cell(&observed, &law)], FitStatistic::PearsonChiSquare).unwrap();
        assert_relative_eq!(x2.statistic, 2.0, epsilon = 1e-12);
        assert_eq!((x2.degrees_of_freedom, x2.observations), (2, 100));
        // Chi-square with 2 dof has the closed-form tail exp(-x / 2).
        assert_relative_eq!(x2.p_value, (-1.0_f64).exp(), epsilon = 1e-9);
        let g = goodness_of_fit(&[cell(&observed, &law)], FitStatistic::LikelihoodRatioG).unwrap();
        assert_relative_eq!(
            g.statistic,
            2.0 * (30.0 * 1.2_f64.ln() + 20.0 * 0.8_f64.ln()),
            epsilon = 1e-12
        );
    }

    #[test]
    fn independent_cells_add_statistics_and_degrees_of_freedom() {
        let (a, law_a) = ([30, 50, 20], [0.25, 0.5, 0.25]);
        let (b, law_b) = ([55, 45], [0.5, 0.5]);
        for statistic in BOTH {
            let one = goodness_of_fit(&[cell(&a, &law_a)], statistic).unwrap();
            let other = goodness_of_fit(&[cell(&b, &law_b)], statistic).unwrap();
            let both = goodness_of_fit(&[cell(&a, &law_a), cell(&b, &law_b)], statistic).unwrap();
            assert_relative_eq!(
                both.statistic,
                one.statistic + other.statistic,
                epsilon = 1e-12
            );
            assert_eq!(both.degrees_of_freedom, 3);
        }
    }

    fn draw(law: &[f64], n: usize, rng: &mut StdRng) -> Vec<u64> {
        let mut counts = vec![0_u64; law.len()];
        for _ in 0..n {
            let u: f64 = rng.gen();
            let mut acc = 0.0;
            let category = law
                .iter()
                .position(|p| {
                    acc += p;
                    u < acc
                })
                .unwrap_or(law.len() - 1);
            counts[category] += 1;
        }
        counts
    }

    /// A sampler that follows the law is rejected at 0.05 about one time in
    /// twenty; one that is biased by a few percent is rejected nearly always.
    #[test]
    fn a_faithful_sampler_passes_at_the_nominal_rate_and_a_biased_one_does_not() {
        let law = [0.1, 0.2, 0.3, 0.4];
        let biased = [0.13, 0.2, 0.3, 0.37];
        for statistic in BOTH {
            let rejected = |sampler: &[f64]| {
                (0..200_u64)
                    .filter(|&seed| {
                        let counts = draw(sampler, 5000, &mut StdRng::seed_from_u64(seed));
                        goodness_of_fit(&[cell(&counts, &law)], statistic)
                            .unwrap()
                            .p_value
                            < 0.05
                    })
                    .count()
            };
            let (faithful, unfaithful) = (rejected(&law), rejected(&biased));
            assert!(
                (2..=20).contains(&faithful),
                "{statistic:?}: {faithful} of 200"
            );
            assert!(unfaithful >= 190, "{statistic:?}: {unfaithful} of 200");
        }
    }

    #[test]
    fn a_zero_count_is_a_valid_observation() {
        let g = goodness_of_fit(
            &[cell(&[0, 100], &[0.06, 0.94])],
            FitStatistic::LikelihoodRatioG,
        )
        .unwrap();
        assert!(g.statistic > 0.0 && g.p_value < 0.05);
    }

    #[test]
    fn refuses_cells_that_are_malformed_or_too_sparse() {
        let s = FitStatistic::PearsonChiSquare;
        assert!(goodness_of_fit(&[], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10], &[1.0])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10, 10], &[0.5])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10, 10], &[0.5, 0.4])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10, 10], &[1.2, -0.2])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[0, 0], &[0.5, 0.5])], s).is_err());
        // 40 draws at p = 0.1 expect 4 in the rare category.
        assert!(goodness_of_fit(&[cell(&[4, 36], &[0.1, 0.9])], s).is_err());
    }
}
