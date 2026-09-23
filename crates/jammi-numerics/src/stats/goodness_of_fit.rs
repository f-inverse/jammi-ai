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
//! observations (Cochran, *Ann. Math. Stat.* 23, 1952; *Biometrics* 10,
//! 1954). A cell whose sparsest categories fall short is pooled rather than
//! refused: its categories are merged, sparsest first, into one category until
//! that merged category and every category left expect at least the floor —
//! the standard remedy, and decided from the law and the cell's total alone,
//! never from the counts, so the pooled categories are fixed before any count
//! is read and each pooled cell is still a multinomial under the law. A cell
//! whose total cannot reach the floor in two categories — or that observed
//! nothing — is untested, and reported as such.
//!
//! Conditioning each cell on its own total is what makes the cells of a
//! Markov chain's transition law independent experiments: the counts out of
//! each state, given how often the state was visited, are multinomial with
//! that state's transition probabilities (Anderson & Goodman, *Ann. Math.
//! Stat.* 28, 1957).

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
    /// `sum (O - E)^2 / E`. Holds its nominal level at moderate expected
    /// counts (Larntz, *JASA* 73, 1978).
    PearsonChiSquare,
    /// `2 sum O ln(O / E)`, a zero count contributing zero. Exceeds its
    /// nominal level at moderate expected counts (Larntz 1978).
    LikelihoodRatioG,
}

impl FitStatistic {
    /// One category's contribution: observed `o` against expected `e > 0`.
    fn term(self, o: f64, e: f64) -> f64 {
        match self {
            Self::PearsonChiSquare => (o - e).powi(2) / e,
            Self::LikelihoodRatioG if o == 0.0 => 0.0,
            Self::LikelihoodRatioG => 2.0 * o * (o / e).ln(),
        }
    }
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
    /// `sum over tested cells of (categories after pooling - 1)`.
    pub degrees_of_freedom: usize,
    /// Upper tail of the chi-square distribution at `statistic`.
    pub p_value: f64,
    /// Observations over the tested cells.
    pub observations: u64,
    /// Cells tested, pooled or not.
    pub cells_tested: usize,
    /// Categories merged into a pooled category, over the tested cells.
    pub categories_pooled: usize,
    /// Cells whose total cannot reach the floor in two categories.
    pub cells_untested: usize,
    /// Observations in the untested cells.
    pub observations_untested: u64,
}

/// A cell's categories after pooling: `(observed, probability)` per category.
type Categories = Vec<(u64, f64)>;

/// `cell`'s categories pooled to [`MIN_EXPECTED_COUNT`] at its total `n`, and
/// how many categories were merged; `None` when fewer than two categories can
/// reach the floor.
fn pooled(cell: &FitCell<'_>, n: u64) -> Option<(Categories, usize)> {
    let n = n as f64;
    let mut order: Vec<usize> = (0..cell.probabilities.len()).collect();
    order.sort_by(|&a, &b| {
        cell.probabilities[a]
            .total_cmp(&cell.probabilities[b])
            .then(a.cmp(&b))
    });
    // Ascending, so once a category reaches the floor every later one does:
    // the merged prefix ends at the first category that reaches the floor
    // with the merged category already there.
    let merged = order
        .iter()
        .scan(0.0, |pool, &i| {
            let pool_short = *pool > 0.0 && *pool * n < MIN_EXPECTED_COUNT;
            let sparse = cell.probabilities[i] * n < MIN_EXPECTED_COUNT;
            (sparse || pool_short).then(|| *pool += cell.probabilities[i])
        })
        .count();
    let (pool, kept) = order.split_at(merged);
    let pool = (!pool.is_empty()).then(|| {
        pool.iter().fold((0, 0.0), |(o, p), &i| {
            (o + cell.observed[i], p + cell.probabilities[i])
        })
    });
    let categories: Categories = kept
        .iter()
        .map(|&i| (cell.observed[i], cell.probabilities[i]))
        .chain(pool)
        .collect();
    (categories.len() >= 2).then_some((categories, merged))
}

/// One cell's part of the pooled fit.
enum CellFit {
    Tested {
        statistic: f64,
        degrees_of_freedom: usize,
        observations: u64,
        merged: usize,
    },
    Untested {
        observations: u64,
    },
}

/// Validate `cell` (the `index`-th) and fit it, pooled, on its own.
fn cell_fit(index: usize, cell: &FitCell<'_>, statistic: FitStatistic) -> Result<CellFit> {
    let k = cell.probabilities.len();
    if cell.observed.len() != k {
        return Err(NumericsError::DimensionMismatch {
            expected: k,
            got: cell.observed.len(),
        });
    }
    let invalid = |reason: String| NumericsError::InvalidInput(format!("cell {index}: {reason}"));
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
    Ok(match pooled(cell, n) {
        None => CellFit::Untested { observations: n },
        Some((categories, merged)) => CellFit::Tested {
            statistic: categories
                .iter()
                .map(|&(o, p)| statistic.term(o as f64, p * n as f64))
                .sum(),
            degrees_of_freedom: categories.len() - 1,
            observations: n,
            merged,
        },
    })
}

impl GoodnessOfFit {
    /// The fit of no cell yet: nothing tested, nothing refused.
    const EMPTY: Self = Self {
        statistic: 0.0,
        degrees_of_freedom: 0,
        p_value: 1.0,
        observations: 0,
        cells_tested: 0,
        categories_pooled: 0,
        cells_untested: 0,
        observations_untested: 0,
    };

    /// This fit with one more independent cell's part added.
    fn with(self, cell: CellFit) -> Self {
        match cell {
            CellFit::Tested {
                statistic,
                degrees_of_freedom,
                observations,
                merged,
            } => Self {
                statistic: self.statistic + statistic,
                degrees_of_freedom: self.degrees_of_freedom + degrees_of_freedom,
                observations: self.observations + observations,
                cells_tested: self.cells_tested + 1,
                categories_pooled: self.categories_pooled + merged,
                ..self
            },
            CellFit::Untested { observations } => Self {
                cells_untested: self.cells_untested + 1,
                observations_untested: self.observations_untested + observations,
                ..self
            },
        }
    }
}

/// Pooled goodness of fit of independent `cells` to their laws.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] when there are no cells; when a cell has
/// fewer than two categories, a probability outside `(0, 1]` or
/// probabilities that do not sum to 1; or when no cell can be tested.
/// [`NumericsError::DimensionMismatch`] when a cell's counts and
/// probabilities differ in length.
pub fn goodness_of_fit(cells: &[FitCell<'_>], statistic: FitStatistic) -> Result<GoodnessOfFit> {
    if cells.is_empty() {
        return Err(NumericsError::InvalidInput(
            "goodness of fit requires at least one cell".into(),
        ));
    }
    let fit = cells
        .iter()
        .enumerate()
        .map(|(index, cell)| cell_fit(index, cell, statistic))
        .try_fold(GoodnessOfFit::EMPTY, |fit, cell| cell.map(|c| fit.with(c)))?;
    if fit.cells_tested == 0 {
        return Err(NumericsError::InvalidInput(format!(
            "none of {} cell(s) can be tested: no cell's total reaches {MIN_EXPECTED_COUNT} expected in two categories",
            cells.len()
        )));
    }
    let chi_square = ChiSquared::new(fit.degrees_of_freedom as f64)
        .map_err(|e| NumericsError::InvalidInput(format!("chi-square distribution: {e}")))?;
    // Rounding can leave G a hair below zero when the counts match the law
    // exactly; the statistic is non-negative by definition.
    let statistic = fit.statistic.max(0.0);
    Ok(GoodnessOfFit {
        statistic,
        p_value: (1.0 - chi_square.cdf(statistic)).clamp(0.0, 1.0),
        ..fit
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
    fn refuses_cells_that_are_malformed_or_untestable() {
        let s = FitStatistic::PearsonChiSquare;
        assert!(goodness_of_fit(&[], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10], &[1.0])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10, 10], &[0.5])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10, 10], &[0.5, 0.4])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[10, 10], &[1.2, -0.2])], s).is_err());
        // Nothing observed, and 40 draws at p = 0.1 (4 expected in the rare
        // category, which pooling can only merge into the whole cell): no
        // cell can be tested.
        assert!(goodness_of_fit(&[cell(&[0, 0], &[0.5, 0.5])], s).is_err());
        assert!(goodness_of_fit(&[cell(&[4, 36], &[0.1, 0.9])], s).is_err());
    }

    /// 100 draws expecting (4, 3, 43, 50): the two sparse categories merge
    /// into one expecting 7, and the cell is tested as that three-category
    /// multinomial.
    #[test]
    fn sparse_categories_pool_into_one_that_reaches_the_floor() {
        let observed = [3, 2, 45, 50];
        let law = [0.04, 0.03, 0.43, 0.5];
        for statistic in BOTH {
            let fit = goodness_of_fit(&[cell(&observed, &law)], statistic).unwrap();
            let by_hand =
                goodness_of_fit(&[cell(&[45, 50, 5], &[0.43, 0.5, 0.07])], statistic).unwrap();
            assert_relative_eq!(fit.statistic, by_hand.statistic, epsilon = 1e-12);
            assert_eq!(fit.degrees_of_freedom, 2);
            assert_eq!(fit.categories_pooled, 2);
            assert_eq!((fit.cells_tested, fit.cells_untested), (1, 0));
        }
    }

    /// A merged category that is still short takes the next-sparsest in.
    #[test]
    fn a_short_pool_absorbs_the_next_sparsest_category() {
        // 100 draws expecting (3, 6, 91): 3 alone is short, 3 + 6 is not.
        let fit = goodness_of_fit(
            &[cell(&[2, 7, 91], &[0.03, 0.06, 0.91])],
            FitStatistic::PearsonChiSquare,
        )
        .unwrap();
        assert_eq!((fit.degrees_of_freedom, fit.categories_pooled), (1, 2));
    }

    #[test]
    fn an_untestable_cell_is_counted_beside_the_tested_ones() {
        let fit = goodness_of_fit(
            &[
                cell(&[30, 50, 20], &[0.25, 0.5, 0.25]),
                cell(&[1, 6], &[0.5, 0.5]),
            ],
            FitStatistic::PearsonChiSquare,
        )
        .unwrap();
        assert_eq!((fit.cells_tested, fit.observations), (1, 100));
        assert_eq!((fit.cells_untested, fit.observations_untested), (1, 7));
        assert_eq!(fit.degrees_of_freedom, 2);
    }

    /// Pooling is decided from the law and the total, never the counts, so a
    /// faithful sampler over a law with many rare categories is still
    /// rejected at the nominal rate, and a biased one still is not missed.
    #[test]
    fn pooling_keeps_the_nominal_rate_on_a_law_with_many_rare_categories() {
        let rare = std::iter::repeat_n(0.01, 10);
        let law: Vec<f64> = rare.clone().chain([0.3, 0.6]).collect();
        let biased: Vec<f64> = std::iter::repeat_n(0.02, 10).chain([0.2, 0.6]).collect();
        let rejected = |sampler: &[f64]| {
            (0..200_u64)
                .filter(|&seed| {
                    let counts = draw(sampler, 200, &mut StdRng::seed_from_u64(seed));
                    let fit =
                        goodness_of_fit(&[cell(&counts, &law)], FitStatistic::PearsonChiSquare)
                            .unwrap();
                    assert_eq!(fit.categories_pooled, 10);
                    fit.p_value < 0.05
                })
                .count()
        };
        let (faithful, unfaithful) = (rejected(&law), rejected(&biased));
        assert!((2..=20).contains(&faithful), "{faithful} of 200");
        assert!(unfaithful >= 190, "{unfaithful} of 200");
    }
}
