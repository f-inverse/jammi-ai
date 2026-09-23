//! Wall-time statistics shared by the serving tiers: the order statistics of a
//! set of repeated serves, and the two-term cost model a row sweep is fitted
//! to.

use serde::{Deserialize, Serialize};

/// The `p`-quantile of `sorted` (ascending) by nearest rank:
/// `ceil(p · n)` clamped into `[1, n]`. The p99 of a short sample is therefore
/// its slowest serve — the honest tail for a small iteration count.
///
/// `sorted` must be non-empty.
pub(crate) fn nearest_rank(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    sorted[((p * n as f64).ceil() as usize).clamp(1, n) - 1]
}

/// The order statistics of one leg's repeated serves, milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ServeStats {
    /// The median serve — the representative steady-state wall time.
    pub p50_ms: f64,
    /// The fastest serve — the cost of the code path with the least
    /// interference from the box, the statistic a same-process ratio is
    /// stable under.
    pub min_ms: f64,
}

impl ServeStats {
    /// `None` for an empty sample: a leg that measured nothing has no
    /// statistic, never a zero.
    pub(crate) fn of(samples_ms: &[f64]) -> Option<Self> {
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let min_ms = *sorted.first()?;
        Some(Self {
            p50_ms: nearest_rank(&sorted, 0.50),
            min_ms,
        })
    }
}

/// A rate at `wall_ms`, or `0.0` when the wall time is not positive (a
/// degenerate measurement a gate fails closed on rather than dividing by
/// zero).
pub(crate) fn per_second(count: usize, wall_ms: f64) -> f64 {
    if wall_ms > 0.0 {
        count as f64 / (wall_ms / 1_000.0)
    } else {
        0.0
    }
}

/// The two-term cost model of a serve, fitted over a row sweep:
/// `serve_ms = fixed_ms + per_row_ms · rows`.
///
/// `fixed_ms` is what one call costs whatever it serves (planning, catalog
/// writes, opening and committing the result table); `per_row_ms` is what each
/// further row costs (tokenize, forward, pool, write). A single serve time
/// cannot separate the two; a sweep can.
///
/// ## Relative least squares
///
/// Timing noise is multiplicative — a serve ten times longer jitters ten times
/// as many milliseconds — so the fit minimises the RELATIVE residual
/// `Σ ((fixed + per_row·rowsᵢ − msᵢ) / msᵢ)²`. An unweighted fit over a sweep
/// spanning three decades of row counts is decided by its largest point alone,
/// and its intercept — the fixed cost, the number the small serves carry — is
/// then whatever error the large point left over.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostFit {
    /// The per-call cost that does not scale with the row count, milliseconds.
    pub fixed_ms: f64,
    /// The cost of one further row, milliseconds.
    pub per_row_ms: f64,
    /// Root-mean-square of the fit's relative residuals — `0.0` for a
    /// two-point sweep, which the model passes through exactly; how far the
    /// serve is from two-term otherwise.
    pub relative_residual_rms: f64,
}

impl CostFit {
    /// Fit `(rows, serve_ms)` points. `None` when the sweep cannot determine
    /// two terms: fewer than two distinct row counts, or a non-positive serve
    /// time.
    pub(crate) fn least_squares(points: &[(usize, f64)]) -> Option<Self> {
        if points.iter().any(|&(_, ms)| !(ms > 0.0 && ms.is_finite())) {
            return None;
        }
        // Weighted normal equations with weights wᵢ = 1 / msᵢ².
        let (sw, swx, swxx, swy, swxy) = points.iter().fold(
            (0.0, 0.0, 0.0, 0.0, 0.0),
            |(sw, swx, swxx, swy, swxy), &(rows, ms)| {
                let (x, w) = (rows as f64, 1.0 / (ms * ms));
                (
                    sw + w,
                    swx + w * x,
                    swxx + w * x * x,
                    swy + w * ms,
                    swxy + w * x * ms,
                )
            },
        );
        // Zero (to rounding) exactly when every point shares one row count.
        let determinant = sw * swxx - swx * swx;
        let two_distinct_row_counts = determinant > 1e-9 * sw * swxx;
        if !two_distinct_row_counts {
            return None;
        }
        let per_row_ms = (sw * swxy - swx * swy) / determinant;
        let fixed_ms = (swy - per_row_ms * swx) / sw;
        let relative_residual_rms = (points
            .iter()
            .map(|&(rows, ms)| ((fixed_ms + per_row_ms * rows as f64 - ms) / ms).powi(2))
            .sum::<f64>()
            / points.len() as f64)
            .sqrt();
        Some(Self {
            fixed_ms,
            per_row_ms,
            relative_residual_rms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_picks_the_median_and_the_tail() {
        let sorted: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(nearest_rank(&sorted, 0.50), 5.0);
        assert_eq!(nearest_rank(&sorted, 0.99), 10.0);
        assert_eq!(nearest_rank(&[7.0], 0.50), 7.0);
    }

    #[test]
    fn serve_stats_of_an_empty_sample_is_none() {
        assert_eq!(ServeStats::of(&[]), None);
        assert_eq!(
            ServeStats::of(&[3.0, 1.0, 2.0]),
            Some(ServeStats {
                p50_ms: 2.0,
                min_ms: 1.0
            })
        );
    }

    /// A sweep that IS two-term recovers both terms with no residual, across
    /// three decades of row counts.
    #[test]
    fn an_exactly_two_term_sweep_recovers_both_terms() {
        let points: Vec<(usize, f64)> = [16, 256, 4096, 16384]
            .iter()
            .map(|&rows| (rows, 7.5 + 0.02 * rows as f64))
            .collect();
        let fit = CostFit::least_squares(&points).expect("four distinct row counts");
        assert!((fit.fixed_ms - 7.5).abs() < 1e-9, "{fit:?}");
        assert!((fit.per_row_ms - 0.02).abs() < 1e-12, "{fit:?}");
        assert!(fit.relative_residual_rms < 1e-12, "{fit:?}");
    }

    /// The reason the fit is relative: the same 5% error on the LARGEST point
    /// moves an unweighted intercept by the whole error (16 ms here, twice the
    /// true fixed cost), while the relative fit keeps the fixed cost the small
    /// serves measured.
    #[test]
    fn noise_on_the_largest_point_does_not_decide_the_fixed_cost() {
        let mut points: Vec<(usize, f64)> = [16, 256, 4096, 16384]
            .iter()
            .map(|&rows| (rows, 7.5 + 0.02 * rows as f64))
            .collect();
        points[3].1 *= 1.05;
        let fit = CostFit::least_squares(&points).expect("four distinct row counts");
        assert!(
            (fit.fixed_ms - 7.5).abs() < 0.5,
            "the fixed cost must stay near the 7.5 ms the small serves carry: {fit:?}"
        );
        assert!(fit.relative_residual_rms > 0.0);
    }

    #[test]
    fn a_sweep_that_cannot_determine_two_terms_has_no_fit() {
        assert_eq!(CostFit::least_squares(&[]), None);
        assert_eq!(CostFit::least_squares(&[(16, 8.0)]), None);
        assert_eq!(CostFit::least_squares(&[(16, 8.0), (16, 8.1)]), None);
        assert_eq!(CostFit::least_squares(&[(16, 8.0), (256, 0.0)]), None);
    }
}
