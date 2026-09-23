//! Where an output series' initial transient ends: the marginal standard
//! error rule (MSER; White, *Simulation* 69, 1997) over batch means (MSER-m;
//! White, Cobb & Spratt, WSC 2000).
//!
//! A series that starts away from its steady state — a cold cache, a clock
//! still settling, an allocator still growing — biases any location taken
//! over it. MSER deletes the prefix whose removal minimises the squared
//! standard error of the mean of what is left,
//!
//! ```text
//! g(k) = (1 / (b − k)²) · Σ_{j ≥ k} (z_j − z̄_k)²
//! ```
//!
//! over the `b` batch means `z_j` of `m` consecutive observations (`m = 1`
//! is White's rule on the observations themselves), `z̄_k` being the mean of
//! the batches from `k` on (Wang & Glynn, *ACM TOMACS* 27, 2016, §3). The
//! argmin runs over the first half only, `k < ⌊b/2⌋`: a minimum at the last
//! point of that range means the transient has not been seen to end within
//! the series (White et al. 2000; Hoad, Robinson & Davies, *JORS* 61, 2010).
//!
//! The rule chooses a point; it tests nothing. Nothing is accepted or
//! rejected at a level, so looking at more points does not inflate an error
//! rate — whether what is left is steady is for a test the caller applies
//! to it afterwards.

use crate::error::{NumericsError, Result};

/// The fewest batch means the rule is run over: a first half of two points
/// is one candidate besides the limit.
const MIN_BATCHES: usize = 4;

/// Where a series' transient ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truncation {
    /// The first observation kept: the series from `at` on is what the rule
    /// keeps.
    pub at: usize,
    /// The minimum fell at the limit of the first half: the transient was
    /// not seen to end, and what is kept is the second half.
    pub at_limit: bool,
}

/// MSER-`batch` over `series` in its given (time) order.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] when `batch` is zero, `series` holds
/// fewer than four batches, or it contains a non-finite value.
pub fn mser_truncation(series: &[f64], batch: usize) -> Result<Truncation> {
    if batch == 0 {
        return Err(NumericsError::InvalidInput(
            "MSER needs a batch size of at least 1".into(),
        ));
    }
    if series.iter().any(|x| !x.is_finite()) {
        return Err(NumericsError::InvalidInput(
            "MSER requires finite observations".into(),
        ));
    }
    let means: Vec<f64> = series
        .chunks_exact(batch)
        .map(|c| c.iter().sum::<f64>() / batch as f64)
        .collect();
    if means.len() < MIN_BATCHES {
        return Err(NumericsError::InvalidInput(format!(
            "MSER-{batch} needs at least {MIN_BATCHES} batches ({} observations), got {}",
            MIN_BATCHES * batch,
            series.len()
        )));
    }
    let half = means.len() / 2;
    // Each suffix's count and sum of squared deviations, accumulated from the
    // end by Welford's update: the textbook `Σz² − (Σz)²/n` cancels, and
    // its rounding would pick the argmin among suffixes that are equal.
    let suffix: Vec<(f64, f64)> = means
        .iter()
        .rev()
        .scan((0.0, 0.0, 0.0), |(count, mean, squares), z| {
            *count += 1.0;
            let delta = z - *mean;
            *mean += delta / *count;
            *squares += delta * (z - *mean);
            Some((*count, *squares))
        })
        .collect();
    let at = (0..half)
        .map(|k| {
            let (count, squares) = suffix[means.len() - 1 - k];
            (squares / (count * count), k)
        })
        .min_by(|(a, _), (b, _)| a.total_cmp(b))
        .map_or(0, |(_, k)| k);
    Ok(Truncation {
        at: at * batch,
        at_limit: at == half - 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| 1.0 + 0.01 * rng.gen::<f64>()).collect()
    }

    /// `g` by its definition, for the argmin to be checked against.
    fn g(means: &[f64], k: usize) -> f64 {
        let rest = &means[k..];
        let mean = rest.iter().sum::<f64>() / rest.len() as f64;
        rest.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (rest.len() * rest.len()) as f64
    }

    #[test]
    fn the_point_is_the_argmin_of_g_over_the_first_half() {
        let series: Vec<f64> = (0..60)
            .map(|i| 1.0 + 0.5 * (-(i as f64) / 2.0).exp() + 0.001 * ((i * 7919) % 13) as f64)
            .collect();
        let means: Vec<f64> = series
            .chunks_exact(5)
            .map(|c| c.iter().sum::<f64>() / 5.0)
            .collect();
        let by_definition = (0..means.len() / 2)
            .min_by(|&a, &b| g(&means, a).total_cmp(&g(&means, b)))
            .unwrap();
        assert_eq!(
            mser_truncation(&series, 5).unwrap(),
            Truncation {
                at: by_definition * 5,
                at_limit: false
            }
        );
    }

    /// A series that starts high and steps down to its level is cut at the
    /// step, by observation and by batch.
    #[test]
    fn a_warming_start_is_cut_at_its_end() {
        let warming: Vec<f64> = noise(80, 1)
            .iter()
            .enumerate()
            .map(|(i, x)| if i < 15 { x + 0.3 } else { *x })
            .collect();
        for batch in [1, 5] {
            assert_eq!(
                mser_truncation(&warming, batch).unwrap(),
                Truncation {
                    at: 15,
                    at_limit: false
                }
            );
        }
    }

    /// A series that drifts the whole way is not seen to settle: the minimum
    /// sits at the limit of the first half.
    #[test]
    fn a_series_that_drifts_throughout_reaches_the_limit() {
        let drifting: Vec<f64> = noise(80, 2)
            .iter()
            .enumerate()
            .map(|(i, x)| x + 0.01 * i as f64)
            .collect();
        assert_eq!(
            mser_truncation(&drifting, 1).unwrap(),
            Truncation {
                at: 39,
                at_limit: true
            }
        );
    }

    /// Every suffix of a constant series has no spread; the first is kept.
    #[test]
    fn a_constant_series_is_kept_whole() {
        assert_eq!(
            mser_truncation(&[0.0123; 16], 1).unwrap(),
            Truncation {
                at: 0,
                at_limit: false
            }
        );
    }

    #[test]
    fn refuses_what_it_cannot_run_over() {
        assert!(mser_truncation(&noise(19, 3), 5).is_err());
        assert!(mser_truncation(&noise(40, 3), 0).is_err());
        assert!(mser_truncation(&[1.0, f64::NAN, 1.0, 1.0], 1).is_err());
    }
}
