//! 1-D Wasserstein distance (Earth-Mover's distance) between two scalar
//! populations, normalised by the reference range so the result is
//! comparable across populations on different scales.

use crate::error::{NumericsError, Result};
use crate::histogram::binning::padded_range;
use crate::histogram::interpolate::interpolate_to;

/// 1-D Wasserstein distance between `reference` and `current`. Both
/// populations are sorted and linearly interpolated onto a common grid
/// of length `max(|reference|, |current|, 1)`; the mean absolute
/// difference is then divided by the reference's padded range so the
/// kernel is scale-invariant.
///
/// An empty `reference` or `current` is refused with
/// [`NumericsError::InvalidInput`]: this MATCHES scipy, which raises
/// `ValueError("Distribution can't be empty.")` from `_validate_distribution`
/// (`scipy/stats/_stats_py.py`) for the same case — jammi is not stricter
/// here. Without the guard, an empty `reference` falls through to
/// [`padded_range`]'s empty-input fallback of `(0.0, 1.0)`, fixing `scale =
/// 1.0` regardless of `current`'s actual magnitude and silently breaking
/// this function's own documented scale-invariance (see the module doc);
/// an empty `current` instead interpolates a length-`|reference|` zero
/// vector against `reference`, producing a plausible-looking finite number
/// with no distributional meaning.
///
/// A non-finite (`NaN` or `±inf`) element in either population is also
/// refused with [`NumericsError::InvalidInput`] rather than silently sorted: a
/// `sort_by` comparator built from `partial_cmp` is not a total order once a
/// `NaN` is present (`NaN.partial_cmp(&x)` is `None` for every `x`), and
/// collapsing that `None` to `Ordering::Equal` — as the prior implementation
/// did — makes the sort order "unspecified" per the `[T]::sort_by` docs,
/// which would silently corrupt the resulting divergence score. This is the
/// same edge scipy's `wasserstein_distance` does not guard: a `NaN` input
/// propagates to a `NaN` distance there rather than being rejected. jammi is
/// stricter than scipy only on this NaN/inf edge, not on the empty-input
/// edge above — a `NaN` divergence score is a "confident wrong number" for
/// every downstream drift-monitor threshold that never fires.
///
/// # Errors
///
/// Returns [`NumericsError::InvalidInput`] if `reference` or `current` is
/// empty, or if either contains a non-finite (`NaN` or `±inf`) element.
pub fn wasserstein_1d(reference: &[f32], current: &[f32]) -> Result<f64> {
    if reference.is_empty() || current.is_empty() {
        return Err(NumericsError::InvalidInput(
            "wasserstein_1d requires non-empty reference and current populations".into(),
        ));
    }
    if reference.iter().any(|x| !x.is_finite()) || current.iter().any(|x| !x.is_finite()) {
        return Err(NumericsError::InvalidInput(
            "wasserstein_1d requires finite (non-NaN, non-infinite) inputs".into(),
        ));
    }
    let mut sorted_ref: Vec<f64> = reference.iter().map(|x| *x as f64).collect();
    let mut sorted_cur: Vec<f64> = current.iter().map(|x| *x as f64).collect();
    sorted_ref.sort_by(f64::total_cmp);
    sorted_cur.sort_by(f64::total_cmp);
    let n = sorted_ref.len().max(sorted_cur.len()).max(1);
    let interp_ref = interpolate_to(&sorted_ref, n);
    let interp_cur = interpolate_to(&sorted_cur, n);
    let sum_abs: f64 = interp_ref
        .iter()
        .zip(interp_cur.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let mean = sum_abs / n as f64;
    let (min, max) = padded_range(reference);
    let scale = (max - min).abs().max(f64::EPSILON);
    Ok(mean / scale)
}
