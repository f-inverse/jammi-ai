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
/// refused with [`NumericsError::InvalidInput`]. Sort-safety is NOT the
/// reason: the sort below uses `total_cmp` (not `partial_cmp`), and
/// `total_cmp` is a genuine total order even over a `NaN`- or
/// `±inf`-containing slice (see [`bootstrap_ci`](crate::stats::bootstrap_ci)
/// for the same point made explicitly) — so a `NaN` or `±inf` element would
/// sort into a well-defined position and NOT corrupt the sort itself. The
/// real reason is downstream: this score exists to be compared against a
/// fixed drift-monitor threshold, and neither a `NaN` nor an unbounded
/// `±inf` distance is meaningful there. A `NaN` comparison against any
/// threshold is always `false` (`NaN > c` never fires, silently defeating
/// the monitor), and an unbounded `±inf` distance carries no *magnitude*
/// information past "already over any finite threshold" — it cannot
/// distinguish a mild distributional shift from a catastrophic one, which is
/// exactly what a drift score exists to do. Both are refused for this one,
/// consistent, "meaningless to the consumer" reason, not two different ones.
/// This is stricter than scipy's `wasserstein_distance`, which propagates
/// both a `NaN` element and a `±inf` element straight through into its
/// return value (`scipy/stats/_stats_py.py` has no finiteness guard) rather
/// than refusing either; the empty-input edge above is the only case where
/// jammi matches scipy's behavior exactly.
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
