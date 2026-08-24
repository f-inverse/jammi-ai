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
/// sort into a well-defined position and NOT corrupt the sort itself. What
/// actually happens without the guard, measured downstream of the sort:
///
/// - A `NaN` element always yields a `NaN` score: it propagates through the
///   `(a - b).abs()` sum (any arithmetic touching `NaN` is `NaN`) into
///   `mean`, and `mean / scale` stays `NaN` regardless of `scale`.
/// - A `±inf` element yields either `NaN` or an unbounded `±inf` score,
///   depending on how the two populations pair up once interpolated onto
///   the common grid — not one fixed outcome. Equal-length populations skip
///   [`interpolate_to`] entirely (see its doc), so `±inf` lands at the same
///   sorted index in both and an aligned `inf - inf` (or `-inf - -inf`)
///   produces `NaN`; e.g. `wasserstein_1d(&[1.0, f32::INFINITY], &[2.0,
///   f32::INFINITY])` is `NaN`. An `inf` in only one population instead
///   produces an unbounded, but non-`NaN`, `±inf` numerator. Separately,
///   [`padded_range`] falls back to a fixed `(0.0, 1.0)` — silently pinning
///   `scale = 1.0` regardless of `current`'s magnitude, the same fallback
///   the empty-`reference` case above defeats — whenever `reference`'s own
///   min/max range is non-finite, which happens whenever `reference`
///   contains `±inf` (see [`padded_range`]'s doc for exactly when the
///   fallback fires).
///
/// In every one of those outcomes the number is meaningless to this
/// function's one consumer: this score exists to be compared against a
/// fixed drift-monitor threshold. A `NaN` comparison against any threshold
/// is always `false` (`NaN > c` never fires, silently defeating the
/// monitor); an unbounded `±inf` distance carries no *magnitude*
/// information past "already over any finite threshold", so it cannot
/// distinguish a mild distributional shift from a catastrophic one, which is
/// exactly what a drift score exists to do; and a scale silently pinned to
/// `1.0` breaks this function's own documented scale-invariance the same
/// way the empty-`reference` case does. All three are refused for this one,
/// consistent, "meaningless to the consumer" reason, not distinct ones. This
/// is stricter than scipy's `wasserstein_distance`, which propagates both a
/// `NaN` element and a `±inf` element straight through into its return value
/// (`scipy/stats/_stats_py.py` has no finiteness guard) rather than refusing
/// either; the empty-input edge above is the only case where jammi matches
/// scipy's behavior exactly.
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
