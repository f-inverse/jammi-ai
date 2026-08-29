//! Exact two-sided sign test.
//!
//! The sign test asks a single question of a set of paired differences
//! `d_i = a_i - b_i`: ignoring magnitude entirely, is the *sign* of `d_i`
//! more consistent than a fair coin flip would produce? Under the null
//! hypothesis each non-tied `d_i` is independently `+` or `-` with
//! probability 1/2, so the count of positive signs among `n` non-tied pairs
//! is `Binomial(n, 0.5)`-distributed. The two-sided p-value is the exact
//! tail probability of that binomial, doubled and capped at `1.0`.
//!
//! Unlike [`crate::stats::welch::welch_t_test`] and
//! [`crate::stats::mannwhitney::mann_whitney_u`], which both fall back to a
//! continuous approximation (Student's t / normal), the sign test's null
//! distribution is discrete and small-`n`-exact by construction, so there is
//! no approximation to fall back to: the p-value here is computed as an
//! exact ratio of `u128` integers (binomial coefficients built via the
//! divisibility-exact multiplicative recurrence `C(n,i) = C(n,i-1) *
//! (n-i+1) / i`, summed, then cast to `f64` only once at the very end) and
//! never a floating-point CDF evaluation. A float CDF (e.g. a beta/incomplete
//! regularized approximation) would introduce approximation error into
//! exactly the digits the pre-registered decision boundary
//! (`n=12, k>=11 => alpha2=0.0064`) depends on.

use crate::error::{NumericsError, Result};
use crate::stats::types::SignTestResult;

/// Exact two-sided sign test over paired differences `diffs[i] = a_i - b_i`.
///
/// Zero-valued differences are ties: excluded from the binomial count `n`
/// (per the standard sign-test convention) but never silently dropped — the
/// count is reported back on [`SignTestResult::ties`], and `n = n_pos +
/// n_neg` is reported on [`SignTestResult::n`] so the caller can see exactly
/// how many pairs the reported `p_value` is conditioned on.
///
/// # Errors
///
/// - Returns `NumericsError::InvalidInput` if `diffs` is empty (`n=0`: there
///   is nothing to test).
/// - Returns `NumericsError::InvalidInput` if every element of `diffs` is an
///   exact tie (`n_pos = n_neg = 0`, i.e. `n=0` again, but reached only
///   after every element resolved to a tie rather than because none were
///   supplied — reported with a distinct message so the two `n=0` causes are
///   not conflated in the returned error).
/// - Returns `NumericsError::InvalidInput` if any element of `diffs` is
///   `NaN`. `NaN` has no sign under IEEE-754 (`NaN > 0.0`, `NaN < 0.0`, and
///   `NaN == 0.0` are all `false`), so it is neither a positive, a negative,
///   nor a tie — silently routing it into any one of those three buckets
///   would misclassify it. `±inf` IS accepted (not refused): its sign is
///   well-defined (`f64::is_sign_positive`/`is_sign_negative`) and it is
///   never a tie, so it classifies exactly like any finite non-zero value.
/// - Returns `NumericsError::InvalidInput` if `n = n_pos + n_neg` is too
///   large for the exact `u128` computation below to represent without
///   overflow (in practice this bites somewhere in the `n ~ 125-127` range —
///   see the overflow note on `binomial_row` (this module) for exactly which
///   quantity overflows first). This is detected by
///   `checked_mul`/`checked_shl` at
///   the point of overflow, not by a hardcoded threshold, so the refusal is
///   exact rather than a guess: this function never silently truncates or
///   falls back to a float approximation of the tail it exists to compute
///   exactly.
pub fn sign_test(diffs: &[f64]) -> Result<SignTestResult> {
    if diffs.is_empty() {
        return Err(NumericsError::InvalidInput(
            "sign test requires at least one paired difference (n=0)".into(),
        ));
    }
    if diffs.iter().any(|d| d.is_nan()) {
        return Err(NumericsError::InvalidInput(
            "sign test requires non-NaN differences (NaN has no sign)".into(),
        ));
    }

    let mut n_pos: usize = 0;
    let mut n_neg: usize = 0;
    let mut ties: usize = 0;
    // Fixed left-to-right fold order over the caller-supplied slice: the
    // classification of any single `d_i` does not depend on any other
    // `d_i`, so this loop order does not affect the resulting counts, but it
    // is pinned explicitly (rather than e.g. `iter().partition()` on an
    // unspecified internal order) so the counting is auditable term-by-term.
    for &d in diffs {
        if d == 0.0 {
            ties += 1;
        } else if d > 0.0 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }

    let n = n_pos + n_neg;
    if n == 0 {
        return Err(NumericsError::InvalidInput(format!(
            "sign test requires at least one non-tied pair; all {ties} difference(s) were exact ties"
        )));
    }

    let p_value = exact_two_sided_tail(n, n_pos.max(n_neg))?;

    Ok(SignTestResult {
        n,
        n_pos,
        n_neg,
        ties,
        p_value,
    })
}

/// `2 * P(X >= t)` under `X ~ Binomial(n, 0.5)`, capped at `1.0`, computed as
/// an exact ratio of `u128` integers with the division to `f64` deferred to
/// the very last step.
///
/// `t = max(n_pos, n_neg)` is symmetric in `n_pos`/`n_neg` by construction
/// (`max(k, n-k) == max(n-k, k)`), so `sign_test` on `diffs` and on
/// `diffs` with every sign flipped agree on `p_value` exactly — this is the
/// two-sidedness symmetry the caller-facing tests pin.
fn exact_two_sided_tail(n: usize, t: usize) -> Result<f64> {
    let row = binomial_row(n)?;
    // `row` sums to exactly `2^n` (the full binomial row) by construction, so
    // `denom` below is recomputed independently from the same closed form
    // rather than by summing `row` again, as a second, cross-checkable
    // derivation of the same quantity. `checked_shl` (rather than a bare
    // `1u128 << n`) makes the `n >= 128` case an explicit refusal instead of
    // Rust's shift-overflow panic.
    let denom: u128 = 1u128.checked_shl(n as u32).ok_or_else(|| {
        NumericsError::InvalidInput(format!(
            "sign test: n={n} too large for exact u128 computation (2^n overflows u128)"
        ))
    })?;
    let tail_sum: u128 = row[t..=n].iter().sum();
    let p_value = match tail_sum.checked_mul(2) {
        Some(numerator) if numerator < denom => numerator as f64 / denom as f64,
        // Either the doubled tail meets or exceeds the full distribution's
        // mass (`numerator >= denom`), or doubling itself overflowed `u128`
        // — which can only happen when `tail_sum` is already more than half
        // of `u128::MAX`, and `tail_sum <= denom` (the row sums to `denom`),
        // so an overflow here is itself proof that `numerator` would have
        // been `>= denom`. Both cases mean the two-sided p-value saturates
        // at the honest cap of `1.0`.
        _ => 1.0,
    };
    Ok(p_value)
}

/// Row `n` of Pascal's triangle, `[C(n,0), C(n,1), ..., C(n,n)]`, computed
/// exactly in `u128` via the multiplicative recurrence `C(n,i) = C(n,i-1) *
/// (n-i+1) / i`. This recurrence is exact integer division at every step —
/// `C(n,i-1) * (n-i+1)` is always evenly divisible by `i`, a standard
/// combinatorial identity — so no rounding is introduced by the `/`, unlike
/// e.g. a `lgamma`-based float approximation of the same coefficient.
///
/// # Overflow
///
/// The *final* row entries are bounded by `2^n`, but the *intermediate*
/// product `C(n,i-1) * (n-i+1)` computed on the way to each entry can exceed
/// the final row's maximum by roughly a factor of `n/2` (it is divided back
/// down by `i` immediately after) — empirically this intermediate product
/// overflows `u128` starting at `n=126`, several rows before `2^n` itself
/// would (`2^n` alone fits up to `n=127`). Using `checked_mul` here (rather
/// than trusting a hand-derived `n < 128` bound on the *final* values) makes
/// the refusal exact at whichever `n` the *intermediate* arithmetic actually
/// overflows, instead of silently wrapping past it.
fn binomial_row(n: usize) -> Result<Vec<u128>> {
    let mut row = Vec::with_capacity(n + 1);
    row.push(1u128); // C(n, 0)
    for i in 1..=n {
        let prev = row[i - 1];
        let numerator = prev.checked_mul((n - i + 1) as u128).ok_or_else(|| {
            NumericsError::InvalidInput(format!(
                "sign test: n={n} too large for exact u128 computation \
                 (intermediate binomial product overflows u128 at term i={i})"
            ))
        })?;
        row.push(numerator / i as u128);
    }
    Ok(row)
}
