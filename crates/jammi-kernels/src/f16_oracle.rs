//! f16 oracle scaffolds for campaign #443's numerics contract (Part 3 / W2a
//! deliverable D4). W2b/W2c fill in the per-op `KO-1`..`KO-8` assertions
//! (see `docs/maintainer/cuda-kernel-guide.md` §3 for the standing
//! checklist and §3.10 for the per-op f16 reference-regime table); this
//! module provides the SHARED primitives every one of those oracles needs,
//! so no op's oracle reinvents its own (and inevitably slightly-different)
//! boundary/ULP logic.
//!
//! ## Two families of helper, two different jobs
//!
//! **Boundary/degenerate contract** (`assert_saturates_to_infinity`,
//! `assert_underflows_to_zero`, `assert_finite_f16`): these are BEHAVIORAL
//! assertions about what f16's own IEEE754 conversion does at a magnitude
//! outside — or at the edge of — its representable range. They are
//! DELIBERATELY never a tolerance comparison against a finite f32
//! reference: past `F16_MAX`, the f32 reference and the f16 arm-under-test
//! no longer share a representable range at all, so "how close is the f16
//! value to the f32 one" is not even a well-formed question there (the f32
//! value is finite; the correct f16 answer is `±inf`). This is the family
//! D domain-validity mandate applied to the 16-bit boundary specifically:
//! pin the actual IEEE754 semantics, not an approximation of them.
//!
//! **ULP-distance / derived-floor helpers** (`ulp_distance_f16`,
//! `f16_ulp_size_at`, `assert_floor_below_f16_gradient_band`): these serve
//! the INTERIOR (non-boundary) parity oracles W2b/W2c write — "how far is
//! the fused f16 arm from the eager f16 (or f32-upcast) reference, at a
//! magnitude where both CAN represent the answer". Per
//! `docs/maintainer/cuda-kernel-guide.md` §3.8 (`KO-3`, "no absolute ULP
//! floor"), a floor here must be DERIVED from f16's own quantization step
//! at the tested magnitude, never a bare integer constant reused across
//! magnitudes, and — the D4-specific addition — never shaped like a
//! COARSER dtype's floor (BF16's ULP is ~8x f16's at the same magnitude;
//! copying a bf16-sized floor into an f16 oracle would silently accept an
//! f16 divergence 8x worse than f16 itself can even represent, hiding
//! exactly the defect class the oracle exists to catch).
//!
//! ## The boundary/degenerate checklist (family D)
//!
//! Every per-op f16 oracle should exercise, at minimum:
//! 1. **Empty input** — the same "zero elements is a no-op, never a panic
//!    or a division-by-zero" contract this crate's F32/BF16 CPU arms
//!    already prove (see e.g. `ops::layer_norm`'s `hidden == 0` tests).
//! 2. **Single point** — the smallest nonempty case; for a reduction op
//!    (layer_norm's variance, softmax's row-sum) this is where a
//!    denominator-of-one or variance-of-zero edge tends to live.
//! 3. **Identical points** — the degenerate zero-variance / zero-spread
//!    case (a constant row into `layer_norm`, an all-equal row into
//!    `softmax`) — distinct from "single point" because it exercises the
//!    reduction machinery at a nontrivial size while still being
//!    analytically exact.
//! 4. **Out-of-range** — a magnitude at or beyond `F16_MAX` (run through
//!    `assert_saturates_to_infinity`), and — for the underflow-to-zero
//!    contract — a magnitude at or below `F16_UNDERFLOW_TIE` (HALF of
//!    `F16_MIN_POSITIVE_SUBNORMAL`, run through `assert_underflows_to_zero`).
//!    A magnitude strictly BETWEEN `F16_UNDERFLOW_TIE` and
//!    `F16_MIN_POSITIVE_SUBNORMAL` is a DIFFERENT boundary case — it rounds
//!    UP to the nonzero subnormal under round-to-nearest-even, not down to
//!    zero (see `F16_UNDERFLOW_TIE`'s own doc) — and must never be run
//!    through `assert_underflows_to_zero`, which now refuses it. Neither
//!    band is covered by the op's ordinary tolerance-vs-reference comparison.

use half::f16;

/// f16's largest finite magnitude (IEEE754 binary16: 5 exponent bits, 10
/// mantissa bits — `(2 - 2^-10) * 2^15`). `no-producer: derived from f16's
/// format, not measured` (see `docs/maintainer/cuda-kernel-guide.md` §3.9's
/// citation convention).
pub const F16_MAX: f32 = 65504.0;

/// f16's smallest positive NORMAL magnitude (`2^-14`). Below this, f16
/// still represents SUBNORMAL values down to [`F16_MIN_POSITIVE_SUBNORMAL`]
/// (a subnormal has no implicit leading-1 mantissa bit, trading precision
/// for a lower floor) — this constant is the normal/subnormal boundary,
/// NOT the underflow-to-zero boundary. `no-producer: derived from f16's
/// format`.
pub const F16_MIN_POSITIVE_NORMAL: f32 = 6.103_515_6e-5; // 2^-14

/// f16's smallest positive SUBNORMAL magnitude (`2^-24`); any nonzero
/// finite value with a smaller magnitude rounds to exact zero. `no-
/// producer: derived from f16's format` (`2^-14 * 2^-10`).
pub const F16_MIN_POSITIVE_SUBNORMAL: f32 = 5.960_464_5e-8; // 2^-24

/// The round-to-zero / round-to-subnormal TIE point for f16's underflow
/// boundary: exactly HALF of [`F16_MIN_POSITIVE_SUBNORMAL`] (`2^-25`).
/// This — NOT `F16_MIN_POSITIVE_SUBNORMAL` itself — is the true domain
/// boundary [`assert_underflows_to_zero`] validates against.
///
/// Under IEEE754 round-to-nearest-EVEN: a magnitude strictly between `0`
/// and this tie rounds DOWN to exact zero; a magnitude strictly between
/// this tie and `F16_MIN_POSITIVE_SUBNORMAL` rounds UP to the NONZERO
/// subnormal (`F16_MIN_POSITIVE_SUBNORMAL` itself) — despite still being
/// smaller than `F16_MIN_POSITIVE_SUBNORMAL`, it does NOT underflow to
/// zero. The exact tie value rounds to zero: the two nearest representable
/// f16 values are `0` and `F16_MIN_POSITIVE_SUBNORMAL`, and zero's mantissa
/// (all bits clear) is the "even" one of the pair, so ties-to-even picks it.
///
/// A phase-4 audit found this crate's stated underflow domain FALSE: an
/// earlier revision of [`assert_underflows_to_zero`] (and this module's own
/// checklist above, and two `tests/cuda_parity.rs` fixtures) claimed the
/// boundary was "at or below `F16_MIN_POSITIVE_SUBNORMAL`" — which silently
/// mis-describes the entire `(F16_UNDERFLOW_TIE, F16_MIN_POSITIVE_SUBNORMAL)`
/// half of that band, where the true IEEE754 outcome is a nonzero subnormal,
/// not zero. `no-producer: derived from f16's format (half of
/// F16_MIN_POSITIVE_SUBNORMAL's own derivation)`.
pub const F16_UNDERFLOW_TIE: f32 = F16_MIN_POSITIVE_SUBNORMAL / 2.0; // 2^-25

/// BEHAVIORAL saturation contract: an `x` STRICTLY BEYOND [`F16_MAX`] in
/// magnitude converts to `±inf` under round-to-nearest (IEEE754 binary16
/// overflow), NEVER a clamped finite value and never a silently wrong
/// number. Note `F16_MAX` ITSELF is finite and exactly representable (it
/// is the largest finite f16, not a saturation trigger) — this contract
/// is about magnitudes beyond it. Panics if `x` is not actually beyond the
/// boundary (a caller error, not a test failure signal) or if the
/// conversion fails to saturate, or saturates with the wrong sign.
pub fn assert_saturates_to_infinity(x: f32) {
    assert!(
        x.abs() > F16_MAX,
        "assert_saturates_to_infinity called on a magnitude ({x}) at or inside f16's finite \
         range (F16_MAX = {F16_MAX}) — this is a boundary assertion for magnitudes STRICTLY \
         BEYOND F16_MAX, not an interior (or exactly-F16_MAX) one; use the op's ordinary \
         tolerance oracle instead"
    );
    let h = f16::from_f32(x);
    assert!(
        h.is_infinite(),
        "expected f16 saturation to +/-inf at magnitude {x} (>= F16_MAX = {F16_MAX}), got \
         {h:?} instead -- a confident WRONG finite number at the boundary is exactly the family \
         D failure mode this oracle exists to catch, not a saturation"
    );
    assert_eq!(
        h.is_sign_negative(),
        x.is_sign_negative(),
        "saturation to infinity must preserve the input's sign (got {h:?} from input {x})"
    );
}

/// BEHAVIORAL underflow contract: an `x` with magnitude strictly greater
/// than `0` and AT OR BELOW [`F16_UNDERFLOW_TIE`] (half of
/// [`F16_MIN_POSITIVE_SUBNORMAL`]) rounds to EXACT zero under
/// round-to-nearest-even (never a subnormal wraparound, never a sign
/// flip) — the mirror-image boundary case to [`assert_saturates_to_infinity`]
/// at the small-magnitude end. A magnitude strictly ABOVE the tie but still
/// below `F16_MIN_POSITIVE_SUBNORMAL` is OUTSIDE this contract's domain —
/// see [`F16_UNDERFLOW_TIE`]'s doc: it rounds UP to the nonzero subnormal,
/// not to zero, and this function refuses (panics on) that input rather
/// than silently asserting the wrong outcome for it.
pub fn assert_underflows_to_zero(x: f32) {
    assert!(
        x != 0.0 && x.abs() <= F16_UNDERFLOW_TIE,
        "assert_underflows_to_zero called on a magnitude ({x}) outside its TRUE domain: \
         nonzero and at or below the round-to-zero/round-to-subnormal tie \
         (F16_UNDERFLOW_TIE = {F16_UNDERFLOW_TIE} = F16_MIN_POSITIVE_SUBNORMAL/2) -- a magnitude \
         in (F16_UNDERFLOW_TIE, F16_MIN_POSITIVE_SUBNORMAL) rounds UP to the nonzero subnormal \
         under round-to-nearest-even, not to zero (see F16_UNDERFLOW_TIE's own doc)"
    );
    let h = f16::from_f32(x);
    assert_eq!(
        h,
        f16::ZERO,
        "expected exact zero (underflow) at magnitude {x}, got {h:?} instead"
    );
    assert_eq!(
        h.is_sign_negative(),
        x.is_sign_negative(),
        "underflow-to-zero must preserve the input's sign (IEEE754 signed zero); got {h:?} from \
         input {x}"
    );
}

/// Non-finite detection (family F: a claimed numeric guarantee is asserted
/// against a NON-VACUOUS negative control, and every control must fail on
/// EVERY bad path including non-finite — `NaN > c` is `false` in IEEE754,
/// so a naive `assert!(!(x > bound))`-style comparison silently "passes"
/// on a `NaN`; see `docs/maintainer/cuda-kernel-guide.md` §3.7's "write
/// comparisons affirmatively" rule). `context` is folded into the panic
/// message so a failure names the call site, not just "some f16 value was
/// non-finite".
pub fn assert_finite_f16(h: f16, context: &str) {
    assert!(
        h.is_finite(),
        "{context}: expected a finite f16 value, got {h:?} (is_nan={}, is_infinite={})",
        h.is_nan(),
        h.is_infinite()
    );
}

/// Maps an f16 bit pattern to a signed, monotonically-ordered integer key
/// (the standard sign-magnitude -> two's-complement-ordering transform,
/// Bruce Dawson's "Comparing Floating Point Numbers" — the SAME idiom this
/// crate's existing f32/bf16 ULP comparisons in `tests/oracles.rs` use,
/// applied at f16's 16-bit width): `+0.0` and `-0.0` both map to `0`, and
/// adjacent representable f16 values map to adjacent integers, so a plain
/// integer subtraction IS the ULP distance.
fn ordered_key_f16(h: f16) -> i32 {
    let bits = h.to_bits() as i16; // reinterpret the bit pattern, two's complement
    let ordered = if bits < 0 {
        i16::MIN.wrapping_sub(bits)
    } else {
        bits
    };
    ordered as i32
}

/// ULP distance between two f16 values, as a non-negative integer count of
/// representable steps. NOT an absolute-value floor by itself (KO-3): a
/// caller comparing this against a threshold must derive that threshold
/// from f16's own quantization step at the tested magnitude — see
/// [`f16_ulp_size_at`] / [`assert_floor_below_f16_gradient_band`] below.
/// `NaN` inputs have no ordered position; this function panics on a `NaN`
/// argument rather than returning a number that reads as a real distance.
pub fn ulp_distance_f16(a: f16, b: f16) -> i32 {
    assert!(
        !a.is_nan() && !b.is_nan(),
        "ulp_distance_f16 is undefined for NaN (a={a:?}, b={b:?})"
    );
    (ordered_key_f16(a) - ordered_key_f16(b)).abs()
}

/// The size, in f32 units, of ONE f16 ULP step at `magnitude` — i.e. the
/// smallest step f16 can represent near that value. Returns the SIZE (not
/// a bare ULP count) so a caller can state its own floor in whichever
/// unit its oracle already reports in: a floor of `k` ULPs means a
/// different absolute gap at `magnitude = 1.0` than at `magnitude =
/// 1000.0`, so a floor expressed as a raw integer copied across
/// magnitudes is exactly the KO-3 "absolute ULP floor" anti-pattern this
/// helper exists to avoid.
pub fn f16_ulp_size_at(magnitude: f32) -> f32 {
    let h = f16::from_f32(magnitude);
    let next = f16::from_bits(h.to_bits().wrapping_add(1));
    (next.to_f32() - h.to_f32()).abs()
}

/// Asserts that a proposed ULP floor sits STRICTLY below f16's own
/// quantization step scaled up by BF16's ULP-to-F16-ULP ratio at
/// `typical_magnitude` (BF16 has 7 mantissa bits vs f16's 10, so BF16's
/// ULP is `2^3 = 8x` coarser than f16's at the same magnitude) — the
/// concrete, checkable form of "no bf16-shaped absolute floor" this
/// crate's D4 deliverable requires. A floor at or above this line would
/// silently accept an f16 divergence at least as coarse as bf16's own
/// rounding noise, hiding exactly the class of f16-specific defect an f16
/// oracle exists to catch (rather than merely re-measuring bf16-scale
/// noise under a different dtype label).
pub fn assert_floor_below_f16_gradient_band(floor_ulps: i32, typical_magnitude: f32) {
    const BF16_TO_F16_ULP_RATIO: f32 = 8.0; // no-producer: derived (2^(10-7))
    let one_f16_ulp = f16_ulp_size_at(typical_magnitude);
    let floor_size = floor_ulps as f32 * one_f16_ulp;
    let bf16_shaped_size = BF16_TO_F16_ULP_RATIO * one_f16_ulp;
    assert!(
        floor_size < bf16_shaped_size,
        "floor ({floor_ulps} f16 ULPs = {floor_size} at magnitude {typical_magnitude}) is at \
         or beyond a bf16-shaped floor ({bf16_shaped_size}, i.e. {BF16_TO_F16_ULP_RATIO}x f16's \
         own ULP at this magnitude) -- re-derive this floor from f16's own quantization step, \
         not from bf16's coarser one"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_max_matches_the_ieee754_binary16_finite_ceiling() {
        // (2 - 2^-10) * 2^15, computed independently of the module's own
        // constant (a numpy-first-style independent recomputation, family
        // F): the largest finite f16 has all-ones exponent-minus-one and
        // all-ones mantissa.
        let expected = (2.0 - 2f64.powi(-10)) * 2f64.powi(15);
        assert!(
            (F16_MAX as f64 - expected).abs() < 1e-6,
            "{F16_MAX} vs {expected}"
        );
        // And it really is the largest FINITE f16 -- one ULP more overflows.
        assert!(f16::from_f32(F16_MAX).is_finite());
        assert!(f16::from_f32(F16_MAX * 1.001).is_infinite());
    }

    #[test]
    fn saturates_to_infinity_at_and_beyond_the_boundary_both_signs() {
        // A tiny nudge past F16_MAX (still well inside f32's own finite
        // range) — F16_MAX itself is finite and exactly representable, so
        // it deliberately is NOT used as a saturation-trigger fixture here.
        assert_saturates_to_infinity(F16_MAX * 1.001);
        assert_saturates_to_infinity(-F16_MAX * 1.001);
        assert_saturates_to_infinity(1.0e6);
        assert_saturates_to_infinity(-1.0e6);
        assert_saturates_to_infinity(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "boundary assertion")]
    fn saturates_to_infinity_refuses_an_interior_magnitude() {
        assert_saturates_to_infinity(1.0);
    }

    #[test]
    fn underflows_to_zero_below_the_subnormal_floor_both_signs() {
        assert_underflows_to_zero(F16_UNDERFLOW_TIE * 0.99);
        assert_underflows_to_zero(-F16_UNDERFLOW_TIE * 0.99);
        assert_underflows_to_zero(1.0e-30);
    }

    /// The exact tie ([`F16_UNDERFLOW_TIE`]) itself rounds to zero
    /// (ties-to-even picks the zero mantissa) — pinned on its own since it
    /// is the boundary VALUE itself, not merely a value close to it.
    #[test]
    fn underflows_to_zero_at_the_exact_tie_both_signs() {
        assert_underflows_to_zero(F16_UNDERFLOW_TIE);
        assert_underflows_to_zero(-F16_UNDERFLOW_TIE);
    }

    /// The band boundary, pinned from BOTH sides (phase-4 audit fix): a
    /// magnitude strictly ABOVE the tie but still strictly BELOW
    /// `F16_MIN_POSITIVE_SUBNORMAL` rounds UP to the nonzero subnormal, not
    /// down to zero — the exact band the old "at or below
    /// F16_MIN_POSITIVE_SUBNORMAL" domain claim got wrong. Two halves: (a)
    /// `assert_underflows_to_zero` must now REFUSE this input (outside its
    /// documented domain); (b) the actual f16 conversion in this band is
    /// independently checked NONZERO, both signs, so "rounds to the
    /// subnormal, not zero" is demonstrated, not merely asserted.
    #[test]
    fn magnitude_between_tie_and_full_subnormal_rounds_up_not_to_zero() {
        let x = (F16_UNDERFLOW_TIE + F16_MIN_POSITIVE_SUBNORMAL) / 2.0; // midpoint of the open band
        assert!(
            x > F16_UNDERFLOW_TIE && x < F16_MIN_POSITIVE_SUBNORMAL,
            "fixture invariant: x must sit strictly inside the (tie, full subnormal) band"
        );

        let r = std::panic::catch_unwind(|| assert_underflows_to_zero(x));
        assert!(
            r.is_err(),
            "assert_underflows_to_zero must refuse a magnitude in the round-UP band, not \
             silently accept it"
        );
        let r_neg = std::panic::catch_unwind(|| assert_underflows_to_zero(-x));
        assert!(r_neg.is_err());

        let h = f16::from_f32(x);
        assert_ne!(
            h,
            f16::ZERO,
            "expected a nonzero subnormal in the round-UP band, got exact zero"
        );
        assert!(h.is_sign_positive());
        let h_neg = f16::from_f32(-x);
        assert_ne!(h_neg, f16::ZERO);
        assert!(h_neg.is_sign_negative());
    }

    #[test]
    fn finite_f16_accepts_finite_and_rejects_nan_and_infinite() {
        assert_finite_f16(f16::from_f32(1.5), "ctx");
        let r = std::panic::catch_unwind(|| assert_finite_f16(f16::NAN, "ctx"));
        assert!(r.is_err(), "NaN must be caught, not silently pass");
        let r = std::panic::catch_unwind(|| assert_finite_f16(f16::INFINITY, "ctx"));
        assert!(r.is_err(), "+inf must be caught, not silently pass");
    }

    #[test]
    fn ulp_distance_is_zero_for_equal_values_and_positive_for_neighbours() {
        let a = f16::from_f32(1.0);
        assert_eq!(ulp_distance_f16(a, a), 0);
        let b = f16::from_bits(a.to_bits().wrapping_add(1));
        assert_eq!(ulp_distance_f16(a, b), 1);
        assert_eq!(ulp_distance_f16(b, a), 1, "distance must be symmetric");
    }

    #[test]
    fn ulp_distance_treats_signed_zero_as_zero_distance() {
        assert_eq!(ulp_distance_f16(f16::from_f32(0.0), f16::from_f32(-0.0)), 0);
    }

    #[test]
    #[should_panic(expected = "undefined for NaN")]
    fn ulp_distance_refuses_nan() {
        ulp_distance_f16(f16::NAN, f16::from_f32(1.0));
    }

    #[test]
    fn ulp_size_grows_with_magnitude_matching_f16s_floating_exponent() {
        let small = f16_ulp_size_at(1.0);
        let large = f16_ulp_size_at(1000.0);
        assert!(
            large > small,
            "f16's ULP step must grow with magnitude (floating-point, not fixed-point): \
             {large} vs {small}"
        );
    }

    #[test]
    fn floor_below_bf16_shaped_band_passes_a_genuinely_f16_sized_floor() {
        // 1 f16 ULP is always < 8x f16 ULPs (the bf16-shaped line).
        assert_floor_below_f16_gradient_band(1, 100.0);
    }

    #[test]
    #[should_panic(expected = "bf16-shaped floor")]
    fn floor_below_bf16_shaped_band_rejects_a_bf16_sized_floor() {
        assert_floor_below_f16_gradient_band(8, 100.0);
    }
}
