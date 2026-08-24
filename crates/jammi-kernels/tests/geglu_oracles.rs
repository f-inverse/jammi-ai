//! CPU-hermetic oracles for `GegluFused` — the same rigor-chain pattern
//! `tests/layer_norm_oracles.rs` establishes, extended for a `CustomOp1`
//! whose backward dispatches into ONE internal `CustomOp2` helper (see
//! `jammi_kernels::ops::geglu`'s module doc).
//!
//!   1. `gradcheck_*` — `bwd` vs. central finite differences, at points
//!      spanning negative `gate`, near-zero `gate`, and `|gate| > 3`
//!      tails — the region where `gelu_erf'`'s two terms partially cancel
//!      and a sign error is likeliest to hide.
//!   2. `eager_vs_fused_*` — fwd+bwd vs. the exact eager composition
//!      (`narrow`+`narrow`+`gelu_erf`+`mul`, reproduced candle-op-for-
//!      candle-op — this crate is a LEAF with no `jammi-encoders` dep, so
//!      it cannot import `ModernBertMlp::forward` itself; the "against the
//!      real call site" oracle lives in `jammi-encoders`' `tests/it` suite
//!      where that function is reachable), f32 tight and bf16 measured-
//!      nonzero + non-uniform `dy`.
//!   3. `chain_rule_through_an_intermediate_wi_out_matches_eager` — the
//!      construction-data / frozen-input regression shape every other
//!      op's oracle suite in this crate carries: `wi_out` is an
//!      INTERMEDIATE (`is_variable() == false`) on a path to a `Var`, and
//!      `bwd`'s `dwi_out` slot must still populate correctly (it is
//!      ALWAYS `Some`, never gated on `is_variable()`).
//!
//! The CUDA↔CPU parity leg (fwd + bwd, contiguous/narrowed/empty/large-n
//! multi-block, bf16+f32) lives in `tests/cuda_parity.rs`, gated the same
//! way every other op's is.

use candle_core::{Device, Tensor, Var, D};
use half::bf16;
use jammi_kernels::ops::{apply1, GegluFused, GeluVariant};

fn fused(wi_out: &Tensor) -> candle_core::Result<Tensor> {
    apply1(wi_out, GegluFused::new(GeluVariant::Erf))
}

/// The exact eager composition `ModernBertMlp::forward`'s eval arm (and
/// `geglu_apply_training`'s eager-fallback arm) runs: narrow into
/// `gate`/`up` (`gate` FIRST, `up` SECOND — the split convention this
/// op's own module doc pins), `gate.gelu_erf()?`, then multiply by `up`.
fn eager(wi_out: &Tensor) -> candle_core::Result<Tensor> {
    let intermediate = wi_out.dim(D::Minus1)? / 2;
    let gate = wi_out.narrow(D::Minus1, 0, intermediate)?;
    let up = wi_out.narrow(D::Minus1, intermediate, intermediate)?;
    (gate.gelu_erf()? * up)?.contiguous()
}

// ---------------------------------------------------------------------
// Oracle 1: gradcheck vs. central finite differences
// ---------------------------------------------------------------------

#[test]
fn gradcheck_dwi_out_f32_spans_negative_near_zero_and_tail_gate_values() {
    let device = Device::Cpu;
    // gate values: negative, near-zero, positive, |gate|>3 tail (both
    // signs) — the region where gelu_erf'(x) = Phi(x) + x*phi(x)'s two
    // terms partially cancel and a sign error is likeliest to hide.
    let gate0: [f32; 6] = [-3.5, -0.5, -0.02, 0.3, 1.2, 4.0];
    let up0: [f32; 6] = [0.7, -1.3, 2.0, -0.4, 3.0, -0.2];
    let intermediate = gate0.len();
    let rows = 1;
    let mut wi0 = gate0.to_vec();
    wi0.extend_from_slice(&up0);

    let wi =
        Var::from_tensor(&Tensor::from_slice(&wi0, (rows, 2 * intermediate), &device).unwrap())
            .unwrap();
    let out = fused(&wi).unwrap();
    let grads = out.backward().unwrap();
    let dwi: Vec<f32> = grads
        .get(&wi)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let sum_fwd = |wi: &Tensor| -> f64 {
        fused(wi)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    let fd_eps = 2e-3f32;
    let tol = 5e-2f64;
    for i in 0..wi0.len() {
        let mut wp = wi0.clone();
        wp[i] += fd_eps;
        let mut wm = wi0.clone();
        wm[i] -= fd_eps;
        let wp_t = Tensor::from_slice(&wp, (rows, 2 * intermediate), &device).unwrap();
        let wm_t = Tensor::from_slice(&wm, (rows, 2 * intermediate), &device).unwrap();
        let numeric = (sum_fwd(&wp_t) - sum_fwd(&wm_t)) / (2.0 * fd_eps as f64);
        assert!(
            (numeric - dwi[i] as f64).abs() < tol,
            "dwi_out[{i}]: numeric {numeric} vs analytic {} (gate0={gate0:?}, up0={up0:?})",
            dwi[i]
        );
    }
}

// ---------------------------------------------------------------------
// Oracle 2: fused vs. eager (candle composition), fwd AND bwd
// ---------------------------------------------------------------------

#[test]
fn eager_vs_fused_f32_fwd_and_bwd_match_within_stated_tolerance() {
    let device = Device::Cpu;
    let v: [f32; 12] = [
        1.0, -2.0, 3.5, -0.25, 0.1, 2.2, -1.1, 0.75, 0.05, -3.2, 2.9, -0.6,
    ];
    let intermediate = v.len() / 2;

    let wi_f =
        Var::from_tensor(&Tensor::from_slice(&v, (1, 2 * intermediate), &device).unwrap()).unwrap();
    let wi_e =
        Var::from_tensor(&Tensor::from_slice(&v, (1, 2 * intermediate), &device).unwrap()).unwrap();

    let out_f = fused(&wi_f).unwrap();
    let out_e = eager(&wi_e).unwrap();
    let vf: Vec<f32> = out_f.flatten_all().unwrap().to_vec1().unwrap();
    let ve: Vec<f32> = out_e.flatten_all().unwrap().to_vec1().unwrap();
    for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
        assert!((f - e).abs() < 1e-5, "fwd[{i}]: fused {f} vs eager {e}");
    }

    // Non-uniform dy (not `backward()`'s default `ones_like` seed): a
    // uniform seed would make `d_up = dy*gelu_val` and `d_gate =
    // dy*up*gelu_deriv` trivially proportional to a constant, masking any
    // bug that only shows up when `dy` varies per element.
    let w0: [f32; 6] = [1.703125, -2.015625, 2.234375, 0.1, -1.5, 3.375];
    let w_f = Tensor::from_slice(&w0, (1, intermediate), &device).unwrap();
    let w_e = Tensor::from_slice(&w0, (1, intermediate), &device).unwrap();
    let loss_f = (&out_f * &w_f).unwrap().sum_all().unwrap();
    let loss_e = (&out_e * &w_e).unwrap().sum_all().unwrap();
    let grads_f = loss_f.backward().unwrap();
    let grads_e = loss_e.backward().unwrap();
    let dwf: Vec<f32> = grads_f
        .get(&wi_f)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dwe: Vec<f32> = grads_e
        .get(&wi_e)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_bwd_diff = dwf
        .iter()
        .zip(dwe.iter())
        .map(|(f, e)| (f - e).abs())
        .fold(0.0f32, f32::max);
    // MEASURED, not assumed: this residual is EXPECTED to be nonzero even
    // in f32, by construction — see `ops::geglu`'s module doc's "backward
    // derivation" section for why candle's own autodiff (`backprop.rs`'s
    // truncated `0.398942` literal vs. this kernel's full-precision
    // `1/sqrt(2*pi)`) can never be bit-exact against this kernel, and how
    // that systematic offset quantitatively explains the residual printed
    // here.
    println!("eager_vs_fused_f32_fwd_and_bwd: measured max |dwi_out diff| = {max_bwd_diff}");
    for (i, (f, e)) in dwf.iter().zip(dwe.iter()).enumerate() {
        assert!((f - e).abs() < 1e-3, "dwi_out[{i}]: fused {f} vs eager {e}");
    }
}

/// Bit-pattern difference, used ONLY for disclosure (`println!`) below —
/// see the forward oracle's doc for why the actual PASS/FAIL bound is
/// RELATIVE (with an absolute floor), not a raw bit-diff/ULP count, for
/// this op's value range.
fn bf16_bit_diff(a: bf16, b: bf16) -> i32 {
    a.to_bits() as i32 - b.to_bits() as i32
}

/// The PASS/FAIL metric both bf16 oracles below use: RELATIVE tolerance
/// with an absolute FLOOR, not a flat scale-free absolute bound (a prior
/// version of this file used `BF16_FWD_ABS_TOL = 1e-3` /
/// `BF16_BWD_ABS_TOL = 0.3`, both flat absolutes — an audit finding: a
/// flat absolute bound is scale-free, so at this op's larger magnitudes
/// (`|dwi_out|` up to ~16 in the production-width backward fixture) `0.3`
/// permits close to a 2% RELATIVE error there, wide enough that a real
/// regression (e.g. a dropped `* 0.5` in the CDF) could plausibly still
/// pass, while ALSO being needlessly loose at the near-zero tail this
/// op's own value range actually produces).
///
/// Both constants are sized from a FULL scan of the backward oracle's
/// production-width fixture (2026-08-24 audit + follow-up measurement),
/// not from one cherry-picked element, because this op's bf16 backward
/// divergence has TWO DIFFERENT mechanisms with different scales:
///
/// 1. Ordinary rounding-order divergence at non-trivial magnitude: the
///    worst such element measured is `2` bf16 ULP (`0.125` absolute) at
///    `|value| ~= 12.3` (`0.125 / 12.3 ~= 1.02%` relative). bf16 has 7
///    explicit mantissa bits, so within any octave `[2^e, 2^(e+1))` the
///    ULP is the constant `2^(e-7)`, and `ULP / value` is LARGEST (worst
///    case for a relative bound) at the BOTTOM of the octave, where it
///    equals exactly `2^-7` regardless of `e` — a `2`-ULP divergence
///    therefore needs `REL >= 2 * 2^-7 = 2^-6 = 1.5625%` to stay covered
///    EVERYWHERE in bf16's range, not merely at this one fixture's own
///    worst point.
/// 2. Eager's own rounding CASCADE (see the module doc's "BF16 backward
///    rounding" section: roughly half a dozen separately-rounded `Tensor`
///    ops) occasionally rounds a small intermediate all the way DOWN TO
///    EXACT bf16 zero at one of those steps, where this kernel's single
///    f32-then-round-once path still resolves a small but genuinely
///    nonzero value. This is NOT well described by a relative bound at
///    all (one side is exactly `0`, so relative error is undefined/
///    unbounded) — measured max `|abs diff|` across every element where
///    EITHER side is exactly zero: `0.0120` (`BF16_ABS_FLOOR = 2^-5 =
///    0.03125` keeps ~2.6x headroom over that).
///
/// `REL = 2^-6` alone is jointly sufficient with `FLOOR = 2^-5` for every
/// element in the production-width fixture (verified directly, not just
/// argued): the maximum `(|diff| - FLOOR) / magnitude` needed across every
/// nonzero-magnitude element, GIVEN `FLOOR = 2^-5`, measures `1.06%` —
/// under `REL`'s `1.5625%` with real margin.
const BF16_REL_TOL: f32 = 0.015625; // 2^-6

/// See `BF16_REL_TOL`'s doc, mechanism 2: exists for elements where
/// EITHER compared value is exactly bf16 zero (eager's rounding cascade
/// underflowing a small intermediate; this kernel's single-rounding path
/// resolving a small nonzero value there instead) — a case a pure
/// relative bound cannot describe at all. Also comfortably covers the
/// forward oracle's own (much smaller) near-zero-tail divergence
/// (measured max `~9.54e-7`, i.e. `2^-20`).
const BF16_ABS_FLOOR: f32 = 0.03125; // 2^-5

/// The shared bf16 comparison both oracles below assert against. See
/// `BF16_REL_TOL`'s doc for the derivation.
fn bf16_close(a: bf16, b: bf16) -> bool {
    let (af, bf) = (a.to_f32(), b.to_f32());
    (af - bf).abs() <= BF16_REL_TOL * af.abs().max(bf.abs()) + BF16_ABS_FLOOR
}

/// bf16's ULP (spacing between adjacent representable values) AT
/// magnitude `x`: within the octave `[2^e, 2^(e+1))`, bf16 (7 explicit
/// mantissa bits) spaces its representable values `2^(e-7)` apart. Used
/// ONLY for the diagnostic `println!`s below (disclosure, not a pass/fail
/// bound — `bf16_close` is that) so a raw "0.125" divergence reads with
/// its actual ULP-count context rather than as a bare number.
fn bf16_ulp_at(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    2f32.powi(x.log2().floor() as i32 - 7)
}

/// BF16 forward: measured, non-vacuous divergence from eager (this op
/// computes the activation in f32 before rounding; candle's own eager
/// `gelu_erf()?` on a bf16 tensor computes in f64 — see the module doc's
/// "bf16 boundary-rounding" section) — a small, stated relative-with-
/// absolute-floor tolerance (`bf16_close`), not an assumed-zero one.
#[test]
fn eager_vs_fused_bf16_fwd_diverges_and_stays_within_the_stated_tolerance() {
    let device = Device::Cpu;
    let intermediate = 2624usize;
    let rows = 2usize;
    let n = rows * 2 * intermediate;
    let v: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f32 * 0.017;
            (x.sin() * 6.0) + (0.3 * (i as f32 * 0.0031).cos())
        })
        .collect();
    let vb: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    let wi = Tensor::from_slice(&vb, (rows, 2 * intermediate), &device).unwrap();

    let fused_out: Vec<bf16> = fused(&wi)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let eager_out: Vec<bf16> = eager(&wi)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let diffs: Vec<i32> = fused_out
        .iter()
        .zip(eager_out.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    let max_diff = diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
    let nonzero = diffs.iter().filter(|&&d| d != 0).count();
    let max_abs: f32 = fused_out
        .iter()
        .zip(eager_out.iter())
        .map(|(&f, &e)| (f.to_f32() - e.to_f32()).abs())
        .fold(0.0, f32::max);
    println!(
        "eager_vs_fused_bf16_fwd (production width, intermediate={intermediate}, \
         n={n}): measured max |bit-diff| = {max_diff}, max |abs diff| = {max_abs}, \
         {nonzero}/{n} elements differ"
    );
    // NON-VACUITY, honestly measured rather than assumed: at BF16's coarse
    // (~8-bit-mantissa) precision, this op's f32-computed activation and
    // candle's own f64-computed activation (see the module doc's "bf16
    // boundary-rounding" section) round to the SAME bf16 value for the
    // overwhelming majority of inputs — the f32-vs-f64 gap is far smaller
    // than one bf16 ULP almost everywhere. A real divergence exists (this
    // assertion requires at least one), but it is RARE, not pervasive;
    // this is the measured reality, not a guess, and it is disclosed here
    // rather than silently dropping the tolerance-oracle shape.
    assert!(
        nonzero > 0,
        "expected at least one element to diverge across {n} elements at \
         production width — the tolerance is not being exercised at all"
    );
    // RELATIVE tolerance with an absolute floor (`bf16_close`, see its own
    // doc), not a raw bit-diff/ULP count: this op's output can be
    // arbitrarily close to zero (`gelu_erf(gate) -> 0` as `gate -> -inf`,
    // times any `up`), and a bf16 BIT-PATTERN diff near zero is a
    // degenerate metric there (crossing between exponent ranges near zero
    // inflates the raw bit-diff by many orders of magnitude for a
    // numerically tiny VALUE difference — MEASURED on this exact fixture:
    // max |bit-diff| = 13554 ULP, but the actual max |value diff| at that
    // SAME element is only ~9.54e-7, i.e. `2^-20`, comfortably inside
    // `BF16_ABS_FLOOR`). `LayerNormFused`/`SoftmaxLastDimFused`'s ULP
    // bounds are sound for THEIR outputs (never structurally near zero the
    // way a GeGLU tail is); this op's own forward oracle uses the metric
    // that is actually meaningful for its own value range.
    for (i, (&f, &e)) in fused_out.iter().zip(eager_out.iter()).enumerate() {
        assert!(
            bf16_close(f, e),
            "element {i}: |fused {f} - eager {e}| = {} exceeds BF16_REL_TOL*max(|a|,|b|) \
             + BF16_ABS_FLOOR",
            (f.to_f32() - e.to_f32()).abs()
        );
    }
}

/// BF16 backward: same non-vacuous-divergence shape, for the DIFFERENT
/// reason stated in the module doc (this op's bf16 backward deliberately
/// does not reproduce eager's multi-op rounding cascade).
///
/// Ships at ModernBERT-large's actual production width
/// (`intermediate = 2624`, per HuggingFace's published
/// `answerdotai/ModernBERT-large` `config.json`) — the fused-kernels
/// contract requires measuring the bf16 bound at production width, not a
/// toy size, since a rounding-order divergence's measured magnitude can
/// depend on how many elements are summed into the loss.
#[test]
fn eager_vs_fused_bf16_bwd_diverges_and_stays_within_the_stated_tolerance_at_production_width() {
    let device = Device::Cpu;
    let intermediate = 2624usize; // ModernBERT-large's real intermediate_size.
    let rows = 2usize;
    let n = rows * 2 * intermediate;

    // Deterministic, non-degenerate fixture spanning a wide range of
    // `gate` magnitudes (negative/near-zero/positive/tails) without a
    // uniform pattern that would mask rounding-order effects.
    let v: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f32 * 0.017;
            (x.sin() * 6.0) + (0.3 * (i as f32 * 0.0031).cos())
        })
        .collect();
    let vb: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    let w0: Vec<f32> = (0..rows * intermediate)
        .map(|i| ((i as f32 * 0.023).cos() * 2.5) + 0.1)
        .collect();
    let wb: Vec<bf16> = w0.iter().map(|&x| bf16::from_f32(x)).collect();

    let wi_f =
        Var::from_tensor(&Tensor::from_slice(&vb, (rows, 2 * intermediate), &device).unwrap())
            .unwrap();
    let wi_e =
        Var::from_tensor(&Tensor::from_slice(&vb, (rows, 2 * intermediate), &device).unwrap())
            .unwrap();
    let w_f = Tensor::from_slice(&wb, (rows, intermediate), &device).unwrap();
    let w_e = Tensor::from_slice(&wb, (rows, intermediate), &device).unwrap();

    let out_f = fused(&wi_f).unwrap();
    let out_e = eager(&wi_e).unwrap();
    let loss_f = (&out_f * &w_f).unwrap().sum_all().unwrap();
    let loss_e = (&out_e * &w_e).unwrap().sum_all().unwrap();
    let grads_f = loss_f.backward().unwrap();
    let grads_e = loss_e.backward().unwrap();

    let dwf: Vec<bf16> = grads_f
        .get(&wi_f)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dwe: Vec<bf16> = grads_e
        .get(&wi_e)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let abs_diffs: Vec<f32> = dwf
        .iter()
        .zip(dwe.iter())
        .map(|(&f, &e)| (f.to_f32() - e.to_f32()).abs())
        .collect();
    let max_abs = abs_diffs.iter().cloned().fold(0.0f32, f32::max);
    let max_val = dwf
        .iter()
        .chain(dwe.iter())
        .map(|v| v.to_f32().abs())
        .fold(0.0f32, f32::max);
    let (argmax_i, _) = abs_diffs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let argmax_mag = dwf[argmax_i]
        .to_f32()
        .abs()
        .max(dwe[argmax_i].to_f32().abs());
    let argmax_ulp = bf16_ulp_at(argmax_mag);
    // MAGNITUDE CONTEXT (an audit finding: a bare "0.125" reads as a large
    // number out of context — it is 1 bf16 ULP at this fixture's own peak
    // magnitude, 16, and the fixture's ACTUAL worst element sits at a
    // smaller magnitude, ~12.2-12.3, where the same 0.125 absolute diff is
    // exactly 2 bf16 ULP (`BF16_REL_TOL`'s doc derives why `2^-6`, not
    // `2^-7`, is the bound sized to cover that). Both facts, printed for
    // the record on every run:
    println!(
        "eager_vs_fused_bf16_bwd (production width, intermediate={intermediate}): \
         measured max |abs diff| = {max_abs} at element {argmax_i} (fused={}, eager={}, \
         magnitude ~{argmax_mag:.3}, bf16 ULP there = {argmax_ulp:.4} => \
         {:.2} ULP); fixture peak magnitude (either side) = {max_val} \
         (1 bf16 ULP there = {:.4})",
        dwf[argmax_i].to_f32(),
        dwe[argmax_i].to_f32(),
        max_abs / argmax_ulp,
        bf16_ulp_at(max_val),
    );
    // Non-vacuity, honestly measured rather than assumed: a real
    // divergence exists (this assertion requires at least one), disclosed
    // above with its exact magnitude rather than a guess.
    assert!(
        max_abs > 0.0,
        "expected fixture to diverge at production width — the tolerance is not \
         being exercised"
    );
    // RELATIVE tolerance with an absolute floor (`bf16_close`), sized in
    // its own doc from a FULL scan of this exact fixture across BOTH
    // divergence mechanisms it can exhibit: ordinary rounding-order
    // divergence at non-trivial magnitude (up to 2 bf16 ULP, covered by
    // `BF16_REL_TOL`), and eager's rounding CASCADE (roughly half a dozen
    // separately-rounded `Tensor` ops — see the module doc's "BF16
    // backward rounding" section) occasionally underflowing a small
    // intermediate to EXACT bf16 zero where this kernel's single-rounding
    // path still resolves a small nonzero value (covered by
    // `BF16_ABS_FLOOR`) — a real, disclosed, mechanistically-understood
    // divergence, not a red flag.
    for (i, (&f, &e)) in dwf.iter().zip(dwe.iter()).enumerate() {
        assert!(
            bf16_close(f, e),
            "dwi_out[{i}]: |fused {f} - eager {e}| = {} exceeds BF16_REL_TOL*max(|a|,|b|) \
             + BF16_ABS_FLOOR (fixture max |abs diff| was {max_abs})",
            (f.to_f32() - e.to_f32()).abs()
        );
    }
}

// ---------------------------------------------------------------------
// Oracle 3: chain-rule through an intermediate wi_out
// ---------------------------------------------------------------------

#[test]
fn chain_rule_through_an_intermediate_wi_out_matches_eager() {
    // `wi_out` is an INTERMEDIATE (`w.affine(1.5, 0.2)`) on a path to a
    // `Var` (`w`) — `is_variable() == false`, the same regression shape
    // `Axpy`/`LayerNormFused`/`SoftmaxLastDimFused` all carry. `bwd`'s
    // `dwi_out` slot must still populate (it is ALWAYS `Some`, never
    // gated on `is_variable()`), and the chain rule through the affine
    // must match the eager composition's own gradient for `w`.
    let device = Device::Cpu;
    let w0: [f32; 8] = [0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 0.75];
    let intermediate = w0.len() / 2;

    let w_f = Var::from_tensor(&Tensor::from_slice(&w0, (1, 2 * intermediate), &device).unwrap())
        .unwrap();
    let wi_f = w_f.affine(1.5, 0.2).unwrap();
    assert!(
        !wi_f.is_variable(),
        "wi_out must be the is_variable()==false case under test"
    );
    let out_f = fused(&wi_f).unwrap();
    let grads_f = out_f.sum_all().unwrap().backward().unwrap(); // must not panic
    let dwf: Vec<f32> = grads_f
        .get(&w_f)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let w_e = Var::from_tensor(&Tensor::from_slice(&w0, (1, 2 * intermediate), &device).unwrap())
        .unwrap();
    let wi_e = w_e.affine(1.5, 0.2).unwrap();
    let out_e = eager(&wi_e).unwrap();
    let grads_e = out_e.sum_all().unwrap().backward().unwrap();
    let dwe: Vec<f32> = grads_e
        .get(&w_e)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for (i, (f, e)) in dwf.iter().zip(dwe.iter()).enumerate() {
        assert!((f - e).abs() < 1e-3, "dw[{i}]: fused {f} vs eager {e}");
    }
}

/// Sanity check that the production-width bf16 fixture above actually
/// exercises F32-vs-BF16-independent behavior — the fused F32 path at the
/// same width must still equal the eager F32 composition (no width-
/// dependent bug hiding in the row/column index arithmetic at a real,
/// non-toy `intermediate`).
#[test]
fn fwd_matches_eager_at_production_width_f32() {
    let device = Device::Cpu;
    let intermediate = 2624usize;
    let rows = 1usize;
    let n = rows * 2 * intermediate;
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 4.0).collect();
    let wi = Tensor::from_slice(&v, (rows, 2 * intermediate), &device).unwrap();

    let out_f: Vec<f32> = fused(&wi)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_e: Vec<f32> = eager(&wi)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (f, e)) in out_f.iter().zip(out_e.iter()).enumerate() {
        assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs eager {e}");
    }
}

/// The backward analog of the forward oracle above, at the SAME
/// production width — the fixture-independent claim this file's module
/// doc makes ("the f32 backward can never be bit-exact vs. eager BY
/// CONSTRUCTION") is about a SYSTEMATIC per-element offset, so measuring
/// it at a realistic element count (rather than the 6-element toy fixture
/// `eager_vs_fused_f32_fwd_and_bwd_match_within_stated_tolerance` uses) is
/// what actually characterizes its typical/worst magnitude — see
/// `ops::geglu`'s module doc's "backward derivation" section for the
/// citation this measurement backs.
#[test]
fn bwd_matches_eager_at_production_width_f32() {
    let device = Device::Cpu;
    let intermediate = 2624usize;
    let rows = 1usize;
    let n = rows * 2 * intermediate;
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 4.0).collect();
    let w0: Vec<f32> = (0..rows * intermediate)
        .map(|i| ((i as f32 * 0.021).cos() * 2.3) + 0.05)
        .collect();

    let wi_f =
        Var::from_tensor(&Tensor::from_slice(&v, (rows, 2 * intermediate), &device).unwrap())
            .unwrap();
    let wi_e =
        Var::from_tensor(&Tensor::from_slice(&v, (rows, 2 * intermediate), &device).unwrap())
            .unwrap();
    let w_f = Tensor::from_slice(&w0, (rows, intermediate), &device).unwrap();
    let w_e = Tensor::from_slice(&w0, (rows, intermediate), &device).unwrap();

    let out_f = fused(&wi_f).unwrap();
    let out_e = eager(&wi_e).unwrap();
    let loss_f = (&out_f * &w_f).unwrap().sum_all().unwrap();
    let loss_e = (&out_e * &w_e).unwrap().sum_all().unwrap();
    let grads_f = loss_f.backward().unwrap();
    let grads_e = loss_e.backward().unwrap();
    let dwf: Vec<f32> = grads_f
        .get(&wi_f)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dwe: Vec<f32> = grads_e
        .get(&wi_e)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_diff = dwf
        .iter()
        .zip(dwe.iter())
        .map(|(f, e)| (f - e).abs())
        .fold(0.0f32, f32::max);
    println!("bwd_matches_eager_at_production_width_f32: measured max |dwi_out diff| = {max_diff}");
    for (i, (f, e)) in dwf.iter().zip(dwe.iter()).enumerate() {
        assert!((f - e).abs() < 1e-3, "dwi_out[{i}]: fused {f} vs eager {e}");
    }
}
