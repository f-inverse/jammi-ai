//! esc-045 (GH#374 round 3) op-level RED reproducer — `SoftmaxLastDimFused`'s
//! own internal `SoftmaxBwdDScores` kernel rounds `dScores = y * (dy -
//! Σ(dy·y))` at a DIFFERENT point than ATen's own bf16 softmax backward,
//! even though both compute the same algebraic identity.
//!
//! ## The ATen reference, quoted at source (v2.13.0)
//!
//! `aten/src/ATen/native/transformers/cuda/attention_backward.cu`'s math
//! path differentiates through ordinary `at::softmax`, whose CUDA backward
//! (`aten/src/ATen/native/cuda/SoftMax.cu`) is:
//!
//! ```text
//! TORCH_IMPL_FUNC(softmax_backward_cuda_out)
//! (const Tensor& grad, const Tensor& output, int64_t dim,
//!  ScalarType input_dtype, const Tensor& grad_input) {
//!   ...
//!   Tensor tmp = grad * output;
//!   host_softmax_backward<SoftMaxBackwardEpilogue, false>(tmp, output, dim, half_to_float, grad_input);
//! }
//! ```
//!
//! `Tensor tmp = grad * output;` is a SEPARATE elementwise kernel
//! (`mul_kernel_cuda`, `aten/src/ATen/native/cuda/BinaryMulKernel.cu`):
//! `using opmath_t = at::opmath_type<scalar_t>; opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, MulFunctor<opmath_t>());`
//! — `opmath_type<BFloat16> == float`, and the TensorIterator machinery
//! computes each element in `opmath_t` (float) and casts to the OUTPUT
//! dtype (BFloat16) exactly once when it stores into `tmp`. So `tmp` is a
//! genuinely BF16-ROUNDED tensor: `tmp_i = round_bf16(f32(dy_i) * f32(y_i))`,
//! rounded BEFORE the reduction below ever runs.
//!
//! `host_softmax_backward` then reduces THAT already-rounded `tmp` (renamed
//! `grad` inside the function) in `accscalar_t` (float —
//! `acc_type<BFloat16, true> == float`): `cunn_SoftMaxBackward`'s
//! `threadSum = ilpReduce<AddFloat, ...>(..., gradOutput, ...)` where
//! `gradOutput` is `tmp`'s data pointer (per the kernel's own comment,
//! `SoftMaxBackwardEpilogue`: "gradOutput that we get here is really
//! gradOutput * output" — i.e. the caller already folded the multiply into
//! `tmp` before this kernel ever runs). The epilogue then computes, per
//! element, `static_cast<T>(gradOutput - output * sum)` with `gradOutput`
//! STILL `tmp_i` (not raw `dy_i`), casting the result to BF16 once. So:
//!
//! ```text
//! tmp_i = round_bf16(f32(dy_i) * f32(y_i))          // ONE rounding, PRE-reduction
//! sum_k = Σ_i f32(tmp_i)                              // f32 accumulate of the ROUNDED products
//! dS_i  = round_bf16(f32(tmp_i) - f32(y_i) * sum_k)   // a SECOND, independent rounding, final
//! ```
//!
//! ## What jammi actually computes
//!
//! `dscores_row_bf16` (`crates/jammi-kernels/src/ops/softmax.rs:1277-1285`,
//! identically `softmax_bwd_dscores_bf16` in
//! `crates/jammi-kernels/src/cuda/softmax.cu:319-338`) instead accumulates
//! the UNROUNDED product directly in a float register, never materializing
//! `dy*y` at BF16 width at all:
//!
//! ```text
//! dot  = Σ_i f32(dy_i) * f32(y_i)                    // full-precision product, no intermediate round
//! dS_i = round_bf16((f32(dy_i) - dot) * f32(y_i))    // ONE rounding, final
//! ```
//!
//! Both formulas are the same algebraic identity (`y·dy − y·Σ(dy·y)`), and
//! jammi's own module doc (`ops/softmax.rs:417-429`) states "f32-accumulate,
//! round-once" is a DELIBERATE, load-bearing crate convention. But esc-045's
//! own round-2 finding (`ops/geglu.rs`'s "esc-045 fix" section,
//! `tests/geglu_torch_bwd_rounding.rs`) already established at ATen source
//! that "round-once" is not always torch's own convention: torch's
//! `mul_tensor_backward` differentiates the SAVED, already-rounded operand,
//! not an unrounded one. The same shape recurs here — torch genuinely
//! rounds the per-element product to BF16 BEFORE reducing it (an extra,
//! real rounding boundary this op's own doc does not mention); jammi never
//! does. This is the op-level oracle for that divergence.
//!
//! `attention_block.rs`'s `bwd_core` (`ops/attention_block.rs:995`,
//! `apply2(&p, &dp, SoftmaxBwdDScores)`) is what actually feeds this kernel
//! in the real ModernBERT-large training step — it is `dS` on the way to
//! `dQ`/`dK`. This test drives the SAME internal kernel through the public,
//! production dispatch path (`SoftmaxLastDimFused`'s own `CustomOp2::bwd`,
//! via a real `Tensor::backward()` call — never a hand-rederived softmax
//! forward) rather than re-deriving `attention_block.rs`'s whole GEMM chain,
//! since `SoftmaxBwdDScores` itself is `pub(crate)` and this is exactly the
//! function both call sites share.
//!
//! ## The fixture (family L: generic/synthetic, no captured production data)
//!
//! This op names no consumer, so no "real ModernBERT layer-N" capture is
//! required or claimed (contrast `tests/geglu_torch_bwd_rounding.rs`, whose
//! FIXTURE explicitly IS a real capture because GeGLU's OWN doc cites a
//! validated-coverage amplitude ceiling). `last = 512` matches this
//! project's own b8-s512 training leg (`scratchpad/HANDOFF.md` §1) —
//! production WIDTH. `AMPLITUDES` below (`0.5` .. `300`) matches
//! `ops/attention_block.rs`'s own module doc citation ("`S_max` is
//! `O(1)`-`O(4)`" at `|qkv|<=1`, "`O(400)`" at `|qkv|=10`, with a REAL
//! captured `max|qkv| ≈ 9`-`18` on ModernBERT-large) — production AMPLITUDE,
//! not the `0.1×` trap this project's own record names as the reason a
//! prior oracle round missed a real defect. Every row's `scores` and `dy`
//! come from an in-file, seeded `xorshift64` PRNG (the producer is THIS
//! file, fully inspectable — no untracked external generator), with `dy`
//! deliberately non-uniform (a uniform seed would make `dS` trivially
//! proportional to a constant and mask exactly this class of bug, per
//! `tests/geglu_torch_bwd_rounding.rs`'s identical note).
//!
//! ## What this test does NOT claim
//!
//! ATen's real block reduction (`blockReduce`/warp shuffles) does not fold
//! in the same linear ascending order this test's reference (or jammi's own
//! kernel, which explicitly documents "Fixed fold order" —
//! `ops/softmax.rs:1061-1064`) uses; a genuine GPU-vs-reference fold-order
//! mismatch could contribute a SEPARATE, much smaller (sub-ULP-in-f32-scale)
//! discrepancy this test does not isolate. The rounding-PLACEMENT effect
//! this test measures (an extra BF16 quantization of up to 512 per-element
//! products before they are summed) is the dominant, easily-separable
//! mechanism — see the non-vacuous control below, which shows the two
//! formulas disagree on the large majority of rows at `last = 512`.

use candle_core::{Device, Tensor, Var};
use half::bf16;
use jammi_kernels::ops::{apply2, FullyMaskedPolicy, SoftmaxLastDimFused};

const LAST: usize = 512; // production width — this project's own b8-s512 leg

/// Production-amplitude sweep for raw `scores`, cited against
/// `ops/attention_block.rs`'s own `S_max` doc (see module doc above).
const AMPLITUDES: &[f32] = &[
    0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0, 150.0, 220.0, 300.0,
];

/// Deterministic, in-file xorshift64 PRNG — the producer for every random
/// value this fixture uses (family L: no untracked external generator).
/// Seeded with the splitmix64 golden-ratio constant, a fixed, disclosed
/// choice, not a "random random" seed.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // xorshift64 requires a nonzero state.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }
}

/// ATen-faithful reference for one row: `tmp_i = round_bf16(dy_i * y_i)`
/// (a genuine, separate BF16-width intermediate — this is `Tensor tmp =
/// grad * output;`), `sum_k = Σ f32(tmp_i)`, `dS_i = round_bf16(tmp_i -
/// y_i * sum_k)`. Ascending fold order (module doc's disclosed caveat).
fn dscores_row_aten_reference(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
    let tmp: Vec<bf16> = y
        .iter()
        .zip(dy)
        .map(|(&yv, &dyv)| bf16::from_f32(dyv.to_f32() * yv.to_f32()))
        .collect();
    let mut sum_k = 0f32;
    for &t in &tmp {
        sum_k += t.to_f32();
    }
    tmp.iter()
        .zip(y)
        .map(|(&t, &yv)| bf16::from_f32(t.to_f32() - yv.to_f32() * sum_k))
        .collect()
}

/// jammi's PRE-FIX formula (hand-transcribed from `dscores_row_bf16` as it
/// read before esc-045 round 3 — `dot = Σ f32(dy)*f32(y)`, one rounding at
/// the end), kept ONLY to build the non-vacuous discrimination control
/// below (proving the fixture actually separates the two formulas). The
/// REAL kernel value asserted against `dscores_row_aten_reference` further
/// down always comes from the real `.backward()` dispatch, never this
/// function — after the fix, the live kernel computes the ATen formula
/// directly, not this one.
fn dscores_row_pre_fix_buggy_formula(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
    let mut dot = 0f32;
    for i in 0..y.len() {
        dot += dy[i].to_f32() * y[i].to_f32();
    }
    (0..y.len())
        .map(|i| bf16::from_f32((dy[i].to_f32() - dot) * y[i].to_f32()))
        .collect()
}

/// Non-vacuous discrimination floor — measured below; leaves headroom for a
/// different libm/toolchain build while refusing a fixture that has
/// degenerated to "the two formulas always agree" (kernel guide §3.7 /
/// AGENTS.md's standing non-vacuous-negative-control clause).
const MIN_DISCRIMINATING_ROWS: usize = 8;

#[test]
fn bf16_bwd_dscores_matches_aten_softmax_backward_rounding_not_jammis_own() {
    let device = Device::Cpu;
    let rows = AMPLITUDES.len();
    let mut rng = XorShift64::new(0x9E37_79B9_7F4A_7C15);

    // Build `rows` independent softmax rows at production width, each at
    // its own amplitude — see module doc's fixture section.
    let mut scores_v: Vec<f32> = Vec::with_capacity(rows * LAST);
    let mut dy_v: Vec<f32> = Vec::with_capacity(rows * LAST);
    for &amp in AMPLITUDES {
        for _ in 0..LAST {
            scores_v.push(amp * (2.0 * rng.next_unit() - 1.0));
        }
        // `dy` amplitude sampled independently per row (0.01..5.0) and
        // cubed within the row to skew toward small values with occasional
        // large ones — deliberately non-uniform, see module doc.
        let dy_amp = 0.01 + 4.99 * rng.next_unit();
        for _ in 0..LAST {
            let u = 2.0 * rng.next_unit() - 1.0;
            dy_v.push(dy_amp * u * u.abs());
        }
    }

    let scores_bf16: Vec<bf16> = scores_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_bf16: Vec<bf16> = dy_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let mask_bf16: Vec<bf16> = vec![bf16::ZERO; rows * LAST]; // fully unmasked

    let scores_var =
        Var::from_tensor(&Tensor::from_slice(&scores_bf16, (rows, LAST), &device).unwrap())
            .unwrap();
    let mask_t = Tensor::from_slice(&mask_bf16, (rows, LAST), &device).unwrap();

    // Real forward dispatch — the production entry point, never a
    // hand-rederived softmax formula. `FullyMaskedPolicy::default()` /
    // `SoftmaxLastDimFused::new`'s default `scale == 1.0` means `dscores`
    // is `SoftmaxBwdDScores`'s raw output bit-for-bit (module doc,
    // `ops/softmax.rs:483-489`).
    let p = apply2(
        scores_var.as_tensor(),
        &mask_t,
        SoftmaxLastDimFused::new(FullyMaskedPolicy::default()),
    )
    .unwrap();
    let y_bf16: Vec<bf16> = p.flatten_all().unwrap().to_vec1().unwrap();

    let dy_t = Tensor::from_slice(&dy_bf16, (rows, LAST), &device).unwrap();
    let loss = (&p * &dy_t).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dscores_kernel: Vec<bf16> = grads
        .get(scores_var.as_tensor())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut discriminating_rows = 0usize;
    let mut first_mismatch: Option<(usize, usize, f32, bf16, bf16, bf16)> = None;
    let mut mismatches = 0usize;
    let mut any_nonzero_signal = false;

    for r in 0..rows {
        let yr = &y_bf16[r * LAST..(r + 1) * LAST];
        let dyr = &dy_bf16[r * LAST..(r + 1) * LAST];
        let kr = &dscores_kernel[r * LAST..(r + 1) * LAST];
        let aten_ref = dscores_row_aten_reference(yr, dyr);
        let pre_fix_buggy = dscores_row_pre_fix_buggy_formula(yr, dyr);

        let mut row_discriminates = false;
        for i in 0..LAST {
            // Affirmative finiteness first (never a negated comparison —
            // `NaN > c` reads `false`, which would silently "pass" here).
            assert!(
                kr[i].to_f32().is_finite()
                    && aten_ref[i].to_f32().is_finite()
                    && yr[i].to_f32().is_finite(),
                "row {r} col {i}: a non-finite value slipped through \
                 (kernel={:?}, aten_ref={:?}, y={:?}, amp={})",
                kr[i],
                aten_ref[i],
                yr[i],
                AMPLITUDES[r]
            );
            if aten_ref[i] != pre_fix_buggy[i] {
                row_discriminates = true;
            }
            if kr[i].to_f32() != 0.0 {
                any_nonzero_signal = true;
            }
            if kr[i] != aten_ref[i] {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((r, i, AMPLITUDES[r], kr[i], aten_ref[i], yr[i]));
                }
            }
        }
        if row_discriminates {
            discriminating_rows += 1;
        }
    }

    assert!(
        any_nonzero_signal,
        "every dscores element read exactly zero — the fixture carries no signal \
         (ZerosB-style vacuous pass); strengthen it before trusting this oracle"
    );
    assert!(
        discriminating_rows >= MIN_DISCRIMINATING_ROWS,
        "fixture is not discriminating: only {discriminating_rows}/{rows} rows separate the \
         ATen-consistent formula from jammi's own — this fixture would read GREEN on a broken \
         build regardless of the kernel; strengthen it before trusting this oracle"
    );

    let total_elements = rows * LAST;
    assert_eq!(
        mismatches,
        0,
        "SoftmaxBwdDScores's bf16 backward does NOT match ATen's softmax backward rounding \
         placement on {mismatches}/{total_elements} elements across {rows} rows at last={LAST} \
         (esc-045/GH#374 round 3). jammi rounds `dS` ONCE, from the UNROUNDED product `dy*y`; \
         ATen materializes `tmp = grad*output` (BF16-rounded) BEFORE reducing it, then rounds \
         AGAIN in the epilogue — see this file's module doc for the exact ATen source quote. \
         First mismatch: row={} col={} amplitude={} kernel_dS={:?} aten_dS={:?} y={:?}.",
        first_mismatch.map(|m| m.0).unwrap_or(usize::MAX),
        first_mismatch.map(|m| m.1).unwrap_or(usize::MAX),
        first_mismatch.map(|m| m.2).unwrap_or(f32::NAN),
        first_mismatch.map(|m| m.3),
        first_mismatch.map(|m| m.4),
        first_mismatch.map(|m| m.5),
    );
}
