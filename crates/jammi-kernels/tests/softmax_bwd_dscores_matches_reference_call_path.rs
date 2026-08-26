//! esc-045 round 4 (GH#374) — `SoftmaxLastDimFused`'s internal
//! `SoftmaxBwdDScores` kernel must match the REFERENCE MODEL's actual call
//! path, not an ATen kernel read in isolation. Round 3's oracle
//! (`softmax_bwd_dscores_aten_rounding.rs`, deleted this round) pinned the
//! wrong reference: it quoted `softmax_backward_cuda_out`'s BF16-native
//! `Tensor tmp = grad * output;` step as if it were reachable from this
//! project's target model, when it is not.
//!
//! ## The reference model's actual call, quoted at source
//!
//! `transformers` 5.15.1 (this project's pinned checkpoint runtime, see
//! `crates/jammi-bench/reference/torch_grad_oracle.py`),
//! `src/transformers/models/modernbert/modeling_modernbert.py:180`,
//! `eager_attention_forward`:
//!
//! ```text
//! attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
//! ```
//!
//! `attn_weights` is `BF16` (ModernBERT's training dtype). The explicit
//! `dtype=torch.float32` keyword argument routes this call through ATen's
//! `Tensor softmax(const Tensor& input_, int64_t dim_, optional<ScalarType>
//! dtype)` (`aten/src/ATen/native/SoftMax.cpp`):
//!
//! ```text
//! if ((input_.is_cuda() || input_.is_xpu()) && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
//!     return at::_softmax(input_, dim_, true);
//! } else {
//!     Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
//!     return at::_softmax(converted, dim_, false);
//! }
//! ```
//!
//! The fused `half_to_float=true` kernel is gated on `ScalarType::Half`
//! ONLY — `BFloat16` is EXCLUDED from that fast path (verified against
//! `pytorch/pytorch`'s own `aten/src/ATen/native/SoftMax.cpp`, `main`
//! branch, the `softmax()` free function). So ModernBERT's `BF16`
//! `attn_weights` take the `else` branch: an explicit `Tensor converted =
//! input_.toType(Float)` upcast (its own autograd node,
//! `ToCopyBackward`), then a PLAIN `_softmax(converted, dim,
//! half_to_float=false)` that never sees a `BF16` value — both its input
//! and output are already `F32`.
//!
//! Backward for THAT call: `softmax_backward_cuda_out` computes
//! `half_to_float = (grad.scalar_type() != input_dtype)`. Here `grad`
//! (upcast from the downstream `BF16` gradient by the `.to(query.dtype)`
//! cast's own trivial backward) and `input_dtype` (`F32`, since the
//! forward input was `converted`) are BOTH `F32`, so `half_to_float` is
//! FALSE — `Tensor tmp = grad * output;` is an `F32 * F32 -> F32` product,
//! genuinely NO `BF16` rounding anywhere inside `softmax_backward`. The
//! ENTIRE `dy·y` reduction (materialize `tmp`, sum it, subtract
//! `output*sum`) runs in pure `F32`. The ONLY `BF16` rounding in this
//! whole chain happens OUTSIDE `softmax_backward`, at the backward of the
//! initial `.toType(Float)` upcast, which downcasts the final `F32`
//! `grad_input` to `BF16` exactly ONCE, after the reduction is already
//! done.
//!
//! So the correct reference formula is:
//!
//! ```text
//! dot   = Σ_i f32(dy_i) * f32(y_i)        // f32 accumulate, NO intermediate bf16 rounding
//! dS_i  = round_bf16((f32(dy_i) - dot) * f32(y_i))   // ONE rounding, at the very end
//! ```
//!
//! — NOT round 3's two-rounding formula (`tmp_i = round_bf16(dy_i*y_i)`
//! summed, then rounded again in the epilogue), which quoted
//! `softmax_backward_cuda_out`'s `BF16`-native (`half_to_float=false` with
//! a RAW `BF16` grad/output, no `dtype=torch.float32` upcast) case — a
//! code path `modeling_modernbert.py:180`'s actual call never reaches.
//!
//! ## What jammi's kernel now computes
//!
//! `dscores_row_bf16` (`crates/jammi-kernels/src/ops/softmax.rs`,
//! identically `softmax_bwd_dscores_bf16` in
//! `crates/jammi-kernels/src/cuda/softmax.cu`) accumulates `dot` fully in
//! `f32` and rounds once at the end — exactly the reference formula above.
//!
//! ## The fixture (family L: generic/synthetic, no captured production data)
//!
//! Same shape as round 3's fixture: `last = 512` (production width, this
//! project's own b8-s512 training leg), `AMPLITUDES` `0.5..300` (matching
//! `ops/attention_block.rs`'s own `S_max` doc citation — a REAL captured
//! `max|qkv| ≈ 9`-`18` on ModernBERT-large), and an in-file seeded
//! `xorshift64` PRNG (family L: no untracked external generator) with `dy`
//! deliberately non-uniform per row (a uniform `dy` would make `dS`
//! trivially proportional to a constant and mask this whole class of
//! rounding-placement bug).
//!
//! ## What this test does NOT claim
//!
//! Same disclosure as round 3's fixture: ATen's real GPU block reduction
//! does not fold in the same linear ascending order this test's reference
//! (or jammi's own kernel, which explicitly documents "Fixed fold order")
//! uses; a genuine fold-order mismatch could contribute a separate,
//! much smaller (sub-ULP-in-f32-scale) discrepancy this test does not
//! isolate. The rounding-PLACEMENT effect this test measures (an extra
//! BF16 quantization of up to 512 per-element products before they are
//! summed) is the dominant, easily-separable mechanism — see the
//! non-vacuous RED control below.

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

/// The CORRECT reference, matching `modeling_modernbert.py:180`'s actual
/// call path (module doc above): `dot = Σ f32(dy)*f32(y)` accumulated with
/// NO intermediate `BF16` rounding, then ONE rounding at the very end.
fn dscores_row_reference_call_path(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
    let mut dot = 0f32;
    for i in 0..y.len() {
        dot += dy[i].to_f32() * y[i].to_f32();
    }
    (0..y.len())
        .map(|i| bf16::from_f32((dy[i].to_f32() - dot) * y[i].to_f32()))
        .collect()
}

/// The round-3 formula (RED control): materializes a separate `BF16`-width
/// `tmp = round_bf16(dy*y)` BEFORE the row reduction, then rounds a SECOND
/// time in the epilogue. This is `softmax_backward_cuda_out`'s BF16-native
/// case — reachable only when the CALLER passes raw `BF16` grad/output
/// with no `dtype=torch.float32` upcast, which `modeling_modernbert.py:180`
/// never does. Kept ONLY to build the non-vacuous discrimination control
/// below; the real kernel value asserted against
/// `dscores_row_reference_call_path` always comes from the live
/// `.backward()` dispatch.
fn dscores_row_round3_two_rounding_formula(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
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

/// Non-vacuous discrimination floor — leaves headroom for a different
/// libm/toolchain build while refusing a fixture that has degenerated to
/// "the two formulas always agree" (kernel guide §3.7 / AGENTS.md's
/// standing non-vacuous-negative-control clause).
const MIN_DISCRIMINATING_ROWS: usize = 8;

fn build_fixture(device: &Device) -> (Var, Tensor, Tensor) {
    let rows = AMPLITUDES.len();
    let mut rng = XorShift64::new(0x9E37_79B9_7F4A_7C15);

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
        Var::from_tensor(&Tensor::from_slice(&scores_bf16, (rows, LAST), device).unwrap()).unwrap();
    let mask_t = Tensor::from_slice(&mask_bf16, (rows, LAST), device).unwrap();
    let dy_t = Tensor::from_slice(&dy_bf16, (rows, LAST), device).unwrap();
    (scores_var, mask_t, dy_t)
}

/// GREEN: the live `SoftmaxLastDimFused::bwd` dispatch (real `.backward()`
/// call, never a hand-rederived softmax) matches
/// `dscores_row_reference_call_path` — the reference model's actual call
/// path — on every element.
#[test]
fn bf16_bwd_dscores_matches_reference_model_call_path() {
    let device = Device::Cpu;
    let rows = AMPLITUDES.len();
    let (scores_var, mask_t, dy_t) = build_fixture(&device);

    let p = apply2(
        scores_var.as_tensor(),
        &mask_t,
        SoftmaxLastDimFused::new(FullyMaskedPolicy::default()),
    )
    .unwrap();
    let y_bf16: Vec<bf16> = p.flatten_all().unwrap().to_vec1().unwrap();
    let dy_bf16: Vec<bf16> = dy_t.flatten_all().unwrap().to_vec1().unwrap();

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
        let reference = dscores_row_reference_call_path(yr, dyr);
        let round3_control = dscores_row_round3_two_rounding_formula(yr, dyr);

        let mut row_discriminates = false;
        for i in 0..LAST {
            // Affirmative finiteness first (never a negated comparison —
            // `NaN > c` reads `false`, which would silently "pass" here).
            assert!(
                kr[i].to_f32().is_finite()
                    && reference[i].to_f32().is_finite()
                    && yr[i].to_f32().is_finite(),
                "row {r} col {i}: a non-finite value slipped through \
                 (kernel={:?}, reference={:?}, y={:?}, amp={})",
                kr[i],
                reference[i],
                yr[i],
                AMPLITUDES[r]
            );
            if reference[i] != round3_control[i] {
                row_discriminates = true;
            }
            if kr[i].to_f32() != 0.0 {
                any_nonzero_signal = true;
            }
            if kr[i] != reference[i] {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((r, i, AMPLITUDES[r], kr[i], reference[i], yr[i]));
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
         reference-call-path formula from round 3's two-rounding formula — this fixture would \
         read GREEN regardless of which formula the kernel computes; strengthen it before \
         trusting this oracle"
    );

    let total_elements = rows * LAST;
    assert_eq!(
        mismatches,
        0,
        "SoftmaxBwdDScores's bf16 backward does NOT match the reference model's actual call \
         path on {mismatches}/{total_elements} elements across {rows} rows at last={LAST} \
         (esc-045/GH#374 round 4). jammi should round `dS` ONCE (f32-accumulate throughout, \
         matching modeling_modernbert.py:180's dtype=torch.float32 upcast); see this file's \
         module doc for the exact HF+ATen source quotes. First mismatch: row={} col={} \
         amplitude={} kernel_dS={:?} reference_dS={:?} y={:?}.",
        first_mismatch.map(|m| m.0).unwrap_or(usize::MAX),
        first_mismatch.map(|m| m.1).unwrap_or(usize::MAX),
        first_mismatch.map(|m| m.2).unwrap_or(f32::NAN),
        first_mismatch.map(|m| m.3),
        first_mismatch.map(|m| m.4),
        first_mismatch.map(|m| m.5),
    );
}

/// RED control: the round-3 two-rounding formula (a real, deleted revision
/// of `dscores_row_bf16`) must DIVERGE from the reference-call-path formula
/// on at least `MIN_DISCRIMINATING_ROWS` rows at production width. This is
/// the fixture's own non-vacuity proof — if this test failed (the two
/// formulas always agreed), the GREEN test above would be worthless as an
/// oracle regardless of which formula the live kernel computed.
#[test]
fn round3_two_rounding_formula_diverges_from_reference_call_path_red_control() {
    let device = Device::Cpu;
    let rows = AMPLITUDES.len();
    let (scores_var, mask_t, dy_t) = build_fixture(&device);

    let p = apply2(
        scores_var.as_tensor(),
        &mask_t,
        SoftmaxLastDimFused::new(FullyMaskedPolicy::default()),
    )
    .unwrap();
    let y_bf16: Vec<bf16> = p.flatten_all().unwrap().to_vec1().unwrap();
    let dy_bf16: Vec<bf16> = dy_t.flatten_all().unwrap().to_vec1().unwrap();

    let mut discriminating_rows = 0usize;
    let mut total_mismatches = 0usize;
    for r in 0..rows {
        let yr = &y_bf16[r * LAST..(r + 1) * LAST];
        let dyr = &dy_bf16[r * LAST..(r + 1) * LAST];
        let reference = dscores_row_reference_call_path(yr, dyr);
        let round3_control = dscores_row_round3_two_rounding_formula(yr, dyr);
        let mut row_mismatches = 0usize;
        for i in 0..LAST {
            if reference[i] != round3_control[i] {
                row_mismatches += 1;
            }
        }
        if row_mismatches > 0 {
            discriminating_rows += 1;
        }
        total_mismatches += row_mismatches;
    }

    eprintln!(
        "esc-045 round 4 RED control: round-3 two-rounding formula diverges from the \
         reference-call-path formula on {discriminating_rows}/{rows} rows, \
         {total_mismatches}/{} elements at last={LAST}",
        rows * LAST
    );
    assert!(
        discriminating_rows >= MIN_DISCRIMINATING_ROWS,
        "RED control failed to discriminate: only {discriminating_rows}/{rows} rows separate \
         round 3's two-rounding formula from the reference-call-path formula \
         ({total_mismatches} total element mismatches) — the fixture carries no signal for \
         esc-045 round 4's regression"
    );
}
