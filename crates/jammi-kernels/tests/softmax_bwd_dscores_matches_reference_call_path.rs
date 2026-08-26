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
//! ## esc-045 round 5 (GH#374 phase-4 re-audit, ledger row 241): round 4
//! got WHERE right and the internal expression ORDER wrong
//!
//! Round 4's `dS_i = (dy_i − dot) * y_i` (subtract before multiply) is
//! algebraically the reference formula above but NOT what ATen's own
//! `softmax_backward_cuda_out` computes, even in the all-`F32` case: that
//! function (`aten/src/ATen/native/cuda/SoftMax.cu`, unconditional on
//! `half_to_float`) computes `Tensor tmp = grad * output;` FIRST (its own
//! elementwise step), sums `tmp`, then (`SoftMaxBackwardEpilogue::
//! operator()`) `dS_i = tmp_i − y_i·sum` — multiply before subtract.
//! Algebraically identical to round 4's formula, numerically DIFFERENT
//! (`F32` multiplication does not distribute over subtraction exactly).
//! This file's `dscores_row_reference_call_path` below now implements
//! `tmp_i − y_i·sum`, matching ATen's real epilogue order, computed
//! INDEPENDENTLY of jammi's own kernel implementation both in dtype (`F64`
//! with Kahan-compensated summation, not the kernel's `F32`/naive-sum) and
//! in provenance (`tests/fixtures/softmax_bwd_torch_dx.safetensors`, a
//! REAL torch-produced `dx` from `modeling_modernbert.py:180`'s exact call
//! path, checked separately below — not a second hand-rederived formula
//! restating the first). A prior revision of this reference (round 4) was
//! character-identical to `dscores_row_bf16`'s own implementation at the
//! time — tautological, unable to distinguish "the kernel is right" from
//! "the kernel and this file's restatement of it happen to agree" (esc-045
//! phase-4 re-audit finding).
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

use candle_core::{DType, Device, Tensor, Var};
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

/// The CORRECT, INDEPENDENT reference, matching `modeling_modernbert.py:180`'s
/// actual call path AND ATen's actual epilogue expression order (module
/// doc's "esc-045 round 5" section): `tmp_i = f64(dy_i) * f64(y_i)`
/// computed first, `sum = Σ tmp_i` (Kahan-compensated `F64` summation —
/// deliberately a DIFFERENT algorithm from the kernel's own naive `F32`
/// ascending accumulation, not merely a higher-precision restatement of
/// it), THEN `dS_i = tmp_i − f64(y_i)·sum`, rounded to `BF16` only at the
/// very end. Computing in `F64` throughout means rounding-ORDER
/// differences within this reference itself are negligible at `BF16`'s
/// ~8-bit mantissa, so what this measures is genuinely "how close is the
/// kernel to the true real-number answer", not "does the kernel match
/// this file's own `F32` restatement of itself" (esc-045 phase-4
/// re-audit: a prior revision of this function was character-identical to
/// `dscores_row_bf16`'s own implementation — tautological).
fn dscores_row_reference_call_path(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
    let n = y.len();
    let mut tmp = vec![0f64; n];
    let mut sum = 0f64;
    let mut compensation = 0f64;
    for (i, t_slot) in tmp.iter_mut().enumerate() {
        let t = (dy[i].to_f32() as f64) * (y[i].to_f32() as f64);
        *t_slot = t;
        let y_k = t - compensation;
        let t_k = sum + y_k;
        compensation = (t_k - sum) - y_k;
        sum = t_k;
    }
    (0..n)
        .map(|i| bf16::from_f32((tmp[i] - (y[i].to_f32() as f64) * sum) as f32))
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

/// The SAME independent `F64`/Kahan-summed ATen-epilogue-order formula as
/// [`dscores_row_reference_call_path`], but taking `y` as `F32` rather
/// than `BF16` — because the quantity `_softmax_backward_data`'s `output`
/// argument REALLY binds to is the UNROUNDED `F32` intermediate
/// `_softmax(converted, dim, false)` itself produces, retained by
/// autograd BEFORE the separate `.to(query.dtype)` cast node rounds it to
/// `BF16` (esc-045 phase-4 re-audit finding: an earlier draft of this
/// file's torch-fixture test fed the ROUNDED-then-rewidened `y` here and
/// measured only ~74% bit-match against torch's real `dx` — not a formula
/// bug, but exactly this op's OWN SEPARATE, already-documented "`BF16`-
/// native softmax" divergence, `ops/softmax.rs`'s module doc, leaking into
/// what was meant to be an isolated backward-formula check).
fn dscores_row_reference_call_path_f32_y(y: &[f32], dy: &[bf16]) -> Vec<bf16> {
    let n = y.len();
    let mut tmp = vec![0f64; n];
    let mut sum = 0f64;
    let mut compensation = 0f64;
    for (i, t_slot) in tmp.iter_mut().enumerate() {
        let t = (dy[i].to_f32() as f64) * (y[i] as f64);
        *t_slot = t;
        let y_k = t - compensation;
        let t_k = sum + y_k;
        compensation = (t_k - sum) - y_k;
        sum = t_k;
    }
    (0..n)
        .map(|i| bf16::from_f32((tmp[i] - (y[i] as f64) * sum) as f32))
        .collect()
}

/// esc-045 round 5 (GH#374 phase-4 re-audit, ledger row 241): an
/// INDEPENDENT torch-produced fixture, not a second hand-rederived
/// formula. `tests/fixtures/softmax_bwd_torch_dx.safetensors`
/// (`.json` sidecar carries full generation provenance -- torch/
/// transformers version, seed, shape, amplitudes) was produced by running
/// `modeling_modernbert.py:180`'s EXACT call --
/// `nn.functional.softmax(attn_weights, dim=-1,
/// dtype=torch.float32).to(query.dtype)` -- forward AND backward, in real
/// `transformers` 5.15.1 + `torch` (this project's pinned checkpoint
/// runtime), at this SAME shape/amplitude/skew convention (`last=512`,
/// `AMPLITUDES` `0.5..300`, non-uniform `dy`).
///
/// This test asserts [`dscores_row_reference_call_path_f32_y`] — the SAME
/// independent `F64`, ATen-epilogue-order formula
/// [`dscores_row_reference_call_path`] implements, fed the fixture's
/// `y_f32_true` field (the REAL UNROUNDED `F32` softmax probabilities
/// `_softmax_backward_data`'s `output` argument actually binds to,
/// captured via `retain_grad()` BEFORE the `.to(query.dtype)` cast, never
/// itself `BF16`-rounded) — reaches a high bit-match rate against torch's
/// REAL captured `dx`. This ISOLATES the backward EPILOGUE formula from
/// this op's SEPARATE, already-documented forward divergence (`ops/
/// softmax.rs`'s "This op's own `BF16`-native softmax is a SEPARATE,
/// known divergence" section: jammi's OWN forward is `BF16`-native
/// end-to-end and never retains an unrounded `F32` `y` the way torch
/// does, so jammi's live kernel dispatch feeding its OWN `y` into this
/// SAME formula reaches a materially lower end-to-end bit-match against
/// torch — a real, disclosed, SEPARATE mechanism this test does not
/// claim to close). Transitively, with the exact-match test above: the
/// real kernel dispatch matches [`dscores_row_reference_call_path`]
/// bit-for-bit on jammi's own `y` (proven, `BF16` in both places), and
/// THIS reference matches torch's real `dx` at a high rate on the TRUE
/// `F32` `y` (measured here) — together these show the backward FORMULA
/// itself, not the forward's separately-tracked rounding divergence, is
/// torch-correct.
///
/// Loaded via `candle_core::safetensors::load` -- already a `candle_core`
/// API this crate depends on regardless (`crates/jammi-encoders` already
/// loads model weights this way), so this fixture format adds NO new
/// Cargo dependency.
#[test]
fn reference_call_path_matches_torch_dx_on_torchs_own_f32_y_from_fixture() {
    let device = Device::Cpu;
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/softmax_bwd_torch_dx.safetensors"
    );
    let tensors = candle_core::safetensors::load(fixture_path, &device)
        .expect("softmax_bwd_torch_dx.safetensors must be present and loadable (see its .json sidecar for provenance)");
    let y_f32_t = tensors
        .get("y_f32_true")
        .expect("fixture missing 'y_f32_true'");
    let dy_t = tensors.get("dy").expect("fixture missing 'dy'");
    let dx_torch_t = tensors.get("dx_torch").expect("fixture missing 'dx_torch'");
    assert_eq!(
        y_f32_t.dtype(),
        DType::F32,
        "fixture's y_f32_true must be F32"
    );

    let (rows, last) = y_f32_t.dims2().unwrap();
    assert_eq!((rows, last), (AMPLITUDES.len(), LAST));

    let y_f32: Vec<f32> = y_f32_t.flatten_all().unwrap().to_vec1().unwrap();
    let dy: Vec<bf16> = dy_t.flatten_all().unwrap().to_vec1().unwrap();
    let dx_torch: Vec<bf16> = dx_torch_t.flatten_all().unwrap().to_vec1().unwrap();
    let n = rows * last;
    assert_eq!(y_f32.len(), n);
    assert_eq!(dy.len(), n);
    assert_eq!(dx_torch.len(), n);

    let mut dx_reference: Vec<bf16> = Vec::with_capacity(n);
    for r in 0..rows {
        let yr = &y_f32[r * last..(r + 1) * last];
        let dyr = &dy[r * last..(r + 1) * last];
        dx_reference.extend(dscores_row_reference_call_path_f32_y(yr, dyr));
    }

    let mut mismatches = 0usize;
    let mut first: Option<(usize, bf16, bf16)> = None;
    for i in 0..n {
        assert!(
            dx_reference[i].to_f32().is_finite() && dx_torch[i].to_f32().is_finite(),
            "non-finite at [{i}]: reference={:?} torch={:?}",
            dx_reference[i],
            dx_torch[i]
        );
        if dx_reference[i].to_bits() != dx_torch[i].to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, dx_reference[i], dx_torch[i]));
            }
        }
    }
    let bit_match_pct = 100.0 * (n - mismatches) as f64 / n as f64;
    eprintln!(
        "reference_call_path_matches_torch_dx_on_torchs_own_f32_y_from_fixture: {}/{n} \
         bit-match ({bit_match_pct:.4}%), {mismatches} mismatches, first={first:?}",
        n - mismatches
    );
    // Measured live on this exact fixture (esc-045 phase-4 re-audit, on
    // this pod's own torch/transformers-pinned venv): the independent F64
    // ATen-epilogue-order reference reaches 100% (6144/6144) bit-match
    // against torch's real dx when fed the TRUE unrounded F32 y -- proving
    // the `tmp - y*sum` formula (round 5's fix) is exactly what ATen
    // computes, not merely close to it. 99.5% leaves real margin below
    // the measured rate while still catching a real regression: round 4's
    // wrong epilogue order (`(dy-dot)*y` instead of `tmp-y*dot`) is
    // algebraically equivalent and, on THIS specific fixture, happens to
    // round to the identical bf16 value everywhere too (rounding-boundary
    // crossings between the two forms are rare, measured separately in
    // `tests/cuda_parity.rs`'s CUDA legs, where they DO show up) -- so
    // this assertion's real value is proving the FORMULA CHOICE is
    // correct against ground truth, not discriminating round 4 from round
    // 5 on this specific data; see this file's module doc.
    assert!(
        bit_match_pct >= 99.5,
        "the independent ATen-epilogue-order reference bit-matches torch's real dx (fed the \
         true unrounded f32 y) on only {bit_match_pct:.4}% of elements ({mismatches}/{n} \
         mismatches) -- below the measured-with-margin floor; first mismatch at [{}]: \
         reference={:?} torch={:?}",
        first.map(|f| f.0).unwrap_or(usize::MAX),
        first.map(|f| f.1),
        first.map(|f| f.2),
    );
}
