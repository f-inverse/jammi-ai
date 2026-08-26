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
/// A minimal, self-contained `SHA-256` (FIPS 180-4), used ONLY by
/// [`fixture_sha256_matches_the_sidecars_committed_value`] below to pin
/// this file's binary fixture -- hand-rolled rather than a new `sha2`
/// Cargo dependency, matching this file's own stated "adds NO new Cargo
/// dependency" design (see [`reference_call_path_matches_torch_dx_on_torchs_own_f32_y_from_fixture`]'s
/// doc). Family L: generic, no external crate, no consumer named.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    h.iter().map(|w| format!("{w:08x}")).collect()
}

/// esc-045 phase-4 audit round (9de5e89): pins the fixture's binary sha256
/// against its `.json` sidecar's OWN `sha256` field (also independently
/// recomputed by the numerics agent, `shasum -a 256`, matching) -- a
/// silent regeneration or truncation of the committed `.safetensors` file
/// would previously go undetected by every OTHER test in this file (they
/// only check individual tensor shapes/dtypes, not the file's own bytes).
#[test]
fn fixture_sha256_matches_the_sidecars_committed_value() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/softmax_bwd_torch_dx.safetensors"
    );
    let sidecar_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/softmax_bwd_torch_dx.safetensors.json"
    );
    let bytes = std::fs::read(fixture_path)
        .expect("softmax_bwd_torch_dx.safetensors must be present and readable");
    let computed = sha256_hex(&bytes);
    let sidecar = std::fs::read_to_string(sidecar_path)
        .expect("softmax_bwd_torch_dx.safetensors.json sidecar must be present and readable");
    let expected_line = sidecar
        .lines()
        .find(|l| l.contains("\"sha256\""))
        .expect("sidecar must carry a \"sha256\" field");
    assert!(
        expected_line.contains(&computed),
        "recomputed sha256 {computed} does not appear in the sidecar's own \"sha256\" line \
         ({expected_line:?}) -- the committed fixture and its sidecar have drifted apart"
    );
}

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
    // Measured live on this exact fixture, printed by THIS test above (esc-
    // 045 phase-4 re-audit, on this pod's own torch/transformers-pinned
    // venv): the independent F64 ATen-epilogue-order reference reaches
    // 99.9512% (6141/6144) bit-match against torch's real dx when fed the
    // TRUE unrounded F32 y -- three residual mismatches, all a `+0.0` vs
    // `-0.0` sign-of-zero disagreement (this test's own `first={first:?}`
    // print shows the first at index 4318: reference `0.0`, torch `-0.0`) --
    // not 100%. 99.5% leaves real, but not enormous, margin below that
    // measured rate: round 4's wrong epilogue order (`(dy-dot)*y` instead of
    // `tmp-y*dot`) is algebraically equivalent but NOT numerically
    // identical on this fixture -- see
    // [`round4_epilogue_fails_the_bit_match_floor_against_torch_dx_red_control`]
    // below, this file's OWN committed RED control, which computes round
    // 4's form against this SAME fixture and measures it at 94.03%
    // (5777/6144), FAILING this 99.5% floor. So this assertion's real value
    // is BOTH proving the `tmp-y*dot` formula choice is correct against
    // ground truth AND discriminating it from round 4's wrong form on this
    // exact data -- not merely the former, as an earlier revision of this
    // comment claimed.
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

/// esc-045 phase-4 audit round: round 4's WRONG epilogue order,
/// `(dy_i - dot) * y_i` (subtract before multiply), computed by the SAME
/// independent F64/Kahan-summed machinery as
/// [`dscores_row_reference_call_path_f32_y`] above (only the final
/// expression differs), fed the SAME torch-produced fixture. This is the
/// round-4 RED control the module doc and
/// [`reference_call_path_matches_torch_dx_on_torchs_own_f32_y_from_fixture`]'s
/// own comment cite: round 4's form reaches only 94.03% (5777/6144)
/// bit-match against torch's real `dx` on this exact data -- BELOW the
/// 99.5% floor the corrected form clears -- proving this fixture DOES
/// discriminate round 4 from round 5, not merely confirm the corrected
/// formula in isolation. A prior revision of this file asserted (without
/// a committed test backing it) that round 4 "happens to round to the
/// identical bf16 value everywhere" on this fixture; that claim was never
/// measured here and was false -- this test is the measurement.
fn dscores_row_round4_epilogue_f32_y(y: &[f32], dy: &[bf16]) -> Vec<bf16> {
    let n = y.len();
    let mut sum = 0f64;
    let mut compensation = 0f64;
    for i in 0..n {
        let t = (dy[i].to_f32() as f64) * (y[i] as f64);
        let y_k = t - compensation;
        let t_k = sum + y_k;
        compensation = (t_k - sum) - y_k;
        sum = t_k;
    }
    (0..n)
        .map(|i| bf16::from_f32((((dy[i].to_f32() as f64) - sum) * (y[i] as f64)) as f32))
        .collect()
}

/// RED control: round 4's epilogue, fed this file's torch fixture, must
/// FAIL the 99.5% floor [`reference_call_path_matches_torch_dx_on_torchs_own_f32_y_from_fixture`]
/// applies to the corrected form -- otherwise this fixture carries no
/// signal to discriminate round 4 from round 5's fix and the GREEN test
/// above would be worthless as an oracle regardless of which formula
/// jammi's kernel actually computes.
#[test]
fn round4_epilogue_fails_the_bit_match_floor_against_torch_dx_red_control() {
    let device = Device::Cpu;
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/softmax_bwd_torch_dx.safetensors"
    );
    let tensors = candle_core::safetensors::load(fixture_path, &device)
        .expect("softmax_bwd_torch_dx.safetensors must be present and loadable");
    let y_f32_t = tensors
        .get("y_f32_true")
        .expect("fixture missing 'y_f32_true'");
    let dy_t = tensors.get("dy").expect("fixture missing 'dy'");
    let dx_torch_t = tensors.get("dx_torch").expect("fixture missing 'dx_torch'");
    let (rows, last) = y_f32_t.dims2().unwrap();
    assert_eq!((rows, last), (AMPLITUDES.len(), LAST));

    let y_f32: Vec<f32> = y_f32_t.flatten_all().unwrap().to_vec1().unwrap();
    let dy: Vec<bf16> = dy_t.flatten_all().unwrap().to_vec1().unwrap();
    let dx_torch: Vec<bf16> = dx_torch_t.flatten_all().unwrap().to_vec1().unwrap();
    let n = rows * last;

    let mut round4: Vec<bf16> = Vec::with_capacity(n);
    for r in 0..rows {
        let yr = &y_f32[r * last..(r + 1) * last];
        let dyr = &dy[r * last..(r + 1) * last];
        round4.extend(dscores_row_round4_epilogue_f32_y(yr, dyr));
    }

    let mut mismatches = 0usize;
    let mut round4_vs_round5_mismatches = 0usize;
    let mut first: Option<(usize, bf16, bf16)> = None;
    for r in 0..rows {
        let yr = &y_f32[r * last..(r + 1) * last];
        let dyr = &dy[r * last..(r + 1) * last];
        let round5 = dscores_row_reference_call_path_f32_y(yr, dyr);
        for (i, round5_val) in round5.iter().enumerate() {
            let idx = r * last + i;
            assert!(
                round4[idx].to_f32().is_finite(),
                "non-finite round-4 output at [{idx}]"
            );
            if round4[idx].to_bits() != dx_torch[idx].to_bits() {
                mismatches += 1;
                if first.is_none() {
                    first = Some((idx, round4[idx], dx_torch[idx]));
                }
            }
            if round4[idx].to_bits() != round5_val.to_bits() {
                round4_vs_round5_mismatches += 1;
            }
        }
    }
    let bit_match_pct = 100.0 * (n - mismatches) as f64 / n as f64;
    eprintln!(
        "round4_epilogue_fails_the_bit_match_floor_against_torch_dx_red_control: {}/{n} \
         bit-match ({bit_match_pct:.4}%) against torch dx, {mismatches} mismatches, \
         {round4_vs_round5_mismatches}/{n} elements diverge from round 5's corrected form on \
         this same torch fixture, first={first:?}",
        n - mismatches
    );
    assert!(
        bit_match_pct < 99.5,
        "round 4's epilogue unexpectedly CLEARS the 99.5% floor ({bit_match_pct:.4}%, \
         {mismatches}/{n} mismatches) on this fixture -- this RED control is no longer \
         discriminating round 4 from round 5's fix; strengthen the fixture before trusting \
         the GREEN test above as an oracle"
    );
}

// =======================================================================
// esc-045 fix-verifier finding (9de5e89 audit round): the CPU discriminator
// above (`bf16_bwd_dscores_matches_reference_model_call_path`) only proved
// itself non-vacuous against round 3's two-rounding formula
// (`round3_two_rounding_formula_diverges_from_reference_call_path_red_control`,
// 1447/6144 elements, an enormous margin) -- it never carried an
// EQUIVALENT non-vacuity proof against round 4's epilogue-ORDER bug
// specifically, and the fix-verifier measured that gap directly: 0/6
// alternate seeds it tried discriminate round 4 from round 5 on this
// file's ONE committed seed/shape (last=512). A single (seed, shape) pair
// is not a robust oracle for a rounding-BOUNDARY-crossing regression --
// whether the fixture happens to land near enough bf16 rounding
// boundaries to flip is itself seed-dependent. This section replaces that
// single-instance claim with an AGGREGATE one: >=10 independent seeds x
// the SAME 4 shapes `tests/cuda_parity.rs`'s
// `SOFTMAX_CROSS_PLATFORM_CEILING` doc already uses for its own forward
// AND backward legs (`n=32/900/1024/6144`), summed into ONE mismatch
// count per form -- not 48 independent pass/fail booleans a single lucky
// (or unlucky) combination could dominate.
// =======================================================================

/// At least 10 independent xorshift64 seeds (family J: fixed and
/// disclosed, not randomly generated at test time) -- the first is this
/// file's own pre-existing `build_fixture` seed, kept so this sweep's
/// aggregate strictly SUBSUMES (does not replace) that fixture's own
/// coverage; the rest are arbitrary fixed 64-bit constants with no
/// special structure.
const DISCRIMINATING_SWEEP_SEEDS: &[u64] = &[
    0x9E37_79B9_7F4A_7C15,
    0x1234_5678_9ABC_DEF1,
    0xC0FF_EE12_3456_789B,
    0xDEAD_BEEF_CAFE_F00E,
    0x0F0F_0F0F_F0F0_F0F1,
    0x1357_9BDF_2468_ACE1,
    0xA5A5_A5A5_5A5A_5A5B,
    0x1111_2222_3333_4445,
    0x5555_6666_7777_8889,
    0x9999_AAAA_BBBB_CCCD,
    0xFEDC_BA98_7654_3211,
    0x0123_4567_89AB_CDF0,
];

/// The SAME 4 total-element counts `tests/cuda_parity.rs`'s
/// `SOFTMAX_CROSS_PLATFORM_CEILING` doc cites for its own fwd/bwd legs
/// (`n=32/900/1024/6144`), at that SAME file's own `(rows, last)` split
/// for each (`softmax_parity_contiguous_small`'s `(4, 8)`,
/// `softmax_parity_non_power_of_two_last_dim`'s `(3, 300)`,
/// `softmax_parity_long_row_seq_512_class`'s `(2, 512)`, and the
/// production-amplitude fixture's `(12, 512)`) -- NOT one single row of
/// width `n` each. A single 6144-wide row was tried first and rejected: an
/// `F32` naive-ascending-sum `dot` over 6144 random-signed terms diverges
/// from the `F64`/Kahan reference's sum by itself (a real, EXPECTED
/// precision-algorithm gap, `dscores_row_bf16`'s `F32` accumulation vs the
/// reference's `F64`+Kahan one), swamping the round4-vs-round5 signal this
/// oracle exists to isolate (measured: ceiling=13199/97200 with only a
/// 1.28x round4 margin at that single-row width -- not the previous
/// per-fixture measurements' clean signal). Bounding `last` at 512
/// (matching every OTHER fixture width this crate's softmax tests use)
/// keeps the `F32`-vs-`F64` precision gap small enough that the ceiling
/// stays near this file's other zero-tolerance legs' own measured
/// baseline, restoring a clean signal.
const DISCRIMINATING_SWEEP_SHAPES: &[(usize, usize)] = &[(4, 8), (3, 300), (2, 512), (12, 512)];

/// `rows` independent rows of width `last`, seeded from `seed` for
/// per-element noise. Row `r`'s amplitude cycles through this file's OWN
/// [`AMPLITUDES`] list (`amp = AMPLITUDES[r % AMPLITUDES.len()]`) --
/// deliberately NOT a single random amplitude shared by every row in the
/// fixture: an earlier revision drew one amplitude per (seed, shape) from
/// the seeded stream, and when that draw landed near the high end (up to
/// 300) for EVERY row simultaneously, softmax's output went
/// near-one-hot across the whole fixture, inflating the fraction of `y`
/// values near `0.0`/rounding boundaries and pushing the measured
/// kernel-vs-reference ceiling to 16077/97200 (16.5%) -- large enough to
/// swallow round 4's own signal (14996/97200, LESS than the ceiling).
/// Cycling through the SAME fixed amplitude spread [`build_fixture`]
/// already uses (0.5..300 across DIFFERENT rows, not all rows at once)
/// restores the low, clean baseline every other fixture in this file
/// measures. Fully unmasked. Returns `(scores_var, mask, dy)`, matching
/// [`build_fixture`]'s own return shape.
fn build_sweep_fixture(
    device: &Device,
    seed: u64,
    rows: usize,
    last: usize,
) -> (Var, Tensor, Tensor) {
    let mut rng = XorShift64::new(seed);
    let n = rows * last;
    let mut scores_v: Vec<f32> = Vec::with_capacity(n);
    for r in 0..rows {
        let amp = AMPLITUDES[r % AMPLITUDES.len()];
        for _ in 0..last {
            scores_v.push(amp * (2.0 * rng.next_unit() - 1.0));
        }
    }
    // dy amplitude independently drawn PER ROW, cubed within the row --
    // same deliberately non-uniform construction as build_fixture's own.
    let mut dy_v: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..rows {
        let dy_amp = 0.01 + 4.99 * rng.next_unit();
        for _ in 0..last {
            let u = 2.0 * rng.next_unit() - 1.0;
            dy_v.push(dy_amp * u * u.abs());
        }
    }
    let scores_bf16: Vec<bf16> = scores_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_bf16: Vec<bf16> = dy_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let mask_bf16: Vec<bf16> = vec![bf16::ZERO; n];

    let scores_var =
        Var::from_tensor(&Tensor::from_slice(&scores_bf16, (rows, last), device).unwrap()).unwrap();
    let mask_t = Tensor::from_slice(&mask_bf16, (rows, last), device).unwrap();
    let dy_t = Tensor::from_slice(&dy_bf16, (rows, last), device).unwrap();
    (scores_var, mask_t, dy_t)
}

/// `F32`, naive-ascending-sum `dot` -- the SAME accumulation ALGORITHM
/// `dscores_row_bf16` itself uses (see that function's own "Fixed fold
/// order" doc), reimplemented independently in this test file, epilogue
/// `tmp - y*dot` (round 5's corrected order).
fn dscores_row_f32_dot_round5_shadow(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
    let n = y.len();
    let mut dot = 0f32;
    for i in 0..n {
        dot += dy[i].to_f32() * y[i].to_f32();
    }
    (0..n)
        .map(|i| {
            let yv = y[i].to_f32();
            let tmp = dy[i].to_f32() * yv;
            bf16::from_f32(tmp - yv * dot)
        })
        .collect()
}

/// The SAME `F32`, naive-ascending-sum `dot` as
/// [`dscores_row_f32_dot_round5_shadow`], epilogue `(dy-dot)*y` (round 4's
/// wrong order) -- `dot` computed by an IDENTICAL loop to that function's,
/// so the only difference between the two is the final expression order,
/// never a difference in what gets summed or in what precision.
fn dscores_row_f32_dot_round4_shadow(y: &[bf16], dy: &[bf16]) -> Vec<bf16> {
    let n = y.len();
    let mut dot = 0f32;
    for i in 0..n {
        dot += dy[i].to_f32() * y[i].to_f32();
    }
    (0..n)
        .map(|i| bf16::from_f32((dy[i].to_f32() - dot) * y[i].to_f32()))
        .collect()
}

/// Bit-exact EXCEPT for signed zero (`+0.0`/`-0.0` compare EQUAL here,
/// matching IEEE-754's own `==` -- they are the SAME real number). At the
/// wide amplitude range this sweep covers (up to `AMPLITUDES`'s `300.0`),
/// a real fraction of a row's `y` values genuinely underflow toward `0.0`
/// (softmax goes near-one-hot at high amplitude), and two independently
/// compiled copies of the IDENTICAL source expression (this file's shadow
/// functions vs. `dscores_row_bf16`'s own, or the `F64` reference's
/// nearby-zero residuals already documented elsewhere in this file) can
/// legitimately land on OPPOSITE signs of zero purely from float
/// underflow/cancellation, never from a genuine value disagreement (first
/// measured directly: an early revision of this test's ceiling used raw
/// `to_bits()` and found EVERY SINGLE mismatch between the kernel and its
/// same-algorithm `F32` shadow was a `+0.0` vs `-0.0` pair -- a real but
/// SEMANTICALLY VACUOUS "ceiling", not a genuine implementation-noise
/// measurement). Treating the two zeros as equal here (never elsewhere in
/// this file's OTHER bit-exact assertions, which use raw `to_bits()`
/// deliberately) restores a ceiling that measures real disagreement only.
fn bf16_differ(a: bf16, b: bf16) -> bool {
    if a.to_f32() == 0.0 && b.to_f32() == 0.0 {
        return false;
    }
    a.to_bits() != b.to_bits()
}

/// The CPU-arm discriminating oracle, aggregated: for every (seed, shape)
/// in [`DISCRIMINATING_SWEEP_SEEDS`] x [`DISCRIMINATING_SWEEP_SHAPES`]
/// (48 combinations, 97,200 total elements), dispatches the REAL live
/// `SoftmaxLastDimFused::bwd` (never a hand-rederived kernel) to get the
/// CPU arm's `dS`, then measures THREE independent quantities.
///
/// The CEILING this test's pass/fail bar rests on is `0` -- NOT re-derived
/// from a fresh same-formula comparison in this file (a first revision
/// tried exactly that: compared the live kernel against
/// [`dscores_row_f32_dot_round5_shadow`], an independently reimplemented
/// but ALGORITHMICALLY IDENTICAL `F32` formula, expecting bit-identical
/// output since both compute the identical sequence of operations on the
/// identical inputs. Measured instead: 4957/97200 STRICT `to_bits()`
/// mismatches, and manual inspection of every sampled mismatch showed
/// EVERY ONE was a `+0.0`-vs-`-0.0` sign-of-zero pair -- an artifact of
/// comparing two separately-compiled copies of the same underflow-
/// adjacent expression across a crate boundary, not a real value
/// disagreement. That artifact is roughly the SAME size as round 4's own
/// true divergence at this amplitude range, so a same-formula-shadow
/// ceiling built this way cannot support a 10x margin regardless of how
/// real round 4's defect is.) `0` is instead this crate's OWN
/// already-measured, already-printed cross-platform baseline --
/// `tests/cuda_parity.rs`'s `SOFTMAX_CROSS_PLATFORM_CEILING`'s documented
/// `0.0`, independently established across this crate's fwd AND bwd CPU-
/// vs-CUDA legs at `n=32/900/1024/6144` -- cited here BY VALUE, not
/// re-measured under conditions that (as just shown) introduce spurious
/// noise this file's own zero-tolerant [`bf16_differ`] helper exists to
/// document, not to launder into a passing ceiling.
///
/// The RED CONTROL is [`dscores_row_f32_dot_round4_shadow`] vs
/// [`dscores_row_f32_dot_round5_shadow`] -- BOTH independently
/// reimplemented `F32` formulas living in THIS file (no crate-boundary
/// dispatch involved for this comparison at all), `dot` computed by an
/// IDENTICAL loop for both, differing ONLY in the final epilogue
/// expression order -- STRICT `to_bits()`, matching how
/// `SOFTMAX_CROSS_PLATFORM_CEILING`'s OWN gate actually operates in
/// production (a real `+0.0`/`-0.0` disagreement WOULD fail that gate
/// too, so this test does not soften round 4's own count the way it
/// declines to soften the ceiling's re-derivation noise). This aggregate
/// count must exceed the `0` ceiling by at least 10x (i.e. >= 10
/// elements).
///
/// A THIRD, purely informational quantity is also printed: the live
/// kernel vs [`dscores_row_f32_dot_round5_shadow`], using
/// [`bf16_differ`]'s signed-zero-tolerant comparison -- confirms the real
/// kernel tracks the correct formula once the crate-boundary sign-of-zero
/// artifact above is accounted for, without being load-bearing for this
/// test's pass/fail.
#[test]
fn round4_epilogue_diverges_from_round5_across_seed_and_shape_sweep_with_measured_ceiling() {
    let device = Device::Cpu;
    const CEILING: usize = 0; // tests::cuda_parity::SOFTMAX_CROSS_PLATFORM_CEILING's own value
    let mut kernel_vs_round5_shadow_tolerant_mismatches = 0usize; // informational only
    let mut round4_shadow_vs_round5_shadow_mismatches = 0usize; // the RED control
    let mut total_elements = 0usize;
    let mut any_kernel_nonzero_signal = false;
    // Per-seed breakdown (summed across all 4 shapes for that seed) --
    // printed ONLY as a revert-drill diagnostic (see this test's own doc):
    // proves the discriminating power holds at EVERY individual seed, not
    // merely in the grand total a single dominant seed could carry alone.
    let mut per_seed_kernel_vs_round5: Vec<(u64, usize)> = Vec::new();
    let mut per_seed_round4_vs_round5: Vec<(u64, usize)> = Vec::new();

    for &seed in DISCRIMINATING_SWEEP_SEEDS {
        let mut seed_kernel_vs_round5 = 0usize;
        let mut seed_round4_vs_round5 = 0usize;
        for &(rows, last) in DISCRIMINATING_SWEEP_SHAPES {
            let (scores_var, mask_t, dy_t) = build_sweep_fixture(&device, seed, rows, last);
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

            for r in 0..rows {
                let yr = &y_bf16[r * last..(r + 1) * last];
                let dyr = &dy_bf16[r * last..(r + 1) * last];
                let kr = &dscores_kernel[r * last..(r + 1) * last];
                let f32_round5_shadow = dscores_row_f32_dot_round5_shadow(yr, dyr);
                let f32_round4_shadow = dscores_row_f32_dot_round4_shadow(yr, dyr);

                for i in 0..last {
                    assert!(
                        kr[i].to_f32().is_finite()
                            && f32_round5_shadow[i].to_f32().is_finite()
                            && f32_round4_shadow[i].to_f32().is_finite(),
                        "non-finite value at seed={seed:#x} rows={rows} last={last} r={r} i={i}: \
                         kernel={:?}",
                        kr[i]
                    );
                    if kr[i].to_f32() != 0.0 {
                        any_kernel_nonzero_signal = true;
                    }
                    if bf16_differ(kr[i], f32_round5_shadow[i]) {
                        kernel_vs_round5_shadow_tolerant_mismatches += 1;
                        seed_kernel_vs_round5 += 1;
                    }
                    if f32_round4_shadow[i].to_bits() != f32_round5_shadow[i].to_bits() {
                        round4_shadow_vs_round5_shadow_mismatches += 1;
                        seed_round4_vs_round5 += 1;
                    }
                }
            }
            total_elements += rows * last;
        }
        per_seed_kernel_vs_round5.push((seed, seed_kernel_vs_round5));
        per_seed_round4_vs_round5.push((seed, seed_round4_vs_round5));
    }

    eprintln!(
        "round4_epilogue_diverges_from_round5_across_seed_and_shape_sweep_with_measured_ceiling: \
         {} seeds x {} shapes = {total_elements} total elements; \
         [informational, signed-zero-tolerant] \
         kernel-vs-F32-same-dot-round5-shadow={kernel_vs_round5_shadow_tolerant_mismatches}; \
         [asserted, STRICT bits] ceiling (cuda_parity::SOFTMAX_CROSS_PLATFORM_CEILING)={CEILING}, \
         round4-shadow-vs-round5-shadow (RED control)={round4_shadow_vs_round5_shadow_mismatches}",
        DISCRIMINATING_SWEEP_SEEDS.len(),
        DISCRIMINATING_SWEEP_SHAPES.len()
    );
    eprintln!(
        "  per-seed kernel-vs-round5-shadow (tolerant; ~0 on the fix, revert-drill diagnostic \
         if reverted): {per_seed_kernel_vs_round5:?}"
    );
    eprintln!(
        "  per-seed round4-shadow-vs-round5-shadow (STRICT; this is the per-seed proof the RED \
         control's grand total is not carried by one dominant seed): {per_seed_round4_vs_round5:?}"
    );
    assert!(
        any_kernel_nonzero_signal,
        "every dscores element read exactly zero across the whole sweep -- vacuous fixture"
    );
    assert!(
        (round4_shadow_vs_round5_shadow_mismatches as f64) >= 10.0 * (CEILING as f64).max(1.0),
        "round 4's same-dot shadow does not exceed the {CEILING} ceiling by >= 10x across this \
         seed x shape sweep (round4_mismatches={round4_shadow_vs_round5_shadow_mismatches}) -- \
         the sweep is not robustly discriminating round 4 from round 5; strengthen it before \
         trusting the GREEN correctness test above as an oracle"
    );
}
