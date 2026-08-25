//! esc-046 (GH#374) legs (2) DEFECT, (3) NON-VACUITY, (4)
//! FINITENESS-AFFIRMATIVE, from `.jammi/escapes.jsonl`'s
//! `esc-046-lora-epilogue-rounds-delta-before-add` row. Leg (1) BLINDNESS
//! lives in `crates/jammi-lora/tests/esc046_epilogue_blindness_gpu.rs`
//! (needs `LoraLinear` + `JAMMI_KERNELS_DISABLE`, CUDA-gated) — this file
//! is CPU-hermetic, testing [`ScaledCastAdd`] directly.
//!
//! ## The mechanism, quoted at PEFT source
//!
//! `peft/tuners/lora/layer.py`'s `Linear.forward` (`peft==0.20.0`, lines
//! 1044-1069):
//! ```text
//! result = self.base_layer(x, *args, **kwargs)   # bf16 GEMM output
//! torch_result_dtype = result.dtype                # bf16
//! ...
//! result = result + lora_B(lora_A(dropout(x))) * scaling   # f32 delta
//! result = result.to(torch_result_dtype)           # ONE cast, at the end
//! ```
//! torch's `+` PROMOTES the bf16 `result` to the delta's `f32` dtype
//! (standard type promotion — no rounding lost on `result`'s side), adds
//! in `f32`, and only THEN casts back down ONCE. That is ONE round point,
//! not two.
//!
//! An earlier revision of `ScaledCastAdd` rounded the SCALED DELTA to
//! `base`'s dtype FIRST (`bf16::from_f32(lora*scaling)`), then added and
//! rounded AGAIN — two round points, an extra one PEFT's own source never
//! takes. Fixed on both the CPU (`ops/scaled_cast_add.rs`) and CUDA
//! (`cuda/scaled_cast_add.cu`) arms in the same change (esc-046).
//!
//! ## The fixture
//!
//! `base ~ N(0, 100²)` (bf16), `delta ~ N(0, 3²)` (f32), `scaling =
//! 32/16 = 2.0` (a realistic LoRA `alpha/rank`), `4096` elements — matching
//! esc-046's own lead-measured reproduction (176/4096 elements differed,
//! max `|diff| = 1.0` at this exact amplitude). A Box-Muller transform over
//! an in-file, seeded `xorshift64` PRNG (family L: no untracked external
//! generator — the producer is this file, fully inspectable).

use candle_core::{Device, Tensor};
use half::bf16;
use jammi_kernels::ops::{apply2, ScaledCastAdd};

const N: usize = 4096;
const SCALING: f64 = 32.0 / 16.0;

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
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
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Box-Muller: one standard-normal draw per call.
    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// `base ~ N(0, sigma_base^2)` (bf16), `delta ~ N(0, sigma_delta^2)` (f32).
fn fixture(seed: u64, sigma_base: f64, sigma_delta: f64) -> (Vec<bf16>, Vec<f32>) {
    let mut rng = XorShift64::new(seed);
    let base: Vec<bf16> = (0..N)
        .map(|_| bf16::from_f32((rng.next_gauss() * sigma_base) as f32))
        .collect();
    let delta: Vec<f32> = (0..N)
        .map(|_| (rng.next_gauss() * sigma_delta) as f32)
        .collect();
    (base, delta)
}

/// PEFT-ordered reference (correct, post-fix): promote `base` to f32, add
/// the f32-scaled delta, round to bf16 ONCE.
fn peft_ordered(base: &[bf16], delta: &[f32], scaling: f64) -> Vec<bf16> {
    let scaling = scaling as f32;
    base.iter()
        .zip(delta)
        .map(|(&b, &d)| bf16::from_f32(b.to_f32() + d * scaling))
        .collect()
}

/// The pre-fix, mis-ordered formula (round the scaled delta to bf16 FIRST,
/// then add and round again) — kept ONLY to prove the fixture is
/// discriminating (clause 3, NON-VACUITY), never asserted as correct.
fn mis_ordered(base: &[bf16], delta: &[f32], scaling: f64) -> Vec<bf16> {
    let scaling = scaling as f32;
    base.iter()
        .zip(delta)
        .map(|(&b, &d)| {
            let rounded_delta = bf16::from_f32(d * scaling);
            bf16::from_f32(b.to_f32() + rounded_delta.to_f32())
        })
        .collect()
}

fn kernel_dispatch(base: &[bf16], delta: &[f32], scaling: f64) -> Vec<bf16> {
    let device = Device::Cpu;
    let base_t = Tensor::from_slice(base, base.len(), &device).unwrap();
    let delta_t = Tensor::from_slice(delta, delta.len(), &device).unwrap();
    let out = apply2(&base_t, &delta_t, ScaledCastAdd::new(scaling)).unwrap();
    out.flatten_all().unwrap().to_vec1().unwrap()
}

/// Non-vacuous discrimination floor (clause 3) — measured below; leaves
/// headroom for a different PRNG/toolchain build while refusing a fixture
/// that has degenerated to "the two formulas always agree".
const MIN_DISCRIMINATING: usize = 20;

#[test]
fn bf16_epilogue_matches_peft_rounding_not_the_round_delta_first_formula() {
    // esc-046's own lead-measured amplitude: base~N(0,100^2), delta~N(0,3^2).
    let (base, delta) = fixture(0x5EED_046A_u64, 100.0, 3.0);

    let peft = peft_ordered(&base, &delta, SCALING);
    let buggy = mis_ordered(&base, &delta, SCALING);
    let kernel = kernel_dispatch(&base, &delta, SCALING);

    assert_eq!(kernel.len(), N);

    // Clause (4): finiteness-affirmative, before any comparison.
    for i in 0..N {
        assert!(
            base[i].to_f32().is_finite()
                && delta[i].is_finite()
                && peft[i].to_f32().is_finite()
                && kernel[i].to_f32().is_finite(),
            "index {i}: a non-finite value slipped through (base={:?} delta={} peft={:?} \
             kernel={:?})",
            base[i],
            delta[i],
            peft[i],
            kernel[i]
        );
    }

    // Clause (3) NON-VACUITY: the fixture must actually separate the two
    // candidate formulas — computed from the two HAND formulas alone,
    // independent of what the real kernel returns.
    let discriminating = (0..N).filter(|&i| peft[i] != buggy[i]).count();
    assert!(
        discriminating >= MIN_DISCRIMINATING,
        "fixture is not discriminating: only {discriminating}/{N} elements separate the \
         PEFT-ordered formula from the round-delta-first one — this fixture would read GREEN \
         on a broken build regardless of the kernel; strengthen it before trusting this oracle"
    );

    // Clause (2) DEFECT (post-fix: GREEN). Raw bit pattern, never a
    // tolerance (esc-046's own stated assertion: `bf16::to_bits()`,
    // elementwise `==`).
    let mismatches: Vec<usize> = (0..N)
        .filter(|&i| kernel[i].to_bits() != peft[i].to_bits())
        .collect();
    let max_ulp_gap = mismatches
        .iter()
        .map(|&i| (kernel[i].to_bits() as i32 - peft[i].to_bits() as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        mismatches.is_empty(),
        "ScaledCastAdd's bf16 forward does NOT match PEFT's rounding order on {}/{N} elements \
         (esc-046/GH#374) — max raw-bit gap {max_ulp_gap}. First mismatch: idx={} base={:?} \
         delta={} kernel={:?} peft_ordered={:?}. Reverting `scaled_cast_add_bf16_f32` to round \
         the scaled delta to bf16 BEFORE the add reproduces every one of these mismatches — see \
         `ops/scaled_cast_add.rs`'s module doc.",
        mismatches.len(),
        mismatches.first().copied().unwrap_or(0),
        mismatches.first().map(|&i| base[i]),
        mismatches.first().map(|&i| delta[i]).unwrap_or(f32::NAN),
        mismatches.first().map(|&i| kernel[i]),
        mismatches.first().map(|&i| peft[i]),
    );
}

/// Second amplitude point: half the elements at ModernBERT-large's own
/// layer-18 residual magnitude (esc-045, `-6688`, ULP there `32`, not
/// `1.0`) and half at esc-046's own `base~100` fixture. A single-population
/// `base~N(0,6688²)` fixture was tried first and measured NON-discriminating
/// (9/4096, below the floor) — honestly reported, not hidden: at this
/// extreme a `delta~N(0,3²)` product is almost always far below one bf16
/// ULP (32), so both rounding orders round the delta's contribution away to
/// the SAME nearest representable value almost every time, independent of
/// which order the rounding happens in. Mixing in the `base~100` population
/// keeps the fixture discriminating while STILL exercising (and asserting
/// bit-exactness on) the layer-18-amplitude half.
#[test]
fn bf16_epilogue_matches_peft_rounding_at_layer18_residual_amplitude() {
    let (mut base, mut delta) = fixture(0x5EED_046B_u64, 6688.0, 3.0);
    let (base2, delta2) = fixture(0x5EED_046C_u64, 100.0, 3.0);
    base.extend(base2);
    delta.extend(delta2);
    let n = base.len();

    let peft = peft_ordered(&base, &delta, SCALING);
    let buggy = mis_ordered(&base, &delta, SCALING);
    let kernel = kernel_dispatch(&base, &delta, SCALING);

    for i in 0..n {
        assert!(
            base[i].to_f32().is_finite() && delta[i].is_finite() && kernel[i].to_f32().is_finite(),
            "index {i}: a non-finite value slipped through"
        );
    }

    let discriminating = (0..n).filter(|&i| peft[i] != buggy[i]).count();
    assert!(
        discriminating >= MIN_DISCRIMINATING,
        "fixture is not discriminating at layer-18 amplitude: only {discriminating}/{n} \
         elements separate the two formulas"
    );

    let mismatches = (0..n)
        .filter(|&i| kernel[i].to_bits() != peft[i].to_bits())
        .count();
    assert_eq!(
        mismatches, 0,
        "ScaledCastAdd's bf16 forward does NOT match PEFT's rounding order on {mismatches}/{n} \
         elements at layer-18 residual amplitude (|base|~6688, one bf16 ULP there = 32)"
    );
}
