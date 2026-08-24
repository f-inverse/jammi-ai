//! Device-side, counter-based dropout: `y = x * mask * (1 / p_keep)`
//! (inverted dropout), where `mask` is NEVER materialized as a tensor —
//! each element's KEEP/DROP decision is computed in-kernel from
//! [`crate::philox::philox_draw`] and discarded immediately after use.
//!
//! This is the fused replacement for `jammi-lora`'s old
//! `DropoutStream::draw_mask` + `Tensor::from_vec` + elementwise `mul`
//! composition: a host-side `SplitMix64` PRNG filled a full-size `Vec<f32>`
//! mask, copied it to the activation's device, and multiplied — a real H2D
//! transfer and a retained mask tensor on the backward tape (candle's
//! `Binary::Mul` backward computes and stores a full-size gradient FOR the
//! mask operand and never frees it, `backprop.rs:197-204`, since the mask
//! is not itself a graph leaf `sorted_nodes` walks). See `jammi-lora`'s
//! `LoraLinear::forward` module doc for the measured cost this replaces
//! (2.9x step time, 16.7 GB at dropout 0.05, #352).
//!
//! ## The rejected mechanism, and why (wip/device-side-dropout, superseded)
//!
//! A prior attempt (`wip/device-side-dropout`, preserved, not merged) tried
//! `Device::set_seed(key)` followed by `Tensor::rand` to draw a per-device
//! mask. That is a NON-ATOMIC read-modify-write of process-global device
//! RNG state (`set_seed` takes and releases a mutex, `rand_uniform`
//! re-takes it separately) — `LoraLinear` is deliberately `Sync`, so a
//! concurrent draw on the same device could silently yield the wrong mask,
//! and a regenerating backward could regenerate a DIFFERENT mask than
//! forward applied: wrong gradients, green tests. This op has NO such
//! state: every draw is a pure function of `(seed, layer_id, forward_idx,
//! element_index)` via Philox, so there is nothing global to race.
//!
//! ## Counter mapping and provenance
//!
//! See `crate::philox`'s module doc for the full Random123 citation
//! (BSD-3-Clause; NOT curand) and the exact `(seed, layer_id, forward_idx,
//! element_index) -> counter/key` mapping. `element_index` is this
//! tensor's LOGICAL flat index (`0..elem_count`), not a raw storage
//! offset — the domain below requires contiguous storage precisely so
//! "logical index" and "storage index" coincide (offset by the layout's
//! own `start_offset`, handled by `contiguous_offsets()` before any
//! per-element loop runs).
//!
//! ## KEEP/DROP: an INTEGER threshold, computed once, host-side, in f64
//!
//! `threshold: u64 = round(p_keep * 2^32)`, computed ONCE in
//! [`DropoutFused::new`] (never recomputed per-element or per-launch, and
//! never recomputed device-side — CUDA reads the identical `u64` this
//! constructor computes). The decision is `(draw as u64) < threshold`,
//! comparing entirely in `u64` space so `p == 0.0` (`p_keep == 1.0`,
//! `threshold == 2^32`) fits WITHOUT wrapping (a `u32` threshold would
//! wrap `2^32` to `0`, silently dropping every element at `p == 0.0` — the
//! exact failure mode a `u64` threshold structurally avoids; see
//! [`tests::p_zero_is_a_bit_exact_no_op`]).
//!
//! Research context (2026-08-24, cited per the C7 contract): PyTorch/JAX
//! compute the decision as a FLOAT compare (`curand_uniform` is `(0,1]`,
//! compared `< p`; JAX's `uniform < p`). FlashAttention-2's own dropout
//! (`csrc/flash_attn/src/dropout.h`) instead compares an 8-BIT INTEGER
//! against `floor(255*p)` — an explicitly disclosed, measurable rescale
//! bias of `(floor(255p)+1)/256` vs the requested `1/p_keep`, at 2^-8
//! granularity. This op's `u64`-at-2^-32-granularity threshold has the
//! IDENTICAL shape of approximation (an integer CDF cutoff, not a
//! constructed float uniform), just four billion times finer — negligible
//! in practice, but the same category of bias, disclosed rather than
//! assumed away. This is a DELIBERATE choice for a different reason than
//! FA2's (which optimizes for warp-level 8-bit packing): here the goal is
//! EXACT CPU/CUDA parity at the DECISION level with no float construction
//! on either device — no `curand_uniform`, no `uint_to_uniform_float`,
//! nothing that could round differently on the two backends. Bit-parity
//! with PyTorch's own dropout stream is explicitly NOT a goal (unattainable
//! without curand's exact philox-offset/launch-striding mapping); parity
//! with a PyTorch reference is DISTRIBUTIONAL ONLY (keep-rate and the
//! inverted-dropout rescale, checked statistically — see
//! [`tests::keep_rate_matches_p_within_a_binomial_bound`]).
//!
//! `p` is validated to `[0.0, 1.0)` in [`DropoutFused::new`] (a typed
//! `candle_core::Error`, not a silently-clamped or silently-NaN-propagated
//! value) — `jammi-lora`'s `lora_dropout` was UNVALIDATED before this
//! commit (`config.rs`'s `lora_dropout: Option<f32>` field took any `f32`),
//! and this op independently re-validates rather than trusting its caller,
//! per family D's "validate at every numeric edge".
//!
//! ## The applied scale: pinned bit-identical CPU/CUDA
//!
//! The KEPT-element value is `x * scale` — a single, lone multiply (never
//! part of a longer fused expression on this op, so there is no fmad
//! contraction question: nvcc's default `--fmad=true` only contracts a
//! multiply INTO a following add, and there is no add here). Plain Rust
//! `f32 * f32` and CUDA's `__fmul_rn` are the SAME IEEE-754
//! round-to-nearest-even single operation — this op's CUDA kernel uses
//! `__fmul_rn` explicitly (rather than relying on nvcc's default `*`,
//! which this build's un-pinned `--fmad` default could in principle treat
//! differently in a future nvcc version if this expression ever grew a
//! neighboring add) so the pinning is stated in the kernel text itself,
//! not merely inferred from "there happens to be no add nearby" — the same
//! doctrine C1 established for `Axpy`'s FMA-contraction disclosure, applied
//! here as a POSITIVE guarantee instead of a disclosed gap.
//!
//! ## No save-for-backward (candle 0.11): bwd IS fwd
//!
//! Dropout's forward is `y_i = mask_i * scale * x_i`, a per-element LINEAR
//! map whose "matrix" (`mask_i * scale`) depends ONLY on `(seed, layer_id,
//! forward_idx, i)` — never on `x`'s value. Its Jacobian-vector product is
//! therefore the IDENTICAL map applied to the upstream gradient:
//! `dx_i = mask_i * scale * dy_i`, using the SAME decision (regenerated
//! from the SAME counter, never stored) and the SAME scale. Concretely:
//! [`DropoutFused::bwd`] calls [`super::apply1`] with `*self` — THE VERY
//! SAME construction data — applied to `grad_res` instead of `x`. This is
//! not a coincidental shortcut; it is the mathematical content of "bwd
//! regenerates the SAME decision from the SAME counter, dx = dy * mask *
//! scale" (this commit's own design), and it structurally eliminates wip
//! branch's finding #2 (candle's `Binary::Mul` backward retaining a
//! full-size gradient FOR the mask): there is no mask tensor anywhere on
//! the tape to retain — `DropoutFused` is one `CustomOp1` node forward,
//! and the identical op is one MORE `CustomOp1` node backward, with no
//! third tensor (no mask) ever created.
//!
//! ## Domain (family D)
//!
//! Contiguous storage only (`Layout::contiguous_offsets`) — a raw-pointer
//! per-element kernel has no flat linear index for a strided view, and
//! contiguity is what makes "logical index" and "storage index" coincide
//! (see the counter-mapping section above). CPU/CUDA both support `F32`
//! and `BF16` (this crate's two production activation dtypes); any other
//! dtype is a typed `Error::UnsupportedDTypeForOp`. An empty tensor
//! (`elem_count == 0`) is a no-op, not an error.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, Error, Layout, Result, Shape, Tensor};
use half::bf16;

use crate::philox::{philox4x32_10, philox_draw};

/// `2^32` as an `f64` — used exactly once per [`DropoutFused::new`] call to
/// compute `threshold`, never per-element or per-launch.
const TWO_POW_32: f64 = 4_294_967_296.0;

/// Device-side counter-based dropout. See the module doc for the full
/// design (counter mapping, integer-threshold decision, why `bwd` is
/// literally `fwd` applied to the upstream gradient).
///
/// STATELESS BY CONSTRUCTION (`Copy`, the same [`super::KernelOp`] argument
/// every op in this crate makes): `seed`/`layer_id`/`forward_idx` are the
/// counter-mapping's own construction data (never mutated after
/// construction — the CALLER, `jammi-lora`'s `DropoutMasks`, owns the
/// advancing forward counter and passes a snapshot of it in here fresh
/// for every training forward); `threshold`/`scale` are derived from `p`
/// ONCE, host-side, at construction.
#[derive(Debug, Clone, Copy)]
pub struct DropoutFused {
    seed: u64,
    layer_id: u32,
    forward_idx: u32,
    /// `round(p_keep * 2^32)`, computed once in `f64` — see the module
    /// doc's "KEEP/DROP" section. Compared against a `u32` draw widened to
    /// `u64`, so `p == 0.0` (`threshold == 2^32`) never wraps.
    threshold: u64,
    /// `1.0 / p_keep`, computed once — the inverted-dropout rescale
    /// applied to every KEPT element.
    scale: f32,
}

impl DropoutFused {
    /// `p` must be finite and in `[0.0, 1.0)` — `p == 1.0` would drop
    /// every element and make `scale` infinite; `p < 0.0` or `p` non-finite
    /// (including `NaN`, which fails every ordinary comparison silently —
    /// family F's "a naive comparison silently passes on NaN" hazard,
    /// refused here by requiring `is_finite()` explicitly rather than
    /// relying on the range check alone) is refused with a typed error
    /// rather than silently clamped.
    pub fn new(seed: u64, layer_id: u32, forward_idx: u32, p: f32) -> Result<Self> {
        if !p.is_finite() || !(0.0..1.0).contains(&p) {
            return Err(Error::Msg(format!(
                "dropout_fused: p must be finite and in [0.0, 1.0), got {p}"
            )));
        }
        let p_keep = 1.0_f64 - f64::from(p);
        let threshold = (p_keep * TWO_POW_32).round() as u64;
        let scale = (1.0_f64 / p_keep) as f32;
        Ok(Self {
            seed,
            layer_id,
            forward_idx,
            threshold,
            scale,
        })
    }

    /// `true` when `draw` (Philox's first output word, widened to `u64` —
    /// see `crate::philox::philox_draw`) falls under this instance's
    /// threshold, i.e. the element is KEPT.
    #[inline]
    fn keeps(&self, element_index: u64) -> bool {
        let draw = philox_draw(self.seed, self.layer_id, self.forward_idx, element_index);
        u64::from(draw) < self.threshold
    }

    /// `(seed, layer_id, forward_idx, threshold, scale)` — this instance's
    /// full construction data, as the CUDA launch's own scalar arguments.
    /// `pub(crate)`: only `crate::cuda::dropout::cuda_fwd` reaches this
    /// (the CPU arm above reads the private fields directly, being in the
    /// same module); the CUDA glue lives in a sibling module and needs an
    /// explicit accessor rather than reaching through private fields.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_launch_args(&self) -> (u64, u32, u32, u64, f32) {
        (
            self.seed,
            self.layer_id,
            self.forward_idx,
            self.threshold,
            self.scale,
        )
    }
}

impl super::sealed::Sealed for DropoutFused {}

impl CustomOp1 for DropoutFused {
    fn name(&self) -> &'static str {
        "dropout_fused"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match s1 {
            CpuStorage::F32(x) => Ok((
                CpuStorage::F32(dropout_f32(self, &x[o1..o2])),
                l1.shape().clone(),
            )),
            CpuStorage::BF16(x) => Ok((
                CpuStorage::BF16(dropout_bf16(self, &x[o1..o2])),
                l1.shape().clone(),
            )),
            s => Err(Error::UnsupportedDTypeForOp(s.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::dropout::cuda_fwd(self, s1, l1)
    }

    /// See the module doc's "no save-for-backward: bwd IS fwd" section:
    /// applying THIS SAME `Copy` instance to `grad_res` regenerates the
    /// identical KEEP/DROP decision from the identical counter and applies
    /// the identical scale — exactly `dx = dy * mask * scale`, with no
    /// mask tensor materialized on either pass. `arg`/`res` are unused:
    /// the decision depends on neither (it is a pure function of this
    /// op's own construction data and the element's index), which is the
    /// structural reason no mask needs to be cached anywhere.
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(super::apply1(grad_res, *self)?))
    }
}

/// Fixed fold order (family J): ascending logical index, no reduction —
/// every element is independent, so this is simply "iterate 0..n".
fn dropout_f32(params: &DropoutFused, x: &[f32]) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            if params.keeps(i as u64) {
                v * params.scale
            } else {
                0.0
            }
        })
        .collect()
}

/// BF16: widen to `f32`, apply the same lone multiply, round to `bf16`
/// once — this crate's usual single-rounding convention (`Axpy`'s bf16
/// arm), and there is only ever ONE rounding point here regardless (unlike
/// `ScaledCastAdd`'s two-round PEFT-matching model): a KEPT element is
/// `bf16::from_f32(x_i_as_f32 * scale)`, a DROPPED element is an exact
/// `bf16::ZERO`, never a rounded quantity.
fn dropout_bf16(params: &DropoutFused, x: &[bf16]) -> Vec<bf16> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            if params.keeps(i as u64) {
                bf16::from_f32(v.to_f32() * params.scale)
            } else {
                bf16::ZERO
            }
        })
        .collect()
}

/// TEST-SUPPORT ONLY — not part of the dropout feature itself. A
/// `CustomOp1` whose sole purpose is putting `crate::philox`'s CPU
/// implementation AND its CUDA device-function counterpart
/// (`cuda/dropout.cu`'s `philox_kat`) through the SAME `apply1` dispatch
/// path every real op in this crate uses, so Random123's published
/// known-answer test vectors can be asserted against BOTH without a
/// separate raw-buffer-download API. It ignores its input tensor entirely
/// (any 1-element tensor on the target device is a valid dummy argument)
/// and always returns the 4 raw Philox output words for the `(counter,
/// key)` this instance was constructed with, as a `U32` tensor of shape
/// `(4,)`. See `crate::philox`'s module doc for the KAT vectors themselves
/// and `tests/cuda_parity.rs`'s `philox_kat_vectors_match_on_cuda` (this
/// op's only real caller) for the CUDA-side half of the proof.
#[derive(Debug, Clone, Copy)]
pub struct PhiloxKatProbe {
    counter: [u32; 4],
    key: [u32; 2],
}

impl PhiloxKatProbe {
    pub fn new(counter: [u32; 4], key: [u32; 2]) -> Self {
        Self { counter, key }
    }
}

impl super::sealed::Sealed for PhiloxKatProbe {}

impl CustomOp1 for PhiloxKatProbe {
    fn name(&self) -> &'static str {
        "philox_kat_probe"
    }

    fn cpu_fwd(&self, _s1: &CpuStorage, _l1: &Layout) -> Result<(CpuStorage, Shape)> {
        let out = philox4x32_10(self.counter, self.key);
        Ok((CpuStorage::U32(out.to_vec()), Shape::from((4,))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        _l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::dropout::cuda_philox_kat(self.counter, self.key, s1.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    fn dropout(seed: u64, layer_id: u32, forward_idx: u32, p: f32, x: &Tensor) -> Result<Tensor> {
        let op = DropoutFused::new(seed, layer_id, forward_idx, p)?;
        crate::ops::apply1(x, op)
    }

    /// `PhiloxKatProbe` through the CPU arm of the SAME `apply1` dispatch
    /// path the CUDA parity suite uses — a sanity check that the op
    /// wrapper itself (shape/dtype plumbing) doesn't perturb the raw
    /// `philox4x32_10` values already pinned directly in
    /// `crate::philox::tests`.
    #[test]
    fn philox_kat_probe_matches_philox4x32_10_on_cpu() {
        let device = Device::Cpu;
        let dummy = Tensor::from_slice(&[0.0f32], (1,), &device).unwrap();
        let vectors: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
            ),
            (
                [0xffffffff; 4],
                [0xffffffff, 0xffffffff],
                [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0],
                [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
            ),
        ];
        for (ctr, key, expected) in vectors {
            let op = PhiloxKatProbe::new(ctr, key);
            let out: Vec<u32> = crate::ops::apply1(&dummy, op).unwrap().to_vec1().unwrap();
            assert_eq!(out, expected.to_vec(), "ctr={ctr:?} key={key:?}");
        }
    }

    #[test]
    fn p_zero_is_a_bit_exact_no_op() {
        // p_keep == 1.0 => threshold == 2^32 exactly, which must NOT wrap
        // to 0 in a u32 (it never does here: the comparison runs in u64
        // space) — a wrapped threshold would silently drop EVERY element
        // at p == 0.0, the exact failure this test would catch.
        let device = Device::Cpu;
        let v = [1.0f32, -2.5, 3.75, 0.0, 100.0];
        let x = Tensor::from_slice(&v, (5,), &device).unwrap();
        let out: Vec<f32> = dropout(1, 0, 0, 0.0, &x).unwrap().to_vec1().unwrap();
        assert_eq!(out, v, "p=0.0 must be a bit-exact no-op on every element");
    }

    #[test]
    fn p_at_or_above_one_is_a_typed_refusal() {
        assert!(
            DropoutFused::new(1, 0, 0, 1.0).is_err(),
            "p == 1.0 must be refused"
        );
        assert!(
            DropoutFused::new(1, 0, 0, 1.5).is_err(),
            "p > 1.0 must be refused"
        );
        assert!(
            DropoutFused::new(1, 0, 0, -0.1).is_err(),
            "p < 0.0 must be refused"
        );
    }

    /// Family F: a non-finite `p` must fail the validation explicitly, not
    /// silently pass a naive comparison (`NaN < 1.0` and `NaN >= 0.0` are
    /// both `false`, so a range check ALONE would let NaN through the
    /// `!(0.0..1.0).contains` branch as "in range" — this op additionally
    /// requires `is_finite()`).
    #[test]
    fn nan_p_is_refused_not_silently_accepted() {
        assert!(
            DropoutFused::new(1, 0, 0, f32::NAN).is_err(),
            "NaN must be refused, not silently treated as in-range"
        );
        assert!(DropoutFused::new(1, 0, 0, f32::INFINITY).is_err());
    }

    #[test]
    fn same_counter_reproduces_the_same_mask() {
        let device = Device::Cpu;
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_slice(&v, (1000,), &device).unwrap();
        let a: Vec<f32> = dropout(11, 3, 5, 0.3, &x).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = dropout(11, 3, 5, 0.3, &x).unwrap().to_vec1().unwrap();
        assert_eq!(
            a, b,
            "identical (seed, layer, forward_idx) must draw the identical mask"
        );
    }

    #[test]
    fn different_forward_idx_draws_a_different_mask() {
        let device = Device::Cpu;
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01 + 1.0).collect();
        let x = Tensor::from_slice(&v, (1000,), &device).unwrap();
        let a: Vec<f32> = dropout(11, 3, 5, 0.3, &x).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = dropout(11, 3, 6, 0.3, &x).unwrap().to_vec1().unwrap();
        assert_ne!(a, b, "a different forward index must draw a different mask");
    }

    /// The keep-rate oracle (a non-vacuous, measured, numpy-comparable
    /// statistic — family F): over a large draw, the fraction kept must be
    /// within a stated binomial bound of `p_keep`. `n = 1_000_000`,
    /// `p = 0.05` (the shipped default, per #352): std dev of a Binomial
    /// `(n, p_keep)` count is `sqrt(n*p*(1-p))` ≈ 217 for `p=0.05`, so 6
    /// std devs ≈ 1302 elements ≈ 0.0013 of `n` — a generous, explicitly
    /// derived, non-arbitrary bound (not a "just wide enough to pass" one).
    #[test]
    fn keep_rate_matches_p_within_a_binomial_bound() {
        let device = Device::Cpu;
        let n = 1_000_000usize;
        let p = 0.05f32;
        let v = vec![1.0f32; n];
        let x = Tensor::from_slice(&v, (n,), &device).unwrap();
        let out: Vec<f32> = dropout(4242, 7, 1, p, &x).unwrap().to_vec1().unwrap();
        let kept = out.iter().filter(|&&y| y != 0.0).count();
        let keep_rate = kept as f64 / n as f64;
        let p_keep = 1.0 - p as f64;
        let std_dev = (n as f64 * p_keep * (1.0 - p_keep)).sqrt() / n as f64;
        let bound = 6.0 * std_dev;
        assert!(
            (keep_rate - p_keep).abs() < bound,
            "keep_rate {keep_rate} vs p_keep {p_keep}, bound {bound}"
        );
        // Inverted-dropout scaling: every kept element is scaled by 1/p_keep,
        // so the mean of `out` over ALL n elements (kept and dropped) should
        // track the mean of the all-ones input (an unbiased estimator
        // argument, not a per-element identity).
        let mean: f64 = out.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.02,
            "inverted-dropout mean {mean} should track the input mean (1.0) within 0.02"
        );
    }

    #[test]
    fn empty_tensor_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let out: Vec<f32> = dropout(1, 0, 0, 0.3, &x).unwrap().to_vec1().unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn unsupported_dtype_is_refused_with_a_typed_error() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1u8, 2, 3], (3,), &device).unwrap();
        let op = DropoutFused::new(1, 0, 0, 0.3).unwrap();
        let err =
            crate::ops::apply1(&x, op).expect_err("U8 has no dropout_fused CPU implementation");
        assert!(matches!(err, Error::UnsupportedDTypeForOp(..)));
    }

    #[test]
    fn non_contiguous_view_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!x.is_contiguous());
        let op = DropoutFused::new(1, 0, 0, 0.3).unwrap();
        let err = crate::ops::apply1(&x, op).expect_err("non-contiguous input must be refused");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    /// Oracle 3 (regeneration): backward through a `Var` must reproduce
    /// `dx = dy * mask * scale` EXACTLY where the element was kept, and
    /// exactly `0` where it was dropped — proving `bwd`'s regenerated
    /// decision matches `fwd`'s (structurally guaranteed by `bwd` calling
    /// `apply1(grad_res, *self)`, but pinned here as an end-to-end
    /// autograd oracle, not just a unit-level code-reading argument).
    #[test]
    fn backward_reproduces_the_same_decision_as_forward() {
        let device = Device::Cpu;
        let n = 5000usize;
        let xv: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.001).collect();
        let x = Var::from_tensor(&Tensor::from_slice(&xv, (n,), &device).unwrap()).unwrap();
        let op = DropoutFused::new(99, 2, 4, 0.4).unwrap();
        let y = crate::ops::apply1(x.as_tensor(), op).unwrap();
        // Seed the upstream gradient with a non-trivial pattern so a sign
        // or index error would not hide behind an all-ones dy.
        let dy_v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let dy = Tensor::from_slice(&dy_v, (n,), &device).unwrap();
        let loss = (&y * &dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dx: Vec<f32> = grads.get(&x).unwrap().to_vec1().unwrap();
        let y_v: Vec<f32> = y.to_vec1().unwrap();

        for i in 0..n {
            if y_v[i] == 0.0 {
                assert_eq!(
                    dx[i], 0.0,
                    "dx must be exactly 0 where forward dropped element {i}"
                );
            } else {
                let expected = dy_v[i] * op.scale;
                assert!(
                    (dx[i] - expected).abs() < 1e-4,
                    "dx[{i}] = {} vs expected dy*scale = {expected}",
                    dx[i]
                );
            }
        }
    }

    /// Eval-mode / dtype-domain oracle at the kernel level: `F32` and
    /// `BF16` both work, and a fixed configuration produces the SAME
    /// KEEP/DROP pattern regardless of the tensor's dtype (the decision is
    /// index-keyed, not value-keyed, so dtype cannot perturb it).
    #[test]
    fn keep_drop_pattern_is_identical_across_supported_dtypes() {
        let device = Device::Cpu;
        let n = 256usize;
        let v: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
        let vb: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
        let x_f32 = Tensor::from_slice(&v, (n,), &device).unwrap();
        let x_bf16 = Tensor::from_slice(&vb, (n,), &device).unwrap();

        let out_f32: Vec<f32> = dropout(5, 1, 1, 0.25, &x_f32).unwrap().to_vec1().unwrap();
        let out_bf16: Vec<f32> = dropout(5, 1, 1, 0.25, &x_bf16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        for i in 0..n {
            assert_eq!(
                out_f32[i] == 0.0,
                out_bf16[i] == 0.0,
                "element {i}: KEEP/DROP must agree across dtypes"
            );
        }
    }
}
