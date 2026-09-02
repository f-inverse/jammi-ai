use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

use crate::layout_walk::StridedOffsets;

/// `out = base + cast_to(base.dtype())(lora * scaling)`, elementwise.
/// `scaling` is fixed at construction; `base` and `lora` may differ in
/// dtype (the whole reason this op exists — see below).
///
/// A generic Tensor-API primitive (family L: this crate names no consumer),
/// but the shape it was designed to fuse away is `jammi-lora`'s LoRA-site
/// epilogue: `base_out + scaling * lora_out`, where `base_out` is the frozen
/// matmul's output at the backbone dtype and `lora_out` is the small A/B
/// GEMM's output at the LoRA adapter's own dtype (today, always `F32` in
/// this workspace — a call-site fact, not a `candle_nn::VarBuilder::
/// from_varmap` API guarantee: its dtype is caller-supplied, not hardcoded;
/// see `jammi-lora`'s `LoraLinear::forward` module doc for the exact call
/// sites that make it `F32` today). The eager composition this replaces is
/// `[mul, cast, add]` — three tape nodes, one `CustomOp2` here.
///
/// STATELESS BY CONSTRUCTION: `Copy`, the argument `ops`'s module doc
/// makes — `scaling` is construction data, not runtime state.
///
/// ## Domain (family D)
///
/// NOT a broadcasting op: `base` and `lora` must have identical shape (a
/// mismatch is `Error::ShapeMismatchBinaryOp`, never silently broadcast).
/// CPU forward supports `base`/`lora` each independently `F32`, `BF16`, or
/// `F16` (seven combinations — the four `{F32,BF16}x{F32,BF16}` plus three
/// new `F16`+`F16`, `F16`+`F32`, `F32`+`F16`; `BF16`+`F16` and `F16`+`BF16`
/// are NOT implemented, on either arm — no kernel exists for mixing those
/// two 16-bit dtypes); any other dtype, or an unimplemented pair, is a
/// typed `Error::UnsupportedDTypeForOp`. The CUDA forward (feature-gated)
/// supports the SAME seven combinations (campaign #443 W2c added the three
/// F16 combinations via the SEPARATE `cuda/scaled_cast_add_f16.cu`
/// translation unit — monomorphic kernels, not template instantiations
/// sharing code with the F32/BF16 kernels) and additionally requires
/// contiguous storage.
///
/// ## The bf16 rounding model: f32-accumulate, round ONCE (esc-046 fix)
///
/// This now follows this crate's "f32-accumulate, round once" convention
/// after all — an EARLIER revision of this doc claimed PEFT
/// rounds the scaled delta to the base dtype BEFORE the add (two round
/// points) and cited that as the reason this op deliberately diverged from
/// that convention. That claim was never checked at PEFT source and is FALSE
/// (esc-046, GH#374): `peft/tuners/lora/layer.py`'s `Linear.forward`
/// (`peft==0.20.0`, lines 1044-1069, re-read at source on pod a100e
/// 2026-08-26) computes
/// `result = result + lora_B(lora_A(dropout(x))) * scaling` (line 1058,
/// inside the per-adapter loop) — torch's `+` PROMOTES the bf16 `result`
/// to the delta's `f32` dtype (standard type promotion, no rounding lost
/// on `result`'s side), adds in `f32`, and only THEN does
/// `result = result.to(torch_result_dtype)` (line 1069, AFTER the loop)
/// cast back down — ONE round point, at the very end, not two. This op
/// now reproduces THAT model: widen `base` to `f32` (lossless), add the
/// already-`f32` scaled `lora`, round the sum to `base`'s own dtype once.
/// The CPU (`F32`,`F32`)/(`BF16`,`F32`) combinations `jammi-lora`'s
/// admission predicate actually reaches stay BIT-EXACT against the eager
/// `[mul, add]` composition (see `tests/scaled_cast_add_oracles.rs`), not
/// merely within a stated ULP tolerance — the composition itself lost its
/// separate `cast` step for the same reason (see `jammi-lora`'s
/// `eager_epilogue`, corrected in the same round).
///
/// Confirmed with a live torch experiment (torch 2.11.0+cu128, peft
/// 0.20.0, A100, 2026-08-26): `base~N(0,100²)` bf16, `delta~N(0,3²)` f32,
/// `n=4096` → 176/4096 elements differ between the once-rounded
/// (`(base.float()+delta).to(bf16)`) and round-then-add
/// (`base.float()+delta.to(bf16).float()).to(bf16)`) formulas, max
/// `|diff| = 1.0` (one bf16 ULP at `|base|~100`; at ModernBERT-large's own
/// layer-18 residual magnitude, `-6688` (esc-045), one ULP there is `32` —
/// the extra rounding point this fix removes sat directly on the residual
/// stream every subsequent layer's forward AND recomputed-backward reads).
/// The same amplitude and discriminating-fixture claim is reproduced
/// in-tree, with its own independently-seeded fixture — see
/// `bf16_epilogue_matches_peft_rounding_not_the_round_delta_first_formula`
/// in `tests/scaled_cast_add_peft_rounding.rs` (a plain backtick span, not
/// an intra-doc link: that test lives in a separate integration-test
/// crate rustdoc cannot resolve into — an `[`...`]` link form here would
/// fail `cargo doc`'s `-D warnings` gate).
///
/// ## The other two combinations are NOT bit-exact — disclosed, not assumed
///
/// `(F32, BF16)` (a `BF16`-dtype `lora`) and `(BF16, BF16)` are accepted by
/// this op's domain (`cpu_fwd` implements all four `{F32,BF16} x {F32,BF16}`
/// combinations) but are UNREACHABLE today — `jammi-lora`'s admission
/// predicate never dispatches them, because `lora_a`/`lora_b` are always
/// `F32` in this workspace (see the "generic Tensor-API primitive" note
/// above). Should a future caller reach either combination, it will NOT be
/// bit-exact against eager: candle-core 0.11.0's own CPU `Affine` impl
/// (`cpu_backend/mod.rs`, `impl Map1 for Affine`: `let mul =
/// T::from_f64(self.0);` then `v * mul + add` in `T`'s own arithmetic)
/// rounds the SCALING CONSTANT itself to `lora`'s storage dtype BEFORE
/// multiplying — an extra bf16 rounding of `scaling` this op never
/// performs (it always widens `lora` to `f32` and multiplies by `scaling
/// as f32`, matching what `Affine` does when `lora`'s dtype is `F32`, but
/// NOT when it is `BF16`). The divergence is therefore keyed on LORA'S
/// dtype being `BF16`, not on `base`'s — both `(F32, BF16)` and `(BF16,
/// BF16)` inherit it identically. Measured and bounded (relative-with-
/// floor, the C4/C5 `bf16_close` pattern) in
/// `tests/scaled_cast_add_oracles.rs`'s `f32_base_bf16_lora_diverges_…`
/// and `bf16_base_bf16_lora_diverges_…` — not silently assumed equal just
/// because the crate is publishable and a future caller could reach this
/// combination.
#[derive(Debug, Clone, Copy)]
pub struct ScaledCastAdd {
    pub scaling: f64,
}

impl ScaledCastAdd {
    pub fn new(scaling: f64) -> Self {
        Self { scaling }
    }
}

impl super::sealed::Sealed for ScaledCastAdd {}

impl CustomOp2 for ScaledCastAdd {
    fn name(&self) -> &'static str {
        "scaled_cast_add"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1.dims() != l2.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: self.name(),
            });
        }
        match (s1, s2) {
            (CpuStorage::F32(base), CpuStorage::F32(lora)) => {
                let out = scaled_cast_add_f32_f32(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::F32(base), CpuStorage::BF16(lora)) => {
                let out = scaled_cast_add_f32_bf16(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(base), CpuStorage::F32(lora)) => {
                let out = scaled_cast_add_bf16_f32(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(base), CpuStorage::BF16(lora)) => {
                let out = scaled_cast_add_bf16_bf16(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (CpuStorage::F32(base), CpuStorage::F16(lora)) => {
                let out = scaled_cast_add_f32_f16(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::F16(base), CpuStorage::F32(lora)) => {
                let out = scaled_cast_add_f16_f32(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::F16(out), l1.shape().clone()))
            }
            (CpuStorage::F16(base), CpuStorage::F16(lora)) => {
                let out = scaled_cast_add_f16_f16(self.scaling, base, l1, lora, l2);
                Ok((CpuStorage::F16(out), l1.shape().clone()))
            }
            (s1, s2) => {
                let unsupported = |d: DType| !matches!(d, DType::F32 | DType::BF16 | DType::F16);
                if unsupported(s1.dtype()) {
                    Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name()))
                } else {
                    Err(Error::UnsupportedDTypeForOp(s2.dtype(), self.name()))
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::scaled_cast_add::cuda_fwd(self.scaling, s1, l1, s2, l2)
    }

    /// `d_base = dy` (identity: the add is linear in `base` with unit
    /// coefficient, and the straight-through convention this op's rounding
    /// model uses treats every `round_to(..)` as the identity function for
    /// gradient purposes — the same convention a plain `to_dtype` cast's own
    /// backward uses in this codebase, e.g. `LoraLinear::forward`'s eager
    /// `.to_dtype(..)` composes through ordinary differentiable candle ops
    /// whose cast backward is exactly this).
    ///
    /// `d_lora = cast_to(lora.dtype())(dy) * scaling` — the chain rule
    /// through the (straight-through) round and the scalar multiply, in
    /// THAT order (cast first, then scale), matching the C6 contract's
    /// stated backward and the eager composition's own gradient (`dy` ->
    /// `to_dtype(f32)` -> `* scaling`, since `lora_out` before its own cast
    /// is `F32` in every reachable configuration).
    ///
    /// ALWAYS returns `Some` for both slots — `Tensor::is_variable()` is
    /// NOT a safe gate here (see `ops`'s module doc for the
    /// full argument: it cannot distinguish a true frozen leaf from an
    /// INTERMEDIATE on a path to a `Var`, and candle's own backward walk
    /// requires a populated gradient for the latter regardless).
    fn bwd(
        &self,
        _base: &Tensor,
        lora: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let d_base = grad_res.clone();
        let dy_lora_dtype = if grad_res.dtype() == lora.dtype() {
            grad_res.clone()
        } else {
            grad_res.to_dtype(lora.dtype())?
        };
        let d_lora = dy_lora_dtype.affine(self.scaling, 0.0)?;
        Ok((Some(d_base), Some(d_lora)))
    }
}

/// Fixed fold order (family J): `StridedOffsets` walked in the same
/// sequence for a given pair of layouts every time.
fn scaled_cast_add_f32_f32(
    scaling: f64,
    base: &[f32],
    lb: &Layout,
    lora: &[f32],
    ll: &Layout,
) -> Vec<f32> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| base[ib] + lora[il] * scaling)
        .collect()
}

fn scaled_cast_add_f32_bf16(
    scaling: f64,
    base: &[f32],
    lb: &Layout,
    lora: &[bf16],
    ll: &Layout,
) -> Vec<f32> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| base[ib] + lora[il].to_f32() * scaling)
        .collect()
}

/// The reachable production combination (`BF16` backbone, `F32` LoRA
/// adapter): esc-046 fix (GH#374) — widens `base` to `f32` (lossless),
/// adds the already-`f32` scaled `lora` (no intermediate bf16-rounded
/// `delta`), and rounds the sum to `bf16` ONCE. Matches PEFT's
/// `Linear.forward` (`result + lora_out*scaling` under torch's own bf16->
/// f32 promotion, THEN one `.to(torch_result_dtype)` cast — see this op's
/// own module doc) bit-for-bit on the CPU (`F32`,`F32`)/(`BF16`,`F32`)
/// combinations `jammi-lora`'s admission predicate actually reaches (see
/// `tests/scaled_cast_add_oracles.rs`).
fn scaled_cast_add_bf16_f32(
    scaling: f64,
    base: &[bf16],
    lb: &Layout,
    lora: &[f32],
    ll: &Layout,
) -> Vec<bf16> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| bf16::from_f32(base[ib].to_f32() + lora[il] * scaling))
        .collect()
}

fn scaled_cast_add_bf16_bf16(
    scaling: f64,
    base: &[bf16],
    lb: &Layout,
    lora: &[bf16],
    ll: &Layout,
) -> Vec<bf16> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| bf16::from_f32(base[ib].to_f32() + lora[il].to_f32() * scaling))
        .collect()
}

/// [`scaled_cast_add_f32_bf16`]'s exact twin, substituting `half::f16`.
fn scaled_cast_add_f32_f16(
    scaling: f64,
    base: &[f32],
    lb: &Layout,
    lora: &[f16],
    ll: &Layout,
) -> Vec<f32> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| base[ib] + lora[il].to_f32() * scaling)
        .collect()
}

/// [`scaled_cast_add_bf16_f32`]'s exact twin, substituting `half::f16` —
/// this is the F16-backbone analog of the reachable production
/// combination, per the per-op f16 reference-regime table
/// (`docs/maintainer/cuda-kernel-guide.md`): f32-accumulate, round-once.
fn scaled_cast_add_f16_f32(
    scaling: f64,
    base: &[f16],
    lb: &Layout,
    lora: &[f32],
    ll: &Layout,
) -> Vec<f16> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| f16::from_f32(base[ib].to_f32() + lora[il] * scaling))
        .collect()
}

/// [`scaled_cast_add_bf16_bf16`]'s exact twin, substituting `half::f16`.
fn scaled_cast_add_f16_f16(
    scaling: f64,
    base: &[f16],
    lb: &Layout,
    lora: &[f16],
    ll: &Layout,
) -> Vec<f16> {
    let scaling = scaling as f32;
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(ll))
        .map(|(ib, il)| f16::from_f32(base[ib].to_f32() + lora[il].to_f32() * scaling))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn scaled_cast_add(scaling: f64, base: &Tensor, lora: &Tensor) -> Result<Tensor> {
        crate::ops::apply2(base, lora, ScaledCastAdd::new(scaling))
    }

    #[test]
    fn cpu_fwd_f32_f32_matches_hand_computed_values() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap();
        let lora = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let out = scaled_cast_add(2.0, &base, &lora)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn cpu_fwd_bf16_f32_matches_hand_computed_values() {
        let device = Device::Cpu;
        let base =
            Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(2.0)], (2,), &device).unwrap();
        let lora = Tensor::from_slice(&[4.0f32, -8.0], (2,), &device).unwrap();
        let out = scaled_cast_add(0.5, &base, &lora)
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap();
        let expected = [
            bf16::from_f32(1.0 + bf16::from_f32(0.5 * 4.0).to_f32()),
            bf16::from_f32(2.0 + bf16::from_f32(0.5 * -8.0).to_f32()),
        ];
        assert_eq!(out, expected);
    }

    /// F16's exact twin of `cpu_fwd_bf16_f32_matches_hand_computed_values`
    /// above — the F16-backbone analog of the reachable production
    /// combination.
    #[test]
    fn cpu_fwd_f16_f32_matches_hand_computed_values() {
        let device = Device::Cpu;
        let base =
            Tensor::from_slice(&[f16::from_f32(1.0), f16::from_f32(2.0)], (2,), &device).unwrap();
        let lora = Tensor::from_slice(&[4.0f32, -8.0], (2,), &device).unwrap();
        let out = scaled_cast_add(0.5, &base, &lora)
            .unwrap()
            .to_vec1::<f16>()
            .unwrap();
        let expected = [
            f16::from_f32(1.0 + f16::from_f32(0.5 * 4.0).to_f32()),
            f16::from_f32(2.0 + f16::from_f32(0.5 * -8.0).to_f32()),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn zero_scaling_leaves_base_unchanged_at_every_supported_combo() {
        // The esc-031 golden's premise, exercised directly at the kernel
        // level: `scaling == 0` (or, equivalently, `lora == 0`) must be a
        // bit-exact no-op on `base` for every supported dtype pair.
        let device = Device::Cpu;
        for (base_bf16, lora_bf16) in [(false, false), (false, true), (true, false), (true, true)] {
            let base_v = [1.5f32, -2.25, 100.0];
            let lora_v = [7.0f32, -3.5, 0.25];
            let base = if base_bf16 {
                let v: Vec<bf16> = base_v.iter().map(|&x| bf16::from_f32(x)).collect();
                Tensor::from_slice(&v, (3,), &device).unwrap()
            } else {
                Tensor::from_slice(&base_v, (3,), &device).unwrap()
            };
            let lora = if lora_bf16 {
                let v: Vec<bf16> = lora_v.iter().map(|&x| bf16::from_f32(x)).collect();
                Tensor::from_slice(&v, (3,), &device).unwrap()
            } else {
                Tensor::from_slice(&lora_v, (3,), &device).unwrap()
            };
            let out = scaled_cast_add(0.0, &base, &lora).unwrap();
            assert_eq!(out.dtype(), base.dtype());
            let out_f32 = out.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
            let base_f32 = base.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(
                out_f32, base_f32,
                "base_bf16={base_bf16} lora_bf16={lora_bf16}: zero scaling must be a bit-exact no-op"
            );
        }
    }

    #[test]
    fn empty_tensor_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let lora = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let out = scaled_cast_add(3.0, &base, &lora)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn shape_mismatch_is_refused_not_broadcast() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let lora = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let err = scaled_cast_add(1.0, &base, &lora)
            .expect_err("mismatched shapes must not silently broadcast");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn unsupported_dtype_is_refused_with_a_typed_error() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[1u8, 2], (2,), &device).unwrap();
        let lora = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let err = scaled_cast_add(1.0, &base, &lora)
            .expect_err("U8 has no scaled_cast_add CPU implementation for `base`");
        assert!(matches!(err, Error::UnsupportedDTypeForOp(..)));
    }

    #[test]
    fn non_contiguous_view_is_still_correct_on_cpu() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)
            .unwrap()
            .t()
            .unwrap();
        let lora = Tensor::from_slice(&[0.0f32; 6], (3, 2), &device).unwrap();
        assert!(!base.is_contiguous());
        let out = scaled_cast_add(1.0, &base, &lora)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }
}
