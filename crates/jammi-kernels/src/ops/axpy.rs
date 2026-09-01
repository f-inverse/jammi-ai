use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

use crate::layout_walk::StridedOffsets;

/// `y' = alpha * x + y`, elementwise, `alpha` fixed at construction.
///
/// STATELESS BY CONSTRUCTION (see `ops`'s module doc, and
/// [`KernelOp`](crate::ops::KernelOp)): `Axpy` is `Copy`, which
/// structurally forbids any OWNED `Cell`/`RefCell`/`Mutex`/atomic/
/// heap-owned field — there is no per-instance mutable state this type
/// could use to leak information from one `fwd` call into a later,
/// unrelated `bwd` call. `alpha` is construction data, not runtime state.
/// `Axpy` implements the crate-private `Sealed` marker below, which is
/// what makes it satisfy [`KernelOp`](crate::ops::KernelOp) — the bound
/// [`apply2`](crate::ops::apply2) requires.
///
/// Domain (family D): this is NOT a broadcasting op. `x` and `y` must have
/// identical shape; a shape mismatch is refused (`Error::ShapeMismatchBinaryOp`)
/// rather than silently broadcast or truncated. CPU forward supports F32,
/// F64, BF16, F16 (typed error for any other dtype, or a lhs/rhs dtype
/// mismatch); the CUDA forward (feature-gated) supports F32, BF16 and
/// additionally requires contiguous storage (a raw-pointer kernel has no
/// flat linear index for a strided/broadcast view — `Error::RequiresContiguous`).
/// F16's CPU arm (`axpy_f16`, added as an oracle reference — see
/// `docs/maintainer/cuda-kernel-guide.md`'s per-op f16 reference-regime
/// table) mirrors `axpy_bf16`'s f32-accumulate-round-once regime; no CUDA
/// F16 dispatch arm exists yet, so admission never routes an F16 tensor
/// here on CUDA (K2: no Hold-without-dispatch).
#[derive(Debug, Clone, Copy)]
pub struct Axpy {
    pub alpha: f64,
}

impl Axpy {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl super::sealed::Sealed for Axpy {}

impl CustomOp2 for Axpy {
    fn name(&self) -> &'static str {
        "axpy"
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
            (CpuStorage::F32(x), CpuStorage::F32(y)) => {
                let out = axpy_f32(self.alpha as f32, x, l1, y, l2);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::F64(x), CpuStorage::F64(y)) => {
                let out = axpy_f64(self.alpha, x, l1, y, l2);
                Ok((CpuStorage::F64(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(y)) => {
                let out = axpy_bf16(self.alpha, x, l1, y, l2);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (CpuStorage::F16(x), CpuStorage::F16(y)) => {
                let out = axpy_f16(self.alpha, x, l1, y, l2);
                Ok((CpuStorage::F16(out), l1.shape().clone()))
            }
            (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            }),
            (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
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
        crate::cuda::axpy::cuda_fwd(self.alpha, s1, l1, s2, l2)
    }

    /// `bwd` composes ordinary `Tensor` ops (`affine`, `clone`), which
    /// already dispatch through candle's own CPU / CUDA / Metal backends —
    /// this is the single backward for every device; `CustomOp2` has no
    /// separate `cuda_bwd` slot.
    ///
    /// ALWAYS returns `Some` for both slots. `Tensor::is_variable()` looks
    /// like the right predicate for "does this input need a gradient" but
    /// is NOT: it is two-state (is / is not a `Var`) over a three-state
    /// reality — a true external constant leaf (no upstream op, never
    /// needs a gradient), a `Var` (always needs one), and an INTERMEDIATE
    /// on a path to a `Var` (e.g. `x = w.affine(2.0, 0.0)` — `is_variable()
    /// == false`, exactly like a true constant, but candle's own backward
    /// walk (`Tensor::sorted_nodes`, backprop.rs:47-158) still marks it
    /// `track_grad = true` and REQUIRES a populated gradient for it once
    /// its turn comes up in the backward loop — `grads.remove(node)`
    /// panics with "candle internal error - grad not populated"
    /// (backprop.rs:174) if nothing upstream ever called
    /// `grads.or_insert`/`insert` on it first. Gating on `is_variable()`
    /// (the design this replaced) returned `None` for exactly that
    /// intermediate case and reproduced that panic — see
    /// `tests/oracles.rs::bwd_chains_through_an_intermediate_non_variable_node`.
    /// The only predicate that COULD make the real three-way distinction —
    /// `Tensor::op()` — is `pub(crate)` in candle-core 0.11 and unreachable
    /// from this crate. Correctness over the micro-optimization: always
    /// compute and return both gradients. For a true constant leaf this is
    /// a harmless wasted allocation (candle's own walk never pushes a
    /// `track_grad = false` node onto `sorted_nodes`, so the extra
    /// `GradStore` entry this creates is simply never consumed).
    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((
            Some(grad_res.affine(self.alpha, 0.0)?),
            Some(grad_res.clone()),
        ))
    }
}

/// Fixed fold order (family J): the offsets from `StridedOffsets` are
/// walked in the same sequence for a given pair of layouts every time, so
/// repeated forwards over identical inputs are bit-reproducible.
fn axpy_f32(alpha: f32, x: &[f32], lx: &Layout, y: &[f32], ly: &Layout) -> Vec<f32> {
    StridedOffsets::from_layout(lx)
        .zip(StridedOffsets::from_layout(ly))
        .map(|(ix, iy)| alpha * x[ix] + y[iy])
        .collect()
}

fn axpy_f64(alpha: f64, x: &[f64], lx: &Layout, y: &[f64], ly: &Layout) -> Vec<f64> {
    StridedOffsets::from_layout(lx)
        .zip(StridedOffsets::from_layout(ly))
        .map(|(ix, iy)| alpha * x[ix] + y[iy])
        .collect()
}

/// BF16 accumulates in f32 (scope decision 4 of the fused-kernels plan: the
/// CPU implementation matches the CUDA kernel's accumulation semantics),
/// rounding back to bf16 once. This is a deliberate precision choice, not
/// bit-identical to candle's own native-bf16 `affine`+`add` composition
/// (which multiplies/adds directly in bf16, with more intermediate
/// rounding) — the fused-vs-eager oracle states a tolerance for this dtype
/// rather than asserting exact equality, and says why.
fn axpy_bf16(alpha: f64, x: &[bf16], lx: &Layout, y: &[bf16], ly: &Layout) -> Vec<bf16> {
    let alpha = alpha as f32;
    StridedOffsets::from_layout(lx)
        .zip(StridedOffsets::from_layout(ly))
        .map(|(ix, iy)| bf16::from_f32(alpha * x[ix].to_f32() + y[iy].to_f32()))
        .collect()
}

/// F16 accumulates in f32, rounding back to f16 once — the exact same
/// regime as [`axpy_bf16`] above, substituting `half::f16`. Not
/// bit-identical to candle's own native-f16 `affine`+`add` composition
/// either, for the same reason `axpy_bf16`'s doc states.
fn axpy_f16(alpha: f64, x: &[f16], lx: &Layout, y: &[f16], ly: &Layout) -> Vec<f16> {
    let alpha = alpha as f32;
    StridedOffsets::from_layout(lx)
        .zip(StridedOffsets::from_layout(ly))
        .map(|(ix, iy)| f16::from_f32(alpha * x[ix].to_f32() + y[iy].to_f32()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn axpy(alpha: f64, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        crate::ops::apply2(x, y, Axpy::new(alpha))
    }

    #[test]
    fn cpu_fwd_f32_matches_hand_computed_values() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let y = Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap();
        let out = axpy(2.0, &x, &y).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn cpu_fwd_f64_matches_hand_computed_values() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f64, -2.0], (2,), &device).unwrap();
        let y = Tensor::from_slice(&[0.5f64, 0.5], (2,), &device).unwrap();
        let out = axpy(-1.5, &x, &y).unwrap().to_vec1::<f64>().unwrap();
        assert_eq!(out, vec![-1.0, 3.5]);
    }

    #[test]
    fn cpu_fwd_bf16_matches_f32_accumulation_rounded_once() {
        let device = Device::Cpu;
        let xv = [bf16::from_f32(1.5), bf16::from_f32(-2.25)];
        let yv = [bf16::from_f32(0.25), bf16::from_f32(1.0)];
        let x = Tensor::from_slice(&xv, (2,), &device).unwrap();
        let y = Tensor::from_slice(&yv, (2,), &device).unwrap();
        let out = axpy(2.0, &x, &y).unwrap().to_vec1::<bf16>().unwrap();
        let expected = [
            bf16::from_f32(2.0 * 1.5 + 0.25),
            bf16::from_f32(2.0 * -2.25 + 1.0),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn cpu_fwd_f16_matches_f32_accumulation_rounded_once() {
        let device = Device::Cpu;
        let xv = [f16::from_f32(1.5), f16::from_f32(-2.25)];
        let yv = [f16::from_f32(0.25), f16::from_f32(1.0)];
        let x = Tensor::from_slice(&xv, (2,), &device).unwrap();
        let y = Tensor::from_slice(&yv, (2,), &device).unwrap();
        let out = axpy(2.0, &x, &y).unwrap().to_vec1::<f16>().unwrap();
        let expected = [
            f16::from_f32(2.0 * 1.5 + 0.25),
            f16::from_f32(2.0 * -2.25 + 1.0),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn empty_tensor_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let y = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let out = axpy(3.0, &x, &y).unwrap().to_vec1::<f32>().unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn shape_mismatch_is_refused_not_broadcast() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let err = axpy(1.0, &x, &y).expect_err("mismatched shapes must not silently broadcast");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn dtype_mismatch_between_inputs_is_refused() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32], (1,), &device).unwrap();
        let y = Tensor::from_slice(&[1.0f64], (1,), &device).unwrap();
        let err = axpy(1.0, &x, &y).expect_err("mismatched input dtypes must be refused");
        assert!(matches!(err, Error::DTypeMismatchBinaryOp { .. }));
    }

    #[test]
    fn unsupported_dtype_is_refused_with_a_typed_error() {
        let device = Device::Cpu;
        // U8 is a real candle dtype this op does not implement — refuse
        // rather than silently reinterpret the bytes (family D / K2).
        let x = Tensor::from_slice(&[1u8, 2], (2,), &device).unwrap();
        let y = Tensor::from_slice(&[1u8, 2], (2,), &device).unwrap();
        let err = axpy(1.0, &x, &y).expect_err("U8 has no axpy CPU implementation");
        assert!(matches!(err, Error::UnsupportedDTypeForOp(..)));
    }

    #[test]
    fn non_contiguous_view_is_still_correct_on_cpu() {
        // The CPU forward walks arbitrary strides (StridedOffsets); a
        // transposed, non-contiguous view must still compute the right
        // values, not silently read the wrong elements.
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)
            .unwrap()
            .t()
            .unwrap();
        let y = Tensor::from_slice(&[0.0f32; 6], (3, 2), &device).unwrap();
        assert!(!x.is_contiguous());
        let out = axpy(1.0, &x, &y).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }
}
