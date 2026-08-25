//! Fused, in-place AdamW step kernels — the multi-tensor-AdamW lever.
//!
//! See `scratchpad/design-multi-tensor-adamw.md` for the full design study
//! (launch/memcpy reconciliation against the perf census, the `CustomOp` vs
//! `InplaceOp` decision, and the CPU-bit-exact/CUDA-tolerance acceptance
//! split). Summary: `candle_core::CustomOp1/2/3::{cpu_fwd,cuda_fwd}` each
//! return a BRAND-NEW `(Storage, Shape)` — applying one to a `Var`'s tensor
//! still needs a `Var::set` afterward (a D2D memcpy) to splice the result
//! back in. `candle_core::InplaceOp1/2/3` (`Tensor::inplace_op1/2/3`) mutate
//! an existing tensor's storage directly, through candle's own
//! `Arc<RwLock<Storage>>` write guard — the ONLY sound, public (no
//! vendoring/patching candle) path to a writable device buffer, since
//! `Tensor::storage_mut`/`storage_mut_and_layout` are `pub(crate)` in
//! candle-core 0.11.0.
//!
//! AdamW's update splits into three in-place mutations, each already
//! matching one `InplaceOpN` arity:
//!
//! 1. `m[i] = beta1*m[i] + (1-beta1)*g[i]` — [`AdamMomentUpdate`] via
//!    `InplaceOp2` (`m` mutated, `g` read).
//! 2. `v[i] = beta2*v[i] + (1-beta2)*g[i]^2` — [`AdamMomentUpdate`] again
//!    (`square_grad = true`).
//! 3. `theta[i] = theta[i]*(1-lr*wd) - lr*(m[i]*scale_m)/(sqrt(v[i]*scale_v)+eps)`
//!    — [`AdamThetaUpdate`] via `InplaceOp3` (`theta` mutated, `m`/`v` read
//!    — called AFTER (1)/(2), so `m`/`v` hold this step's freshly-EMA'd
//!    values). [`adamw_step_fused`] composes the three, in order, for one
//!    `Var`.
//!
//! Domain (family D): F32 only (the optimizer is not gated by esc-045's
//! BF16 boundary — LoRA `theta`/moments are always F32, `adapter.rs`'s
//! `ComputePrecision::backbone_dtype` only affects the frozen backbone). Not
//! a broadcasting op: every tensor triple must share one shape (refused,
//! `Error::ShapeMismatchBinaryOp`, never silently broadcast/truncated). CPU
//! forward walks arbitrary strides (`StridedOffsets`, matching every other
//! op in this crate); the CUDA forward (feature-gated) requires contiguous
//! storage — LoRA's `theta`/`first_moment`/`second_moment` `Var`s are always
//! freshly allocated and therefore always contiguous
//! (`jammi-lora/src/lora_linear.rs:374-375`,`422-423`); a non-contiguous
//! gradient (e.g. a transposed backward output) on CUDA is refused with
//! `Error::RequiresContiguous` rather than silently misread.
//!
//! **Bit-identity**: per-element, each of the three update rules above is
//! evaluated as the SAME sequence of individually-rounded `f32` operations
//! (`*`, `+`, `-`, `/`, `.sqrt()`) that `candle_nn`-style eager composition
//! (`m*beta1 + g*(1-beta1)`, etc.) already performs — floating-point
//! ELEMENTWISE operations have no cross-element interaction, so folding
//! candle's separate full-array passes (affine, affine, add) into one
//! per-element expression changes nothing about that element's own
//! rounding (`a+b` is exactly commutative in IEEE 754, unlike associativity
//! across MULTIPLE elements, which this op never reorders). On the CPU arm
//! this is provably bit-exact (Rust's `f32` arithmetic does not
//! auto-contract into FMA without an explicit `.mul_add()` call — see the
//! design study's §3, and every existing op in this module,
//! e.g. `ops::axpy`, already relies on the same fact for its own
//! `assert_eq!` oracles). On the CUDA arm this is a documented TOLERANCE
//! claim, not bit-exact: nvcc's `--fmad=true` default (on regardless of
//! `--use_fast_math`, which stays off — `build.rs`) may contract an
//! `a*b+c`-shaped sub-expression into a single-rounding FMA, and — the key
//! correction versus a naive reading — candle's OWN eager CUDA kernels
//! (`candle-kernels-0.11.0/src/affine.cu`'s `x*mul+add`) are compiled by a
//! build script this crate does not control and are exposed to the exact
//! same contraction, so "bit-identical to the eager CUDA chain" is not an
//! honestly provable claim either way; this crate's established convention
//! (every C2-C7 fused op's `tests/cuda_parity.rs`) states a tolerance
//! instead, and this op follows it.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, Error, InplaceOp2, InplaceOp3, Layout, Result, Tensor};

use crate::layout_walk::StridedOffsets;

/// In-place `dst[i] = beta*dst[i] + (1-beta)*(g[i] or g[i]^2)` — the Adam
/// first/second-moment EMA update. `square_grad = false` for the first
/// moment (`m`), `true` for the second (`v`, where `g[i]^2` matches
/// candle's own `Tensor::sqr` — `op.rs:591`, `v * v`, a single `f32`
/// rounding, reproduced verbatim below).
///
/// STATELESS BY CONSTRUCTION (see `ops`'s module doc): `Copy`, so no owned
/// interior-mutable field can leak state between calls. `beta`/
/// `square_grad` are construction data, not runtime state.
#[derive(Debug, Clone, Copy)]
pub struct AdamMomentUpdate {
    pub beta: f64,
    pub square_grad: bool,
}

impl AdamMomentUpdate {
    pub fn new(beta: f64, square_grad: bool) -> Self {
        Self { beta, square_grad }
    }
}

impl super::sealed::Sealed for AdamMomentUpdate {}

impl InplaceOp2 for AdamMomentUpdate {
    fn name(&self) -> &'static str {
        "adamw_moment_update"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<()> {
        if l1.dims() != l2.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: self.name(),
            });
        }
        match (s1, s2) {
            (CpuStorage::F32(m), CpuStorage::F32(g)) => {
                let beta = self.beta as f32;
                let one_minus_beta = (1.0 - self.beta) as f32;
                let square_grad = self.square_grad;
                let m_offsets: Vec<usize> = StridedOffsets::from_layout(l1).collect();
                for (im, ig) in m_offsets.into_iter().zip(StridedOffsets::from_layout(l2)) {
                    let gv = g[ig];
                    // `Tensor::sqr` is `v * v` (op.rs:591) — one `f32`
                    // rounding, reproduced here exactly (not `gv.powi(2)`,
                    // which LLVM may lower differently).
                    let gv = if square_grad { gv * gv } else { gv };
                    m[im] = beta * m[im] + one_minus_beta * gv;
                }
                Ok(())
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
        s1: &mut candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<()> {
        crate::cuda::adamw_step::moment_update_cuda_fwd(self.beta, self.square_grad, s1, l1, s2, l2)
    }
}

/// In-place `theta[i] = theta[i]*one_minus_lr_lambda - lr*m_hat[i]/(sqrt(v_hat[i])+eps)`
/// where `m_hat[i] = m[i]*scale_m`, `v_hat[i] = v[i]*scale_v` — the
/// bias-corrected, decoupled-weight-decay AdamW parameter update. Reads
/// `m`/`v` (must already hold THIS step's post-EMA values — i.e. this op
/// runs after two [`AdamMomentUpdate`] calls, never before).
///
/// `one_minus_lr_lambda`, `scale_m`, `scale_v`, `eps`, `lr` are all f64,
/// each cast to `f32` at use time exactly as candle's own `affine`/
/// `Add<f64>`/`Mul<f64>` do (`WithDType::from_f64` for `f32` is a plain
/// `v as f32` — `dtype.rs:235`), so the constants match the eager chain's
/// rounding bit-for-bit on top of matching its operation order.
#[derive(Debug, Clone, Copy)]
pub struct AdamThetaUpdate {
    pub one_minus_lr_lambda: f64,
    pub scale_m: f64,
    pub scale_v: f64,
    pub eps: f64,
    pub lr: f64,
}

impl AdamThetaUpdate {
    /// `lr_lambda = lr * weight_decay` (candle's own formula,
    /// `adamw.rs:84`, computed in f64 exactly as here) folded into
    /// `one_minus_lr_lambda = 1.0 - lr_lambda` up front — matching eager's
    /// `theta.as_tensor() * (1f64 - lr_lambda)` operand exactly.
    pub fn new(lr: f64, weight_decay: f64, scale_m: f64, scale_v: f64, eps: f64) -> Self {
        let lr_lambda = lr * weight_decay;
        Self {
            one_minus_lr_lambda: 1.0 - lr_lambda,
            scale_m,
            scale_v,
            eps,
            lr,
        }
    }
}

impl super::sealed::Sealed for AdamThetaUpdate {}

impl InplaceOp3 for AdamThetaUpdate {
    fn name(&self) -> &'static str {
        "adamw_theta_update"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()> {
        if l1.dims() != l2.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: self.name(),
            });
        }
        if l1.dims() != l3.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l3.shape().clone(),
                op: self.name(),
            });
        }
        match (s1, s2, s3) {
            (CpuStorage::F32(theta), CpuStorage::F32(m), CpuStorage::F32(v)) => {
                let one_minus_lr_lambda = self.one_minus_lr_lambda as f32;
                let scale_m = self.scale_m as f32;
                let scale_v = self.scale_v as f32;
                let eps = self.eps as f32;
                let lr = self.lr as f32;
                let theta_offsets: Vec<usize> = StridedOffsets::from_layout(l1).collect();
                let m_offsets: Vec<usize> = StridedOffsets::from_layout(l2).collect();
                for ((it, im), iv) in theta_offsets
                    .into_iter()
                    .zip(m_offsets)
                    .zip(StridedOffsets::from_layout(l3))
                {
                    let m_hat = m[im] * scale_m;
                    let v_hat = v[iv] * scale_v;
                    let denom = v_hat.sqrt() + eps;
                    let adjusted_grad = m_hat / denom;
                    theta[it] = theta[it] * one_minus_lr_lambda - adjusted_grad * lr;
                }
                Ok(())
            }
            (s1, s2, s3) if s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype() => {
                let (lhs, rhs) = if s1.dtype() != s2.dtype() {
                    (s1.dtype(), s2.dtype())
                } else {
                    (s1.dtype(), s3.dtype())
                };
                Err(Error::DTypeMismatchBinaryOp {
                    lhs,
                    rhs,
                    op: self.name(),
                })
            }
            (s1, _, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &mut candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<()> {
        crate::cuda::adamw_step::theta_update_cuda_fwd(*self, s1, l1, s2, l2, s3, l3)
    }
}

/// The whole fused AdamW step for ONE `Var`: two [`AdamMomentUpdate`] calls
/// (first then second moment — order matters only in that both must finish
/// before the third call, since [`AdamThetaUpdate`] reads their post-EMA
/// values) followed by one [`AdamThetaUpdate`] call. Three kernel launches,
/// zero `Var::set`/memcpy — see this module's doc and the design study for
/// why. `scale_m`/`scale_v` are the caller's bias-correction terms
/// (`1/(1-beta1^t)`, `1/(1-beta2^t)` — depend on the step counter `t`, so
/// the caller computes them fresh each step, exactly as `adamw.rs:87-88`
/// does).
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_fused(
    theta: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    grad: &Tensor,
    beta1: f64,
    beta2: f64,
    scale_m: f64,
    scale_v: f64,
    lr: f64,
    weight_decay: f64,
    eps: f64,
) -> Result<()> {
    super::apply_inplace2(first_moment, grad, AdamMomentUpdate::new(beta1, false))?;
    super::apply_inplace2(second_moment, grad, AdamMomentUpdate::new(beta2, true))?;
    super::apply_inplace3(
        theta,
        first_moment,
        second_moment,
        AdamThetaUpdate::new(lr, weight_decay, scale_m, scale_v, eps),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The literal eager chain `AdamW::step` runs
    /// (`crates/jammi-ai/src/fine_tune/adamw.rs:81-107` at HEAD `2c1a68d`),
    /// reproduced here as an independent oracle using plain `candle_core`
    /// tensor ops (NOT this crate's fused op) — this is the numpy-first-style
    /// reference this op's bit-identity claim is measured against (family
    /// F), not a re-statement of the fused kernel's own code.
    #[allow(clippy::too_many_arguments)]
    fn eager_step(
        theta: &Tensor,
        m: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta1: f64,
        beta2: f64,
        lr: f64,
        weight_decay: f64,
        eps: f64,
        t: i32,
    ) -> (Tensor, Tensor, Tensor) {
        let lr_lambda = lr * weight_decay;
        let scale_m = 1f64 / (1f64 - beta1.powi(t));
        let scale_v = 1f64 / (1f64 - beta2.powi(t));
        let next_m = ((m * beta1).unwrap() + (g * (1.0 - beta1)).unwrap()).unwrap();
        let next_v = ((v * beta2).unwrap() + (g.sqr().unwrap() * (1.0 - beta2)).unwrap()).unwrap();
        let m_hat = (&next_m * scale_m).unwrap();
        let v_hat = (&next_v * scale_v).unwrap();
        let next_theta = (theta * (1f64 - lr_lambda)).unwrap();
        let adjusted_grad = (m_hat / (v_hat.sqrt().unwrap() + eps).unwrap()).unwrap();
        let next_theta = (next_theta - (adjusted_grad * lr).unwrap()).unwrap();
        (next_theta, next_m, next_v)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_fused(
        theta: &Tensor,
        m: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta1: f64,
        beta2: f64,
        lr: f64,
        weight_decay: f64,
        eps: f64,
        t: i32,
    ) {
        let scale_m = 1f64 / (1f64 - beta1.powi(t));
        let scale_v = 1f64 / (1f64 - beta2.powi(t));
        adamw_step_fused(
            theta,
            m,
            v,
            g,
            beta1,
            beta2,
            scale_m,
            scale_v,
            lr,
            weight_decay,
            eps,
        )
        .unwrap();
    }

    fn assert_bit_identical(a: &Tensor, b: &Tensor) {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "element {i}: fused={x} ({:#010x}) eager={y} ({:#010x})",
                x.to_bits(),
                y.to_bits()
            );
        }
    }

    /// Fresh `theta`/`m`/`v`/`g` Vars for one test — `m`/`v` start at zero
    /// (matching `AdamW::new`'s zero-init, `adamw.rs:52-53`), `theta`/`g`
    /// take caller-supplied values.
    fn setup(theta_v: &[f32], g_v: &[f32], shape: (usize,)) -> (Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(theta_v, shape, &dev).unwrap();
        let m = Tensor::zeros(shape, DType::F32, &dev).unwrap();
        let v = Tensor::zeros(shape, DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(g_v, shape, &dev).unwrap();
        (theta, m, v, g)
    }

    #[test]
    fn one_step_matches_the_eager_chain_bit_for_bit() {
        let (theta, m, v, g) = setup(&[0.5, -1.25, 3.0, 0.0], &[0.1, -0.2, 0.05, 0.0], (4,));
        let (want_theta, want_m, want_v) =
            eager_step(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        assert_bit_identical(&theta, &want_theta);
        assert_bit_identical(&m, &want_m);
        assert_bit_identical(&v, &want_v);
    }

    /// At least 3 consecutive steps (the bias-correction scalars change
    /// with `t`), lr changing mid-run, weight_decay not equal to zero —
    /// reproduces the resume-sensitive part of `AdamW::step`'s state
    /// (step_t-dependent `scale_m`/`scale_v`) at production-realistic
    /// cadence.
    #[test]
    fn three_consecutive_steps_with_a_changing_lr_match_bit_for_bit() {
        let (theta, m, v, g) = setup(&[1.0, -2.0, 0.0, 4.5], &[0.3, 0.1, -0.4, 0.02], (4,));
        let (mut theta_ref, mut m_ref, mut v_ref, g_ref) =
            setup(&[1.0, -2.0, 0.0, 4.5], &[0.3, 0.1, -0.4, 0.02], (4,));
        let lrs = [1e-3, 5e-4, 2e-3];
        for (t, &lr) in lrs.iter().enumerate() {
            let t = (t + 1) as i32;
            let (want_theta, want_m, want_v) = eager_step(
                &theta_ref, &m_ref, &v_ref, &g_ref, 0.9, 0.999, lr, 0.05, 1e-8, t,
            );
            run_fused(&theta, &m, &v, &g, 0.9, 0.999, lr, 0.05, 1e-8, t);
            assert_bit_identical(&theta, &want_theta);
            assert_bit_identical(&m, &want_m);
            assert_bit_identical(&v, &want_v);
            theta_ref = want_theta;
            m_ref = want_m;
            v_ref = want_v;
        }
    }

    #[test]
    fn zero_weight_decay_matches_bit_for_bit() {
        let (theta, m, v, g) = setup(&[2.0, -3.0], &[0.5, -0.5], (2,));
        let (want_theta, want_m, want_v) =
            eager_step(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.0, 1e-8, 1);
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.0, 1e-8, 1);
        assert_bit_identical(&theta, &want_theta);
        assert_bit_identical(&m, &want_m);
        assert_bit_identical(&v, &want_v);
    }

    /// Boundary/degenerate oracle (family D): a zero-element tensor is a
    /// clean no-op, not a crash or a spurious element.
    #[test]
    fn empty_tensor_is_a_no_op_not_an_error() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[] as &[f32], (0,), &dev).unwrap();
        let m = Tensor::zeros((0,), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((0,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[] as &[f32], (0,), &dev).unwrap();
        adamw_step_fused(&theta, &m, &v, &g, 0.9, 0.999, 1.0, 1.0, 1e-3, 0.01, 1e-8).unwrap();
        assert!(theta.to_vec1::<f32>().unwrap().is_empty());
    }

    /// Boundary oracle: a single-element ("single point") tensor.
    #[test]
    fn single_element_matches_bit_for_bit() {
        let (theta, m, v, g) = setup(&[7.5], &[-0.01], (1,));
        let (want_theta, want_m, want_v) =
            eager_step(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        assert_bit_identical(&theta, &want_theta);
        assert_bit_identical(&m, &want_m);
        assert_bit_identical(&v, &want_v);
    }

    /// Boundary oracle: `theta`/`g` identical everywhere (a common
    /// initialization for the LoRA `B` matrix, which starts at zero — see
    /// `lora_linear.rs:408`), so `g == 0` too — degenerate zero-gradient
    /// step, must not divide by zero or blow up (`eps` guards the
    /// denominator).
    #[test]
    fn identical_values_and_zero_gradient_stay_finite() {
        let (theta, m, v, g) = setup(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], (3,));
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        let out: Vec<f32> = theta.to_vec1().unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
        // theta*(1-lr_lambda) - 0 = 0*(1-lr_lambda) = 0 exactly.
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    /// Finiteness affirmative + signal (family F): a real (nonzero)
    /// gradient must move theta away from its start (‖Δθ‖ > 0), and every
    /// output must be finite.
    #[test]
    fn a_real_gradient_moves_theta_and_stays_finite() {
        let (theta, m, v, g) = setup(&[1.0, 1.0, 1.0], &[0.2, -0.3, 0.1], (3,));
        let before: Vec<f32> = theta.to_vec1().unwrap();
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-2, 0.01, 1e-8, 1);
        let after: Vec<f32> = theta.to_vec1().unwrap();
        assert!(after.iter().all(|x| x.is_finite()));
        let delta: f32 = before.iter().zip(&after).map(|(b, a)| (a - b).abs()).sum();
        assert!(delta > 0.0, "theta did not move: {before:?} -> {after:?}");
    }

    #[test]
    fn shape_mismatch_between_theta_and_grad_is_refused() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[1.0f32, 2.0], (2,), &dev).unwrap();
        let m = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
        let err = adamw_step_fused(&theta, &m, &v, &g, 0.9, 0.999, 1.0, 1.0, 1e-3, 0.01, 1e-8)
            .expect_err("mismatched theta/grad shapes must not silently broadcast");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn unsupported_dtype_is_refused_with_a_typed_error() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let m = Tensor::zeros((2,), DType::F64, &dev).unwrap();
        let v = Tensor::zeros((2,), DType::F64, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let err = adamw_step_fused(&theta, &m, &v, &g, 0.9, 0.999, 1.0, 1.0, 1e-3, 0.01, 1e-8)
            .expect_err("F64 has no adamw_step_fused CPU implementation (F32 only)");
        assert!(matches!(err, Error::UnsupportedDTypeForOp(..)));
    }

    /// NEGATIVE CONTROL (family F): a deliberately WRONG reference — the
    /// exact single-rounding FMA contraction the design study's §3 warns
    /// CUDA's default `--fmad=true` may apply to an `a*b+c`-shaped
    /// sub-expression, computed here explicitly via `f32::mul_add` for the
    /// moment-update step (`beta*m + one_minus_beta*g` becomes a single
    /// fused-rounding `beta.mul_add(m, one_minus_beta*g)`) instead of the
    /// kernel's two separately-rounded multiplies + one add. This proves
    /// the bit-identity oracle has power: if a fused-vs-eager comparison
    /// could not distinguish an FMA-contracted computation from the
    /// non-contracted one this op actually performs, the CPU bit-identity
    /// claim in every other test in this file would be unfalsifiable. Swept
    /// over several representative `(m, g)` pairs (a single unlucky pair
    /// CAN coincide bit-for-bit by chance — generic inputs do not) and
    /// requires at least one mismatch, which is the only way to
    /// demonstrate a REAL power difference rather than assert a specific
    /// magic value forever.
    #[test]
    fn negative_control_an_fma_contracted_reference_is_not_bit_identical() {
        let beta: f32 = 0.9;
        let one_minus_beta: f32 = 0.1;
        let pairs: [(f32, f32); 6] = [
            (0.1234567, -0.7654321),
            (std::f32::consts::PI / 4.0, std::f32::consts::E / 3.0),
            (-2.0_f32.sqrt(), 3.0_f32.sqrt()),
            (0.000123, 987.654),
            (-0.5, 0.5),
            (1.0000001, -1.0000002),
        ];
        let mut any_mismatch = false;
        for &(m, g) in &pairs {
            // The real kernel's sequence (`AdamMomentUpdate::cpu_fwd`):
            // two separately-rounded multiplies, then one rounded add.
            let real = beta * m + one_minus_beta * g;
            // WRONG: a single-rounding FMA contraction of the first
            // product into the add — exactly what nvcc's `--fmad=true`
            // default risks doing to an `a*b+c` pattern, and exactly what
            // this crate's bit-identity claim must NOT silently accept.
            let fma_contracted = beta.mul_add(m, one_minus_beta * g);
            if real.to_bits() != fma_contracted.to_bits() {
                any_mismatch = true;
            }
        }
        assert!(
            any_mismatch,
            "an FMA-contracted reference must diverge from the kernel's \
             two-separate-roundings computation on at least one of these \
             representative inputs (this is the negative control proving \
             the bit-identity oracle has power)"
        );
    }

    /// Non-contiguous view oracle (CPU walks arbitrary strides): a
    /// transposed `g` must still read the right elements.
    #[test]
    fn non_contiguous_grad_view_is_still_correct_on_cpu() {
        let dev = Device::Cpu;
        let theta = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let m = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let g_base = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev).unwrap();
        let g = g_base.t().unwrap();
        assert!(!g.is_contiguous());
        let g_contig = g.contiguous().unwrap();

        let want = eager_step(&theta, &m, &v, &g_contig, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        let scale_m = 1f64 / (1f64 - 0.9f64.powi(1));
        let scale_v = 1f64 / (1f64 - 0.999f64.powi(1));
        adamw_step_fused(
            &theta, &m, &v, &g, 0.9, 0.999, scale_m, scale_v, 1e-3, 0.01, 1e-8,
        )
        .unwrap();
        assert_bit_identical(&theta, &want.0);
    }
}
