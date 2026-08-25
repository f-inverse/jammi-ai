//! Fused, in-place AdamW step kernels — the multi-tensor-AdamW lever.
//!
//! See `scratchpad/design-multi-tensor-adamw.md` for the full design study
//! (launch/memcpy reconciliation against the perf census, the `CustomOp` vs
//! `InplaceOp` decision). Summary: `candle_core::CustomOp1/2/3::{cpu_fwd,
//! cuda_fwd}` each return a BRAND-NEW `(Storage, Shape)` — applying one to a
//! `Var`'s tensor still needs a `Var::set` afterward (a D2D memcpy) to
//! splice the result back in. `candle_core::InplaceOp1/2/3`
//! (`Tensor::inplace_op1/2/3`) mutate an existing tensor's storage directly,
//! through candle's own `Arc<RwLock<Storage>>` write guard — the ONLY
//! sound, public (no vendoring/patching candle) path to a writable device
//! buffer, since `Tensor::storage_mut`/`storage_mut_and_layout` are
//! `pub(crate)` in candle-core 0.11.0.
//!
//! **The unit of fusion is one `Var`'s step, not a multi-tensor batch**:
//! despite the design study's working name, this crate does NOT batch
//! multiple `Var`s into one launch (candle has no such primitive without
//! vendoring). Each `Var` still gets its own three kernel launches
//! ([`adamw_step_fused_t`], below) — the lever is eliminating the
//! `Var::set` D2D memcpy `CustomOp` would otherwise force per `Var` per
//! step (672 `Var`s × steps in the design study's census), not eliminating
//! per-`Var` launch overhead itself.
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
//!    values). [`adamw_step_fused_t`] composes the three, in order, for one
//!    `Var`, deriving `scale_m`/`scale_v` from the step counter `t` itself
//!    (see that function's doc for why a caller-supplied `scale_m`/`scale_v`
//!    — [`adamw_step_fused`]'s now-`#[deprecated]` shape — is a
//!    bias-correction footgun). [`AdamWParams`] bundles the five scalar
//!    hyperparameters that stay fixed across steps.
//!
//! Domain (family D), validated ONCE, up front, before any of the three
//! `InplaceOpN` calls mutates anything ([`validate_step_domain`]): `theta`/
//! `first_moment`/`second_moment`/`grad` share one dtype (F32 only — the
//! optimizer is not gated by esc-045's BF16 boundary; LoRA `theta`/moments
//! are always F32, `adapter.rs`'s `ComputePrecision::backbone_dtype` only
//! affects the frozen backbone), one shape (refused,
//! `Error::ShapeMismatchBinaryOp`, never silently broadcast/truncated), and
//! one device (`Error::DeviceMismatchBinaryOp`). No two of the four may
//! ALIAS (share storage) — refused with `Error::CannotSetVar`, mirroring
//! `Var::set`'s own aliasing guard (candle-core 0.11.0 `variable.rs:130-
//! 135`), checked via read-lock pointer identity BEFORE any `InplaceOpN`
//! call so a write-vs-read alias on the SAME `RwLock<Storage>` cannot
//! deadlock candle's own locking (see [`refuse_aliased`]'s doc). The three
//! MUTATED tensors (`theta`/`first_moment`/`second_moment`) additionally
//! need an INJECTIVE layout on both arms: CUDA requires full contiguity
//! (`Error::RequiresContiguous` — unchanged from before this fix); CPU
//! forward walks arbitrary strides for **reads** (`StridedOffsets`,
//! matching every other op in this crate) but a non-injective
//! **destination** (a stride-0/broadcast dimension mapping multiple logical
//! indices onto ONE storage slot) is refused the same way a broadcast
//! `Var::set` target would be — writing through it would silently
//! overwrite the same slot multiple times per step in an order this op
//! does not control. `grad` (read-only) has no injectivity requirement on
//! CPU — a transposed/broadcast gradient view reads correctly through
//! `StridedOffsets` either way.
//!
//! **Bit-identity, both arms, not a CUDA tolerance.** Per-element, each of
//! the three update rules above is evaluated as the SAME SEQUENCE of
//! individually-rounded `f32` operations that candle's eager composition
//! (`(m*beta1) + (g*(1-beta1))`, etc. — the literal `adamw.rs:94-100` chain
//! this module's CPU unit tests reproduce as an independent oracle)
//! performs — floating-point ELEMENTWISE operations have no cross-element
//! interaction, so folding candle's separate full-array passes into one
//! per-element expression changes nothing about that element's own
//! rounding PROVIDED the expression preserves both candle's OPERATION ORDER
//! and its ROUNDING COUNT. Two corrections versus the previous version of
//! this doc, both closed by the adversarial audit at `perf/multi-tensor-
//! adamw`@0498f8b (`.jammi/ledger/perf-s2-20260825.jsonl`):
//!
//! - **Every `Tensor * f64` in the eager chain is `Affine(mul, 0.0)`**
//!   (`candle-core-0.11.0/src/cpu_backend/mod.rs:311-317`'s CPU map is
//!   literally `v * mul + add` with `add = T::from_f64(0.0)`; the CUDA
//!   kernel — `candle-kernels-0.11.0/src/affine.cu`'s `AFFINE_OP(float,
//!   affine_f32, x * mul + add)` — is the same expression). The trailing
//!   `+ 0.0` is not a no-op: it LAUNDERS a `-0.0` product (from an
//!   underflowed or exact-zero multiply) to `+0.0`, per IEEE-754's
//!   opposite-sign-zero-sum rule. This op's CPU and CUDA arms both now
//!   reproduce that `+ 0.0` explicitly at every scalar-multiply site
//!   (`AdamMomentUpdate`/`AdamThetaUpdate`'s bodies below; the `.cu` file's
//!   matching comment), not just the multiply — skipping it would silently
//!   diverge on a `-0.0` input the eager chain launders away.
//! - **On the CUDA arm this is provable, not a tolerance claim.** nvcc's
//!   `--fmad=true` default (on regardless of `-use_fast_math`, which stays
//!   off — `build.rs`) may silently contract an `a*b+c`-shaped C-source
//!   sub-expression into a single-rounding hardware FMA — measured on
//!   jammi-a100, this previously diverged from candle's own eager CUDA
//!   chain on 5145/16384 `m` elements at t=3 with nonzero prior moments.
//!   Per `build.rs`'s pinned-flags comment and `docs/maintainer/cuda-
//!   kernel-guide.md`, the fix is explicit-rounding PTX intrinsics IN THE
//!   EXPRESSION (`__fmul_rn`/`__fadd_rn`/`__fsub_rn`/`__fdiv_rn`, `sqrtf` —
//!   already correctly-rounded without `-use_fast_math`), not a TU-wide
//!   `--fmad=false` (which would tax every OTHER kernel in this crate for a
//!   guarantee only this one needs). Each intrinsic is a single, non-fusable
//!   IEEE round-to-nearest op, so ptxas cannot merge two of them into an
//!   FMA the way it silently could with bare `*`/`+` — see `cuda/
//!   adamw_step.cu` for the full per-site mapping against `adamw.rs:94-100`
//!   and the acceptance harness in `tests/cuda_parity.rs` (fused-CUDA vs
//!   eager-CUDA `to_bits()` equality, plus fused-CPU vs fused-CUDA) and this
//!   file's own CPU-side oracle for the CPU arm.
//!
//! On the CPU arm, Rust's `f32` arithmetic does not auto-contract into FMA
//! without an explicit `.mul_add()` call, so no intrinsic-pinning is needed
//! there — but the `+ 0.0` laundering above is required on CPU too (Rust
//! does not add it for you either).

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, Error, InplaceOp2, InplaceOp3, Layout, Result, Tensor};

use crate::layout_walk::StridedOffsets;

/// Fixed hyperparameters for one [`adamw_step_fused_t`] call — everything
/// EXCEPT the step counter `t` (which changes every step and therefore
/// stays a separate argument rather than a field a caller could forget to
/// update).
#[derive(Debug, Clone, Copy)]
pub struct AdamWParams {
    pub beta1: f64,
    pub beta2: f64,
    pub lr: f64,
    pub weight_decay: f64,
    pub eps: f64,
}

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
                    // which LLVM may lower differently). It is its own
                    // standalone unary kernel in the eager chain, never
                    // followed by "+ 0.0", so no laundering site here.
                    let gv = if square_grad { gv * gv } else { gv };
                    // `Affine(beta, 0.0)` / `Affine(one_minus_beta, 0.0)`:
                    // two INDEPENDENTLY rounded, INDEPENDENTLY zero-laundered
                    // terms (candle's own `Tensor * f64` is `v*mul + 0.0`,
                    // `cpu_backend/mod.rs:311-317` — see this module's doc),
                    // each its own kernel launch in the eager chain, THEN a
                    // genuine standalone `Tensor + Tensor` add.
                    let term_m = beta * m[im] + 0.0f32;
                    let term_g = one_minus_beta * gv + 0.0f32;
                    m[im] = term_m + term_g;
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
                    // `Affine(next_m, scale_m, 0.0)` / `Affine(next_v,
                    // scale_v, 0.0)` — see this module's doc.
                    let m_hat = m[im] * scale_m + 0.0f32;
                    let v_hat = v[iv] * scale_v + 0.0f32;
                    // `Affine(sqrt(v_hat), 1.0, eps)` == `sqrt(v_hat) + eps`
                    // bit-for-bit: `x * 1.0` is exact at any finite `x` (no
                    // rounding, no sign change), so no separate laundering
                    // step is needed for THIS site — unlike the `mul=<other>,
                    // add=0.0` sites, `mul=1.0` cannot itself introduce a new
                    // `-0.0` `x` did not already carry.
                    let denom = v_hat.sqrt() + eps;
                    // Standalone binary div — one rounding, no affine site.
                    let adjusted_grad = m_hat / denom;
                    // `Affine(theta, one_minus_lr_lambda, 0.0)` /
                    // `Affine(adjusted_grad, lr, 0.0)`, then a genuine
                    // standalone `Tensor - Tensor` sub.
                    let theta_scaled = theta[it] * one_minus_lr_lambda + 0.0f32;
                    let adj_scaled = adjusted_grad * lr + 0.0f32;
                    theta[it] = theta_scaled - adj_scaled;
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

/// TEST-ONLY NEGATIVE CONTROL — never wired into [`adamw_step_fused_t`] or
/// any admission/dispatch path; the ONLY callers are this module's own
/// `negative_control_...` unit test and `tests/cuda_parity.rs`'s CUDA
/// bit-identity RED control. It exists solely to prove those harnesses
/// have the POWER to detect the exact defect class this fix closes:
/// deliberately forces the single-rounding FMA contraction commit 0498f8b
/// risked leaving to nvcc's `--fmad=true` discretion (CPU: `f32::mul_add`;
/// CUDA: `fmaf()`, `cuda/adamw_step.cu`'s `adamw_moment_update_f32_fma_
/// contracted_red_control`) instead of [`AdamMomentUpdate`]'s real
/// two-separately-rounded-multiplies-then-add. A normal, always-compiled
/// (not `#[cfg(test)]`) public type — `#[cfg(test)]` items are invisible to
/// `tests/*.rs` integration tests, which compile this crate as an ordinary
/// external dependency (see `PhiloxKatProbe`'s identical precedent,
/// `ops::dropout`), so this has to be reachable the same way.
#[derive(Debug, Clone, Copy)]
pub struct AdamMomentUpdateFmaContractedRedControl {
    pub beta: f64,
    pub square_grad: bool,
}

impl AdamMomentUpdateFmaContractedRedControl {
    pub fn new(beta: f64, square_grad: bool) -> Self {
        Self { beta, square_grad }
    }
}

impl super::sealed::Sealed for AdamMomentUpdateFmaContractedRedControl {}

impl InplaceOp2 for AdamMomentUpdateFmaContractedRedControl {
    fn name(&self) -> &'static str {
        "adamw_moment_update_fma_contracted_red_control"
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
                    let gv = if square_grad { gv * gv } else { gv };
                    // WRONG on purpose: one rounding for the whole
                    // expression (`f32::mul_add`), contracting the
                    // `beta*m[im]` term into the add — exactly the defect
                    // class the real op must NOT exhibit.
                    m[im] = beta.mul_add(m[im], one_minus_beta * gv);
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
        crate::cuda::adamw_step::moment_update_fma_contracted_red_control_cuda_fwd(
            self.beta,
            self.square_grad,
            s1,
            l1,
            s2,
            l2,
        )
    }
}

/// `true` iff `layout` is INJECTIVE as a map from logical multi-index to
/// storage offset — i.e. no dimension of length > 1 has stride 0. A
/// stride-0 dimension is exactly what `Tensor::broadcast_as`/`expand`
/// produce: every logical index along that axis reads (or, if this were a
/// write target, WRITES) the SAME storage slot. Reading through such a
/// layout is fine (every read returns the same value regardless of which
/// logical index asked); writing through one is not (the last write along
/// the collapsed axis wins, in an order this op does not control) — so
/// this check gates [`validate_step_domain`]'s three MUTATED destinations
/// (`theta`/`first_moment`/`second_moment`) only, never `grad`.
fn layout_is_injective(l: &Layout) -> bool {
    l.dims()
        .iter()
        .zip(l.stride().iter())
        .all(|(&d, &s)| d <= 1 || s != 0)
}

/// `true` iff `a` and `b` share the same underlying storage — a public-API
/// substitute for `Tensor::same_storage` (candle-core 0.11.0 `pub(crate)`,
/// unreachable from this crate). Acquires a READ lock on `a`, records the
/// address the guard derefs to, drops it, then does the same for `b` —
/// SEQUENTIALLY, never holding both simultaneously, so this cannot deadlock
/// even when `a` and `b` are in fact the same `RwLock` (two read guards on
/// the same lock, held one at a time, are always sound; it is a
/// simultaneous read-vs-write pair on the SAME lock — exactly what an
/// un-checked `InplaceOpN` call with an aliased mutated/read argument would
/// attempt — that candle's `std::sync::RwLock` cannot resolve). Two
/// `Tensor`s that merely CONTAIN equal values but own independent storage
/// (e.g. two separate `Tensor::zeros` calls) get different addresses here,
/// same as `same_storage` would report.
fn same_storage(a: &Tensor, b: &Tensor) -> bool {
    let addr = |t: &Tensor| -> usize {
        let (storage, _layout) = t.storage_and_layout();
        &*storage as *const _ as usize
    };
    addr(a) == addr(b)
}

/// Refuses ANY pairwise aliasing among `tensors` — mirrors `Var::set`'s own
/// aliasing guard (`Error::CannotSetVar`, candle-core 0.11.0 `variable.rs:
/// 130-135`: "cannot set a variable to a tensor that is derived from its
/// value"). Called from [`validate_step_domain`] BEFORE any `InplaceOpN`
/// call, so an aliased write-vs-read pair (e.g. a caller passing the SAME
/// `Var` as both `first_moment` and `grad`) is refused with a typed error
/// instead of candle's own write-lock-then-read-lock machinery deadlocking
/// on the same `RwLock<Storage>` inside `Tensor::inplace_op2/3`.
fn refuse_aliased(tensors: &[&Tensor]) -> Result<()> {
    for i in 0..tensors.len() {
        for j in (i + 1)..tensors.len() {
            if same_storage(tensors[i], tensors[j]) {
                let msg = "adamw_step_fused: theta/first_moment/second_moment/grad \
                           must not alias one another (two arguments share storage)";
                return Err(Error::CannotSetVar { msg }.bt());
            }
        }
    }
    Ok(())
}

/// The whole-domain check (family D), run ONCE before any of the three
/// `InplaceOpN` calls mutates anything — see this module's doc for the full
/// rationale. Checks, in order: pairwise dtype/shape/device agreement
/// (against `theta`) for all four tensors; pairwise aliasing across all
/// four; then, device-conditional, the structural requirement on the THREE
/// MUTATED tensors only (`theta`/`first_moment`/`second_moment`) — full
/// contiguity on CUDA (mirroring the CUDA glue's own per-kernel
/// `require_contiguous_f32`, but checked here across ALL FOUR tensors up
/// front rather than lazily per launch, so a `grad`/`theta` contiguity
/// problem that would only have surfaced on the THIRD launch cannot leave
/// `first_moment`/`second_moment` already mutated by the first two);
/// injective layout on CPU (see [`layout_is_injective`]'s doc).
fn validate_step_domain(theta: &Tensor, m: &Tensor, v: &Tensor, g: &Tensor) -> Result<()> {
    const OP: &str = "adamw_step";
    for t in [m, v, g] {
        if t.shape() != theta.shape() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: theta.shape().clone(),
                rhs: t.shape().clone(),
                op: OP,
            });
        }
        if t.dtype() != theta.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: theta.dtype(),
                rhs: t.dtype(),
                op: OP,
            });
        }
        if t.device().location() != theta.device().location() {
            return Err(Error::DeviceMismatchBinaryOp {
                lhs: theta.device().location(),
                rhs: t.device().location(),
                op: OP,
            });
        }
    }

    refuse_aliased(&[theta, m, v, g])?;

    if theta.device().is_cuda() {
        for t in [theta, m, v, g] {
            if !t.is_contiguous() {
                return Err(Error::RequiresContiguous { op: OP });
            }
        }
    } else {
        for t in [theta, m, v] {
            if !layout_is_injective(t.layout()) {
                let msg = "adamw_step_fused: theta/first_moment/second_moment must have an \
                           injective layout (no stride-0/broadcast dimension) — a broadcast \
                           destination would overwrite the same storage slot multiple times \
                           per step in an order this op does not control";
                return Err(Error::CannotSetVar { msg }.bt());
            }
        }
    }
    Ok(())
}

/// The whole fused AdamW step for ONE `Var`: `validate_step_domain` once,
/// up front, then two [`AdamMomentUpdate`] calls (first then second moment
/// — order matters only in that both must finish before the third call,
/// since [`AdamThetaUpdate`] reads their post-EMA values) followed by one
/// [`AdamThetaUpdate`] call. Three kernel launches, zero `Var::set`/memcpy
/// (see this module's doc). `t` is the 1-indexed step counter — matching
/// `candle_nn::AdamW`'s own convention, `t = 1` for the very first step —
/// from which `scale_m = 1/(1-beta1^t)`, `scale_v = 1/(1-beta2^t)` are
/// derived HERE rather than left to the caller: a caller-supplied
/// `scale_m`/`scale_v` (the deprecated [`adamw_step_fused`]'s shape) is a
/// bias-correction footgun — nothing stops it from being computed from the
/// WRONG `t` (stale, off-by-one, or simply never advanced), and the
/// resulting number is silently "unrepresentable-wrong": every value in
/// its domain is a plausible bias-correction scalar, so a wrong one carries
/// no signal that it is wrong. `t == 0` is refused outright (`1/(1-x^0) =
/// 1/0 = inf`, which would poison every downstream product with `inf`/`NaN`
/// rather than error).
pub fn adamw_step_fused_t(
    theta: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    grad: &Tensor,
    t: usize,
    params: AdamWParams,
) -> Result<()> {
    if t == 0 {
        candle_core::bail!(
            "adamw_step_fused_t: t must be >= 1 (candle_nn::AdamW's convention — t=1 for the \
             first step); t=0 makes scale_m/scale_v = 1/(1-beta^0) = 1/0 = inf"
        );
    }
    validate_step_domain(theta, first_moment, second_moment, grad)?;
    let t_i32 = i32::try_from(t).map_err(|_| {
        Error::Msg(format!(
            "adamw_step_fused_t: t={t} overflows i32::powi's exponent"
        ))
    })?;
    let scale_m = 1f64 / (1f64 - params.beta1.powi(t_i32));
    let scale_v = 1f64 / (1f64 - params.beta2.powi(t_i32));
    super::apply_inplace2(
        first_moment,
        grad,
        AdamMomentUpdate::new(params.beta1, false),
    )?;
    super::apply_inplace2(
        second_moment,
        grad,
        AdamMomentUpdate::new(params.beta2, true),
    )?;
    super::apply_inplace3(
        theta,
        first_moment,
        second_moment,
        AdamThetaUpdate::new(params.lr, params.weight_decay, scale_m, scale_v, params.eps),
    )
}

/// DEPRECATED shape of [`adamw_step_fused_t`]: takes `scale_m`/`scale_v`
/// (the bias-correction terms) directly from the caller instead of deriving
/// them from a step counter `t` — see [`adamw_step_fused_t`]'s doc for why
/// that is a footgun. Kept for one commit because `jammi-ai::fine_tune::
/// adamw`'s wiring (in flight on this same branch) already calls this exact
/// signature; behaviour is UNCHANGED (now goes through
/// `validate_step_domain` too, closing the same-branch audit's upfront-
/// validation and aliasing findings for this call shape as well) — only the
/// bias-correction-footgun concern is deprecated away, not the numerics.
#[deprecated(
    note = "compute t (the 1-indexed step counter) and call adamw_step_fused_t with AdamWParams \
            instead — scale_m/scale_v computed by the caller from a stale or wrong t is a \
            silent, unrepresentable-wrong bias-correction bug"
)]
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
    validate_step_domain(theta, first_moment, second_moment, grad)?;
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
        adamw_step_fused_t(
            theta,
            m,
            v,
            g,
            t as usize,
            AdamWParams {
                beta1,
                beta2,
                lr,
                weight_decay,
                eps,
            },
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
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
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

    /// MUT-triage (cargo-mutants): `InplaceOp2::name`/`InplaceOp3::name`
    /// feed every typed error's `op` field — a mutant that replaces either
    /// `name()` body with `""` or `"xyzzy"` survives unless some test pins
    /// the exact string, not just the error VARIANT. Exercises both ops'
    /// `name()` through their `ShapeMismatchBinaryOp` path (the cheapest
    /// domain check to trigger for both).
    #[test]
    fn error_op_field_names_are_pinned_exactly() {
        let dev = Device::Cpu;
        let m = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
        let err =
            super::super::apply_inplace2(&m, &g, AdamMomentUpdate::new(0.9, false)).unwrap_err();
        match err {
            Error::ShapeMismatchBinaryOp { op, .. } => assert_eq!(op, "adamw_moment_update"),
            other => panic!("expected ShapeMismatchBinaryOp, got {other:?}"),
        }

        let theta = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let m2 = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let v2 = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
        let err = super::super::apply_inplace3(
            &theta,
            &m2,
            &v2,
            AdamThetaUpdate::new(1e-3, 0.01, 1.0, 1.0, 1e-8),
        )
        .unwrap_err();
        match err {
            Error::ShapeMismatchBinaryOp { op, .. } => assert_eq!(op, "adamw_theta_update"),
            other => panic!("expected ShapeMismatchBinaryOp, got {other:?}"),
        }
    }

    /// MUT-triage: a dtype mismatch BETWEEN the two `InplaceOp2` inputs
    /// (not just an unsupported-but-agreeing dtype like F64/F64) must hit
    /// the `s1.dtype() != s2.dtype()` guard specifically — the previous
    /// `unsupported_dtype_is_refused_with_a_typed_error` test used F64 for
    /// EVERY tensor, which never took this branch (all dtypes agreed, so
    /// it fell straight to the catch-all `UnsupportedDTypeForOp` arm),
    /// leaving `s1.dtype() != s2.dtype() -> false`, `!=` -> `==`, and
    /// `||`/`&&` swap mutants on this guard undetected.
    #[test]
    fn mismatched_dtype_between_moment_and_grad_is_refused() {
        let dev = Device::Cpu;
        let m = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let err = super::super::apply_inplace2(&m, &g, AdamMomentUpdate::new(0.9, false))
            .expect_err("F32 moment vs F64 grad must be refused, not silently reinterpreted");
        assert!(matches!(err, Error::DTypeMismatchBinaryOp { .. }));
    }

    /// MUT-triage: same class of gap as
    /// `mismatched_dtype_between_moment_and_grad_is_refused`, for
    /// `AdamThetaUpdate`'s two-condition OR guard
    /// (`s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype()`) — covers
    /// BOTH disjuncts (m mismatched, then v mismatched) so a `||` -> `&&`
    /// swap or either `!=` -> `==` swap cannot hide behind the other
    /// disjunct still catching it.
    #[test]
    fn mismatched_dtype_between_theta_and_either_moment_is_refused() {
        let dev = Device::Cpu;
        let theta = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let v_ok = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let m_wrong = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let err = super::super::apply_inplace3(
            &theta,
            &m_wrong,
            &v_ok,
            AdamThetaUpdate::new(1e-3, 0.01, 1.0, 1.0, 1e-8),
        )
        .expect_err("F32 theta vs F64 m must be refused");
        // MUT-triage: pin the exact `lhs`/`rhs` reported, not just the
        // error VARIANT — the inner `if s1.dtype() != s2.dtype() { .. }
        // else { .. }` (line 241) selects WHICH mismatched pair to report;
        // a `!=` -> `==` mutant there flips it to report the (agreeing)
        // theta/v pair instead of the actually-mismatched theta/m pair,
        // invisible to a `matches!` check alone.
        match err {
            Error::DTypeMismatchBinaryOp { lhs, rhs, .. } => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::F64);
            }
            other => panic!("expected DTypeMismatchBinaryOp, got {other:?}"),
        }

        let m_ok = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let v_wrong = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let err = super::super::apply_inplace3(
            &theta,
            &m_ok,
            &v_wrong,
            AdamThetaUpdate::new(1e-3, 0.01, 1.0, 1.0, 1e-8),
        )
        .expect_err("F32 theta vs F64 v must be refused");
        match err {
            Error::DTypeMismatchBinaryOp { lhs, rhs, .. } => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::F64);
            }
            other => panic!("expected DTypeMismatchBinaryOp, got {other:?}"),
        }
    }

    /// MUT-triage: `AdamThetaUpdate::cpu_fwd`'s dtype-mismatch guard
    /// (`s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype()`) must be
    /// `false` — reaching the catch-all `UnsupportedDTypeForOp` arm, NOT
    /// `DTypeMismatchBinaryOp` — when theta/m/v all AGREE with each other
    /// but on a dtype other than F32. `adamw_step_fused`'s own
    /// `unsupported_dtype_is_refused_with_a_typed_error` test never
    /// exercises this: it feeds F64 to `first_moment`/`grad` too, so
    /// `AdamMomentUpdate`'s OWN catch-all fires first and short-circuits
    /// before `AdamThetaUpdate` is ever called — this test calls
    /// `AdamThetaUpdate` directly to close that gap.
    #[test]
    fn all_agreeing_non_f32_dtype_hits_the_catchall_not_a_mismatch_error() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let m = Tensor::zeros((2,), DType::F64, &dev).unwrap();
        let v = Tensor::zeros((2,), DType::F64, &dev).unwrap();
        let err = super::super::apply_inplace3(
            &theta,
            &m,
            &v,
            AdamThetaUpdate::new(1e-3, 0.01, 1.0, 1.0, 1e-8),
        )
        .expect_err("F64 has no AdamThetaUpdate CPU implementation (F32 only)");
        assert!(
            matches!(err, Error::UnsupportedDTypeForOp(..)),
            "all-agreeing-but-non-F32 must hit the catch-all arm, not \
             DTypeMismatchBinaryOp (which requires an actual disagreement \
             between the three dtypes) — got {err:?}"
        );
    }

    /// MUT-triage: `eps=1e-8` in every other test in this file is far
    /// below `v_hat.sqrt()`'s magnitude at `f32` precision, so `+ eps` and
    /// a mutated `- eps` round to the SAME bits — the mutant on
    /// `crates/jammi-kernels/src/ops/adamw_step.rs:234` (`+` -> `-` in
    /// `let denom = v_hat.sqrt() + eps;`) survived every bit-identity test
    /// for exactly this reason. `eps` here is large enough, relative to a
    /// small `v`, that `+`/`-` produce OBSERVABLY different `f32` bits.
    #[test]
    fn large_eps_relative_to_v_hat_matches_bit_for_bit() {
        let (theta, m, v, g) = setup(&[1.0, -0.5], &[0.01, 0.02], (2,));
        let eps = 0.25; // v starts at 0, so v_hat after one step is tiny — eps dominates.
        let (want_theta, want_m, want_v) =
            eager_step(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, eps, 1);
        run_fused(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, eps, 1);
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

    /// Peel candle's `Error::WithBacktrace` wrapper (if present) down to the
    /// error it carries — identical to `low_rank_residual_linear.rs`'s
    /// helper of the same name/doc. `Error::bt()` (this module's
    /// `refuse_aliased`/`validate_step_domain` both call it, mirroring
    /// `Var::set`'s own `CannotSetVar` convention) boxes the original error
    /// into `WithBacktrace` whenever `RUST_BACKTRACE`/`RUST_LIB_BACKTRACE`
    /// leaves backtrace capture enabled — an environment property, not a
    /// platform one — so a bare `matches!(err, Error::CannotSetVar { .. })`
    /// is only reliable after peeling.
    fn peel_backtrace(err: &Error) -> &Error {
        let mut e = err;
        while let Error::WithBacktrace { inner, .. } = e {
            e = inner;
        }
        e
    }

    fn default_params() -> AdamWParams {
        AdamWParams {
            beta1: 0.9,
            beta2: 0.999,
            lr: 1e-3,
            weight_decay: 0.01,
            eps: 1e-8,
        }
    }

    #[test]
    fn shape_mismatch_between_theta_and_grad_is_refused() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[1.0f32, 2.0], (2,), &dev).unwrap();
        let m = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
        let err = adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params())
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
        let err = adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params())
            .expect_err("F64 has no adamw_step_fused_t CPU implementation (F32 only)");
        // theta/m/v/g all AGREE on dtype (F64) — validate_step_domain's
        // pairwise-vs-theta check passes; the F32-only InplaceOp2's own
        // catch-all is what actually refuses this, so the caller sees
        // `UnsupportedDTypeForOp`, not `DTypeMismatchBinaryOp` (that variant
        // is reserved for an ACTUAL disagreement between two tensors — see
        // `mismatched_dtype_between_moment_and_grad_is_refused`, below).
        assert!(matches!(err, Error::UnsupportedDTypeForOp(..)));
    }

    /// `t == 0` is a domain violation (family D): `1/(1-beta^0) = 1/0 =
    /// inf`, which would otherwise poison every downstream product with
    /// `inf`/`NaN` silently rather than error.
    #[test]
    fn zero_step_counter_is_refused() {
        let (theta, m, v, g) = setup(&[1.0], &[0.1], (1,));
        let err = adamw_step_fused_t(&theta, &m, &v, &g, 0, default_params())
            .expect_err("t=0 must be refused, not silently produce inf/NaN scale_m/scale_v");
        assert!(err.to_string().contains("t must be >= 1"), "got: {err}");
    }

    /// The upfront, single-function domain check (family D / all-or-
    /// nothing): a shape mismatch that would only have surfaced on the
    /// THIRD `InplaceOp3` call (`theta` vs `first_moment`/`second_moment`/
    /// `grad`, which all agree with EACH OTHER) must be caught before the
    /// first two `InplaceOp2` calls run at all — `first_moment`/
    /// `second_moment` must be observably UNTOUCHED (still their initial
    /// nonzero values) after the refused call, not half-advanced.
    #[test]
    fn mismatched_shape_on_the_theta_leg_leaves_moments_untouched() {
        let dev = Device::Cpu;
        // theta has a DIFFERENT shape from m/v/g, which all agree with each
        // other — so a lazy, per-call validation would let the two
        // AdamMomentUpdate calls (m, v) succeed before AdamThetaUpdate's
        // shape check ever runs.
        let theta = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
        let m = Tensor::from_slice(&[0.5f32, -0.5], (2,), &dev).unwrap();
        let v = Tensor::from_slice(&[0.25f32, 0.75], (2,), &dev).unwrap();
        let g = Tensor::from_slice(&[0.1f32, 0.2], (2,), &dev).unwrap();
        let m_before: Vec<f32> = m.to_vec1().unwrap();
        let v_before: Vec<f32> = v.to_vec1().unwrap();

        let err = adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params())
            .expect_err("theta/moment shape mismatch must be refused");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));

        let m_after: Vec<f32> = m.to_vec1().unwrap();
        let v_after: Vec<f32> = v.to_vec1().unwrap();
        assert_eq!(
            m_before, m_after,
            "first_moment was mutated despite the whole step being refused"
        );
        assert_eq!(
            v_before, v_after,
            "second_moment was mutated despite the whole step being refused"
        );
    }

    /// Aliasing (family D): `first_moment` and `grad` sharing storage (the
    /// SAME `Var`/`Tensor` passed as both) must be refused with a typed
    /// error, not deadlock candle's own write-then-read locking inside
    /// `InplaceOp2::cpu_fwd`'s dispatch. `Tensor::clone()` shares storage
    /// (candle's `Tensor` is `Arc<Tensor_>`), so `g.clone()` is a genuine
    /// alias, not merely an equal-valued independent tensor.
    #[test]
    fn aliased_first_moment_and_grad_is_refused_not_deadlocked() {
        let (theta, m, v, _g) = setup(&[1.0, 2.0], &[0.1, 0.2], (2,));
        let aliased_grad = m.clone();
        let err = adamw_step_fused_t(&theta, &m, &v, &aliased_grad, 1, default_params())
            .expect_err("first_moment aliasing grad must be refused");
        assert!(
            matches!(peel_backtrace(&err), Error::CannotSetVar { .. }),
            "got {err:?}"
        );
    }

    /// Aliasing: `theta` and `first_moment` sharing storage must also be
    /// refused (the mutated-vs-read hazard `AdamThetaUpdate` would hit).
    #[test]
    fn aliased_theta_and_first_moment_is_refused() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[1.0f32, 2.0], (2,), &dev).unwrap();
        let m = theta.clone();
        let v = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[0.1f32, 0.2], (2,), &dev).unwrap();
        let err = adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params())
            .expect_err("theta aliasing first_moment must be refused");
        assert!(
            matches!(peel_backtrace(&err), Error::CannotSetVar { .. }),
            "got {err:?}"
        );
    }

    /// CPU non-injective destination (family D): a broadcast `first_moment`
    /// (stride 0 along the expanded dimension) must be refused, the same
    /// way the CUDA arm already refuses a non-contiguous one — writing
    /// through it would silently overwrite the same storage slot multiple
    /// times per step.
    #[test]
    fn non_injective_first_moment_destination_is_refused_on_cpu() {
        let dev = Device::Cpu;
        let theta = Tensor::zeros((4,), DType::F32, &dev).unwrap();
        let m = Tensor::zeros((1,), DType::F32, &dev)
            .unwrap()
            .broadcast_as((4,))
            .unwrap();
        assert!(!m.is_contiguous());
        let v = Tensor::zeros((4,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], (4,), &dev).unwrap();
        let err = adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params())
            .expect_err("a stride-0 broadcast destination must be refused, not silently walked");
        assert!(
            matches!(peel_backtrace(&err), Error::CannotSetVar { .. }),
            "got {err:?}"
        );
    }

    /// A broadcast `grad` (read-only) is fine — reading the same storage
    /// slot repeatedly is not a hazard the way writing through one is.
    #[test]
    fn non_injective_grad_is_fine_read_only() {
        let dev = Device::Cpu;
        let theta = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (4,), &dev).unwrap();
        let m = Tensor::zeros((4,), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((4,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[0.1f32], (1,), &dev)
            .unwrap()
            .broadcast_as((4,))
            .unwrap();
        assert!(!g.is_contiguous());
        adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params())
            .expect("a broadcast (read-only) grad must be accepted on CPU");
        let out: Vec<f32> = theta.to_vec1().unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    /// The DEPRECATED `adamw_step_fused` shape still works and still goes
    /// through the same [`validate_step_domain`]/bit-identity path — kept
    /// for one commit because the in-flight `jammi-ai::fine_tune::adamw`
    /// wiring on this branch already calls this exact signature (see this
    /// function's own doc). `#[allow(deprecated)]`: the whole point of this
    /// test is to exercise the deprecated item, not to avoid it.
    #[test]
    #[allow(deprecated)]
    fn deprecated_signature_still_matches_the_eager_chain_bit_for_bit() {
        let (theta, m, v, g) = setup(&[0.5, -1.25, 3.0, 0.0], &[0.1, -0.2, 0.05, 0.0], (4,));
        let (want_theta, want_m, want_v) =
            eager_step(&theta, &m, &v, &g, 0.9, 0.999, 1e-3, 0.01, 1e-8, 1);
        let scale_m = 1f64 / (1f64 - 0.9f64.powi(1));
        let scale_v = 1f64 / (1f64 - 0.999f64.powi(1));
        adamw_step_fused(
            &theta, &m, &v, &g, 0.9, 0.999, scale_m, scale_v, 1e-3, 0.01, 1e-8,
        )
        .unwrap();
        assert_bit_identical(&theta, &want_theta);
        assert_bit_identical(&m, &want_m);
        assert_bit_identical(&v, &want_v);
    }

    /// NEGATIVE CONTROL (family F), CALLING THE REAL OP on both sides —
    /// the previous version of this test computed two bare `f32`
    /// expressions locally and never invoked `AdamMomentUpdate`/
    /// `apply_inplace2` at all, which the adversarial audit correctly
    /// flagged as tautological (it could never fail regardless of what the
    /// real kernel did). This version runs [`AdamMomentUpdate`] (the REAL
    /// kernel) and [`AdamMomentUpdateFmaContractedRedControl`] (the
    /// deliberately WRONG, FMA-contracted kernel) through the SAME
    /// `apply_inplace2` dispatch on the SAME nonzero starting `m`/`g`, and
    /// requires at least one bit mismatch across a representative sweep (a
    /// single unlucky pair CAN coincide bit-for-bit by chance) — proving
    /// the bit-identity oracle actually has the power the deprecated
    /// tautological version only claimed.
    #[test]
    fn negative_control_the_fma_contracted_kernel_diverges_from_the_real_one() {
        let dev = Device::Cpu;
        let pairs: [(f32, f32); 6] = [
            (0.1234567, -0.7654321),
            (std::f32::consts::PI / 4.0, std::f32::consts::E / 3.0),
            (-2.0_f32.sqrt(), 3.0_f32.sqrt()),
            (0.000123, 987.654),
            (-0.5, 0.5),
            (1.0000001, -1.0000002),
        ];
        let mut any_mismatch = false;
        for &(m0, g0) in &pairs {
            let m_real = Tensor::from_slice(&[m0], (1,), &dev).unwrap();
            let g = Tensor::from_slice(&[g0], (1,), &dev).unwrap();
            super::super::apply_inplace2(&m_real, &g, AdamMomentUpdate::new(0.9, false)).unwrap();

            let m_wrong = Tensor::from_slice(&[m0], (1,), &dev).unwrap();
            super::super::apply_inplace2(
                &m_wrong,
                &g,
                AdamMomentUpdateFmaContractedRedControl::new(0.9, false),
            )
            .unwrap();

            let real: f32 = m_real.to_vec1::<f32>().unwrap()[0];
            let wrong: f32 = m_wrong.to_vec1::<f32>().unwrap()[0];
            if real.to_bits() != wrong.to_bits() {
                any_mismatch = true;
            }
        }
        assert!(
            any_mismatch,
            "AdamMomentUpdateFmaContractedRedControl must diverge from the real \
             AdamMomentUpdate kernel on at least one of these representative inputs \
             — otherwise the bit-identity oracle has no power to catch this defect class"
        );
    }

    /// MUT-triage: `negative_control_the_fma_contracted_kernel_diverges_
    /// from_the_real_one` (above) only proves the red-control's output
    /// DIFFERS from the real kernel's — true even for a stub `cpu_fwd`
    /// that never mutates `m` at all (the stale `m0` differs from the
    /// real kernel's updated value too). This test is a DIRECT value
    /// oracle instead (family F, numpy-first-style): the red control's
    /// output must equal the EXACT `f32::mul_add`-contracted formula,
    /// computed independently here — covering both `square_grad` branches
    /// (`gv*gv` is untested by every other red-control test, which all use
    /// `square_grad=false`) and killing the `cpu_fwd -> Ok(())` stub
    /// mutant as a side effect (a stub leaves `m` at its stale `m0`, which
    /// will not bit-match the computed formula for any of these nonzero,
    /// non-idempotent fixtures).
    #[test]
    fn red_control_matches_the_exact_fma_contracted_formula() {
        let dev = Device::Cpu;
        let cases: [(f32, f32, f64, bool); 4] = [
            (0.5, 0.1, 0.9, false),
            (-1.25, -0.3, 0.9, false),
            (0.5, 0.1, 0.9, true),
            (-1.25, -0.3, 0.999, true),
        ];
        for &(m0, g0, beta, square_grad) in &cases {
            let m = Tensor::from_slice(&[m0], (1,), &dev).unwrap();
            let g = Tensor::from_slice(&[g0], (1,), &dev).unwrap();
            super::super::apply_inplace2(
                &m,
                &g,
                AdamMomentUpdateFmaContractedRedControl::new(beta, square_grad),
            )
            .unwrap();
            let got: f32 = m.to_vec1::<f32>().unwrap()[0];

            let beta_f32 = beta as f32;
            let one_minus_beta_f32 = (1.0 - beta) as f32;
            let gv = if square_grad { g0 * g0 } else { g0 };
            let want = beta_f32.mul_add(m0, one_minus_beta_f32 * gv);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "red control (m0={m0}, g0={g0}, beta={beta}, square_grad={square_grad}): \
                 got {got} ({:#010x}), want {want} ({:#010x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    /// MUT-triage: pins `AdamMomentUpdateFmaContractedRedControl::name`'s
    /// exact string through its `ShapeMismatchBinaryOp` path — mirrors
    /// `error_op_field_names_are_pinned_exactly`'s pattern for the real
    /// ops, which never exercises the red control's own `name()` at all.
    #[test]
    fn red_control_name_is_pinned_exactly() {
        let dev = Device::Cpu;
        let m = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
        let err = super::super::apply_inplace2(
            &m,
            &g,
            AdamMomentUpdateFmaContractedRedControl::new(0.9, false),
        )
        .unwrap_err();
        match err {
            Error::ShapeMismatchBinaryOp { op, .. } => {
                assert_eq!(op, "adamw_moment_update_fma_contracted_red_control")
            }
            other => panic!("expected ShapeMismatchBinaryOp, got {other:?}"),
        }
    }

    /// MUT-triage: the red control's own `s1.dtype() != s2.dtype()` guard
    /// (a separate copy from `AdamMomentUpdate`'s, since this type
    /// implements its own `cpu_fwd`) must be exercised the same way.
    #[test]
    fn red_control_mismatched_dtype_is_refused() {
        let dev = Device::Cpu;
        let m = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let g = Tensor::from_slice(&[1.0f64, 2.0], (2,), &dev).unwrap();
        let err = super::super::apply_inplace2(
            &m,
            &g,
            AdamMomentUpdateFmaContractedRedControl::new(0.9, false),
        )
        .expect_err("F32 moment vs F64 grad must be refused, not silently reinterpreted");
        assert!(matches!(err, Error::DTypeMismatchBinaryOp { .. }));
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
        adamw_step_fused_t(&theta, &m, &v, &g, 1, default_params()).unwrap();
        assert_bit_identical(&theta, &want.0);
    }
}
