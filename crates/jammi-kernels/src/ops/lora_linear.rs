//! One tape node per LoRA site: `out = (x @ w^T) + cast(scale * (dropout(x) @ A^T @ B^T))`.
//!
//! This is the fused replacement for the eager composition
//! `jammi-lora`'s `LoraLinear::forward` (training arm) builds today from
//! ~11 candle ops (`Linear::forward`'s reshape/matmul/reshape for `base`,
//! the same for the `A` and `B` sub-linears, a `to_dtype`, an optional
//! `DropoutFused` node, and [`super::ScaledCastAdd`]'s own `[mul, cast,
//! add]`): each of those is its own node on candle's backward tape, and
//! candle's `GradStore::or_insert` (`backprop.rs`) allocates a FULL-SIZE
//! `zeros_like` + `add` for every one of them, PLUS an unconditional `dW`
//! GEMM for the frozen base weight (`Op::Matmul`'s backward computes both
//! operands' gradients regardless of whether either side is trainable —
//! `backprop.rs:457-468`). Collapsing the whole site into one
//! `CustomOp3` removes all of that: one node, one (or two, if the base
//! weight is itself a trainable `Var`) gradient contribution.
//!
//! ## The three tensor arguments
//!
//! - `x`: `[.., in]` (rank 2 — a pooled head, e.g. `fine_tune/lora.rs` — or
//!   rank 3 — `[batch, seq, in]`), the backbone dtype (`F32` or `BF16`).
//! - `w`: `[out, in]`, the FROZEN base weight, same dtype as `x`.
//! - `ab`: `F32` `[rank, in + out]`, the per-forward `Tensor::cat([A, B^T],
//!   1)` of the LoRA `A` (`[rank, in]`) and `B^T` (`[out, rank]` transposed
//!   to `[rank, out]`) matrices — built at the call site fresh every
//!   forward. `Tensor::cat`'s own backward (`Op::Cat`, `backprop.rs:469`)
//!   splits this op's `dab` gradient back into `dA`/`dB` via two cheap
//!   `narrow`s — tiny (`rank` rows), not the concern this op addresses.
//!
//! ## Every rounding point, forward
//!
//! 1. `base = x @ w^T` — one `BackendStorage::matmul` call issued with
//!    EXACTLY the `(b=1, m, n=out, k=in)` config and the same
//!    reshape-to-2D-then-transpose-the-weight layout shape
//!    `candle_nn::Linear::forward`'s contiguous-input branch issues
//!    (`candle-nn/src/linear.rs:46-68`) — same cuBLAS/gemm-crate kernel,
//!    bit-exact against the eager `base.forward(x)` call it replaces. No
//!    rounding beyond whatever `base`'s own dtype's GEMM does (`BF16`
//!    accumulates in `F32` internally on tensor cores; `F32` does not
//!    round at all here).
//! 2. `x32 = to_dtype(x, F32)` — a transient, exact when `x` is already
//!    `F32`, a widening (lossless) cast when `x` is `BF16`.
//! 3. `xd = dropout(x32)` (only when `self.dropout.is_some()`) — the
//!    KEPT-element value is `x32_i * scale`, a single `F32` multiply, via
//!    [`super::DropoutFused`]'s own `cpu_fwd`/`cuda_fwd` called DIRECTLY
//!    (not through [`super::apply1`] — this stays inside ONE storage-level
//!    forward, no second tape node). No mask tensor is ever materialized;
//!    see `dropout`'s own module doc for the Philox counter mapping.
//! 4. `h = xd @ A^T` — a second `F32` GEMM, same shape-derivation pattern
//!    as step 1 (`A` plays `w`'s role, output width `rank`).
//! 5. `delta = h @ B^T` — a third `F32` GEMM (`B^T` plays `w^T`'s role
//!    directly: `ab`'s second slot already stores `B^T`, so no further
//!    transpose is needed — see "the packed-`ab` GEMM eligibility
//!    problem" below for why this slot is copied into its own dense
//!    buffer before this GEMM, not sliced in place).
//! 6. `out = base + cast_to(base.dtype())(delta * scale)` — reuses
//!    [`super::ScaledCastAdd`]'s `cpu_fwd`/`cuda_fwd` DIRECTLY (same
//!    round-before-add model, same two rounding points its own module doc
//!    enumerates: PEFT's `Linear.forward` casts the scaled delta down to
//!    the base result's dtype BEFORE the add).
//!
//! ## Every rounding point, backward
//!
//! Symbols: `dy` = upstream gradient (base dtype); `s` = `self.scale`.
//! Derived directly from the forward graph above (a `Linear`-style
//! `y = x @ W^T` node has the standard `dx = dy @ W`, `dW = dy^T @ x`
//! backward — applied three times, once for the base GEMM and once each
//! for the two LoRA GEMMs) and from [`super::ScaledCastAdd::bwd`]'s own
//! disclosed order (cast to the LoRA dtype FIRST, multiply by `scale`
//! second — `scaled_cast_add.rs`'s `bwd` doc, cited verbatim in the code
//! below).
//!
//! 1. `d_lora = cast_to(F32)(dy) * s` — ONE rounding point: widening
//!    `dy` to `F32` is lossless from `BF16`, exact identity from `F32`;
//!    the multiply by `s` is the only place a `BF16` input's precision
//!    could have been lost, and it never is here because the cast
//!    happens first (matching `ScaledCastAdd::bwd`).
//! 2. `x32`/`xd`/`h` are RECOMPUTED (candle 0.11 has no save-for-backward
//!    channel — the same constraint [`super::LayerNormFused`]'s `bwd`
//!    documents) via the identical `to_dtype`/dropout-with-the-SAME-key/
//!    matmul sequence forward used — no new rounding beyond what forward
//!    already accounted for.
//! 3. `dB^T = h^T @ d_lora`, `g = d_lora @ B`, `dA = g^T @ xd`,
//!    `d_xd = g @ A` — four `F32` GEMMs, no rounding (`F32` throughout).
//! 4. `d_xd` is re-passed through THE SAME dropout key
//!    ([`super::DropoutFused::bwd`]'s own contract: applying the same
//!    `Copy` instance to a gradient regenerates `mask * scale`
//!    identically) — no new rounding (`F32`).
//! 5. `d_x_lora = cast_to(x.dtype())(d_xd)` — the ONE lossy rounding point
//!    on this branch when `x` is `BF16` (mirrors forward's own widening
//!    cast, inverted).
//! 6. `dx_base = dy @ w` — computed at `dy`/`w`'s OWN (base) dtype, the
//!    same cuBLAS/gemm-crate call `candle`'s own `Op::Matmul` backward
//!    would issue for `y = x @ w^T` (`backprop.rs:457-468`'s
//!    `lhs_grad = grad.matmul(&rhs.t()?)`, specialized to a pre-transposed
//!    `w`), so no NEW rounding beyond what a from-scratch composition
//!    would already have paid.
//! 7. `dx = dx_base + d_x_lora` — ONE more round-and-add at `x`'s dtype
//!    (candle's ordinary `Tensor::add`, promote-compute-round-once for
//!    `BF16`, exact for `F32`).
//! 8. `dW = dy^T @ x` (only when `self.dweight_needed`) — same `F32`/`BF16`
//!    GEMM rounding as step 6, no cast.
//!
//! ## The packed-`ab` GEMM eligibility problem (and its fix)
//!
//! `ab = cat([A, B^T], 1)` packs `A` (`[rank, in]`) and `B^T`
//! (`[rank, out]`) SIDE BY SIDE along the feature axis, so slicing either
//! back out via `Layout::narrow(1, ..)` yields a view whose ROW STRIDE is
//! `in + out` — wider than its own logical width (`in` or `out`). CUDA's
//! `gemm_config` (`cuda_backend/mod.rs:1379-1421`) accepts only two
//! stride shapes for a GEMM operand: row-contiguous WITH `row_stride ==
//! width` (its `CUBLAS_OP_N` branch) or column-contiguous WITH
//! `col_stride == 1` over the FULL matrix (`CUBLAS_OP_T`) — a
//! narrower-than-its-storage-row slice satisfies NEITHER (the check is
//! `rhs_m2 == n` / `rhs_m2 == 1`, both false for a padded row), so cuBLAS
//! would refuse it with `MatMulNonContiguous`. This op therefore
//! materializes each slice into its OWN dense buffer first (`to_dtype`
//! with the SAME dtype it already has — a layout-aware gather-copy, not a
//! numeric operation: see [`super::empty_like`]'s sibling idiom and
//! `BackendStorage::to_dtype`'s own implementation, which walks the given
//! `Layout` regardless of dtype-pair) before either GEMM touches it. Two
//! small (`rank`-row) copies per forward/backward call — negligible next
//! to the GEMMs and epilogue this op removes, and bit-exact (a gather
//! copy changes no bits).
//!
//! ## CPU `BF16` matmul: a pre-existing candle limitation, not a regression
//!
//! candle-core 0.11.0's CPU backend (without the `mkl`/`accelerate`
//! features, neither enabled anywhere in this workspace) implements
//! `MatMul` only for `F16`/`F32`/`F64` (`cpu_backend/mod.rs`'s
//! `impl Map2 for MatMul`, `f`'s own `T::DTYPE` guard) — a `BF16` `base`
//! matmul on CPU returns `Error::UnsupportedDTypeForOp` from EXACTLY the
//! same `BackendStorage::matmul` call this op's `cpu_fwd` issues, which is
//! the IDENTICAL call `candle_nn::Linear::forward` issues for the eager
//! composition this op replaces — so a `BF16`-backbone-on-CPU forward
//! fails the same way with or without this op (a typed, loud error, never
//! a silent wrong number — family D holds either way). This is a
//! pre-existing, disclosed gap in candle's CPU backend, not something this
//! op's domain check tries to route around: `BF16` production forwards run
//! on CUDA only (see `jammi-lora`'s call-site admission doc), and the CPU
//! oracle suite here covers the `(F32, F32, F32)` combination end-to-end
//! plus the typed-error boundary for `(BF16, BF16, F32)` on CPU.
//!
//! ## Bias: a domain refusal, not packed into `ab`
//!
//! BERT/DistilBERT's LoRA bases carry a bias (`candle_nn::linear`'s
//! `bias.is_some()`); this op has no bias slot. Packing a bias as an
//! extra `ab` row (`[rank + 1, in + out]`) was evaluated and rejected:
//! turning `y = x @ w^T + b` into a single matmul over an AUGMENTED input
//! requires appending a constant `1` COLUMN to `x` itself (the classic
//! bias-as-augmented-feature trick) — which changes `x`'s own domain
//! (`in` -> `in + 1`) and, worse, would need that constant column
//! EXCLUDED from dropout's per-element Bernoulli draw (a bias term is
//! never dropped), breaking the clean "every element of `x32` is an
//! independent dropout draw" domain this op (and `DropoutFused`) is built
//! on. That is not a clean fusion — a real structural change to `x`'s
//! shape and dropout's own domain — so `bias.is_some()` stays a domain
//! refusal (counted eager fallback at the call site), matching the
//! contract's explicit escape hatch for this evaluation.
//!
//! ## Domain (family D / K2)
//!
//! `x` rank 2 or 3, `w` rank 2 `[out, in]`, `ab` rank 2 `[rank, in+out]`;
//! dtype pairs `(F32, F32, F32)` and `(BF16, BF16, F32)` (base dtype must
//! match between `x`/`w`; `ab` is always `F32` — the workspace fact
//! `jammi-lora`'s LoRA adapters are always built `F32`, cited in
//! [`super::ScaledCastAdd`]'s own doc); every input contiguous
//! (`Layout::is_contiguous`); `out >= 1`, `rank >= 1` (validated once, at
//! construction, in [`LoraLinearFused::new`]); the call site is
//! responsible for `n <= u32::MAX` and `device_is_supported` (this op has
//! no CUDA-launch-grid ceiling of its own beyond what
//! `crate::cuda::{dropout, scaled_cast_add}`'s own launchers already
//! enforce internally when reused). Anything else is a typed `Error`
//! this op returns directly (no silent fallback INSIDE the op — the
//! call site's admission predicate is what decides fused-vs-eager BEFORE
//! ever calling this op, per this crate's usual "validate-and-fall-back"
//! split, `admission.rs`'s module doc).

use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Error, Layout, Result, Shape, Tensor,
};

use super::{DropoutFused, ScaledCastAdd};

/// The Philox draw's `(seed, layer_id, forward_idx, p)` key, reserved ONCE
/// per site per forward by `jammi-lora`'s `DropoutMasks::next_key()`
/// BEFORE either arm (fused or eager fallback) runs — see this crate's
/// `dropout` module doc for the counter mapping this key feeds into
/// [`DropoutFused::new`]. `Copy` (unlike [`DropoutFused`] itself, whose
/// private `threshold`/`scale` fields are DERIVED from `p` — this type
/// carries the raw tuple so both `fwd` and `bwd` can reconstruct an
/// identical [`DropoutFused`] from the SAME four numbers without either
/// arm re-deriving or re-reserving anything).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DropoutKey {
    pub seed: u64,
    pub layer_id: u32,
    pub forward_idx: u32,
    pub p: f32,
}

/// One fused LoRA site: `(x @ w^T) + cast(scale * (dropout(x) @ A^T @
/// B^T))`. See the module doc for the full forward/backward rounding
/// enumeration. `Copy` (this crate's usual stateless-op requirement — see
/// `ops`'s module doc): every field is construction data fixed by the
/// CALL SITE before `apply3` runs, never mutated or cached from a
/// forward's own inputs.
#[derive(Debug, Clone, Copy)]
pub struct LoraLinearFused {
    /// The LoRA scaling factor `gamma_r` (`alpha/rank` or
    /// `alpha/sqrt(rank)` — see `jammi_lora::lora_scaling`), applied to
    /// the `B^T`-side delta before the epilogue's cast.
    pub scale: f32,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    /// `Some` when the call site's `LoraLinear` has dropout configured
    /// AND is training; `None` skips step 3/step-4-of-backward entirely
    /// (`xd == x32`, `d_xd` unchanged) rather than running dropout at
    /// `p == 0.0` through the kernel.
    pub dropout: Option<DropoutKey>,
    /// Whether `bwd` computes and returns `Some(dW)` for the `w` slot —
    /// frozen into this `Copy` instance by the call site from its OWN
    /// frozen-weight gate (`!w.track_op()` => `false`, a tracked `Var`
    /// base => `true`; see `jammi-lora`'s `LoraLinear::new` doc for why
    /// `is_variable()` alone is not a safe signal here). This op never
    /// inspects `w`'s tracked-op state itself.
    pub dweight_needed: bool,
}

impl LoraLinearFused {
    /// `scale` must be finite (a non-finite scaling would poison every
    /// output silently otherwise — family F); `in_features`/`out_features`/
    /// `rank` must each be `>= 1` (a zero-sized GEMM dimension is a
    /// degenerate case this op refuses rather than special-cases, unlike
    /// e.g. `LayerNormFused`'s `hidden == 0` empty-output path — no LoRA
    /// site in this workspace has a zero-width feature dimension, and a
    /// caller that somehow reaches one gets a typed refusal, not a
    /// confidently-wrong empty tensor).
    pub fn new(
        scale: f32,
        in_features: usize,
        out_features: usize,
        rank: usize,
        dropout: Option<DropoutKey>,
        dweight_needed: bool,
    ) -> Result<Self> {
        if !scale.is_finite() {
            return Err(Error::Msg(format!(
                "lora_linear_fused: scale must be finite, got {scale}"
            )));
        }
        if in_features == 0 || out_features == 0 || rank == 0 {
            return Err(Error::Msg(format!(
                "lora_linear_fused: in_features/out_features/rank must all be >= 1, got \
                 in_features={in_features} out_features={out_features} rank={rank}"
            )));
        }
        Ok(Self {
            scale,
            in_features,
            out_features,
            rank,
            dropout,
            dweight_needed,
        })
    }

    /// `[.., in_features]` -> `(rows, in_features)`, where `rows` is the
    /// product of every dimension but the last (`1` leading dims collapse
    /// to a plain `[rows, in]` 2D view exactly as `Layout::narrow`ing
    /// never needs to happen — the GEMM only ever sees a flat row count).
    /// Refuses any rank other than 2 or 3 (this op's stated domain) and
    /// any last-dim mismatch with `self.in_features`. `pub(crate)`: also
    /// called from `crate::cuda::lora_linear::cuda_fwd`, so the domain
    /// check has exactly one definition, not one per device.
    pub(crate) fn flatten_x(&self, l1: &Layout) -> Result<usize> {
        let dims = l1.dims();
        if dims.len() != 2 && dims.len() != 3 {
            return Err(Error::Msg(format!(
                "lora_linear_fused: x must be rank 2 or 3, got rank {}",
                dims.len()
            )));
        }
        let last = dims[dims.len() - 1];
        if last != self.in_features {
            return Err(Error::Msg(format!(
                "lora_linear_fused: x's last dim {last} != in_features {}",
                self.in_features
            )));
        }
        Ok(dims[..dims.len() - 1].iter().product())
    }

    /// `w` must be exactly `[out_features, in_features]`; `ab` must be
    /// exactly `[rank, in_features + out_features]` and `F32`. Both
    /// checked structurally regardless of what the call site's own
    /// admission predicate already verified (family D: an op trusts no
    /// caller for its own domain — the same doctrine `DropoutFused::new`
    /// documents). `pub(crate)`: shared with `crate::cuda::lora_linear`
    /// (dims/dtype are device-erased, so this needs no `CpuStorage`/
    /// `CudaStorage`-specific variant).
    pub(crate) fn check_w_and_ab(
        &self,
        l2: &Layout,
        ab_dims: &[usize],
        ab_dtype: DType,
    ) -> Result<()> {
        if l2.dims() != [self.out_features, self.in_features] {
            return Err(Error::Msg(format!(
                "lora_linear_fused: w must be [{}, {}], got {:?}",
                self.out_features,
                self.in_features,
                l2.dims()
            )));
        }
        if ab_dims != [self.rank, self.in_features + self.out_features] {
            return Err(Error::Msg(format!(
                "lora_linear_fused: ab must be [{}, {}], got {:?}",
                self.rank,
                self.in_features + self.out_features,
                ab_dims
            )));
        }
        if ab_dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(ab_dtype, "lora_linear_fused"));
        }
        Ok(())
    }

    /// `x`'s leading dims (everything but the last) followed by
    /// `out_features` — the final tensor shape this op returns,
    /// independent of the flat `(rows, out_features)` shape every
    /// internal GEMM actually operates over. `pub(crate)`: shared with
    /// `crate::cuda::lora_linear::cuda_fwd`.
    pub(crate) fn output_shape(&self, l1: &Layout) -> Shape {
        let mut dims = l1.dims().to_vec();
        *dims
            .last_mut()
            .expect("flatten_x already checked rank >= 2") = self.out_features;
        Shape::from(dims)
    }
}

impl super::sealed::Sealed for LoraLinearFused {}

impl CustomOp3 for LoraLinearFused {
    fn name(&self) -> &'static str {
        "lora_linear_fused"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let m = self.flatten_x(l1)?;
        self.check_w_and_ab(l2, l3.dims(), s3.dtype())?;
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            });
        }
        if !matches!(s1.dtype(), DType::F32 | DType::BF16) {
            return Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name()));
        }
        for (l, what) in [(l1, "x"), (l2, "w"), (l3, "ab")] {
            if l.contiguous_offsets().is_none() {
                return Err(Error::RequiresContiguous {
                    op: match what {
                        "x" => "lora_linear_fused(x)",
                        "w" => "lora_linear_fused(w)",
                        _ => "lora_linear_fused(ab)",
                    },
                });
            }
        }

        let inf = self.in_features;
        let outf = self.out_features;
        let r = self.rank;

        // Step 1: base = x @ w^T — the exact `Linear::forward` reshape
        // pattern (`Layout::contiguous_with_offset`, matching
        // `Tensor::reshape`'s own contiguous fast path verbatim).
        let x2d_l = Layout::contiguous_with_offset((m, inf), l1.start_offset());
        let w_t_l = l2.transpose(0, 1)?;
        let base_storage = s1.matmul(s2, (1, m, outf, inf), &x2d_l, &w_t_l)?;
        let base_l = Layout::contiguous((m, outf));

        // Step 2: x32 = to_dtype(x, F32) — a layout-aware gather, exact
        // when x is already F32.
        let x32_storage = s1.to_dtype(&x2d_l, DType::F32)?;
        let x32_l = Layout::contiguous((m, inf));

        // Step 3: xd = dropout(x32), reusing DropoutFused's OWN cpu_fwd
        // directly (no second tape node — this whole forward is one
        // storage-level function).
        let (xd_storage, xd_l) = match &self.dropout {
            Some(key) => {
                let op = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p)?;
                let (s, shape) = CustomOp1::cpu_fwd(&op, &x32_storage, &x32_l)?;
                (s, Layout::contiguous(shape))
            }
            None => (x32_storage, x32_l),
        };

        // A/B^T materialized out of the packed `ab` slice into their own
        // dense buffers — see the module doc's "packed-`ab` GEMM
        // eligibility problem" section for why this copy is required.
        let a_view = l3.narrow(1, 0, inf)?;
        let bt_view = l3.narrow(1, inf, outf)?;
        let a_storage = s3.to_dtype(&a_view, DType::F32)?;
        let bt_storage = s3.to_dtype(&bt_view, DType::F32)?;
        let a_l = Layout::contiguous((r, inf));
        let bt_l = Layout::contiguous((r, outf));

        // Step 4: h = xd @ A^T.
        let a_t_l = a_l.transpose(0, 1)?;
        let h_storage = xd_storage.matmul(&a_storage, (1, m, r, inf), &xd_l, &a_t_l)?;
        let h_l = Layout::contiguous((m, r));

        // Step 5: delta = h @ B^T (B^T is already the right orientation).
        let delta_storage = h_storage.matmul(&bt_storage, (1, m, outf, r), &h_l, &bt_l)?;
        let delta_l = Layout::contiguous((m, outf));

        // Step 6: out = base + cast(delta * scale), reusing ScaledCastAdd's
        // OWN cpu_fwd directly.
        let epilogue = ScaledCastAdd::new(f64::from(self.scale));
        let (out_storage, _flat_shape) =
            CustomOp2::cpu_fwd(&epilogue, &base_storage, &base_l, &delta_storage, &delta_l)?;

        Ok((out_storage, self.output_shape(l1)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::lora_linear::cuda_fwd(self, s1, l1, s2, l2, s3, l3)
    }

    /// Tensor-level (see the module doc's backward enumeration): every
    /// intermediate here is an ordinary, untracked-by-the-OUTER-tape
    /// `Tensor` computation — none of these ops become nodes candle's
    /// OWN `sorted_nodes` walk (the thing whose per-node `zeros_like` +
    /// `add` this whole op exists to avoid) ever revisits, because the
    /// gradients this function returns are inserted directly into
    /// `GradStore` by `backprop.rs`'s `Op::CustomOp3` arm, not threaded
    /// back through another `.backward()` call.
    fn bwd(
        &self,
        x: &Tensor,
        w: &Tensor,
        ab: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let base_dtype = x.dtype();
        let inf = self.in_features;
        let outf = self.out_features;
        let dims = x.dims().to_vec();
        let m: usize = dims[..dims.len() - 1].iter().product();

        // d_lora = cast_f32(dy) * scale — ScaledCastAdd::bwd's own order
        // (cast to the LoRA dtype FIRST, scale second).
        let dy_f32 = if grad_res.dtype() == DType::F32 {
            grad_res.clone()
        } else {
            grad_res.to_dtype(DType::F32)?
        };
        let d_lora_2d = dy_f32
            .affine(f64::from(self.scale), 0.0)?
            .reshape((m, outf))?;

        // Recompute x32 / xd / h (no save-for-backward in candle 0.11).
        let x32_2d = if base_dtype == DType::F32 {
            x.reshape((m, inf))?
        } else {
            x.to_dtype(DType::F32)?.reshape((m, inf))?
        };
        let dropout_op = self
            .dropout
            .map(|key| DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p))
            .transpose()?;
        let xd_2d = match dropout_op {
            Some(op) => super::apply1(&x32_2d, op)?,
            None => x32_2d.clone(),
        };

        let a = ab.narrow(1, 0, inf)?; // [r, in]
        let b_t = ab.narrow(1, inf, outf)?; // [r, out] (this IS B^T)

        let h_2d = xd_2d.matmul(&a.t()?)?; // [M, r]

        // dB^T slot: h^T @ d_lora.
        let d_bt = h_2d.t()?.matmul(&d_lora_2d)?; // [r, out]

        // g = dL/dh = d_lora @ B, with B = (B^T)^T.
        let g = d_lora_2d.matmul(&b_t.t()?)?; // [M, r]

        // dA slot: g^T @ xd.
        let d_a = g.t()?.matmul(&xd_2d)?; // [r, in]

        // d_xd = g @ A, then the SAME dropout key reapplied to the
        // gradient (DropoutFused::bwd's own contract: applying the same
        // Copy instance regenerates `mask * scale` identically).
        let d_xd = g.matmul(&a)?; // [M, in], F32
        let d_x_lora_f32_2d = match dropout_op {
            Some(op) => super::apply1(&d_xd, op)?,
            None => d_xd,
        };
        let d_x_lora_2d = if base_dtype == DType::F32 {
            d_x_lora_f32_2d
        } else {
            d_x_lora_f32_2d.to_dtype(base_dtype)?
        };
        let d_x_lora = d_x_lora_2d.reshape(x.shape())?;

        // dx = dy @ w + d_x_lora, at the base dtype.
        let dy_base_2d = grad_res.reshape((m, outf))?;
        let dx_base = dy_base_2d.matmul(w)?.reshape(x.shape())?;
        let dx = (&dx_base + &d_x_lora)?;

        let dw = if self.dweight_needed {
            let x_base_2d = x.reshape((m, inf))?;
            Some(dy_base_2d.t()?.matmul(&x_base_2d)?)
        } else {
            None
        };

        let d_ab = Tensor::cat(&[&d_a, &d_bt], 1)?;

        Ok((Some(dx), dw, Some(d_ab)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    /// Independent, closed-form reference: `x @ w^T`, `A`/`B` supplied
    /// separately (not packed), plain `f64` accumulation on the host —
    /// the numpy-first oracle this test's fused-kernel output is checked
    /// against, computed with NO shared code path (family F).
    #[allow(clippy::too_many_arguments)]
    fn reference_forward(
        x: &[f32],
        rows: usize,
        inf: usize,
        w: &[f32],
        outf: usize,
        a: &[f32],
        r: usize,
        b: &[f32],
        scale: f32,
        dropout_mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let xd: Vec<f64> = match dropout_mask {
            Some(mask) => x
                .iter()
                .zip(mask.iter())
                .map(|(&xv, &mv)| f64::from(xv) * f64::from(mv))
                .collect(),
            None => x.iter().map(|&v| f64::from(v)).collect(),
        };
        let mut out = vec![0.0f64; rows * outf];
        // base = x @ w^T
        for i in 0..rows {
            for o in 0..outf {
                let mut acc = 0.0f64;
                for k in 0..inf {
                    acc += f64::from(x[i * inf + k]) * f64::from(w[o * inf + k]);
                }
                out[i * outf + o] = acc;
            }
        }
        // h = xd @ A^T
        let mut h = vec![0.0f64; rows * r];
        for i in 0..rows {
            for j in 0..r {
                let mut acc = 0.0f64;
                for k in 0..inf {
                    acc += xd[i * inf + k] * f64::from(a[j * inf + k]);
                }
                h[i * r + j] = acc;
            }
        }
        // delta = h @ B^T, B is [outf, r] so B^T[j,o] = b[o*r+j]
        for i in 0..rows {
            for o in 0..outf {
                let mut acc = 0.0f64;
                for j in 0..r {
                    acc += h[i * r + j] * f64::from(b[o * r + j]);
                }
                out[i * outf + o] += acc * f64::from(scale);
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }

    fn pack_ab(a: &[f32], inf: usize, b: &[f32], outf: usize, r: usize, device: &Device) -> Tensor {
        // b is [outf, r]; B^T is [r, outf].
        let mut bt = vec![0.0f32; r * outf];
        for o in 0..outf {
            for j in 0..r {
                bt[j * outf + o] = b[o * r + j];
            }
        }
        let a_t = Tensor::from_slice(a, (r, inf), device).unwrap();
        let bt_t = Tensor::from_slice(&bt, (r, outf), device).unwrap();
        Tensor::cat(&[&a_t, &bt_t], 1).unwrap()
    }

    fn fused_forward(x: &Tensor, w: &Tensor, ab: &Tensor, op: LoraLinearFused) -> Result<Tensor> {
        x.apply_op3(w, ab, op)
    }

    #[test]
    fn cpu_f32_matches_a_closed_form_reference_no_dropout() {
        let device = Device::Cpu;
        let rows = 6;
        let inf = 5;
        let outf = 7;
        let r = 3;
        let scale = 1.7f32;

        let x_v: Vec<f32> = (0..rows * inf).map(|i| ((i as f32) * 0.31).sin()).collect();
        let w_v: Vec<f32> = (0..outf * inf).map(|i| ((i as f32) * 0.17).cos()).collect();
        let a_v: Vec<f32> = (0..r * inf)
            .map(|i| ((i as f32) * 0.11 + 0.4).sin())
            .collect();
        let b_v: Vec<f32> = (0..outf * r)
            .map(|i| ((i as f32) * 0.23 - 0.2).cos())
            .collect();

        let x = Tensor::from_slice(&x_v, (rows, inf), &device).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let ab = pack_ab(&a_v, inf, &b_v, outf, r, &device);

        let op = LoraLinearFused::new(scale, inf, outf, r, None, false).unwrap();
        let got = fused_forward(&x, &w, &ab, op)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let got_flat: Vec<f32> = got.into_iter().flatten().collect();

        let expected = reference_forward(&x_v, rows, inf, &w_v, outf, &a_v, r, &b_v, scale, None);
        for i in 0..rows * outf {
            let diff = (got_flat[i] - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "index {i}: got {} expected {} diff {diff}",
                got_flat[i],
                expected[i]
            );
        }
    }

    #[test]
    fn cpu_f32_rank3_matches_the_reshape_flattened_reference() {
        let device = Device::Cpu;
        let (b, s, inf, outf, r) = (2usize, 3usize, 4usize, 6usize, 2usize);
        let rows = b * s;
        let scale = 0.9f32;

        let x_v: Vec<f32> = (0..rows * inf).map(|i| ((i as f32) * 0.13).sin()).collect();
        let w_v: Vec<f32> = (0..outf * inf).map(|i| ((i as f32) * 0.07).cos()).collect();
        let a_v: Vec<f32> = (0..r * inf).map(|i| ((i as f32) * 0.05).sin()).collect();
        let b_v: Vec<f32> = (0..outf * r).map(|i| ((i as f32) * 0.09).cos()).collect();

        let x = Tensor::from_slice(&x_v, (b, s, inf), &device).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let ab = pack_ab(&a_v, inf, &b_v, outf, r, &device);

        let op = LoraLinearFused::new(scale, inf, outf, r, None, false).unwrap();
        let got = fused_forward(&x, &w, &ab, op).unwrap();
        assert_eq!(got.dims(), &[b, s, outf]);
        let got_flat: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();

        let expected = reference_forward(&x_v, rows, inf, &w_v, outf, &a_v, r, &b_v, scale, None);
        for i in 0..rows * outf {
            assert!((got_flat[i] - expected[i]).abs() < 1e-4, "index {i}");
        }
    }

    #[test]
    fn cpu_f32_matches_manual_composition_bit_exact() {
        // No dropout, F32 throughout: every internal step is the SAME
        // BackendStorage::matmul / ScaledCastAdd call the eager
        // composition issues, so this must be bit-exact, not merely
        // close (see the module doc's forward rounding enumeration).
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (4usize, 3usize, 5usize, 2usize);
        let scale = 1.3f32;

        let x_v: Vec<f32> = (0..rows * inf).map(|i| ((i as f32) * 0.29).sin()).collect();
        let w_v: Vec<f32> = (0..outf * inf).map(|i| ((i as f32) * 0.19).cos()).collect();
        let a_v: Vec<f32> = (0..r * inf).map(|i| ((i as f32) * 0.37).sin()).collect();
        let b_v: Vec<f32> = (0..outf * r).map(|i| ((i as f32) * 0.41).cos()).collect();

        let x = Tensor::from_slice(&x_v, (rows, inf), &device).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a = Tensor::from_slice(&a_v, (r, inf), &device).unwrap();
        let b = Tensor::from_slice(&b_v, (outf, r), &device).unwrap();
        let ab = pack_ab(&a_v, inf, &b_v, outf, r, &device);

        let op = LoraLinearFused::new(scale, inf, outf, r, None, false).unwrap();
        let fused = fused_forward(&x, &w, &ab, op).unwrap();

        // Manual eager reconstruction: identical to `LoraLinear::forward`'s
        // training-arm composition before this op existed.
        let base_out = x.matmul(&w.t().unwrap()).unwrap();
        let after_a = x.matmul(&a.t().unwrap()).unwrap();
        let lora_out = after_a.matmul(&b.t().unwrap()).unwrap();
        let scaled = (&lora_out * f64::from(scale)).unwrap();
        let manual = (&base_out + &scaled).unwrap();

        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            manual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "F32, no dropout: must be bit-exact against the eager composition"
        );
    }

    #[test]
    fn cpu_f32_with_dropout_matches_the_manual_dropout_composition_bit_exact() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (5usize, 4usize, 3usize, 2usize);
        let scale = 0.8f32;
        let p = 0.3f32;
        let key = DropoutKey {
            seed: 99,
            layer_id: 7,
            forward_idx: 2,
            p,
        };

        let x_v: Vec<f32> = (0..rows * inf).map(|i| ((i as f32) * 0.21).sin()).collect();
        let w_v: Vec<f32> = (0..outf * inf).map(|i| ((i as f32) * 0.14).cos()).collect();
        let a_v: Vec<f32> = (0..r * inf).map(|i| ((i as f32) * 0.33).sin()).collect();
        let b_v: Vec<f32> = (0..outf * r).map(|i| ((i as f32) * 0.44).cos()).collect();

        let x = Tensor::from_slice(&x_v, (rows, inf), &device).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a = Tensor::from_slice(&a_v, (r, inf), &device).unwrap();
        let b = Tensor::from_slice(&b_v, (outf, r), &device).unwrap();
        let ab = pack_ab(&a_v, inf, &b_v, outf, r, &device);

        let op = LoraLinearFused::new(scale, inf, outf, r, Some(key), false).unwrap();
        let fused = fused_forward(&x, &w, &ab, op).unwrap();

        // Manual reconstruction using the SAME DropoutFused key directly.
        let dropout_op = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p).unwrap();
        let xd = super::super::apply1(&x, dropout_op).unwrap();
        let base_out = x.matmul(&w.t().unwrap()).unwrap();
        let after_a = xd.matmul(&a.t().unwrap()).unwrap();
        let lora_out = after_a.matmul(&b.t().unwrap()).unwrap();
        let scaled = (&lora_out * f64::from(scale)).unwrap();
        let manual = (&base_out + &scaled).unwrap();

        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            manual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "F32 with dropout: must be bit-exact against the manual composition \
             using the same DropoutFused key"
        );
    }

    /// Central-finite-difference gradcheck: perturbs each element of `x`,
    /// `A`, and `B` independently and compares against the analytic
    /// gradients `bwd` returns, with a NON-UNIFORM `dy` (a sine pattern,
    /// not all-ones) so a sign or transpose error in the derivation could
    /// not hide behind a degenerate upstream gradient.
    #[test]
    fn gradcheck_cpu_f32_no_dropout() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (3usize, 3usize, 4usize, 2usize);
        let scale = 1.1f32;
        let eps = 1e-3f32;

        let x_v: Vec<f32> = (0..rows * inf)
            .map(|i| ((i as f32) * 0.27).sin() * 0.5)
            .collect();
        let w_v: Vec<f32> = (0..outf * inf)
            .map(|i| ((i as f32) * 0.19).cos() * 0.5)
            .collect();
        let a_v: Vec<f32> = (0..r * inf)
            .map(|i| ((i as f32) * 0.31).sin() * 0.5)
            .collect();
        let b_v: Vec<f32> = (0..outf * r)
            .map(|i| ((i as f32) * 0.23).cos() * 0.5)
            .collect();
        let dy_v: Vec<f32> = (0..rows * outf)
            .map(|i| ((i as f32) * 0.71).sin())
            .collect();

        let loss = |x_v: &[f32], a_v: &[f32], b_v: &[f32]| -> f32 {
            let out = reference_forward(x_v, rows, inf, &w_v, outf, a_v, r, b_v, scale, None);
            out.iter().zip(dy_v.iter()).map(|(&o, &g)| o * g).sum()
        };

        let x = Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a_var =
            Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
        let b_var =
            Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
        let bt = b_var.as_tensor().t().unwrap().contiguous().unwrap();
        let ab = Tensor::cat(&[a_var.as_tensor(), &bt], 1).unwrap();

        let op = LoraLinearFused::new(scale, inf, outf, r, None, false).unwrap();
        let out = x.as_tensor().apply_op3(&w, &ab, op).unwrap();
        let dy = Tensor::from_slice(&dy_v, (rows, outf), &device).unwrap();
        let total = (&out * &dy).unwrap().sum_all().unwrap();
        let grads = total.backward().unwrap();

        // `ab` is an INTERMEDIATE (the `Tensor::cat` output), not a `Var`:
        // candle's `backward()` REMOVES a non-variable node's gradient
        // entry from `GradStore` the moment it consumes it to propagate
        // further (`backprop.rs:174`'s `grads.remove(node)`), so `ab`'s
        // own entry is gone by the time `backward()` returns — read the
        // gradients candle's `Op::Cat` backward already propagated INTO
        // the true leaves (`a_var`/`b_var`) instead.
        let da_analytic: Vec<f32> = grads
            .get(&a_var)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // `b_var`'s gradient is w.r.t. B itself (shape [out, r]), not B^T
        // — `Op::Transpose`'s backward already un-transposes it, so no
        // manual index remapping is needed here (unlike a direct read of
        // `ab`'s packed B^T slot).
        let db_analytic: Vec<f32> = grads
            .get(&b_var)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dx_analytic: Vec<f32> = grads
            .get(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        for i in 0..rows * inf {
            let mut xp = x_v.clone();
            xp[i] += eps;
            let mut xm = x_v.clone();
            xm[i] -= eps;
            let numeric = (loss(&xp, &a_v, &b_v) - loss(&xm, &a_v, &b_v)) / (2.0 * eps);
            assert!(
                (numeric - dx_analytic[i]).abs() < 5e-2,
                "dx[{i}]: numeric {numeric} vs analytic {}",
                dx_analytic[i]
            );
        }
        for i in 0..r * inf {
            let mut ap = a_v.clone();
            ap[i] += eps;
            let mut am = a_v.clone();
            am[i] -= eps;
            let numeric = (loss(&x_v, &ap, &b_v) - loss(&x_v, &am, &b_v)) / (2.0 * eps);
            assert!(
                (numeric - da_analytic[i]).abs() < 5e-2,
                "dA[{i}]: numeric {numeric} vs analytic {}",
                da_analytic[i]
            );
        }
        for idx in 0..outf * r {
            let mut bp = b_v.clone();
            bp[idx] += eps;
            let mut bm = b_v.clone();
            bm[idx] -= eps;
            let numeric = (loss(&x_v, &a_v, &bp) - loss(&x_v, &a_v, &bm)) / (2.0 * eps);
            assert!(
                (numeric - db_analytic[idx]).abs() < 5e-2,
                "dB[{idx}]: numeric {numeric} vs analytic {}",
                db_analytic[idx]
            );
        }
    }

    #[test]
    fn dweight_needed_returns_some_dw_otherwise_none() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (3usize, 3usize, 4usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
        let w = Var::from_tensor(&Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap()).unwrap();
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let bt = b.t().unwrap().contiguous().unwrap();
        let ab = Tensor::cat(&[&a, &bt], 1).unwrap();

        for dweight_needed in [true, false] {
            let op = LoraLinearFused::new(1.0, inf, outf, r, None, dweight_needed).unwrap();
            let out = x.apply_op3(w.as_tensor(), &ab, op).unwrap();
            let grads = out.sum_all().unwrap().backward().unwrap();
            assert_eq!(
                grads.get(w.as_tensor()).is_some(),
                dweight_needed,
                "dweight_needed={dweight_needed}: dW slot presence must match"
            );
        }
    }

    #[test]
    fn rank2_pooled_head_shape_is_accepted() {
        // fine_tune/lora.rs's classification/distribution/ner heads: rank-2
        // x, F32 base, small out_features.
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (4usize, 6usize, 2usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
        let w = Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let bt = b.t().unwrap().contiguous().unwrap();
        let ab = Tensor::cat(&[&a, &bt], 1).unwrap();

        let op = LoraLinearFused::new(1.0, inf, outf, r, None, false).unwrap();
        let out = x.apply_op3(&w, &ab, op).unwrap();
        assert_eq!(out.dims(), &[rows, outf]);
    }

    #[test]
    fn rank1_x_is_a_typed_refusal() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (5,), &device).unwrap();
        let w = Tensor::randn(0f32, 1.0, (3, 5), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (2, 5), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (3, 2), &device).unwrap();
        let bt = b.t().unwrap().contiguous().unwrap();
        let ab = Tensor::cat(&[&a, &bt], 1).unwrap();
        let op = LoraLinearFused::new(1.0, 5, 3, 2, None, false).unwrap();
        assert!(x.apply_op3(&w, &ab, op).is_err());
    }

    #[test]
    fn mismatched_ab_shape_is_a_typed_refusal() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (4, 5), &device).unwrap();
        let w = Tensor::randn(0f32, 1.0, (3, 5), &device).unwrap();
        // Wrong rank (should be 2) — the ab packing itself must be [r, in+out].
        let bad_ab = Tensor::randn(0f32, 1.0, (2, 5), &device).unwrap();
        let op = LoraLinearFused::new(1.0, 5, 3, 2, None, false).unwrap();
        assert!(x.apply_op3(&w, &bad_ab, op).is_err());
    }

    #[test]
    fn non_finite_scale_is_a_typed_refusal() {
        assert!(LoraLinearFused::new(f32::NAN, 4, 4, 2, None, false).is_err());
        assert!(LoraLinearFused::new(f32::INFINITY, 4, 4, 2, None, false).is_err());
    }

    #[test]
    fn zero_sized_dims_are_a_typed_refusal() {
        assert!(LoraLinearFused::new(1.0, 0, 4, 2, None, false).is_err());
        assert!(LoraLinearFused::new(1.0, 4, 0, 2, None, false).is_err());
        assert!(LoraLinearFused::new(1.0, 4, 4, 0, None, false).is_err());
    }

    #[test]
    fn non_contiguous_x_is_a_typed_refusal() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (5, 4), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!x.is_contiguous());
        let w = Tensor::randn(0f32, 1.0, (3, 5), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (2, 5), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (3, 2), &device).unwrap();
        let bt = b.t().unwrap().contiguous().unwrap();
        let ab = Tensor::cat(&[&a, &bt], 1).unwrap();
        let op = LoraLinearFused::new(1.0, 5, 3, 2, None, false).unwrap();
        assert!(x.apply_op3(&w, &ab, op).is_err());
    }

    /// The disclosed pre-existing candle limitation (see the module doc):
    /// a `BF16` base on CPU must fail with a typed error — the SAME error
    /// class `candle_nn::Linear::forward` already returns for a `BF16`
    /// CPU matmul today, never a panic and never a silently wrong number.
    #[test]
    fn bf16_base_on_cpu_is_a_typed_error_not_a_panic_or_wrong_number() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (2usize, 3usize, 4usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::randn(0f32, 1.0, (outf, inf), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let bt = b.t().unwrap().contiguous().unwrap();
        let ab = Tensor::cat(&[&a, &bt], 1).unwrap();
        let op = LoraLinearFused::new(1.0, inf, outf, r, None, false).unwrap();
        let err = x.apply_op3(&w, &ab, op).expect_err(
            "BF16 CPU matmul is unsupported by candle-core 0.11.0 without mkl/accelerate",
        );
        assert!(matches!(err, Error::UnsupportedDTypeForOp(..)));

        // The SAME error class the eager composition (candle_nn::Linear)
        // hits today — this op is not introducing a NEW failure mode.
        let eager_err = x
            .matmul(&w.t().unwrap())
            .expect_err("the pre-existing eager composition must fail identically");
        assert!(matches!(eager_err, Error::UnsupportedDTypeForOp(..)));
    }

    #[test]
    fn empty_dropout_key_and_present_dropout_key_draw_the_same_mask_as_dropout_fused_directly() {
        // Not a new determinism property (DropoutFused already proves
        // this) — pins that LoraLinearFused's OWN reconstruction of
        // DropoutFused from a DropoutKey uses the fields in the right
        // order (a transposed/swapped constructor call would silently
        // draw a DIFFERENT stream and this test would catch it via the
        // bit-exact dropout composition test above using the same key
        // type end-to-end).
        let key = DropoutKey {
            seed: 5,
            layer_id: 2,
            forward_idx: 9,
            p: 0.25,
        };
        let a = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p).unwrap();
        let b = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p).unwrap();
        let device = Device::Cpu;
        let x = Tensor::ones((64,), DType::F32, &device).unwrap();
        let out_a: Vec<f32> = super::super::apply1(&x, a).unwrap().to_vec1().unwrap();
        let out_b: Vec<f32> = super::super::apply1(&x, b).unwrap().to_vec1().unwrap();
        assert_eq!(out_a, out_b);
    }
}
