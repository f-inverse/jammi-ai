//! One tape node per low-rank residual site: a frozen linear map `x @
//! w^T` plus a trainable low-rank correction, `out = (x @ w^T) +
//! cast(scale * (dropout(x) @ A^T @ B^T))` — `w` is `[out, in]` and
//! FROZEN, `A` is `[rank, in]` and `B` is `[out, rank]`, both trainable,
//! `rank << min(in, out)`.
//!
//! This is the fused replacement for the eager composition this math
//! builds from ~11 candle ops (a `Linear`-style reshape/matmul/reshape for
//! `base`, the same for the `A` and `B` sub-linears, a `to_dtype`, an
//! optional dropout node, and [`super::ScaledCastAdd`]'s own `[mul, cast,
//! add]`): each of those is its own node on candle's backward tape, and
//! candle's `GradStore::or_insert` (`backprop.rs`) allocates a FULL-SIZE
//! `zeros_like` + `add` for every one of them, PLUS an unconditional `dW`
//! GEMM for the frozen base weight (`Op::Matmul`'s backward computes both
//! operands' gradients regardless of whether either side is trainable —
//! `backprop.rs:457-468`). Collapsing the whole site into one
//! `CustomOp3` removes all of that: one node, one (or two, if the base
//! weight is itself a trainable `Var`) gradient contribution. MEASURED
//! (not estimated) via `Tensor::sorted_nodes().len()` — the exact list
//! `Tensor::backward` walks — at `x`/`A`/`B` as `Var`s and `w` a plain
//! (frozen, untracked) leaf, `F32`, no dropout: the eager composition
//! this op replaces retains 10 tape nodes end to end (3 `Var` leaves + 7
//! tracked intermediates/output); this op's own site retains 6 (the same
//! 3 leaves + `A.t()` + the `ab` pack + this op's own single node) — see
//! `fused_site_retains_fewer_tape_nodes_than_the_eager_composition`'s own
//! doc for the full per-node accounting.
//!
//! ## The three tensor arguments
//!
//! - `x`: `[.., in]` (rank 2 — a pooled/flattened activation — or rank 3
//!   — `[batch, seq, in]`), the backbone dtype (`F32` or `BF16`).
//! - `w`: `[out, in]`, the FROZEN base weight, same dtype as `x`.
//! - `ab`: `F32` `[in + out, rank]`, THE pack layout — row 0 through
//!   `in - 1` holds `A^T` (`[in, rank]`), row `in` through `in + out - 1`
//!   holds `B` AS-IS (`[out, rank]`, no pre-transpose): `Tensor::cat([A.t(),
//!   B], 0)`, built at the call site fresh every forward. See "the packed-
//!   `ab` GEMM eligibility problem" below for why THIS orientation (stack
//!   along the ROW axis, dim 0) rather than the column axis is what makes
//!   both slices GEMM-eligible with zero copies. `Tensor::cat`'s own
//!   backward (`Op::Cat`, `backprop.rs:469`) splits this op's `dab`
//!   gradient back into `dA^T`/`dB` via two cheap `narrow`s — tiny (`rank`
//!   columns), not the concern this op addresses. The row-stacked layout
//!   also leaves room for a future bias block to be appended as EXTRA
//!   leading or trailing dim-0 rows without disturbing either existing
//!   slice's own stride (a dim-0 narrow's stride is invariant to how many
//!   OTHER rows exist elsewhere in the buffer) — noted for the record, not
//!   implemented: see "Bias: a domain refusal" below for why this op still
//!   refuses a bias today.
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
//! 4. `h = xd @ A^T` — a second `F32` GEMM. `ab`'s first `in` rows ARE
//!    `A^T` already (`[in, rank]`, row-contiguous), so this GEMM's rhs is
//!    `ab.narrow(0, 0, in)` used DIRECTLY — no copy, no further transpose
//!    (see "the packed-`ab` GEMM eligibility problem" below for why this
//!    slicing is zero-cost, unlike the column-packed layout it replaces).
//! 5. `delta = h @ B^T` — a third `F32` GEMM. `ab`'s remaining `out` rows
//!    ARE `B` (`[out, rank]`, row-contiguous); `B^T` is that slice's
//!    `.transpose(0, 1)` VIEW (`[rank, out]`, column-contiguous) — a
//!    zero-copy reinterpretation `gemm_config`'s `CUBLAS_OP_T` branch
//!    accepts directly (same citation below).
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
//! 3. `dB = d_lora^T @ h` (matching `B`'s own `[out, rank]` orientation in
//!    `ab`), `g = d_lora @ B`, `dA^T = xd^T @ g` (matching `A^T`'s own
//!    `[in, rank]` orientation in `ab`), `d_xd = g @ A` — four `F32`
//!    GEMMs, no rounding (`F32` throughout). (`dA^T`/`dB` here are
//!    ALGEBRAICALLY the same values as a `dA`/`dB^T`-shaped derivation
//!    would produce, just transposed to match `ab`'s own packed
//!    orientation — the underlying math is unchanged by which slot the
//!    pack layout stores.)
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
//! An EARLIER version of this op packed `ab = cat([A, B^T], 1)` — `A`
//! (`[rank, in]`) and `B^T` (`[rank, out]`) SIDE BY SIDE along the feature
//! axis (`[rank, in + out]`). Slicing either back out via
//! `Layout::narrow(1, ..)` yields a view whose ROW STRIDE is `in + out` —
//! wider than its own logical width (`in` or `out`). CUDA's `gemm_config`
//! (`cuda_backend/mod.rs:1398-1408`, `1412-1422`) admits an operand in
//! EXACTLY two stride shapes: `rhs_m1 == 1 && rhs_m2 == n` (row-contiguous
//! WITH `row_stride == width`, `CUBLAS_OP_N`) or `rhs_m1 == k && rhs_m2 ==
//! 1` (column-contiguous over the FULL matrix, `CUBLAS_OP_T`) — a
//! narrower-than-its-storage-row slice satisfies NEITHER (`rhs_m2` is the
//! padded `in + out`, matching neither `n` nor `1`), so cuBLAS refused it
//! with `MatMulNonContiguous`. This was confirmed on-device (an A100 pod
//! run against production transformer-encoder widths): EVERY
//! `lora_linear_parity_*` CUDA leg failed with exactly this error at the
//! `h = xd @ A^T` GEMM.
//!
//! **THE FIX: pack along the ROW axis (dim 0) instead.** `ab =
//! cat([A.t(), B], 0)` has shape `[in + out, rank]`. A leading-row slice
//! of a row-major matrix (`Layout::narrow(0, start, len)`) NEVER changes
//! the row stride — narrowing dim 0 only moves `start_offset` and shrinks
//! the row COUNT, leaving `stride == [rank, 1]` untouched — so BOTH slices
//! come out GEMM-eligible with **zero copies**, verified against the same
//! `gemm_config` rules the failure above cites:
//!
//! - `a_t_l = ab.narrow(0, 0, in)`: shape `[in, rank]`, stride `[rank, 1]`
//!   — `rhs_m1 == 1 && rhs_m2 == rank == n` (the GEMM's own `n` for
//!   `h = xd @ A^T`) → `CUBLAS_OP_N`, admitted directly.
//! - `b_l = ab.narrow(0, in, out)`: shape `[out, rank]`, stride
//!   `[rank, 1]`; `b_t_l = b_l.transpose(0, 1)`: shape `[rank, out]`,
//!   stride `[1, rank]` — `rhs_m1 == rank == k && rhs_m2 == 1` (the GEMM's
//!   own `k` for `delta = h @ B^T`) → `CUBLAS_OP_T`, admitted directly.
//!
//! Backward mirrors this exactly: `g = d_lora @ B` reads `ab`'s `B` block
//! (`CUBLAS_OP_N`, same shape as forward's `b_l`); `d_xd = g @ A` needs
//! `A = a_t_l.transpose(0, 1)` (`CUBLAS_OP_T`, same pattern as `b_t_l`
//! above); `dA^T = xd^T @ g` and `dB = d_lora^T @ h` are FRESH `F32`
//! tensors (not slices of `ab`), so they carry no eligibility question of
//! their own — `Tensor::cat(&[dA^T, dB], 0)` reassembles `d_ab` at the
//! Tensor level (candle's `Op::Cat` backward, not this op's own concern).
//!
//! This is strictly better than the copy-based workaround the column
//! layout required: no `to_dtype` gather-copy, no extra tape/storage
//! allocation, and BIT-EXACT relative to whatever the un-copied GEMM
//! itself would have produced (there is no separate "copy" step left to
//! introduce a divergent rounding order at all).
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
//! op's domain check tries to route around: `BF16` production forwards are
//! expected to run on CUDA only (a call site's own admission predicate is
//! what decides this — see the domain section below), and the CPU
//! oracle suite here covers the `(F32, F32, F32)` combination end-to-end
//! plus the typed-error boundary for `(BF16, BF16, F32)` on CPU.
//!
//! ## Bias: a domain refusal, not packed into `ab`
//!
//! A frozen linear base MAY carry a bias (`candle_nn::linear`'s
//! `bias.is_some()`); this op has no bias slot. Packing a bias
//! contribution into a single augmented matmul was evaluated and
//! rejected: turning `y = x @ w^T + b` into a single matmul over an
//! AUGMENTED input
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
//! `x` rank 2 or 3, `w` rank 2 `[out, in]`, `ab` rank 2 `[in+out, rank]`
//! (see "the packed-`ab` GEMM eligibility problem" above for why THIS
//! orientation, not `[rank, in+out]`); dtype pairs `(F32, F32, F32)` and
//! `(BF16, BF16, F32)` (base dtype must match between `x`/`w`; `ab` is
//! always `F32` by this op's own domain requirement, regardless of the
//! base dtype — see [`super::ScaledCastAdd`]'s own doc for the analogous
//! epilogue requirement);
//! `w`/`ab` contiguous (`Layout::is_contiguous`) — a hard refusal, since
//! the call site is expected to control their layout by construction
//! (a `VarBuilder`-loaded weight, a freshly `Tensor::cat`-ed `ab`); `x` MAY
//! be non-contiguous (e.g. a transposed activation view) — this op
//! materializes a dense copy of `x` internally rather than refusing (see
//! [`materialize_contiguous_if_needed`]'s doc), since `x` is the one
//! argument a caller does not fully control the layout of. `out >= 1`,
//! `rank >= 1` (validated once, at construction, in
//! [`LowRankResidualLinear::new`]); the call site is
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
/// per site per forward by the CALL SITE's own key-reservation function
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

/// One fused low-rank residual site: `(x @ w^T) + cast(scale *
/// (dropout(x) @ A^T @ B^T))`. See the module doc for the full
/// forward/backward rounding enumeration. `Copy` (this crate's usual
/// stateless-op requirement — see `ops`'s module doc): every field is
/// construction data fixed by the CALL SITE before `apply3` runs, never
/// mutated or cached from a forward's own inputs.
#[derive(Debug, Clone, Copy)]
pub struct LowRankResidualLinear {
    /// The scaling factor `gamma_r` applied to the `B^T`-side delta
    /// before the epilogue's cast — a caller-supplied constant (typically
    /// `alpha/rank` or `alpha/sqrt(rank)` in a LoRA-style
    /// parameterization; this op has no opinion on which formula produced
    /// it).
    pub scale: f32,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    /// `Some` when the call site has dropout configured AND is training;
    /// `None` skips step 3/step-4-of-backward entirely (`xd == x32`,
    /// `d_xd` unchanged) rather than running dropout at `p == 0.0`
    /// through the kernel.
    pub dropout: Option<DropoutKey>,
    /// Whether `bwd` computes and returns `Some(dW)` for the `w` slot —
    /// frozen into this `Copy` instance by the call site from its OWN
    /// frozen-weight gate (`!w.track_op()` => `false`, a tracked `Var`
    /// base => `true`; `is_variable()` alone is not a safe signal here,
    /// since a tracked-but-non-`Var` intermediate is neither definitely
    /// frozen nor definitely trainable — see the call site's own gate
    /// for the full three-way classification). This op never inspects
    /// `w`'s tracked-op state itself.
    pub dweight_needed: bool,
}

impl LowRankResidualLinear {
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
                "low_rank_residual_linear: scale must be finite, got {scale}"
            )));
        }
        if in_features == 0 || out_features == 0 || rank == 0 {
            return Err(Error::Msg(format!(
                "low_rank_residual_linear: in_features/out_features/rank must all be >= 1, got \
                 in_features={in_features} out_features={out_features} rank={rank}"
            )));
        }
        // `dropout.p` is validated HERE too, at construction, the same
        // way `scale` is — not only later, lazily, when `bwd`/`cpu_fwd`
        // happens to construct a `DropoutFused` from it (which would
        // itself catch a bad `p`, but only on the FIRST forward that
        // takes this instance, not at the point the caller actually made
        // the mistake).
        if let Some(key) = dropout {
            if !key.p.is_finite() || !(0.0..1.0).contains(&key.p) {
                return Err(Error::Msg(format!(
                    "low_rank_residual_linear: dropout.p must be finite and in [0.0, 1.0), got {}",
                    key.p
                )));
            }
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
    /// called from `crate::cuda::low_rank_residual_linear::cuda_fwd`, so the domain
    /// check has exactly one definition, not one per device.
    pub(crate) fn flatten_x(&self, l1: &Layout) -> Result<usize> {
        let dims = l1.dims();
        if dims.len() != 2 && dims.len() != 3 {
            return Err(Error::Msg(format!(
                "low_rank_residual_linear: x must be rank 2 or 3, got rank {}",
                dims.len()
            )));
        }
        let last = dims[dims.len() - 1];
        if last != self.in_features {
            return Err(Error::Msg(format!(
                "low_rank_residual_linear: x's last dim {last} != in_features {}",
                self.in_features
            )));
        }
        Ok(dims[..dims.len() - 1].iter().product())
    }

    /// `w` must be exactly `[out_features, in_features]`; `ab` must be
    /// exactly `[in_features + out_features, rank]` (the row-packed
    /// layout — see the module doc's "packed-`ab` GEMM eligibility
    /// problem" section) and `F32`. Both
    /// checked structurally regardless of what the call site's own
    /// admission predicate already verified (family D: an op trusts no
    /// caller for its own domain — the same doctrine `DropoutFused::new`
    /// documents). `pub(crate)`: shared with `crate::cuda::low_rank_residual_linear`
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
                "low_rank_residual_linear: w must be [{}, {}], got {:?}",
                self.out_features,
                self.in_features,
                l2.dims()
            )));
        }
        if ab_dims != [self.in_features + self.out_features, self.rank] {
            return Err(Error::Msg(format!(
                "low_rank_residual_linear: ab must be [{}, {}], got {:?}",
                self.in_features + self.out_features,
                self.rank,
                ab_dims
            )));
        }
        if ab_dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(
                ab_dtype,
                "low_rank_residual_linear",
            ));
        }
        Ok(())
    }

    /// `x`'s leading dims (everything but the last) followed by
    /// `out_features` — the final tensor shape this op returns,
    /// independent of the flat `(rows, out_features)` shape every
    /// internal GEMM actually operates over. `pub(crate)`: shared with
    /// `crate::cuda::low_rank_residual_linear::cuda_fwd`.
    pub(crate) fn output_shape(&self, l1: &Layout) -> Shape {
        let mut dims = l1.dims().to_vec();
        *dims
            .last_mut()
            .expect("flatten_x already checked rank >= 2") = self.out_features;
        Shape::from(dims)
    }
}

/// If `l` is already contiguous, returns `None` (no copy — the common
/// case). Otherwise materializes a dense, offset-0, contiguous copy of `s`
/// respecting `l`'s real (possibly transposed/narrowed) layout: a
/// same-dtype `to_dtype` call is a pure layout-aware gather-copy, not a
/// numeric operation (`BackendStorage::to_dtype` walks the given `Layout`
/// regardless of dtype-pair; see `s3.to_dtype` calls elsewhere in this
/// crate for the same idiom applied to `ab`'s narrowed slices before this
/// op's row-packed layout made those specific copies unnecessary) — so the
/// materialized result is bit-exact relative to what a from-scratch
/// contiguous `x` would have produced.
///
/// Only THIS op's `x` argument (the GEMM *lhs*) ever calls this: `w`/`ab`
/// stay a hard domain refusal (see the module doc's domain section) —
/// `w` is a `VarBuilder`-loaded weight and `ab` is freshly built by the
/// call site every forward, so both are contiguous by construction and a
/// caller reaching a non-contiguous one has a real bug worth surfacing
/// loudly. `x`, in contrast, can legitimately be a narrowed OR transposed
/// view of an activation the caller does not fully control (e.g. an
/// upstream batch-major reshape) — refusing it outright would push a
/// common, numerically harmless shape back to the eager fallback for no
/// reason beyond this op's own convenience.
pub(crate) fn materialize_contiguous_if_needed<S: BackendStorage>(
    s: &S,
    l: &Layout,
) -> Result<Option<(S, Layout)>> {
    if l.contiguous_offsets().is_some() {
        return Ok(None);
    }
    let dtype = s.dtype();
    let owned = s.to_dtype(l, dtype)?;
    Ok(Some((owned, Layout::contiguous(l.shape().clone()))))
}

impl super::sealed::Sealed for LowRankResidualLinear {}

impl CustomOp3 for LowRankResidualLinear {
    fn name(&self) -> &'static str {
        "low_rank_residual_linear"
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
        for (l, what) in [(l2, "w"), (l3, "ab")] {
            if l.contiguous_offsets().is_none() {
                return Err(Error::RequiresContiguous {
                    op: match what {
                        "w" => "low_rank_residual_linear(w)",
                        _ => "low_rank_residual_linear(ab)",
                    },
                });
            }
        }

        // `x` may be a non-contiguous (e.g. transposed/narrowed) view;
        // materialize it at THIS op's storage level rather than refusing —
        // see `materialize_contiguous_if_needed`'s doc for why only `x`
        // (never `w`/`ab`) gets this treatment.
        let x_owned = materialize_contiguous_if_needed(s1, l1)?;
        let (s1, l1): (&CpuStorage, Layout) = match &x_owned {
            Some((owned, contig_l)) => (owned, contig_l.clone()),
            None => (s1, l1.clone()),
        };
        let l1 = &l1;

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

        // `ab`'s row-packed layout: the first `inf` rows ARE `A^T`
        // (`[in, rank]`) and the remaining `outf` rows ARE `B` (`[out,
        // rank]`) — both dim-0 narrows of a contiguous matrix, hence
        // themselves contiguous with NO copy (see the module doc's
        // "packed-`ab` GEMM eligibility problem" section).
        let a_t_l = l3.narrow(0, 0, inf)?;
        let b_l = l3.narrow(0, inf, outf)?;
        let b_t_l = b_l.transpose(0, 1)?;

        // Step 4: h = xd @ A^T — `a_t_l` used directly, zero-copy.
        let h_storage = xd_storage.matmul(s3, (1, m, r, inf), &xd_l, &a_t_l)?;
        let h_l = Layout::contiguous((m, r));

        // Step 5: delta = h @ B^T — `b_t_l` used directly, zero-copy.
        let delta_storage = h_storage.matmul(s3, (1, m, outf, r), &h_l, &b_t_l)?;
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
        crate::cuda::low_rank_residual_linear::cuda_fwd(self, s1, l1, s2, l2, s3, l3)
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
        // This op trusts no caller for its own domain (the same doctrine
        // `check_w_and_ab`'s doc states): `dweight_needed` is CALL-SITE
        // data, frozen into this `Copy` instance at construction, not
        // re-derived here — but `w.is_variable() && !self.dweight_needed`
        // is a self-contradictory combination (a trainable base weight
        // whose gradient this op was told to DROP) that would silently
        // starve `w` of its own gradient contribution forever, so it is
        // checked and refused rather than trusted.
        if !self.dweight_needed && w.is_variable() {
            return Err(Error::Msg(
                "low_rank_residual_linear: bwd called with dweight_needed=false but w is a \
                 trainable Var — the call site's frozen_weight_gate disagrees with the \
                 tensor it actually passed; refusing rather than silently dropping w's \
                 gradient"
                    .into(),
            ));
        }
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

        // `ab`'s row-packed layout: rows `[0, inf)` are `A^T` (`[in, r]`),
        // rows `[inf, inf+outf)` are `B` (`[out, r]`) — see the module
        // doc's "packed-`ab` GEMM eligibility problem" section for why
        // this orientation is what makes the FORWARD's slices copy-free;
        // the backward derivation below is the algebraic transpose of the
        // `dA`/`dB^T`-shaped version, matching `ab`'s own slots.
        let a_t = ab.narrow(0, 0, inf)?; // [in, r] == A^T
        let b = ab.narrow(0, inf, outf)?; // [out, r] == B

        let h_2d = xd_2d.matmul(&a_t)?; // [M, r]

        // dB slot (matches `b`'s own [out, r] orientation): d_lora^T @ h.
        let d_b = d_lora_2d.t()?.matmul(&h_2d)?; // [out, r]

        // g = dL/dh = d_lora @ B.
        let g = d_lora_2d.matmul(&b)?; // [M, r]

        // dA^T slot (matches `a_t`'s own [in, r] orientation): xd^T @ g.
        let d_a_t = xd_2d.t()?.matmul(&g)?; // [in, r]

        // d_xd = g @ A (A = a_t^T), then the SAME dropout key reapplied to
        // the gradient (DropoutFused::bwd's own contract: applying the
        // same Copy instance regenerates `mask * scale` identically).
        let d_xd = g.matmul(&a_t.t()?)?; // [M, in], F32
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

        // The MIRROR gate (family D — the same doctrine as the
        // `!dweight_needed && w.is_variable()` refusal above, but this
        // combination is wasteful rather than dangerous): `dweight_needed
        // == true` while `w` is not actually `track_op()`'d means the
        // call site's `frozen_weight_gate` disagrees with the tensor it
        // passed in the OTHER direction — `w` carries no `Op` and is not
        // a `Var`, so NOTHING downstream (`backprop.rs`'s `sorted_nodes`
        // walk never visits an untracked leaf, and there is no `Var` to
        // `grads.get`) will ever read a `dW` computed for it. Computing
        // the full `dy^T @ x` GEMM and allocating its output anyway would
        // be silently wasted work every backward pass — skipped here
        // rather than trusted, matching this op's usual "an op trusts no
        // caller for its own domain" doctrine.
        let dw = if self.dweight_needed && w.track_op() {
            let x_base_2d = x.reshape((m, inf))?;
            Some(dy_base_2d.t()?.matmul(&x_base_2d)?)
        } else {
            None
        };

        let d_ab = Tensor::cat(&[&d_a_t, &d_b], 0)?;

        Ok((Some(dx), dw, Some(d_ab)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    /// Small-integer-valued, deterministic fixture (`{-4, .., 4}`, no
    /// `rand` dependency): every product/partial-sum this crate's LoRA
    /// GEMMs form from these values stays a small exact integer in `f32`
    /// (well under the 24-bit mantissa's exact range at the tiny
    /// dimensions this module's own tests use), so a `f32` sum of these
    /// values is IDENTICAL regardless of summation order — the property
    /// that makes a `diff == 0.0` (bit-exact) assertion architecture-
    /// independent rather than an accident of one CPU's `gemm` kernel
    /// selection (see the module doc's "packed-`ab` GEMM eligibility
    /// problem" section: the fused op and a hand-rolled "manual"
    /// composition can legitimately hand `gemm` different stride
    /// patterns, and different stride patterns can select different
    /// internal blocking/summation orders — exact-integer inputs sidestep
    /// that entirely rather than merely hoping it does not manifest).
    fn exact_fixture(n: usize, phase: i64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = (i as i64 * 7 + phase * 13).rem_euclid(9);
                (v - 4) as f32
            })
            .collect()
    }

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

    /// Packs `a` (`[r, inf]`) and `b` (`[outf, r]`) into `ab`'s ROW-packed
    /// `[inf + outf, r]` layout: rows `[0, inf)` are `A^T`, rows `[inf,
    /// inf+outf)` are `B` AS-IS (no transpose of `b` needed — see the
    /// module doc's "packed-`ab` GEMM eligibility problem" section for why
    /// this orientation is THE pack layout).
    fn pack_ab(a: &[f32], inf: usize, b: &[f32], outf: usize, r: usize, device: &Device) -> Tensor {
        let mut a_t = vec![0.0f32; inf * r];
        for j in 0..r {
            for k in 0..inf {
                a_t[k * r + j] = a[j * inf + k];
            }
        }
        let a_t_tensor = Tensor::from_slice(&a_t, (inf, r), device).unwrap();
        let b_tensor = Tensor::from_slice(b, (outf, r), device).unwrap();
        Tensor::cat(&[&a_t_tensor, &b_tensor], 0).unwrap()
    }

    fn fused_forward(
        x: &Tensor,
        w: &Tensor,
        ab: &Tensor,
        op: LowRankResidualLinear,
    ) -> Result<Tensor> {
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

        let op = LowRankResidualLinear::new(scale, inf, outf, r, None, false).unwrap();
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

        let op = LowRankResidualLinear::new(scale, inf, outf, r, None, false).unwrap();
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
        // No dropout, F32 throughout, EXACT-INTEGER fixtures (see
        // `exact_fixture`'s doc): every internal GEMM sum stays an exact
        // integer regardless of which stride pattern `gemm` sees, so this
        // must be bit-exact on ANY architecture, not merely the one this
        // suite happens to run on.
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (4usize, 3usize, 5usize, 2usize);
        let scale = 2.0f32; // exact in binary: no epilogue rounding either.

        let x_v = exact_fixture(rows * inf, 1);
        let w_v = exact_fixture(outf * inf, 2);
        let a_v = exact_fixture(r * inf, 3);
        let b_v = exact_fixture(outf * r, 4);

        let x = Tensor::from_slice(&x_v, (rows, inf), &device).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a = Tensor::from_slice(&a_v, (r, inf), &device).unwrap();
        let b = Tensor::from_slice(&b_v, (outf, r), &device).unwrap();
        let ab = pack_ab(&a_v, inf, &b_v, outf, r, &device);

        let op = LowRankResidualLinear::new(scale, inf, outf, r, None, false).unwrap();
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
        // EXACT-INTEGER fixtures (see `exact_fixture`'s doc) AND `p = 0.5`
        // (so the inverted-dropout scale `1/(1-p) == 2.0` is itself exact
        // in binary): a kept element becomes `x * 2.0` (exact), a dropped
        // one becomes exactly `0.0` — the dropout step introduces no
        // rounding of its own either, so the WHOLE forward stays exact-
        // integer end to end and this bit-exact claim is architecture-
        // independent.
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (5usize, 4usize, 3usize, 2usize);
        let scale = 2.0f32;
        let p = 0.5f32;
        let key = DropoutKey {
            seed: 99,
            layer_id: 7,
            forward_idx: 2,
            p,
        };

        let x_v = exact_fixture(rows * inf, 5);
        let w_v = exact_fixture(outf * inf, 6);
        let a_v = exact_fixture(r * inf, 7);
        let b_v = exact_fixture(outf * r, 8);

        let x = Tensor::from_slice(&x_v, (rows, inf), &device).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a = Tensor::from_slice(&a_v, (r, inf), &device).unwrap();
        let b = Tensor::from_slice(&b_v, (outf, r), &device).unwrap();
        let ab = pack_ab(&a_v, inf, &b_v, outf, r, &device);

        let op = LowRankResidualLinear::new(scale, inf, outf, r, Some(key), false).unwrap();
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
        let a_t = a_var.as_tensor().t().unwrap();
        let ab = Tensor::cat(&[&a_t, b_var.as_tensor()], 0).unwrap();

        let op = LowRankResidualLinear::new(scale, inf, outf, r, None, false).unwrap();
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
        // `a_var`'s gradient is w.r.t. A itself (shape [r, in]), not A^T
        // — `Op::Transpose`'s backward already un-transposes it (the `.t()`
        // above was applied to `a_var`, not `b_var`, in THIS pack
        // orientation), so no manual index remapping is needed here.
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

    /// The dropout-arm counterpart to `gradcheck_cpu_f32_no_dropout`: closes
    /// a real coverage gap an adversarial mutation audit found —
    /// neither an unmasked `d_xd` in `bwd` nor skipping the dropout
    /// re-application in the recomputed `xd` moved the dropout-less
    /// gradcheck above, so this test is what actually exercises that
    /// branch's correctness. A Philox draw is a pure function of
    /// `(seed, layer_id, forward_idx, element_index, shape)` — NEVER of
    /// `x`'s own VALUES — so the mask is legitimately fixed across the
    /// finite-difference probe below: it is extracted ONCE (via
    /// `DropoutFused` applied to an all-ones tensor of the same shape) and
    /// held constant while `x`/`A`/`B` are perturbed.
    #[test]
    fn gradcheck_cpu_f32_with_dropout() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (3usize, 3usize, 4usize, 2usize);
        let scale = 1.1f32;
        let eps = 1e-3f32;
        let key = DropoutKey {
            seed: 42,
            layer_id: 5,
            forward_idx: 0,
            p: 0.4,
        };

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

        let dropout_op = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p).unwrap();
        let ones = Tensor::ones((rows, inf), DType::F32, &device).unwrap();
        let mask: Vec<f32> = super::super::apply1(&ones, dropout_op)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let loss = |x_v: &[f32], a_v: &[f32], b_v: &[f32]| -> f32 {
            let out =
                reference_forward(x_v, rows, inf, &w_v, outf, a_v, r, b_v, scale, Some(&mask));
            out.iter().zip(dy_v.iter()).map(|(&o, &g)| o * g).sum()
        };

        let x = Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
        let w = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a_var =
            Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
        let b_var =
            Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
        let a_t = a_var.as_tensor().t().unwrap();
        let ab = Tensor::cat(&[&a_t, b_var.as_tensor()], 0).unwrap();

        let op = LowRankResidualLinear::new(scale, inf, outf, r, Some(key), false).unwrap();
        let out = x.as_tensor().apply_op3(&w, &ab, op).unwrap();
        let dy = Tensor::from_slice(&dy_v, (rows, outf), &device).unwrap();
        let total = (&out * &dy).unwrap().sum_all().unwrap();
        let grads = total.backward().unwrap();

        let da_analytic: Vec<f32> = grads
            .get(&a_var)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
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
                "dx[{i}] with dropout: numeric {numeric} vs analytic {}",
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
                "dA[{i}] with dropout: numeric {numeric} vs analytic {}",
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
                "dB[{idx}] with dropout: numeric {numeric} vs analytic {}",
                db_analytic[idx]
            );
        }
    }

    #[test]
    fn dweight_needed_returns_some_dw_otherwise_none() {
        // `dweight_needed=true` needs a trainable `Var` `w` (the real
        // "also fine-tune the base" case); `dweight_needed=false` needs a
        // true frozen leaf (`w.is_variable() == false`) — `bwd`'s own
        // domain check (added alongside this test: "an op trusts no
        // caller for its own domain") now REFUSES the self-contradictory
        // `dweight_needed=false` + `w.is_variable()` combination, so the
        // two branches can no longer share one `w`.
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (3usize, 3usize, 4usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();

        // dweight_needed = true: w is a trainable Var, dW must be Some.
        let w_trainable =
            Var::from_tensor(&Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap()).unwrap();
        let op_true = LowRankResidualLinear::new(1.0, inf, outf, r, None, true).unwrap();
        let out_true = x.apply_op3(w_trainable.as_tensor(), &ab, op_true).unwrap();
        let grads_true = out_true.sum_all().unwrap().backward().unwrap();
        assert!(
            grads_true.get(w_trainable.as_tensor()).is_some(),
            "dweight_needed=true: dW slot must be Some"
        );

        // dweight_needed = false: w is a true frozen leaf, dW must be
        // None (and w never appears in GradStore at all — nothing to
        // fetch).
        let w_frozen = Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap();
        assert!(!w_frozen.is_variable() && !w_frozen.track_op());
        let op_false = LowRankResidualLinear::new(1.0, inf, outf, r, None, false).unwrap();
        let out_false = x.apply_op3(&w_frozen, &ab, op_false).unwrap();
        // A frozen leaf carries no gradient entry to inspect via `grads`
        // (it is never a `Var`); the real assertion is that this forward
        // and backward SUCCEED at all with `dweight_needed=false` and a
        // genuinely frozen `w` — the self-contradiction check added
        // alongside this test only refuses `!dweight_needed &&
        // w.is_variable()`, never a true frozen leaf.
        let _ = out_false.sum_all().unwrap().backward().unwrap();
    }

    /// The domain check `bwd` added alongside this test (family D: an op
    /// trusts no caller for its own domain, cited in `check_w_and_ab`'s
    /// own doc): `dweight_needed=false` combined with an ACTUALLY
    /// trainable `w` (a `Var`) is refused with a typed error rather than
    /// silently dropping `w`'s gradient forever — the call site's own
    /// frozen-weight gate is expected to keep this flag truthful, but
    /// this op does not simply trust it.
    #[test]
    fn dweight_needed_false_with_a_trainable_w_is_a_typed_refusal() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (3usize, 3usize, 4usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
        let w = Var::from_tensor(&Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap()).unwrap();
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();

        let op = LowRankResidualLinear::new(1.0, inf, outf, r, None, false).unwrap();
        let out = x.apply_op3(w.as_tensor(), &ab, op).unwrap();
        let err = out.sum_all().unwrap().backward().expect_err(
            "dweight_needed=false with a trainable w must be refused, not silently \
                          drop w's gradient",
        );
        assert!(matches!(err, Error::Msg(_)));
    }

    /// The MIRROR gate (round-2 audit finding A1): `dweight_needed=true`
    /// combined with a `w` that is NOT actually `track_op()`'d (a true
    /// frozen leaf) must skip computing `dW` entirely — no `dW` slot is
    /// ever returned for a `w` nothing downstream could read one for —
    /// rather than silently wasting a full `dy^T @ x` GEMM and its output
    /// allocation every backward pass. Calls `CustomOp3::bwd` DIRECTLY
    /// (bypassing `Tensor::backward`'s whole machinery) so this asserts
    /// the WORK ITSELF was skipped (`dw` is `None` INSIDE `bwd`'s own
    /// return), not merely that candle's `GradStore` later discards an
    /// entry nothing reads — a black-box `grads.get(&w).is_none()` check
    /// would pass identically whether or not the wasted GEMM ran.
    #[test]
    fn dweight_needed_true_with_an_untracked_w_skips_dw_without_wasted_work() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (3usize, 3usize, 4usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
        let w = Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap();
        assert!(
            !w.is_variable() && !w.track_op(),
            "w must be a true frozen leaf"
        );
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();

        let op = LowRankResidualLinear::new(1.0, inf, outf, r, None, true).unwrap();
        let res = x.apply_op3(&w, &ab, op).unwrap();
        let grad_res = Tensor::ones_like(&res).unwrap();
        let (_dx, dw, _dab) = op.bwd(&x, &w, &ab, &res, &grad_res).unwrap();
        assert!(
            dw.is_none(),
            "dweight_needed=true with an untracked w must skip the dW GEMM \
             entirely, not merely compute-then-discard it"
        );
    }

    #[test]
    fn rank2_pooled_head_shape_is_accepted() {
        // A rank-2 pooled classification-head-shaped call: rank-2 x, F32
        // base, small out_features.
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (4usize, 6usize, 2usize, 2usize);
        let x = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
        let w = Tensor::randn(0f32, 1.0, (outf, inf), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();

        let op = LowRankResidualLinear::new(1.0, inf, outf, r, None, false).unwrap();
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
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();
        let op = LowRankResidualLinear::new(1.0, 5, 3, 2, None, false).unwrap();
        assert!(x.apply_op3(&w, &ab, op).is_err());
    }

    #[test]
    fn mismatched_ab_shape_is_a_typed_refusal() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (4, 5), &device).unwrap();
        let w = Tensor::randn(0f32, 1.0, (3, 5), &device).unwrap();
        // Wrong shape (should be [in+out, r]) — the ab packing itself must
        // be [in+out, r], the row-packed layout (see the module doc).
        let bad_ab = Tensor::randn(0f32, 1.0, (2, 5), &device).unwrap();
        let op = LowRankResidualLinear::new(1.0, 5, 3, 2, None, false).unwrap();
        assert!(x.apply_op3(&w, &bad_ab, op).is_err());
    }

    #[test]
    fn non_finite_scale_is_a_typed_refusal() {
        assert!(LowRankResidualLinear::new(f32::NAN, 4, 4, 2, None, false).is_err());
        assert!(LowRankResidualLinear::new(f32::INFINITY, 4, 4, 2, None, false).is_err());
    }

    /// `dropout.p` is validated at CONSTRUCTION, the same way `scale` is
    /// — not only lazily, the first time `bwd`/`cpu_fwd` happens to build
    /// a `DropoutFused` from it.
    #[test]
    fn non_finite_or_out_of_range_dropout_p_is_a_typed_refusal_at_construction() {
        let key = |p: f32| {
            Some(DropoutKey {
                seed: 1,
                layer_id: 1,
                forward_idx: 1,
                p,
            })
        };
        assert!(LowRankResidualLinear::new(1.0, 4, 4, 2, key(f32::NAN), false).is_err());
        assert!(LowRankResidualLinear::new(1.0, 4, 4, 2, key(1.0), false).is_err());
        assert!(LowRankResidualLinear::new(1.0, 4, 4, 2, key(-0.1), false).is_err());
        assert!(LowRankResidualLinear::new(1.0, 4, 4, 2, key(0.5), false).is_ok());
    }

    #[test]
    fn zero_sized_dims_are_a_typed_refusal() {
        assert!(LowRankResidualLinear::new(1.0, 0, 4, 2, None, false).is_err());
        assert!(LowRankResidualLinear::new(1.0, 4, 0, 2, None, false).is_err());
        assert!(LowRankResidualLinear::new(1.0, 4, 4, 0, None, false).is_err());
    }

    /// Family D, revised: a non-contiguous `x` (e.g. a transposed view) is
    /// no longer refused — see `materialize_contiguous_if_needed`'s doc
    /// for why `x` (unlike `w`/`ab`) gets a documented internal copy
    /// instead of a hard refusal. Proven by bit-exact equality against the
    /// SAME `x` made contiguous by the CALLER first: both paths gather the
    /// identical values, so there is no rounding difference to tolerate.
    #[test]
    fn non_contiguous_x_is_materialized_not_refused() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (5, 4), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!x.is_contiguous());
        let w = Tensor::randn(0f32, 1.0, (3, 5), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (2, 5), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (3, 2), &device).unwrap();
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();

        let op = LowRankResidualLinear::new(1.0, 5, 3, 2, None, false).unwrap();
        let fused_noncontig = x.apply_op3(&w, &ab, op).unwrap();

        let op_ref = LowRankResidualLinear::new(1.0, 5, 3, 2, None, false).unwrap();
        let x_contig = x.contiguous().unwrap();
        let fused_contig = x_contig.apply_op3(&w, &ab, op_ref).unwrap();

        assert_eq!(
            fused_noncontig
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            fused_contig
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            "a non-contiguous x must be materialized internally to the SAME \
             values a caller-side .contiguous() would have produced"
        );
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
        let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();
        let op = LowRankResidualLinear::new(1.0, inf, outf, r, None, false).unwrap();
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
        // this) — pins that LowRankResidualLinear's OWN reconstruction of
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

    /// Mirrors `cuda_backend::gemm_config`'s per-operand admissibility rule
    /// (candle-core 0.11.0's `cuda_backend/mod.rs:1398-1422`) purely off a
    /// `Layout`'s shape/stride: an operand is admissible iff it is
    /// row-major contiguous (`CUBLAS_OP_N`) OR is a single transposed VIEW
    /// of a row-major contiguous matrix (`CUBLAS_OP_T`, stride `[1,
    /// leading_dim]`) — device-independent (the same `Layout` struct is
    /// what any backend's `matmul` call receives), so a CPU test using
    /// this predicate can still catch the CUDA-only `MatMulNonContiguous`
    /// class this op's own history hit (see the module doc's "packed-`ab`
    /// GEMM eligibility problem" section).
    fn is_gemm_operand_admissible(l: &Layout) -> bool {
        let dims = l.dims();
        let stride = l.stride();
        if dims.len() != 2 {
            return false;
        }
        let (p, q) = (dims[0], dims[1]);
        let (sp, sq) = (stride[0], stride[1]);
        // N: row-major contiguous.
        (sq == 1 && sp == q) ||
        // T: a transposed view of a contiguous [q, p] matrix.
        (sp == 1 && sq == p)
    }

    /// Every 2D operand `bwd` hands `Tensor::matmul` — forward's own
    /// slices were already proven admissible by construction (module
    /// doc); this test proves EVERY BACKWARD product is too, at `rank`
    /// 1/2/3 (this op's own domain boundary — `rank >= 1`) and at
    /// production width (a transformer-encoder-scale `in=1024`). Reconstructs
    /// `bwd`'s exact slicing sequence (`a_t = ab.narrow(0, 0, inf)`,
    /// `b = ab.narrow(0, inf, outf)`, and their single-transpose views)
    /// rather than calling the private `bwd` method directly, since the
    /// property under test is about the LAYOUT each intermediate carries
    /// going INTO a GEMM, not the gradient values themselves (those are
    /// covered by the gradcheck tests above).
    #[test]
    fn bwd_every_gemm_operand_is_admissible_at_boundary_and_production_ranks() {
        let device = Device::Cpu;
        for &(rows, inf, outf, r) in &[
            (3usize, 4usize, 5usize, 1usize),
            (3usize, 4usize, 5usize, 2usize),
            (3usize, 4usize, 5usize, 3usize),
            (256usize, 1024usize, 3072usize, 16usize),
        ] {
            let a = Tensor::randn(0f32, 1.0, (r, inf), &device).unwrap();
            let b = Tensor::randn(0f32, 1.0, (outf, r), &device).unwrap();
            let ab = Tensor::cat(&[&a.t().unwrap(), &b], 0).unwrap();
            let xd_2d = Tensor::randn(0f32, 1.0, (rows, inf), &device).unwrap();
            let d_lora_2d = Tensor::randn(0f32, 1.0, (rows, outf), &device).unwrap();

            // ab's own slices (mirrors bwd's `a_t`/`b`).
            let a_t = ab.narrow(0, 0, inf).unwrap();
            let b_slice = ab.narrow(0, inf, outf).unwrap();

            // h = xd @ A^T: lhs xd_2d, rhs a_t.
            assert!(
                is_gemm_operand_admissible(xd_2d.layout()),
                "h = xd @ A^T: lhs xd at rank={r}"
            );
            assert!(
                is_gemm_operand_admissible(a_t.layout()),
                "h = xd @ A^T: rhs a_t at rank={r}"
            );
            let h_2d = xd_2d.matmul(&a_t).unwrap();

            // dB = d_lora^T @ h: lhs d_lora_2d.t(), rhs h_2d.
            let d_lora_t = d_lora_2d.t().unwrap();
            assert!(
                is_gemm_operand_admissible(d_lora_t.layout()),
                "dB = d_lora^T @ h: lhs d_lora^T at rank={r}"
            );
            assert!(
                is_gemm_operand_admissible(h_2d.layout()),
                "dB = d_lora^T @ h: rhs h at rank={r}"
            );
            let _d_b = d_lora_t.matmul(&h_2d).unwrap();

            // g = d_lora @ B: lhs d_lora_2d, rhs b_slice.
            assert!(
                is_gemm_operand_admissible(d_lora_2d.layout()),
                "g = d_lora @ B: lhs d_lora at rank={r}"
            );
            assert!(
                is_gemm_operand_admissible(b_slice.layout()),
                "g = d_lora @ B: rhs b at rank={r}"
            );
            let g = d_lora_2d.matmul(&b_slice).unwrap();

            // dA^T = xd^T @ g: lhs xd_2d.t(), rhs g.
            let xd_t = xd_2d.t().unwrap();
            assert!(
                is_gemm_operand_admissible(xd_t.layout()),
                "dA^T = xd^T @ g: lhs xd^T at rank={r}"
            );
            assert!(
                is_gemm_operand_admissible(g.layout()),
                "dA^T = xd^T @ g: rhs g at rank={r}"
            );
            let _d_a_t = xd_t.matmul(&g).unwrap();

            // d_xd = g @ A (A = a_t^T): lhs g, rhs a_t.t().
            let a_t_t = a_t.t().unwrap();
            assert!(
                is_gemm_operand_admissible(g.layout()),
                "d_xd = g @ A: lhs g at rank={r}"
            );
            assert!(
                is_gemm_operand_admissible(a_t_t.layout()),
                "d_xd = g @ A: rhs A at rank={r}"
            );
            let _d_xd = g.matmul(&a_t_t).unwrap();
        }
    }

    /// MEMORY ORACLE (mirrors `jammi_encoders::modernbert`'s own
    /// `fused_training_softmax_call_site_drops_the_affine_node`):
    /// `Tensor::sorted_nodes()` is candle's own PUBLIC topological-sort-
    /// for-backward API (the exact list `Tensor::backward` walks, and the
    /// list `GradStore::or_insert` allocates a full-size `zeros_like` +
    /// `add` for) — a direct, honest count of what backward keeps
    /// resident, not a proxy. `x`/`A`/`B` are `Var`s (so they, and every
    /// TRACKED intermediate derived from them, appear in the walk); `w`
    /// is a PLAIN (non-`Var`, untracked) leaf, so it and any
    /// exclusively-`w`-derived intermediate (`w.t()`) contribute NOTHING
    /// to either count — matching this op's actual frozen-base contract.
    /// `F32`, no dropout.
    ///
    /// EAGER: the composition this op replaces, reproduced directly
    /// (`base = x@w^T`, `after_a = x@A^T`, `lora_out = after_a@B^T`,
    /// `scaled = lora_out*scale`, `out = base+scaled`) — every `.t()` on
    /// a TRACKED operand (`A`, `B`) is itself a distinct tracked node
    /// (`Op::Transpose`), not free.
    ///
    /// FUSED: `ab = cat([A.t(), B], 0)` then ONE `apply_op3` call.
    #[test]
    fn fused_site_retains_fewer_tape_nodes_than_the_eager_composition() {
        let device = Device::Cpu;
        let (rows, inf, outf, r) = (4usize, 5usize, 3usize, 2usize);
        let scale = 1.3f32;

        let x_v: Vec<f32> = (0..rows * inf).map(|i| ((i as f32) * 0.29).sin()).collect();
        let w_v: Vec<f32> = (0..outf * inf).map(|i| ((i as f32) * 0.19).cos()).collect();
        let a_v: Vec<f32> = (0..r * inf).map(|i| ((i as f32) * 0.37).sin()).collect();
        let b_v: Vec<f32> = (0..outf * r).map(|i| ((i as f32) * 0.41).cos()).collect();

        // EAGER.
        let x_eager =
            Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
        let w_plain = Tensor::from_slice(&w_v, (outf, inf), &device).unwrap();
        let a_eager =
            Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
        let b_eager =
            Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
        assert!(
            !w_plain.is_variable() && !w_plain.track_op(),
            "w must be a plain leaf"
        );

        let base_out = x_eager.as_tensor().matmul(&w_plain.t().unwrap()).unwrap();
        let after_a = x_eager
            .as_tensor()
            .matmul(&a_eager.as_tensor().t().unwrap())
            .unwrap();
        let lora_out = after_a.matmul(&b_eager.as_tensor().t().unwrap()).unwrap();
        let scaled = (&lora_out * f64::from(scale)).unwrap();
        let y_eager = (&base_out + &scaled).unwrap();
        let nodes_eager = y_eager.sorted_nodes().len();

        // FUSED.
        let x_fused =
            Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
        let a_fused =
            Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
        let b_fused =
            Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
        let ab_fused =
            Tensor::cat(&[&a_fused.as_tensor().t().unwrap(), b_fused.as_tensor()], 0).unwrap();
        let op = LowRankResidualLinear::new(scale, inf, outf, r, None, false).unwrap();
        let y_fused = x_fused
            .as_tensor()
            .apply_op3(&w_plain, &ab_fused, op)
            .unwrap();
        let nodes_fused = y_fused.sorted_nodes().len();

        assert!(
            nodes_fused < nodes_eager,
            "the fused site must retain FEWER tape nodes than the eager composition: \
             eager={nodes_eager} fused={nodes_fused}"
        );
        // Pin the MEASURED constants directly, not just "fewer than" (see
        // this test's own doc for the leaf-vs-tracked-intermediate
        // accounting these numbers follow from).
        assert_eq!(
            nodes_eager, 10,
            "measured EAGER node count: x, A, B (3 leaves) + A.t(), base_out, \
             after_a, B.t(), lora_out, scaled, out (7 tracked intermediates/output) \
             — w and w.t() contribute 0 (w is a plain, untracked leaf)"
        );
        assert_eq!(
            nodes_fused, 6,
            "measured FUSED node count: x, A, B (3 leaves) + A.t(), ab (Op::Cat), \
             out (CustomOp3) (3 tracked intermediates/output) — w contributes 0"
        );
    }
}
