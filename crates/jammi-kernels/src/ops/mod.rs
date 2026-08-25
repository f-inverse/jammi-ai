//! Fused `CustomOp` implementations. `Axpy` is the proof op establishing the
//! pattern every later fused op (LayerNorm, RoPE, softmax, GeGLU) copies:
//! real CPU forward+backward and a feature-gated CUDA forward loaded from
//! build-time PTX.
//!
//! **Statelessness is TOTAL for the `applyN` family** ([`apply1`],
//! [`apply2`], [`apply3`]): every op reachable through them is [`KernelOp`]
//! (`Copy`-bounded — see below), so NONE of them may carry a `Saved` field.
//! An op that genuinely needs one (P6 Stage B's FlashAttention-2 varlen op,
//! `crate::ops::flash_attention`) runs ONLY through [`apply_stateful1`]/
//! [`apply_stateful3`], which require [`StatefulKernelOp`] instead — a
//! SEPARATE sealed trait with no `Copy`/`Clone` bound (see its own doc).
//! The two families are mutually exclusive by construction: `KernelOp`
//! requires `Copy`, `StatefulKernelOp`'s whole point is that a type holding
//! interior-mutable state (`Saved<T>`) CANNOT be `Copy`, so no type
//! satisfies both, and there is exactly one `applyN`/`apply_statefulN`
//! entry point per arity a real op ever needs.
//!
//! ## Every STATELESS op in this module implements `KernelOp`
//!
//! `CustomOp2::apply_op2` takes the op BY VALUE, so a fresh instance backs
//! every call regardless of the type's own properties — that alone does
//! NOT prove an op is stateless (an earlier version of this crate shipped
//! an "interleaving oracle" meant to catch a save-state-in-the-op-struct
//! bug by running two forwards before either backward; it could never
//! actually fail, because `apply_op2` consuming the op by value means each
//! call already gets its own independent instance no matter what the type
//! looks like — the oracle was deleted rather than kept as a fake proof).
//!
//! The REAL guarantee is structural: every op here is required to
//! implement [`KernelOp`], a SEALED (see the private `sealed` module —
//! unnameable outside this crate, so a downstream crate cannot implement
//! `KernelOp` for its own type) supertrait of `CustomOp2 + Copy + Send +
//! Sync`. This is TOTAL, not per-op opt-in: [`apply2`] — the sanctioned way
//! to run a binary op — requires `T: KernelOp`, so a new op added to this
//! module without also implementing the (crate-private) `Sealed` marker
//! for it fails to COMPILE the moment anything tries to run it through
//! `apply2`, rather than silently shipping unconstrained because an author
//! forgot to add a separate assertion line. This matters because this is
//! the template C2-C7 (LayerNorm, RoPE, softmax, GeGLU, LoRA-site cleanup,
//! device-side dropout) copy, and each of those ops is under real pressure
//! to want a cache (LayerNorm's bwd recomputes mean/invvar from `x`
//! specifically because there is nowhere stateful to stash them from
//! `fwd`).
//!
//! ### What `Copy` (`+ Send + Sync`) proves, and what it does not
//!
//! - PROVES: no OWNED interior-mutable or heap-allocated field anywhere in
//!   the type — `Cell` / `RefCell` / `Mutex` / an atomic / `Box` / `Vec` /
//!   ... are all `!Copy`. That is the exact shape a "cache something in
//!   `fwd` for `bwd` to read later" design needs for PER-INSTANCE state,
//!   and `Copy` forbids it at the type level.
//! - Does NOT prove the op's methods touch no shared mutable state at
//!   all: a module-level `static`, or a `&'static Mutex<T>` FIELD, is
//!   itself `Copy + Send + Sync` (a reference is a plain pointer-sized
//!   value, regardless of what it points to). That class of statefulness
//!   is a REVIEW concern this bound cannot catch at compile time — stated
//!   here explicitly rather than left as an overclaim.

use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, Error, Layout, Result, Shape, Tensor,
};

// `pub(crate)`, not private like `axpy`/`layer_norm`/`rope`: `crate::cuda::attention_block`
// imports `attention_dims`/`check_mask`/`check_rope_pack` directly from here
// (the SAME domain checks the CPU arm applies), mirroring `ops::softmax`'s
// identical `pub(crate)` rationale.
pub(crate) mod attention_block;
mod axpy;
mod dropout;
// `flash-attn`-gated: `crate::flash` (the FFI boundary) exists only under
// that feature, so the op composing it can only exist under it too — see
// this submodule's own doc for why it is a `StatefulKernelOp`, not a
// `KernelOp`.
#[cfg(feature = "flash-attn")]
pub(crate) mod flash_attention;
// `pub(crate)`, not private like `axpy`/`scaled_cast_add`: each op's CUDA
// glue (`crate::cuda::geglu`/`layer_norm`/`rope`/`softmax`) imports its
// shared dims/domain helper (`geglu_dims`/`output_shape`/`check_variant`,
// `hidden_of`, `rope_dims`, `softmax_dims`) directly from here rather than
// carrying an independently-maintained CUDA-side copy — one definition per
// op's domain check, not two that could silently drift apart.
pub(crate) mod geglu;
pub(crate) mod layer_norm;
pub(crate) mod low_rank_residual_linear;
pub(crate) mod rope;
mod saved;
mod scaled_cast_add;
pub(crate) mod softmax;

pub use attention_block::AttentionBlockFused;
/// Test/introspection-only (P3 fix round 4, deliverable 3's "mechanism
/// pin" — see `bwd_gradient_gemm_layouts`'s own doc): `#[doc(hidden)]`
/// re-exports so `tests/cuda_parity.rs` can capture `bwd`'s own gradient-
/// GEMM operand `Layout`s without depending on `CustomOp3::bwd`'s private
/// trait-method signature. `matmul_grad_lhs`/`matmul_grad_rhs` are the
/// shared `Op::Matmul`-backward definition `bwd` itself calls — see their
/// own doc for the shared-definition rationale.
#[doc(hidden)]
pub use attention_block::{
    bwd_gradient_gemm_layouts, matmul_grad_lhs, matmul_grad_rhs, BwdGemmLayoutsParams,
};
/// Re-exported under this name (rather than `ops::attention_block::HEAD_DIM`
/// directly) so a call site's admission predicate reads `ATTENTION_BLOCK_HEAD_DIM`
/// without a `attention_block::` path segment, mirroring `MAX_HEAD_DIM`/`MAX_LAST_DIM`'s
/// own flat re-export shape.
pub const ATTENTION_BLOCK_HEAD_DIM: usize = attention_block::HEAD_DIM;
/// See [`ATTENTION_BLOCK_HEAD_DIM`]'s doc for why this is a real `const`
/// definition here rather than a `pub use ... as` rename.
pub const ATTENTION_BLOCK_MAX_SEQ: usize = attention_block::MAX_SEQ;
/// See [`ATTENTION_BLOCK_HEAD_DIM`]'s doc for the re-export shape. A call
/// site combining its own padding mask with a sliding-window band before
/// calling [`AttentionBlockFused`] (the op has no `window` construction
/// data of its own — see that op's module doc) needs this SAME sentinel
/// so the combined mask's out-of-window contribution matches what
/// [`AttentionBlockFused`]'s own `< 0.0` fully-masked-row rule expects;
/// pinned by value, not merely by sign, so a caller can assert the two
/// crates agree exactly (family F: a measured, asserted equality, not an
/// assumed one).
pub const ATTENTION_BLOCK_WINDOW_MASKED_VALUE: f32 = attention_block::WINDOW_MASKED_VALUE;
pub use axpy::Axpy;
pub use dropout::{DropoutFused, PhiloxKatProbe};
#[cfg(feature = "flash-attn")]
pub use flash_attention::flash_attention_varlen;
pub use geglu::{GegluFused, GeluVariant};
pub use layer_norm::{LayerNormFused, MAX_HIDDEN};
pub use low_rank_residual_linear::{DropoutKey, LowRankResidualLinear};
pub use rope::{RopeFused, MAX_HEAD_DIM};
pub use saved::{Saved, SavedError};
pub use scaled_cast_add::ScaledCastAdd;
pub use softmax::{
    mask_broadcast_class_holds, FullyMaskedPolicy, SoftmaxLastDimFused, MAX_LAST_DIM, MAX_RANK,
};

mod sealed {
    //! Not `pub` at the `ops` level, so `Sealed` is unreachable outside
    //! this crate (and outside `ops` and its submodules, even within this
    //! crate) — the sealing boundary for [`super::KernelOp`].
    pub trait Sealed {}
}

/// Every fused op this crate exports must implement this. See the module
/// doc for what the `Copy` bound does and does not guarantee, and why the
/// enforcement is total (via [`apply2`]/[`apply3`]) rather than a per-op
/// opt-in assertion. `'static` matches `CustomOp2`/`CustomOp3`'s own bound
/// on `apply_op2`/`apply_op3` (an op carries no borrowed data — every
/// field is owned, plain construction data like `Axpy::alpha` or
/// `LayerNormFused::eps`).
///
/// Deliberately NOT bounded by `CustomOp2` (or any specific arity) here:
/// `LayerNormFused`'s backward recomputes `dx` through an internal
/// `CustomOp3` helper (`x`, `gamma`, `grad_output` — three tensor
/// arguments, since candle 0.11 has no save-for-backward channel to stash
/// the recomputed mean/invvar in), so the arity-agnostic statelessness
/// guarantee (`Copy + Send + Sync + Sealed`) lives here, and each
/// arity-specific `applyN` function below adds its own `CustomOpN` bound.
pub trait KernelOp: Copy + Send + Sync + 'static + sealed::Sealed {}

impl<T> KernelOp for T where T: Copy + Send + Sync + 'static + sealed::Sealed {}

/// The sanctioned way to run a unary (`CustomOp1`) fused op — the same
/// enforcement point as [`apply2`]/[`apply3`], for an op that takes
/// exactly one tensor argument (e.g. [`GegluFused`], whose split happens
/// INSIDE the kernel rather than at the call site).
pub fn apply1<T: KernelOp + CustomOp1>(x: &Tensor, op: T) -> Result<Tensor> {
    x.apply_op1(op)
}

/// The sanctioned way to run a binary (`CustomOp2`) fused op: requires
/// `T: KernelOp + CustomOp2`, so this is the compile-time enforcement
/// point every call site (present and future) goes through.
pub fn apply2<T: KernelOp + CustomOp2>(x: &Tensor, y: &Tensor, op: T) -> Result<Tensor> {
    x.apply_op2(y, op)
}

/// The sanctioned way to run a ternary (`CustomOp3`) fused op — the same
/// enforcement point as [`apply2`], for ops that take three tensor
/// arguments (e.g. `LayerNormFused`'s internal backward-dx kernel: `x`,
/// `gamma`, `grad_output`).
pub fn apply3<T: KernelOp + CustomOp3>(
    x: &Tensor,
    y: &Tensor,
    z: &Tensor,
    op: T,
) -> Result<Tensor> {
    x.apply_op3(y, z, op)
}

/// A SECOND, DISJOINT op family from [`KernelOp`]: ops that legitimately
/// carry per-instance interior-mutable state (a [`Saved`] field) between
/// their own `fwd` and their own `bwd`. `KernelOp`'s `Copy` bound exists
/// PRECISELY to forbid that shape (see this module's top doc, "What
/// `Copy` … proves, and what it does not") — so a stateful op cannot and
/// must not implement `KernelOp`; it implements this trait instead.
///
/// Bound: `Send + Sync + 'static + Sealed`, deliberately WITHOUT `Copy`
/// **or `Clone`**. This is not merely "we don't need Clone" — Clone is
/// actively refused:
///
/// - Every call site in this crate constructs a fresh instance and passes
///   it BY VALUE into [`apply_stateful1`] (mirroring [`apply1`]/[`apply2`]/
///   [`apply3`]'s own by-value shape and every existing op's inline
///   `::new()`-at-the-call-site convention — e.g. `AttentionBlockFused::new`
///   at `crates/jammi-encoders/src/modernbert.rs:960`, `DropoutFused::new`
///   at `crates/jammi-lora/src/lora_linear.rs:690`); nothing in this crate
///   ever clones an op value, stateful or not.
/// - If a stateful op were `Clone`, a caller could hold one instance in a
///   struct field (`struct Layer { op: FlashVarlenAttention }`) and reuse
///   a CLONE of it across multiple calls — reintroducing exactly the
///   hazard `Saved`'s per-call freshness argument depends on (see
///   `saved`'s module doc): a cloned `Saved<T>` shares its `Arc` with the
///   original, so `set()` on a REUSED clone would fail with
///   `SavedError::AlreadySet` (harmless — a typed error, not silent
///   corruption) but only because `Saved` itself refuses to be `Clone`
///   either; the real defence is one level up, at THIS bound, making the
///   reuse impossible to WRITE in the first place: without `Copy` or
///   `Clone`, `op: FlashVarlenAttention` cannot be moved out of `&self` in
///   a method (`cannot move out of `self.op` which is behind a shared
///   reference` — a compile error, not a runtime one), so a struct field
///   holding one can never be handed to [`apply_stateful1`] (which takes
///   the op BY VALUE) more than once. Hoisting a stateful op into a
///   long-lived field is therefore unrepresentable at compile time, not
///   merely discouraged by convention.
///
/// `candle_core::Tensor::apply_op1_arc`/`apply_op3_arc` (`custom_op.rs:
/// 216-234,236-243`) are PUBLIC candle APIs that take an already-
/// constructed `Arc<Box<dyn CustomOpN + Send + Sync>>` and can back
/// arbitrarily many forward calls from ONE such `Arc` — nothing in
/// `CustomOpN`'s own trait bounds stops that. This module's enforcement is
/// therefore a DISCIPLINE over this crate's own call sites (never call
/// `apply_opN_arc`/`apply_opN_no_bwd` directly on a `Saved`-bearing op),
/// not a type-system guarantee candle itself enforces — see
/// `crates/jammi-kernels/tests/stateful_op_discipline.rs`'s grep-based
/// regression test for the mechanical check that no call site in this
/// crate's `src/` does.
///
/// The doctest below proves the move-out-of-`&self` closure using
/// [`Saved`] directly (a PUBLIC type) rather than
/// `crate::ops::flash_attention::FlashVarlenAttention` itself: that op
/// type is `pub(crate)` (crate-private by construction, see its own
/// module doc), and doctests compile as an EXTERNAL crate regardless of
/// which item's doc comment they are attached to — a `pub(crate)` path is
/// UNREACHABLE from one (verified: an earlier draft of this doctest tried
/// `jammi_kernels::ops::flash_attention::FlashVarlenAttention` directly
/// and failed with `E0603 module is private`, a DIFFERENT and weaker
/// error than the `E0507 cannot move out of ... behind a shared reference`
/// this doctest exists to prove). The STRUCTURAL reason
/// `FlashVarlenAttention` cannot be hoisted is exactly the shape below —
/// it holds a `Saved<T>` field and derives neither `Copy` nor `Clone` —
/// so this generic minimal reproduction is a faithful proof of the
/// mechanism, not a proof about `FlashVarlenAttention` by name specifically:
///
/// ```compile_fail,E0507
/// use jammi_kernels::ops::Saved;
///
/// struct HoldsSaved {
///     lse: Saved<u32>,
/// }
///
/// struct Layer {
///     op: HoldsSaved,
/// }
///
/// impl Layer {
///     // The GradCache-detached-pass-1 hazard's shape: a method taking
///     // `&self` trying to hand the hoisted op to a by-value consumer
///     // (`apply_stateful1`'s own signature) a SECOND time on a later
///     // call. `self.op` cannot be moved out of `&self` because
///     // `HoldsSaved` is neither `Copy` nor `Clone` (it holds a `Saved<T>`,
///     // and `Saved<T>` itself deliberately implements neither).
///     fn take_op(&self) -> HoldsSaved {
///         self.op
///     }
/// }
/// ```
pub trait StatefulKernelOp: Send + Sync + 'static + sealed::Sealed {}

impl<T> StatefulKernelOp for T where T: Send + Sync + 'static + sealed::Sealed {}

/// The sanctioned way to run a unary (`CustomOp1`) STATEFUL fused op —
/// same shape as [`apply1`], but for [`StatefulKernelOp`] (no `Copy`
/// bound). `crate::ops::flash_attention::FlashVarlenAttention`'s FORWARD
/// takes exactly one differentiable `Tensor` argument (the packed `qkv`)
/// — mirroring [`crate::ops::AttentionBlockFused`]'s own precedent of
/// taking `qkv` as ONE `CustomOp3` argument and splitting Q/K/V INTERNALLY
/// (`crate::cuda::attention_block::slot_view`), not as three top-level
/// tensor arguments — so `CustomOp1` is its honest arity. Its BACKWARD
/// needs a different, THREE-tensor seam; see [`apply_stateful3`].
pub fn apply_stateful1<T: StatefulKernelOp + CustomOp1>(x: &Tensor, op: T) -> Result<Tensor> {
    x.apply_op1(op)
}

/// The sanctioned way to run a ternary (`CustomOp3`) STATEFUL fused op.
/// `crate::ops::flash_attention::FlashVarlenAttention::bwd` uses this: the
/// vendored backward genuinely needs THREE tensors (`qkv`, `o`, `d_o` —
/// candle already hands `bwd` exactly these three, `res` being `o`) PLUS
/// the forward's stashed `lse`, and constructing the returned `d_qkv`
/// `Tensor` through candle's own `apply_op3` (rather than hand-building a
/// `Tensor` from raw `Storage`) is the SAME "recompute via composing an
/// inner op" idiom `LayerNormFused::bwd`'s internal `CustomOp3` dx/dgamma
/// helpers already use — `lse` moves from the outer op's `Saved` slot to
/// the inner helper's OWN `Saved` slot at construction (`take()` then
/// `set()`, both typed, both proven — see `ops::saved`'s module doc): a
/// `Saved<T>` SPANS two op types here, never a raw field smuggled between
/// them.
pub fn apply_stateful3<T: StatefulKernelOp + CustomOp3>(
    x: &Tensor,
    y: &Tensor,
    z: &Tensor,
    op: T,
) -> Result<Tensor> {
    x.apply_op3(y, z, op)
}

/// The "output dtype must match `s1`, and equal `s2`" degenerate-input
/// empty-storage builder shared by [`layer_norm`] (`hidden == 0`),
/// [`softmax`] (`last == 0`), and [`rope`] (`hidden == 0`) — all bail to an
/// empty CPU output of `l1`'s own shape (preserving every OTHER dimension,
/// unlike a bare `[0]`) once their reduction axis itself is zero-length.
/// `pub(crate)`: byte-identical across all three files before this, now
/// one definition. `rope`'s pre-existing local copy returned a bare
/// `Shape::from(0)` instead of `l1.shape().clone()` and took only ONE
/// storage argument — checked and confirmed NOT load-bearing (its call
/// site discarded that returned shape and substituted `l1.shape().clone()`
/// itself, and by the time it ran `s1`'s dtype was already known to equal
/// `s2`/`s3`'s), so `rope::cpu_fwd` now calls this directly with `(s1, s1)`
/// rather than keeping a second, narrower copy.
pub(crate) fn empty_like(
    s1: &CpuStorage,
    s2: &CpuStorage,
    l1: &Layout,
    op: &'static str,
) -> Result<(CpuStorage, Shape)> {
    match (s1, s2) {
        (CpuStorage::F32(_), CpuStorage::F32(_)) => {
            Ok((CpuStorage::F32(Vec::new()), l1.shape().clone()))
        }
        (CpuStorage::BF16(_), CpuStorage::BF16(_)) => {
            Ok((CpuStorage::BF16(Vec::new()), l1.shape().clone()))
        }
        (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s2.dtype(),
            op,
        }),
        (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), op)),
    }
}
