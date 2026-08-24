//! Fused `CustomOp` implementations. `Axpy` is the proof op establishing the
//! pattern every later fused op (LayerNorm, RoPE, softmax, GeGLU) copies:
//! real CPU forward+backward and a feature-gated CUDA forward loaded from
//! build-time PTX.
//!
//! ## Every op in this module implements `KernelOp`
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

use candle_core::{CustomOp1, CustomOp2, CustomOp3, Result, Tensor};

mod axpy;
mod dropout;
// `pub(crate)`, not private like `axpy`/`layer_norm`/`rope`: `crate::cuda::geglu`
// imports `geglu_dims`/`output_shape`/`check_variant` directly from here
// rather than duplicating them — the same "shared, not duplicated" choice
// `ops::softmax` makes for its own (more complex) broadcast-domain logic.
pub(crate) mod geglu;
mod layer_norm;
mod rope;
mod scaled_cast_add;
// `pub(crate)`, not private like the three above: `crate::cuda::softmax`
// imports `softmax_dims` directly from here rather than duplicating it (a
// deliberate, disclosed departure from `layer_norm`'s/`rope`'s per-file
// duplication precedent — see `ops::softmax`'s module doc for why this
// op's broadcast-domain logic is complex enough that a second,
// independently-maintained copy is a real divergence risk, not just
// boilerplate).
pub(crate) mod softmax;

pub use axpy::Axpy;
pub use dropout::{DropoutFused, PhiloxKatProbe};
pub use geglu::{GegluFused, GeluVariant};
pub use layer_norm::{LayerNormFused, MAX_HIDDEN};
pub use rope::{RopeFused, MAX_HEAD_DIM};
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
