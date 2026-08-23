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

use candle_core::{CustomOp2, Result, Tensor};

mod axpy;

pub use axpy::Axpy;

mod sealed {
    //! Not `pub` at the `ops` level, so `Sealed` is unreachable outside
    //! this crate (and outside `ops` and its submodules, even within this
    //! crate) — the sealing boundary for [`super::KernelOp`].
    pub trait Sealed {}
}

/// Every fused op this crate exports must implement this. See the module
/// doc for what the `Copy` bound does and does not guarantee, and why the
/// enforcement is total (via [`apply2`]) rather than a per-op opt-in
/// assertion. `'static` matches `CustomOp2`'s own bound on `apply_op2`
/// (an op carries no borrowed data — every field is owned, plain
/// construction data like `Axpy::alpha`).
pub trait KernelOp: CustomOp2 + Copy + Send + Sync + 'static + sealed::Sealed {}

impl<T> KernelOp for T where T: CustomOp2 + Copy + Send + Sync + 'static + sealed::Sealed {}

/// The sanctioned way to run a binary (`CustomOp2`) fused op: requires
/// `T: KernelOp`, so this is the compile-time enforcement point every
/// call site (present and future) goes through. A future op of a
/// different arity (`CustomOp1`/`CustomOp3`) gets its own `apply1`/
/// `apply3` following the same pattern.
pub fn apply2<T: KernelOp>(x: &Tensor, y: &Tensor, op: T) -> Result<Tensor> {
    x.apply_op2(y, op)
}
