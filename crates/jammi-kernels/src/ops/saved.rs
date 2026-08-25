//! `Saved<T>`: a write-once-read-once slot for state a [`StatefulKernelOp`]'s
//! forward computes and its own backward alone consumes — the channel
//! `candle_core`'s `CustomOp1::bwd`/`CustomOp2::bwd`/`CustomOp3::bwd`
//! (`candle-core` 0.11.0 `custom_op.rs:34-38,79-86,142-151`) does not
//! provide: `bwd` receives only `(arg…, res, grad_res)`, and `res` is the
//! op's TENSOR output — there is nowhere to carry an auxiliary value of a
//! different shape/dtype (FlashAttention-2's `lse`, `[H, total_q]` f32,
//! alongside a bf16 `o`) from `fwd` to `bwd` through that signature alone.
//!
//! [`StatefulKernelOp`]: super::StatefulKernelOp
//!
//! # Why this is safe (not just convenient)
//!
//! `Tensor::apply_op3` (`custom_op.rs:236-243` in `candle-core` 0.11.0)
//! allocates a FRESH `Arc<Box<dyn CustomOp3 + Send + Sync>>` on every call
//! (`Arc::new(Box::new(c))`) and stores that `Arc` (cloned, never
//! recreated) into the returned tensor's `Op::CustomOp3(.., c)`. A
//! `Saved<T>` FIELD inside an op struct `c` is therefore scoped to exactly
//! the one forward call that constructed `c` — a DIFFERENT forward call
//! constructs a DIFFERENT `c` with its own, independent `Saved<T>`. The
//! remaining hazard this type itself must close is *within* that scope:
//! could the SAME node's `bwd` run twice (double backward), or could
//! `bwd` run before `fwd` ever set the slot (a malformed graph, or a
//! caller who calls `.backward()` on a tensor produced by ONLY `fwd`, e.g.
//! GradCache's detached pass 1 — `crates/jammi-ai/src/fine_tune/
//! gradcache.rs:78-81` — which forwards without ever differentiating that
//! output)? Both are answered by the ONE typed error each direction:
//! `set()` errors instead of silently overwriting if the slot already
//! holds a value; `take()` errors instead of returning a stale/default
//! value if the slot is empty (never set, or already taken by an earlier
//! `bwd`). Neither direction ever panics or returns a silently-wrong `T`.
//!
//! `Arc<Mutex<Option<T>>>` rather than `RefCell<Option<T>>`: `StatefulKernelOp`
//! requires `Send + Sync` (mirroring `KernelOp`'s own bound, and
//! `CustomOp3 + Send + Sync` is `Tensor::apply_op3`'s own requirement) —
//! `Mutex` is the `Sync` cell; the `Arc` is needed because `set`/`take`
//! take `&self` (candle's `CustomOp3::bwd` signature is `&self`, never
//! `&mut self` — there is no exclusive borrow to hand a plain `Cell`-style
//! type here) and `Saved<T>` must therefore use interior mutability
//! reachable through a shared reference, same as any other lock-based
//! shared-state primitive.

use std::sync::{Arc, Mutex};

use thiserror::Error;

/// The two ways a [`Saved`] slot can be misused — both mean the calling
/// code violated the one-fwd-then-at-most-one-bwd contract, not that
/// anything in this type itself went wrong.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SavedError {
    /// `set()` was called on a slot that already holds a value — the
    /// forward that owns this `Saved` ran its save-for-backward step
    /// twice, which should never happen (a `StatefulKernelOp`'s `fwd` is
    /// called at most once per instance, since every instance is used
    /// through `apply_op3` exactly once — see the module doc).
    #[error(
        "Saved::set called on a slot that already holds a value — this op's forward saved \
         state twice for the same instance, which the fresh-Arc-per-call invariant \
         (see the `saved` module doc) says can never legitimately happen"
    )]
    AlreadySet,
    /// `take()` was called on an empty slot: either `fwd` never ran (a
    /// malformed graph), `fwd` did not save anything (a bug in the op),
    /// or a PRIOR `bwd` on this same node already took the value (a
    /// double backward on one node — candle's public `Tensor::backward()`
    /// does not itself prevent calling it twice on the same output).
    #[error(
        "Saved::take called on an empty slot — either forward never ran, forward never saved \
         state, or a previous backward on this same op instance already consumed it (a double \
         backward on one node); this op's contract is at most one backward per forward"
    )]
    Empty,
}

/// A write-once-read-once slot. See the module doc for the full safety
/// argument. `T` carries no bound of its own — `Saved<T>` adds `Send +
/// Sync` regardless of `T`'s own properties are (the `Mutex` provides
/// `Sync` for any `T: Send`; `Arc<Mutex<T>>: Send` requires `T: Send`,
/// which every real user of this type satisfies since the value crosses
/// the same `fwd`-to-`bwd` boundary `StatefulKernelOp` itself already
/// requires `Send` for).
pub struct Saved<T> {
    inner: Arc<Mutex<Option<T>>>,
}

impl<T> Saved<T> {
    /// A fresh, empty slot — the only way to construct one (there is no
    /// `Saved::new(value)` that starts pre-filled: every real caller's
    /// `fwd` computes the value it saves, so a pre-filled constructor
    /// would only invite skipping the domain check `set()` performs).
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Fills the slot. Errors with [`SavedError::AlreadySet`], never
    /// silently overwrites, if the slot already holds a value.
    pub fn set(&self, value: T) -> Result<(), SavedError> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_some() {
            return Err(SavedError::AlreadySet);
        }
        *guard = Some(value);
        Ok(())
    }

    /// Empties the slot and returns its value. Errors with
    /// [`SavedError::Empty`], never returns a default/stale `T`, if the
    /// slot is empty.
    pub fn take(&self) -> Result<T, SavedError> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.take().ok_or(SavedError::Empty)
    }

    /// `true` iff the slot currently holds a value. Test/introspection
    /// only — no production call site needs to peek without consuming.
    #[cfg(test)]
    pub(crate) fn is_set(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }
}

// Deliberately NOT `#[derive(Clone)]`: a `Saved<T>` field inside a
// `StatefulKernelOp` must not be duplicable (see that trait's own doc for
// why `StatefulKernelOp` excludes both `Copy` and `Clone` entirely, not
// just `Copy`) — cloning the `Arc` would let two op VALUES share one slot,
// reopening exactly the aliasing hazard `Saved` exists to close. If a
// future caller genuinely needs to hold two independent handles to the
// SAME underlying slot (not the case for any op in this crate today), that
// is a new, explicit method here, not a blanket `Clone`.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slot_starts_unset_and_take_errors() {
        let s: Saved<u32> = Saved::empty();
        assert!(!s.is_set());
        assert_eq!(s.take().unwrap_err(), SavedError::Empty);
    }

    #[test]
    fn set_then_take_round_trips_the_value() {
        let s: Saved<String> = Saved::empty();
        s.set("lse".to_string()).unwrap();
        assert!(s.is_set());
        assert_eq!(s.take().unwrap(), "lse".to_string());
        assert!(!s.is_set());
    }

    #[test]
    fn double_set_errors_and_keeps_the_first_value() {
        let s: Saved<u32> = Saved::empty();
        s.set(1).unwrap();
        assert_eq!(s.set(2).unwrap_err(), SavedError::AlreadySet);
        // The first value survives a rejected second `set` — a caller
        // that ignores the error (it shouldn't) still gets the correct
        // value, not the discarded one, on a later `take`.
        assert_eq!(s.take().unwrap(), 1);
    }

    #[test]
    fn double_take_errors_on_the_second_call() {
        let s: Saved<u32> = Saved::empty();
        s.set(7).unwrap();
        assert_eq!(s.take().unwrap(), 7);
        assert_eq!(s.take().unwrap_err(), SavedError::Empty);
    }

    /// The exact GradCache detached-pass-1 shape (`gradcache.rs:78-81`):
    /// forward runs (so `set()` fires), the output is dropped WITHOUT ever
    /// calling `.backward()` — the slot is left `Some` when the `Saved`
    /// (and the op instance that owns it) is dropped. This must not
    /// panic and must not leak in any way `Drop` can't handle: dropping
    /// an `Arc<Mutex<Option<T>>>` with `Some(T)` inside just drops `T`
    /// normally, which is exactly `#[derive(Drop)]`'s default behaviour
    /// for every field here (no explicit `Drop` impl exists, or needs to).
    #[test]
    fn set_without_take_drops_cleanly_forward_only_gradcache_shape() {
        let s: Saved<Vec<u8>> = Saved::empty();
        s.set(vec![1, 2, 3]).unwrap();
        assert!(s.is_set());
        drop(s); // must not panic
    }

    #[test]
    fn saved_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Saved<u32>>();
    }
}
