//! The shared optimizer-step seam: global-L2 gradient clipping followed by an
//! AdamW step, plus the loss-level convenience that runs `backward` first.
//!
//! Both the token-coupled text trainer and the non-text parallel loop reduce a
//! batch to a single update through this one place, so the clip→step contract
//! (and the `torch.nn.utils.clip_grad_norm_` semantics it implements) lives in
//! exactly one location rather than being copy-pasted per call site.
//!
//! ## Device-side clip
//!
//! [`clip_gradients`] used to end every trainable [`Var`]'s contribution to
//! the global norm with a `to_scalar::<f32>()` — a full-pipeline device→host
//! sync, once per `Var` (224 of them on a ModernBERT-large r16 `Wqkv`/`Wo`/`Wi`
//! LoRA config at the default `max_grad_norm = 1.0`). PyTorch's own
//! `torch.nn.utils.clip_grad._clip_grads_with_norm_` computes `clip_coef =
//! max_norm / (total_norm + 1e-6)`, clamps it to at most `1.0`, and multiplies
//! every gradient by the clamped coefficient **unconditionally** — its comment
//! is explicit that this "avoids a `if clip_coef < 1:` conditional which can
//! require a CPU <=> device synchronization" (`torch/nn/utils/clip_grad.py`,
//! `_clip_grads_with_norm_`, called from `clip_grad_norm_`; pinned to
//! `torch==2.13.0`, the version this repo's own PyTorch reference harness
//! targets — see `crates/jammi-bench/reference/README.md`). [`clip_gradients`]
//! now does the same: the
//! whole computation — per-`Var` squared sums, the global-norm reduction,
//! `sqrt`, the `max_norm / (total_norm + eps)` coefficient, its `<= 1.0`
//! clamp, and every gradient's rescale — stays on the device as one tensor
//! program with **zero** `to_scalar`/`to_vec` calls. `torch.nn.utils.
//! clip_grad_norm_`'s default is `error_if_nonfinite=False`: a NaN gradient is
//! silently scaled by a NaN coefficient and training continues. jammi keeps
//! a stricter contract — a non-finite total norm is a typed refusal — but
//! pays for that check with the one sync [`refuse_nonfinite_norm`] performs,
//! and therefore holds it to a CADENCE, not every step: [`clip_and_step`]
//! reads the norm back on step 1, every `check_every_n_steps`, and on the
//! run's last step (see that function's doc). Between cadence points a
//! NaN norm is NOT caught here — the clip scales every gradient by NaN and
//! `optimizer.step` writes the poisoned update into the parameters, exactly
//! as torch would. What bounds the damage is the trainer's epoch-boundary
//! backstop, `TrainingLoop::refuse_nonfinite_params` (`trainer.rs`): one
//! host read per epoch over the trainable parameters, a typed refusal before
//! any `checkpoint_best` can hold a non-finite weight. So the honest
//! statement is "refused within one cadence interval of steps, and never
//! checkpointed", not "never silent".
//!
//! ## Accumulator precision: an undisclosed change the device rewrite also
//! made, and why the new value is the correct one to keep
//!
//! The pre-rewrite host implementation read EVERY `Var`'s own `f32` squared
//! sum back with `to_scalar::<f32>()` and accumulated the cross-`Var` fold
//! in an `f64` HOST scalar (`let mut total_sq = 0.0f64; … total_sq += sq as
//! f64;`, then `total_sq.sqrt()` — also `f64`). [`clip_gradients`] now folds
//! that SAME cross-`Var` sum entirely as `f32` device tensors (`Some(acc) =>
//! (&acc + &sq)…`, `total_sq.sqrt()` on the `f32` tensor) — a genuine
//! precision change to the fold's accumulator this doc did not previously
//! call out. It is not a regression to fix, though: PyTorch's OWN reference
//! implementation never promotes to `f64` either — `torch.linalg.
//! vector_norm` computes each per-parameter norm AND the outer fold of the
//! stacked per-parameter norms (`_get_total_norm`, `torch/nn/utils/
//! clip_grad.py`) entirely in the gradient's own dtype (`f32`, for
//! full-precision training) — so the pre-rewrite `f64` host accumulation
//! was an ACCIDENTAL side effect of syncing every `Var` back individually,
//! never a deliberate higher-precision design choice, and it was already a
//! (small) DEPARTURE from torch's own `f32`-throughout behavior, not a
//! closer match to it (family K: parity target is PyTorch, not a candle
//! implementation detail some earlier revision of this file happened to
//! have). The device-side `f32` fold this file now performs is the one that
//! matches torch's own accumulator precision for `f32` gradients. It is NOT
//! bit-identical to torch, and this doc does not claim it: (1) the
//! coefficient is now computed with torch's own rounding COUNT — `denom =
//! total_norm.affine(1.0, 1e-6)` (one add) then `max_norm_t.div(&denom)`
//! (one division, candle's `Div` kernel, `v1 / v2` — not `recip()` then a
//! second `affine`-multiply, an earlier revision of this function used,
//! which cost a SECOND `f32` rounding torch's own `max_norm / (total_norm +
//! 1e-6)` never pays; see [`clip_gradients`]'s own doc for the derivation)
//! — so this fold's ONLY remaining source of drift from torch's coefficient
//! is the `f32` values `total_norm`/`max_norm` themselves being the result
//! of a differently-shaped reduction, not an extra rounding in the
//! coefficient's own arithmetic; (2) the fold shape
//! differs (`sqrt(Σ_i Σ g_i²)` here vs torch's norm-of-per-parameter-norms,
//! bounded in [`clip_gradients`]'s doc); and (3) for a NON-`f32` gradient
//! (bf16 LoRA training) torch keeps the whole computation in the gradient's
//! own dtype — `torch._foreach_norm` returns bf16 per-parameter norms, the
//! coefficient is bf16, `foreach_mul_` is a bf16 × bf16 — while this file
//! upcasts each gradient to `f32`, folds/scales in `f32`, and rounds back to
//! bf16 once at the end, a genuinely different (fewer-rounding) arithmetic
//! that is closer to the real-number clip than torch's, not equal to it.
//! `tests::multi_var_clip_
//! matches_host_reference_on_cpu` and `tests::clip_gradients_device_and_
//! host_agree_bit_identically_on_cpu` (below) pin that the fold's own
//! op-sequence behavior matches an independent host reference at the
//! per-op level this file's doc derives; this crate does not carry a
//! dedicated test pinning the f64-host-vs-f32-device magnitude at full
//! 224-`Var` production width specifically — see this PR's own
//! description for why that apparatus was dropped in favor of the
//! narrower, per-op derivation these two tests check directly.

use candle_core::{backprop::GradStore, DType, Tensor, Var};
use candle_nn::VarMap;
use jammi_db::error::{JammiError, Result};

use crate::fine_tune::adamw::AdamW;

/// Snapshot every trainable `Var` in `varmap`, in a DETERMINISTIC order —
/// sorted by its `VarBuilder`-path NAME, never `VarMap::all_vars()`'s raw
/// `HashMap` iteration order.
///
/// Why this matters (esc-182, a sibling audit finding): `VarMap::data()` is a
/// `std::collections::HashMap<String, Var>` (candle-nn 0.11.0, `var_map.rs`)
/// — `all_vars()` is `tensor_data.values().collect()`, and `std::HashMap`'s
/// default hasher is keyed by a PER-PROCESS random seed (`RandomState`), so
/// its iteration order is stable within one process's lifetime for a given
/// table (never mutated between reads) but is NOT reproducible across two
/// separate process invocations of the identical program with the identical
/// `seed` — a second `cargo test`/training run gets a DIFFERENT `all_vars()`
/// order from the first, purely from HashMap's own randomized hashing, wholly
/// independent of the caller's `seed` parameter. [`clip_gradients`]'s own doc
/// claims a "fixed left-to-right fold order… deterministic run to run" — that
/// claim is only as true as the ORDER its `trainable_vars: &[Var]` argument
/// arrives in, which this function's caller (not `clip_gradients` itself,
/// which cannot see names) is responsible for pinning. `trainer.rs`'s
/// `TrainingLoop::run` and `parallel_train.rs`'s `run_parallel_training` both
/// snapshot `trainable_vars` ONCE (`self.varmap.all_vars()`, pre-esc-182) and
/// reuse that same `Vec` for gradient accumulation, the clip's fold, AND the
/// `AdamW` optimizer's positional moment vector — self-consistent WITHIN one
/// run (the snapshot itself never reorders), but the clip's fold order (and
/// therefore the last bits of `total_norm`, and therefore the last bits of
/// every clipped gradient) still differs BETWEEN independent process
/// invocations of the same seed. Sorting by name here removes that
/// process-level nondeterminism from a parity-critical path (family J: no
/// unseeded RNG) — the optimizer's OWN moment-restoration path already had to
/// solve this same problem for a different reason (`trainer.rs`'s
/// `optim_param_names`, keying `AdamW`'s resume-from-checkpoint moments by
/// name rather than by this same unstable position) — this closes the
/// matching gap for the clip's own fold.
pub fn sorted_trainable_vars(varmap: &VarMap) -> Vec<Var> {
    let data = varmap.data().lock().unwrap_or_else(|e| e.into_inner());
    let mut named: Vec<(&String, &Var)> = data.iter().collect();
    named.sort_by(|a, b| a.0.cmp(b.0));
    named.into_iter().map(|(_, v)| v.clone()).collect()
}

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

/// Test-only counter of calls to [`refuse_nonfinite_norm`] — the *only*
/// function on this file's per-step path allowed to call
/// `to_scalar`/`to_vec`, and the ONLY site that bumps this counter. A CPU
/// test cannot observe "no CUDA sync happened" directly (there is no CUDA
/// stream to inspect), so this counter is a structural proxy for exactly
/// one claim: that [`clip_gradients`] and an off-cadence [`clip_and_step`]
/// never reach `refuse_nonfinite_norm`. It does NOT count `to_scalar`/
/// `to_vec` calls themselves — candle's host reads are not instrumented —
/// so a future edit that adds a host read somewhere ELSE on the hot path
/// would leave this counter at `0` and the tests below green. That gap is
/// closed by review, not by this counter: the module doc's "zero
/// `to_scalar`/`to_vec` calls" claim is checked by reading
/// [`clip_gradients`]'s body (`grep -n 'to_scalar\|to_vec'` over this file
/// must match only `refuse_nonfinite_norm` and `#[cfg(test)]` code), and
/// the CUDA leg `clip_gradients_never_reads_the_norm_back_on_cuda` proves the
/// same proxy on a real device, no more.
#[cfg(test)]
static SYNC_READ_COUNT: AtomicU64 = AtomicU64::new(0);

/// Snapshot [`SYNC_READ_COUNT`]. Test-only.
#[cfg(test)]
pub(crate) fn sync_read_count() -> u64 {
    SYNC_READ_COUNT.load(Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// The per-THREAD twin of [`SYNC_READ_COUNT`]: a run-level test drives
    /// `TrainingLoop::run` on its own thread and reads this, so the count it
    /// asserts on is exactly that run's reads — unperturbed by every other
    /// test's training in the same process, which the global counter cannot
    /// separate.
    static THREAD_SYNC_READ_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Snapshot [`THREAD_SYNC_READ_COUNT`] for the calling thread. Test-only.
#[cfg(test)]
pub(crate) fn thread_sync_read_count() -> u64 {
    THREAD_SYNC_READ_COUNT.with(|c| c.get())
}

#[cfg(test)]
thread_local! {
    /// Test-only counter of calls to [`clip_gradients`] itself (not just its
    /// on-cadence `refuse_nonfinite_norm` sub-call): incremented once at the
    /// top of every invocation, regardless of which [`ClipOutcome`] arm it
    /// returns (`Disabled`/`NoGradients`/`Clipped`). `clip_and_step` calls
    /// `clip_gradients` exactly once per optimizer step, so this counter is
    /// a direct proxy for "how many optimizer steps ran" on the clip side of
    /// the clip→step seam — the pin `trainer.rs`'s `last_step_horizon_run_
    /// oracles` module uses to prove a fixed `(n_pairs, batch_size, epochs)`
    /// config's `result.total_steps` and its clip-call count agree exactly,
    /// i.e. that step counting is not a second, independently-drifting
    /// count.
    static THREAD_CLIP_CALL_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Snapshot [`THREAD_CLIP_CALL_COUNT`] for the calling thread. Test-only.
#[cfg(test)]
pub(crate) fn thread_clip_call_count() -> u64 {
    THREAD_CLIP_CALL_COUNT.with(|c| c.get())
}

/// The outcome of a [`clip_gradients`] call.
///
/// Before this split, `clip_gradients` returned `Option<Tensor>` and
/// collapsed TWO semantically different situations into the SAME `None`:
/// `max_norm <= 0.0` (a deliberate operator choice — clipping is off) and
/// "clipping was requested but not one of `trainable_vars` had a gradient
/// present" (which, whenever `trainable_vars` is non-empty, is AMBIGUOUS —
/// a bug signal such as a detached graph, an all-frozen adapter, or a
/// `GradStore` that was never populated, OR a batch whose loss legitimately
/// never routes through any of these `Var`s by design — see
/// [`clip_and_step`]'s own doc for that ambiguity's resolution). A caller
/// downstream of that `None` had no way to tell "I asked for this" apart
/// from "clipping ran but touched nothing", so [`clip_and_step`] silently
/// skipped BOTH the clip and the non-finite check in either case.
#[derive(Debug)]
#[must_use = "a ClipOutcome says whether the clip actually ran (`Clipped`) or was skipped \
              (`Disabled`/`NoGradients`); a caller that drops it cannot tell a clip-on step \
              from a no-op — match on it (see `clip_and_step`)"]
pub enum ClipOutcome {
    /// `max_norm <= 0.0` — clipping was turned off by the caller.
    Disabled,
    /// `max_norm > 0.0`, but not one of `trainable_vars` had a gradient
    /// present in `grads`. Unambiguously benign when `trainable_vars`
    /// itself was empty (nothing was ever asked to be clipped); AMBIGUOUS
    /// otherwise — see [`clip_and_step`]'s own handling of this arm.
    NoGradients,
    /// Clipping ran; the on-device `total_norm` scalar tensor, for the
    /// caller to read back — at whatever cadence it chooses — through
    /// [`refuse_nonfinite_norm`].
    Clipped(Tensor),
}

impl ClipOutcome {
    /// The on-device `total_norm` tensor when clipping actually ran —
    /// `None` for BOTH `Disabled` and `NoGradients` (a caller that does not
    /// need to distinguish those two, e.g. to decide whether to read the
    /// norm back, can use this; a caller that DOES need to distinguish
    /// them, like [`clip_and_step`]'s own handling, matches on `self`
    /// directly instead).
    pub fn total_norm(&self) -> Option<&Tensor> {
        match self {
            ClipOutcome::Clipped(t) => Some(t),
            ClipOutcome::Disabled | ClipOutcome::NoGradients => None,
        }
    }

    /// Unwraps the `Clipped` arm's tensor, panicking with the OTHER arm
    /// named otherwise. Test-only ergonomic sugar for the many fixtures in
    /// this file that always expect clipping to have run — production code
    /// must match all three arms explicitly (see [`clip_and_step`]).
    #[cfg(test)]
    fn unwrap_clipped(self) -> Tensor {
        match self {
            ClipOutcome::Clipped(t) => t,
            other => panic!("expected ClipOutcome::Clipped, got {other:?}"),
        }
    }
}

/// Clip gradients by global L2 norm in-place, matching
/// `torch.nn.utils.clip_grad_norm_(params, max_norm)`, entirely on-device.
///
/// Computes `total_norm = sqrt(sum ||g||² for all g)` as a device scalar
/// tensor (a fixed left-to-right fold over `trainable_vars` IN THE ORDER THE
/// CALLER SUPPLIES IT). This function has no way to see `Var` names and
/// cannot enforce ordering itself — the "deterministic run to run" half of
/// that claim is only as true as its caller's ordering. Every production
/// caller in this crate (`trainer.rs`, `parallel_train.rs`) now builds
/// `trainable_vars` via [`sorted_trainable_vars`], never a raw
/// `VarMap::all_vars()` — see that function's own doc (esc-182) for why a raw
/// `all_vars()` order is stable WITHIN one process's use of one `VarMap` but
/// NOT reproducible ACROSS separate process invocations of the identical
/// `seed`, which would otherwise make this fold's last bits (and therefore
/// every clipped gradient's last bits) a function of `HashMap`'s per-instance
/// hash-randomization rather than of `seed` alone.
/// Reduction order versus torch: `clip_grad_norm_` takes a norm of
/// per-parameter norms (`torch._foreach_norm` per gradient, then
/// `vector_norm` of the stacked norms — `sqrt(Σ_i (sqrt(Σ g_i²))²)`), while
/// this folds the raw per-`Var` sums of squares and takes ONE `sqrt`
/// (`sqrt(Σ_i Σ g_i²)`); the two agree in real arithmetic and differ in
/// `f32` by the extra `sqrt`-then-square per `Var` torch rounds through —
/// at most ~1 ulp of each squared term, so `|total_norm_jammi -
/// total_norm_torch| <= (k + 2) · ε_f32 · total_norm` for `k` `Var`s
/// (the `+ 2` for the shared final `sqrt` and the sum's own rounding),
/// well inside the 4-ulp band `tests::multi_var_clip_matches_host_reference_
/// on_cpu` asserts against a host reference.
/// Every gradient is then rescaled by `clip_coef = (max_norm / (total_norm +
/// 1e-6)).min(1.0)` **unconditionally**, matching torch's own avoidance of a
/// host-syncing `if total_norm > max_norm` branch (`torch/nn/utils/
/// clip_grad.py`, `_clip_grads_with_norm_`; see the module doc's citation).
///
/// **Bit-identity guarantee — ONE DIRECTION ONLY.** Real-number algebra says
/// `clip_coef` clamps to EXACTLY `1.0` — making the rescale bit-identical to
/// skipping it entirely, since `x * 1.0 == x` for every finite `x` — exactly
/// when `max_norm / (total_norm + 1e-6) >= 1.0`, i.e. when `total_norm <=
/// max_norm - 1e-6`. In `f32`, only the FORWARD implication is true:
/// `total_norm <= max_norm - 1e-6` (with enough margin that `f32` rounding of
/// `total_norm + 1e-6` cannot push the ratio back above `1.0`) DOES guarantee
/// `clip_coef == 1.0` exactly. The CONVERSE — `clip_coef == 1.0` implies
/// `total_norm <= max_norm - 1e-6` — is **FALSE** in `f32`: `total_norm =
/// 0.999999046` (`max_norm = 1.0`) is strictly GREATER than `max_norm - 1e-6
/// = 0.999999` in real-number terms, yet `total_norm + 1e-6` rounds to
/// exactly `1.0` in `f32` (the true sum, `1.000000046`, is within half a
/// `f32` ULP of `1.0` at that magnitude), so `clip_coef` still comes out
/// EXACTLY `1.0` — a counterexample to the converse, not an edge case this
/// doc can wave away. Do not read "clip_coef == 1.0" as a certificate that
/// `total_norm` was safely below the threshold; it only certifies the
/// rescale was a no-op THIS time.
///
/// Separately from that boundary: on the half-open band `total_norm ∈
/// (max_norm - 1e-6, max_norm]`, `clip_coef` is generally strictly less than
/// `1.0` (torch's own `1e-6` epsilon in the denominator puts it there) and the
/// rescale perturbs every element. Concretely: 4 gradients of `0.5` each give
/// `total_norm == max_norm == 1.0`, which sits inside that band — `clip_coef
/// == 0.9999990463256836` (`f32` bits `0x3f7ffff0`) scales each `0.5` down to
/// exactly `0.49999952316284180`, NOT bit-identical to the unclipped `0.5`.
/// See `tests::at_max_norm_boundary_coef_is_not_bit_identical_to_no_clip` for
/// that pinned exact-bits value, and
/// `tests::below_max_norm_clip_is_bit_identical_to_no_clip` for a `total_norm`
/// safely inside the `<= max_norm - 1e-6` region, where the forward guarantee
/// does hold. `max_norm <= 0.0` disables clipping entirely (unchanged from
/// before — no compute is issued; this is [`ClipOutcome::Disabled`]).
///
/// Returns a [`ClipOutcome`] — `Disabled`/`NoGradients`/`Clipped(total_norm)`
/// — rather than collapsing the first two into the SAME `None` a caller
/// cannot tell apart (see that enum's own doc for why the distinction
/// matters). The `Clipped` arm's on-device `total_norm` scalar tensor is for
/// a caller to read back — at whatever cadence it chooses — through
/// [`refuse_nonfinite_norm`]. `clip_gradients` itself never reads it.
///
/// Device op count for `n` trainable `Var`s with a gradient present: `n` ×
/// (`sqr` + `sum_all`) for the per-`Var` squared sums, `n - 1` adds to fold
/// them into one scalar, then `sqrt` + `affine` (+ eps) + `full` (materialize
/// `max_norm` as a same-shape device scalar — no scalar-numerator `div` exists
/// to skip this, see [`clip_gradients`]'s own doc) + `div` (torch's exact
/// coefficient, one rounding, not `recip` + a second `affine`) + `minimum`
/// (the ≤ 1.0 clamp) = 5 fixed ops for the coefficient (the SAME count the
/// prior `recip`-then-`affine` sequence used — `full` replaces one `affine`
/// rather than adding a sixth op), and `n` × `broadcast_mul` to rescale every
/// gradient — `4n + 4` device ops total, zero of them a host read. (A
/// gradient that is not
/// already `F32` pays one extra `to_dtype` in the squared-sum loop AND one
/// more in the rescale loop, not counted above — `coef` is always `F32`
/// [`total_sq`'s dtype, from the per-`Var` upconvert above], so a non-`F32`
/// gradient must upconvert for `broadcast_mul` and downconvert back to its
/// OWN dtype afterward, the same round trip torch's `foreach_mul_` leaves a
/// non-`F32` `.grad` in: gradient dtype is a contract with whatever built the
/// `GradStore` — typically the `Var`'s own dtype, which the optimizer's
/// moments are shaped to — not something this function may silently change.)
pub fn clip_gradients(
    trainable_vars: &[Var],
    grads: &mut GradStore,
    max_norm: f64,
) -> Result<ClipOutcome> {
    #[cfg(test)]
    THREAD_CLIP_CALL_COUNT.with(|c| c.set(c.get() + 1));

    // Domain-validity at the edge (family D): `max_norm.is_nan()` makes
    // `max_norm <= 0.0` below `false` (NaN compares false against
    // everything), so a NaN `max_norm` would otherwise fall THROUGH the
    // disable-clipping guard and into the clip computation, where
    // `clip_coef = max_norm / (total_norm + eps)` is NaN unconditionally —
    // every gradient silently scaled by NaN, forever, on every step. Worse:
    // `total_norm` itself (the value [`refuse_nonfinite_norm`] later checks)
    // is computed from the GRADIENTS, not from `max_norm`, so it stays
    // perfectly finite even while the coefficient corrupts every parameter —
    // the existing non-finite-norm check cannot see this class of bug at
    // all. `max_norm = ±inf` has the same defect in kind even though its
    // arithmetic happens to clamp back to a merely-wasteful (not corrupting)
    // `coef == 1.0`: it is still a caller passing a non-finite tuning
    // parameter into a numeric contract that promises finite behavior, and a
    // typed refusal here is cheap (checked once per call, off the per-`Var`
    // loop) versus silently accepting it. Refuse both up front, at this
    // function's own boundary — the abstraction `clip_and_step`/`optimizer_
    // step`/every training loop shares — rather than trusting every future
    // caller (or `FineTuneConfig`'s deserialization, which does not validate
    // this field's finiteness) to pre-filter it.
    if !max_norm.is_finite() {
        return Err(JammiError::FineTune(format!(
            "GradClip: max_norm must be finite, got {max_norm}"
        )));
    }
    if max_norm <= 0.0 {
        return Ok(ClipOutcome::Disabled);
    }

    // Fold the per-`Var` squared sums into one device scalar, in
    // `trainable_vars` order — fixed fold order, so this is deterministic
    // (family J: no unseeded/order-dependent reduction).
    let mut total_sq: Option<Tensor> = None;
    for var in trainable_vars {
        let t: &Tensor = var;
        if let Some(g) = grads.get(t) {
            let g_f32 = if g.dtype() == DType::F32 {
                g.clone()
            } else {
                g.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("GradClip dtype: {e}")))?
            };
            let sq = g_f32
                .sqr()
                .map_err(|e| JammiError::FineTune(format!("GradClip sqr: {e}")))?
                .sum_all()
                .map_err(|e| JammiError::FineTune(format!("GradClip sum: {e}")))?;
            total_sq = Some(match total_sq {
                None => sq,
                Some(acc) => {
                    (&acc + &sq).map_err(|e| JammiError::FineTune(format!("GradClip acc: {e}")))?
                }
            });
        }
    }

    // No trainable gradient was present this step (e.g. an empty
    // `trainable_vars`, or none of them appear in `grads`) — nothing to
    // clip. `ClipOutcome::NoGradients`, NOT `Disabled`: whether this is
    // benign (an empty `trainable_vars`) or a bug signal (a non-empty one
    // with nothing present in `grads`) is for the CALLER to decide — see
    // `clip_and_step`'s own handling of this arm.
    let Some(total_sq) = total_sq else {
        return Ok(ClipOutcome::NoGradients);
    };

    let total_norm = total_sq
        .sqrt()
        .map_err(|e| JammiError::FineTune(format!("GradClip sqrt: {e}")))?;

    // clip_coef = max_norm / (total_norm + 1e-6), clamped to at most 1.0 —
    // torch's EXACT op sequence, rounding count included (`torch/nn/utils/
    // clip_grad.py`, `_clip_grads_with_norm_`; see the module doc's
    // citation): `clip_coef = max_norm / (total_norm + 1e-6)` is Python
    // `float / Tensor`, ATen's binary `div` kernel — ONE add (the `+ 1e-6`)
    // then ONE division, never a reciprocal. An earlier revision of this
    // function computed `(total_norm + 1e-6).recip() * max_norm` — a
    // reciprocal then a multiply, TWO `f32` roundings where torch performs
    // one — which could differ from torch's coefficient by ~1 ULP for
    // `max_norm != 1.0` (at the shipped default `max_norm == 1.0`, `x * 1.0
    // == x` exactly, so that revision's extra rounding was invisible at the
    // production default specifically, not fixed by it). This computes the
    // same op sequence, and the same rounding COUNT, as torch:
    //  1. `denom = total_norm.affine(1.0, 1e-6)` — `total_norm * 1.0` is
    //     exact (multiplying any finite `f32` by `1.0` introduces no
    //     rounding), so this is ONE rounding, matching torch's `total_norm +
    //     1e-6`.
    //  2. `max_norm_t.div(&denom)` — candle has no scalar-numerator `div`
    //     that keeps `max_norm` a host constant the way `affine`/`recip`
    //     bake their constants as kernel-launch scalars (`Tensor::div`/
    //     `broadcast_div` are both `binary_op!(div, Div)` — same-shape
    //     TENSOR/TENSOR only, `candle-core-0.11.0` `tensor.rs`; there is no
    //     `TensorOrScalar`-accepting `div` the way `minimum`/`maximum` get
    //     one through `binary_op_scalar!`) — so `max_norm` is materialized
    //     as a same-shape device scalar via `Tensor::full` (a device-side
    //     fill, `alloc_uninit` + `const_set` — no host array upload, the
    //     same class of small op as `minimum`'s own scalar-to-tensor
    //     promotion below) and divided by `denom` through candle's `Div`
    //     kernel — `bin_op!(Div, div, |v1, v2| v1 / v2)` (`op.rs`) — a
    //     genuine single-rounding division, NOT `Recip`'s `v.recip()`
    //     (`== 1.0 / v` in Rust's own `f32::recip`, still one rounding on
    //     its own, but paired with the second `affine`-multiply rounding
    //     the old sequence needed to fold in `max_norm`). ONE rounding here,
    //     matching torch's ONE division.
    // Total: two roundings before the clamp (the add, the division) —
    // identical to torch's rounding count. `minimum`'s own `1.0` is a
    // third small device scalar (materialized the same way, via
    // `binary_op_scalar!`'s own `TensorOrScalar` promotion) — one small
    // H2D-adjacent op per call, same as before this change; not cached
    // (nothing in this crate holds candle's `CUDA_GRAPH_HTOD_CACHE` for the
    // run — see [`crate::fine_tune::trainer`]'s doc on why an unbounded,
    // never-evicted, run-lifetime cache is the wrong trade). Below,
    // `broadcast_mul` has the same shape on the CUDA backend: `coef` is a
    // stride-0 scalar broadcast against every gradient, which candle routes
    // through its general strided-binary-op kernel (a fresh per-call
    // dims/strides upload) rather than `Tensor::affine`'s host-`f64`-constant
    // fast path — kept as `broadcast_mul` rather than rewritten to a
    // `CustomOp`, with that extra upload's cost meant to be measured on
    // device (not assumed) before it is optimized away.
    let denom = total_norm
        .affine(1.0, 1e-6)
        .map_err(|e| JammiError::FineTune(format!("GradClip denom: {e}")))?;
    let max_norm_t = Tensor::full(max_norm as f32, denom.shape(), denom.device())
        .map_err(|e| JammiError::FineTune(format!("GradClip max_norm const: {e}")))?;
    let clip_coef = max_norm_t
        .div(&denom)
        .map_err(|e| JammiError::FineTune(format!("GradClip coef: {e}")))?;
    let coef = clip_coef
        .minimum(1.0)
        .map_err(|e| JammiError::FineTune(format!("GradClip clamp: {e}")))?;

    for var in trainable_vars {
        let t: &Tensor = var;
        if let Some(g) = grads.remove(t) {
            // `coef` is always F32 (folded from the per-`Var` squared sums,
            // upconverted above): a non-F32 gradient must upconvert here too
            // — the SAME class of bug this function's `is_finite` guard
            // exists to catch, just a dtype-domain edge instead of a
            // value-domain one (family D). Round-trip back to the
            // gradient's OWN dtype afterward rather than leaving it F32:
            // dtype is part of the `GradStore` contract this function did
            // not create and must not silently change.
            let orig_dtype = g.dtype();
            let g_f32 = if orig_dtype == DType::F32 {
                g
            } else {
                g.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("GradClip dtype (scale): {e}")))?
            };
            let scaled_f32 = g_f32
                .broadcast_mul(&coef)
                .map_err(|e| JammiError::FineTune(format!("GradClip scale: {e}")))?;
            let scaled = if orig_dtype == DType::F32 {
                scaled_f32
            } else {
                scaled_f32.to_dtype(orig_dtype).map_err(|e| {
                    JammiError::FineTune(format!("GradClip dtype (scale, downconvert): {e}"))
                })?
            };
            grads.insert(t, scaled);
        }
    }

    Ok(ClipOutcome::Clipped(total_norm))
}

/// The one deliberate device→host sync this file keeps: read `total_norm`
/// back and refuse a non-finite value with a typed error naming `step`.
///
/// `torch.nn.utils.clip_grad_norm_`'s default (`error_if_nonfinite=False`)
/// silently scales every gradient by a NaN/Inf coefficient and trains on;
/// jammi diverges from that default deliberately — a NaN gradient must never
/// train silently — but the read this requires is exactly the kind of
/// mid-step stall [`clip_gradients`] was rewritten to remove, so callers must
/// NOT invoke this every step. Every call site in this crate checks on the
/// same cadence — every [`DEFAULT_NORM_CHECK_INTERVAL`] *optimizer* steps
/// (not micro-batches: `step` is `global_step + 1`, the 1-based index of the
/// optimizer step about to run) — via [`clip_and_step`]'s
/// `check_every_n_steps` parameter, never every micro-batch.
pub fn refuse_nonfinite_norm(total_norm: &Tensor, step: usize) -> Result<()> {
    #[cfg(test)]
    SYNC_READ_COUNT.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    THREAD_SYNC_READ_COUNT.with(|c| c.set(c.get() + 1));

    let norm: f32 = total_norm
        .to_scalar::<f32>()
        .map_err(|e| JammiError::FineTune(format!("GradClip norm read: {e}")))?;
    if !norm.is_finite() {
        return Err(JammiError::FineTune(format!(
            "GradClip: non-finite total gradient norm ({norm}) at optimizer step {step} — \
             refusing to train on it. (torch's clip_grad_norm_ default, \
             error_if_nonfinite=False, would silently scale every gradient by a NaN/Inf \
             coefficient here instead.)"
        )));
    }
    Ok(())
}

/// Default cadence, in optimizer steps, for [`refuse_nonfinite_norm`] where a
/// call site has no more natural boundary (e.g. [`optimizer_step`]'s
/// one-batch-per-step loop, which has no epoch). Chosen to surface a diverged
/// run within a small, bounded number of steps while keeping the sync off all
/// but a `1 / DEFAULT_NORM_CHECK_INTERVAL` fraction of them — every step is
/// explicitly not acceptable (that is the sync this file exists to remove).
pub const DEFAULT_NORM_CHECK_INTERVAL: usize = 50;

/// Clip an already-computed gradient store, then take one AdamW step.
///
/// This is the seam both training loops share: whatever produced `grads` (a
/// single backward, an accumulation window, or the GradCache two-pass
/// backward), the clip-then-step that turns them into a parameter update is
/// identical. `max_grad_norm <= 0.0` skips clipping.
///
/// `check_every_n_steps` gates [`refuse_nonfinite_norm`]: the norm is read
/// back and checked when `step` (the 1-based index of the optimizer step
/// about to run — pass `global_step + 1`) is `1`, is a multiple of
/// `check_every_n_steps`, OR `is_last_step` is `true` — and never when
/// `check_every_n_steps == 0` (an explicit full opt-out every call site in
/// this crate currently leaves unused). The `step == 1` and `is_last_step`
/// arms exist because the modulo cadence alone silently skips every run
/// shorter than `check_every_n_steps` steps end to end: with only the modulo
/// check, a run of, say, 12 steps against the default interval of 50 would
/// never call [`refuse_nonfinite_norm`] even once, and a NaN gradient on its
/// very last step would train silently and get saved into the adapter. `step
/// == 1` catches a bad start immediately; `is_last_step` (the caller states
/// whether `step` is the final optimizer step of the whole run — trainer.rs's
/// callers know this from the LR-schedule horizon they already compute)
/// catches a bad end even when the run never reaches a full interval.
///
/// [`ClipOutcome::NoGradients`] is handled OUTSIDE the cadence gate above —
/// unconditionally, on every call, off-cadence or not — because it is not a
/// non-finite-norm finding (there is no norm tensor to even read back).
/// Whenever `trainable_vars` is non-empty, this is an AMBIGUOUS state this
/// function cannot resolve on its own (see [`ClipOutcome::NoGradients`]'s
/// own doc): it could be a genuine bug (a detached graph, an all-frozen
/// adapter, an unpopulated `GradStore`), or a batch whose loss legitimately
/// never routes through any of these `Var`s by DESIGN (e.g.
/// `TrainingBatch::Contrastive`'s `contrastive_loss` scores raw precomputed
/// embeddings directly, never through a `ProjectionHead`'s LoRA layers — a
/// real, common shape across this crate's own step-counting/schedule test
/// oracles, never a bug in those tests). Since a hard refusal here would
/// break every one of those legitimate call sites, this is a COUNTED FACT
/// instead — a `tracing::warn!` an operator can grep/alert on — and the
/// step proceeds (`optimizer.step(grads)` over an empty `GradStore` is a
/// no-op, unchanged from before `ClipOutcome` existed). An EMPTY
/// `trainable_vars` is the one UNAMBIGUOUSLY benign reading (nothing was
/// ever asked to be clipped) and does not warn.
pub fn clip_and_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    grads: &mut GradStore,
    max_grad_norm: f64,
    check_every_n_steps: usize,
    step: usize,
    is_last_step: bool,
) -> Result<()> {
    let outcome = clip_gradients(trainable_vars, grads, max_grad_norm)?;
    if matches!(outcome, ClipOutcome::NoGradients) && !trainable_vars.is_empty() {
        // A COUNTED FACT (`tracing::warn!`, a structured field a log
        // pipeline/dashboard can grep and alert on), never a hard refusal:
        // an earlier revision of this branch returned `Err` here, which
        // broke every legitimate loss that does not route through EVERY
        // trainable `Var` for a given batch — e.g. `TrainingBatch::
        // Contrastive`'s `contrastive_loss` scores raw precomputed
        // embeddings directly, never through `TrainingTarget::
        // ProjectionHead`'s LoRA layers, so its trainable Vars carry no
        // gradient by DESIGN, not by bug (`ft_correctness_sweep.rs`'s
        // step-counting oracles, `ft_determinism.rs`, and
        // `encoder_adapters.rs` all exercise exactly this shape). The
        // AMBIGUITY this branch cannot resolve on its own — "this batch's
        // loss legitimately never touches these Vars" versus "a detached
        // graph / an all-frozen adapter / an unpopulated GradStore" — is
        // real, so refusing outright is not the answer to that ambiguity;
        // making it OBSERVABLE (searchable in logs, countable by a
        // dashboard) is. The step proceeds exactly as it did before
        // `ClipOutcome` existed: `optimizer.step(grads)` over an empty
        // `GradStore` is a harmless no-op.
        tracing::warn!(
            step,
            trainable_vars = trainable_vars.len(),
            "GradClip: trainable Var(s) were passed but NONE had a gradient present at this \
             optimizer step — either this step's loss legitimately does not route through any \
             of them, or this is a detached graph / an all-frozen adapter / a GradStore that \
             was never populated. Proceeding (the optimizer step over an empty GradStore is a \
             no-op); this line is the counted fact an operator investigates if it recurs \
             unexpectedly."
        );
    }
    let on_cadence = check_every_n_steps > 0
        && (step == 1 || step.is_multiple_of(check_every_n_steps) || is_last_step);
    if on_cadence {
        if let Some(norm) = outcome.total_norm() {
            refuse_nonfinite_norm(norm, step)?;
        }
    }
    optimizer
        .step(grads)
        .map_err(|e| JammiError::FineTune(format!("Optimizer step: {e}")))
}

/// Merge a freshly-computed [`GradStore`] into a running accumulator,
/// summing per-var over `vars`. The shared gradient-accumulation merge: both
/// the text trainer's micro-batch loop and GradCache's per-chunk
/// accumulation pass reduce to this one place, sitting next to the
/// clip→step seam it feeds.
///
/// `acc` starts as `GradStore::default()` (a public, `#[derive(Default)]`
/// constructor — `GradStore::new()` itself is private, but `Default` is not)
/// and is never re-seeded from a fresh store: for the first call, every var
/// present in `fresh` takes the `else` branch below (`acc.remove(t)` finds
/// nothing in the still-empty accumulator) and is inserted directly, with no
/// addition performed — numerically identical to the wholesale
/// `acc = fresh` a first-call special case would have done, but expressed as
/// the same per-var loop every subsequent call takes. `fresh` is consumed
/// (not borrowed): its values are moved into `acc` via `remove` rather than
/// cloned, since nothing else needs `fresh` after this call.
pub(crate) fn accumulate_grads(
    acc: &mut GradStore,
    mut fresh: GradStore,
    vars: &[Var],
) -> Result<()> {
    for var in vars {
        let t: &Tensor = var;
        if let Some(g_new) = fresh.remove(t) {
            if let Some(g_acc) = acc.remove(t) {
                let summed = (&g_acc + &g_new)
                    .map_err(|e| JammiError::FineTune(format!("Grad acc: {e}")))?;
                acc.insert(t, summed);
            } else {
                acc.insert(t, g_new);
            }
        }
    }
    Ok(())
}

/// Backward `loss`, clip the resulting gradients, and take one AdamW step — the
/// whole batch→update sequence for a loop without gradient accumulation.
///
/// The text trainer accumulates micro-batch gradients before stepping, so it
/// calls [`clip_and_step`] at its window boundaries rather than this; the
/// non-text parallel loop has one batch per step and uses this directly. That
/// loop has no epoch boundary to hang the non-finite check on, so this checks
/// every [`DEFAULT_NORM_CHECK_INTERVAL`] steps (`optimizer.step_t() + 1`,
/// the 1-based index of the step about to run), plus step `1` and
/// `is_last_step` unconditionally (see [`clip_and_step`]'s doc — a run
/// shorter than the interval must still be checked at least once).
/// `is_last_step`: the caller states whether this call is the final optimizer
/// step of the whole run (this loop has no epoch boundary of its own to
/// derive that from, so the caller — which knows the batch/epoch count —
/// supplies it).
pub fn optimizer_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    loss: &Tensor,
    max_grad_norm: f64,
    is_last_step: bool,
) -> Result<()> {
    let mut grads = loss
        .backward()
        .map_err(|e| JammiError::FineTune(format!("Backward: {e}")))?;
    let step = optimizer.step_t() + 1;
    clip_and_step(
        optimizer,
        trainable_vars,
        &mut grads,
        max_grad_norm,
        DEFAULT_NORM_CHECK_INTERVAL,
        step,
        is_last_step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ParamsAdamW;
    use serial_test::serial;

    fn params_adamw(lr: f64) -> ParamsAdamW {
        ParamsAdamW {
            lr,
            ..Default::default()
        }
    }

    /// esc-182: [`sorted_trainable_vars`] must return the SAME `Var` sequence
    /// regardless of the order names were INSERTED into the `VarMap` — the
    /// property that removes the `HashMap`-iteration-order dependence
    /// `VarMap::all_vars()` has. Two `VarMap`s, built by inserting the SAME
    /// four names in OPPOSITE orders (a stand-in for two different
    /// processes' independently-randomized `HashMap` hashers producing
    /// different `all_vars()` orders for the identical variable set — this
    /// test cannot literally launch two processes, but insertion order is
    /// the one lever that visibly perturbs a `HashMap`'s internal bucket
    /// layout without needing to), must still walk their SHAPES (each name
    /// here gets a distinct element count, so the SHAPE sequence pins the
    /// NAME sequence indirectly) in IDENTICAL order.
    #[test]
    fn sorted_trainable_vars_is_independent_of_varmap_insertion_order() {
        let dev = Device::Cpu;
        let names_shapes: [(&str, usize); 4] =
            [("zed", 4), ("alpha", 1), ("mid.b", 3), ("mid.a", 2)];

        let forward = VarMap::new();
        for &(name, n) in &names_shapes {
            forward
                .get((n,), name, candle_nn::Init::Const(0.0), DType::F32, &dev)
                .unwrap();
        }
        let backward = VarMap::new();
        for &(name, n) in names_shapes.iter().rev() {
            backward
                .get((n,), name, candle_nn::Init::Const(0.0), DType::F32, &dev)
                .unwrap();
        }

        let forward_shapes: Vec<usize> = sorted_trainable_vars(&forward)
            .iter()
            .map(|v| v.as_tensor().dims1().unwrap())
            .collect();
        let backward_shapes: Vec<usize> = sorted_trainable_vars(&backward)
            .iter()
            .map(|v| v.as_tensor().dims1().unwrap())
            .collect();

        // The expected order is the NAME-sorted order:
        // "alpha" < "mid.a" < "mid.b" < "zed" -> shapes [1, 2, 3, 4].
        let expected = vec![1, 2, 3, 4];
        assert_eq!(
            forward_shapes, expected,
            "forward-inserted VarMap did not come back name-sorted"
        );
        assert_eq!(
            backward_shapes, expected,
            "reverse-inserted VarMap did not come back name-sorted"
        );
        assert_eq!(
            forward_shapes, backward_shapes,
            "sorted_trainable_vars must be independent of VarMap insertion order — a mutant \
             that reads VarMap::all_vars() directly (raw HashMap order) would very likely \
             diverge here"
        );
    }

    /// A single trainable `Var` with a gradient whose norm we control exactly
    /// (`grad_value` on every element of a `count`-length vector, so
    /// `||g|| = grad_value * sqrt(count)`).
    fn one_var_with_grad(grad_value: f32, count: usize) -> (Var, GradStore, Tensor) {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((count,), DType::F32, &dev).unwrap()).unwrap();
        // Build a real backward pass so `grads` is a genuine `GradStore`
        // (matches how every production caller obtains one), with a gradient
        // equal to `grad_value` everywhere: d/dw[sum(grad_value * w)] = grad_value.
        let coeff = Tensor::full(grad_value, (count,), &dev).unwrap();
        let loss = (w.as_tensor() * &coeff).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g = grads.get(w.as_tensor()).unwrap().clone();
        (w, grads, g)
    }

    #[test]
    fn below_max_norm_clip_is_bit_identical_to_no_clip() {
        // ||g|| = 0.5 * sqrt(4) = 1.0, max_norm = 10.0 → clip_coef clamps to
        // exactly 1.0, so the rescaled gradient must be BIT-IDENTICAL to the
        // unclipped one (x * 1.0 == x exactly for every finite f32 x).
        let (w, mut grads, g_before) = one_var_with_grad(0.5, 4);
        let total_norm = clip_gradients(std::slice::from_ref(&w), &mut grads, 10.0)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f32 = total_norm.to_scalar().unwrap();
        assert!(norm_val <= 10.0, "test setup: norm must be below max_norm");

        let g_after = grads.get(w.as_tensor()).unwrap();
        let before: Vec<f32> = g_before.to_vec1().unwrap();
        let after: Vec<f32> = g_after.to_vec1().unwrap();
        assert_eq!(
            before, after,
            "sub-threshold clip must not perturb a single bit"
        );
    }

    #[test]
    fn at_max_norm_boundary_coef_is_not_bit_identical_to_no_clip() {
        // total_norm == max_norm == 1.0 sits INSIDE the (max_norm - 1e-6,
        // max_norm] band where clip_coef is strictly < 1.0 — the doc's exact
        // predicate is `total_norm <= max_norm - 1e-6`, not the naive (and
        // FALSE) `total_norm <= max_norm` reading. Closed form pinned by hand
        // in f32: total_norm + 1e-6 = 1.0000009536743164, recip =
        // 0.9999990463256836 (bits `0x3f7ffff0`), * max_norm (1.0) = same,
        // min(1.0) = same (already < 1.0) → clip_coef = 0.9999990463256836.
        // `0.5 * clip_coef` is an EXACT `f32` operation (multiplying by 0.5 —
        // a power of two — only decrements the exponent; it introduces no
        // rounding for a normal, non-underflowing value), so the result is
        // pinned EXACTLY too: `0.49999952316284180` (bits `0x3efffff0`).
        // Pinned to the exact bit patterns (not a `<= N ulp` tolerance band):
        // this whole chain is deterministic `f32` arithmetic with no
        // data-dependent rounding, so there is no reason to accept slop here.
        let (w, mut grads, _g_before) = one_var_with_grad(0.5, 4);
        let max_norm = 1.0f64;
        let total_norm = clip_gradients(std::slice::from_ref(&w), &mut grads, max_norm)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f32 = total_norm.to_scalar().unwrap();
        assert_eq!(
            norm_val, 1.0,
            "test setup: total_norm must equal max_norm exactly to be on the boundary"
        );

        // The host-computed coefficient, `max_norm / (total_norm + eps)` —
        // `clip_gradients`'s current op sequence (one add, one division) —
        // pinned to its exact bits — independent corroboration of the doc's
        // closed form above. At `max_norm == 1.0` specifically this is
        // bit-identical to the PRIOR `(total_norm + eps).recip() * max_norm`
        // sequence too: `recip(d) == 1.0 / d` (Rust's own `f32::recip`), and
        // multiplying that by `max_norm == 1.0` is an exact no-op — so this
        // fixture cannot by itself distinguish the two op sequences; see
        // `clip_gradients`'s own doc for a `max_norm != 1.0` case where they
        // differ.
        let host_coef: f32 = max_norm as f32 / (norm_val + 1e-6);
        assert_eq!(
            host_coef.to_bits(),
            0x3f7f_fff0,
            "clip_coef at the boundary must be EXACTLY 0.9999990463256836 (bits 0x3f7ffff0), \
             got {host_coef} (bits {:#010x})",
            host_coef.to_bits()
        );

        let expected: f32 = 0.499_999_52; // == 0.49999952316284180, bits 0x3efffff0
        assert_eq!(expected.to_bits(), 0x3eff_fff0);
        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        for a in &after {
            assert_ne!(
                *a, 0.5,
                "boundary batch (total_norm == max_norm) must NOT be bit-identical to \
                 the unclipped gradient — a regression to the FALSE `total_norm <= \
                 max_norm` reading of the guarantee would make this pass"
            );
            assert_eq!(
                a.to_bits(),
                0x3eff_fff0,
                "expected the boundary batch scaled to EXACTLY {expected} (bits 0x3efffff0), \
                 got {a} (bits {:#010x})",
                a.to_bits()
            );
        }
    }

    #[test]
    fn clipping_batch_matches_host_reference_within_f32_ulps() {
        // ||g|| = 2.0 * sqrt(4) = 4.0, max_norm = 1.0 → clip_coef =
        // 1.0 / (4.0 + 1e-6) ≈ 0.25, well below the 1.0 clamp.
        let (w, mut grads, g_before) = one_var_with_grad(2.0, 4);
        let max_norm = 1.0f64;
        let total_norm = clip_gradients(std::slice::from_ref(&w), &mut grads, max_norm)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f32 = total_norm.to_scalar().unwrap();

        // Host reference computed from the SAME grads, torch's exact formula
        // (`torch/nn/utils/clip_grad.py`, see the module doc's citation):
        // max_norm / (total_norm + 1e-6), never clamped here since it is well
        // under 1.0.
        let host_coef = (max_norm / (norm_val as f64 + 1e-6)) as f32;
        let before: Vec<f32> = g_before.to_vec1().unwrap();
        let expected: Vec<f32> = before.iter().map(|x| x * host_coef).collect();
        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();

        // Tolerance: the device path now computes the coefficient with the
        // SAME rounding count as this host reference — one `f32` add (the
        // `+ 1e-6`), one `f32` division — but the two are still independent
        // implementations of that sequence (candle's `Affine`/`Div` kernels
        // vs this file's own `f64`-then-round-to-`f32` arithmetic), each
        // rounding by up to 0.5 ULP; up to ~1 ULP of drift in the coefficient
        // is expected from that, not from a bug; propagated through one more
        // multiply against `before`, 4 ULP absolute-relative tolerance covers
        // it with headroom (a looser bound than the ~1 ULP this specific op
        // sequence needs, kept as a round, documented ceiling rather than
        // fitted to what the test happened to observe).
        for (e, a) in expected.iter().zip(after.iter()) {
            let ulp = f32::EPSILON * e.abs().max(f32::MIN_POSITIVE);
            assert!(
                (e - a).abs() <= 4.0 * ulp.max(f32::EPSILON),
                "expected {e}, got {a} (device/host coefficient diverged beyond 4 ULP)"
            );
        }
    }

    /// Four trainable `Var`s of UNEQUAL shapes on `device`, each with a
    /// constant gradient (`d/dw[sum(c * w)] = c`), obtained through a real
    /// `backward` exactly as every production caller obtains its
    /// `GradStore`. The per-`Var` sums of squares are exact small integers
    /// (`2 + 24 + 1 + 45 = 72`), so every partial sum in the device fold is
    /// exactly representable — the fold's `+` is the ONLY thing that can
    /// combine them, and `+`→`*` (`2 * 24 * 1 * 45 = 2160`) or `+`→`-`
    /// (`2 - 24 - 1 - 45 = -68`, `sqrt` → NaN) is visible in the
    /// coefficient.
    fn four_vars_with_grads(device: &Device) -> (Vec<Var>, GradStore, Vec<Vec<f32>>) {
        let specs: [(&[usize], f32); 4] =
            [(&[2], 1.0), (&[3, 2], 2.0), (&[1, 4], 0.5), (&[5], 3.0)];
        let mut vars = Vec::new();
        let mut loss: Option<Tensor> = None;
        let mut expected_grads = Vec::new();
        for (dims, c) in specs {
            let w = Var::from_tensor(&Tensor::zeros(dims, DType::F32, device).unwrap()).unwrap();
            let coeff = Tensor::full(c, dims, device).unwrap();
            let term = (w.as_tensor() * &coeff).unwrap().sum_all().unwrap();
            loss = Some(match loss {
                None => term,
                Some(acc) => (&acc + &term).unwrap(),
            });
            expected_grads.push(vec![c; dims.iter().product()]);
            vars.push(w);
        }
        let grads = loss.unwrap().backward().unwrap();
        (vars, grads, expected_grads)
    }

    /// Bound derivation (f32, exact op sequence `clip_gradients` issues; see
    /// clause "every doc claim about float behaviour is derived from the op
    /// sequence in the tensor dtype"):
    ///  - per-`Var` `sqr` + `sum_all`: exact here (squares of 1, 2, 0.5, 3 and
    ///    their partial sums are all exactly representable integers or halves
    ///    below 2^24) — 0 ulp; in general `(n_i - 1) * 0.5` ulp per `Var`;
    ///  - the fold over 4 `Var`s: exact here (integer partial sums 2, 26, 27,
    ///    72) — 0 ulp; in general `(k - 1) * 0.5` ulp;
    ///  - `sqrt(72)`: correctly rounded, 0.5 ulp;
    ///  - `affine(1.0, 1e-6)` (`x * 1.0 + 1e-6`): `x * 1.0` is exact, one
    ///    rounding, 0.5 ulp;
    ///  - `div` (`max_norm / denom`, torch's own rounding count — see
    ///    [`clip_gradients`]'s own doc): 0.5 ulp;
    ///  - `minimum(1.0)`: exact (no rounding), 0 ulp;
    ///  - `broadcast_mul`: one rounding per element, 0.5 ulp.
    ///
    /// The host reference computes the same chain in f64 (error ≪ 1 f32 ulp)
    /// and rounds to f32 once (0.5 ulp). Total: ≤ 2 ulp device + 0.5 ulp
    /// host = 2.5 ulp of the result; asserted at 4 ulp (headroom for the
    /// ulp-of-result vs ulp-of-intermediate mismatch at exponent boundaries —
    /// not a fitted number).
    fn assert_multi_var_clip_matches_host(device: &Device) -> Vec<Vec<f32>> {
        let (vars, mut grads, before) = four_vars_with_grads(device);
        let max_norm = 1.0f64;
        let total_norm = clip_gradients(&vars, &mut grads, max_norm)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f32 = total_norm.to_scalar().unwrap();

        // Host reference: torch's formula over the SAME per-element grads,
        // `max_norm / (sqrt(Σ_i Σ g_i²) + 1e-6)`, in f64.
        let total_sq: f64 = before
            .iter()
            .flat_map(|g| g.iter())
            .map(|&x| (x as f64) * (x as f64))
            .sum();
        assert_eq!(
            total_sq, 72.0,
            "test setup: the exact-integer sum of squares"
        );
        let host_norm = total_sq.sqrt();
        let host_coef = max_norm / (host_norm + 1e-6);
        assert!(
            host_coef < 1.0 - 1e-3,
            "test setup: the coefficient must be STRICTLY below the 1.0 clamp, got {host_coef}"
        );
        let ulps = |e: f32, a: f32| -> f32 {
            let ulp = f32::EPSILON * e.abs().max(f32::MIN_POSITIVE);
            (e - a).abs() / ulp
        };
        assert!(
            ulps(host_norm as f32, norm_val) <= 4.0,
            "total_norm: host {host_norm} vs device {norm_val}"
        );

        let mut after = Vec::new();
        for (w, g_before) in vars.iter().zip(&before) {
            let g_after: Vec<f32> = grads
                .get(w.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (x, a) in g_before.iter().zip(&g_after) {
                let e = ((*x as f64) * host_coef) as f32;
                assert!(a.is_finite(), "clipped gradient must be finite, got {a}");
                assert!(
                    ulps(e, *a) <= 4.0,
                    "expected {e} (host), got {a} (device) — {} ulp apart",
                    ulps(e, *a)
                );
            }
            after.push(g_after);
        }
        after
    }

    /// The multi-`Var` fold oracle on CPU: `clip_gradients` over FOUR `Var`s
    /// of unequal shapes at a coefficient strictly below the `1.0` clamp,
    /// against the host reference `max_norm / (sqrt(Σ_i Σ g_i²) + 1e-6)` and
    /// every gradient `g_i * coef`, within the bound derived above.
    ///
    /// Mutation tried: `+` → `*` in the fold over per-`Var` squared sums
    /// (`&acc + &sq` → `&acc * &sq`) — RED (`total_sq` becomes `2160`, the
    /// coefficient a factor `sqrt(30)` too small). A single-`Var` fixture
    /// cannot see that mutant: the fold is never entered with a `Some(acc)`.
    #[test]
    fn multi_var_clip_matches_host_reference_on_cpu() {
        assert_multi_var_clip_matches_host(&Device::Cpu);
    }

    /// A HOST-side Rust scalar replica of [`clip_gradients`]'s EXACT formula
    /// and op ORDER — `sqr` + `sum_all` per Var (f32), fold across Vars
    /// (f32), `sqrt`, `affine(1.0, 1e-6)`, `div` (`max_norm / denom`,
    /// torch's own rounding count), `minimum(1.0)`, then `broadcast_mul` per
    /// element — written in plain f32 arithmetic so each step is the SAME
    /// single IEEE-754 rounding candle's own CPU backend performs
    /// (`Affine`'s `v * mul + add`, `Div`'s `v1 / v2`, `Minimum`'s `if v1 >
    /// v2 { v2 } else { v1 }` — candle-core-0.11.0 `cpu_backend/mod.rs`/
    /// `op.rs`). This is the
    /// "drop-in" oracle for the device-side clip REFACTOR itself (did moving
    /// the computation onto the device change the answer the "replace 225
    /// D2H syncs" lever actually needs to be equivalence-preserving?), never
    /// a torch-parity claim (that is `multi_var_clip_matches_host_reference_
    /// on_cpu`'s and `clipping_batch_matches_host_reference_within_f32_
    /// ulps`'s job, both against an independent host `f64` reference of
    /// torch's own formula) — so the fixture only needs the SAME formula computed twice through two
    /// independent implementations, at a scale where every partial sum is
    /// an exactly-representable integer, so NEITHER implementation's own
    /// internal reduction order (candle's SIMD-lane `vec_sum` vs this
    /// function's plain sequential fold) can perturb a single bit — making
    /// bit-identity an honest claim rather than an accident of a lucky seed.
    fn host_clip_gradients_f32(before: &[Vec<f32>], max_norm: f64) -> (f32, Vec<Vec<f32>>) {
        let mut total_sq: Option<f32> = None;
        for g in before {
            let sum_i: f32 = g.iter().map(|&x| x * x).fold(0.0f32, |acc, sq| acc + sq);
            total_sq = Some(match total_sq {
                None => sum_i,
                Some(acc) => acc + sum_i,
            });
        }
        let total_sq = total_sq.expect("at least one Var");
        let total_norm = total_sq.sqrt();
        let d = total_norm * 1.0f32 + 1e-6f32; // affine(1.0, 1e-6)
        let c = max_norm as f32 / d; // div: max_norm / denom, torch's own rounding count
        let coef = if c > 1.0 { 1.0f32 } else { c }; // minimum(1.0), candle's own predicate
        let after = before
            .iter()
            .map(|g| g.iter().map(|&x| x * coef).collect())
            .collect();
        (total_norm, after)
    }

    /// The "drop-in" acceptance leg (esc-182 finding item 2 / phase-6
    /// tautology close-out): [`clip_gradients`]'s device-side tensor-op path
    /// against [`host_clip_gradients_f32`]'s independent host-scalar replica
    /// of the IDENTICAL formula, over the SAME four-Var exact-integer
    /// fixture [`assert_multi_var_clip_matches_host`] uses — `total_norm`
    /// AND every clipped gradient BIT-IDENTICAL, not merely within a
    /// tolerance. This is the criterion the "replace 225 D2H syncs"
    /// optimization itself needs (device compute must reproduce the SAME
    /// formula, exactly, not merely something close to torch); CUDA is
    /// covered separately — `multi_var_clip_matches_host_reference_on_cuda_
    /// and_is_bit_identical_to_cpu` (below) holds CUDA to the SAME host
    /// reference AND to CPU bit-identity on this exact-integer fixture (see
    /// that test's own doc for why bit-identity is an honest claim there but
    /// would not be at production element counts, where CPU's and CUDA's
    /// reduction orders are mathematically but not bitwise equivalent).
    ///
    /// Mutation tried: `+` → `*` in the fold over per-`Var` squared sums
    /// (the same mutant `multi_var_clip_matches_host_reference_on_cpu`
    /// guards) — RED: the host replica (never touched by the mutant) still
    /// reports `total_sq == 72.0`, while the mutated device path reports
    /// `2160`, and `sqrt(2160) != sqrt(72)` is very much not bit-identical.
    #[test]
    fn clip_gradients_device_and_host_agree_bit_identically_on_cpu() {
        let (vars, mut grads, before) = four_vars_with_grads(&Device::Cpu);
        let max_norm = 1.0f64;

        let (host_norm, host_after) = host_clip_gradients_f32(&before, max_norm);

        let device_norm_tensor = clip_gradients(&vars, &mut grads, max_norm)
            .unwrap()
            .unwrap_clipped();
        let device_norm: f32 = device_norm_tensor.to_scalar().unwrap();

        assert_eq!(
            device_norm.to_bits(),
            host_norm.to_bits(),
            "device total_norm {device_norm} (bits {:#010x}) must be BIT-IDENTICAL to the host \
             replica's {host_norm} (bits {:#010x})",
            device_norm.to_bits(),
            host_norm.to_bits()
        );

        for (i, (w, host_g)) in vars.iter().zip(&host_after).enumerate() {
            let device_g: Vec<f32> = grads
                .get(w.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let device_bits: Vec<u32> = device_g.iter().map(|x| x.to_bits()).collect();
            let host_bits: Vec<u32> = host_g.iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                device_bits, host_bits,
                "Var {i}: device-clipped gradient must be BIT-IDENTICAL to the host replica's \
                 (device {device_g:?} vs host {host_g:?})"
            );
        }
    }

    /// A gradient that is NOT already `F32` (the AMP path this function's own
    /// doc calls out) must be upconverted before `.sqr()`/`.sum_all()` in the
    /// squared-sum loop AND before `broadcast_mul` in the rescale loop, then
    /// downconverted back to its OWN dtype before it is stored — every other
    /// test in this file only ever sees F32 gradients, so neither `else` arm
    /// (the actual conversions) is otherwise entered.
    ///
    /// This test found a REAL bug, not just a mutant-shaped one: before this
    /// fix, the rescale loop multiplied the gradient's ORIGINAL dtype
    /// against `coef` (always `F32`, folded from the upconverted squared
    /// sums) with no matching upconvert of its own — candle's
    /// `broadcast_mul` refuses mismatched dtypes, so `clip_gradients` errored
    /// on every call where any trainable `Var`'s gradient was not already
    /// `F32`, unconditionally, clipping disabled or not test coverage aside.
    ///
    /// Mutation tried: `replace == with !=` on the squared-sum loop's
    /// `g.dtype() == DType::F32` — the mutant keeps a non-F32 gradient's
    /// dtype through `.sqr()`/`.sum_all()` (BF16 arithmetic) instead of
    /// upconverting it, and routes an already-F32 gradient through a
    /// redundant (no-op, invisible) `to_dtype` instead. RED here: the
    /// untouched-dtype BF16 squared-sum and the second `Var`'s F32 squared-sum
    /// land in the SAME `Option<Tensor>` fold, and candle's `+` refuses
    /// mismatched dtypes — the fold itself errors, so this test's `.unwrap()`
    /// panics instead of returning `Ok`. The rescale loop's OWN `==`/dtype
    /// round-trip is pinned directly below by asserting the returned
    /// gradient's dtype.
    #[test]
    fn non_f32_gradient_is_upconverted_before_the_fold() {
        let dev = Device::Cpu;
        let w_bf16 = Var::from_tensor(&Tensor::zeros((4,), DType::F32, &dev).unwrap()).unwrap();
        let w_f32 = Var::from_tensor(&Tensor::zeros((4,), DType::F32, &dev).unwrap()).unwrap();
        let coeff_bf16 = Tensor::full(3.0f32, (4,), &dev).unwrap();
        let coeff_f32 = Tensor::full(2.0f32, (4,), &dev).unwrap();
        let loss = ((w_bf16.as_tensor() * &coeff_bf16)
            .unwrap()
            .sum_all()
            .unwrap()
            + (w_f32.as_tensor() * &coeff_f32).unwrap().sum_all().unwrap())
        .unwrap();
        let mut grads = loss.backward().unwrap();

        // Downcast the first Var's real gradient to BF16 in place — the
        // class of gradient the function's doc calls out. `3.0` round-trips
        // through BF16 exactly, so the host reference below needs no BF16
        // emulation of its own: it reads back the SAME bf16-then-f32 value
        // `clip_gradients` itself would compute.
        let g_f32 = grads.get(w_bf16.as_tensor()).unwrap().clone();
        let g_bf16 = g_f32.to_dtype(DType::BF16).unwrap();
        let g_bf16_roundtrip: Vec<f32> = g_bf16.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        assert_eq!(
            g_bf16_roundtrip,
            vec![3.0; 4],
            "test setup: 3.0 must round-trip through BF16 exactly"
        );
        grads.insert(w_bf16.as_tensor(), g_bf16);

        let vars = vec![w_bf16.clone(), w_f32.clone()];
        let total_norm = clip_gradients(&vars, &mut grads, 1.0)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f32 = total_norm.to_scalar().unwrap();

        let bf16_sq: f64 = 4.0 * (3.0f64 * 3.0);
        let f32_sq: f64 = 4.0 * (2.0f64 * 2.0);
        let host_norm = (bf16_sq + f32_sq).sqrt();
        assert!(
            (norm_val as f64 - host_norm).abs() < 1e-3,
            "expected {host_norm} (host), got {norm_val} (device)"
        );

        // The rescale loop's own dtype contract: the clipped gradient must
        // come back in the SAME dtype it went in with, not silently promoted
        // to F32 by the upconvert-for-`broadcast_mul` round trip.
        let clipped_bf16 = grads.get(w_bf16.as_tensor()).unwrap();
        assert_eq!(
            clipped_bf16.dtype(),
            DType::BF16,
            "a BF16 gradient must come back BF16, not silently promoted to F32"
        );
        let clipped_f32 = grads.get(w_f32.as_tensor()).unwrap();
        assert_eq!(clipped_f32.dtype(), DType::F32);
    }

    /// Acquire CUDA device 0, or skip — unless `JAMMI_REQUIRE_CUDA` is set
    /// (the pod session that is this leg's landing proof), in which case a
    /// missing device is a failure, never a silent skip.
    fn cuda_device(test: &str) -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(d) => Some(d),
            Err(e) => {
                if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                    panic!(
                        "{test}: JAMMI_REQUIRE_CUDA is set but no CUDA device could be \
                         acquired — a silent skip is not acceptable here: {e}"
                    );
                }
                eprintln!("{test}: skipping — no CUDA device available ({e})");
                None
            }
        }
    }

    /// The SAME oracle body on CUDA, plus CPU/CUDA BIT-identity of the
    /// clipped gradients in this finite cell. Why bit-identity is the right
    /// expectation and not a tolerance: every partial sum in the fixture is
    /// an exactly-representable integer, so the per-`Var` `sum_all` and the
    /// fold are exact on both backends regardless of their reduction order;
    /// `sqrt`/`div` are correctly rounded on both (nvcc's default
    /// `-prec-sqrt=true`/`-prec-div=true`; candle-kernels does not build with
    /// `-use_fast_math`); `affine(1.0, 1e-6)` is `x * 1.0 + 1e-6`, where
    /// `x * 1.0` is exact, so an FMA contraction on the GPU rounds once to
    /// the same value the CPU's two ops produce; `Tensor::full` materializing
    /// `max_norm` is a pure memory fill (`const_set`, no arithmetic — the
    /// SAME bit pattern on both backends) and `minimum(1.0)` are exact; and
    /// the final `broadcast_mul` is one rounding per element on both.
    ///
    /// The NaN cell is a DELIBERATE CPU≠CUDA divergence, not covered by the
    /// identity assertion: candle-core-0.11.0 `src/op.rs:460` implements
    /// `Minimum` as `if v1 > v2 { v2 } else { v1 }` — a NaN `v1` fails the
    /// comparison and is returned, so on CPU a NaN coefficient stays NaN and
    /// poisons EVERY gradient — while candle-kernels-0.11.0
    /// `src/cuda_utils.cuh:144` implements it as `fminf`, IEEE `minNum`,
    /// which returns the non-NaN operand: on CUDA the coefficient clamps to
    /// `1.0` and only the gradient that was already NaN stays NaN. Neither
    /// arm changes what `refuse_nonfinite_norm` sees — `total_norm` itself is
    /// NaN on both, computed before the clamp — so the typed refusal is
    /// device-independent ON A CADENCE STEP. OFF cadence, `clip_and_step`
    /// skips `refuse_nonfinite_norm` and DOES consume the rescaled store:
    /// `optimizer.step` runs over it, so on CPU every parameter is poisoned
    /// at once while on CUDA only the already-NaN gradient's parameter is
    /// — a real CPU≠CUDA divergence in the parameters until the next
    /// cadence point (step 1, every `check_every_n_steps`, or the last
    /// step) refuses the norm, with `TrainingLoop::refuse_nonfinite_params`'
    /// epoch-boundary read as the backstop that keeps either arm out of a
    /// checkpoint. `nan_gradient_poisons_every_gradient_through_the_cpu_
    /// minimum` pins the CPU arm; this leg pins the CUDA arm when a device
    /// is present.
    #[test]
    fn multi_var_clip_matches_host_reference_on_cuda_and_is_bit_identical_to_cpu() {
        let Some(cuda) = cuda_device("multi_var_clip_cuda") else {
            return;
        };
        let after_cuda = assert_multi_var_clip_matches_host(&cuda);
        let after_cpu = assert_multi_var_clip_matches_host(&Device::Cpu);
        for (i, (c, g)) in after_cuda.iter().zip(&after_cpu).enumerate() {
            let c_bits: Vec<u32> = c.iter().map(|x| x.to_bits()).collect();
            let g_bits: Vec<u32> = g.iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                c_bits, g_bits,
                "Var {i}: CUDA-clipped gradient must be bit-identical to the CPU-clipped one \
                 in the finite cell (cuda {c:?} vs cpu {g:?})"
            );
        }

        // The NaN cell's CUDA arm: `fminf(NaN, 1.0) == 1.0`, so the finite
        // Var's gradient passes through UNCHANGED (bit-identical to before)
        // while `total_norm` is still NaN for the refusal to see.
        let (vars, mut grads, before) = one_nan_var_one_finite_var(&cuda);
        let total_norm = clip_gradients(&vars, &mut grads, 1.0)
            .unwrap()
            .unwrap_clipped();
        assert!(total_norm.to_scalar::<f32>().unwrap().is_nan());
        let finite_after: Vec<f32> = grads.get(vars[1].as_tensor()).unwrap().to_vec1().unwrap();
        assert_eq!(
            finite_after, before,
            "CUDA's fminf clamps a NaN coefficient to 1.0: the finite gradient is untouched"
        );
    }

    /// Two `Var`s: the first with an all-NaN gradient, the second with a
    /// finite one (`2.0` × 3). Returns the finite gradient's pre-clip values.
    fn one_nan_var_one_finite_var(device: &Device) -> (Vec<Var>, GradStore, Vec<f32>) {
        let w_nan = Var::from_tensor(&Tensor::zeros((2,), DType::F32, device).unwrap()).unwrap();
        let w_fin = Var::from_tensor(&Tensor::zeros((3,), DType::F32, device).unwrap()).unwrap();
        let nan_coeff = Tensor::full(f32::NAN, (2,), device).unwrap();
        let fin_coeff = Tensor::full(2.0f32, (3,), device).unwrap();
        let loss = ((w_nan.as_tensor() * &nan_coeff).unwrap().sum_all().unwrap()
            + (w_fin.as_tensor() * &fin_coeff).unwrap().sum_all().unwrap())
        .unwrap();
        let grads = loss.backward().unwrap();
        let before: Vec<f32> = grads.get(w_fin.as_tensor()).unwrap().to_vec1().unwrap();
        assert_eq!(before, vec![2.0; 3], "test setup: the finite gradient");
        (vec![w_nan, w_fin], grads, before)
    }

    /// The NaN cell's CPU arm (see the CUDA leg's doc for the pair): a NaN
    /// in ANY `Var`'s gradient makes `total_norm` NaN, the coefficient NaN,
    /// and — through candle's CPU `Minimum` (`if v1 > v2 { v2 } else
    /// { v1 }`, candle-core-0.11.0 `src/op.rs:460`) — the clamp returns the
    /// NaN, so every OTHER gradient is rescaled to NaN too. The refusal is
    /// what keeps that from mattering; this pins the arm the doc names.
    #[test]
    fn nan_gradient_poisons_every_gradient_through_the_cpu_minimum() {
        let (vars, mut grads, _before) = one_nan_var_one_finite_var(&Device::Cpu);
        let total_norm = clip_gradients(&vars, &mut grads, 1.0)
            .unwrap()
            .unwrap_clipped();
        assert!(total_norm.to_scalar::<f32>().unwrap().is_nan());
        let finite_after: Vec<f32> = grads.get(vars[1].as_tensor()).unwrap().to_vec1().unwrap();
        assert!(
            finite_after.iter().all(|x| x.is_nan()),
            "CPU Minimum returns the NaN coefficient: the finite gradient is poisoned, got {finite_after:?}"
        );
    }

    // The five tests below all read or mutate the process-wide, test-only
    // `SYNC_READ_COUNT` (directly, or indirectly by calling
    // `refuse_nonfinite_norm`). `cargo test` runs tests in parallel threads
    // within the SAME process, so an unmarked pair racing on that counter
    // would make these tests flaky; `#[serial(..)]` under a shared key forces
    // them to run one at a time relative to each other (never relative to
    // the rest of the file, which does not touch the counter).
    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn refuse_nonfinite_norm_is_red_when_the_check_is_removed() {
        // A NaN total norm must produce a typed error when
        // `refuse_nonfinite_norm` runs. The mutation this guards against is
        // deleting the `is_finite()` guard inside that function — simulate
        // it by asserting the *positive* behavior here, and rely on
        // `clip_and_step_skips_the_check_off_cadence` below to prove the
        // gating actually suppresses the call (so "remove the check" and
        // "remove the call" are both covered).
        let dev = Device::Cpu;
        let nan_norm = Tensor::new(f32::NAN, &dev).unwrap();
        let err = refuse_nonfinite_norm(&nan_norm, 7).unwrap_err().to_string();
        assert!(err.contains("non-finite"), "got: {err}");
        assert!(err.contains("step 7"), "must name the step, got: {err}");

        let inf_norm = Tensor::new(f32::INFINITY, &dev).unwrap();
        assert!(refuse_nonfinite_norm(&inf_norm, 1).is_err());

        let finite_norm = Tensor::new(3.0f32, &dev).unwrap();
        assert!(refuse_nonfinite_norm(&finite_norm, 1).is_ok());
    }

    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_gradients_never_reads_the_norm_back() {
        // Structural proxy for "no D2H sync on the per-step path" (a CPU
        // test cannot observe a CUDA sync directly): SYNC_READ_COUNT only
        // moves inside `refuse_nonfinite_norm`, so a bare `clip_gradients`
        // call — the thing the per-micro-batch step actually runs — must
        // leave it untouched.
        let before = sync_read_count();
        let (w, mut grads, _g) = one_var_with_grad(3.0, 4);
        // `unwrap_clipped` only pattern-matches the arm — no host read.
        let _norm_on_device = clip_gradients(&[w], &mut grads, 1.0)
            .unwrap()
            .unwrap_clipped();
        assert_eq!(
            sync_read_count(),
            before,
            "clip_gradients must not perform any device→host read"
        );
    }

    /// The CUDA leg of the same structural-proxy claim (esc-182 finding item
    /// 3 / phase-6 tautology close-out): `SYNC_READ_COUNT` is the only way a
    /// test can observe "no device→host read happened" on ANY backend — there
    /// is no CUDA stream a `#[test]` can inspect directly — so this counts
    /// `to_scalar`/`to_vec` calls through the exact SAME structural proxy
    /// [`clip_gradients_never_reads_the_norm_back`] uses, on a real CUDA
    /// device, across the FULL timed path (`clip_and_step`, not a bare
    /// `clip_gradients`): a bare clip (0 reads), an OFF-cadence `clip_and_
    /// step` (0 reads — the cadence gate must suppress the read entirely,
    /// not merely skip acting on it), and an ON-cadence `clip_and_step`
    /// (EXACTLY 1 read — [`refuse_nonfinite_norm`] is the only permitted
    /// device→host call on this path, never zero and never more than one
    /// per cadence-gated step).
    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_gradients_never_reads_the_norm_back_on_cuda() {
        let Some(cuda) = cuda_device("clip_gradients_sync_cuda") else {
            return;
        };

        // Bare clip_gradients: 0 reads.
        let before = sync_read_count();
        let (vars, mut grads, _before) = four_vars_with_grads(&cuda);
        let _norm_on_device = clip_gradients(&vars, &mut grads, 1.0)
            .unwrap()
            .unwrap_clipped();
        assert_eq!(
            sync_read_count(),
            before,
            "CUDA: clip_gradients must not perform any device→host read"
        );

        // OFF-cadence clip_and_step: still 0 reads, even with a fresh
        // GradStore and a real AdamW step riding along.
        let (vars2, mut grads2, _before2) = four_vars_with_grads(&cuda);
        let mut optimizer = AdamW::new(vars2.clone(), params_adamw(0.1)).unwrap();
        let before_off = sync_read_count();
        clip_and_step(&mut optimizer, &vars2, &mut grads2, 1.0, 10, 3, false).unwrap();
        assert_eq!(
            sync_read_count(),
            before_off,
            "CUDA: an off-cadence clip_and_step must not read the norm back at all"
        );

        // ON-cadence clip_and_step: EXACTLY 1 read (refuse_nonfinite_norm),
        // never more.
        let (vars3, mut grads3, _before3) = four_vars_with_grads(&cuda);
        let mut optimizer3 = AdamW::new(vars3.clone(), params_adamw(0.1)).unwrap();
        let before_on = sync_read_count();
        clip_and_step(&mut optimizer3, &vars3, &mut grads3, 1.0, 10, 10, false).unwrap();
        assert_eq!(
            sync_read_count(),
            before_on + 1,
            "CUDA: an on-cadence clip_and_step must read the norm back EXACTLY once, through \
             refuse_nonfinite_norm — never zero (the finite check would be silently skipped) \
             and never more than one (a second sync would defeat the whole optimization this \
             module exists for)"
        );
    }

    /// Build a `Var`/`grads` pair whose gradient is NaN everywhere (mirrors
    /// the setup shared by every cadence test below).
    fn one_var_with_nan_grad() -> (Var, GradStore, AdamW) {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &dev).unwrap()).unwrap();
        let coeff = Tensor::new(f32::NAN, &dev)
            .unwrap()
            .broadcast_as((2,))
            .unwrap();
        let loss = (w.as_tensor() * &coeff).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let optimizer = AdamW::new(vec![w.clone()], params_adamw(0.1)).unwrap();
        (w, grads, optimizer)
    }

    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_and_step_skips_the_check_off_cadence() {
        // check_every_n_steps = 10, step = 3, is_last_step = false → not step
        // 1, not a multiple of 10, not the run's last step, so the check must
        // not run even though the norm happens to be NaN — this is the
        // "acceptable off the critical path" contract: the sync is opt-in per
        // step, gated by cadence, not by the norm's value.
        let (w, mut grads, mut optimizer) = one_var_with_nan_grad();
        let before = sync_read_count();
        clip_and_step(&mut optimizer, &[w], &mut grads, 1.0, 10, 3, false).unwrap();
        assert_eq!(
            sync_read_count(),
            before,
            "off-cadence step must not read the norm back at all"
        );
    }

    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_and_step_refuses_a_nonfinite_norm_on_cadence() {
        let (w, mut grads, mut optimizer) = one_var_with_nan_grad();
        let err = clip_and_step(&mut optimizer, &[w], &mut grads, 1.0, 10, 10, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-finite"), "got: {err}");
    }

    /// `clip_and_step` must WARN (a `tracing::warn!` — a COUNTED FACT, not a
    /// hard refusal, see `clip_and_step`'s own doc for why a hard `Err` here
    /// broke every legitimate loss that does not route through every
    /// trainable `Var` for a given batch) when `trainable_vars` is
    /// NON-EMPTY but not one of them has a gradient present — the
    /// AMBIGUOUS `ClipOutcome::NoGradients` case (phase-6 class-census
    /// addition #2, ledger row 215; and the round-2 pivot on the SAME
    /// finding once `ft_correctness_sweep.rs`/`ft_determinism.rs`/
    /// `encoder_adapters.rs` proved a hard refusal there is not viable).
    /// Before `ClipOutcome` existed, `clip_gradients` returned the SAME
    /// `None` for this case as it did for `max_grad_norm <= 0.0`, so
    /// `clip_and_step` silently skipped BOTH the clip AND the non-finite
    /// check with NO trace of it anywhere — indistinguishable from an
    /// intentional "clipping is off" configuration AND unobservable. The
    /// step must still SUCCEED (an `AdamW` step over an empty `GradStore`
    /// is a harmless no-op) — this is a warning, not a failure. Off-cadence
    /// OR on-cadence: the warning is not gated by the norm-check cadence at
    /// all (there is no norm tensor to even read back).
    #[test]
    fn clip_and_step_warns_no_gradients_with_nonempty_trainable_vars() {
        use std::io;
        use std::sync::Mutex;
        use tracing::subscriber::DefaultGuard;
        use tracing_subscriber::fmt::MakeWriter;

        #[derive(Clone)]
        struct BufferWriter(std::sync::Arc<Mutex<Vec<u8>>>);
        impl io::Write for BufferWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        impl<'w> MakeWriter<'w> for BufferWriter {
            type Writer = BufferWriter;
            fn make_writer(&'w self) -> Self::Writer {
                self.clone()
            }
        }

        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((4,), DType::F32, &dev).unwrap()).unwrap();
        let mut empty_grads = GradStore::default();
        let mut optimizer = AdamW::new(vec![w.clone()], params_adamw(0.1)).unwrap();

        let buffer = std::sync::Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(BufferWriter(buffer.clone()))
            .with_ansi(false)
            .finish();
        let _guard: DefaultGuard = tracing::subscriber::set_default(subscriber);

        // Off-cadence (step=3, interval=10): still must warn, and must
        // still SUCCEED.
        clip_and_step(&mut optimizer, &[w], &mut empty_grads, 1.0, 10, 3, false)
            .expect("NoGradients with a non-empty trainable_vars must WARN, never refuse");

        let logs = String::from_utf8(buffer.lock().unwrap().clone()).expect("utf-8 logs");
        assert!(
            logs.contains("NONE had a gradient") && logs.contains("WARN"),
            "expected a WARN-level counted-fact log line, got: {logs}"
        );
    }

    /// The counterpart: an EMPTY `trainable_vars` is the UNAMBIGUOUSLY
    /// benign reading of `NoGradients` (nothing was ever asked to be
    /// clipped) and `clip_and_step` must not even warn about it — matching
    /// the pre-esc-182 behavior for this one case (an eval-only /
    /// all-adapters-frozen-by-design call site).
    #[test]
    fn clip_and_step_tolerates_no_gradients_with_empty_trainable_vars() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((4,), DType::F32, &dev).unwrap()).unwrap();
        let mut grads = GradStore::default();
        let mut optimizer = AdamW::new(vec![w], params_adamw(0.1)).unwrap();
        clip_and_step(&mut optimizer, &[], &mut grads, 1.0, 10, 3, false)
            .expect("an empty trainable_vars must not be refused");
    }

    /// Why `step == 1` is its own arm: with the modulo cadence alone
    /// (`check_every_n_steps > 0 && step.is_multiple_of(check_every_n_steps)`
    /// as the WHOLE gate), `step == 1` against `check_every_n_steps == 50` is
    /// never checked. Mutation tried: delete the `step == 1 ||` disjunct from
    /// `clip_and_step`'s `on_cadence` — this test goes red (an `Ok` where it
    /// expects `Err`).
    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_and_step_always_checks_step_one_regardless_of_cadence() {
        let (w, mut grads, mut optimizer) = one_var_with_nan_grad();
        let err = clip_and_step(
            &mut optimizer,
            &[w],
            &mut grads,
            1.0,
            DEFAULT_NORM_CHECK_INTERVAL,
            1,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("non-finite"), "got: {err}");
    }

    /// Why `is_last_step` is its own arm: a run shorter than
    /// `check_every_n_steps` steps end to end (e.g. 12 steps against the
    /// default interval of 50) never hits a multiple of the interval and, on
    /// the modulo cadence alone, would train an entire run — including a NaN
    /// on its very last step — without ever calling `refuse_nonfinite_norm`. Mutation tried: delete the `|| is_last_step`
    /// disjunct — this test goes red.
    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_and_step_checks_the_last_step_of_a_short_run() {
        let (w, mut grads, mut optimizer) = one_var_with_nan_grad();
        // step = 12, well short of the default interval (50) and not a
        // multiple of it — only `is_last_step = true` should trigger the check.
        let err = clip_and_step(
            &mut optimizer,
            &[w],
            &mut grads,
            1.0,
            DEFAULT_NORM_CHECK_INTERVAL,
            12,
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("non-finite"), "got: {err}");
    }

    #[test]
    fn clip_does_not_entangle_the_var_in_a_new_backward_graph() {
        // The clip must operate on the FETCHED GRADIENT tensor, never on
        // `var` itself — a `Var` is a graph leaf, and calling ops directly on
        // it (instead of on `grads.get(var)`) would grow its dependency
        // graph every step. `sorted_nodes()` on the Var, called as if it were
        // itself the tape's output, must report only the Var (length 1)
        // before and after the clip.
        let (w, mut grads, _g) = one_var_with_grad(2.0, 4);
        let before = w.as_tensor().sorted_nodes().len();
        let _norm_on_device = clip_gradients(std::slice::from_ref(&w), &mut grads, 1.0)
            .unwrap()
            .unwrap_clipped();
        let after = w.as_tensor().sorted_nodes().len();
        assert_eq!(before, 1, "a Var is its own sole node");
        assert_eq!(
            after, before,
            "clip_gradients must not grow the Var's graph"
        );
    }

    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn max_norm_le_zero_disables_clipping_and_returns_none() {
        let (w, mut grads, g_before) = one_var_with_grad(5.0, 4);
        let before = sync_read_count();
        let result = clip_gradients(std::slice::from_ref(&w), &mut grads, 0.0).unwrap();
        assert!(
            matches!(result, ClipOutcome::Disabled),
            "max_norm <= 0.0 must be the Disabled outcome, never NoGradients (this Var DID have \
             a gradient present — collapsing the two would re-introduce the ambiguity \
             ClipOutcome exists to remove), got {result:?}"
        );
        assert_eq!(sync_read_count(), before, "disabled clipping must not sync");
        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        let before_vals: Vec<f32> = g_before.to_vec1().unwrap();
        assert_eq!(before_vals, after);
    }

    /// The NEW half of the ClipOutcome split (phase-6 class-census addition
    /// #2, ledger row 215): a NON-EMPTY `trainable_vars` where NOT ONE `Var`
    /// has a gradient present in `grads` must come back `NoGradients`, never
    /// silently collapsed into the SAME `None` `Disabled` produces — that
    /// collapse is exactly what let a detached-graph / all-frozen-adapter
    /// bug hide behind an "operator turned clipping off" reading.
    #[test]
    fn no_gradient_present_is_distinct_from_disabled() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((4,), DType::F32, &dev).unwrap()).unwrap();
        // A GradStore that was never populated for `w` at all — the
        // "detached graph" / "forgot to call backward" shape.
        let mut empty_grads = GradStore::default();
        let result = clip_gradients(std::slice::from_ref(&w), &mut empty_grads, 1.0).unwrap();
        assert!(
            matches!(result, ClipOutcome::NoGradients),
            "a non-empty trainable_vars with no gradient present must be NoGradients, got \
             {result:?}"
        );

        // An EMPTY trainable_vars is the one genuinely benign reading —
        // still NoGradients (nothing WAS clipped), but `clip_and_step`
        // treats this arm differently based on emptiness, not on the
        // ClipOutcome variant alone (see that function's own doc).
        let result_empty = clip_gradients(&[], &mut empty_grads, 1.0).unwrap();
        assert!(matches!(result_empty, ClipOutcome::NoGradients));
    }

    /// The non-finite `max_norm` cell, NaN: `max_norm.is_nan()` makes
    /// `max_norm <= 0.0` `false` (family F: a NaN comparison is always
    /// false, in EITHER direction), so a NaN `max_norm` would fall through
    /// the disable-clipping guard and into the clip computation — silently
    /// scaling every gradient by a NaN coefficient forever, invisible to
    /// [`refuse_nonfinite_norm`] because `total_norm` (what that function
    /// checks) is computed from the GRADIENTS, not from `max_norm`, and stays
    /// finite throughout. Mutation tried: delete the `!max_norm.is_finite()`
    /// guard — this test goes red (`Ok` instead of the expected `Err`, and
    /// the returned gradient is silently all-NaN).
    #[test]
    fn clip_gradients_refuses_nan_max_norm() {
        let (w, mut grads, _g_before) = one_var_with_grad(2.0, 4);
        let err = clip_gradients(std::slice::from_ref(&w), &mut grads, f64::NAN)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("finite"),
            "expected a typed non-finite max_norm refusal, got: {err}"
        );
    }

    /// The non-finite `max_norm` cell, ±inf: `+inf`/`-inf` are equally
    /// non-finite `max_norm` values — `f64::INFINITY <= 0.0` is `false` too,
    /// so this would also fall through the same guard (its
    /// arithmetic happens to clamp back to a merely-wasteful `coef == 1.0`
    /// rather than corrupting every gradient, but it is still a non-finite
    /// tuning parameter this abstraction's contract should never silently
    /// accept). Mutation tried: same as above — this test goes red.
    #[test]
    fn clip_gradients_refuses_infinite_max_norm() {
        let (w, mut grads, _g_before) = one_var_with_grad(2.0, 4);
        let err_pos = clip_gradients(std::slice::from_ref(&w), &mut grads, f64::INFINITY)
            .unwrap_err()
            .to_string();
        assert!(err_pos.contains("finite"), "got: {err_pos}");

        let err_neg = clip_gradients(std::slice::from_ref(&w), &mut grads, f64::NEG_INFINITY)
            .unwrap_err()
            .to_string();
        assert!(err_neg.contains("finite"), "got: {err_neg}");
    }

    /// The near-zero-`total_norm` formula discriminator (RED control —
    /// acceptance criterion 3: "the old conditional/no-eps formula fails").
    /// A small, DETERMINISTIC (no RNG, no production-scale apparatus)
    /// fixture whose `total_norm` sits at the `~1e-9` scale — deep enough
    /// below torch's `1e-6` epsilon that the epsilon term DOMINATES the
    /// denominator, the exact regime where the pre-PR conditional,
    /// no-epsilon formula (`clip_coef = max_norm / total_norm`, applied only
    /// `if total_norm > max_norm`) and torch's own formula (`clip_coef =
    /// max_norm / (total_norm + 1e-6)`, applied unconditionally) diverge by
    /// ORDERS OF MAGNITUDE rather than a few ulp — no probabilistic bound
    /// needed to see it. `max_norm` is set to a FIXED FRACTION of
    /// `total_norm` (`0.5 * total_norm`), so the OLD formula's coefficient
    /// is EXACTLY `0.5` by construction, independent of scale (`max_norm /
    /// total_norm` cancels the common factor); the NEW formula's
    /// coefficient instead collapses toward `total_norm / 1e-6` as
    /// `total_norm -> 0` — a scale-DEPENDENT divergence the old formula
    /// cannot see at all.
    #[test]
    fn old_no_epsilon_formula_diverges_from_device_at_near_zero_norm() {
        let (w, mut grads, _g_before) = one_var_with_grad(1e-9, 4);
        // ||g|| = grad_value * sqrt(count), exact by this fixture's own
        // construction (see `one_var_with_grad`'s doc).
        let total_norm_host = 1e-9f64 * 4.0f64.sqrt();
        let max_norm = total_norm_host * 0.5; // deep active regime, not boundary-adjacent

        let old_coef = max_norm / total_norm_host; // pre-PR: no epsilon
        assert!(
            (old_coef - 0.5).abs() < 1e-9,
            "test setup: old_coef must be exactly 0.5 by this fixture's construction, got \
             {old_coef}"
        );
        let new_coef = max_norm / (total_norm_host + 1e-6); // torch's own formula

        let relative_diff = (old_coef - new_coef).abs() / old_coef;
        assert!(
            relative_diff > 0.9,
            "RED control did not fire: the old no-epsilon formula ({old_coef}) should diverge \
             from torch's own formula ({new_coef}) by more than 90% at this near-zero \
             total_norm ({total_norm_host}), got only {relative_diff:.3e} relative difference — \
             the fixture's total_norm needs to be smaller relative to the 1e-6 epsilon"
        );

        // The DEVICE path under test must match the NEW formula, not the
        // OLD one — assert every clipped element lands closer to the
        // new-formula prediction than to the old-formula one.
        let total_norm = clip_gradients(std::slice::from_ref(&w), &mut grads, max_norm)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f64 = total_norm.to_scalar::<f32>().unwrap() as f64;
        assert!(
            (norm_val - total_norm_host).abs() / total_norm_host < 1e-3,
            "test setup: device total_norm {norm_val} must match host {total_norm_host}"
        );

        let expected_new = (1e-9f64 * new_coef) as f32;
        let expected_old = (1e-9f64 * old_coef) as f32;
        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        for a in &after {
            let dist_to_new = (*a as f64 - expected_new as f64).abs();
            let dist_to_old = (*a as f64 - expected_old as f64).abs();
            assert!(
                dist_to_new < dist_to_old,
                "device-clipped gradient {a} must land closer to the NEW (torch) formula's \
                 value {expected_new} than to the OLD (pre-PR, no-epsilon) formula's value \
                 {expected_old} — the code under test must match torch at this near-zero \
                 total_norm, not the pre-PR formula it replaced"
            );
        }
    }

    /// The rounding-COUNT fix ([`clip_gradients`]'s doc: `denom.div(&max_norm_t)`
    /// replacing an earlier `recip().affine(max_norm, 0.0)`) is INVISIBLE at
    /// `max_norm == 1.0` — the shipped default (`jammi_wire::fine_tune::
    /// FineTuneConfig::default().max_grad_norm`) — because `x * 1.0` is an
    /// exact no-op in `f32`: `recip(d) * 1.0 == recip(d) == 1.0 / d ==
    /// max_norm / d` bit-for-bit whenever `max_norm == 1.0`, so every
    /// `max_norm == 1.0` fixture in this file (`below_max_norm_clip_is_bit_
    /// identical_to_no_clip`, `at_max_norm_boundary_coef_is_not_bit_
    /// identical_to_no_clip`, `clipping_batch_matches_host_reference_within_
    /// f32_ulps`, `multi_var_clip_matches_host_reference_on_cpu`,
    /// `clip_gradients_device_and_host_agree_bit_identically_on_cpu`) passes
    /// UNCHANGED before and after the fix — none of them can be cited as
    /// evidence the rounding-count fix does anything. This fixture picks a
    /// `max_norm != 1.0` (`7.5`) and a `total_norm` (found by an offline
    /// search, `1799.50146484375`, chosen so `sqr` + `sum_all` + `sqrt`
    /// round-trips to itself exactly — no per-element rounding to reason
    /// through) where the OLD `recip`-then-`affine` sequence and the NEW
    /// single `div` genuinely disagree: the coefficient differs by exactly 1
    /// ULP (`0x3af8894e` vs `0x3af8894d`), and — unlike most such 1-ULP
    /// coefficient differences, which the final `broadcast_mul`'s own
    /// rounding absorbs — this one SURVIVES into the clipped gradient itself
    /// (`7.500000476837158` vs `7.499999523162842`, bits `0x40f00001` vs
    /// `0x40efffff`). Pinned to exact bits, not a tolerance: this is
    /// deterministic `f32` arithmetic with no data-dependent rounding.
    ///
    /// Mutation tried: revert `clip_gradients`'s `div` back to `recip()` +
    /// `affine(max_norm, 0.0)` — RED (the device path lands on the OLD
    /// bits, not the NEW ones this test pins).
    #[test]
    fn rounding_count_fix_changes_the_result_only_when_max_norm_is_not_one() {
        let grad_value = 1_799.501_5_f32; // == 1799.50146484375 exactly (see doc)
        let (w, mut grads, _g_before) = one_var_with_grad(grad_value, 1);
        let max_norm = 7.5f64;

        let total_norm = clip_gradients(std::slice::from_ref(&w), &mut grads, max_norm)
            .unwrap()
            .unwrap_clipped();
        let norm_val: f32 = total_norm.to_scalar().unwrap();
        assert_eq!(
            norm_val.to_bits(),
            grad_value.to_bits(),
            "test setup: a single-element gradient's total_norm must round-trip to the \
             gradient's own value exactly"
        );

        // Host replicas of both op sequences, over the SAME total_norm/max_norm.
        let d = norm_val.mul_add(1.0, 1e-6); // affine(1.0, 1e-6) -- `x * 1.0` exact, one rounding
        let old_coef = (1.0f32 / d) * max_norm as f32; // recip() then affine(max_norm, 0.0)
        let new_coef = max_norm as f32 / d; // div: max_norm / denom
        assert_ne!(
            old_coef.to_bits(),
            new_coef.to_bits(),
            "test setup: at max_norm = 7.5 (!= 1.0) the two op sequences must disagree by \
             construction — old {old_coef} (bits {:#010x}) vs new {new_coef} (bits {:#010x})",
            old_coef.to_bits(),
            new_coef.to_bits()
        );
        assert!(
            old_coef < 1.0 && new_coef < 1.0,
            "test setup: below the clamp"
        );

        let expected_new = grad_value * new_coef;
        let expected_old = grad_value * old_coef;
        assert_ne!(
            expected_old.to_bits(),
            expected_new.to_bits(),
            "test setup: the 1-ULP coefficient difference must survive the final multiply \
             here — old {expected_old} (bits {:#010x}) vs new {expected_new} (bits {:#010x})",
            expected_old.to_bits(),
            expected_new.to_bits()
        );

        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(
            after[0].to_bits(),
            expected_new.to_bits(),
            "clip_gradients must match the NEW (single-division, torch-rounding-count) \
             formula's exact bits {expected_new} (bits {:#010x}), got {} (bits {:#010x}) — the \
             OLD (recip-then-multiply) formula would have given {expected_old} (bits \
             {:#010x})",
            expected_new.to_bits(),
            after[0],
            after[0].to_bits(),
            expected_old.to_bits()
        );
    }
}
