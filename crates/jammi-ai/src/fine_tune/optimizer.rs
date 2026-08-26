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
//! its stricter contract — a non-finite total norm is a typed refusal, never
//! silent — but pays for that check with the one sync [`refuse_nonfinite_norm`]
//! performs; that function's docs cover the cadence this file (and its
//! callers) hold that sync to, so it never reappears on every step.

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

/// Test-only counter of device→host reads issued by [`refuse_nonfinite_norm`]
/// — the *only* function on this file's per-step path allowed to call
/// `to_scalar`/`to_vec`. A CPU test cannot observe "no CUDA sync happened"
/// directly (there is no CUDA stream to inspect), so this counter is the
/// structural proxy: [`clip_gradients`] and an unchecked [`clip_and_step`]
/// call must never move it. If a future edit adds a host read anywhere else
/// on the hot path, a test asserting this counter stays at `0` across
/// [`clip_gradients`] goes red.
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
/// before — no compute is issued, and this is the ONE remaining place
/// `clip_gradients` returns `None`).
///
/// Returns the on-device `total_norm` scalar tensor (`None` when clipping was
/// disabled) so a caller can read it back — at whatever cadence it chooses —
/// through [`refuse_nonfinite_norm`]. `clip_gradients` itself never reads it.
///
/// Device op count for `n` trainable `Var`s with a gradient present: `n` ×
/// (`sqr` + `sum_all`) for the per-`Var` squared sums, `n - 1` adds to fold
/// them into one scalar, then `sqrt` + `affine` (+ eps) + `recip` + `affine`
/// (× `max_norm`) + `minimum` (the ≤ 1.0 clamp) = 5 fixed ops for the
/// coefficient, and `n` × `broadcast_mul` to rescale every gradient — `4n + 4`
/// device ops total, zero of them a host read. (A gradient that is not
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
) -> Result<Option<Tensor>> {
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
        return Ok(None);
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
    // `trainable_vars`, or none of them appear in `grads`) — nothing to clip.
    let Some(total_sq) = total_sq else {
        return Ok(None);
    };

    let total_norm = total_sq
        .sqrt()
        .map_err(|e| JammiError::FineTune(format!("GradClip sqrt: {e}")))?;

    // clip_coef = max_norm / (total_norm + 1e-6), clamped to at most 1.0 —
    // torch's exact order and epsilon (`torch/nn/utils/clip_grad.py`, see the
    // module doc's citation). `affine`/`recip` bake their
    // constants as kernel-launch scalars (no tensor upload); only `minimum`'s
    // `1.0` is materialized as a tensor — one small H2D upload per call. That
    // upload is NOT cached: nothing in this crate holds candle's
    // `CUDA_GRAPH_HTOD_CACHE` for the run (see [`crate::fine_tune::trainer`]'s
    // doc on why an unbounded, never-evicted, run-lifetime cache is the wrong
    // trade). Below, `broadcast_mul` has the same shape on the CUDA backend:
    // `coef` is a stride-0 scalar broadcast against every gradient, which
    // candle routes through its general strided-binary-op kernel (a fresh
    // per-call dims/strides upload) rather than `Tensor::affine`'s
    // host-`f64`-constant fast path — kept as `broadcast_mul` rather than
    // rewritten to a `CustomOp`, with that extra upload's cost meant to be
    // measured on device (not assumed) before it is optimized away.
    let clip_coef = total_norm
        .affine(1.0, 1e-6)
        .and_then(|d| d.recip())
        .and_then(|r| r.affine(max_norm, 0.0))
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

    Ok(Some(total_norm))
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
pub fn clip_and_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    grads: &mut GradStore,
    max_grad_norm: f64,
    check_every_n_steps: usize,
    step: usize,
    is_last_step: bool,
) -> Result<()> {
    let total_norm = clip_gradients(trainable_vars, grads, max_grad_norm)?;
    let on_cadence = check_every_n_steps > 0
        && (step == 1 || step.is_multiple_of(check_every_n_steps) || is_last_step);
    if on_cadence {
        if let Some(norm) = &total_norm {
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
            .expect("clipping was enabled");
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
            .expect("clipping was enabled");
        let norm_val: f32 = total_norm.to_scalar().unwrap();
        assert_eq!(
            norm_val, 1.0,
            "test setup: total_norm must equal max_norm exactly to be on the boundary"
        );

        // The host-computed coefficient, same op sequence `clip_gradients`
        // uses (`(total_norm + eps).recip() * max_norm`), pinned to its exact
        // bits — independent corroboration of the doc's closed form above.
        let host_coef: f32 = (1.0f32 / (norm_val + 1e-6)) * max_norm as f32;
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
            .expect("clipping was enabled");
        let norm_val: f32 = total_norm.to_scalar().unwrap();

        // Host reference computed from the SAME grads, torch's exact formula
        // (`torch/nn/utils/clip_grad.py`, see the module doc's citation):
        // max_norm / (total_norm + 1e-6), never clamped here since it is well
        // under 1.0.
        let host_coef = (max_norm / (norm_val as f64 + 1e-6)) as f32;
        let before: Vec<f32> = g_before.to_vec1().unwrap();
        let expected: Vec<f32> = before.iter().map(|x| x * host_coef).collect();
        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();

        // Tolerance: the device path computes the coefficient through one
        // extra `recip` + two `affine`s versus the host's single division —
        // each elementary f32 op can round by up to 0.5 ULP, so up to ~2 ULP
        // of drift in the coefficient is expected from the different op
        // sequence, not from a bug; propagated through one more multiply
        // against `before`, 4 ULP absolute-relative tolerance covers it with
        // headroom. This is *reasoned* from the op-count difference, not
        // fitted to what the test happened to observe.
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
    ///  - `recip`: 0.5 ulp;
    ///  - `affine(max_norm, 0.0)` with `max_norm == 1.0`: exact, 0 ulp;
    ///  - `minimum(1.0)`: exact (no rounding), 0 ulp;
    ///  - `broadcast_mul`: one rounding per element, 0.5 ulp.
    ///
    /// The host reference computes the same chain in f64 (error ≪ 1 f32 ulp)
    /// and rounds to f32 once (0.5 ulp). Total: ≤ 2.5 ulp device + 0.5 ulp
    /// host = 3 ulp of the result; asserted at 4 ulp (one ulp of headroom for
    /// the ulp-of-result vs ulp-of-intermediate mismatch at exponent
    /// boundaries — not a fitted number).
    fn assert_multi_var_clip_matches_host(device: &Device) -> Vec<Vec<f32>> {
        let (vars, mut grads, before) = four_vars_with_grads(device);
        let max_norm = 1.0f64;
        let total_norm = clip_gradients(&vars, &mut grads, max_norm)
            .unwrap()
            .expect("clipping was enabled");
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
            .expect("clipping enabled");
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
    fn cuda_device_or_skip(test: &str) -> Option<Device> {
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
    /// `sqrt`/`recip` are correctly rounded on both (nvcc's default
    /// `-prec-sqrt=true`/`-prec-div=true`; candle-kernels does not build with
    /// `-use_fast_math`); `affine(1.0, 1e-6)` is `x * 1.0 + 1e-6`, where
    /// `x * 1.0` is exact, so an FMA contraction on the GPU rounds once to
    /// the same value the CPU's two ops produce; `affine(max_norm, 0.0)`
    /// with `max_norm == 1.0` and `minimum(1.0)` are exact; and the final
    /// `broadcast_mul` is one rounding per element on both.
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
    /// device-independent; only the (never-consumed, since the step is
    /// refused on cadence) rescaled gradients differ. `nan_gradient_poisons_
    /// every_gradient_through_the_cpu_minimum` pins the CPU arm; this leg
    /// pins the CUDA arm when a device is present.
    #[test]
    fn multi_var_clip_matches_host_reference_on_cuda_and_is_bit_identical_to_cpu() {
        let Some(cuda) = cuda_device_or_skip("multi_var_clip_cuda") else {
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
        let total_norm = clip_gradients(&vars, &mut grads, 1.0).unwrap().unwrap();
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
        let total_norm = clip_gradients(&vars, &mut grads, 1.0).unwrap().unwrap();
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
        clip_gradients(&[w], &mut grads, 1.0).unwrap();
        assert_eq!(
            sync_read_count(),
            before,
            "clip_gradients must not perform any device→host read"
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
        clip_gradients(std::slice::from_ref(&w), &mut grads, 1.0).unwrap();
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
        assert!(result.is_none());
        assert_eq!(sync_read_count(), before, "disabled clipping must not sync");
        let after: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        let before_vals: Vec<f32> = g_before.to_vec1().unwrap();
        assert_eq!(before_vals, after);
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

    // ── Production-shape torch-parity acceptance leg (esc-182 finding item 1) ──
    //
    // Every oracle above proves the op-sequence derivation at a TOY scale: 4
    // Vars, a handful of elements each, values chosen so every intermediate
    // sum is an EXACTLY representable integer (0 ulp of rounding at every
    // step, by construction — see `assert_multi_var_clip_matches_host`'s own
    // doc). That is a mechanism proof, not a SCALE proof: it says nothing
    // about whether the same derivation still bounds the same code at the
    // tensor COUNT and ELEMENT COUNT this module's own top-of-file doc cites
    // as the motivating case — "224 [Vars] on a ModernBERT-large r16
    // Wqkv/Wo/Wi LoRA config" — with REAL (seeded-random, non-integer)
    // per-element gradient values, never a hand-picked constant.
    //
    // ## Real production shape
    //
    // `production_lora_site_shapes` is ModernBERT-large's actual dimensions
    // (`hidden_size = 1024`, `intermediate_size = 2624`, 28 layers) at
    // `lora_rank = 16`, four LoRA sites per layer (`jammi-encoders`'
    // `ModernBert::build` names the MLP output projection `"mlp.Wo"`,
    // distinct from attention's own `"Wo"` — see `grad_oracle.rs`'s
    // "Wqkv/Wo/Wi" table in `jammi-bench` for the same four-site
    // enumeration), two Vars per site (`lora_a`, `lora_b`):
    // `4 * 2 * 28 == 224`, pinned by
    // `production_shape_224_vars_matches_torch_parity_within_probabilistic_bound`'s
    // own `assert_eq!(k, 224, ..)` rather than assumed.
    //
    // ## Bound derivation, generalized from the toy fixture's own doc to real k/n
    //
    // The toy fixture's derivation states the WORST-CASE per-step rounding as
    // `(n_i - 1) * 0.5` ulp for a Var's own `n_i`-element `sqr`+`sum_all`
    // fold and `(k - 1) * 0.5` ulp for the cross-Var fold over `k` Vars — a
    // fully-correlated, every-rounding-in-the-same-direction bound. At
    // `k = 224` and `Σn_i ≈ 7.2M` (this fixture's real element count — LoRA
    // A/B matrices at the dimensions above), that worst-case additive bound
    // evaluates to several MILLION ulp (tens of percent, relative) —
    // vacuous: a bound that loose would accept almost any output, catching
    // nothing. The worst-case bound that was TIGHT at 4 toy Vars with a
    // handful of elements each is USELESS at production scale; a meaningful
    // acceptance criterion here needs a DIFFERENT (still-derived, never
    // fitted) bound.
    //
    // Standard numerical-analysis practice (Higham, *Accuracy and Stability
    // of Numerical Algorithms*, 2nd ed., the probabilistic summation-error
    // bound of §4.2) treats per-step rounding errors as independent,
    // zero-mean perturbations when the summands' signs are not
    // adversarially correlated — true here: every summand is `g_i²` for a
    // genuinely-random `g_i`, never engineered to round in one direction —
    // so the accumulated error grows like `sqrt(n) * u`, not `n * u`, with
    // high probability. `production_bound_ulp` uses THAT bound instead:
    // `0.5 * (Σ sqrt(n_i - 1) + sqrt(k - 1)) + 2.0` (the intra-Var folds'
    // probabilistic term, the cross-Var fold's probabilistic term, and the
    // toy fixture's own fixed `sqrt`/`affine(eps)`/`recip`/`broadcast_mul`
    // roundings, none of which scale with `n`/`k`) — computed FROM the
    // fixture's actual measured `n_i`/`k` in code, never a literal. This is
    // still a DERIVED bound (no absolute floor, nothing fitted to what a run
    // happened to produce), just the appropriate ONE for this scale.
    //
    // ## RED control
    //
    // `production_shape_old_formula_fails_the_bound_at_small_total_norm`
    // shows the pre-PR host-clip formula (`clip_coef = max_norm /
    // total_norm`, no epsilon) applied to the SAME 224-Var production-shaped
    // topology, at a `total_norm` small enough that torch's `+ 1e-6` term is
    // a material fraction of the denominator, EXCEEDS this same bound by
    // ORDERS of magnitude — the mechanism (an entirely missing additive
    // term, not merely a different rounding order) a scale-invariant ulp
    // bound is built to catch. That same leg's disclosure in the PARITY
    // test's own `eprintln!` reports the honest complement: at the SHIPPED
    // DEFAULT `max_grad_norm = 1.0` and a REALISTIC per-element gradient
    // magnitude, the epsilon term is only a few ulp — comparable to or
    // smaller than ordinary rounding noise — so the pre-PR formula's defect
    // is NOT reliably detectable at that specific (common) operating point;
    // it is real and material specifically in the small-`total_norm`
    // (near-convergence / small-`max_grad_norm`) regime this RED control
    // targets. That is the honest finding, reported rather than papered
    // over (family K).

    /// ModernBERT-large's real hidden width.
    const PROD_HIDDEN: usize = 1024;
    /// ModernBERT-large's real (per-branch) MLP intermediate width — the
    /// GeGLU `Wi` projection packs TWO of these (gate + up).
    const PROD_INTERMEDIATE: usize = 2624;
    /// The `r16` LoRA rank this module's own top-of-file doc cites.
    const PROD_LORA_RANK: usize = 16;
    /// ModernBERT-large's real depth.
    const PROD_NUM_LAYERS: usize = 28;

    /// Every LoRA-site `(lora_a_shape, lora_b_shape)` pair at the constants
    /// above, matching jammi-lora's own `A: [rank, in_features]` /
    /// `B: [out_features, rank]` orientation. Four sites per layer: packed
    /// `Wqkv` (`out = 3 * hidden`), attention `Wo`, packed GeGLU `mlp.Wi`
    /// (`out = 2 * intermediate`), `mlp.Wo`.
    fn production_lora_site_shapes() -> [([usize; 2], [usize; 2]); 4] {
        [
            (
                [PROD_LORA_RANK, PROD_HIDDEN],
                [3 * PROD_HIDDEN, PROD_LORA_RANK],
            ),
            ([PROD_LORA_RANK, PROD_HIDDEN], [PROD_HIDDEN, PROD_LORA_RANK]),
            (
                [PROD_LORA_RANK, PROD_HIDDEN],
                [2 * PROD_INTERMEDIATE, PROD_LORA_RANK],
            ),
            (
                [PROD_LORA_RANK, PROD_INTERMEDIATE],
                [PROD_HIDDEN, PROD_LORA_RANK],
            ),
        ]
    }

    /// Build the REAL 224-Var production trainable set (see
    /// `production_lora_site_shapes`) on `device`, each Var's gradient a
    /// GENUINE `backward()`-sourced value (never a literal): `d/dw[sum(c ⊙
    /// w)] = c` for a per-element coefficient tensor `c` drawn from a seeded
    /// `Normal(0, element_std)` — real (non-integer, non-hand-picked)
    /// floating-point noise, not the toy fixtures' friendly constants.
    /// Returns the Vars, the GradStore, and each Var's PRE-clip gradient as
    /// `Vec<f32>` (flattened, same order as the Vars) for an INDEPENDENT
    /// host reference.
    fn production_shaped_vars_with_grads(
        element_std: f64,
        seed: u64,
        device: &Device,
    ) -> (Vec<Var>, GradStore, Vec<Vec<f32>>) {
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f64, element_std).expect("valid std");

        let mut vars = Vec::with_capacity(PROD_NUM_LAYERS * 8);
        let mut before = Vec::with_capacity(PROD_NUM_LAYERS * 8);
        let mut loss: Option<Tensor> = None;

        for _layer in 0..PROD_NUM_LAYERS {
            for (a_shape, b_shape) in production_lora_site_shapes() {
                for shape in [a_shape, b_shape] {
                    let n = shape[0] * shape[1];
                    let w =
                        Var::from_tensor(&Tensor::zeros(&shape[..], DType::F32, device).unwrap())
                            .unwrap();
                    let coeff_vals: Vec<f32> =
                        (0..n).map(|_| normal.sample(&mut rng) as f32).collect();
                    before.push(coeff_vals.clone());
                    let coeff = Tensor::from_vec(coeff_vals, &shape[..], device).unwrap();
                    let term = (w.as_tensor() * &coeff).unwrap().sum_all().unwrap();
                    loss = Some(match loss {
                        None => term,
                        Some(acc) => (&acc + &term).unwrap(),
                    });
                    vars.push(w);
                }
            }
        }
        let grads = loss.unwrap().backward().unwrap();
        (vars, grads, before)
    }

    /// The scale-appropriate ulp bound for this fixture's ACTUAL measured
    /// `n_i`/`k` (see this section's leading doc: the toy fixture's
    /// worst-case `(n-1)*0.5` additive bound is vacuous at `Σn_i` in the
    /// millions; this uses the standard probabilistic `sqrt(n)`
    /// summation-error bound instead — still derived from the op sequence,
    /// never fitted to a measured value).
    fn production_bound_ulp(n_per_var: &[usize]) -> f64 {
        let k = n_per_var.len();
        let intra_var: f64 = n_per_var
            .iter()
            .map(|&n| (n.saturating_sub(1) as f64).sqrt() * 0.5)
            .sum();
        let cross_var = (k.saturating_sub(1) as f64).sqrt() * 0.5;
        // `sqrt`, `affine(1.0, eps)`, `recip`, `broadcast_mul`: 4 further
        // roundings at 0.5 ulp each that do not scale with n/k (see
        // `clip_gradients`'s own doc's op-count derivation); `affine(max_norm,
        // 0.0)`/`minimum` are exact here (max_norm == 1.0 or the clamp is not
        // engaged in either leg below).
        intra_var + cross_var + 4.0 * 0.5
    }

    /// ULP distance between a host-truth value `e` and a device-measured
    /// value `a`, in `f64` (the bounds this section computes can run into
    /// the tens of thousands of ulp — see `production_bound_ulp`'s own doc —
    /// well past where `f32` ulp arithmetic would itself lose precision).
    fn ulp_distance(e: f32, a: f32) -> f64 {
        let u = (f32::EPSILON as f64) * (e.abs() as f64).max(f32::MIN_POSITIVE as f64);
        ((e as f64) - (a as f64)).abs() / u
    }

    /// THE production-shape torch-parity acceptance leg. 224 real
    /// ModernBERT-large-r16-shaped LoRA Vars, real (seeded-random, non-toy)
    /// per-element gradients at a REALISTIC per-element magnitude
    /// (`element_std = 1e-3`), clipped at the SHIPPED DEFAULT `max_grad_norm
    /// = 1.0`, in the ACTIVE-clipping regime (`total_norm > max_norm` —
    /// asserted below, never assumed). Compared against an INDEPENDENT host
    /// `f64` reference of the SAME torch formula (`max_norm / (total_norm +
    /// 1e-6)`, clamped to at most `1.0` — never reached here since active),
    /// within `production_bound_ulp`.
    #[test]
    fn production_shape_224_vars_matches_torch_parity_within_probabilistic_bound() {
        let device = Device::Cpu;
        let (vars, mut grads, before) = production_shaped_vars_with_grads(1e-3, 20260825, &device);
        let n_per_var: Vec<usize> = before.iter().map(|v| v.len()).collect();
        let k = vars.len();
        assert_eq!(
            k, 224,
            "esc-182: this fixture must build EXACTLY the 224 Vars this module's own doc cites \
             (28 layers * 4 LoRA sites * 2 (A, B))"
        );

        let total_sq_host: f64 = before
            .iter()
            .flat_map(|g| g.iter())
            .map(|&x| (x as f64) * (x as f64))
            .sum();
        let host_norm = total_sq_host.sqrt();

        let max_norm = 1.0f64; // the shipped default (`max_grad_norm = 1.0`)
        assert!(
            host_norm > max_norm,
            "test setup: must be in the ACTIVE-clipping regime (total_norm > max_norm), got \
             total_norm={host_norm} at max_norm={max_norm}"
        );

        let host_coef = max_norm / (host_norm + 1e-6); // torch's exact formula, f64
        assert!(
            host_coef < 1.0,
            "test setup: coefficient must be unclamped (active, not at the 1.0 ceiling)"
        );

        let total_norm = clip_gradients(&vars, &mut grads, max_norm)
            .unwrap()
            .expect("clipping enabled");
        let norm_val: f32 = total_norm.to_scalar().unwrap();

        let bound = production_bound_ulp(&n_per_var);
        let norm_ulp = ulp_distance(host_norm as f32, norm_val);
        assert!(
            norm_ulp <= bound,
            "total_norm: host {host_norm} vs device {norm_val} — {norm_ulp} ulp exceeds the \
             derived bound {bound} ulp"
        );

        let mut max_grad_ulp = 0.0f64;
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
                let u = ulp_distance(e, *a);
                max_grad_ulp = max_grad_ulp.max(u);
                assert!(
                    u <= bound,
                    "clipped gradient: host {e} vs device {a} — {u} ulp exceeds the derived \
                     bound {bound} ulp"
                );
            }
        }

        // Honest disclosure (family K / esc-182 item 2's own instruction —
        // "report, do not paper over" — applies here too): at THIS realistic
        // production amplitude and the shipped default max_norm, torch's `+
        // 1e-6` term is TINY relative to `total_norm`. Measure it rather
        // than silently leave it implicit.
        let old_coef = max_norm / host_norm; // pre-PR formula, no epsilon
        let old_vs_new_relative = (old_coef - host_coef).abs() / host_coef;
        eprintln!(
            "esc-182 production-scale disclosure: total_norm={host_norm:.6}, max_norm=\
             {max_norm}, torch_coef={host_coef:.10}, old_coef={old_coef:.10}, relative diff=\
             {old_vs_new_relative:.3e} (~{:.1} f32 ulp of the coefficient) — measured max \
             device/host deviation this run: total_norm {norm_ulp:.2} ulp, gradients \
             {max_grad_ulp:.2} ulp, against a derived bound of {bound:.1} ulp",
            old_vs_new_relative / (f32::EPSILON as f64)
        );
    }

    /// CUDA leg for the production-shape acceptance test — bounded against
    /// the SAME independent host `f64` reference, NOT asserted bit-identical
    /// to CPU. Why: the toy fixture's CPU/CUDA bit-identity claim
    /// (`multi_var_clip_matches_host_reference_on_cuda_and_is_bit_identical_to_cpu`)
    /// holds ONLY because its sums are of EXACTLY representable small
    /// integers, so CPU's and CUDA's differently-ordered reductions land on
    /// the identical value regardless of associativity. At PRODUCTION scale
    /// with millions of genuinely-random `f32` values, CPU's
    /// (BLAS/sequential) and CUDA's (parallel-tree) reduction orders are
    /// mathematically equivalent but NOT bitwise so (`f32` addition is not
    /// associative) — bit-identity is not a claim this fixture can honestly
    /// make, and asserting it would test an accident of the toy fixture's
    /// exact integers, not a property of `clip_gradients` itself. This leg
    /// instead holds CUDA to the SAME truth-relative `production_bound_ulp`
    /// the CPU leg above uses.
    #[test]
    fn production_shape_224_vars_cuda_matches_host_within_bound() {
        let Some(cuda) = cuda_device_or_skip("production_shape_cuda") else {
            return;
        };
        let (vars, mut grads, before) = production_shaped_vars_with_grads(1e-3, 20260825, &cuda);
        let n_per_var: Vec<usize> = before.iter().map(|v| v.len()).collect();

        let total_sq_host: f64 = before
            .iter()
            .flat_map(|g| g.iter())
            .map(|&x| (x as f64) * (x as f64))
            .sum();
        let host_norm = total_sq_host.sqrt();
        let max_norm = 1.0f64;
        assert!(
            host_norm > max_norm,
            "test setup: active regime on CUDA too, got total_norm={host_norm}"
        );
        let host_coef = max_norm / (host_norm + 1e-6);

        let total_norm = clip_gradients(&vars, &mut grads, max_norm)
            .unwrap()
            .expect("clipping enabled");
        let norm_val: f32 = total_norm.to_scalar().unwrap();
        let bound = production_bound_ulp(&n_per_var);
        let norm_ulp = ulp_distance(host_norm as f32, norm_val);
        assert!(
            norm_ulp <= bound,
            "CUDA total_norm: host {host_norm} vs device {norm_val} — {norm_ulp} ulp exceeds \
             bound {bound}"
        );

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
                let u = ulp_distance(e, *a);
                assert!(
                    u <= bound,
                    "CUDA clipped gradient exceeds bound: host {e} vs device {a} — {u} ulp > \
                     {bound}"
                );
            }
        }
    }

    /// RED control (esc-182 finding item 1): the SAME 224 production-shaped
    /// LoRA Vars/topology, at a `total_norm` small enough (`element_std =
    /// 1e-9`, deliberately far below the realistic-amplitude leg above —
    /// see this section's leading doc for why the realistic-amplitude leg
    /// alone cannot fire this control) that torch's `1e-6` epsilon is a
    /// LARGE fraction of the denominator. Still `total_norm > max_norm`
    /// (active — asserted below, chosen well away from the boundary band so
    /// this is unambiguously the "> max_norm" regime, not the
    /// already-covered `at_max_norm_boundary_coef_is_not_bit_identical_to_
    /// no_clip` edge).
    #[test]
    fn production_shape_old_formula_fails_the_bound_at_small_total_norm() {
        let device = Device::Cpu;
        let (vars, mut grads, before) = production_shaped_vars_with_grads(1e-9, 20260826, &device);
        let n_per_var: Vec<usize> = before.iter().map(|v| v.len()).collect();

        let total_sq_host: f64 = before
            .iter()
            .flat_map(|g| g.iter())
            .map(|&x| (x as f64) * (x as f64))
            .sum();
        let host_norm = total_sq_host.sqrt();
        let max_norm = host_norm * 0.5; // deep active, not boundary-adjacent
        assert!(
            host_norm > max_norm,
            "test setup: active regime, got total_norm={host_norm} max_norm={max_norm}"
        );
        assert!(
            host_norm < 1e-4,
            "test setup: total_norm must be small enough that the 1e-6 epsilon term is \
             MATERIAL (got {host_norm}) — see this section's leading doc for the derivation \
             of why realistic (production-amplitude) total_norm cannot fire this control"
        );

        let new_coef = max_norm / (host_norm + 1e-6);
        let old_coef = max_norm / host_norm;
        let bound = production_bound_ulp(&n_per_var);

        // The device path (NEW formula) — must stay INSIDE the bound at this
        // SAME small-norm scale, proving the bound itself is not simply too
        // tight to ever pass anything at this magnitude (the non-vacuous
        // control half of this leg).
        let total_norm = clip_gradients(&vars, &mut grads, max_norm)
            .unwrap()
            .expect("clipping enabled");
        let norm_val: f32 = total_norm.to_scalar().unwrap();
        let norm_ulp = ulp_distance(host_norm as f32, norm_val);
        assert!(
            norm_ulp <= bound,
            "sanity: the NEW formula must stay within the bound at this same scale (got \
             {norm_ulp} ulp vs bound {bound})"
        );

        // The OLD formula's OUTPUT — computed host-side from the SAME
        // pre-clip gradients — compared against the SAME truth (the
        // new/torch formula's host reference) at the SAME bound.
        // `max_grad_ulp` tracks the WORST element so one clean number
        // carries the "must fail" claim.
        let mut max_grad_ulp = 0.0f64;
        for g_before in &before {
            for &x in g_before {
                let truth = ((x as f64) * new_coef) as f32;
                let old_val = ((x as f64) * old_coef) as f32;
                max_grad_ulp = max_grad_ulp.max(ulp_distance(truth, old_val));
            }
        }
        assert!(
            max_grad_ulp > bound,
            "esc-182 RED control did not fire: the pre-PR (no-epsilon) formula's output should \
             exceed the SAME bound the new formula clears at this total_norm scale (host_norm \
             {host_norm}, old_coef {old_coef:.10} vs torch_coef {new_coef:.10}), but only \
             reached {max_grad_ulp} ulp against a bound of {bound} ulp — the epsilon-driven \
             divergence (relative ~{:.3e}) was not large enough at this scale; the leg's \
             total_norm choice needs to be smaller",
            (old_coef - new_coef).abs() / new_coef
        );
        eprintln!(
            "esc-182 RED control fired as expected: pre-PR formula {max_grad_ulp:.1} ulp from \
             truth (bound {bound:.1} ulp) at total_norm={host_norm:.6}, max_norm={max_norm:.6}"
        );
    }
}
