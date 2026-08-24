//! The shared optimizer-step seam: global-L2 gradient clipping followed by an
//! AdamW step, plus the loss-level convenience that runs `backward` first.
//!
//! Both the token-coupled text trainer and the non-text parallel loop reduce a
//! batch to a single update through this one place, so the clip→step contract
//! (and the `torch.nn.utils.clip_grad_norm_` semantics it implements) lives in
//! exactly one location rather than being copy-pasted per call site.
//!
//! ## Device-side clip (P4b)
//!
//! [`clip_gradients`] used to end every trainable [`Var`]'s contribution to
//! the global norm with a `to_scalar::<f32>()` — a full-pipeline device→host
//! sync, once per `Var` (224 of them on a ModernBERT-large r16 `Wqkv`/`Wo`/`Wi`
//! LoRA config at the default `max_grad_norm = 1.0`). PyTorch's own
//! `torch.nn.utils.clip_grad._clip_grads_with_norm_` computes `clip_coef =
//! max_norm / (total_norm + 1e-6)`, clamps it to at most `1.0`, and multiplies
//! every gradient by the clamped coefficient **unconditionally** — its comment
//! is explicit that this "avoids a `if clip_coef < 1:` conditional which can
//! require a CPU <=> device synchronization" (ref-077,
//! `torch/nn/utils/clip_grad.py`). [`clip_gradients`] now does the same: the
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
use jammi_db::error::{JammiError, Result};

use crate::fine_tune::adamw::AdamW;

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

/// Clip gradients by global L2 norm in-place, matching
/// `torch.nn.utils.clip_grad_norm_(params, max_norm)`, entirely on-device.
///
/// Computes `total_norm = sqrt(sum ||g||² for all g)` as a device scalar
/// tensor (a fixed left-to-right fold over `trainable_vars` in order, so the
/// reduction order — and therefore the bits — is deterministic run to run).
/// Every gradient is then rescaled by `clip_coef = (max_norm / (total_norm +
/// 1e-6)).min(1.0)` **unconditionally**, matching torch's own avoidance of a
/// host-syncing `if total_norm > max_norm` branch (ref-077). When `total_norm
/// <= max_norm`, `clip_coef` clamps to exactly `1.0` and the multiply is a
/// no-op in the bits (`x * 1.0 == x` for every finite `x`), so a non-clipping
/// step here is bit-identical to skipping the rescale entirely — the same
/// guarantee the old early-return gave, without needing the host read that
/// made that branch possible. `max_norm <= 0.0` disables clipping entirely
/// (unchanged from before — no compute is issued, and this is the ONE
/// remaining place `clip_gradients` returns `None`).
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
/// device ops total, zero of them a host read. (Grads that are not already
/// `F32` pay one extra `to_dtype` each, not counted above — unchanged from
/// the pre-existing behavior.)
pub fn clip_gradients(
    trainable_vars: &[Var],
    grads: &mut GradStore,
    max_norm: f64,
) -> Result<Option<Tensor>> {
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
    // torch's exact order and epsilon (ref-077). `affine`/`recip` bake their
    // constants as kernel-launch scalars (no tensor upload); only `minimum`'s
    // `1.0` is materialized as a tensor, and that one is exactly the kind of
    // tiny constant [`candle_core::CudaDevice::enable_cuda_graph_htod_cache`]
    // is meant to cache across steps (see [`crate::fine_tune::trainer`]'s
    // guard).
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
            let scaled = g
                .broadcast_mul(&coef)
                .map_err(|e| JammiError::FineTune(format!("GradClip scale: {e}")))?;
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
/// back and checked only when `step` (the 1-based index of the optimizer step
/// about to run — pass `global_step + 1`) is a multiple of it, and never when
/// `check_every_n_steps == 0`. Every call site decides its own cadence; see
/// each call site for why.
pub fn clip_and_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    grads: &mut GradStore,
    max_grad_norm: f64,
    check_every_n_steps: usize,
    step: usize,
) -> Result<()> {
    let total_norm = clip_gradients(trainable_vars, grads, max_grad_norm)?;
    if check_every_n_steps > 0 && step.is_multiple_of(check_every_n_steps) {
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
/// the 1-based index of the step about to run).
pub fn optimizer_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    loss: &Tensor,
    max_grad_norm: f64,
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
        // (ref-077): max_norm / (total_norm + 1e-6), never clamped here since
        // it is well under 1.0.
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
        // RED-first: a NaN total norm must produce a typed error when
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

    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_and_step_skips_the_check_off_cadence() {
        // check_every_n_steps = 10, step = 3 → not a multiple of 10, so the
        // check must not run even though the norm happens to be NaN — this
        // is the "acceptable off the critical path" contract: the sync is
        // opt-in per step, gated by cadence, not by the norm's value.
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &dev).unwrap()).unwrap();
        let coeff = Tensor::new(f32::NAN, &dev)
            .unwrap()
            .broadcast_as((2,))
            .unwrap();
        let loss = (w.as_tensor() * &coeff).unwrap().sum_all().unwrap();
        let mut grads = loss.backward().unwrap();
        let mut optimizer = AdamW::new(vec![w.clone()], params_adamw(0.1)).unwrap();

        let before = sync_read_count();
        clip_and_step(&mut optimizer, &[w], &mut grads, 1.0, 10, 3).unwrap();
        assert_eq!(
            sync_read_count(),
            before,
            "off-cadence step must not read the norm back at all"
        );
    }

    #[test]
    #[serial(grad_clip_sync_read_count)]
    fn clip_and_step_refuses_a_nonfinite_norm_on_cadence() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &dev).unwrap()).unwrap();
        let coeff = Tensor::new(f32::NAN, &dev)
            .unwrap()
            .broadcast_as((2,))
            .unwrap();
        let loss = (w.as_tensor() * &coeff).unwrap().sum_all().unwrap();
        let mut grads = loss.backward().unwrap();
        let mut optimizer = AdamW::new(vec![w.clone()], params_adamw(0.1)).unwrap();

        let err = clip_and_step(&mut optimizer, &[w], &mut grads, 1.0, 10, 10)
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
}
