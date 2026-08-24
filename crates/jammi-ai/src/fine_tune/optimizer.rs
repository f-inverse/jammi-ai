//! The shared optimizer-step seam: global-L2 gradient clipping followed by an
//! AdamW step, plus the loss-level convenience that runs `backward` first.
//!
//! Both the token-coupled text trainer and the non-text parallel loop reduce a
//! batch to a single update through this one place, so the clip→step contract
//! (and the `torch.nn.utils.clip_grad_norm_` semantics it implements) lives in
//! exactly one location rather than being copy-pasted per call site.

use candle_core::{backprop::GradStore, DType, Tensor, Var};
use jammi_db::error::{JammiError, Result};

use crate::fine_tune::adamw::AdamW;

/// Clip gradients by global L2 norm in-place, matching
/// `torch.nn.utils.clip_grad_norm_(params, max_norm)`.
///
/// Computes `total_norm = sqrt(sum ||g||² for all g)`. If `total_norm > max_norm`,
/// every gradient is scaled by `max_norm / total_norm`. `max_norm <= 0.0`
/// disables clipping entirely.
pub fn clip_gradients(trainable_vars: &[Var], grads: &mut GradStore, max_norm: f64) -> Result<()> {
    if max_norm <= 0.0 {
        return Ok(());
    }

    // Compute the global L2 norm.
    let mut total_sq = 0.0f64;
    for var in trainable_vars {
        let t: &Tensor = var;
        if let Some(g) = grads.get(t) {
            let g_f32 = if g.dtype() == DType::F32 {
                g.clone()
            } else {
                g.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("GradClip dtype: {e}")))?
            };
            let sq: f32 = g_f32
                .sqr()
                .map_err(|e| JammiError::FineTune(format!("GradClip sqr: {e}")))?
                .sum_all()
                .map_err(|e| JammiError::FineTune(format!("GradClip sum: {e}")))?
                .to_scalar::<f32>()
                .map_err(|e| JammiError::FineTune(format!("GradClip scalar: {e}")))?;
            total_sq += sq as f64;
        }
    }

    let total_norm = total_sq.sqrt();
    if total_norm <= max_norm {
        return Ok(());
    }

    let clip_coef = max_norm / total_norm;
    for var in trainable_vars {
        let t: &Tensor = var;
        if let Some(g) = grads.remove(t) {
            let scaled = (&g * clip_coef)
                .map_err(|e| JammiError::FineTune(format!("GradClip scale: {e}")))?;
            grads.insert(t, scaled);
        }
    }

    Ok(())
}

/// Clip an already-computed gradient store, then take one AdamW step.
///
/// This is the seam both training loops share: whatever produced `grads` (a
/// single backward, an accumulation window, or the GradCache two-pass backward),
/// the clip-then-step that turns them into a parameter update is identical.
/// `max_grad_norm <= 0.0` skips clipping.
pub fn clip_and_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    grads: &mut GradStore,
    max_grad_norm: f64,
) -> Result<()> {
    clip_gradients(trainable_vars, grads, max_grad_norm)?;
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
/// non-text parallel loop has one batch per step and uses this directly.
pub fn optimizer_step(
    optimizer: &mut AdamW,
    trainable_vars: &[Var],
    loss: &Tensor,
    max_grad_norm: f64,
) -> Result<()> {
    let mut grads = loss
        .backward()
        .map_err(|e| JammiError::FineTune(format!("Backward: {e}")))?;
    clip_and_step(optimizer, trainable_vars, &mut grads, max_grad_norm)
}
