//! GradCache: in-batch-negative training with the negative pool decoupled from
//! the activation-memory budget.
//!
//! In-batch-negative loss quality scales with the batch size — every other
//! row's positive is a negative, so a larger batch means harder, more
//! plentiful negatives. But a single backward over the whole batch holds every
//! row's encoder activation graph at once, so the batch is memory-bound.
//! GradCache ([Gao et al. 2021]) breaks that coupling:
//!
//! 1. **Representation pass (no encoder graph kept).** Encode every row chunk
//!    by chunk, detaching each chunk's representation so the encoder graph is
//!    freed immediately. Collect all representations into leaf tensors, compute
//!    the in-batch loss over the *whole* batch, and read each representation's
//!    loss-gradient `∂L/∂rep`. Only representations — not activation graphs —
//!    are resident, so the negative pool can be the entire batch.
//! 2. **Accumulation pass (one chunk's graph at a time).** Re-encode each chunk
//!    *with* grad and backpropagate the surrogate `Σ rep · cached_grad`. By the
//!    chain rule `∂(Σ rep·∂L/∂rep)/∂θ = ∂L/∂θ`, so the optimiser sees exactly
//!    the single-pass gradient — the [`crate::fine_tune`] gradient-equivalence
//!    test pins this — while only one chunk's encoder graph is ever alive.
//!
//! This is distinct from gradient accumulation, which sums losses over
//! micro-batches that each see only their own negatives: GradCache's loss in
//! step 1 sees every negative in the batch at once.
//!
//! [Gao et al. 2021]: https://arxiv.org/abs/2101.06983

use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};
use jammi_db::error::{JammiError, Result};

/// One group of rows to embed — the anchors, the positives, or (for a triplet
/// batch) the hard negatives. Each group encodes independently into an
/// `[n_rows, d]` representation; the loss closure consumes all groups together.
pub struct EncodeGroup<'a> {
    /// Number of rows in the group.
    pub rows: usize,
    /// Encode rows `[start, start + len)` of this group into an `[len, d]`
    /// representation. The closure is called twice per chunk over the run — once
    /// detached in the representation pass, once with grad in the accumulation
    /// pass — so it must be deterministic (dropout disabled) for the two passes
    /// to agree.
    pub encode: &'a dyn Fn(usize, usize) -> Result<Tensor>,
}

/// Run GradCache over `groups`, chunking each group at `chunk_size` rows.
///
/// `loss` receives the full per-group representations (in group order) and
/// returns the scalar in-batch loss. `trainable_vars` are the parameters whose
/// gradients are accumulated. Returns a [`GradStore`] holding `∂loss/∂θ` for
/// every trainable var — identical to a single-pass `loss(reps).backward()`,
/// but computed without holding all encoder graphs at once.
pub fn gradcache_backward(
    groups: &[EncodeGroup<'_>],
    chunk_size: usize,
    loss: &dyn Fn(&[Tensor]) -> Result<Tensor>,
    trainable_vars: &[Var],
) -> Result<GradStore> {
    if chunk_size == 0 {
        return Err(JammiError::FineTune(
            "gradcache chunk_size must be > 0".into(),
        ));
    }

    // ── Pass 1: detached representations + their loss-gradients ──────────────
    //
    // Encode every chunk with the encoder graph discarded (detach), so only the
    // representations stay resident. Each group's representation is wrapped in a
    // leaf `Var` so the loss backward yields `∂L/∂rep` for it.
    let mut rep_vars: Vec<Var> = Vec::with_capacity(groups.len());
    for group in groups {
        let mut chunks: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start < group.rows {
            let len = chunk_size.min(group.rows - start);
            let rep = (group.encode)(start, len)?;
            // Drop the encoder graph: only the representation values survive.
            let rep = rep
                .detach()
                .to_dtype(DType::F32)
                .map_err(|e| JammiError::FineTune(format!("gradcache rep dtype: {e}")))?;
            chunks.push(rep);
            start += len;
        }
        let rep = if chunks.len() == 1 {
            chunks.into_iter().next().expect("one chunk")
        } else {
            let refs: Vec<&Tensor> = chunks.iter().collect();
            Tensor::cat(&refs, 0)
                .map_err(|e| JammiError::FineTune(format!("gradcache cat: {e}")))?
        };
        rep_vars.push(
            Var::from_tensor(&rep)
                .map_err(|e| JammiError::FineTune(format!("gradcache rep var: {e}")))?,
        );
    }

    let rep_tensors: Vec<Tensor> = rep_vars.iter().map(|v| v.as_tensor().clone()).collect();
    let total_loss = loss(&rep_tensors)?;
    let rep_grad_store = total_loss
        .backward()
        .map_err(|e| JammiError::FineTune(format!("gradcache rep backward: {e}")))?;

    // The cached loss-gradient of each group's representation. A group whose
    // representation does not reach the loss (no gradient) contributes nothing.
    let cached_grads: Vec<Option<Tensor>> = rep_vars
        .iter()
        .map(|v| rep_grad_store.get(v.as_tensor()).cloned())
        .collect();

    // ── Pass 2: re-encode per chunk with grad, backprop the surrogate ────────
    //
    // For each chunk, the surrogate `Σ rep · cached_grad` has parameter
    // gradient equal to the chunk's contribution to `∂loss/∂θ` (chain rule),
    // and only this chunk's encoder graph is alive while it is computed.
    let mut accumulated: Option<GradStore> = None;
    for (group, cached) in groups.iter().zip(cached_grads.iter()) {
        let Some(cached_grad) = cached else {
            continue;
        };
        let mut start = 0;
        while start < group.rows {
            let len = chunk_size.min(group.rows - start);
            let rep = (group.encode)(start, len)?;
            let rep = if rep.dtype() == DType::F32 {
                rep
            } else {
                rep.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("gradcache rep2 dtype: {e}")))?
            };
            let grad_slice = cached_grad
                .narrow(0, start, len)
                .map_err(|e| JammiError::FineTune(format!("gradcache grad slice: {e}")))?;
            let surrogate = (&rep * &grad_slice)
                .map_err(|e| JammiError::FineTune(format!("gradcache surrogate mul: {e}")))?
                .sum_all()
                .map_err(|e| JammiError::FineTune(format!("gradcache surrogate sum: {e}")))?;
            let grads = surrogate
                .backward()
                .map_err(|e| JammiError::FineTune(format!("gradcache chunk backward: {e}")))?;
            accumulated = Some(merge_grads(accumulated.take(), grads, trainable_vars)?);
            start += len;
        }
    }

    accumulated.ok_or_else(|| {
        JammiError::FineTune(
            "gradcache produced no gradients — no representation reached the loss".into(),
        )
    })
}

/// Merge a fresh [`GradStore`] into a running accumulator, summing per-var.
/// Seeds the accumulator from the first store (candle's `GradStore::new` is
/// private), matching the trainer's gradient-accumulation merge.
fn merge_grads(
    accumulated: Option<GradStore>,
    fresh: GradStore,
    trainable_vars: &[Var],
) -> Result<GradStore> {
    match accumulated {
        None => Ok(fresh),
        Some(mut acc) => {
            for var in trainable_vars {
                let t: &Tensor = var;
                if let Some(g_new) = fresh.get(t) {
                    if let Some(g_acc) = acc.remove(t) {
                        let summed = (&g_acc + g_new)
                            .map_err(|e| JammiError::FineTune(format!("gradcache merge: {e}")))?;
                        acc.insert(t, summed);
                    } else {
                        acc.insert(t, g_new.clone());
                    }
                }
            }
            Ok(acc)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Module, Tensor};
    use candle_nn::Linear;

    use crate::fine_tune::trainer::mnrl_loss_for_test;

    /// GradCache must reproduce the single-pass gradient. This is the contract
    /// that gates the feature: a trainable linear "encoder" embeds anchors and
    /// positives, MNRL scores them, and the parameter gradient from the two-pass
    /// GradCache path must match the one-pass `loss.backward()` within tolerance.
    #[test]
    fn gradcache_gradient_matches_single_pass() {
        let device = Device::Cpu;

        // A trainable linear encoder shared by both paths. Distinct anchor and
        // positive inputs so the similarity matrix is non-trivial.
        let w = Var::from_tensor(
            &Tensor::new(
                &[
                    [0.5f32, -0.2, 0.1, 0.3],
                    [0.1, 0.4, -0.3, 0.2],
                    [-0.2, 0.1, 0.5, -0.1],
                    [0.3, 0.2, -0.1, 0.4],
                ],
                &device,
            )
            .unwrap(),
        )
        .unwrap();
        let linear = Linear::new(w.as_tensor().clone(), None);

        let anchors = Tensor::new(
            &[
                [1.0f32, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            &device,
        )
        .unwrap();
        let positives = Tensor::new(
            &[
                [0.9f32, 0.1, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.1, 0.9],
            ],
            &device,
        )
        .unwrap();

        let encode_rows = |src: &Tensor, start: usize, len: usize| -> Result<Tensor> {
            let rows = src.narrow(0, start, len).unwrap();
            Ok(linear.forward(&rows).unwrap())
        };

        // ── Single pass: encode everything, one backward over the batch. ──────
        let a_rep = linear.forward(&anchors).unwrap();
        let p_rep = linear.forward(&positives).unwrap();
        let single_loss = mnrl_loss_for_test(&a_rep, &p_rep, None, 20.0, true).unwrap();
        let single_grads = single_loss.backward().unwrap();
        let single_w_grad = single_grads.get(w.as_tensor()).unwrap();

        // ── GradCache: two-pass, chunked at 2 rows. ───────────────────────────
        let a_enc = |start: usize, len: usize| encode_rows(&anchors, start, len);
        let p_enc = |start: usize, len: usize| encode_rows(&positives, start, len);
        let groups = vec![
            EncodeGroup {
                rows: 4,
                encode: &a_enc,
            },
            EncodeGroup {
                rows: 4,
                encode: &p_enc,
            },
        ];
        let loss = |reps: &[Tensor]| mnrl_loss_for_test(&reps[0], &reps[1], None, 20.0, true);
        let cached_grads = gradcache_backward(&groups, 2, &loss, std::slice::from_ref(&w)).unwrap();
        let cached_w_grad = cached_grads.get(w.as_tensor()).unwrap();

        // Gradients must agree element-wise within tolerance.
        let diff = (single_w_grad - cached_w_grad)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "GradCache gradient must match single-pass within tolerance, max abs diff = {diff}"
        );

        // And the gradient must be non-trivial — otherwise the match is vacuous.
        let mag = single_w_grad
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            mag > 1e-3,
            "single-pass gradient should be non-trivial, got {mag}"
        );
    }
}
