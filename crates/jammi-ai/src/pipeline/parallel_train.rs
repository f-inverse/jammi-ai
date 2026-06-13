//! Parallel (non-text) training loop: autograd + optimizer over precomputed
//! feature/target tensors, with no tokenizer, no `LoadedModel`, and no
//! `input_ids` anywhere in its signature.
//!
//! The text trainer's loop is welded to a per-batch tokenizer and a
//! `forward(input_ids, attention_mask)` encode step, so it cannot drive a model
//! whose inputs are already tensors (a graph/context predictor over precomputed
//! features). This loop is that path: it owns only the autograd/optimizer
//! mechanics and takes the model and loss as caller closures, so what is being
//! trained is entirely the caller's concern.
//!
//! It shares the clip→step seam with the text trainer through
//! [`crate::fine_tune::optimizer::optimizer_step`] — the two loops differ in
//! how a batch becomes a loss, not in how a loss becomes a parameter update.

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{Tensor, Var};
use candle_nn::VarMap;

use crate::fine_tune::adamw::{AdamW, ParamsAdamW};
use jammi_db::error::{JammiError, Result};

use crate::fine_tune::optimizer::optimizer_step;

/// One precomputed training example batch: a feature tensor and the target it
/// should predict. Both are already tensors — there is no encode/tokenize step
/// between the data and the model.
///
/// The simplest batch shape [`train_loop`] drives: a flat feature/target pair.
/// [`train_loop`] is generic over the batch type, so an episodic sampler over a
/// richer batch (a context set plus its held-out target) is just another batch
/// type the consumer passes — this struct is the one provided for the flat case.
#[derive(Debug, Clone)]
pub struct TensorBatch {
    /// Model inputs, `[batch, ..]` — whatever the caller's `model_fn` consumes.
    pub features: Tensor,
    /// Supervision targets, `[batch, ..]` — whatever the caller's `loss_fn`
    /// scores predictions against.
    pub targets: Tensor,
}

/// Configuration for [`train_loop`].
#[derive(Debug, Clone)]
pub struct ParallelTrainConfig {
    /// Passes over the batch set.
    pub epochs: usize,
    /// AdamW learning rate.
    pub learning_rate: f64,
    /// AdamW decoupled weight decay.
    pub weight_decay: f64,
    /// Global-L2 gradient-clip norm; `<= 0.0` disables clipping.
    pub grad_clip: f64,
}

impl Default for ParallelTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            grad_clip: 1.0,
        }
    }
}

/// What one finished [`train_loop`] run reports back.
#[derive(Debug, Clone)]
pub struct ParallelTrainReport {
    /// Mean loss across the final epoch's batches.
    pub final_loss: f64,
    /// Total optimizer steps taken (one per batch per epoch).
    pub total_steps: usize,
}

/// Train the parameters held in `varmap` over `batches` for `config.epochs`.
///
/// Generic over the batch type `B`, so the loop owns only the
/// autograd/optimizer mechanics and what a batch *is* stays the caller's
/// concern: `model_fn(&batch) -> predictions` runs the forward pass over a whole
/// batch (the [`TensorBatch`] flat case, an episodic context-set batch, or any
/// other shape), and `loss_fn(&preds, &batch) -> loss` scores the predictions
/// against the same batch's supervision. Each batch is one forward → loss →
/// [`optimizer_step`] (backward + clip + AdamW). The optimizer is built once
/// from the varmap's trainable variables and reused across every step so AdamW's
/// moment estimates persist, exactly as the text trainer's does.
///
/// Neither argument nor return type names a tokenizer, a `LoadedModel`, or
/// `input_ids`: the decoupling from the text path is structural, readable off
/// this signature alone.
///
/// `cancel` is the cooperative-cancellation flag the loop checks at every epoch
/// boundary — the worker's heartbeat sets it when the lease is lost so the loop
/// bails (a `spawn_blocking` thread cannot be force-aborted). Pass an
/// always-false flag for a run that cannot be cancelled.
pub fn train_loop<B, M, L>(
    varmap: &VarMap,
    batches: &[B],
    config: &ParallelTrainConfig,
    cancel: &AtomicBool,
    model_fn: M,
    loss_fn: L,
) -> Result<ParallelTrainReport>
where
    M: Fn(&B) -> Result<Tensor>,
    L: Fn(&Tensor, &B) -> Result<Tensor>,
{
    let trainable_vars: Vec<Var> = varmap.all_vars();
    let mut optimizer = AdamW::new(
        trainable_vars.clone(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )
    .map_err(|e| JammiError::FineTune(format!("Optimizer init: {e}")))?;

    let mut total_steps = 0usize;
    let mut last_epoch_loss = 0.0f64;

    for _epoch in 0..config.epochs {
        if cancel.load(Ordering::Relaxed) {
            return Err(JammiError::FineTune(
                "training cancelled: lease lost before epoch boundary".into(),
            ));
        }
        let mut epoch_loss = 0.0f64;
        let mut batch_count = 0usize;

        for batch in batches {
            let preds = model_fn(batch)?;
            let loss = loss_fn(&preds, batch)?;

            epoch_loss += scalar_loss(&loss)?;
            batch_count += 1;

            optimizer_step(&mut optimizer, &trainable_vars, &loss, config.grad_clip)?;
            total_steps += 1;
        }

        last_epoch_loss = epoch_loss / batch_count.max(1) as f64;
    }

    Ok(ParallelTrainReport {
        final_loss: last_epoch_loss,
        total_steps,
    })
}

/// Read a scalar loss tensor to `f64`, casting to f32 first if it carries a
/// reduced-precision dtype (the same convention the text trainer uses).
fn scalar_loss(loss: &Tensor) -> Result<f64> {
    let loss = if loss.dtype() == candle_core::DType::F32 {
        loss.clone()
    } else {
        loss.to_dtype(candle_core::DType::F32)
            .map_err(|e| JammiError::FineTune(format!("Loss dtype cast: {e}")))?
    };
    Ok(loss
        .to_scalar::<f32>()
        .map_err(|e| JammiError::FineTune(format!("Loss scalar: {e}")))? as f64)
}
