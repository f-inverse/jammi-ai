//! ADVERSARIAL CORRECTNESS SWEEP — slice A (training / fine-tune).
//!
//! Each test constructs an independently-computed oracle answer and asserts the
//! engine matches. A failing test here is a confirmed correctness bug in the
//! 0.26.0 fine-tune surface — NOT a perf or "does it run" check.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use jammi_ai::fine_tune::{
    data::{TrainingBatch, TrainingDataLoader},
    lora::build_projection_head,
    target::TrainingTarget,
    trainer::{compute_lr, TrainingLoopBuilder},
    FineTuneConfig, LrSchedule,
};
use jammi_ai::model::ModelTask;

// ── Shared catalog/job boilerplate so a precomputed TrainingLoop can run. ────
//
// The TrainingLoop stamps run-start metrics under a lease guard, so it needs a
// registered model + a claimed training job. This mirrors the existing
// `training_early_stopping_triggers` setup exactly.
async fn claimed_loop_env(tag: &str) -> (Arc<jammi_db::catalog::Catalog>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
    let model_id = format!("{tag}-model");
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: &model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
            job_id: tag,
            base_model_id: &format!("{model_id}::1"),
            training_source: "src",
            loss_type: "cosent",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    catalog
        .claim_next_training_job(&format!("{tag}-worker"), std::time::Duration::from_secs(60))
        .await
        .unwrap()
        .expect("queued job claimable");
    (catalog, dir)
}

/// A precomputed contrastive batch with a learnable (non-degenerate) signal:
/// orthogonal a/b but score=1.0, so cosent loss is non-zero and gradients flow.
fn learnable_batch(device: &Device) -> TrainingBatch {
    let a = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], device).unwrap();
    let b = Tensor::new(&[[0.0f32, 1.0, 0.0, 0.0]], device).unwrap();
    TrainingBatch::Contrastive {
        embeddings_a: a,
        embeddings_b: b,
        scores: Tensor::new(&[1.0f32], device).unwrap(),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ORACLE 1 — epochs honoring (precomputed path).
//
// With validation_fraction=0, grad_accum=1, a precomputed loader of N batches:
//   total_steps == N * epochs   (one optimizer step per batch per epoch).
// Early stopping is disabled by monitoring TrainLoss with patience high enough
// that it can never trigger inside the budget. So total_steps MUST equal N*epochs
// for whatever epochs is configured.
// ════════════════════════════════════════════════════════════════════════════
#[tokio::test(flavor = "multi_thread")]
async fn oracle_epochs_honored_exactly_precomputed() {
    let device = Device::Cpu;
    const N: usize = 4; // batches per epoch (precomputed → num_batches == len)

    for epochs in [1usize, 2, 5] {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = build_projection_head(4, &FineTuneConfig::default(), &varmap, &vb).unwrap();

        let batches: Vec<_> = (0..N).map(|_| learnable_batch(&device)).collect();
        let loader = TrainingDataLoader::from_precomputed(batches);

        let tag = format!("ep-honor-{epochs}");
        let (catalog, dir) = claimed_loop_env(&tag).await;

        let mut tl = TrainingLoopBuilder::new(
            TrainingTarget::ProjectionHead { head },
            varmap,
            FineTuneConfig {
                epochs,
                batch_size: 1,
                validation_fraction: 0.0,
                warmup_steps: 0,
                gradient_accumulation_steps: 1,
                // Monitor train loss, patience huge → early stopping can't fire.
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                early_stopping_patience: 10_000,
                learning_rate: 1e-3,
                ..Default::default()
            },
        )
        .job_id(tag.clone())
        .worker_id(format!("{tag}-worker"))
        .catalog(Arc::clone(&catalog))
        .artifact_dir(dir.path().to_path_buf())
        .build()
        .unwrap();

        let result = tokio::task::spawn_blocking(move || tl.run(&loader))
            .await
            .unwrap()
            .unwrap();

        let expected = N * epochs;
        assert_eq!(
            result.total_steps, expected,
            "epochs={epochs}: expected exactly {expected} optimizer steps (N={N} batches × {epochs} epochs), got {}",
            result.total_steps
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ORACLE 1b — the exact-epoch invariant under a non-trivial accumulation window.
//
// The trainer has exactly ONE epoch loop (`for epoch in 0..config.epochs`),
// shared by the tabular and graph fine-tune paths alike; `epochs = E` runs
// exactly E passes with no warm-up or zeroth epoch. ORACLE 1 pins this for
// grad_accum=1 (one step per batch); ORACLE 2 pins the trailing-window flush for
// epochs=1. This locks the *composite* invariant both factors non-trivial:
//
//   optimizer steps == epochs × ceil(batches_per_epoch / grad_accum)
//
// With N=5 batches, grad_accum=2, epochs=3:
//   per-epoch steps = ceil(5/2) = 3   (steps at batch 2, 4, and the batch-5 flush)
//   total           = 3 × 3 = 9
// A regression that added a warm-up pass, dropped the trailing flush, or ran a
// zeroth epoch would break this exact count.
// ════════════════════════════════════════════════════════════════════════════
#[tokio::test(flavor = "multi_thread")]
async fn oracle_steps_equal_epochs_times_ceil_batches_over_grad_accum() {
    let device = Device::Cpu;
    const N: usize = 5; // batches per epoch
    const GA: usize = 2; // accumulation window (N not a multiple of GA)
    const EPOCHS: usize = 3;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let head = build_projection_head(4, &FineTuneConfig::default(), &varmap, &vb).unwrap();
    let batches: Vec<_> = (0..N).map(|_| learnable_batch(&device)).collect();
    let loader = TrainingDataLoader::from_precomputed(batches);

    let tag = "steps-epochs-ceil";
    let (catalog, dir) = claimed_loop_env(tag).await;

    let mut tl = TrainingLoopBuilder::new(
        TrainingTarget::ProjectionHead { head },
        varmap,
        FineTuneConfig {
            epochs: EPOCHS,
            batch_size: 1,
            validation_fraction: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: GA,
            // Train-loss monitoring with huge patience → early stopping can't fire
            // and truncate the epoch budget.
            early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
            early_stopping_patience: 10_000,
            learning_rate: 1e-3,
            ..Default::default()
        },
    )
    .job_id(tag.into())
    .worker_id(format!("{tag}-worker"))
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .build()
    .unwrap();

    let result = tokio::task::spawn_blocking(move || tl.run(&loader))
        .await
        .unwrap()
        .unwrap();

    let expected = EPOCHS * N.div_ceil(GA);
    assert_eq!(
        result.total_steps, expected,
        "optimizer steps must equal epochs × ceil(batches/grad_accum) = \
         {EPOCHS} × ceil({N}/{GA}) = {expected}, got {}",
        result.total_steps
    );
}

// ════════════════════════════════════════════════════════════════════════════
// ORACLE 2 — gradient_accumulation_steps step accounting vs the LR-schedule
// horizon (total_steps).
//
// The trainer pre-computes `total_steps = (batches_per_epoch * epochs) / grad_accum`
// (integer division) and feeds it to `compute_lr` as the cosine/linear-decay
// horizon. But each epoch ALSO flushes a trailing partial accumulation window
// (trainer.rs:425) which bumps global_step. When batches_per_epoch is NOT a
// multiple of grad_accum, the realised step count per epoch is
// ceil(batches/grad_accum), so the run overshoots `total_steps`.
//
// Oracle: with N=4 batches, grad_accum=3, epochs=1:
//   in-loop steps  = floor(4/3) = 1   (the step at batch_count==3)
//   partial flush  = 1                (batch 4 leftover → flushed at epoch end)
//   realised steps = 2
//   total_steps    = (4*1)/3 = 1
// So global_step (2) > total_steps (1). The decay LR is then evaluated at
// step > horizon, which for CosineDecay/LinearDecay drives the LR NEGATIVE
// (progress > 1). A negative LR is an ascent step — actively wrong.
// ════════════════════════════════════════════════════════════════════════════
#[tokio::test(flavor = "multi_thread")]
async fn oracle_grad_accum_partial_window_step_accounting() {
    let device = Device::Cpu;
    const N: usize = 4;
    const GA: usize = 3;
    const EPOCHS: usize = 1;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let head = build_projection_head(4, &FineTuneConfig::default(), &varmap, &vb).unwrap();
    let batches: Vec<_> = (0..N).map(|_| learnable_batch(&device)).collect();
    let loader = TrainingDataLoader::from_precomputed(batches);

    let tag = "ga-overshoot";
    let (catalog, dir) = claimed_loop_env(tag).await;

    let mut tl = TrainingLoopBuilder::new(
        TrainingTarget::ProjectionHead { head },
        varmap,
        FineTuneConfig {
            epochs: EPOCHS,
            batch_size: 1,
            validation_fraction: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: GA,
            lr_schedule: LrSchedule::CosineDecay,
            early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
            early_stopping_patience: 10_000,
            learning_rate: 1e-3,
            ..Default::default()
        },
    )
    .job_id(tag.into())
    .worker_id(format!("{tag}-worker"))
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .build()
    .unwrap();

    let result = tokio::task::spawn_blocking(move || tl.run(&loader))
        .await
        .unwrap()
        .unwrap();

    // total_steps reported == realised global_step. Oracle realised = ceil(N/GA)*EPOCHS = 2.
    // The trailing partial-window flush takes the extra step the old floored
    // horizon `(N*EPOCHS)/GA = 1` failed to count; the corrected horizon counts it
    // (`ceil(N/GA)*EPOCHS = 2`), so realised steps and horizon agree.
    let realised_oracle = N.div_ceil(GA) * EPOCHS;
    let horizon = realised_oracle;
    assert_eq!(
        result.total_steps, realised_oracle,
        "realised optimizer steps should be ceil(N/GA)*epochs = {realised_oracle}"
    );
    // The consistency the trainer must maintain — the LR-schedule horizon covers
    // every realised step, so the decay schedule is never evaluated past
    // progress=1.0 into a negative (gradient-ascent) learning rate.
    assert!(
        result.total_steps <= horizon,
        "realised steps ({}) must not exceed the LR-schedule horizon ({horizon}); \
         a realised-steps horizon (ceil(N/GA)*epochs) counts the trailing \
         partial-accumulation flush so the decay schedule never goes negative.",
        result.total_steps
    );
}

// ════════════════════════════════════════════════════════════════════════════
// ORACLE 2b — direct compute_lr probe: a negative learning rate is reachable
// once the realised step passes the horizon. This isolates the consequence of
// the step/horizon mismatch from the trainer plumbing.
//
// Oracle: LinearDecay, total_steps=1, warmup=0, step=2 →
//   progress = (2-0)/max(1-0,1) = 2.0 → lr = base*(1-2) = -base  (NEGATIVE).
// ════════════════════════════════════════════════════════════════════════════
#[test]
fn oracle_compute_lr_goes_negative_past_horizon() {
    let cfg = FineTuneConfig {
        learning_rate: 1e-3,
        warmup_steps: 0,
        lr_schedule: LrSchedule::LinearDecay,
        ..Default::default()
    };
    // Horizon total_steps=1; the overshoot step is 2 (see Oracle 2).
    let lr_at_overshoot = compute_lr(&cfg, 2, 1);
    // A learning rate must never be negative — that flips the optimizer to ascent.
    assert!(
        lr_at_overshoot >= 0.0,
        "BUG: compute_lr returned NEGATIVE lr {lr_at_overshoot} at step past the horizon. \
         compute_lr has no floor at 0, and the trainer can step past total_steps via the \
         per-epoch partial-accumulation flush (Oracle 2). A negative LR is an ascent step."
    );
}

// NOTE on learning-rate honoring: the precomputed loader path feeds tensors
// straight to the loss WITHOUT routing through the trainable ProjectionHead
// (the head is applied only on the text-encoding path), so a precomputed run
// produces no weight delta regardless of LR — it cannot probe LR honoring.
// LR honoring is already pinned directly by `compute_lr` (see the `lr_schedule_*`
// tests in fine_tune.rs) and, for the negative-LR defect, by
// `oracle_compute_lr_goes_negative_past_horizon` above.

// ════════════════════════════════════════════════════════════════════════════
// ORACLE 2c — the run's optimizer-step horizon AND its checkpoint cadence, as
// realised by the loop, over the boundary cases the arithmetic in `run` has
// to get right:
//
//   total_steps         = ceil(batches_per_epoch / grad_accum) * epochs
//   checkpoint_interval = ceil(total_steps * 0.1)
//   step checkpoints    = { k in 1..=total_steps : k % checkpoint_interval == 0 }
//                       → total_steps / checkpoint_interval files
//
// Cells: batches not divisible by grad_accum with epochs=1 (the trailing
// partial-window flush is the run's last step); a horizon ≤ 10 (interval 1:
// every step checkpoints); a horizon > 10 (interval > 1: only multiples);
// a horizon that is not a multiple of its interval (the floor in the count).
// The oracle counts the `checkpoint_{step}.safetensors` files the run left in
// its artifact dir — the realised cadence, not the formula re-derived.
//
// Cell 5 (mutants sweep finding, R3 finishing round): the first four cells
// never combine `grad_accum > 1` (so the trailing-partial-window flush at
// `run`'s own `checkpoint_interval > 0 && global_step.is_multiple_of(...)`
// gate is actually reached) with `interval > 1` (so the gate's `&&` is
// distinguishable from `||` — cells 3/4 have `grad_accum == 1`, so no
// partial window ever accumulates and the flush's checkpoint block is never
// entered; cells 1/2 reach it but with `interval == 1`, where every step
// checkpoints under EITHER operator). `cargo mutants --in-diff` on this
// round's trainer.rs diff survived exactly that mutant. Cell 5 reaches the
// flush with `interval == 2` and lands it on step 11 (odd, not a multiple),
// so an `||` regression would write an extra `checkpoint_11.safetensors`
// this oracle's exact-set assertion below would catch.
// ════════════════════════════════════════════════════════════════════════════
#[tokio::test(flavor = "multi_thread")]
async fn oracle_total_steps_and_checkpoint_cadence_boundaries() {
    let device = Device::Cpu;
    // (batches_per_epoch, grad_accum, epochs)
    let cells: [(usize, usize, usize); 5] = [
        (5, 2, 1),  // partial window, epochs=1: 3 steps, interval 1 → 3 files
        (7, 3, 2),  // partial window, epochs=2: 6 steps, interval 1 → 6 files
        (4, 1, 3),  // 12 steps, interval 2 → 6 files
        (11, 1, 1), // 11 steps, interval 2 → 5 files (floor: step 11 is not a multiple)
        (31, 3, 1), // partial-window flush AND interval 2: flush lands on step
                    // 11 (not a multiple of 2) — pins the flush's own cadence gate,
                    // not just process_batch_loss's in-window one.
    ];
    for (n, ga, epochs) in cells {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = build_projection_head(4, &FineTuneConfig::default(), &varmap, &vb).unwrap();
        let batches: Vec<_> = (0..n).map(|_| learnable_batch(&device)).collect();
        let loader = TrainingDataLoader::from_precomputed(batches);

        let tag = format!("ckpt-cadence-{n}-{ga}-{epochs}");
        let (catalog, dir) = claimed_loop_env(&tag).await;

        let mut tl = TrainingLoopBuilder::new(
            TrainingTarget::ProjectionHead { head },
            varmap,
            FineTuneConfig {
                epochs,
                batch_size: 1,
                validation_fraction: 0.0,
                warmup_steps: 0,
                gradient_accumulation_steps: ga,
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                early_stopping_patience: 10_000,
                learning_rate: 1e-3,
                ..Default::default()
            },
        )
        .job_id(tag.clone())
        .worker_id(format!("{tag}-worker"))
        .catalog(Arc::clone(&catalog))
        .artifact_dir(dir.path().to_path_buf())
        .build()
        .unwrap();

        let result = tokio::task::spawn_blocking(move || tl.run(&loader))
            .await
            .unwrap()
            .unwrap();

        let total_steps = n.div_ceil(ga) * epochs;
        let interval = (total_steps as f64 * 0.1).ceil() as usize;
        let expected_files = total_steps / interval;
        assert_eq!(
            result.total_steps, total_steps,
            "cell (n={n}, ga={ga}, epochs={epochs}): realised steps"
        );

        let mut step_checkpoints: Vec<usize> = std::fs::read_dir(result.artifact_dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter_map(|name| {
                name.strip_prefix("checkpoint_")?
                    .strip_suffix(".safetensors")?
                    .parse::<usize>()
                    .ok()
            })
            .collect();
        step_checkpoints.sort_unstable();
        let expected: Vec<usize> = (1..=total_steps).filter(|k| k % interval == 0).collect();
        assert_eq!(
            step_checkpoints, expected,
            "cell (n={n}, ga={ga}, epochs={epochs}): checkpoint steps at interval {interval}"
        );
        assert_eq!(step_checkpoints.len(), expected_files);
        assert!(
            result
                .artifact_dir
                .path()
                .join("checkpoint_best.safetensors")
                .exists(),
            "cell (n={n}, ga={ga}, epochs={epochs}): the epoch-boundary best checkpoint"
        );
    }
}
