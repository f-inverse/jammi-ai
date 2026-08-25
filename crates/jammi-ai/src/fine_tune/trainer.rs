//! Training loop: gradient descent with LR scheduling, early stopping, and checkpointing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::collections::HashMap;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::VarMap;
use jammi_db::catalog::Catalog;
use jammi_db::store::ArtifactStore;

use crate::fine_tune::adamw::{AdamW, ParamsAdamW};
use jammi_db::error::{JammiError, Result};

use super::data::{TextChunk, TrainingDataLoader};
use super::optimizer::{clip_and_step, DEFAULT_NORM_CHECK_INTERVAL};
use super::regression_loss::{crps_gaussian_loss, gaussian_nll_loss, pinball_loss, TargetScaler};
use super::resume::{
    capture_bundle, NamedMoments, RestoredCheckpoint, ResumeState, RESUME_STATE_SCHEMA_VERSION,
};
use super::target::TrainingTarget;
use super::{EarlyStoppingMetric, FineTuneConfig, LrSchedule};
use crate::model::{LoadedModel, ModelTask};

#[cfg(test)]
use std::sync::atomic::AtomicU64;

/// Test-only counter of device→host reads issued on the per-micro-batch
/// path this file owns (`process_batch_loss`'s post-backward loss read is the
/// only production one; `accumulate_sim_stats` must never move it — see its
/// doc). Mirrors `optimizer::SYNC_READ_COUNT`'s role for the clip: a CPU test
/// cannot observe a CUDA sync directly, so this is the structural proxy.
#[cfg(test)]
static PER_MICRO_BATCH_HOST_READ_COUNT: AtomicU64 = AtomicU64::new(0);

/// Snapshot [`PER_MICRO_BATCH_HOST_READ_COUNT`]. Test-only.
#[cfg(test)]
fn per_micro_batch_host_read_count() -> u64 {
    PER_MICRO_BATCH_HOST_READ_COUNT.load(Ordering::Relaxed)
}

/// Result of a completed training run.
///
/// The loop trains and persists the adapter into a worker-private local
/// directory, but does **not** write the job's terminal status, register the
/// output model, or publish the artifact to the object store — those are the
/// worker's single lease-guarded finalization. The worker reads the final
/// files out of [`Self::artifact_dir`], writes them to the artifact store under
/// a unique per-attempt prefix, and records that prefix as the model row's
/// `artifact_path` in the same compare-and-set that flips the job to
/// `completed`. The directory is a tempdir the result owns, so it is cleaned up
/// when the worker drops the result after publishing. The run metrics it
/// computed (final loss, step count, timestamps) are returned here so the
/// worker records them in that same compare-and-set.
#[derive(Debug)]
pub struct TrainingResult {
    /// The local directory holding the final adapter files
    /// (`adapter.safetensors` + `adapter_config.json`) plus run checkpoints. The
    /// worker reads the adapter files from here to publish them; the tempdir is
    /// removed on drop.
    pub artifact_dir: tempfile::TempDir,
    /// Best validation loss achieved.
    pub final_loss: f64,
    /// Total optimizer steps taken.
    pub total_steps: usize,
    /// The run metrics JSON the worker writes alongside the terminal status.
    pub metrics_json: String,
}

/// Compute the learning rate for a given step.
///
/// Warmup: linear ramp from 0 to base LR over `warmup_steps`.
/// After warmup: decay per `lr_schedule` (Constant, CosineDecay, or LinearDecay).
pub fn compute_lr(config: &FineTuneConfig, step: usize, total_steps: usize) -> f64 {
    let base_lr = config.learning_rate;

    // Warmup phase: linear ramp
    if step < config.warmup_steps {
        return base_lr * (step as f64 / config.warmup_steps.max(1) as f64);
    }

    // Decay phase. `progress` is clamped to [0, 1] and the returned lr floored at
    // 0 so the schedule is total-domain-valid for any step: stepping past the
    // horizon holds the lr at its end-of-schedule value rather than continuing
    // the curve into negative (gradient-ascent) territory.
    let decay_steps = total_steps.saturating_sub(config.warmup_steps);
    let decay_step = step - config.warmup_steps;
    let progress = (decay_step as f64 / decay_steps.max(1) as f64).clamp(0.0, 1.0);

    let lr = match config.lr_schedule {
        LrSchedule::Constant => base_lr,
        LrSchedule::CosineDecay => base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
        LrSchedule::LinearDecay => base_lr * (1.0 - progress),
    };
    lr.max(0.0)
}

/// Mutable per-epoch state passed into [`TrainingLoop::process_batch_loss`].
///
/// All four fields are borrowed mutably so the function can update batch
/// counts, loss accumulators, and the gradient store in place.
struct EpochState<'a> {
    batch_count: &'a mut usize,
    epoch_loss: &'a mut f64,
    accumulated_grads: &'a mut Option<GradStore>,
    global_step: &'a mut usize,
}

/// Immutable per-step context (except for the optimizer, which mutates on
/// every step). Constructed fresh for each call to
/// [`TrainingLoop::process_batch_loss`] and dropped at function return so the
/// caller can keep using `optimizer` directly between iterations.
struct StepContext<'a> {
    trainable_vars: &'a [Var],
    optimizer: &'a mut AdamW,
    checkpoint_dir: &'a Path,
    checkpoint_interval: usize,
    total_steps: usize,
    /// Micro-batches this epoch's loader yields. Needed so the trailing partial
    /// accumulation window divides its loss by its actual micro-batch count
    /// (`batches_per_epoch % grad_accum`) rather than the full `grad_accum` — the
    /// partial window averages over fewer micro-batches.
    batches_per_epoch: usize,
}

/// Running epoch-level cosine-similarity statistics, folded entirely on
/// device by [`TrainingLoop::accumulate_sim_stats`].
///
/// Reshaping what used to be three parallel variables
/// (`epoch_pos_sim: Option<Tensor>`, `epoch_neg_sim: Option<Tensor>`,
/// `triplet_batch_count: usize`) into one `Option<SimStats>` makes "a nonzero
/// count implies both running sums are populated" a STRUCTURAL invariant
/// instead of a runtime one pinned by an `.expect(...)` at the read site:
/// there is no way to construct a `SimStats` with `count > 0` and `pos`/`neg`
/// unset, so the epoch-boundary read can never observe the three variables
/// having drifted out of sync with each other.
struct SimStats {
    /// Running device-side sum of per-micro-batch mean positive-pair cosine
    /// similarity. Always a graph leaf (`track_op() == false`) — see
    /// [`TrainingLoop::accumulate_sim_stats`]'s doc.
    pos: Tensor,
    /// Running device-side sum of per-micro-batch mean negative-pair cosine
    /// similarity. Same leaf guarantee as `pos`.
    neg: Tensor,
    /// Number of triplet micro-batches folded into `pos`/`neg` so far this
    /// epoch — the divisor for the epoch-boundary average.
    count: usize,
}

/// The training loop: runs LoRA fine-tuning with gradient accumulation,
/// early stopping, LR scheduling, and checkpointing.
pub struct TrainingLoop {
    target: TrainingTarget,
    /// Provides the tokenizer for both target variants, plus the frozen
    /// forward path consumed by [`TrainingTarget::ProjectionHead`]. `None`
    /// is only valid when the data loader yields pre-built tensor batches
    /// (`is_precomputed()` is `true`) — used by trainer-internals tests.
    base_model: Option<Arc<LoadedModel>>,
    varmap: VarMap,
    config: FineTuneConfig,
    job_id: String,
    /// The lease holder's id (`claimed_by`). The run-start metrics write is
    /// gated on `claimed_by == worker_id AND status = 'running'`, so a worker
    /// whose lease was reclaimed mid-run cannot stamp `running` metrics over a
    /// job the winner already finalized.
    worker_id: String,
    catalog: Arc<Catalog>,
    /// The local directory training scratch (the per-run tempdir holding
    /// checkpoints and the final adapter) is created under. The run owns a
    /// fresh tempdir within it, so two workers training the same `job_id` never
    /// share a training-time path; the worker publishes the final files from
    /// there to the artifact store under a unique per-attempt prefix.
    artifact_dir: PathBuf,
    divergence_count: usize,
    /// Fixed, dataset-level target standardiser for the regression path, derived
    /// once from all training targets at the start of [`Self::run`]. `None` until
    /// the run computes it (and for every non-regression target). It maps each
    /// regression loss into a z-space the zero-initialised head can reach, while
    /// the head itself stays in raw space — so serving needs no de-standardisation.
    target_scaler: Option<TargetScaler>,
    device: Device,
    /// Cooperative-cancellation flag the worker's heartbeat task sets when the
    /// lease is lost. Checked at every epoch boundary; once set the loop bails
    /// without recording a terminal status, leaving the job for lease-based
    /// reclaim. A `spawn_blocking` thread cannot be force-aborted, so this is the
    /// coarsest safe interruption point.
    cancel: Arc<AtomicBool>,
    /// The durable artifact store the epoch-boundary resume checkpoint is written
    /// to (under `{job_id}/_resume/`). `None` disables durable checkpointing — the
    /// run trains but leaves nothing to resume from (used by trainer-internal
    /// tests that drive the loop without a worker/store).
    artifact_store: Option<Arc<ArtifactStore>>,
    /// A resume bundle this run restores from before the first epoch, or `None`
    /// for a from-scratch run. When present, training starts at
    /// `state.last_completed_epoch + 1` with weights, optimizer moments, scaler,
    /// and dropout positions restored.
    resume: Option<RestoredCheckpoint>,
}

/// Builder for [`TrainingLoop`].
pub struct TrainingLoopBuilder {
    target: TrainingTarget,
    base_model: Option<Arc<LoadedModel>>,
    varmap: VarMap,
    config: FineTuneConfig,
    job_id: Option<String>,
    worker_id: Option<String>,
    catalog: Option<Arc<Catalog>>,
    artifact_dir: Option<PathBuf>,
    device: Device,
    cancel: Arc<AtomicBool>,
    artifact_store: Option<Arc<ArtifactStore>>,
    resume: Option<RestoredCheckpoint>,
}

impl TrainingLoopBuilder {
    /// Start building a training loop with the chosen [`TrainingTarget`].
    /// Call [`Self::base_model`] before [`Self::build`] for the production
    /// path; omit it only when supplying a precomputed-batches data loader
    /// to the trainer (test affordance — the loader yields tensors directly
    /// instead of texts that need to be encoded).
    pub fn new(target: TrainingTarget, varmap: VarMap, config: FineTuneConfig) -> Self {
        Self {
            target,
            base_model: None,
            varmap,
            config,
            job_id: None,
            worker_id: None,
            catalog: None,
            artifact_dir: None,
            device: Device::Cpu,
            cancel: Arc::new(AtomicBool::new(false)),
            artifact_store: None,
            resume: None,
        }
    }

    /// Set the durable artifact store the epoch-boundary resume checkpoint is
    /// written to. Omit it for a run that should not checkpoint durably (a
    /// trainer-internal test).
    pub fn artifact_store(mut self, store: Arc<ArtifactStore>) -> Self {
        self.artifact_store = Some(store);
        self
    }

    /// Restore from a discovered resume bundle: training continues from the
    /// persisted epoch boundary instead of starting fresh.
    pub fn resume(mut self, restored: RestoredCheckpoint) -> Self {
        self.resume = Some(restored);
        self
    }

    /// Set the device all training tensors should live on.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the cooperative-cancellation flag the loop checks at every epoch
    /// boundary. The worker's heartbeat task sets it when the lease is lost so
    /// the loop bails and the job is left for reclaim. Omit it for a run that
    /// cannot be cancelled (the loop then uses a never-set flag).
    pub fn cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = cancel;
        self
    }

    /// Set the base model. Required for text-data training (supplies the
    /// tokenizer and, for `ProjectionHead` targets, the frozen forward
    /// pass).
    pub fn base_model(mut self, model: Arc<LoadedModel>) -> Self {
        self.base_model = Some(model);
        self
    }

    /// Set the job ID for catalog tracking.
    pub fn job_id(mut self, id: String) -> Self {
        self.job_id = Some(id);
        self
    }

    /// Set the lease holder's id (`claimed_by`). The run-start metrics write is
    /// gated on it so a reclaimed (zombie) worker cannot disturb the job row.
    pub fn worker_id(mut self, id: String) -> Self {
        self.worker_id = Some(id);
        self
    }

    /// Set the catalog for status persistence.
    pub fn catalog(mut self, catalog: Arc<Catalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// Set the artifact directory for checkpoint and adapter storage.
    pub fn artifact_dir(mut self, dir: PathBuf) -> Self {
        self.artifact_dir = Some(dir);
        self
    }

    /// Build the training loop. All infrastructure params must be set.
    pub fn build(self) -> Result<TrainingLoop> {
        let job_id = self
            .job_id
            .ok_or_else(|| JammiError::FineTune("TrainingLoopBuilder: job_id required".into()))?;
        let worker_id = self.worker_id.ok_or_else(|| {
            JammiError::FineTune("TrainingLoopBuilder: worker_id required".into())
        })?;
        let catalog = self
            .catalog
            .ok_or_else(|| JammiError::FineTune("TrainingLoopBuilder: catalog required".into()))?;
        let artifact_dir = self.artifact_dir.ok_or_else(|| {
            JammiError::FineTune("TrainingLoopBuilder: artifact_dir required".into())
        })?;
        // Audit advisory (post-4aa1303 round): a whole-run, one-time check
        // — not a hot-path cost — that no two of this target's dropout
        // layers hash to the same `layer_id` (see the method's own doc).
        // Runs before the loop is handed back to the caller, so a
        // collision is a hard, typed refusal at construction time, never
        // a silent correlated-dropout defect discovered later.
        self.target.assert_dropout_layer_ids_are_collision_free()?;
        Ok(TrainingLoop {
            target: self.target,
            base_model: self.base_model,
            varmap: self.varmap,
            config: self.config,
            job_id,
            worker_id,
            catalog,
            artifact_dir,
            divergence_count: 0,
            target_scaler: None,
            device: self.device,
            cancel: self.cancel,
            artifact_store: self.artifact_store,
            resume: self.resume,
        })
    }
}

impl TrainingLoop {
    /// Run the training loop. Returns the path to the saved adapter.
    ///
    /// Dual-path:
    /// - With `base_model`: text-based loaders encode through the frozen base
    ///   model, project through LoRA, and compute loss on the projected embeddings.
    /// - Without `base_model`: precomputed tensor batches go directly to loss.
    pub fn run(&mut self, data_loader: &TrainingDataLoader) -> Result<TrainingResult> {
        // Stamp run-start metrics under the lease guard. The claim already set
        // the status to `running`; this records `started_at` only while this
        // worker still holds the lease (`claimed_by == worker_id AND status =
        // 'running'`). A worker whose lease was reclaimed mid-run (a zombie) thus
        // cannot regress a job the winner already finalized back to `running`.
        let started_at = chrono::Utc::now().to_rfc3339();
        let metrics_json = serde_json::json!({"started_at": started_at}).to_string();
        tokio::runtime::Handle::current().block_on(self.catalog.mark_training_running(
            &self.job_id,
            &self.worker_id,
            Some(&metrics_json),
        ))?;

        // Split training/validation
        let total_rows = data_loader.len();
        let (train_loader, val_loader) = data_loader.split(self.config.validation_fraction)?;

        // A validation split can come out empty even when `validation_fraction`
        // is non-zero, because the split rounds: `round(rows * fraction)` is 0
        // for any dataset small enough — at the default 0.1, fewer than five
        // rows. `FineTuneConfig::validate` refuses the explicit zero but cannot
        // see the row count, so the row-dependent case has to be refused here,
        // where it is known. Without this the run monitors a loss that is never
        // measured, stops on the first non-improvement, and publishes the
        // epoch-0 adapter as its result.
        if self.config.early_stopping_metric == EarlyStoppingMetric::ValLoss
            && val_loader.is_empty()
        {
            return Err(JammiError::FineTune(format!(
                "early_stopping_metric=val_loss requires a non-empty validation split, but \
                 validation_fraction={} over {total_rows} row(s) holds out none. Set \
                 early_stopping_metric=train_loss, raise validation_fraction, or train on \
                 more rows.",
                self.config.validation_fraction
            )));
        }

        // Reduce all training targets into one fixed standardiser before the loop
        // (a regression run only). Computed from the train split — the val split
        // is held out — so every regression-loss call scores in a z-space the
        // zero-init head can reach, while the head stays in raw space.
        self.target_scaler = match train_loader.regression_targets() {
            Some(targets) if !targets.is_empty() => {
                let n = targets.len();
                let tensor = Tensor::from_vec(targets, (n,), &self.device)
                    .map_err(|e| JammiError::FineTune(format!("scaler targets tensor: {e}")))?;
                Some(TargetScaler::from_targets(&tensor)?)
            }
            _ => None,
        };

        // The LR-schedule horizon is the number of optimizer steps the run will
        // actually take, not the floor of `batches / grad_accum`. Each epoch takes
        // one step per full accumulation window plus one trailing step for the
        // partial window when `batches_per_epoch` is not a multiple of
        // `grad_accum`, i.e. `ceil(batches / grad_accum)` steps. Counting the
        // realised steps keeps `compute_lr`'s `progress` within [0, 1] for every
        // step the loop takes, and makes the reported `result.total_steps` equal
        // this horizon. Computed after the train/validation split, since
        // `validation_fraction` changes `train_batches_per_epoch`.
        let train_batches_per_epoch = train_loader.num_batches(self.config.batch_size);
        let total_steps = train_batches_per_epoch
            .div_ceil(self.config.gradient_accumulation_steps.max(1))
            * self.config.epochs;
        let checkpoint_interval = (total_steps as f64 * 0.1).ceil() as usize;

        // `total_steps` above is the ACCUMULATION-WINDOW arm's horizon (one
        // step per full `grad_accum`-sized window, `div_ceil` also covering a
        // trailing partial window). GradCache takes exactly ONE optimizer
        // step per EPOCH (`run_gradcache_epoch`), so its true horizon is
        // `self.config.epochs`, not `total_steps` — using `total_steps` there
        // was B2's bug: `global_step + 1 >= total_steps` was false on
        // GradCache's actual last step whenever `num_batches > 1`, so the
        // run's real final step never forced `clip_and_step`'s non-finite
        // check the way a run shorter than `DEFAULT_NORM_CHECK_INTERVAL`
        // needs it to (see `optimizer::clip_and_step`'s doc). Computed ONCE,
        // here, and threaded to every `is_last_step` call site below instead
        // of each site re-deriving it — `gradcache_eligible()` is a fixed
        // property of `self.config`/`self.base_model` for the run's whole
        // lifetime (it never toggles per-epoch), so one branch here is
        // correct for every epoch. The `!train_loader.is_precomputed()` guard
        // mirrors the epoch loop's own arm precedence below (`is_precomputed()`
        // is checked BEFORE `gradcache_eligible()` there), so a test run that
        // happens to set both a `base_model` and a precomputed loader still
        // gets the accumulation-window horizon, matching the arm it actually
        // takes.
        //
        // LAST-STEP LATTICE — every arm/edge `is_last_step` must be right
        // for, and this fix's disposition for each:
        //
        //  - **accumulation window** (`process_batch_loss`'s flush): one step
        //    per full `grad_accum`-sized window. `total_steps`'s `div_ceil`
        //    already counts these correctly; unaffected by this fix.
        //  - **partial last batch** (trailing window smaller than
        //    `grad_accum`, `batches_per_epoch % grad_accum != 0`): the same
        //    `div_ceil` above already accounts for it as one extra step;
        //    unaffected by this fix.
        //  - **GradCache**: one step per epoch. THE BUG THIS FIX CLOSES —
        //    `total_optimizer_steps` uses `self.config.epochs` on this arm
        //    instead of `total_steps`.
        //  - **mined loader** (`trainer.rs`'s hard-negative refresh; see
        //    `mine_hard_negative_loader`'s doc): a row whose mined pool is
        //    entirely excluded is DROPPED, so a mined epoch's row (and
        //    therefore batch/window) count can differ from `train_loader`'s —
        //    what `total_steps` was computed from, once, before the loop.
        //    This is a PRE-EXISTING desync this fix does NOT resolve: an
        //    epoch whose mining drops enough rows to change its window count
        //    can still make `is_last_step` wrong on that epoch, because the
        //    true per-epoch window count is only known once that epoch's
        //    `text_chunks()`/mined triplets are built, not upfront. Flagged
        //    as a known, tracked gap — not silently absorbed into "fixed" by
        //    this commit.
        //  - **early stopping** (`break` on patience exhaustion): an
        //    early-stopped run's actual last step is whatever `global_step`
        //    reached before the `break`, which `total_optimizer_steps`
        //    (fixed upfront from `self.config.epochs`) does not know ahead of
        //    time — `is_last_step` therefore never fires `true` on an
        //    early-stopped run's TRUE final step. Judged ACCEPTABLE, not a
        //    residual defect: `run` always restores `checkpoint_best` (saved
        //    at whichever epoch boundary had the best monitored loss so far)
        //    before saving the final adapter — on EVERY run, early-stopped or
        //    not (see the `best_path.exists()` restore below) — so a
        //    divergence on an early-stopped run's true last step cannot
        //    silently reach the published artifact: a non-finite
        //    `monitor_loss` makes `monitor_loss < best_val_loss` `false`
        //    (family F: `NaN > c` is `false`, and so is `NaN < c`), so that
        //    epoch is never saved as `checkpoint_best`. The diagnostic gap (a
        //    divergence not immediately surfaced as a typed `Err` on that
        //    exact step) remains open — the modulo cadence still catches it
        //    within `DEFAULT_NORM_CHECK_INTERVAL` steps on a long-enough run,
        //    and `step == 1` catches an early one — but the PUBLISHED
        //    adapter's correctness does not depend on it.
        //  - **cancel** (`self.cancel` checked at the epoch boundary): a
        //    cancelled run returns `Err` BEFORE the epoch that would have
        //    followed cancellation ever calls `process_batch_loss` or
        //    `run_gradcache_epoch` — no `is_last_step` is evaluated for a
        //    step that never runs, so there is nothing to pin. No adapter is
        //    published on this path (the worker's lease-guarded finalization
        //    never runs for an `Err` return).
        //  - **resume** (`global_step` restored from a checkpoint): `self.config
        //    .epochs`/`total_steps` are both absolute counts fixed at THIS
        //    run's start, independent of where `global_step` restarts from —
        //    resume changes where `global_step` counts FROM, not what it is
        //    compared against, so a resumed run's `is_last_step` uses exactly
        //    the same horizon a fresh run would. Already correct; unaffected
        //    by this fix.
        let total_optimizer_steps = if !train_loader.is_precomputed() && self.gradcache_eligible() {
            self.config.epochs
        } else {
            total_steps
        };

        // Snapshot the trainable variables ONCE. `VarMap::all_vars()` iterates a
        // HashMap, so a second call could return a different order — and `AdamW`'s
        // optimizer state is positional in the order it was built from. Building
        // the optimizer and `trainable_vars` from one snapshot keeps the gradient
        // accumulation, clipping, and the optimizer's moment vector all aligned to
        // the same parameter order within this process. The cross-process
        // correlation that makes resume safe is `optim_param_names` below — the
        // moments serialize/restore BY NAME, never by this in-process order.
        let trainable_vars = self.varmap.all_vars();

        // weight_decay matches train_embedding_model.py: AdamW(weight_decay=0.01).
        let mut optimizer = AdamW::new(
            trainable_vars.clone(),
            ParamsAdamW {
                lr: self.config.learning_rate,
                weight_decay: self.config.weight_decay,
                ..Default::default()
            },
        )
        .map_err(|e| JammiError::FineTune(format!("Optimizer init: {e}")))?;

        // The parameter NAME for each entry of `optimizer.state()`'s moment
        // vector, in that exact order. `AdamW::new` keeps the float subset of
        // `trainable_vars` in order, and `state()` reports moments in that same
        // order — so zipping `optim_param_names` with the moment vector keys every
        // `(m, v)` by its parameter name. This is the R1 fix: a `VarMap`'s
        // `all_vars()` order is not stable across processes, so the resume bundle
        // must never serialize moments positionally; it serializes them by this
        // name. The names come from `varmap.data()` keyed by tensor identity, so
        // the correlation is independent of any HashMap iteration order.
        let optim_param_names = self.optimizer_param_names(&trainable_vars)?;

        // Restore from a discovered resume bundle (weights + optimizer moments +
        // scaler + dropout positions). The persisted scaler is authoritative — it
        // overrides the just-computed one so a source mutated between crash and
        // resume cannot perturb the de-standardisation (R7). Returns the epoch the
        // resumed run starts at (`last_completed + 1`) and its step counter.
        let (start_epoch, mut global_step) = match self.resume.take() {
            Some(restored) => {
                self.restore_from_checkpoint(restored, &mut optimizer, &optim_param_names)?
            }
            None => (0, 0),
        };
        let mut best_val_loss = f64::MAX;
        let mut patience_counter = 0;
        // Train into a fresh worker-private tempdir, never a shared path: two
        // workers on the same `job_id` must not share a training-time file.
        // Checkpoints and the final adapter land here; the worker publishes the
        // final files to the artifact store under a unique per-attempt prefix
        // after the loop returns, on a finalize-CAS win. The tempdir sits under
        // `artifact_dir` so it shares the deployment's training scratch disk.
        std::fs::create_dir_all(&self.artifact_dir)?;
        let artifact_tmp = tempfile::Builder::new()
            .prefix("train-")
            .tempdir_in(&self.artifact_dir)?;
        let checkpoint_dir = artifact_tmp.path().to_path_buf();

        // Hard negatives mined from the current model, re-mined every
        // `refresh_every` epochs. Held across epochs so a non-refresh epoch
        // reuses the last mining (the staleness/cost trade).
        let mut mined_loader: Option<TrainingDataLoader> = None;

        // NOT held: candle's `CUDA_GRAPH_HTOD_CACHE`.
        //
        // candle-core 0.11.0's `CudaDevice`-scoped HtoD-cache-enabling method
        // (`cuda_backend/device.rs:92-95`) turns on a THREAD-LOCAL,
        // content-keyed `HashMap<(DeviceId, TypeId, bytes), Box<dyn Any>>` of
        // every H2D upload ≤ `CUDA_GRAPH_HTOD_CACHE_MAX_BYTES` (4096) bytes
        // (`device.rs:31,209`) — every distinct `input_ids`/mask/score/label
        // micro-batch the trainer ever uploads becomes a permanent entry.
        // `CudaGraphHtodCacheGuard::drop` (`device.rs:45-52`) only decrements
        // a re-entrancy depth counter; nothing in candle's public API clears
        // or bounds the map's contents, and 0.11.0 is the newest release on
        // crates.io, so there is no pin bump that adds one. Holding that
        // guard for a whole run would grow the map by one entry per distinct
        // micro-batch shape/content for the run's lifetime, unbounded, on a
        // pooled `spawn_blocking` thread that outlives the training job
        // (family E: bound the term that grows, not a sum around it — there
        // is no bound to add here without candle exposing one). The
        // `cuda_htod_cache_premise_pin::candle_core_is_still_the_audited_0_11_0`
        // test below fails the moment `candle-core` moves off `0.11.0`, so a
        // future upgrade is forced to re-read `cuda_backend/device.rs` for an
        // eviction/bounded-capacity API before that guard is ever reinstated.
        //
        // What the trainer pays instead, per micro-batch, uncached: the
        // dims/strides H2D uploads `params_from_layout` issues per kernel
        // launch (`cuda_backend/mod.rs:63-88` in the same release) plus the
        // tiny scalar constants `clip_gradients` materializes (`.minimum
        // (1.0)`; see `optimizer::clip_gradients`'s doc for that op count).
        // A per-step tiny-H2D-copy count for a representative LoRA config
        // (e.g. ModernBERT-large r16) is a real, GPU-measured number, not one
        // to re-derive from this comment — measure it per pod run against a
        // committed census artifact when one exists in this tree, rather than
        // citing a figure or a doc path that is not (currently unresolvable
        // from this repository). Nothing in this file changes that count.

        for epoch in start_epoch..self.config.epochs {
            // Cooperative cancellation: the worker's heartbeat sets this when the
            // lease is lost. Bail at the epoch boundary, leaving the job for
            // lease-based reclaim rather than recording a (wrong) terminal status.
            if self.cancel.load(Ordering::Relaxed) {
                return Err(JammiError::FineTune(
                    "training cancelled: lease lost before epoch boundary".into(),
                ));
            }
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            // Accumulated gradients across micro-batches. Seeded from the first
            // backward call (avoids needing a private GradStore::new()).
            let mut accumulated_grads: Option<GradStore> = None;
            // Running device-side cosine-similarity stats — read back to
            // `f64` exactly once, at the epoch boundary below (see
            // `Self::accumulate_sim_stats`'s and `SimStats`'s docs).
            let mut sim_stats: Option<SimStats> = None;

            // Re-mine hard negatives at refresh boundaries. Mining replaces the
            // epoch's data with (anchor, positive, mined-negative) triplets fed
            // through the MNRL hard-negative path.
            if self.mining_eligible()
                && super::hard_negative_miner::should_refresh(
                    epoch,
                    self.config.hard_negatives.refresh_every,
                )
            {
                mined_loader = Some(self.mine_hard_negative_loader(&train_loader)?);
            }
            // The loader this epoch trains on: the freshly/last-mined triplets
            // when mining is active, otherwise the original data.
            let epoch_loader: &TrainingDataLoader = mined_loader.as_ref().unwrap_or(&train_loader);

            if epoch_loader.is_precomputed() {
                // Test path: direct tensor batches, no encoding.
                let train_batches = epoch_loader.batches(self.config.batch_size)?;
                for batch in train_batches {
                    let batch = batch?;
                    Self::accumulate_sim_stats(&batch, &mut sim_stats);
                    let loss = self.compute_loss(&batch)?;
                    self.process_batch_loss(
                        loss,
                        EpochState {
                            batch_count: &mut batch_count,
                            epoch_loss: &mut epoch_loss,
                            accumulated_grads: &mut accumulated_grads,
                            global_step: &mut global_step,
                        },
                        StepContext {
                            trainable_vars: &trainable_vars,
                            optimizer: &mut optimizer,
                            checkpoint_dir: &checkpoint_dir,
                            checkpoint_interval,
                            total_steps: total_optimizer_steps,
                            batches_per_epoch: train_batches_per_epoch,
                        },
                    )?;
                }
            } else if self.gradcache_eligible() {
                // GradCache path: the whole dataset is one in-batch-negative
                // batch, chunked at `batch_size` for memory. One optimiser step
                // per epoch over the full negative pool.
                let lr = compute_lr(&self.config, global_step, total_steps);
                optimizer.set_learning_rate(lr);
                let loss_val = self.run_gradcache_epoch(
                    epoch_loader,
                    &trainable_vars,
                    &mut optimizer,
                    total_optimizer_steps,
                    global_step,
                )?;
                epoch_loss += loss_val;
                batch_count += 1;
                global_step += 1;
                if checkpoint_interval > 0 && global_step % checkpoint_interval == 0 {
                    self.save_checkpoint(&checkpoint_dir, global_step)?;
                }
            } else {
                // Production path: encode text through the target, then compute loss.
                let text_chunks = epoch_loader.text_chunks(self.config.batch_size);
                for chunk in &text_chunks {
                    let batch = self.encode_chunk(chunk)?;
                    let loss = self.compute_loss(&batch)?;
                    Self::accumulate_sim_stats(&batch, &mut sim_stats);
                    self.process_batch_loss(
                        loss,
                        EpochState {
                            batch_count: &mut batch_count,
                            epoch_loss: &mut epoch_loss,
                            accumulated_grads: &mut accumulated_grads,
                            global_step: &mut global_step,
                        },
                        StepContext {
                            trainable_vars: &trainable_vars,
                            optimizer: &mut optimizer,
                            checkpoint_dir: &checkpoint_dir,
                            checkpoint_interval,
                            total_steps: total_optimizer_steps,
                            batches_per_epoch: train_batches_per_epoch,
                        },
                    )?;
                }
            }

            // Flush any remaining micro-batch gradients that didn't fill a full
            // accumulation window (last partial window of the epoch). This
            // block is only reached via `accumulated_grads` — which only
            // `process_batch_loss` populates — so it is never reached on the
            // GradCache arm; `total_optimizer_steps` equals `total_steps`
            // here (see the lattice doc above `total_optimizer_steps`'s
            // definition), used for consistency with the other two call
            // sites rather than because this arm needs the distinction.
            if let Some(mut acc) = accumulated_grads.take() {
                let lr = compute_lr(&self.config, global_step, total_steps);
                optimizer.set_learning_rate(lr);
                // `total_optimizer_steps` is the whole run's ACTUAL
                // optimizer-step horizon for the arm this run takes (see the
                // lattice doc above its definition), so
                // `global_step + 1 >= total_optimizer_steps` names the run's
                // actual final optimizer step, not just this epoch's — the
                // non-finite check must not be skippable by a run shorter
                // than `DEFAULT_NORM_CHECK_INTERVAL` steps (see
                // `clip_and_step`'s doc).
                let is_last_step = global_step + 1 >= total_optimizer_steps;
                clip_and_step(
                    &mut optimizer,
                    &trainable_vars,
                    &mut acc,
                    self.config.max_grad_norm,
                    DEFAULT_NORM_CHECK_INTERVAL,
                    global_step + 1,
                    is_last_step,
                )?;
                global_step += 1;
            }

            let avg_train_loss = epoch_loss / batch_count.max(1) as f64;
            // The ONE host read for the whole epoch's sim stats — every
            // per-micro-batch contribution above stayed on device (see
            // `Self::accumulate_sim_stats`'s doc). `SimStats` makes "a
            // populated count implies both sums are populated" structural, so
            // there is no `.expect()` here to fire.
            let (avg_pos_sim, avg_neg_sim) =
                match &sim_stats {
                    Some(stats) => {
                        let pos_sum: f32 = stats.pos.to_scalar().map_err(|e| {
                            JammiError::FineTune(format!("epoch pos sim read: {e}"))
                        })?;
                        let neg_sum: f32 = stats.neg.to_scalar().map_err(|e| {
                            JammiError::FineTune(format!("epoch neg sim read: {e}"))
                        })?;
                        (
                            pos_sum as f64 / stats.count as f64,
                            neg_sum as f64 / stats.count as f64,
                        )
                    }
                    None => (0.0, 0.0),
                };

            // Validation — skip entirely when monitoring train loss to avoid wasting time.
            // `None` when no validation pass ran. Not `0.0`: a sentinel that
            // shares a type with a real measurement is a measurement everywhere
            // downstream.
            let avg_val_loss: Option<f64> = match self.config.early_stopping_metric {
                EarlyStoppingMetric::TrainLoss => None,
                EarlyStoppingMetric::ValLoss => {
                    // Disable dropout for the validation pass.
                    self.target.set_training(false);
                    let val_loss = self.evaluate(&val_loader)?;
                    self.target.set_training(true);
                    Some(val_loss)
                }
            };

            // Decide which loss to monitor for early stopping.
            let (monitor_loss, monitor_label) = match self.config.early_stopping_metric {
                EarlyStoppingMetric::TrainLoss => (avg_train_loss, "train"),
                EarlyStoppingMetric::ValLoss => (
                    avg_val_loss.expect("ValLoss runs always measure — guarded at the split"),
                    "val",
                ),
            };

            let lr = compute_lr(&self.config, global_step, total_steps);
            tracing::info!(
                epoch,
                avg_train_loss,
                avg_val_loss,
                avg_pos_sim,
                avg_neg_sim,
                monitor_loss,
                monitor_label,
                global_step,
                lr,
                "Epoch complete"
            );

            // Early stopping on the chosen metric.
            if monitor_loss < best_val_loss {
                best_val_loss = monitor_loss;
                patience_counter = 0;
                self.save_checkpoint_tagged(&checkpoint_dir, "best")?;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    tracing::info!(
                        epoch,
                        patience_counter,
                        best_loss = best_val_loss,
                        monitor_label,
                        "Early stopping: no improvement for {} epochs",
                        patience_counter
                    );
                    break;
                }
            }

            // Durable resume checkpoint at the epoch boundary. Gated on the
            // lease: a worker whose lease was reclaimed during this epoch must not
            // overwrite the durable checkpoint with stale state. The trainer
            // already checks `cancel` at the TOP of the next iteration; checking it
            // again HERE, before the write, closes the window where a lease lost
            // mid-epoch would still let this (now-zombie) attempt regress the
            // shared `{job_id}/_resume/` bundle below the lease-winner's epoch (R5).
            // A `None` store disables durable checkpointing (trainer-internal tests).
            if !self.cancel.load(Ordering::Relaxed) {
                self.save_resume_checkpoint(
                    &checkpoint_dir,
                    epoch,
                    global_step,
                    &optimizer,
                    &optim_param_names,
                )?;
            }
        }

        // Restore best checkpoint before saving final adapter
        let best_path = checkpoint_dir.join("checkpoint_best.safetensors");
        if best_path.exists() {
            self.load_checkpoint(&best_path)?;
        }

        // Save the final adapter — both target variants persist their
        // trainable weights alongside a `SavedAdapter` metadata JSON.
        let final_weights = self.target.named_trainable_weights()?;
        // The form is persisted exactly when the scaler is — both are the
        // regression head's de-standardisation state. A non-regression head has
        // no scaler and no form, so its adapter config round-trips unchanged.
        let regression_form = self.target_scaler.map(|_| self.regression_form());
        let saved = self
            .target
            .saved_adapter(&self.config, self.target_scaler, regression_form);
        jammi_lora::save_adapter(&checkpoint_dir, &final_weights, &saved)
            .map_err(|e| JammiError::FineTune(format!("Save adapter: {e}")))?;

        // The loop does not write the terminal status, register the output
        // model, or publish the artifact to the object store. All three are the
        // worker's single lease-guarded finalization: it writes the final files
        // to the artifact store under a unique per-attempt prefix, registers the
        // model row pointing at that prefix, and runs the compare-and-set that
        // flips the job to `completed` only while it still holds the lease.
        // Computing the run metrics here (and returning them) keeps the rich loss
        // / step / timing detail the worker records in that same CAS.
        let completed_at = chrono::Utc::now().to_rfc3339();
        let early_stopping_metric_label = match self.config.early_stopping_metric {
            EarlyStoppingMetric::TrainLoss => "train_loss",
            EarlyStoppingMetric::ValLoss => "val_loss",
        };
        let metrics_json = serde_json::json!({
            "final_loss": best_val_loss,
            "early_stopping_metric": early_stopping_metric_label,
            "total_steps": global_step,
            "started_at": started_at,
            "completed_at": completed_at,
        })
        .to_string();

        Ok(TrainingResult {
            artifact_dir: artifact_tmp,
            final_loss: best_val_loss,
            total_steps: global_step,
            metrics_json,
        })
    }

    /// Whether this run should mine hard negatives: `mine` is on, the objective
    /// is the in-batch-negative one (mining only feeds that path), and a base
    /// model is present to embed the corpus. Mining replaces the epoch's data
    /// with mined triplets, so it requires a text loader — the precomputed test
    /// path skips it.
    fn mining_eligible(&self) -> bool {
        self.base_model.is_some()
            && self.config.hard_negatives.mine
            && matches!(
                self.config.embedding_loss,
                Some(super::EmbeddingLoss::MultipleNegativesRanking { .. })
            )
    }

    /// Mine hard negatives from the current model and build a triplet loader of
    /// `(anchor, positive, mined-negative)` rows.
    ///
    /// Indexes the positives as the candidate corpus (jammi's own cosine ANN),
    /// then streams the anchors in `batch_size` chunks: each chunk is embedded,
    /// mined against the index, and dropped before the next. Only one chunk of
    /// anchor vectors is resident at a time, and the positive vectors live solely
    /// in the index (no second copy) — peak working-set is bounded by the index
    /// plus one batch, not by holding the whole anchor + positive corpora in RAM.
    /// The positive and its `exclude_hops`-hop neighbourhood are excluded per
    /// anchor as the false-negative guard. A row whose pool is entirely excluded
    /// is dropped; if mining yields no usable rows the original loader is
    /// returned unchanged rather than training on an empty set.
    fn mine_hard_negative_loader(
        &mut self,
        loader: &TrainingDataLoader,
    ) -> Result<TrainingDataLoader> {
        use super::hard_negative_miner::{AnchorQuery, Candidate, HardNegativeMiner};

        let (anchors, positives, _existing_neg) = loader.in_batch_negative_texts()?;
        if anchors.is_empty() {
            return Ok(TrainingDataLoader::from_triplets(Vec::new()));
        }

        // Embed with dropout off — the model state the negatives are mined
        // against. Returns owned per-row vectors, consumed into the index or the
        // per-batch anchor queries below.
        self.target.set_training(false);
        let embed = |this: &Self, texts: &[String]| -> Result<Vec<Vec<f32>>> {
            let t = this.encode_texts(texts)?;
            let t = if t.dtype() == DType::F32 {
                t
            } else {
                t.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("mine dtype: {e}")))?
            };
            t.to_vec2::<f32>()
                .map_err(|e| JammiError::FineTune(format!("mine to_vec2: {e}")))
        };
        let batch = self.config.batch_size.max(1);
        let result = (|| {
            // Candidate corpus = the positives, keyed by row index so a mined id
            // maps back to its positive text. The positive vectors are moved into
            // the index and dropped here — the index is their only owner.
            let candidates: Vec<Candidate> = embed(self, &positives)?
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| Candidate {
                    id: i.to_string(),
                    embedding,
                })
                .collect();
            let miner = HardNegativeMiner::build(&candidates, self.config.hard_negatives)?;
            drop(candidates);

            // Stream the anchors in batches: embed one chunk, mine it, drop it.
            // Only one chunk of anchor vectors is resident at any moment.
            let mut rows = Vec::with_capacity(anchors.len());
            for (chunk_idx, chunk) in anchors.chunks(batch).enumerate() {
                let base = chunk_idx * batch;
                let anchor_vecs = embed(self, chunk)?;
                for (offset, anchor_vec) in anchor_vecs.into_iter().enumerate() {
                    let i = base + offset;
                    let query = AnchorQuery {
                        embedding: anchor_vec,
                        positive_id: i.to_string(),
                    };
                    let mined = miner.mine(&query)?;
                    if let Some(neg_id) = mined.first() {
                        let neg_idx: usize = neg_id
                            .parse()
                            .map_err(|e| JammiError::FineTune(format!("mine id parse: {e}")))?;
                        rows.push((
                            anchors[i].clone(),
                            positives[i].clone(),
                            positives[neg_idx].clone(),
                        ));
                    }
                }
            }
            Ok::<_, JammiError>(rows)
        })();
        self.target.set_training(true);
        let rows = result?;

        if rows.is_empty() {
            // Nothing minable (e.g. every candidate excluded) — fall back to the
            // original data rather than train on an empty epoch.
            tracing::warn!(
                job_id = %self.job_id,
                "hard-negative mining produced no rows; training on original data this epoch"
            );
            return Ok(self.clone_text_loader(loader));
        }
        Ok(TrainingDataLoader::from_triplets(rows))
    }

    /// Re-materialise a text loader's in-batch-negative rows as a fresh loader,
    /// used as the mining fall-back. Pairs become a `Pairs` loader; triplets
    /// keep their explicit negatives.
    fn clone_text_loader(&self, loader: &TrainingDataLoader) -> TrainingDataLoader {
        match loader.in_batch_negative_texts() {
            Ok((anchors, positives, Some(negatives))) => {
                let rows = anchors
                    .into_iter()
                    .zip(positives)
                    .zip(negatives)
                    .map(|((a, p), n)| (a, p, n))
                    .collect();
                TrainingDataLoader::from_triplets(rows)
            }
            Ok((anchors, positives, None)) => {
                TrainingDataLoader::from_pairs(anchors.into_iter().zip(positives).collect())
            }
            Err(_) => TrainingDataLoader::from_triplets(Vec::new()),
        }
    }

    /// Whether this run should take the GradCache path: `cached` is on, the
    /// configured objective is the in-batch-negative one, and a base model is
    /// present to re-encode chunks (the test/precomputed path has no encoder).
    /// `cached` only enlarges an *in-batch-negative* pool, so it is a no-op for
    /// graded-pair or triplet-margin objectives — those take the standard path.
    fn gradcache_eligible(&self) -> bool {
        self.base_model.is_some()
            && self.config.cached
            && matches!(
                self.config.embedding_loss,
                Some(super::EmbeddingLoss::MultipleNegativesRanking { .. })
            )
    }

    /// Run one GradCache epoch: treat the whole training set as a single
    /// in-batch-negative batch, compute the MNRL loss and its parameter
    /// gradient in two memory-bounded passes, then take one optimiser step.
    /// Returns the epoch's MNRL loss value for logging.
    ///
    /// The negative pool is the entire dataset — that is the point of GradCache
    /// over plain gradient accumulation — so each anchor is contrasted against
    /// every other positive (and every explicit hard negative). The per-chunk
    /// re-encode keeps peak activation memory at one chunk regardless of the
    /// pool size; the gradient equals the single-pass one (pinned by the
    /// gradient-equivalence test in the `gradcache` module).
    fn run_gradcache_epoch(
        &mut self,
        train_loader: &TrainingDataLoader,
        trainable_vars: &[Var],
        optimizer: &mut AdamW,
        total_optimizer_steps: usize,
        global_step: usize,
    ) -> Result<f64> {
        use super::gradcache::{gradcache_backward, EncodeGroup};

        let (anchors, positives, negatives) = train_loader.in_batch_negative_texts()?;
        let scale = self.mnrl_scale();
        let has_negatives = negatives.is_some();

        // Dropout off for the whole GradCache region so the two encode passes
        // (and the logging re-encode) agree. Toggled while no encode closure
        // borrows `self`, so it does not collide with the immutable borrows
        // below. Concretely, this means a WHOLE GradCache epoch trains with
        // LoRA dropout OFF — an undocumented (until now) behavior difference
        // from the standard per-batch training path, flagged by #352; left
        // as-is here (out of this commit's scope — a follow-up question,
        // not a defect this commit fixes).
        self.target.set_training(false);

        // Immutable-borrow region: the encode closures borrow `self`, so no
        // `&mut self` call may appear until they are dropped at the block end.
        let outcome: Result<(GradStore, f64)> = (|| {
            let enc = |texts: &[String], start: usize, len: usize| -> Result<Tensor> {
                self.encode_texts(&texts[start..start + len])
            };
            let a_enc = |start: usize, len: usize| enc(&anchors, start, len);
            let p_enc = |start: usize, len: usize| enc(&positives, start, len);

            let mut groups = vec![
                EncodeGroup {
                    rows: anchors.len(),
                    encode: &a_enc,
                },
                EncodeGroup {
                    rows: positives.len(),
                    encode: &p_enc,
                },
            ];
            // A triplet GradCache run also embeds the explicit hard negatives as
            // a third group so they join the row-direction candidate set.
            let n_enc;
            if let Some(ref negs) = negatives {
                n_enc = move |start: usize, len: usize| enc(negs, start, len);
                groups.push(EncodeGroup {
                    rows: negs.len(),
                    encode: &n_enc,
                });
            }

            let loss_fn = |reps: &[Tensor]| -> Result<Tensor> {
                let neg = if has_negatives { Some(&reps[2]) } else { None };
                mnrl_loss(&reps[0], &reps[1], neg, scale, true)
            };

            let grads = gradcache_backward(
                &groups,
                self.config.batch_size.max(1),
                &loss_fn,
                trainable_vars,
            )?;

            // Loss value for logging from a no-grad re-encode of the full batch
            // — cheap relative to the two-pass backward and outside its graph.
            let a_rep = self.encode_texts(&anchors)?;
            let p_rep = self.encode_texts(&positives)?;
            let neg_rep = match &negatives {
                Some(negs) => Some(self.encode_texts(negs)?),
                None => None,
            };
            let loss = mnrl_loss(&a_rep, &p_rep, neg_rep.as_ref(), scale, true)?;
            let loss = if loss.dtype() == DType::F32 {
                loss
            } else {
                loss.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("GradCache loss dtype: {e}")))?
            };
            let loss_val = loss
                .to_scalar::<f32>()
                .map_err(|e| JammiError::FineTune(format!("GradCache loss scalar: {e}")))?
                as f64;
            Ok((grads, loss_val))
        })();

        self.target.set_training(true);
        let (mut grads, loss_val) = outcome?;

        // GradCache takes exactly ONE optimizer step per EPOCH, not one per
        // accumulation window — the caller passes `total_optimizer_steps` as
        // `self.config.epochs` on this arm specifically BECAUSE of that (see
        // the lattice doc at its call site in `run`, above the definition of
        // `total_optimizer_steps`). Before B2's fix, this call site was
        // passed the accumulation-window arm's `total_steps`
        // (`ceil(batches / grad_accum) * epochs`) instead, which
        // overcounts GradCache's real per-epoch step count whenever a epoch
        // has more than one micro-batch/accumulation-window worth of data —
        // making `global_step + 1 >= total_steps` false on GradCache's actual
        // last step, silently skipping the run's final non-finite check on
        // any GradCache run shorter than `DEFAULT_NORM_CHECK_INTERVAL` steps
        // (every multi-epoch GradCache run, since it takes exactly one step
        // per epoch and `DEFAULT_NORM_CHECK_INTERVAL` is 50).
        let is_last_step = global_step + 1 >= total_optimizer_steps;
        clip_and_step(
            optimizer,
            trainable_vars,
            &mut grads,
            self.config.max_grad_norm,
            DEFAULT_NORM_CHECK_INTERVAL,
            global_step + 1,
            is_last_step,
        )?;

        Ok(loss_val)
    }

    /// Encode a slice of texts into a `[batch, hidden]` embedding tensor,
    /// dispatched on the active [`TrainingTarget`]:
    ///
    /// - `ProjectionHead`: run the texts through the frozen base model to
    ///   produce pooled embeddings, then project through the head's first
    ///   LoRA layer (shared with the audio path via
    ///   [`Self::project_frozen_embedding`]).
    /// - `EncoderAdapters`: tokenize the texts via the base model's tokenizer,
    ///   then forward through the LoRA-injected encoder directly (the encoder
    ///   does its own pooling and normalisation).
    fn encode_texts(&self, texts: &[String]) -> Result<Tensor> {
        let base = self
            .base_model
            .as_ref()
            .ok_or_else(|| JammiError::FineTune("encode_texts requires a base model".into()))?;
        match &self.target {
            TrainingTarget::ProjectionHead { .. } => {
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let arr = Arc::new(StringArray::from(text_refs)) as ArrayRef;
                self.project_frozen_embedding(base, arr, ModelTask::TextEmbedding)
            }
            TrainingTarget::EncoderAdapters(state) => {
                let encoder = &state.encoder;
                let tokenizer = match base.as_ref() {
                    crate::model::LoadedModel::Candle(m) => m
                        .tokenizer
                        .as_ref()
                        .ok_or_else(|| JammiError::FineTune("No tokenizer in base model".into()))?,
                    _ => return Err(JammiError::FineTune(
                        "Encoder-adapters training requires a Candle base model with a tokenizer"
                            .into(),
                    )),
                };

                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let effective_max = self.config.max_seq_length.min(encoder.max_seq_length());
                let encoding = tokenizer.encode_batch(&text_refs, Some(effective_max))?;

                let rows = encoding.input_ids.len();
                let cols = encoding.input_ids.first().map_or(0, |v| v.len());

                let input_ids = Tensor::from_vec(
                    encoding
                        .input_ids
                        .into_iter()
                        .flatten()
                        .collect::<Vec<u32>>(),
                    (rows, cols),
                    &self.device,
                )
                .map_err(|e| JammiError::FineTune(format!("input_ids tensor: {e}")))?;

                let attention_mask = Tensor::from_vec(
                    encoding
                        .attention_masks
                        .into_iter()
                        .flatten()
                        .collect::<Vec<u32>>(),
                    (rows, cols),
                    &self.device,
                )
                .map_err(|e| JammiError::FineTune(format!("attention_mask tensor: {e}")))?;

                encoder
                    .forward(&input_ids, &attention_mask)
                    .map_err(|e| JammiError::FineTune(format!("Encoder forward: {e}")))
            }
        }
    }

    /// Encode a slice of audio clips into a `[batch, hidden]` embedding
    /// tensor through the frozen audio base model and the LoRA projection
    /// head.
    ///
    /// Each clip is encoded audio bytes (WAV/FLAC/MP3/Ogg); the base model
    /// owns decode → resample → log-mel → audio-tower forward, exactly as the
    /// `encode_audio_query` inference path does. Only the `ProjectionHead`
    /// target trains an audio adapter — LoRA injected *inside* an audio
    /// encoder is not supported, so `EncoderAdapters` here is a typed error
    /// rather than a silent wrong path.
    fn encode_audio(&self, clips: &[Vec<u8>]) -> Result<Tensor> {
        let base = self
            .base_model
            .as_ref()
            .ok_or_else(|| JammiError::FineTune("encode_audio requires a base model".into()))?;
        match &self.target {
            TrainingTarget::ProjectionHead { .. } => {
                let clip_refs: Vec<&[u8]> = clips.iter().map(|c| c.as_slice()).collect();
                let arr = Arc::new(BinaryArray::from(clip_refs)) as ArrayRef;
                self.project_frozen_embedding(base, arr, ModelTask::AudioEmbedding)
            }
            TrainingTarget::EncoderAdapters(_) => Err(JammiError::FineTune(
                "Audio fine-tuning trains a projection head on a frozen audio encoder; \
                 LoRA injected inside the audio encoder is not supported. \
                 Leave `target_modules` empty for audio tasks."
                    .into(),
            )),
        }
    }

    /// Run a content column through the frozen base model for `task`, then
    /// project the pooled embeddings through the projection head's first LoRA
    /// layer. Shared by the text and audio projection-head paths — the only
    /// difference between modalities is the Arrow array type and the
    /// `ModelTask`, both supplied by the caller.
    fn project_frozen_embedding(
        &self,
        base: &Arc<LoadedModel>,
        content: ArrayRef,
        task: ModelTask,
    ) -> Result<Tensor> {
        let head = match &self.target {
            TrainingTarget::ProjectionHead { head } => head,
            TrainingTarget::EncoderAdapters(_) => {
                return Err(JammiError::FineTune(
                    "project_frozen_embedding is only valid for a projection-head target".into(),
                ))
            }
        };
        let output = base
            .forward(&[content], task)
            .map_err(|e| JammiError::FineTune(format!("Encode: {e}")))?;
        let n = output.shapes[0].0;
        let dim = output.shapes[0].1;
        let raw = Tensor::from_vec(output.float_outputs[0].clone(), (n, dim), &self.device)
            .map_err(|e| JammiError::FineTune(format!("Encode tensor: {e}")))?;
        head.layers[0]
            .1
            .forward(&raw)
            .map_err(|e| JammiError::FineTune(format!("LoRA projection: {e}")))
    }

    /// Encode a text chunk into a `TrainingBatch` ready for loss computation.
    /// Encode several text groups in ONE forward, returning one
    /// `[group_rows, hidden]` tensor per group.
    ///
    /// The groups are concatenated before tokenisation and split after pooling,
    /// so a triplet micro-batch costs one encoder forward instead of three.
    /// Measured on an A100 with ModernBERT-large (batch 8, seq 128, bf16, LoRA
    /// r=16 on Wqkv/Wo/Wi, dropout off): 0.715 s/step -> 0.593 s/step, and peak
    /// device memory 39.6 GB -> 35.8 GB.
    ///
    /// The result is identical to encoding each group separately. Joining pads
    /// every group to one common length rather than to its own longest, and that
    /// extra padding is inert: pad positions are masked out of attention and out
    /// of the pooling mean, RoPE positions are absolute, and the sliding-window
    /// band is an absolute distance — so no real token's attended set or
    /// position encoding depends on how much padding trails it.
    ///
    /// Peak memory does not rise from holding all groups at once: every group's
    /// graph was already alive simultaneously, since the loss consumes all of
    /// them before `backward`.
    fn encode_groups(&self, groups: &[&Vec<String>]) -> Result<Vec<Tensor>> {
        let rows: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        let joined: Vec<String> = groups.iter().flat_map(|g| g.iter().cloned()).collect();
        let all = self.encode_texts(&joined)?;

        let mut out = Vec::with_capacity(groups.len());
        let mut offset = 0usize;
        for n in rows {
            out.push(
                all.narrow(0, offset, n)
                    .map_err(|e| JammiError::FineTune(format!("split encoded groups: {e}")))?,
            );
            offset += n;
        }
        Ok(out)
    }

    fn encode_chunk(&self, chunk: &TextChunk) -> Result<super::data::TrainingBatch> {
        let encode = |texts: &Vec<String>| -> Result<Tensor> { self.encode_texts(texts) };

        match chunk {
            TextChunk::Contrastive {
                texts_a,
                texts_b,
                scores,
            } => {
                let mut e = self.encode_groups(&[texts_a, texts_b])?.into_iter();
                let proj_a = e.next().expect("group 0");
                let proj_b = e.next().expect("group 1");
                let scores_tensor = Tensor::from_vec(scores.clone(), (scores.len(),), &self.device)
                    .map_err(|e| JammiError::FineTune(format!("Scores tensor: {e}")))?;
                Ok(super::data::TrainingBatch::Contrastive {
                    embeddings_a: proj_a,
                    embeddings_b: proj_b,
                    scores: scores_tensor,
                })
            }
            TextChunk::Pairs { anchors, positives } => {
                let mut e = self.encode_groups(&[anchors, positives])?.into_iter();
                let proj_a = e.next().expect("group 0");
                let proj_p = e.next().expect("group 1");
                Ok(super::data::TrainingBatch::Pairs {
                    anchors: proj_a,
                    positives: proj_p,
                })
            }
            TextChunk::Triplet {
                anchors,
                positives,
                negatives,
            } => {
                let mut e = self
                    .encode_groups(&[anchors, positives, negatives])?
                    .into_iter();
                let proj_a = e.next().expect("group 0");
                let proj_p = e.next().expect("group 1");
                let proj_n = e.next().expect("group 2");
                Ok(super::data::TrainingBatch::Triplet {
                    anchor: proj_a,
                    positive: proj_p,
                    negative: proj_n,
                })
            }
            TextChunk::AudioTriplet {
                anchors,
                positives,
                negatives,
            } => {
                // Audio triplets reuse the triplet contrastive objective
                // verbatim — only the encode step differs (audio bytes →
                // frozen audio tower → projection head, vs text → text tower).
                let proj_a = self.encode_audio(anchors)?;
                let proj_p = self.encode_audio(positives)?;
                let proj_n = self.encode_audio(negatives)?;
                Ok(super::data::TrainingBatch::Triplet {
                    anchor: proj_a,
                    positive: proj_p,
                    negative: proj_n,
                })
            }
            TextChunk::Classification { texts, labels } => {
                let proj = encode(texts)?;
                let labels_tensor = Tensor::from_vec(labels.clone(), (labels.len(),), &self.device)
                    .map_err(|e| JammiError::FineTune(format!("Labels tensor: {e}")))?;
                Ok(super::data::TrainingBatch::Classification {
                    embeddings: proj,
                    labels: labels_tensor,
                })
            }
            TextChunk::Ner { .. } => Err(JammiError::FineTune(
                "NER fine-tuning is not yet available. \
                 Token-level training requires sequence-level encoding."
                    .into(),
            )),
            TextChunk::Regression { texts, targets } => {
                let proj = encode(texts)?;
                // Score in standardized (z) space: the head emits its RAW z-output
                // (no de-standardise), and the target is z-scored with the run's
                // scaler. The optimizer then sees O(1) residuals regardless of the
                // target's raw scale — de-standardisation is moved entirely to the
                // serve path (mirroring the in-context predictor's
                // `target_context_z`/`destandardize_distribution` split).
                let head_out = self.head_forward(&proj)?;
                let scaler = self.target_scaler.as_ref().ok_or_else(|| {
                    JammiError::FineTune(
                        "regression batch reached without a target scaler (run did not set one)"
                            .into(),
                    )
                })?;
                let z_targets: Vec<f32> = targets
                    .iter()
                    .map(|&y| scaler.standardize_value(y as f64) as f32)
                    .collect();
                let target_tensor = Tensor::from_vec(z_targets, (targets.len(),), &self.device)
                    .map_err(|e| JammiError::FineTune(format!("Target tensor: {e}")))?;
                Ok(super::data::TrainingBatch::Regression {
                    input: head_out,
                    target: target_tensor,
                })
            }
        }
    }

    /// The predictive distribution form this run's regression head emits, read
    /// off the configured objective: `Pinball` trains the quantile head over the
    /// configured levels; every other arm trains the parametric Gaussian head.
    /// This is the single gaussian-vs-quantile dispatch — the de-standardisation
    /// (here and at serving) and the persisted head metadata all derive from it,
    /// so the served form can never disagree with the trained one.
    fn regression_form(&self) -> crate::inference::adapter::DistributionForm {
        use super::target::StandardizableHead;
        use crate::inference::adapter::DistributionForm;
        // Route the gaussian-vs-quantile decision through the offset-bearing-head
        // classifier — the same closed enum the standardisation-contract guards
        // and oracle pin — so the trained form, the persisted form, and the
        // contract's notion of "which head this is" are one mapping. The quantile
        // arm carries this run's configured levels.
        let loss = self.config.regression_loss.unwrap_or_default();
        if StandardizableHead::for_regression_loss(loss).is_gaussian() {
            DistributionForm::Gaussian
        } else {
            DistributionForm::Quantile {
                levels: self.config.quantile_levels.clone(),
            }
        }
    }

    /// Apply the distributional regression head to projected embeddings,
    /// producing the head's RAW `(batch, k)` z-space output — the parameter the
    /// LoRA layer actually learns, with **no** de-standardisation. Mirrors
    /// [`Self::classify`]: only a `ProjectionHead` target with a second (head)
    /// layer can regress.
    ///
    /// The training loss scores this z-output directly against a z-scored target
    /// (`embed_chunk` z-scores the target via [`TargetScaler::standardize_value`]),
    /// so the optimizer sees O(1) residuals regardless of the target's raw scale.
    /// De-standardisation (`μ_y + σ_y·z` on the mean/quantile columns, `σ_y·σ_z`
    /// on the served σ) happens **only at serve** — the backend's de-standardising
    /// affine and the inference adapter's σ scaling — so this method is the single
    /// raw-head forward shared by training, determinism, and resume.
    fn head_forward(&self, embeddings: &Tensor) -> Result<Tensor> {
        match &self.target {
            TrainingTarget::ProjectionHead { head } if head.layers.len() > 1 => head.layers[1]
                .1
                .forward(embeddings)
                .map_err(|e| JammiError::FineTune(format!("LoRA regression head: {e}"))),
            TrainingTarget::ProjectionHead { .. } => Err(JammiError::FineTune(
                "No regression head in projection target".into(),
            )),
            TrainingTarget::EncoderAdapters(_) => Err(JammiError::FineTune(
                "Regression with encoder adapters is not supported".into(),
            )),
        }
    }

    /// Accumulate cosine similarity stats from a triplet batch for epoch-level
    /// logging, entirely ON DEVICE — this runs once per micro-batch, so it
    /// sits on the same hot path `process_batch_loss` does, and must issue
    /// ZERO `to_scalar`/`to_vec*` calls (unlike the old version, which did two
    /// per call, BEFORE `backward` ever ran). Each batch's per-pair cosine
    /// similarity is reduced to a mean (`mean_all`, a device scalar tensor)
    /// and folded into [`SimStats::pos`]/[`SimStats::neg`] with a device add —
    /// a fixed left-to-right fold across the epoch's micro-batches (family J:
    /// deterministic reduction order), mirroring `optimizer::clip_gradients`'s
    /// fold. The running sums are read back to `f64` exactly ONCE, at the
    /// epoch boundary (see the `avg_pos_sim`/`avg_neg_sim` computation in
    /// `run`), dividing by [`SimStats::count`] there — so this function moves
    /// the *number* of per-micro-batch host reads for the sim-stats path from
    /// 2 to 0, not just their timing relative to `backward`.
    ///
    /// **Graph retention (B1).** In production `anchor`/`positive`/`negative`
    /// come from `encode_chunk` over the LoRA `Var`s, so `cosine_similarity`'s
    /// output is TRACKED (`track_op() == true`): its forward subgraph reaches
    /// all the way back to the LoRA weights. Folding a tracked scalar into an
    /// epoch-lifetime accumulator with `(prev + new)` and no `detach()` would
    /// retain every micro-batch's activation graph until the epoch boundary —
    /// `sorted_nodes().len()` on the accumulator growing by one micro-batch's
    /// subgraph size on every call, unbounded over the epoch, even though
    /// `process_batch_loss` already dropped its own graph via `backward()`
    /// immediately after computing the loss. `batch_mean_f32` below
    /// `.detach()`s the per-batch mean BEFORE it ever touches the
    /// accumulator, and the fold detaches its own output too (belt and
    /// suspenders: two detached operands under candle's `BackpropOp::new2`
    /// never re-attach — `op` stays `None` when neither operand tracks — but
    /// detaching both ends keeps that invariant true even if the fold's
    /// implementation changes to something that isn't a plain `+`). The
    /// result: [`SimStats`] is always a graph LEAF (`track_op() == false`) at
    /// every point in the epoch, regardless of how many micro-batches have
    /// folded into it — and, being neither `is_variable()` nor carrying an
    /// `op`, its own `sorted_nodes()` is EMPTY (candle's `sorted_nodes` only
    /// pushes a node when it is a `Var` or reaches one; a detached non-`Var`
    /// leaf is neither), not a length of `1`.
    ///
    /// **Precision.** The running sum is an on-device `f32` accumulation of
    /// per-micro-batch means, each already averaged from a cosine similarity
    /// in `[-1, 1]`; the epoch-boundary read (`to_scalar::<f32>()`) divides
    /// that `f32` sum by `count` in `f64`. The old code accumulated in host
    /// `f64` directly. Every candle op this fold issues (`mean_all`, `+`) is a
    /// single IEEE-754 `f32` rounding per element per op — no compensated
    /// (Kahan) summation — so after `N` folds the `f32` running sum's error
    /// versus an exact real-number sum is bounded by `O(N · eps_f32)` on
    /// values whose magnitude never exceeds `N` (the per-batch means are each
    /// in `[-1, 1]`, so the partial sum after `k` folds is in `[-k, k]`,
    /// `eps_f32 ≈ 1.19e-7`) — this is the standard worst-case bound for
    /// unrationalized floating-point summation (no cancellation-aware claim
    /// beyond it), and it is a strictly larger error than the old host-`f64`
    /// path's (`O(N · eps_f64)`, `eps_f64 ≈ 2.22e-16`) for the same `N`. This
    /// is an epoch-logging metric, not a scored loss or a persisted number a
    /// contract pins, so the wider `f32` error band is an accepted trade for
    /// paying zero per-micro-batch host reads.
    ///
    /// Non-triplet batches are silently ignored. Errors in stat computation
    /// (from `cosine_similarity` or the fold) are swallowed so a GPU issue
    /// never aborts training just because of a logging metric — but only
    /// all-or-nothing per batch: if either accumulator's fold would fail, the
    /// batch contributes to NEITHER accumulator nor `count`, so the two
    /// running sums and the count they are later divided by never drift out
    /// of sync (unchanged from before the [`SimStats`] reshape below — the
    /// reshape only makes "count > 0 implies both sums are populated"
    /// structural instead of an `.expect()`-pinned runtime invariant).
    fn accumulate_sim_stats(batch: &super::data::TrainingBatch, stats: &mut Option<SimStats>) {
        let super::data::TrainingBatch::Triplet {
            anchor,
            positive,
            negative,
        } = batch
        else {
            return;
        };

        let batch_mean_f32 = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
            let sim = cosine_similarity(a, b)?;
            let mean = sim
                .mean_all()
                .map_err(|e| JammiError::FineTune(format!("{e}")))?;
            let mean = if mean.dtype() == DType::F32 {
                mean
            } else {
                mean.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("{e}")))?
            };
            // Detach BEFORE this scalar ever reaches the epoch-lifetime
            // accumulator (see the doc's "Graph retention (B1)" section) —
            // this is a logging mean, never a scored loss, so it must never
            // hold the forward graph open.
            Ok(mean.detach())
        };

        let (Ok(ps), Ok(ns)) = (
            batch_mean_f32(anchor, positive),
            batch_mean_f32(anchor, negative),
        ) else {
            return;
        };

        let fold = |acc: &Tensor, new: &Tensor| -> Result<Tensor> {
            (acc + new)
                .map(|t| t.detach())
                .map_err(|e| JammiError::FineTune(format!("{e}")))
        };

        let next = match stats.as_ref() {
            None => Ok(SimStats {
                pos: ps,
                neg: ns,
                count: 1,
            }),
            Some(prev) => (|| -> Result<SimStats> {
                Ok(SimStats {
                    pos: fold(&prev.pos, &ps)?,
                    neg: fold(&prev.neg, &ns)?,
                    count: prev.count + 1,
                })
            })(),
        };

        if let Ok(next) = next {
            *stats = Some(next);
        }
    }

    /// Process a single batch loss: divergence detection, gradient accumulation
    /// via immediate backward, and optimizer step every N micro-batches.
    ///
    /// Each call computes `loss.backward()` immediately so the activation graph
    /// is freed at the end of every micro-batch. Gradients are accumulated in an
    /// `Option<GradStore>` (seeded from the first micro-batch's backward result,
    /// which avoids needing the private `GradStore::new()`) and an optimizer step
    /// is taken once every `gradient_accumulation_steps` micro-batches.
    fn process_batch_loss(
        &mut self,
        loss: Tensor,
        epoch: EpochState<'_>,
        ctx: StepContext<'_>,
    ) -> Result<()> {
        let loss_f32 = if loss.dtype() == DType::F32 {
            loss.clone()
        } else {
            loss.to_dtype(DType::F32)
                .map_err(|e| JammiError::FineTune(format!("Loss dtype cast: {e}")))?
        };

        // Gradient-accumulation window bookkeeping, computed from the
        // PROSPECTIVE batch index (this micro-batch counts unless it turns
        // out diverged below) so the loss scale is known before `backward` —
        // without reading the loss off the device first. `epoch.batch_count`
        // itself is only committed once divergence is known (below): a
        // diverged micro-batch must not advance the window, matching the
        // pre-existing skip semantics exactly.
        //
        // A full accumulation window averages over `grad_accum` micro-batches, so
        // each one's loss is divided by `grad_accum`. The epoch's trailing window
        // — when `batches_per_epoch` is not a multiple of `grad_accum` — contains
        // only `batches_per_epoch % grad_accum` micro-batches, so those divide by
        // that smaller count to keep the window's gradient a true average rather
        // than under-scaling it by the full `grad_accum`. `candidate_batch_count`
        // is the 1-based index this micro-batch WOULD occupy within the epoch.
        let grad_accum = self.config.gradient_accumulation_steps.max(1);
        let candidate_batch_count = *epoch.batch_count + 1;
        let partial_window = ctx.batches_per_epoch % grad_accum;
        let in_trailing_partial =
            partial_window != 0 && candidate_batch_count > ctx.batches_per_epoch - partial_window;
        let scale = if in_trailing_partial {
            partial_window as f64
        } else {
            grad_accum as f64
        };
        let scaled_loss =
            (&loss / scale).map_err(|e| JammiError::FineTune(format!("Loss scale: {e}")))?;

        // `backward` is issued BEFORE the loss is read off the device: the
        // old order (`to_scalar` first, for divergence detection) forced the
        // host to wait for the forward pass before even starting backward's
        // kernel launches, stalling the pipeline mid-step on every single
        // batch. Backward's launches (async on CUDA) now go out first; the
        // D2H read below only has to wait for whatever of the forward pass
        // isn't already done by the time backward finishes issuing — and
        // this is the ONE remaining sync in this loop's per-batch path (the
        // grad-clip sync is gone; see `optimizer::clip_gradients`). Releasing
        // the activation graph here (not later) is still what keeps
        // `gradient_accumulation_steps > 1` from growing memory proportional
        // to the micro-batch count.
        let new_grads = scaled_loss
            .backward()
            .map_err(|e| JammiError::FineTune(format!("Backward: {e}")))?;

        #[cfg(test)]
        PER_MICRO_BATCH_HOST_READ_COUNT.fetch_add(1, Ordering::Relaxed);
        let loss_val = loss_f32
            .to_scalar::<f32>()
            .map_err(|e| JammiError::FineTune(format!("Loss scalar: {e}")))?
            as f64;

        // Divergence detection. A diverged run returns the typed error and the
        // worker records the terminal `failed` status — terminal writes are the
        // worker's single authority, never the loop's.
        //
        // Post-W5-PR5 the regression-arm losses train in z-space (residuals O(1)),
        // so the numeric `>100` branch is now LESS discriminating on finite
        // divergence for those arms — it rarely fires because a healthy z-space
        // regression loss stays O(1)–O(10). The `is_nan()` branch is therefore the
        // load-bearing backstop for the regression arms (an overconfidence collapse
        // or a NaN gradient). The threshold still guards the non-regression arms
        // (CoSENT/MNRL/triplet/CE), whose magnitudes are unchanged.
        //
        // A diverged micro-batch's `new_grads` are dropped here without being
        // merged into `epoch.accumulated_grads` — the extra `backward` this
        // batch cost (versus the old skip-before-backward order) is spent
        // only on the rare diverged batch, never on the healthy common case.
        if loss_val.is_nan() || loss_val > 100.0 {
            self.divergence_count += 1;
            if self.divergence_count >= 3 {
                return Err(JammiError::FineTune(
                    "Training diverged: loss was NaN or >100 for 3 consecutive batches".into(),
                ));
            }
            return Ok(());
        }
        self.divergence_count = 0;

        *epoch.epoch_loss += loss_val;
        *epoch.batch_count = candidate_batch_count;

        // Merge new_grads into the running accumulator.
        // The accumulator is seeded from the first backward call to avoid
        // needing the private GradStore::new().
        match epoch.accumulated_grads {
            None => {
                *epoch.accumulated_grads = Some(new_grads);
            }
            Some(ref mut acc) => {
                for var in ctx.trainable_vars.iter() {
                    let t: &Tensor = var;
                    if let Some(g_new) = new_grads.get(t) {
                        if let Some(g_acc) = acc.remove(t) {
                            let summed = (&g_acc + g_new)
                                .map_err(|e| JammiError::FineTune(format!("Grad acc: {e}")))?;
                            acc.insert(t, summed);
                        } else {
                            acc.insert(t, g_new.clone());
                        }
                    }
                }
            }
        }

        // Optimizer step every N micro-batches.
        if (*epoch.batch_count).is_multiple_of(self.config.gradient_accumulation_steps) {
            let lr = compute_lr(&self.config, *epoch.global_step, ctx.total_steps);
            ctx.optimizer.set_learning_rate(lr);

            if let Some(mut acc) = epoch.accumulated_grads.take() {
                // See the flush-window call site's doc: `ctx.total_steps` is
                // the whole run's optimizer-step horizon, so this names the
                // run's actual final optimizer step, not just this epoch's.
                let is_last_step = *epoch.global_step + 1 >= ctx.total_steps;
                clip_and_step(
                    ctx.optimizer,
                    ctx.trainable_vars,
                    &mut acc,
                    self.config.max_grad_norm,
                    DEFAULT_NORM_CHECK_INTERVAL,
                    *epoch.global_step + 1,
                    is_last_step,
                )?;
            }

            *epoch.global_step += 1;

            // Checkpoint
            if ctx.checkpoint_interval > 0
                && (*epoch.global_step).is_multiple_of(ctx.checkpoint_interval)
            {
                self.save_checkpoint(ctx.checkpoint_dir, *epoch.global_step)?;
            }
        }

        Ok(())
    }

    /// Compute loss for a training batch.
    ///
    /// Contrastive pairs `(a, b, score)` dispatch on the configured
    /// [`EmbeddingLoss`]: CoSENT (default), AnglE, or cosine-MSE — every
    /// graded-pair objective. `Pairs` rows `(anchor, positive)` always train
    /// with [Multiple-Negatives-Ranking](mnrl_loss): the in-batch negatives
    /// *are* the format's contrast. `Triplet` rows use the triplet-margin
    /// objective unless `MultipleNegativesRanking` is selected, in which case
    /// the explicit negatives are appended to the in-batch similarity matrix
    /// (the DPR recipe).
    ///
    /// `MultipleNegativesRanking` is an in-batch-negative objective over
    /// `(anchor, positive)` rows, not a graded-pair one. Selecting it for a
    /// scored `Contrastive` batch is a batch/loss mismatch, so it is a typed
    /// error rather than a silent fall-through to a different loss. The
    /// triplet-margin variant on a graded `Contrastive` batch is the same
    /// mismatch and is rejected the same way.
    ///
    /// When `matryoshka_dims` is set, the chosen embedding objective is
    /// evaluated at each prefix dimension and the losses summed, so the leading
    /// embedding coordinates carry the most information (truncatable at serve
    /// time). The wrapper composes over the objective once — every embedding
    /// loss inherits it.
    fn compute_loss(&self, batch: &super::data::TrainingBatch) -> Result<Tensor> {
        match batch {
            super::data::TrainingBatch::Contrastive {
                embeddings_a,
                embeddings_b,
                scores,
            } => self.matryoshka_wrap(&[embeddings_a, embeddings_b], &|dims| {
                self.contrastive_loss(&dims[0], &dims[1], scores)
            }),
            super::data::TrainingBatch::Pairs { anchors, positives } => self
                .matryoshka_wrap(&[anchors, positives], &|dims| {
                    mnrl_loss(&dims[0], &dims[1], None, self.mnrl_scale(), true)
                }),
            super::data::TrainingBatch::Triplet {
                anchor,
                positive,
                negative,
            } => match self.config.embedding_loss {
                Some(super::EmbeddingLoss::MultipleNegativesRanking { .. }) => self
                    .matryoshka_wrap(&[anchor, positive, negative], &|dims| {
                        mnrl_loss(&dims[0], &dims[1], Some(&dims[2]), self.mnrl_scale(), true)
                    }),
                _ => self.matryoshka_wrap(&[anchor, positive, negative], &|dims| {
                    self.triplet_loss(&dims[0], &dims[1], &dims[2])
                }),
            },
            super::data::TrainingBatch::Classification { embeddings, labels } => {
                let logits = self.classify(embeddings)?;
                self.cross_entropy_loss(&logits, labels)
            }
            super::data::TrainingBatch::Ner {
                hidden_states,
                labels,
            } => self.ner_loss(hidden_states, labels),
            super::data::TrainingBatch::Regression { input, target } => {
                self.regression_loss(input, target)
            }
        }
    }

    /// Proper-scoring regression loss (S18), dispatched on the configured
    /// [`RegressionLoss`]. `input` is the distributional head's raw z-space output
    /// (`(batch, k)`); `target` is the **z-scored** `(batch,)` outcome.
    ///
    /// The three Gaussian arms read `(mean, raw_std)` from a two-wide head and
    /// score the predictive `Normal(mean, σ)`, where `σ = floor + softplus(raw_std)`
    /// — the learnable floor is the head's own trainable bias under `softplus`,
    /// with [`STD_FLOOR`] as the hard numerical guard against exact-zero variance
    /// (the overconfidence collapse). The pinball arm reads one quantile per
    /// head column and scores each against its level.
    ///
    /// Both `input` and `target` are in standardized (z) space — `head_forward`
    /// returns the raw z-output and `embed_chunk` z-scores the target — so this
    /// loss scores O(1) residuals regardless of the target's raw scale. The four
    /// objective fns are unchanged: they are pure functions of `(head, target)`,
    /// already proven in z-space by the in-context predictor. De-standardisation
    /// to raw units lives entirely on the serve path, never here.
    fn regression_loss(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        match self.config.regression_loss.unwrap_or_default() {
            super::RegressionLoss::GaussianNll => gaussian_nll_loss(input, target, 0.0),
            super::RegressionLoss::BetaNll { beta } => gaussian_nll_loss(input, target, beta),
            super::RegressionLoss::Crps => crps_gaussian_loss(input, target),
            super::RegressionLoss::Pinball => {
                pinball_loss(input, target, &self.config.quantile_levels)
            }
        }
    }

    /// The graded-pair embedding objective for a `Contrastive` batch, dispatched
    /// on the configured [`EmbeddingLoss`]. Thin wrapper over the free
    /// [`dispatch_contrastive_loss`] — the CoSENT default is provided by
    /// [`Self::cosent_loss`], the only graded objective that reads `self`.
    fn contrastive_loss(&self, emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
        dispatch_contrastive_loss(
            self.config.embedding_loss,
            emb_a,
            emb_b,
            scores,
            &|a, b, s| self.cosent_loss(a, b, s),
        )
    }

    /// The MNRL similarity scale (`temperature`). `20.0` is the standard
    /// default; a `MultipleNegativesRanking { temperature }` config overrides it.
    fn mnrl_scale(&self) -> f64 {
        match self.config.embedding_loss {
            Some(super::EmbeddingLoss::MultipleNegativesRanking { temperature }) => temperature,
            _ => PAIRWISE_SCALE,
        }
    }

    /// Evaluate `objective` at each configured Matryoshka prefix dimension and
    /// sum the losses, or evaluate it once on the full embeddings when no dims
    /// are set. Thin wrapper over the free [`matryoshka_sum`].
    fn matryoshka_wrap(
        &self,
        embeddings: &[&Tensor],
        objective: &dyn Fn(Vec<Tensor>) -> Result<Tensor>,
    ) -> Result<Tensor> {
        matryoshka_sum(&self.config.matryoshka_dims, embeddings, objective)
    }

    /// CoSENT loss: cross-entropy on cosine similarity ordering.
    fn cosent_loss(&self, emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
        let cos_sim = cosine_similarity(emb_a, emb_b)?;
        // Scale similarities by temperature (20.0 is typical for CoSENT)
        let temperature = 20.0;
        let scaled = (&cos_sim * temperature)
            .map_err(|e| JammiError::FineTune(format!("CoSENT scale: {e}")))?;

        // MSE between scaled cosine similarity and target scores
        let diff = (&scaled / temperature - scores)
            .map_err(|e| JammiError::FineTune(format!("CoSENT diff: {e}")))?;
        let loss = diff
            .sqr()
            .map_err(|e| JammiError::FineTune(format!("CoSENT sqr: {e}")))?
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("CoSENT mean: {e}")))?;

        Ok(loss)
    }

    /// Apply the classification head to projected embeddings.
    ///
    /// Only the `ProjectionHead` target supports classification training,
    /// and only when the head was built with both a projection and a
    /// classifier layer (i.e. `head.layers.len() > 1`).
    fn classify(&self, embeddings: &Tensor) -> Result<Tensor> {
        match &self.target {
            TrainingTarget::ProjectionHead { head } if head.layers.len() > 1 => head.layers[1]
                .1
                .forward(embeddings)
                .map_err(|e| JammiError::FineTune(format!("LoRA classifier: {e}"))),
            TrainingTarget::ProjectionHead { .. } => Err(JammiError::FineTune(
                "No classification head in projection target".into(),
            )),
            TrainingTarget::EncoderAdapters(_) => Err(JammiError::FineTune(
                "Classification with encoder adapters is not supported".into(),
            )),
        }
    }

    /// Cross-entropy loss for classification.
    fn cross_entropy_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        candle_nn::loss::cross_entropy(logits, labels)
            .map_err(|e| JammiError::FineTune(format!("Cross-entropy loss: {e}")))
    }

    /// Token-level cross-entropy loss for NER, ignoring positions with label -100.
    fn ner_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, num_labels) = logits
            .dims3()
            .map_err(|e| JammiError::FineTune(format!("NER logits dims: {e}")))?;

        // Flatten to (batch*seq_len, num_labels) and (batch*seq_len,)
        let flat_logits = logits
            .reshape((batch * seq_len, num_labels))
            .map_err(|e| JammiError::FineTune(format!("NER flatten logits: {e}")))?;
        let flat_labels = labels
            .reshape(batch * seq_len)
            .map_err(|e| JammiError::FineTune(format!("NER flatten labels: {e}")))?;

        // Replace -100 with 0 for safe indexing (masked out below)
        let safe_labels = flat_labels
            .clamp(0i64, (num_labels - 1) as i64)
            .map_err(|e| JammiError::FineTune(format!("NER clamp labels: {e}")))?
            .to_dtype(candle_core::DType::U32)
            .map_err(|e| JammiError::FineTune(format!("NER labels u32: {e}")))?;

        // Cross-entropy on all positions (candle returns mean over elements).
        // Positions with original label -100 are clamped to 0 and contribute noise,
        // but this is a reasonable approximation until masked CE is available.
        candle_nn::loss::cross_entropy(&flat_logits, &safe_labels)
            .map_err(|e| JammiError::FineTune(format!("NER cross-entropy: {e}")))
    }

    /// Triplet loss: `max(0, cos(anchor, negative) - cos(anchor, positive) + margin)`.
    fn triplet_loss(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negative: &Tensor,
    ) -> Result<Tensor> {
        let margin = match self.config.embedding_loss {
            Some(super::EmbeddingLoss::Triplet { margin }) => margin,
            _ => 0.3,
        };

        let pos_sim = cosine_similarity(anchor, positive)?;
        let neg_sim = cosine_similarity(anchor, negative)?;

        // loss = max(0, neg_sim - pos_sim + margin)
        let diff = ((&neg_sim - &pos_sim)
            .map_err(|e| JammiError::FineTune(format!("Triplet diff: {e}")))?
            + margin)
            .map_err(|e| JammiError::FineTune(format!("Triplet margin: {e}")))?;

        let zero = Tensor::zeros_like(&diff)
            .map_err(|e| JammiError::FineTune(format!("Triplet zeros: {e}")))?;
        let loss = diff
            .maximum(&zero)
            .map_err(|e| JammiError::FineTune(format!("Triplet max: {e}")))?
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("Triplet mean: {e}")))?;

        Ok(loss)
    }

    /// Run forward pass over validation set without gradient updates.
    fn evaluate(&self, val_loader: &TrainingDataLoader) -> Result<f64> {
        if val_loader.is_empty() {
            // Unreachable: the ValLoss path is refused at the split, and the
            // TrainLoss path never calls this. Kept as an error rather than a
            // `0.0` because a fabricated measurement is indistinguishable from a
            // real one downstream — it selected checkpoints, stopped runs early,
            // and was reported as the run's final loss.
            return Err(JammiError::FineTune(
                "internal: evaluate() called with an empty validation loader".into(),
            ));
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        let accumulate = |batch, total: &mut f64, count: &mut usize| -> Result<()> {
            let loss = self.compute_loss(&batch)?;
            let loss = if loss.dtype() == DType::F32 {
                loss
            } else {
                loss.to_dtype(DType::F32)
                    .map_err(|e| JammiError::FineTune(format!("Val loss dtype cast: {e}")))?
            };
            *total += loss
                .to_scalar::<f32>()
                .map_err(|e| JammiError::FineTune(format!("Val loss scalar: {e}")))?
                as f64;
            *count += 1;
            Ok(())
        };

        if val_loader.is_precomputed() {
            for batch in val_loader.batches(self.config.batch_size)? {
                accumulate(batch?, &mut total_loss, &mut count)?;
            }
        } else {
            let text_chunks = val_loader.text_chunks(self.config.batch_size);
            for chunk in &text_chunks {
                let batch = self.encode_chunk(chunk)?;
                accumulate(batch, &mut total_loss, &mut count)?;
            }
        }

        Ok(if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        })
    }

    /// Save a numbered intra-epoch checkpoint. Weights only — the metadata
    /// JSON is written once when the final adapter lands.
    fn save_checkpoint(&self, dir: &Path, step: usize) -> Result<()> {
        let path = dir.join(format!("checkpoint_{step}.safetensors"));
        self.save_checkpoint_weights(&path)
    }

    /// Save a named checkpoint (e.g. "best"). Weights only.
    fn save_checkpoint_tagged(&self, dir: &Path, tag: &str) -> Result<()> {
        let path = dir.join(format!("checkpoint_{tag}.safetensors"));
        self.save_checkpoint_weights(&path)
    }

    fn save_checkpoint_weights(&self, path: &Path) -> Result<()> {
        let weights = self.target.named_trainable_weights()?;
        candle_core::safetensors::save(&weights, path)
            .map_err(|e| JammiError::FineTune(format!("Save checkpoint: {e}")))
    }

    /// Load a checkpoint, restoring LoRA weights in place.
    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }
        let weights = candle_core::safetensors::load(path, &self.device)
            .map_err(|e| JammiError::FineTune(format!("Load checkpoint: {e}")))?;
        self.target.load_weights(&weights)
    }

    /// The parameter NAME for each entry of `optimizer.state()`'s moment vector,
    /// in that exact order — the correlation that lets the resume bundle key
    /// optimizer moments by name rather than by the unstable `all_vars()` order.
    ///
    /// `AdamW::new` keeps the float subset of `vars` in their given order, and
    /// `state()` reports moments in that order, so this applies the same float
    /// filter and maps each surviving var to its name via a tensor-identity →
    /// name index built from `varmap.data()`. A trainable var absent from the
    /// `VarMap` (which cannot happen — every trainable LoRA tensor is a registered
    /// `Var`) is a hard error rather than a silently dropped moment.
    fn optimizer_param_names(&self, vars: &[Var]) -> Result<Vec<String>> {
        let data = self.varmap.data().lock().map_err(|_| {
            JammiError::FineTune("optimizer param names: VarMap mutex poisoned".into())
        })?;
        let id_to_name: HashMap<candle_core::TensorId, &String> =
            data.iter().map(|(name, var)| (var.id(), name)).collect();
        vars.iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                id_to_name
                    .get(&var.id())
                    .map(|name| (*name).clone())
                    .ok_or_else(|| {
                        JammiError::FineTune(
                            "optimizer param names: a trainable var is not registered in the \
                             VarMap — cannot key its optimizer moment by name"
                                .into(),
                        )
                    })
            })
            .collect()
    }

    /// Capture the moments `optimizer.state()` reports, keyed by parameter name
    /// (the order-independent correlation `optim_param_names` provides). This is
    /// the single capture routine the durable epoch-boundary save and the resume
    /// test both use, so a reference snapshot and a crash-persist are taken at the
    /// SAME boundary by the SAME code (R4).
    fn capture_moments_by_name(
        optimizer: &AdamW,
        optim_param_names: &[String],
    ) -> Result<(NamedMoments, usize)> {
        let (moments, step_t) = optimizer
            .state()
            .map_err(|e| JammiError::FineTune(format!("capture optimizer state: {e}")))?;
        if moments.len() != optim_param_names.len() {
            return Err(JammiError::FineTune(format!(
                "optimizer reported {} moments for {} named parameters",
                moments.len(),
                optim_param_names.len()
            )));
        }
        let by_name = optim_param_names
            .iter()
            .cloned()
            .zip(moments)
            .collect::<HashMap<_, _>>();
        Ok((by_name, step_t))
    }

    /// Assemble the full resume bundle at an epoch boundary: adapter weights, the
    /// name-keyed optimizer moments, the scaler's `(μ, σ)`, the dropout-stream
    /// positions, and the run counters. The single routine both the durable save
    /// and the test's reference snapshot drive.
    fn capture_resume_bundle(
        &self,
        scratch_dir: &Path,
        last_completed_epoch: usize,
        global_step: usize,
        optimizer: &AdamW,
        optim_param_names: &[String],
    ) -> Result<Vec<(String, bytes::Bytes)>> {
        let weights = self.target.named_trainable_weights()?;
        let (moments, step_t) = Self::capture_moments_by_name(optimizer, optim_param_names)?;
        let state = ResumeState {
            schema_version: RESUME_STATE_SCHEMA_VERSION,
            last_completed_epoch,
            global_step,
            step_t,
            seed: self.config.seed,
            scaler: self.target_scaler.map(|s| (s.mean(), s.std())),
            dropout_positions: self.target.dropout_positions()?,
        };
        capture_bundle(scratch_dir, &weights, &moments, &state)
    }

    /// Write the durable resume checkpoint to `{job_id}/_resume/` via the artifact
    /// store, overwriting the prior epoch. A `None` store is a no-op (a
    /// trainer-internal run with no durable checkpointing). The caller has already
    /// confirmed the lease is held (`!cancel`).
    fn save_resume_checkpoint(
        &self,
        checkpoint_dir: &Path,
        epoch: usize,
        global_step: usize,
        optimizer: &AdamW,
        optim_param_names: &[String],
    ) -> Result<()> {
        let Some(store) = self.artifact_store.as_ref() else {
            return Ok(());
        };
        let scratch = checkpoint_dir.join("_resume_scratch");
        let bundle =
            self.capture_resume_bundle(&scratch, epoch, global_step, optimizer, optim_param_names)?;
        tokio::runtime::Handle::current()
            .block_on(store.put_resume_checkpoint(&self.job_id, &bundle))?;
        Ok(())
    }

    /// Restore weights, optimizer moments (BY NAME), the scaler, and the dropout
    /// positions from a discovered resume bundle, and return the epoch the resumed
    /// run starts at (`last_completed + 1`) and its step counter.
    ///
    /// The optimizer moments are reordered from the persisted name→moment map into
    /// the optimizer's positional order via `optim_param_names` (this process's
    /// `all_vars()` order), so `AdamW::load_state` restores each parameter its OWN
    /// moments regardless of how the two processes' HashMap orders differ (R1). The
    /// scaler is loaded authoritatively, never recomputed (R7).
    fn restore_from_checkpoint(
        &mut self,
        restored: RestoredCheckpoint,
        optimizer: &mut AdamW,
        optim_param_names: &[String],
    ) -> Result<(usize, usize)> {
        let RestoredCheckpoint {
            weights,
            moments,
            state,
        } = restored;

        // Restore weights by writing into the registered `Var`s in place (by
        // name), NOT by replacing the target's tensor fields. The optimizer holds
        // those same `Var`s, and the LoRA layer's `lora_a`/`lora_b` fields share
        // their storage; an in-place `Var::set` updates all three together, so the
        // forward, the gradient, and the optimizer step stay bound to one tensor
        // identity. Replacing the field tensor instead (a fresh `clone`) would
        // sever that binding — the optimizer would step the now-orphaned `Var`
        // while the forward read the stale field, freezing the restored weights.
        {
            let data = self.varmap.data().lock().map_err(|_| {
                JammiError::FineTune("resume: VarMap mutex poisoned restoring weights".into())
            })?;
            for (name, tensor) in &weights {
                let var = data.get(name).ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "resume: restored weight '{name}' is not a registered Var — the head \
                         shape changed between crash and resume"
                    ))
                })?;
                var.set(tensor)
                    .map_err(|e| JammiError::FineTune(format!("resume: set '{name}': {e}")))?;
            }
        }

        // Moments reordered by name into the optimizer's positional order.
        let ordered = optim_param_names
            .iter()
            .map(|name| {
                moments.get(name).cloned().ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "resume: optimizer moment for parameter '{name}' missing from the \
                         checkpoint — cannot restore its trajectory by name"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        optimizer
            .load_state(&ordered, state.step_t)
            .map_err(|e| JammiError::FineTune(format!("resume: load optimizer state: {e}")))?;

        // The persisted scaler is authoritative — overwrite the recomputed one so
        // a source mutated between crash and resume cannot perturb the
        // de-standardisation (R7). A regression run always persists `(μ, σ)`; a
        // non-regression run persists `None` and leaves the scaler unset.
        self.target_scaler = state
            .scaler
            .map(|(mean, std)| TargetScaler::from_mean_std(mean, std));

        // Replay each dropout stream to its epoch-boundary position so the next
        // forwards draw the same masks the uninterrupted run drew (R3).
        self.target
            .restore_dropout_positions(&state.dropout_positions)?;

        Ok((state.last_completed_epoch + 1, state.global_step))
    }
}

/// Temperature scaling the per-pair similarity before the pairwise log-sum-exp
/// ordering. `20.0` is the standard CoSENT/AnglE convention; AnglE reuses it so
/// the two objectives are directly comparable on the same data.
const PAIRWISE_SCALE: f64 = 20.0;

/// CoSENT-style pairwise ordering loss over a per-pair similarity vector.
///
/// Given a per-pair similarity `sim[k]` (already scaled) and the graded target
/// `scores[k]`, penalises every ordered pair `(i, j)` whose targets say `i`
/// should rank below `j` (`scores[i] < scores[j]`) but whose similarities say
/// otherwise. The loss is `log(1 + Σ exp(sim[i] − sim[j]))` over those pairs,
/// computed as a single `log_sum_exp` with a prepended zero (the `1`) and an
/// additive `−∞` mask on the invalid pairs.
///
/// Shared by CoSENT (similarity = cosine) and AnglE (similarity = angle
/// magnitude); only the per-pair similarity differs.
fn pairwise_ordering_loss(sim: &Tensor, scores: &Tensor) -> Result<Tensor> {
    let n = sim
        .dim(0)
        .map_err(|e| JammiError::FineTune(format!("pairwise dim: {e}")))?;

    // Pairwise similarity differences `sim[i] − sim[j]` as an (n, n) matrix.
    let sim_i = sim
        .reshape((n, 1))
        .map_err(|e| JammiError::FineTune(format!("pairwise sim_i: {e}")))?
        .broadcast_as((n, n))
        .map_err(|e| JammiError::FineTune(format!("pairwise sim_i bcast: {e}")))?;
    let sim_j = sim
        .reshape((1, n))
        .map_err(|e| JammiError::FineTune(format!("pairwise sim_j: {e}")))?
        .broadcast_as((n, n))
        .map_err(|e| JammiError::FineTune(format!("pairwise sim_j bcast: {e}")))?;
    let diff =
        (&sim_i - &sim_j).map_err(|e| JammiError::FineTune(format!("pairwise diff: {e}")))?;

    // Valid pairs are those the targets order as `scores[i] < scores[j]`. Build
    // an additive mask: `0` on valid pairs, a large negative elsewhere, so the
    // invalid terms vanish under `exp` inside the log-sum-exp.
    let score_i = scores
        .reshape((n, 1))
        .map_err(|e| JammiError::FineTune(format!("pairwise score_i: {e}")))?
        .broadcast_as((n, n))
        .map_err(|e| JammiError::FineTune(format!("pairwise score_i bcast: {e}")))?;
    let score_j = scores
        .reshape((1, n))
        .map_err(|e| JammiError::FineTune(format!("pairwise score_j: {e}")))?
        .broadcast_as((n, n))
        .map_err(|e| JammiError::FineTune(format!("pairwise score_j bcast: {e}")))?;
    let valid = score_i
        .lt(&score_j)
        .map_err(|e| JammiError::FineTune(format!("pairwise valid: {e}")))?
        .to_dtype(diff.dtype())
        .map_err(|e| JammiError::FineTune(format!("pairwise valid dtype: {e}")))?;
    // `(valid − 1) · 1e12` is `0` where valid, `−1e12` where not.
    let mask = ((&valid - 1.0)
        .map_err(|e| JammiError::FineTune(format!("pairwise mask sub: {e}")))?
        * 1e12)
        .map_err(|e| JammiError::FineTune(format!("pairwise mask scale: {e}")))?;
    let masked = (&diff + &mask)
        .map_err(|e| JammiError::FineTune(format!("pairwise masked: {e}")))?
        .flatten_all()
        .map_err(|e| JammiError::FineTune(format!("pairwise flatten: {e}")))?;

    // Prepend a zero — the `1` inside `log(1 + Σ exp(·))` — then log-sum-exp the
    // whole vector. With no valid pair, every entry is `≈ −∞` except the zero,
    // so the loss is `log(1) = 0`.
    let zero = Tensor::zeros(1, masked.dtype(), masked.device())
        .map_err(|e| JammiError::FineTune(format!("pairwise zero: {e}")))?;
    let stacked = Tensor::cat(&[&zero, &masked], 0)
        .map_err(|e| JammiError::FineTune(format!("pairwise cat: {e}")))?;
    stacked
        .log_sum_exp(0)
        .map_err(|e| JammiError::FineTune(format!("pairwise logsumexp: {e}")))
}

/// AnglE loss: optimise the angle difference between paired embeddings in
/// complex space, applied through the CoSENT pairwise ordering.
///
/// Each embedding is split into real/imaginary halves. The complex quotient
/// `z_a / z_b` has an imaginary component proportional to `sin(Δθ)` of the
/// angle between the two complex vectors; its magnitude is the angle signal
/// AnglE optimises. Crucially this signal does **not** saturate as the cosine
/// similarity approaches ±1 — where a cosine objective's gradient vanishes,
/// the angle gradient stays informative, which is the whole point of AnglE.
///
/// The per-pair angle magnitude is scaled by [`PAIRWISE_SCALE`] and fed to the
/// same pairwise log-sum-exp ordering as CoSENT.
fn angle_loss(emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
    let (a_re, a_im) = split_complex(emb_a)?;
    let (b_re, b_im) = split_complex(emb_b)?;

    // Treat the two halves as complex vectors z_a, z_b and form the per-pair
    // quotient z_a / z_b summed over the embedding dimension. With
    // numerator = z_a · conj(z_b) and denominator = |z_b|²:
    //   Re = Σ(a_re·b_re + a_im·b_im),  Im = Σ(a_im·b_re − a_re·b_im).
    let num_re = ((&a_re * &b_re).map_err(|e| JammiError::FineTune(format!("angle re1: {e}")))?
        + (&a_im * &b_im).map_err(|e| JammiError::FineTune(format!("angle re2: {e}")))?)
    .map_err(|e| JammiError::FineTune(format!("angle re: {e}")))?
    .sum(1)
    .map_err(|e| JammiError::FineTune(format!("angle re sum: {e}")))?;
    let num_im = ((&a_im * &b_re).map_err(|e| JammiError::FineTune(format!("angle im1: {e}")))?
        - (&a_re * &b_im).map_err(|e| JammiError::FineTune(format!("angle im2: {e}")))?)
    .map_err(|e| JammiError::FineTune(format!("angle im: {e}")))?
    .sum(1)
    .map_err(|e| JammiError::FineTune(format!("angle im sum: {e}")))?;

    // Normalise the quotient to unit magnitude: with |z_a/z_b| = 1, its
    // imaginary part is exactly sin(Δθ) of the angle between the vectors. That
    // |sin(Δθ)| is the angle signal — and unlike cosine it does not flatten as
    // the vectors align (the cosine objective's vanishing-gradient zone), since
    // d|sin(Δθ)|/dθ = |cos(Δθ)| stays away from zero there. `num_re` is the
    // partner component that defines the magnitude, so it is part of the graph.
    let mag = ((&num_re
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("angle re sqr: {e}")))?
        + &num_im
            .sqr()
            .map_err(|e| JammiError::FineTune(format!("angle im sqr: {e}")))?)
        .map_err(|e| JammiError::FineTune(format!("angle mag add: {e}")))?
        .sqrt()
        .map_err(|e| JammiError::FineTune(format!("angle mag sqrt: {e}")))?)
    .clamp(1e-8, f64::MAX)
    .map_err(|e| JammiError::FineTune(format!("angle mag clamp: {e}")))?;
    let angle = (num_im
        .abs()
        .map_err(|e| JammiError::FineTune(format!("angle abs: {e}")))?
        / &mag)
        .map_err(|e| JammiError::FineTune(format!("angle div: {e}")))?;

    let scaled =
        (&angle * PAIRWISE_SCALE).map_err(|e| JammiError::FineTune(format!("angle scale: {e}")))?;
    pairwise_ordering_loss(&scaled, scores)
}

/// Split a `[batch, hidden]` embedding into real and imaginary halves along the
/// hidden dimension, as AnglE's complex-space representation requires. The
/// hidden dimension must be even.
fn split_complex(emb: &Tensor) -> Result<(Tensor, Tensor)> {
    let hidden = emb
        .dim(1)
        .map_err(|e| JammiError::FineTune(format!("complex dim: {e}")))?;
    if hidden % 2 != 0 {
        return Err(JammiError::FineTune(format!(
            "AnglE requires an even embedding dimension to split into real/imaginary halves, got {hidden}"
        )));
    }
    let half = hidden / 2;
    let re = emb
        .narrow(1, 0, half)
        .map_err(|e| JammiError::FineTune(format!("complex re: {e}")))?;
    let im = emb
        .narrow(1, half, half)
        .map_err(|e| JammiError::FineTune(format!("complex im: {e}")))?;
    Ok((re, im))
}

/// cosine-MSE loss: regress the scaled cosine similarity of each pair onto its
/// graded target score with mean-squared error.
///
/// `MSE(scale · cos(a, b), score)`. The simplest objective for continuous
/// similarity labels — distinct from CoSENT (pairwise ordering) and MNRL
/// (ranking). Reuses [`PAIRWISE_SCALE`] so the predicted value lives on the
/// same scale as the graded targets the other objectives consume.
fn cosine_mse_loss(emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
    let cos = cosine_similarity(emb_a, emb_b)?;
    let pred = (&cos * PAIRWISE_SCALE)
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE scale: {e}")))?;
    let target = (scores * PAIRWISE_SCALE)
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE target scale: {e}")))?;
    let diff =
        (&pred - &target).map_err(|e| JammiError::FineTune(format!("cosine-MSE diff: {e}")))?;
    diff.sqr()
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE sqr: {e}")))?
        .mean_all()
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE mean: {e}")))
}

/// L2-normalise every row of a `[n, d]` tensor to unit length, sharing the norm
/// computation with [`cosine_similarity`] (sum of squares along dim 1, sqrt,
/// clamped away from zero). The cosine-similarity *matrix* MNRL needs is then a
/// plain matmul of two row-normalised batches — no new distance primitive.
fn l2_normalize_rows(x: &Tensor) -> Result<Tensor> {
    let norm = x
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("l2norm sqr: {e}")))?
        .sum_keepdim(1)
        .map_err(|e| JammiError::FineTune(format!("l2norm sum: {e}")))?
        .sqrt()
        .map_err(|e| JammiError::FineTune(format!("l2norm sqrt: {e}")))?
        .clamp(1e-8, f64::MAX)
        .map_err(|e| JammiError::FineTune(format!("l2norm clamp: {e}")))?;
    x.broadcast_div(&norm)
        .map_err(|e| JammiError::FineTune(format!("l2norm div: {e}")))
}

/// Multiple-Negatives-Ranking loss (InfoNCE / NT-Xent) over a batch of
/// `(anchor, positive)` rows.
///
/// Builds the scaled cosine-similarity matrix `S = normalize(A) ·
/// normalize(P)ᵀ · scale`, an `(n, n)` matrix whose `[i, j]` entry is the
/// scaled similarity of anchor `i` to positive `j`. The correct positive for
/// each anchor sits on the diagonal, so the target labels are `0..n` and the
/// loss is cross-entropy of each row against its diagonal index — every
/// off-diagonal positive is an in-batch negative.
///
/// `symmetric` adds the column-direction cross-entropy (each positive against
/// its anchor), the sentence-transformers default: it trains the embedding to
/// retrieve in both directions. Pass `false` for an asymmetric query→document
/// objective where only the anchor→positive direction is meaningful.
///
/// `hard_negatives`, when present, is an `(n, d)` batch of one explicit hard
/// negative per anchor; its similarities are appended as extra columns of `S`
/// (the DPR recipe), sharpening the contrast without changing the diagonal
/// targets. The column direction only ranks the `n` positives, so the hard
/// negatives participate in the row direction alone.
fn mnrl_loss(
    anchor: &Tensor,
    positive: &Tensor,
    hard_negatives: Option<&Tensor>,
    scale: f64,
    symmetric: bool,
) -> Result<Tensor> {
    let n = anchor
        .dim(0)
        .map_err(|e| JammiError::FineTune(format!("mnrl dim: {e}")))?;

    let a_norm = l2_normalize_rows(anchor)?;
    let p_norm = l2_normalize_rows(positive)?;
    let p_t = p_norm
        .t()
        .map_err(|e| JammiError::FineTune(format!("mnrl transpose: {e}")))?;
    // (n, n) anchor↔positive similarity, scaled. `p_t` is a transpose view, so
    // this routes through the contiguity-safe primitive rather than a raw
    // `matmul` on a transposed RHS.
    let sim = (jammi_encoders::contiguous_matmul(&a_norm, &p_t)
        .map_err(|e| JammiError::FineTune(format!("mnrl matmul: {e}")))?
        * scale)
        .map_err(|e| JammiError::FineTune(format!("mnrl scale: {e}")))?;

    // The positive for anchor i is column i: labels are the diagonal indices.
    let labels = Tensor::arange(0u32, n as u32, anchor.device())
        .map_err(|e| JammiError::FineTune(format!("mnrl labels: {e}")))?;

    // Append explicit hard negatives as extra similarity columns. They extend
    // the row-direction candidate set (more negatives per anchor) but not the
    // positives, so the diagonal labels are unchanged.
    let row_logits = match hard_negatives {
        None => sim.clone(),
        Some(neg) => {
            let neg_norm = l2_normalize_rows(neg)?;
            let neg_t = neg_norm
                .t()
                .map_err(|e| JammiError::FineTune(format!("mnrl neg transpose: {e}")))?;
            let neg_sim = (jammi_encoders::contiguous_matmul(&a_norm, &neg_t)
                .map_err(|e| JammiError::FineTune(format!("mnrl neg matmul: {e}")))?
                * scale)
                .map_err(|e| JammiError::FineTune(format!("mnrl neg scale: {e}")))?;
            Tensor::cat(&[&sim, &neg_sim], 1)
                .map_err(|e| JammiError::FineTune(format!("mnrl neg cat: {e}")))?
        }
    };

    let row_loss = candle_nn::loss::cross_entropy(&row_logits, &labels)
        .map_err(|e| JammiError::FineTune(format!("mnrl row cross-entropy: {e}")))?;

    if !symmetric {
        return Ok(row_loss);
    }

    // Column direction: each positive against the anchors. Transpose the
    // anchor↔positive block only (hard negatives have no anchor to rank
    // against, so they stay out of this direction).
    let col_logits = sim
        .t()
        .map_err(|e| JammiError::FineTune(format!("mnrl col transpose: {e}")))?;
    let col_loss = candle_nn::loss::cross_entropy(&col_logits, &labels)
        .map_err(|e| JammiError::FineTune(format!("mnrl col cross-entropy: {e}")))?;

    ((&row_loss + &col_loss).map_err(|e| JammiError::FineTune(format!("mnrl sum: {e}")))? * 0.5)
        .map_err(|e| JammiError::FineTune(format!("mnrl mean: {e}")))
}

/// Dispatch a graded-pair `(a, b, score)` batch onto the configured
/// [`EmbeddingLoss`]. CoSENT (the default), AnglE, and cosine-MSE consume
/// graded pairs. The in-batch-negative and triplet objectives are not
/// graded-pair shaped, so naming one here is a typed error rather than a silent
/// fall-through to a different loss. `cosent` supplies the CoSENT path (the
/// only graded objective that reads trainer state).
fn dispatch_contrastive_loss(
    loss: Option<super::EmbeddingLoss>,
    emb_a: &Tensor,
    emb_b: &Tensor,
    scores: &Tensor,
    cosent: &dyn Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    match loss {
        Some(super::EmbeddingLoss::AnglE) => angle_loss(emb_a, emb_b, scores),
        Some(super::EmbeddingLoss::CosineMse) => cosine_mse_loss(emb_a, emb_b, scores),
        Some(super::EmbeddingLoss::MultipleNegativesRanking { .. }) => Err(JammiError::FineTune(
            "MultipleNegativesRanking is an in-batch-negative objective over (anchor, positive) \
             rows; it cannot score a graded (text_a, text_b, score) batch. Supply (anchor, \
             positive) pairs, or choose CoSENT/AnglE/cosine-MSE."
                .into(),
        )),
        Some(super::EmbeddingLoss::Triplet { .. }) => Err(JammiError::FineTune(
            "Triplet loss needs (anchor, positive, negative) rows; it cannot score a graded \
             (text_a, text_b, score) batch. Choose CoSENT/AnglE/cosine-MSE for graded pairs."
                .into(),
        )),
        Some(super::EmbeddingLoss::CoSent) | None => cosent(emb_a, emb_b, scores),
    }
}

/// Evaluate `objective` at each Matryoshka prefix dimension in `dims` and sum
/// the losses, or evaluate it once on the full embeddings when `dims` is empty.
///
/// Every input tensor is `narrow`ed to the same prefix width before each call,
/// so the objective sees a consistent reduced embedding. Summing over a nested
/// set of prefixes is what *orders* the coordinates by importance — the leading
/// dims must satisfy the objective at every truncation, so they carry the most
/// signal, and a serve-time truncation to any listed dim stays valid. A dim
/// wider than the embedding is a typed error, not a silent clamp.
fn matryoshka_sum(
    dims: &[usize],
    embeddings: &[&Tensor],
    objective: &dyn Fn(Vec<Tensor>) -> Result<Tensor>,
) -> Result<Tensor> {
    if dims.is_empty() {
        return objective(embeddings.iter().map(|t| (*t).clone()).collect());
    }

    let full_dim = embeddings
        .first()
        .ok_or_else(|| JammiError::FineTune("matryoshka: no embeddings".into()))?
        .dim(1)
        .map_err(|e| JammiError::FineTune(format!("matryoshka dim: {e}")))?;

    let mut total: Option<Tensor> = None;
    for &dim in dims {
        if dim > full_dim {
            return Err(JammiError::FineTune(format!(
                "matryoshka_dims entry {dim} exceeds the embedding width {full_dim}"
            )));
        }
        let truncated: Vec<Tensor> = embeddings
            .iter()
            .map(|t| {
                t.narrow(1, 0, dim)
                    .map_err(|e| JammiError::FineTune(format!("matryoshka narrow: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let loss = objective(truncated)?;
        total = Some(match total {
            None => loss,
            Some(acc) => {
                (&acc + &loss).map_err(|e| JammiError::FineTune(format!("matryoshka sum: {e}")))?
            }
        });
    }
    total.ok_or_else(|| JammiError::FineTune("matryoshka_dims was unexpectedly empty".into()))
}

/// Test-only handle to [`mnrl_loss`] for the GradCache gradient-equivalence
/// test, which lives in the sibling `gradcache` module and needs the exact
/// objective the trainer runs.
#[cfg(test)]
pub(crate) fn mnrl_loss_for_test(
    anchor: &Tensor,
    positive: &Tensor,
    hard_negatives: Option<&Tensor>,
    scale: f64,
    symmetric: bool,
) -> Result<Tensor> {
    mnrl_loss(anchor, positive, hard_negatives, scale, symmetric)
}

/// Compute element-wise cosine similarity between two batches of vectors.
fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dot = (a * b)
        .map_err(|e| JammiError::FineTune(format!("cos_sim mul: {e}")))?
        .sum(1)
        .map_err(|e| JammiError::FineTune(format!("cos_sim sum: {e}")))?;

    let norm_a = a
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_a sqr: {e}")))?
        .sum(1)
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_a sum: {e}")))?
        .sqrt()
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_a sqrt: {e}")))?
        .clamp(1e-8, f64::MAX)
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_a clamp: {e}")))?;

    let norm_b = b
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_b sqr: {e}")))?
        .sum(1)
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_b sum: {e}")))?
        .sqrt()
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_b sqrt: {e}")))?
        .clamp(1e-8, f64::MAX)
        .map_err(|e| JammiError::FineTune(format!("cos_sim norm_b clamp: {e}")))?;

    let denom =
        (&norm_a * &norm_b).map_err(|e| JammiError::FineTune(format!("cos_sim denom: {e}")))?;

    (&dot / &denom).map_err(|e| JammiError::FineTune(format!("cos_sim div: {e}")))
}

/// B2 premise pin: the "no bounded HtoD-cache API exists" reasoning behind
/// removing the run-held cache guard (see the `NOT held: candle's
/// CUDA_GRAPH_HTOD_CACHE` doc above the epoch loop in [`TrainingLoop::run`])
/// is read straight off candle-core 0.11.0's `cuda_backend/device.rs` — a
/// version-specific fact, not a permanent one. `cuda_backend` is private to
/// candle-core, so this crate cannot probe its `CudaGraphHtodCacheGuard` type
/// for a `clear`/`capacity` method directly (and the `cuda` feature cannot
/// build locally anyway — no nvcc); pinning the dependency's resolved version
/// in the workspace lockfile is the compile-time-checkable proxy available
/// everywhere. This fails the moment `candle-core` moves off `0.11.0`,
/// forcing a human to re-read the new `cuda_backend/device.rs` for an
/// eviction/bounded-capacity API before a run-held HtoD-cache guard is ever
/// reinstated.
#[cfg(test)]
mod cuda_htod_cache_premise_pin {
    #[test]
    fn candle_core_is_still_the_audited_0_11_0() {
        let lock = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"));
        let idx = lock
            .find("name = \"candle-core\"")
            .expect("candle-core must appear in the workspace Cargo.lock");
        let version_line = lock[idx..]
            .lines()
            .nth(1)
            .expect("a version line must follow the candle-core package name");
        assert_eq!(
            version_line.trim(),
            "version = \"0.11.0\"",
            "candle-core moved off the audited 0.11.0 — before reinstating a \
             run-held HtoD-cache guard, re-read the new cuda_backend/device.rs \
             for an eviction or bounded-capacity API (see TrainingLoop::run's \
             doc on why this guard was removed)."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::regression_loss::{
        gaussian_params, softplus_std_for_test, TargetScaler, STD_FLOOR,
    };
    use super::*;
    use candle_core::Var;

    /// CoSENT objective expressed through the shared pieces: scaled cosine
    /// similarity fed to the pairwise ordering. Used only to contrast its
    /// gradient against AnglE's near cosine saturation.
    fn cosent_reference(emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
        let cos = cosine_similarity(emb_a, emb_b).unwrap();
        let scaled = (&cos * PAIRWISE_SCALE).unwrap();
        pairwise_ordering_loss(&scaled, scores)
    }

    /// L2 norm of a gradient tensor as an f64 scalar.
    fn grad_norm(g: &Tensor) -> f64 {
        let sq: f32 = g.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        (sq as f64).sqrt()
    }

    /// Near cosine saturation — pairs whose embeddings are almost aligned, so
    /// every cosine similarity sits at ≈1 — CoSENT's gradient w.r.t. the
    /// embeddings collapses (the cosine surface is flat there), while AnglE's
    /// angle objective keeps a meaningful gradient. This is the entire reason
    /// AnglE exists, and the contract this test pins.
    #[test]
    fn angle_gradient_is_non_vanishing_at_cosine_saturation() {
        let device = Device::Cpu;

        // Two pairs whose targets disagree with their (saturated) similarities,
        // so a valid ordering pair exists and both losses are non-trivial. Each
        // `b` is its `a` plus a tiny perturbation → cosine ≈ 1 for both pairs.
        let a = Var::from_tensor(
            &Tensor::new(&[[1.0f32, 0.5, -0.3, 0.8], [0.2, 0.9, 0.4, -0.1]], &device).unwrap(),
        )
        .unwrap();
        let b = Tensor::new(
            &[
                [1.0f32 + 1e-4, 0.5, -0.3, 0.8],
                [0.2, 0.9 + 1e-4, 0.4, -0.1],
            ],
            &device,
        )
        .unwrap();
        // Targets order pair 0 below pair 1.
        let scores = Tensor::new(&[0.0f32, 1.0], &device).unwrap();

        let a_t: &Tensor = &a;

        let cosent = cosent_reference(a_t, &b, &scores).unwrap();
        let cosent_grad = cosent.backward().unwrap();
        let cosent_norm = grad_norm(cosent_grad.get(a_t).unwrap());

        let angle = angle_loss(a_t, &b, &scores).unwrap();
        let angle_grad = angle.backward().unwrap();
        let angle_norm = grad_norm(angle_grad.get(a_t).unwrap());

        // CoSENT's gradient has all but vanished at saturation.
        assert!(
            cosent_norm < 1e-3,
            "expected CoSENT gradient to collapse near saturation, got {cosent_norm}"
        );
        // AnglE keeps an informative gradient there — orders of magnitude larger.
        assert!(
            angle_norm > 1e-2,
            "expected AnglE gradient to stay non-vanishing near saturation, got {angle_norm}"
        );
        assert!(
            angle_norm > cosent_norm * 100.0,
            "AnglE gradient ({angle_norm}) should dominate CoSENT's ({cosent_norm}) at saturation"
        );
    }

    /// cosine-MSE drives the predicted cosine toward the graded target: as the
    /// pair's cosine moves from far below the target up to it, the loss
    /// decreases monotonically and bottoms out near zero on a match.
    #[test]
    fn cosine_mse_tracks_graded_targets() {
        let device = Device::Cpu;
        // A single pair whose target is a graded score of 1.0 (perfectly
        // similar). Cosine of identical vectors is 1.0 → scaled prediction
        // matches the scaled target → loss ≈ 0.
        let aligned = Tensor::new(&[[1.0f32, 0.0, 0.0, 1.0]], &device).unwrap();
        let target_high = Tensor::new(&[1.0f32], &device).unwrap();
        let loss_match = cosine_mse_loss(&aligned, &aligned, &target_high)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            loss_match < 1e-4,
            "cosine-MSE should be ~0 when cosine equals the graded target, got {loss_match}"
        );

        // Orthogonal vectors (cosine 0) against a high target → large loss.
        let ortho = Tensor::new(&[[0.0f32, 1.0, 0.0, 0.0]], &device).unwrap();
        let base = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
        let loss_far = cosine_mse_loss(&base, &ortho, &target_high)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            loss_far > loss_match,
            "cosine-MSE should penalise a mismatched pair ({loss_far}) more than a matched one ({loss_match})"
        );

        // Moving cosine partway toward the target lowers the loss versus
        // orthogonal: the objective tracks the graded score continuously.
        let partial = Tensor::new(&[[1.0f32, 1.0, 0.0, 0.0]], &device).unwrap();
        let loss_partial = cosine_mse_loss(&base, &partial, &target_high)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            loss_partial < loss_far,
            "raising cosine toward the target should lower cosine-MSE: partial {loss_partial} vs far {loss_far}"
        );
    }

    /// AnglE requires an even hidden dimension to split into real/imaginary
    /// halves; an odd dimension is a typed error, not a panic.
    #[test]
    fn angle_rejects_odd_embedding_dimension() {
        let device = Device::Cpu;
        let odd = Tensor::new(&[[1.0f32, 0.0, 0.5]], &device).unwrap();
        let scores = Tensor::new(&[1.0f32], &device).unwrap();
        let err = angle_loss(&odd, &odd, &scores).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("even embedding dimension")),
            "expected an even-dimension error, got {err:?}"
        );
    }

    /// MNRL drives each anchor toward its own positive and away from the other
    /// rows' positives: a batch whose anchors already align with their matched
    /// positives (the diagonal of the similarity matrix dominates) has a far
    /// lower loss than one whose anchor↔positive matching is permuted.
    #[test]
    fn mnrl_rewards_diagonal_matching() {
        let device = Device::Cpu;
        // Three near-orthogonal directions; each anchor equals its positive, so
        // the similarity matrix is diagonal-dominant — the easy case.
        let anchor = Tensor::new(
            &[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        )
        .unwrap();
        let aligned = anchor.clone();
        let matched_loss = mnrl_loss(&anchor, &aligned, None, 20.0, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        // Permute the positives so each anchor's true positive sits off the
        // diagonal: the objective now penalises the mismatch.
        let permuted = Tensor::new(
            &[[0.0f32, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            &device,
        )
        .unwrap();
        let mismatched_loss = mnrl_loss(&anchor, &permuted, None, 20.0, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            matched_loss < mismatched_loss,
            "diagonal-matched batch ({matched_loss}) should score below a permuted one ({mismatched_loss})"
        );
        assert!(
            matched_loss < 0.05,
            "an already-aligned batch should have near-zero MNRL loss, got {matched_loss}"
        );
    }

    /// Appending an explicit hard negative that is *more* similar to the anchor
    /// than the in-batch negatives raises the MNRL loss versus the same batch
    /// without it: the hard negative is an extra, harder column to rank below
    /// the positive. This is the DPR recipe the `Triplet`-with-MNRL path uses.
    #[test]
    fn mnrl_hard_negative_sharpens_contrast() {
        let device = Device::Cpu;
        // A single (anchor, positive) row — no in-batch negatives at all — so
        // the loss without a hard negative is exactly zero (nothing to contrast
        // a single diagonal against).
        let anchor = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
        let positive = Tensor::new(&[[0.9f32, 0.1, 0.0, 0.0]], &device).unwrap();
        let no_neg = mnrl_loss(&anchor, &positive, None, 20.0, false)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            no_neg.abs() < 1e-5,
            "a lone (anchor, positive) row with no negatives has zero row loss, got {no_neg}"
        );

        // A hard negative very close to the anchor introduces a competing column
        // the anchor must rank below its positive — a strictly positive loss.
        let hard_neg = Tensor::new(&[[0.95f32, 0.05, 0.0, 0.0]], &device).unwrap();
        let with_neg = mnrl_loss(&anchor, &positive, Some(&hard_neg), 20.0, false)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            with_neg > no_neg,
            "a hard negative should raise the loss above the no-negative case: {with_neg} vs {no_neg}"
        );
    }

    /// The asymmetric (one-directional) MNRL option ranks only anchor→positive.
    /// On a batch whose anchor↔positive matching is symmetric, dropping the
    /// column direction halves the contribution but keeps the loss finite and
    /// non-negative — the asymmetric query→document objective the docstring
    /// promises.
    #[test]
    fn mnrl_asymmetric_drops_column_direction() {
        let device = Device::Cpu;
        let anchor = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
        let positive = Tensor::new(&[[0.3f32, 0.7], [0.7, 0.3]], &device).unwrap();
        let symmetric = mnrl_loss(&anchor, &positive, None, 20.0, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let one_dir = mnrl_loss(&anchor, &positive, None, 20.0, false)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(one_dir >= 0.0, "MNRL loss is non-negative, got {one_dir}");
        // For a similarity matrix that is its own transpose the two directions
        // are equal, so the symmetric mean equals the one-directional loss.
        assert!(
            (symmetric - one_dir).abs() < 1e-4,
            "with a symmetric similarity matrix both directions match: {symmetric} vs {one_dir}"
        );
    }

    /// Matryoshka wrapping evaluates the objective at each prefix dimension and
    /// sums — so a truncated-dim embedding still carries quality. The summed
    /// loss equals the per-dim losses added by hand (a faithful sum, not an
    /// approximation), which is what orders the leading coordinates by
    /// importance for serve-time truncation.
    #[test]
    fn matryoshka_sums_per_dimension_losses() {
        let device = Device::Cpu;
        // 4-d embeddings whose first 2 coordinates already separate the two
        // rows, so truncating to dim 2 keeps the diagonal dominant.
        let anchor =
            Tensor::new(&[[1.0f32, 0.0, 0.1, 0.2], [0.0, 1.0, 0.2, 0.1]], &device).unwrap();
        let positive =
            Tensor::new(&[[0.9f32, 0.1, 0.0, 0.3], [0.1, 0.9, 0.3, 0.0]], &device).unwrap();

        let objective = |dims: Vec<Tensor>| mnrl_loss(&dims[0], &dims[1], None, 20.0, true);
        let wrapped = matryoshka_sum(&[4, 2], &[&anchor, &positive], &objective)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        // The same objective evaluated by hand at each prefix dim and summed.
        let mut by_hand = 0.0f32;
        for dim in [4usize, 2] {
            let a = anchor.narrow(1, 0, dim).unwrap();
            let p = positive.narrow(1, 0, dim).unwrap();
            by_hand += mnrl_loss(&a, &p, None, 20.0, true)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
        }
        assert!(
            (wrapped - by_hand).abs() < 1e-4,
            "matryoshka wrapper must sum the per-dim losses: {wrapped} vs {by_hand}"
        );

        // No dims = the objective applied once at full width.
        let full = matryoshka_sum(&[], &[&anchor, &positive], &objective)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let direct = mnrl_loss(&anchor, &positive, None, 20.0, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (full - direct).abs() < 1e-6,
            "empty dims must be a no-op wrap"
        );
    }

    /// A Matryoshka dim wider than the embedding is a typed error, not a silent
    /// clamp — truncation must be a true prefix.
    #[test]
    fn matryoshka_rejects_oversized_dim() {
        let device = Device::Cpu;
        let anchor = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
        let positive = Tensor::new(&[[0.9f32, 0.1, 0.0, 0.0]], &device).unwrap();
        let objective = |dims: Vec<Tensor>| mnrl_loss(&dims[0], &dims[1], None, 20.0, true);
        let err = matryoshka_sum(&[8], &[&anchor, &positive], &objective).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("exceeds the embedding width")),
            "expected an oversized-dim error, got {err:?}"
        );
    }

    /// Selecting MNRL for a graded `(text_a, text_b, score)` Contrastive batch
    /// is a typed error rather than a silent fall-through to CoSENT — the
    /// previously-latent silent-wrong-loss bug. The loss/batch mismatch is
    /// surfaced, not quietly satisfied by a different objective.
    #[test]
    fn mnrl_on_graded_batch_is_a_typed_error() {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
        let b = Tensor::new(&[[0.9f32, 0.1], [0.1, 0.9]], &device).unwrap();
        let scores = Tensor::new(&[1.0f32, 0.5], &device).unwrap();
        // The CoSENT fallback must never be reached for an MNRL config — assert
        // the dispatch errors before invoking it.
        let never = |_: &Tensor, _: &Tensor, _: &Tensor| -> Result<Tensor> {
            panic!("CoSENT fallback must not run for an MNRL config — silent fall-through")
        };
        let err = dispatch_contrastive_loss(
            Some(crate::fine_tune::EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
            &a,
            &b,
            &scores,
            &never,
        )
        .unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("in-batch-negative objective")),
            "MNRL on a graded batch must be a typed mismatch error, got {err:?}"
        );

        // The triplet-margin variant on a graded batch is the same mismatch.
        let err2 = dispatch_contrastive_loss(
            Some(crate::fine_tune::EmbeddingLoss::Triplet { margin: 0.3 }),
            &a,
            &b,
            &scores,
            &never,
        )
        .unwrap_err();
        assert!(
            matches!(err2, JammiError::FineTune(ref m) if m.contains("Triplet loss needs")),
            "triplet on a graded batch must be a typed mismatch error, got {err2:?}"
        );
    }

    /// The pairwise ordering loss is zero when no target pair is mis-ordered:
    /// with a single pair (no valid `i<j` ordering), `log(1) = 0`.
    #[test]
    fn pairwise_ordering_loss_is_zero_without_valid_pairs() {
        let device = Device::Cpu;
        let sim = Tensor::new(&[5.0f32], &device).unwrap();
        let scores = Tensor::new(&[1.0f32], &device).unwrap();
        let loss = pairwise_ordering_loss(&sim, &scores)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.abs() < 1e-6, "expected zero loss, got {loss}");
    }

    // ─── Distributional regression objectives (S18) ──────────────────────────

    use crate::fine_tune::adamw::{AdamW, ParamsAdamW};
    use candle_nn::VarMap;

    /// A heteroscedastic synthetic regression set with TWO feature groups that
    /// share ONE regression mean but have different noise: a low-noise group
    /// (targets tightly around `+offset`) and a high-noise group (targets widely
    /// scattered around `−offset`). The true balanced mean is `0`. Because the
    /// two groups disagree on where the mean should sit and one is far noisier,
    /// a *shared* mean is exactly the setting where joint `μ,σ²` NLL down-weights
    /// the noisy group (inflating its variance) and the shared mean drifts toward
    /// the low-noise group — the variance-collapse / mean-starvation pathology.
    /// Returns `(group_id_per_row, targets, true_stds)`; the true shared mean is
    /// `0` and the per-group offsets are `±offset`.
    fn heteroscedastic_set(device: &Device) -> (Vec<usize>, Tensor, [f32; 2]) {
        // Group 0: centred at +2.0, std ≈0.1 (easy, tight).
        // Group 1: centred at −2.0, std ≈3.0 (hard, scattered).
        // The balanced (variance-agnostic) mean of the two centres is 0.0.
        let true_stds = [0.1f32, 3.0];
        let g0: Vec<f32> = [-0.1, 0.1, -0.05, 0.05, -0.08, 0.08]
            .iter()
            .map(|d| 2.0 + d)
            .collect();
        let g1: Vec<f32> = [-3.0, 3.0, -1.5, 1.5, -4.5, 4.5]
            .iter()
            .map(|d| -2.0 + d)
            .collect();
        let mut groups = Vec::new();
        let mut targets = Vec::new();
        for v in g0 {
            groups.push(0);
            targets.push(v);
        }
        for v in g1 {
            groups.push(1);
            targets.push(v);
        }
        let t = Tensor::from_vec(targets, (groups.len(),), device).unwrap();
        (groups, t, true_stds)
    }

    /// Fit ONE shared mean plus a PER-GROUP `raw_std` to the heteroscedastic set
    /// under `loss_fn`. The shared mean is the contested parameter: how far it
    /// drifts toward the low-noise group's centre is the variance-collapse
    /// signature. Returns `(shared_mean, [σ_easy, σ_hard])`.
    fn fit_shared_mean(
        device: &Device,
        groups: &[usize],
        targets: &Tensor,
        loss_fn: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>,
        steps: usize,
    ) -> (f32, [f32; 2]) {
        let varmap = VarMap::new();
        // One shared mean, initialised at 0.
        let mean = varmap
            .get(
                (1,),
                "mean",
                candle_nn::Init::Const(0.0),
                DType::F32,
                device,
            )
            .unwrap();
        // Per-group raw_std, initialised at 0 (σ ≈ ln2 + floor).
        let raw_std = varmap
            .get(
                (2,),
                "raw_std",
                candle_nn::Init::Const(0.0),
                DType::F32,
                device,
            )
            .unwrap();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();

        let n = groups.len();
        let group_idx = Tensor::from_vec(
            groups.iter().map(|&g| g as u32).collect::<Vec<_>>(),
            (n,),
            device,
        )
        .unwrap();

        for _ in 0..steps {
            // Broadcast the shared mean to every row; gather each row's raw_std
            // from its group. Stack into a (n, 2) head output.
            let mean_col = mean.broadcast_as((n, 1)).unwrap().contiguous().unwrap();
            let raw_col = raw_std
                .index_select(&group_idx, 0)
                .unwrap()
                .reshape((n, 1))
                .unwrap();
            let head = Tensor::cat(&[&mean_col, &raw_col], 1).unwrap();
            let loss = loss_fn(&head, targets).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }

        let m = mean.to_vec1::<f32>().unwrap()[0];
        let raws = raw_std.to_vec1::<f32>().unwrap();
        let sigmas = [
            softplus_std_for_test(raws[0] as f64) as f32,
            softplus_std_for_test(raws[1] as f64) as f32,
        ];
        (m, sigmas)
    }

    /// The variance-collapse / mean-starvation pathology, and its fix. With ONE
    /// shared mean over a low-noise group (centred +2) and a high-noise group
    /// (centred −2), naive joint `μ,σ²` NLL inflates the noisy group's variance,
    /// down-weighting its residuals, so the shared mean is pulled toward the
    /// low-noise group (well above the balanced mean of 0) — the
    /// Seitzer/Nix-Weigend pathology. β-NLL restores the noisy group's mean
    /// gradient, so its shared mean sits markedly closer to the balanced 0; CRPS
    /// likewise. This is the [HIGH] contract regression test.
    #[test]
    fn beta_nll_and_crps_avoid_the_naive_nll_mean_starvation() {
        let device = Device::Cpu;
        let (groups, targets, _) = heteroscedastic_set(&device);
        let steps = 2000;

        let naive = |i: &Tensor, t: &Tensor| gaussian_nll_loss(i, t, 0.0);
        let beta = |i: &Tensor, t: &Tensor| gaussian_nll_loss(i, t, 0.5);
        let crps = crps_gaussian_loss;

        let (naive_mean, naive_sigmas) = fit_shared_mean(&device, &groups, &targets, &naive, steps);
        let (beta_mean, _) = fit_shared_mean(&device, &groups, &targets, &beta, steps);
        let (crps_mean, _) = fit_shared_mean(&device, &groups, &targets, &crps, steps);

        // Naive NLL collapses toward the low-noise group: it inflates the hard
        // group's σ (so its residuals barely count) and the shared mean drifts
        // well above the balanced mean of 0.
        assert!(
            naive_sigmas[1] > naive_sigmas[0] * 3.0,
            "naive NLL should inflate the hard group's variance: σ {naive_sigmas:?}"
        );
        assert!(
            naive_mean > 0.7,
            "naive NLL's shared mean should be pulled toward the low-noise \
             group (well above the balanced 0), got {naive_mean}"
        );
        // β-NLL pulls the shared mean back toward the balanced 0 — strictly
        // closer than naive NLL.
        assert!(
            beta_mean.abs() < naive_mean.abs(),
            "β-NLL should pull the shared mean back toward balance \
             (β-NLL mean {beta_mean}, naive mean {naive_mean})"
        );
        // CRPS, the other collapse-resistant objective, likewise.
        assert!(
            crps_mean.abs() < naive_mean.abs(),
            "CRPS should pull the shared mean back toward balance \
             (CRPS mean {crps_mean}, naive mean {naive_mean})"
        );
    }

    /// Heteroscedasticity is the point: the fitted σ is INPUT-DEPENDENT — the
    /// high-noise group gets a much larger predictive std than the low-noise
    /// group, tracking the true noise. A single global σ (collapsed
    /// heteroscedasticity) would fail this. Demonstrated under the default
    /// β-NLL objective.
    #[test]
    fn fitted_variance_is_input_dependent() {
        let device = Device::Cpu;
        let (groups, targets, true_stds) = heteroscedastic_set(&device);
        let beta = |i: &Tensor, t: &Tensor| gaussian_nll_loss(i, t, 0.5);
        let (_, sigmas) = fit_shared_mean(&device, &groups, &targets, &beta, 2000);

        // The hard group's σ is far larger than the easy group's — variance
        // varies with input difficulty.
        assert!(
            sigmas[1] > sigmas[0] * 3.0,
            "predictive std must track input difficulty: easy σ {}, hard σ {}",
            sigmas[0],
            sigmas[1]
        );
        // Both stay in the right ballpark of the true noise (loose bounds — a
        // bounded fit, not a precise one).
        assert!(
            sigmas[1] > 1.0,
            "fitted hard-group σ should be large, tracking true stds {true_stds:?}: got {sigmas:?}"
        );
    }

    /// The predictive σ never collapses to (near) zero even when the head is
    /// pushed toward overconfidence: the `STD_FLOOR` guards every NLL/CRPS term.
    #[test]
    fn predictive_std_respects_the_floor() {
        let device = Device::Cpu;
        // A head with a very negative raw_std → softplus → ≈0, plus the floor.
        let input = Tensor::new(&[[1.0f32, -50.0]], &device).unwrap();
        let (_, sigma) = gaussian_params(&input).unwrap();
        let s = sigma.to_vec1::<f32>().unwrap()[0];
        assert!(
            s >= STD_FLOOR as f32,
            "σ {s} fell below the floor {STD_FLOOR}"
        );
    }

    /// Fit a Gaussian head — carrying the de-standardising affine the trainer
    /// builds on it — to a REALISTIC, high-offset, low-variance regression target:
    /// calendar years 2014..=2020 (true mean ≈ 2017, true std ≈ 2). This is
    /// exactly the shape of a real "predict the filing year" regression. Same
    /// optimiser budget the engine's own regression contract tests use (AdamW
    /// lr=0.05, 2000 steps).
    ///
    /// The head's learnable column is a **z-space** parameter, zero-init; the
    /// head's forward de-standardises it through the [`TargetScaler`]'s affine
    /// (`μ_y + σ_y·z`), so the emitted mean starts at exactly μ_y and the loss
    /// scores that raw-correct output against the raw target. Adam only has to
    /// move the O(1) z-parameter, which is reachable in the budget.
    ///
    /// ORACLE: the fitted *served* mean (the de-standardised head output) must
    /// land near the true mean (2017). A z-param scored through a head that did
    /// NOT de-standardise — i.e. scoring the raw z-space output against the raw
    /// 2017-offset target — is the failure mode this guards: Adam's per-step move
    /// is ≈ lr regardless of loss scale, so an un-reparameterised mean travels at
    /// most ~100 units and stalls thousands short of 2017.
    #[test]
    fn gaussian_head_fits_high_offset_low_variance_target() {
        let device = Device::Cpu;
        // Calendar years — a textbook low-variance, high-offset regression target.
        let years: Vec<f32> = vec![2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0];
        let true_mean: f32 = years.iter().sum::<f32>() / years.len() as f32; // 2017.0
        let n = years.len();
        let targets = Tensor::from_vec(years, (n,), &device).unwrap();
        // The affine the head carries, reduced from the same targets the trainer
        // reduces — so the head emits μ_y at zero-init.
        let scaler = TargetScaler::from_targets(&targets).unwrap();

        // Two learnable z-space columns (z_mean, raw_std), both zero-init — the
        // head's own parameterisation (build_distribution_head starts at 0).
        let varmap = VarMap::new();
        let z_mean = varmap
            .get(
                (1,),
                "z_mean",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &device,
            )
            .unwrap();
        let raw_std = varmap
            .get(
                (1,),
                "raw_std",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &device,
            )
            .unwrap();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();

        for _ in 0..2000 {
            let z_col = z_mean.broadcast_as((n, 1)).unwrap().contiguous().unwrap();
            let raw_col = raw_std.broadcast_as((n, 1)).unwrap().contiguous().unwrap();
            let z_head = Tensor::cat(&[&z_col, &raw_col], 1).unwrap();
            // The head's forward: de-standardise the z-space output to raw units.
            let head = scaler.destandardize_gaussian(&z_head).unwrap();
            // β-NLL with β=0.5 — the engine's default regression loss — scores the
            // raw head output against the raw target, no scaler in the loss.
            let loss = gaussian_nll_loss(&head, &targets, 0.5).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }

        // The SERVED mean is the de-standardised head output, exactly what the
        // serving adapter reads. Reproduce the head forward at the fitted params.
        let z_col = z_mean.broadcast_as((n, 1)).unwrap().contiguous().unwrap();
        let raw_col = raw_std.broadcast_as((n, 1)).unwrap().contiguous().unwrap();
        let z_head = Tensor::cat(&[&z_col, &raw_col], 1).unwrap();
        let served = scaler.destandardize_gaussian(&z_head).unwrap();
        let served_mean = served.to_vec2::<f32>().unwrap()[0][0];
        // ORACLE: a calibrated regression head predicts ≈ the true target mean.
        // Allow a generous ±50 (the true std is ~2, so ±50 is 25σ of slack).
        assert!(
            (served_mean - true_mean).abs() < 50.0,
            "Gaussian head failed to fit a realistic high-offset target. \
             true mean {true_mean}, served mean {served_mean} (off by {:.0}). \
             The head's de-standardised mean must converge to the target mean \
             under the z-space reparameterisation.",
            (served_mean - true_mean).abs()
        );
    }

    /// The pinball objective drives each quantile head toward its level: the
    /// fitted median sits at the data's median, the low quantile below it, the
    /// high quantile above it (monotone, non-crossing). The non-crossing penalty
    /// keeps the order during training; here we assert the trained head is
    /// coherent and well-ordered.
    #[test]
    fn pinball_trains_ordered_quantiles_to_their_levels() {
        let device = Device::Cpu;
        // Targets symmetric around 0 with spread, so the 0.5 quantile → 0, the
        // 0.1 quantile is negative, the 0.9 quantile positive.
        let targets = Tensor::new(&[-4.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], &device).unwrap();
        let n = 7;
        let levels = [0.1f64, 0.5, 0.9];

        let varmap = VarMap::new();
        let q = varmap
            .get(
                (1, 3),
                "q",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &device,
            )
            .unwrap();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..800 {
            let preds = q.broadcast_as((n, 3)).unwrap().contiguous().unwrap();
            let loss = pinball_loss(&preds, &targets, &levels).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        let fitted = q.to_vec2::<f32>().unwrap()[0].clone();
        // Monotone: q10 < q50 < q90 — zero crossings.
        assert!(
            fitted[0] < fitted[1] && fitted[1] < fitted[2],
            "pinball quantiles must be ordered (non-crossing): {fitted:?}"
        );
        // The median sits near the data median (0).
        assert!(
            fitted[1].abs() < 0.8,
            "fitted median should be ≈0, got {}",
            fitted[1]
        );
        // The 0.1 quantile is below the median, the 0.9 above.
        assert!(
            fitted[0] < -0.5 && fitted[2] > 0.5,
            "tails mis-placed: {fitted:?}"
        );
    }

    /// The non-crossing penalty is strictly positive when the head emits a
    /// CROSSING set and zero when ordered — the training-time guard against
    /// quantile crossing.
    #[test]
    fn pinball_penalises_crossing_heads() {
        let device = Device::Cpu;
        let targets = Tensor::new(&[0.0f32], &device).unwrap();
        let levels = [0.1f64, 0.5, 0.9];
        // Ordered head (no crossing).
        let ordered = Tensor::new(&[[-1.0f32, 0.0, 1.0]], &device).unwrap();
        // Crossing head (q10 > q90).
        let crossing = Tensor::new(&[[1.0f32, 0.0, -1.0]], &device).unwrap();
        let l_ordered = pinball_loss(&ordered, &targets, &levels)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let l_crossing = pinball_loss(&crossing, &targets, &levels)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // The crossing head pays the extra non-crossing penalty on top of the
        // pinball term, so its loss is strictly larger.
        assert!(
            l_crossing > l_ordered,
            "a crossing head must cost more than an ordered one \
             (crossing {l_crossing}, ordered {l_ordered})"
        );
    }
}

/// Host-read accounting for the per-micro-batch training path (B3 close):
/// [`TrainingLoop::accumulate_sim_stats`] must issue ZERO device→host reads
/// (it used to do two `to_scalar::<f32>()` per triplet micro-batch, BEFORE
/// `backward` ever ran), and [`TrainingLoop::process_batch_loss`] must issue
/// exactly ONE (its post-backward loss read) — for every loss arm, since
/// `process_batch_loss` takes an already-computed loss [`Tensor`] and never
/// branches on which objective produced it, so exercising it once here
/// structurally covers CoSENT/MNRL/Triplet/CE/regression alike.
#[cfg(test)]
mod host_read_discipline {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor, Var};
    use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
    use serial_test::serial;

    use super::super::data::TrainingBatch;
    use super::super::lora::build_distribution_head;
    use super::super::target::TrainingTarget;
    use super::super::FineTuneConfig;
    use super::{
        per_micro_batch_host_read_count, EpochState, SimStats, StepContext, TrainingLoop,
        TrainingLoopBuilder,
    };
    use crate::fine_tune::adamw::AdamW;

    const HIDDEN: usize = 4;

    /// A minimal real [`TrainingLoop`]. `process_batch_loss` and
    /// `accumulate_sim_stats` never touch the catalog (only `TrainingLoop::run`
    /// does, via `mark_training_running`), so this skips the register/create/
    /// claim job plumbing the production-dispatch oracles elsewhere in this
    /// file use for that reason — an empty, unclaimed catalog is enough here.
    async fn minimal_loop(device: &Device) -> TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let config = FineTuneConfig::default();
        let head = build_distribution_head(HIDDEN, 2, &config, &varmap, &vb).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("host-read-oracle-job".into())
            .worker_id("host-read-oracle-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap()
    }

    // The three tests below all read `PER_MICRO_BATCH_HOST_READ_COUNT`
    // (directly, or indirectly by calling `accumulate_sim_stats`/
    // `process_batch_loss`). `cargo test` runs tests in parallel threads
    // within the SAME process, so an unmarked set racing on that counter
    // would be flaky; `#[serial(..)]` under a shared key forces them to run
    // one at a time relative to each other.

    #[test]
    #[serial(trainer_host_read_count)]
    fn accumulate_sim_stats_never_reads_the_device_back() {
        let device = Device::Cpu;
        let anchor = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
        let positive = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
        let negative = Tensor::new(&[[0.0f32, 1.0], [1.0, 0.0]], &device).unwrap();
        let batch = TrainingBatch::Triplet {
            anchor,
            positive,
            negative,
        };

        let mut stats: Option<SimStats> = None;

        let before = per_micro_batch_host_read_count();
        TrainingLoop::accumulate_sim_stats(&batch, &mut stats);
        assert_eq!(
            per_micro_batch_host_read_count(),
            before,
            "accumulate_sim_stats must not perform any device→host read"
        );
        let stats = stats.unwrap();
        assert_eq!(stats.count, 1);

        // Numeric correctness of the on-device fold survives the move off
        // host `f64` accumulation: cosine similarity of identical rows is
        // 1.0 (positive pair), of the swapped/orthogonal rows is 0.0
        // (negative pair) — read back ONCE here, in the test, which is not
        // the per-micro-batch path this oracle guards.
        let pos_val: f32 = stats.pos.to_scalar().unwrap();
        let neg_val: f32 = stats.neg.to_scalar().unwrap();
        assert!((pos_val - 1.0).abs() < 1e-5, "got {pos_val}");
        assert!(neg_val.abs() < 1e-5, "got {neg_val}");
    }

    #[test]
    #[serial(trainer_host_read_count)]
    fn accumulate_sim_stats_ignores_non_triplet_batches_without_reading() {
        let device = Device::Cpu;
        let batch = TrainingBatch::Regression {
            input: Tensor::zeros((2, 2), DType::F32, &device).unwrap(),
            target: Tensor::zeros((2,), DType::F32, &device).unwrap(),
        };
        let mut stats: Option<SimStats> = None;

        let before = per_micro_batch_host_read_count();
        TrainingLoop::accumulate_sim_stats(&batch, &mut stats);
        assert_eq!(per_micro_batch_host_read_count(), before);
        assert!(stats.is_none());
    }

    /// RED-first (B1): before this fix, `accumulate_sim_stats` folded a
    /// TRACKED per-batch mean (`track_op() == true` whenever
    /// `anchor`/`positive`/`negative` come from a real forward pass, as they
    /// do in production via `encode_chunk`'s LoRA `Var`s) into the epoch
    /// accumulator with no `detach()`, retaining every micro-batch's forward
    /// subgraph for the whole epoch — the auditor measured the accumulator's
    /// `sorted_nodes().len()` growing by a fixed per-call increment (23, 46,
    /// 69, 92, 115) over five calls. This test builds its triplet batch
    /// THROUGH a `Var` (`w.as_tensor()` has `is_variable() == true`, so every
    /// `cosine_similarity`/`mean_all` op stacked on top of it is tracked) — a
    /// leaf `Tensor::new` fixture, as the two tests above use, is untracked
    /// from the start and CANNOT see this regression (the audit's exact
    /// point: "the existing tests use leaf tensors and cannot see this").
    ///
    /// Mutation tried: delete the `.detach()` calls inside
    /// `accumulate_sim_stats` (both the per-batch mean's and the fold's) —
    /// this test goes red on the `track_op()` assertion at fold 1 already
    /// (`sorted_nodes().len()` nonzero and growing with `n` from there).
    #[test]
    fn accumulate_sim_stats_accumulator_is_always_a_graph_leaf() {
        let device = Device::Cpu;
        let w =
            Var::from_tensor(&Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap()).unwrap();
        let neg_w =
            Var::from_tensor(&Tensor::new(&[[0.0f32, 1.0], [1.0, 0.0]], &device).unwrap()).unwrap();

        let mut stats: Option<SimStats> = None;
        for n in 1..=5usize {
            // Fresh ops on top of the SAME Vars every call — mirrors a fresh
            // forward pass through the same LoRA weights each micro-batch, so
            // a real regression (a retained subgraph) grows the
            // accumulator's node count on every fold, not just the first.
            let anchor = w.as_tensor().affine(1.0, 0.0).unwrap();
            let positive = w.as_tensor().affine(1.0, 0.0).unwrap();
            let negative = neg_w.as_tensor().affine(1.0, 0.0).unwrap();
            assert!(
                anchor.track_op(),
                "test setup: the Var-rooted fixture must itself be tracked"
            );
            let batch = TrainingBatch::Triplet {
                anchor,
                positive,
                negative,
            };

            TrainingLoop::accumulate_sim_stats(&batch, &mut stats);

            let s = stats.as_ref().unwrap();
            assert_eq!(s.count, n);
            assert!(
                !s.pos.track_op(),
                "pos accumulator must be a detached leaf after fold {n}"
            );
            assert!(
                !s.neg.track_op(),
                "neg accumulator must be a detached leaf after fold {n}"
            );
            assert_eq!(
                s.pos.sorted_nodes().len(),
                0,
                "pos accumulator must not retain a forward graph after fold {n}"
            );
            assert_eq!(
                s.neg.sorted_nodes().len(),
                0,
                "neg accumulator must not retain a forward graph after fold {n}"
            );
        }
    }

    /// RED-first (B3): before this fix, `accumulate_sim_stats` did two
    /// `to_scalar::<f32>()` calls on every triplet micro-batch, BEFORE
    /// `backward` ever ran. Mutation tried: reinstate a `to_scalar` inside
    /// `accumulate_sim_stats` (re-add the old per-call host read that used to
    /// populate `epoch_pos_sim`/`epoch_neg_sim` as `f64`s directly, instead of
    /// folding device tensors) — `accumulate_sim_stats_never_reads_the_device_
    /// back` above goes red because the counter moves. Separately,
    /// `process_batch_loss`'s ONE remaining read is pinned here: mutation
    /// tried — delete the counter increment at its post-backward
    /// `to_scalar` — this test goes red (`before` instead of `before + 1`).
    ///
    /// Advisory follow-up: `PER_MICRO_BATCH_HOST_READ_COUNT` (this file) only
    /// counts the loss-scalar read this function itself issues — it is NOT
    /// the whole device→host read count for this call. The `StepContext`
    /// below (`total_steps: 1`, `batches_per_epoch: 1`) makes this micro-batch
    /// BOTH `step == 1` and the run's last step, so `clip_and_step`'s cadence
    /// gate (`optimizer.rs`) also fires and reads the grad-norm back through
    /// `refuse_nonfinite_norm` — a SEPARATE counter
    /// (`optimizer::sync_read_count`) this file does not own. Both are
    /// asserted here (against their own before/after deltas, not summed into
    /// one number) so the two-reads-per-call fact for THIS setup is pinned
    /// explicitly rather than left implicit in the first assertion's message.
    #[test]
    #[serial(trainer_host_read_count, grad_clip_sync_read_count)]
    fn process_batch_loss_reads_the_device_exactly_once_per_micro_batch() {
        let device = Device::Cpu;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut loop_ = rt.block_on(minimal_loop(&device));

        // A loss tensor built from an arbitrary trainable var. `process_batch_loss`
        // never branches on which loss function produced its `loss` argument, so
        // this structurally covers every loss arm — the read this test pins is
        // the same call regardless of whether the loss came from CoSENT, MNRL,
        // triplet margin, classification CE, or a regression objective.
        let trainable_vars = loop_.varmap.all_vars();
        let w = trainable_vars[0].clone();
        let loss = w.as_tensor().sqr().unwrap().sum_all().unwrap();

        let mut optimizer = AdamW::new(
            trainable_vars.clone(),
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )
        .unwrap();
        let mut batch_count = 0usize;
        let mut epoch_loss = 0.0f64;
        let mut accumulated_grads = None;
        let mut global_step = 0usize;
        let checkpoint_dir = tempfile::tempdir().unwrap();

        let loss_before = per_micro_batch_host_read_count();
        let clip_before = crate::fine_tune::optimizer::sync_read_count();
        loop_
            .process_batch_loss(
                loss,
                EpochState {
                    batch_count: &mut batch_count,
                    epoch_loss: &mut epoch_loss,
                    accumulated_grads: &mut accumulated_grads,
                    global_step: &mut global_step,
                },
                StepContext {
                    trainable_vars: &trainable_vars,
                    optimizer: &mut optimizer,
                    checkpoint_dir: checkpoint_dir.path(),
                    checkpoint_interval: 0,
                    total_steps: 1,
                    batches_per_epoch: 1,
                },
            )
            .unwrap();

        assert_eq!(
            per_micro_batch_host_read_count(),
            loss_before + 1,
            "process_batch_loss's own post-backward loss read (this file's counter) must be \
             exactly one device→host read per micro-batch"
        );
        // A SEPARATE device→host read: `total_steps: 1` + a single micro-batch
        // makes this call both `step == 1` and the run's last step, so
        // `clip_and_step`'s cadence gate also fires and reads the grad norm
        // back via `optimizer::refuse_nonfinite_norm` — pinned here against
        // its own counter so this call's TRUE read count (two, not one) is
        // explicit rather than left to the assertion above's message alone.
        assert_eq!(
            crate::fine_tune::optimizer::sync_read_count(),
            clip_before + 1,
            "clip_and_step's cadence-gated grad-norm read (optimizer::SYNC_READ_COUNT) must \
             also fire exactly once for this call"
        );
    }
}

/// B2/B3 last-step lattice row: **GradCache**. `run_gradcache_epoch` has no
/// loss-value divergence check of its own (unlike `process_batch_loss`'s
/// `loss_val.is_nan() || loss_val > 100.0` guard) — the ONLY thing that can
/// ever catch a diverged GradCache step is `clip_and_step`'s cadence-gated
/// `refuse_nonfinite_norm`, so getting `is_last_step` right on this arm is
/// load-bearing in a way the accumulation-window arm (which has the
/// loss-value backstop too) is not.
#[cfg(test)]
mod gradcache_last_step_oracle {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{ParamsAdamW, VarBuilder, VarMap};

    use super::super::data::TrainingDataLoader;
    use super::super::lora::build_projection_head;
    use super::super::target::TrainingTarget;
    use super::super::{EmbeddingLoss, FineTuneConfig};
    use super::TrainingLoopBuilder;
    use crate::fine_tune::adamw::AdamW;
    use crate::model::{ModelSource, ModelTask};

    const HIDDEN: usize = 32; // tiny_bert's hidden width.

    /// Load the hermetic `tiny_bert` cookbook fixture through a real
    /// `InferenceSession`'s model cache — the same resolve+backend-load path
    /// serving uses (see `ModelCache::load_owned_for_test`'s doc on
    /// `session.rs`'s equivalent seam). Real, but tiny and local: no network,
    /// sub-second load, matching every other `tiny_bert`-fixture test in
    /// `tests/it`.
    async fn tiny_bert() -> Arc<crate::model::LoadedModel> {
        let dir = tempfile::tempdir().unwrap();
        let config = jammi_test_utils::test_config(dir.path());
        let session = crate::session::InferenceSession::new(config).await.unwrap();
        let source = ModelSource::Local(jammi_test_utils::cookbook_fixture("tiny_bert"));
        let guard = session
            .model_cache()
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .unwrap();
        guard.model.clone()
    }

    /// RED-first (B2/B3): `run_gradcache_epoch`'s `is_last_step` must reflect
    /// GradCache's TRUE per-epoch horizon (`self.config.epochs`), not the
    /// accumulation-window arm's `ceil(batches / grad_accum) * epochs` — the
    /// two differ whenever a GradCache epoch chunks into more than one
    /// `batch_size`-sized memory-bounded pass (6 rows / `batch_size: 2` = 3
    /// chunks here, so the WRONG horizon would be `ceil(3/1) * 2 == 6`
    /// against the correct `2`).
    ///
    /// This drives `run_gradcache_epoch` directly (private, same-file access)
    /// exactly as `run` calls it: epoch 0 first (healthy weights, NOT the
    /// run's last step), then — DETERMINISTICALLY, not relying on organic
    /// numeric divergence — corrupts one trainable `Var` to `NaN` before
    /// calling it a second time for epoch 1 (`epochs: 2`, so this IS the
    /// run's last step). The correct `total_optimizer_steps` (`2`, matching
    /// `self.config.epochs`) must make `is_last_step == true` on that second
    /// call, forcing `clip_and_step`'s non-finite check regardless of the
    /// `DEFAULT_NORM_CHECK_INTERVAL` modulo cadence (which a 2-step run never
    /// reaches).
    ///
    /// Mutation tried: hardcode `is_last_step = false` inside
    /// `run_gradcache_epoch` (the bug B2 fixed, reintroduced) — this test
    /// goes red: the second call returns `Ok` with a NaN weight trained in
    /// silently, instead of the typed non-finite refusal.
    #[test]
    fn run_gradcache_epoch_catches_a_nonfinite_gradient_on_the_runs_last_epoch() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let base_model = tiny_bert().await;

            let device = Device::Cpu;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let config = FineTuneConfig {
                cached: true,
                embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
                batch_size: 2,
                epochs: 2,
                lora_rank: 2,
                ..Default::default()
            };
            let head = build_projection_head(HIDDEN, &config, &varmap, &vb).unwrap();

            let job_dir = tempfile::tempdir().unwrap();
            let catalog = Arc::new(
                jammi_db::catalog::Catalog::open(job_dir.path())
                    .await
                    .unwrap(),
            );
            let mut loop_ =
                TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
                    .device(device.clone())
                    .job_id("gradcache-last-step-job".into())
                    .worker_id("gradcache-last-step-worker".into())
                    .catalog(catalog)
                    .artifact_dir(job_dir.path().to_path_buf())
                    .base_model(base_model)
                    .build()
                    .unwrap();

            // 6 (anchor, positive) pairs — MNRL's in-batch-negative pool —
            // chunked at `batch_size: 2` into 3 memory-bounded GradCache
            // passes per epoch, so the pre-fix (WRONG) horizon
            // (`ceil(3 / 1) * 2 == 6`) visibly differs from the correct one
            // (`self.config.epochs == 2`).
            let rows: Vec<(String, String)> = (0..6)
                .map(|i| (format!("anchor text {i}"), format!("positive text {i}")))
                .collect();
            let loader = TrainingDataLoader::from_pairs(rows);

            let trainable_vars = loop_.varmap.all_vars();
            assert!(
                !trainable_vars.is_empty(),
                "test setup: the projection head must have trainable LoRA vars"
            );
            let mut optimizer = AdamW::new(
                trainable_vars.clone(),
                ParamsAdamW {
                    lr: 0.01,
                    ..Default::default()
                },
            )
            .unwrap();

            // Epoch 0: healthy weights, global_step 0 → 1. `total_optimizer_steps
            // == 2` (== `self.config.epochs`) is exactly what `run` computes
            // for this arm (see `total_optimizer_steps`'s doc in `run`).
            let global_step_after_epoch0 = 0usize;
            loop_
                .run_gradcache_epoch(
                    &loader,
                    &trainable_vars,
                    &mut optimizer,
                    2,
                    global_step_after_epoch0,
                )
                .expect("epoch 0 must complete on healthy weights");

            // Deterministically corrupt one trainable var to NaN — the
            // divergence this test pins, not left to organic numeric chance.
            let w = &trainable_vars[0];
            let nan = Tensor::full(f32::NAN, w.dims(), &device).unwrap();
            w.set(&nan).unwrap();

            // Epoch 1 is THIS run's actual last optimizer step
            // (`global_step + 1 == 2 == total_optimizer_steps`).
            let global_step_after_epoch1 = 1usize;
            let err = loop_
                .run_gradcache_epoch(
                    &loader,
                    &trainable_vars,
                    &mut optimizer,
                    2,
                    global_step_after_epoch1,
                )
                .expect_err(
                    "a NaN weight on the run's last GradCache epoch must surface a typed \
                     non-finite refusal, not train silently",
                );
            assert!(
                err.to_string().contains("non-finite"),
                "expected a non-finite grad-norm refusal, got: {err}"
            );
        });
    }
}

/// The standardisation-contract oracle for the **production fine-tune regression
/// path** (heads 6/7 — Gaussian + quantile).
///
/// This is the genuinely-new coverage W5-PR1 adds. Unlike the MATH-level
/// `gaussian_head_fits_high_offset_low_variance_target` /
/// `pinball_trains_ordered_quantiles_to_their_levels` tests above (which hand-roll
/// a `VarMap` head + scaler and call the loss functions directly), these oracles
/// drive the **actual production dispatch** a real `db.fine_tune(task=regression)`
/// exercises: a [`TrainingLoop`] built by the production [`TrainingLoopBuilder`],
/// holding a real [`TrainingTarget::ProjectionHead`] regression head assembled by
/// the production [`build_distribution_head`], with its [`TargetScaler`] reduced
/// from the training targets exactly as `TrainingLoop::run` does — then each step
/// runs the head forward through the production `TrainingLoop::regress`
/// (which applies `TargetScaler::destandardize` keyed on the production
/// `regression_form`) and scores it through the production `TrainingLoop::compute_loss`
/// → `regression_loss` → the configured [`RegressionLoss`] dispatch.
///
/// The property (per head): a high-offset / low-variance target (calendar years,
/// μ≈2017, σ≈2) is FIT — the served (de-standardised) mean lands within the
/// context oracle's bar (`|mean − 2017| < 50`) and, for the Gaussian head, the
/// served σ — read through the REAL adapter serve path (the σ_y multiply), not the
/// raw z-σ — is off the floor and recovers σ_y exactly. The companion in-context
/// oracle that
/// proves the SAME contract on the other offset-bearing dispatch surface lives in
/// [`crate::pipeline::context_predictor`]'s
/// `gaussian_in_context_head_fits_high_offset_low_variance_target` (it depends on
/// pipeline-private episode/predictor machinery unreachable from this module, so
/// it stays where its dependencies live; both surfaces are pinned, together).
///
/// Heads 1–5 (CoSENT / MNRL / Triplet / Classification-CE / NER-CE) are
/// offset-INVARIANT by construction (cosine / softmax / class-index), carry no
/// `TargetScaler`, and are deliberately excluded from
/// [`super::target::StandardizableHead`] — so this oracle asserts nothing for them.
#[cfg(test)]
mod standardization_contract {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::super::data::TrainingBatch;
    use super::super::lora::build_distribution_head;
    use super::super::regression_loss::TargetScaler;
    use super::super::target::TrainingTarget;
    use super::super::{FineTuneConfig, RegressionLoss};
    use super::{TrainingLoop, TrainingLoopBuilder};
    use crate::fine_tune::adamw::{AdamW, ParamsAdamW};

    const HIDDEN: usize = 8;
    /// Calendar years — the textbook high-offset, low-variance regression target
    /// (μ ≈ 2017, σ ≈ 2). Re-used by both arms.
    const YEARS: [f32; 9] = [
        2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
    ];
    const TRUE_MEAN: f32 = 2017.0;

    /// A HIGH-variance, wide-range target (μ ≈ 34.7, σ_y ≈ 19.2) — an
    /// arxiv-citation-count / wide-outcome analogue. This is the discriminating
    /// scale the σ ≈ 2 `YEARS` fixture never exercised: in RAW-space training the
    /// Gaussian NLL `(y−μ)²/σ²` is O(σ_y²/σ_init²) ≈ O(hundreds) on step 0, past
    /// the divergence guard; in z-space the loss is O(1) for every objective.
    const WIDE: [f32; 9] = [6.0, 13.0, 20.0, 27.0, 34.0, 41.0, 48.0, 55.0, 68.0];

    /// Build a real production [`TrainingLoop`] over a regression
    /// [`TrainingTarget::ProjectionHead`] (projection + distribution head of the
    /// given width), with its [`TargetScaler`] reduced from `targets` exactly as
    /// `TrainingLoop::run` does. The infra (catalog/job/worker/artifact_dir) is the
    /// production builder's required plumbing — the dispatch we exercise
    /// (`regress` / `compute_loss`) never touches it, but we go through the real
    /// constructor so nothing about the head/scaler wiring is synthetic.
    async fn regression_loop(
        config: FineTuneConfig,
        head_width: usize,
        targets: &Tensor,
        device: &Device,
    ) -> (TrainingLoop, VarMap) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_distribution_head(HIDDEN, head_width, &config, &varmap, &vb).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // Leak the tempdir so the artifact path outlives the loop without a drop
        // race; a unit test process is short-lived, so this is contained.
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "oracle-model",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: crate::model::ModelTask::Regression,
                base_model_id: None,
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: "oracle-job",
                base_model_id: "oracle-model::1",
                training_source: "src",
                loss_type: "regression",
                hyperparams: "{}",
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .unwrap();
        catalog
            .claim_next_training_job("oracle-worker", std::time::Duration::from_secs(60))
            .await
            .unwrap()
            .expect("queued job is claimable");

        let mut loop_ = TrainingLoopBuilder::new(
            TrainingTarget::ProjectionHead { head },
            varmap.clone(),
            config,
        )
        .device(device.clone())
        .job_id("oracle-job".into())
        .worker_id("oracle-worker".into())
        .catalog(catalog)
        .artifact_dir(dir_path)
        .build()
        .unwrap();

        // Reduce the scaler from the targets exactly as `run` does before the loop,
        // so the head's z-scored target and the serve de-standardise share μ_y,σ_y.
        loop_.target_scaler = Some(TargetScaler::from_targets(targets).unwrap());
        (loop_, varmap)
    }

    /// Z-score a raw target tensor with the loop's scaler — the exact transform
    /// `embed_chunk` applies before the loss in production.
    fn z_score_targets(loop_: &TrainingLoop, targets: &Tensor) -> Tensor {
        let scaler = loop_.target_scaler.as_ref().unwrap();
        let raw = targets.to_vec1::<f32>().unwrap();
        let z: Vec<f32> = raw
            .iter()
            .map(|&y| scaler.standardize_value(y as f64) as f32)
            .collect();
        Tensor::from_vec(z, (raw.len(),), targets.device()).unwrap()
    }

    /// De-standardise a trained z-space head exactly as the serve path does: the
    /// backend's `TargetScaler::destandardize` (mean / quantile affine, raw σ
    /// passthrough) followed by the inference adapter's post-softplus σ_y scaling
    /// on a Gaussian head. Returns `(served_means, served_sigmas_or_quantiles)`
    /// per row: for Gaussian, the second vec is the served σ (σ_y·softplus(raw));
    /// for quantile, the first vec is unused and the second is the sorted served
    /// quantiles for row 0.
    fn serve_through_production(loop_: &TrainingLoop, z_head: &Tensor) -> Vec<Vec<f32>> {
        use crate::inference::adapter::{
            BackendOutput, DistributionAdapter, DistributionForm, OutputAdapter,
        };
        let scaler = loop_.target_scaler.as_ref().unwrap();
        let form = loop_.regression_form();
        // Backend de-standardise: mean/quantile affine; raw σ passthrough.
        let raw = scaler.destandardize(z_head, &form).unwrap();
        let rows = raw.to_vec2::<f32>().unwrap();
        let n = rows.len();
        let width = rows.first().map_or(0, Vec::len);
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let output = BackendOutput {
            float_outputs: vec![flat],
            string_outputs: vec![],
            row_status: vec![true; n],
            row_errors: vec![String::new(); n],
            shapes: vec![(n, width)],
        };
        let adapter: Box<dyn OutputAdapter> = match &form {
            DistributionForm::Gaussian => {
                Box::new(DistributionAdapter::gaussian_scaled(scaler.std() as f32))
            }
            DistributionForm::Quantile { levels } => {
                Box::new(DistributionAdapter::quantile(levels.clone()).unwrap())
            }
        };
        let cols = adapter.adapt(&output, n).unwrap();
        use arrow::array::{Array, Float32Array};
        cols.iter()
            .map(|c| {
                let a = c.as_any().downcast_ref::<Float32Array>().unwrap();
                (0..a.len()).map(|i| a.value(i)).collect::<Vec<f32>>()
            })
            .collect()
    }

    /// Serve a trained Gaussian z-head through the REAL production adapter
    /// (`gaussian_scaled(σ_y)`) and ALSO compute an INDEPENDENT reference σ_z —
    /// `softplus(raw_std)` read straight off the raw head column, NOT through the
    /// production σ helper — and return `(σ_z_reference, σ_raw_served)` per row.
    /// The ratio `σ_raw/σ_z` must be exactly σ_y for every row: this is the per-row
    /// identity the σ-axis calibration falsifier pins. Computing the σ_z reference
    /// independently (raw softplus, no `destandardize_sigma`) is load-bearing — a
    /// *multiplicative* bug in the production helper (missing, doubled, or wrong
    /// factor) would cancel out of a ratio of two helper outputs, so the reference
    /// must bypass the helper to expose it.
    fn serve_unscaled_and_scaled(loop_: &TrainingLoop, z_head: &Tensor) -> Vec<(f32, f32)> {
        use crate::inference::adapter::{BackendOutput, DistributionAdapter, OutputAdapter};
        use arrow::array::{Array, Float32Array};
        let scaler = loop_.target_scaler.as_ref().unwrap();
        let raw = scaler
            .destandardize(z_head, &loop_.regression_form())
            .unwrap();
        let rows = raw.to_vec2::<f32>().unwrap();
        // Independent σ_z reference: softplus(raw_std) per row, computed here from
        // the raw head column with the test-only softplus, NOT via the production
        // adapter, so a multiplicative bug in the σ helper cannot hide in the ratio.
        let sigma_z_ref: Vec<f32> = rows
            .iter()
            .map(|r| super::super::regression_loss::softplus_std_for_test(r[1] as f64) as f32)
            .collect();
        let n = rows.len();
        let width = rows.first().map_or(0, Vec::len);
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let output = BackendOutput {
            float_outputs: vec![flat],
            string_outputs: vec![],
            row_status: vec![true; n],
            row_errors: vec![String::new(); n],
            shapes: vec![(n, width)],
        };
        // Production serve path: the σ_y-scaled adapter (the number serving emits).
        let cols = DistributionAdapter::gaussian_scaled(scaler.std() as f32)
            .adapt(&output, n)
            .unwrap();
        let served = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();
        (0..n).map(|i| (sigma_z_ref[i], served.value(i))).collect()
    }

    /// Train the production regression dispatch for `steps` AdamW steps on a fixed
    /// batch of `(features, targets)` and return the trained head's RAW z-space
    /// `(batch, k)` output at the fitted parameters. Each step runs the PRODUCTION
    /// `TrainingLoop::head_forward` (the raw z-head, no de-standardise) against a
    /// z-scored target and `TrainingLoop::compute_loss` (→ `regression_loss` →
    /// configured `RegressionLoss`) — the exact chain `db.fine_tune(task=regression)`
    /// runs, scoring O(1) z-residuals. Use [`serve_through_production`] to recover
    /// the served raw distribution from the returned z-head.
    fn train_through_production_dispatch(
        loop_: &TrainingLoop,
        varmap: &VarMap,
        features: &Tensor,
        targets: &Tensor,
        steps: usize,
    ) -> Tensor {
        let z_target = z_score_targets(loop_, targets);
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..steps {
            // PRODUCTION head forward: projection + distribution head, RAW z-output.
            let head_out = loop_.head_forward(features).unwrap();
            let batch = TrainingBatch::Regression {
                input: head_out,
                target: z_target.clone(),
            };
            // PRODUCTION loss dispatch: compute_loss → regression_loss → the
            // configured RegressionLoss arm, scoring z-head vs z-target.
            let loss = loop_.compute_loss(&batch).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        loop_.head_forward(features).unwrap()
    }

    /// Like [`train_through_production_dispatch`], but ALSO drives every step's
    /// loss through the production divergence guard ([`TrainingLoop::process_batch_loss`]'s
    /// `>100`/NaN check, reproduced here as a per-step assertion) and records the
    /// max loss seen. Returns `(trained_z_head, max_loss)`. The guard is the exact
    /// behavioural contract this PR fixes: in z-space every objective stays O(1),
    /// so no step exceeds 100 — RED on current main for GaussianNll/BetaNll on the
    /// WIDE target, GREEN after the z-space loss.
    fn train_tracking_loss(
        loop_: &TrainingLoop,
        varmap: &VarMap,
        features: &Tensor,
        targets: &Tensor,
        steps: usize,
    ) -> (Tensor, f64) {
        let z_target = z_score_targets(loop_, targets);
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        let mut max_loss = 0.0_f64;
        let mut consecutive_diverged = 0u32;
        for step in 0..steps {
            let head_out = loop_.head_forward(features).unwrap();
            let batch = TrainingBatch::Regression {
                input: head_out,
                target: z_target.clone(),
            };
            let loss = loop_.compute_loss(&batch).unwrap();
            let loss_val = loss
                .to_dtype(DType::F32)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            // Reproduce the production divergence guard (trainer.rs `process_batch_loss`):
            // a NaN or `>100` loss is a divergence; three consecutive trips abort.
            if loss_val.is_nan() || loss_val > 100.0 {
                consecutive_diverged += 1;
                assert!(
                    consecutive_diverged < 3,
                    "z-space loss diverged (NaN or >100 for 3 consecutive steps) at step {step}: \
                     loss {loss_val}. The z-space loss must keep every objective O(1) on a \
                     σ_y≈19 target — this is the production divergence guard the PR fixes."
                );
            } else {
                consecutive_diverged = 0;
            }
            max_loss = max_loss.max(loss_val);
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        (loop_.head_forward(features).unwrap(), max_loss)
    }

    /// Deterministic small feature matrix `(n, HIDDEN)` — the projected embeddings
    /// the regression head sits on. Values are O(1) so the projection (identity
    /// base + zero-init LoRA) feeds the distribution head a bounded signal.
    fn features(n: usize, device: &Device) -> Tensor {
        let mut vals = Vec::with_capacity(n * HIDDEN);
        let mut s: u64 = 0x1234_5678;
        for _ in 0..n * HIDDEN {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0;
            vals.push(u * 0.5);
        }
        Tensor::from_vec(vals, (n, HIDDEN), device).unwrap()
    }

    /// ORACLE (head 6 — production fine-tune Gaussian regression): the
    /// β-NLL-trained parametric Gaussian head FITS a high-offset / low-variance
    /// calendar-year target through the real production dispatch. The served
    /// (de-standardised) mean lands within ±50 of μ_y ≈ 2017 and the served σ is
    /// off the floor (> 0.1). A head that did NOT de-standardise would crawl
    /// ≈ lr·steps units from zero-init and stall thousands short of 2017 (Adam's
    /// per-step move is ≈ lr·sign(grad), scale-independent) — that is the failure
    /// this guards on the production path, not the synthetic VarMap head.
    #[tokio::test(flavor = "multi_thread")]
    async fn ft_gaussian_head_fits_high_offset_through_production_dispatch() {
        let device = Device::Cpu;
        let n = YEARS.len();
        let targets = Tensor::from_vec(YEARS.to_vec(), (n,), &device).unwrap();
        let config = FineTuneConfig {
            // β-NLL with β=0.5 is the engine's default regression objective.
            regression_loss: Some(RegressionLoss::BetaNll { beta: 0.5 }),
            ..Default::default()
        };
        let (loop_, varmap) = regression_loop(config, 2, &targets, &device).await;
        let feats = features(n, &device);

        let z_head = train_through_production_dispatch(&loop_, &varmap, &feats, &targets, 1500);
        // Serve exactly as production: backend de-standardise (mean affine) +
        // adapter σ_y·softplus(raw). cols[0] = served means, cols[1] = served σ.
        let cols = serve_through_production(&loop_, &z_head);
        let served_mean = cols[0][0];
        let served_sigma = cols[1][0];

        assert!(
            (served_mean - TRUE_MEAN).abs() < 50.0,
            "production ft Gaussian head failed to fit the high-offset target: \
             true mean {TRUE_MEAN}, served mean {served_mean} (off by {:.0}). \
             The served de-standardisation must converge the served mean to μ_y \
             under the z-space loss.",
            (served_mean - TRUE_MEAN).abs()
        );
        // Read σ through the REAL serve path (the adapter's σ_y multiply), not off
        // the raw head (which would be σ_z ≈ 1, blind to the multiply). The served
        // σ must (a) be off the floor and (b) recover σ_y EXACTLY: σ_raw/σ_z = σ_y
        // per row, the same multiplicative identity the high-variance oracle pins.
        // Reading `gaussian_params(&row0)` here would silently stop testing the
        // served σ post-z-space (it returns σ_z, and σ_z > 0.1 trivially) — this
        // routes through the adapter so the assertion tests the number serving emits.
        assert!(
            served_sigma > 0.1,
            "production ft Gaussian head served a collapsed σ {served_sigma} \
             (a real σ_y-scaled σ must be well off the floor)"
        );
        let sigma_y = {
            let s = loop_.target_scaler.as_ref().unwrap();
            s.std() as f32
        };
        for (row, (sigma_z, sigma_raw)) in serve_unscaled_and_scaled(&loop_, &z_head)
            .iter()
            .enumerate()
        {
            let ratio = sigma_raw / sigma_z;
            assert!(
                (ratio - sigma_y).abs() <= 1e-3 * sigma_y,
                "row {row} served σ {sigma_raw} / σ_z {sigma_z} = {ratio} must equal \
                 σ_y={sigma_y} — the served σ is read through the real adapter (σ_y \
                 multiply), not the raw z-σ; a missing multiply leaves the ratio at 1"
            );
        }
    }

    /// ORACLE (head 7 — production fine-tune quantile regression): the
    /// pinball-trained quantile head FITS the same high-offset calendar-year
    /// target through the real production dispatch. Every served (de-standardised)
    /// quantile column lands within ±50 of μ_y ≈ 2017 (the levels straddle a
    /// σ≈2.6 spread, so all sit within a couple of units of 2017), and the columns
    /// are ordered (non-crossing). A quantile head whose columns were NOT all
    /// de-standardised would leave the upper levels stranded near 0.
    #[tokio::test(flavor = "multi_thread")]
    async fn ft_quantile_head_fits_high_offset_through_production_dispatch() {
        let device = Device::Cpu;
        let n = YEARS.len();
        let targets = Tensor::from_vec(YEARS.to_vec(), (n,), &device).unwrap();
        let levels = vec![0.1, 0.5, 0.9];
        let config = FineTuneConfig {
            regression_loss: Some(RegressionLoss::Pinball),
            quantile_levels: levels.clone(),
            ..Default::default()
        };
        let (loop_, varmap) = regression_loop(config, levels.len(), &targets, &device).await;
        let feats = features(n, &device);

        let z_head = train_through_production_dispatch(&loop_, &varmap, &feats, &targets, 1500);
        // Serve exactly as production: every quantile column is de-standardised by
        // the backend affine, then the adapter sorts per row. `serve_through_production`
        // returns one served column per level; read row 0 across the columns.
        let cols = serve_through_production(&loop_, &z_head);
        let row0: Vec<f32> = cols.iter().map(|c| c[0]).collect();
        assert_eq!(
            row0.len(),
            levels.len(),
            "served quantile head keeps all levels"
        );

        for (i, &q) in row0.iter().enumerate() {
            assert!(
                (q - TRUE_MEAN).abs() < 50.0,
                "production ft quantile column {i} (level {}) failed to fit the \
                 high-offset target: μ_y ≈ {TRUE_MEAN}, served {q} (off by {:.0}). \
                 A column left un-de-standardised would strand near 0.",
                levels[i],
                (q - TRUE_MEAN).abs()
            );
        }
        // Non-crossing: the pinball + non-crossing penalty keeps the served levels
        // ordered, and de-standardisation is a monotone affine so order survives.
        assert!(
            row0[0] <= row0[1] && row0[1] <= row0[2],
            "served quantiles must be non-crossing after de-standardisation: {row0:?}"
        );
    }

    // ─── W5-PR5 high-variance oracle (the scale-robustness deliverable) ───────
    //
    // The σ ≈ 2 `YEARS` oracles above never exercised the divergence the raw-space
    // loss has on a realistic-variance target. These run the SAME production
    // dispatch on the WIDE target (σ_y ≈ 19) and assert, for ALL FOUR objectives:
    // (1) CONVERGES — no divergence-guard trip (RED on current main for
    //     GaussianNll/BetaNll, GREEN for Crps/Pinball; GREEN for all four after
    //     the z-space loss);
    // (2) the served POINT estimate FITS the target (mean / quantile median);
    // (3) for Gaussian, the served σ recovers σ_y EXACTLY (σ_raw/σ_z = σ_y per row,
    //     against an independent σ_z reference) — the calibration assertion that
    //     catches a missing OR mis-scaled post-softplus σ_y multiply;
    // plus a destructive NON-VACUITY guard (untrained head → no served spread) and
    // a raw-vs-z served-PRESERVATION check (within a justified tolerance, since the
    // non-scale-free AdamW perturbs the trajectory) for the scale-equivariant
    // objectives.

    /// σ_y of the WIDE target — the σ-scale the served Gaussian σ must recover.
    fn wide_sigma_y() -> f32 {
        let device = Device::Cpu;
        let t = Tensor::from_vec(WIDE.to_vec(), (WIDE.len(),), &device).unwrap();
        TargetScaler::from_targets(&t).unwrap().std() as f32
    }

    fn wide_mean() -> f32 {
        WIDE.iter().sum::<f32>() / WIDE.len() as f32
    }

    /// ORACLE (W5-PR5 — Gaussian-family scale-robustness): each of the three
    /// Gaussian-form objectives (`GaussianNll`, `BetaNll{0.5}` the default, `Crps`)
    /// trains the production dispatch on the σ_y≈19 WIDE target WITHOUT tripping the
    /// divergence guard, the served mean FITS μ_y, and the served σ is recovered to
    /// the right ORDER (~σ_y, not ~σ_z≈1 — the missing-σ_y-multiply calibration bug).
    ///
    /// RED on current main: GaussianNll/BetaNll trip the `>100` guard within the
    /// first steps (raw `(y−μ)²/σ²` ≈ σ_y²/σ_init² ≈ 800), while Crps converges in
    /// raw too — that asymmetry is the bug fingerprint. GREEN for all three here.
    #[tokio::test(flavor = "multi_thread")]
    async fn ft_gaussian_family_scale_robust_on_high_variance_target() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let sigma_y = wide_sigma_y();
        let mu_y = wide_mean();

        for loss in [
            RegressionLoss::GaussianNll,
            RegressionLoss::BetaNll { beta: 0.5 },
            RegressionLoss::Crps,
        ] {
            let config = FineTuneConfig {
                regression_loss: Some(loss),
                ..Default::default()
            };
            let (loop_, varmap) = regression_loop(config, 2, &targets, &device).await;
            let feats = features(n, &device);

            // (1) Convergence: every step's loss is finite and never trips the
            //     divergence guard. This is RED on raw-space NLL/BetaNll.
            let (z_head, max_loss) = train_tracking_loss(&loop_, &varmap, &feats, &targets, 1500);
            assert!(
                max_loss.is_finite() && max_loss < 100.0,
                "{loss:?}: z-space loss must stay below the divergence guard on a \
                 σ_y≈{sigma_y} target (max loss {max_loss})"
            );

            // (2) Served mean fits μ_y within one spread.
            let cols = serve_through_production(&loop_, &z_head);
            let served_mean = cols[0][0];
            assert!(
                (served_mean - mu_y).abs() < sigma_y,
                "{loss:?}: served mean {served_mean} must fit μ_y≈{mu_y} within one \
                 spread (σ_y≈{sigma_y})"
            );

            // (3) THE σ-AXIS CALIBRATION FALSIFIER. The bug this PR guards is a
            // *multiplicative* error on the served σ: a missing σ_y multiply serves
            // σ_z (≈ σ_y× too tight), a wrong factor serves k·σ_y. A loose order
            // band (σ_y/3 < σ < 3σ_y) would pass a 2×-miscalibrated fit, so instead
            // we pin the multiply EXACTLY: for every row the production-served σ
            // divided by an INDEPENDENT σ_z reference (softplus(raw_std) read
            // straight off the raw head, NOT through the production σ helper) must
            // equal σ_y — the σ_y factor and nothing else. The independent reference
            // is load-bearing: a multiplicative bug in the helper would cancel out of
            // a ratio of two helper outputs. This catches a missing multiply (ratio
            // 1 ≠ σ_y), a doubled multiply (ratio 2σ_y), or a softplus-inside
            // mis-placement (ratio drifts per row). It is the tight, per-row identity
            // that the loose order band approximated; both falsifiers demonstrated
            // RED (ratio 1 and ratio 2σ_y) by neutralizing/doubling `destandardize_sigma`.
            let scaled = serve_unscaled_and_scaled(&loop_, &z_head);
            for (row, (sigma_z, sigma_raw)) in scaled.iter().enumerate() {
                // σ_z is post-softplus, floored ≥ STD_FLOOR, so the ratio is well
                // defined; the raw σ is re-floored, so compare only where the floor
                // did not bind (σ_y·σ_z ≫ STD_FLOOR holds for every row here).
                let ratio = sigma_raw / sigma_z;
                assert!(
                    (ratio - sigma_y).abs() <= 1e-3 * sigma_y,
                    "{loss:?}: row {row} served σ {sigma_raw} / σ_z {sigma_z} = {ratio} \
                     must equal σ_y={sigma_y} EXACTLY (the one multiplicative factor). \
                     A missing post-softplus σ_y multiply leaves the ratio at 1 \
                     (σ_z≈1, ~σ_y× too tight) — the silent under-dispersion bug."
                );
            }
            // And the served σ is genuinely σ_y-scaled in absolute terms (not a
            // collapsed-to-floor degenerate that would make the ratio vacuous): the
            // high-residual row is on the ORDER of σ_y, not σ_z≈1.
            let max_sigma = scaled.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
            assert!(
                max_sigma > sigma_y / 3.0,
                "{loss:?}: the head's largest served σ {max_sigma} must be σ_y-scaled \
                 (≈{sigma_y}), not the z-scale σ_z≈1 — a missing σ_y multiply is the bug"
            );
        }
    }

    // ─── esc-035: distributional K3 standardization oracles ───────────────────
    //
    // `config.seed` reaches ONLY the LoRA-A Kaiming draw (lora.rs:56 ->
    // lora_linear.rs:131); `features()` above is a fixed LCG. So sweeping the
    // seed isolates the A-draw and diverges the two oracles below through
    // gradient dynamics alone — the population these oracles must speak for.
    // The pinned 12-seed set below is the escape's evidence set; the default
    // seed (42, `DEFAULT_FINE_TUNE_SEED`) is the ONE trajectory the pre-rewrite
    // single-seed assertions happened to pass on.
    const PINNED_SEEDS: [u64; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    /// Per-seed measurement for the Pinball scale-robustness oracle.
    #[derive(Debug, Clone, Copy)]
    struct QuantileSeedStats {
        seed: u64,
        max_loss: f64,
        diverged: bool,
        q10: f32,
        q50: f32,
        q90: f32,
    }

    /// Why a seed's quantile measurement failed the checker — named so a
    /// rejected seed points at the specific broken bound, not a bare
    /// `assertion failed`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum QuantileFailure {
        NonFinite,
        Diverged,
        FitOutOfRange,
        MedianOutOfRange,
        Crossing,
    }

    /// THE quantile-oracle judgment, factored out of the training loop into a
    /// pure function of measured stats. This is what lets the real production
    /// run and the permanent mutant controls below share one definition of
    /// "the oracle says no" (required sequence step 2/3). Non-finite counts as
    /// failing (esc-005 class): every field is checked with `is_finite()`
    /// before any numeric comparison, so `NaN < bound` (which is `false` and
    /// would otherwise vacuously read as "not out of range") cannot pass.
    fn check_quantile_seed(
        s: &QuantileSeedStats,
        mu_y: f32,
        sigma_y: f32,
    ) -> std::result::Result<(), QuantileFailure> {
        if !s.max_loss.is_finite() || !s.q10.is_finite() || !s.q50.is_finite() || !s.q90.is_finite()
        {
            return Err(QuantileFailure::NonFinite);
        }
        if s.diverged || s.max_loss >= 100.0 {
            return Err(QuantileFailure::Diverged);
        }
        for q in [s.q10, s.q50, s.q90] {
            if (q - mu_y).abs() >= 2.0 * sigma_y {
                return Err(QuantileFailure::FitOutOfRange);
            }
        }
        if (s.q50 - mu_y).abs() >= sigma_y {
            return Err(QuantileFailure::MedianOutOfRange);
        }
        if !(s.q10 <= s.q50 && s.q50 <= s.q90) {
            return Err(QuantileFailure::Crossing);
        }
        Ok(())
    }

    /// Train + serve one seed of the Pinball/WIDE scenario through the REAL
    /// production dispatch: `TrainingLoop::head_forward` -> `compute_loss` ->
    /// AdamW step, reproducing the production divergence guard
    /// (`process_batch_loss`'s NaN/`>100` trip) as a non-panicking flag so a
    /// diverged seed can be COUNTED rather than aborting the sweep (required
    /// sequence step 2's non-finite/diverged-counts-as-failing rule). This is a
    /// new test-side helper — `train_tracking_loss` above stays untouched
    /// (other in-scope-adjacent tests call it and must keep panicking on
    /// divergence for their own single-run assertions).
    async fn measure_quantile_seed(
        seed: u64,
        targets: &Tensor,
        feats: &Tensor,
        levels: &[f64],
        device: &Device,
    ) -> QuantileSeedStats {
        let config = FineTuneConfig {
            regression_loss: Some(RegressionLoss::Pinball),
            quantile_levels: levels.to_vec(),
            seed,
            ..Default::default()
        };
        let (loop_, varmap) = regression_loop(config, levels.len(), targets, device).await;
        let z_target = z_score_targets(&loop_, targets);
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        let mut max_loss = 0.0_f64;
        let mut consecutive = 0u32;
        let mut diverged = false;
        for _ in 0..1500 {
            let head_out = loop_.head_forward(feats).unwrap();
            let batch = TrainingBatch::Regression {
                input: head_out,
                target: z_target.clone(),
            };
            let loss = loop_.compute_loss(&batch).unwrap();
            let loss_val = loss
                .to_dtype(DType::F32)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            if loss_val.is_nan() || loss_val > 100.0 {
                consecutive += 1;
                if consecutive >= 3 {
                    diverged = true;
                }
            } else {
                consecutive = 0;
            }
            max_loss = max_loss.max(loss_val);
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        let z_head = loop_.head_forward(feats).unwrap();
        // `serve_through_production` panics (`.unwrap()` on the quantile
        // adapter's `adapt`, which errors on a non-finite row — trainer.rs
        // `serve_through_production` -> distribution.rs's non-crossing guard)
        // if the head itself is non-finite. A diverged seed must be COUNTED,
        // not crash the sweep (required sequence step 2), so check finiteness
        // on the raw head BEFORE calling the panicking helper and short-circuit
        // to a non-finite sentinel the checker's `is_finite()` gate catches.
        let z_head_vals = z_head.to_vec2::<f32>().unwrap();
        if z_head_vals.iter().flatten().any(|v| !v.is_finite()) {
            return QuantileSeedStats {
                seed,
                max_loss,
                diverged: true,
                q10: f32::NAN,
                q50: f32::NAN,
                q90: f32::NAN,
            };
        }
        let cols = serve_through_production(&loop_, &z_head);
        let row0: Vec<f32> = cols.iter().map(|c| c[0]).collect();
        QuantileSeedStats {
            seed,
            max_loss,
            diverged,
            q10: row0[0],
            q50: row0[1],
            q90: row0[2],
        }
    }

    /// ORACLE (W5-PR5 — quantile (Pinball) scale-robustness), esc-035
    /// distributional rewrite: the pinball head trains the production dispatch
    /// on the WIDE target without diverging, every served quantile lands within
    /// a spread of μ_y, the median tracks μ_y, and the served columns are
    /// non-crossing after de-standardisation — asserted as a POPULATION claim
    /// over the pinned 12-seed set (the sole randomness source, the LoRA-A
    /// Kaiming draw), not one default-seed trajectory.
    ///
    /// MEASURED pre-rewrite (esc-035 step 1, unmodified assertions, same
    /// pinned-seed sweep — table in this commit's message): 10/12 pinned seeds
    /// pass; seeds 6 and 7 miss (seed 6 on the median bound, seed 7 on both the
    /// fit and median bounds). The count-based bar below (>=9/12) keeps one
    /// seed of headroom under that measured 10/12 for platform floating-point
    /// variance while asserting a strong-majority population claim, not a
    /// vacuous one; per K3, robustness comes from aggregating over seeds, not
    /// from widening the per-seed bounds (already the original, un-loosened
    /// order-of-magnitude bounds: within one/two σ_y of μ_y).
    #[tokio::test(flavor = "multi_thread")]
    async fn ft_quantile_scale_robust_on_high_variance_target() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let levels = vec![0.1, 0.5, 0.9];
        let sigma_y = wide_sigma_y();
        let mu_y = wide_mean();
        let feats = features(n, &device);

        let mut passed = 0usize;
        let mut failures: Vec<(u64, QuantileFailure)> = Vec::new();
        for seed in PINNED_SEEDS {
            let stats = measure_quantile_seed(seed, &targets, &feats, &levels, &device).await;
            match check_quantile_seed(&stats, mu_y, sigma_y) {
                Ok(()) => passed += 1,
                Err(reason) => failures.push((stats.seed, reason)),
            }
        }
        assert!(
            passed >= 9,
            "Pinball scale-robustness must hold across a strong majority of the \
             pinned 12-seed set, not one lucky default-seed trajectory: {passed}/12 \
             pinned seeds passed the checker (need >=9); failures: {failures:?}"
        );
    }

    /// NON-VACUITY (W5-PR5 destructive guard): an UNTRAINED Gaussian head (zero
    /// steps) serves the constant μ_y for EVERY row — zero spread across rows — so
    /// it FAILS a learning bar that a trained head passes. Mirrors the PR4
    /// μ-collapse guard: the fit assertions above would be vacuous against a head
    /// that emits μ_y for all inputs, so this proves the served means actually move
    /// with the input only after training. The trained head separates the rows.
    #[tokio::test(flavor = "multi_thread")]
    async fn untrained_gaussian_head_has_no_served_spread_trained_does() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let config = FineTuneConfig {
            regression_loss: Some(RegressionLoss::Crps),
            ..Default::default()
        };
        let feats = features(n, &device);

        // Untrained: serve the zero-init head directly (no steps). z = 0 → served
        // μ_y for every row → zero spread of served means.
        let (loop0, _vm0) = regression_loop(config.clone(), 2, &targets, &device).await;
        let z_head0 = loop0.head_forward(&feats).unwrap();
        let served0 = serve_through_production(&loop0, &z_head0);
        let spread0 = spread(&served0[0]);
        assert!(
            spread0 < 1e-3,
            "untrained head must emit the constant μ_y for every row (≈0 spread), \
             got spread {spread0} — the fit bar would be vacuous otherwise"
        );

        // Trained: the served means now separate the rows (non-trivial spread).
        let (loop1, vm1) = regression_loop(config, 2, &targets, &device).await;
        let z_head1 = train_through_production_dispatch(&loop1, &vm1, &feats, &targets, 1500);
        let served1 = serve_through_production(&loop1, &z_head1);
        let spread1 = spread(&served1[0]);
        assert!(
            spread1 > 1.0,
            "a trained head must SEPARATE the rows (served-mean spread {spread1} ≫ \
             the untrained {spread0}) — proving it learned input→target, not μ-regurgitation"
        );
    }

    /// max − min of a served column — the row-to-row spread the non-vacuity guard
    /// reads (an untrained head emits the constant μ_y → spread ≈ 0).
    fn spread(col: &[f32]) -> f32 {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in col {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    }

    /// THE BUG FINGERPRINT (W5-PR5 non-vacuity): the RAW-space loss DIVERGES on the
    /// high-variance target — exactly the failure the z-space loss fixes. This
    /// reconstructs the pre-PR5 flow (de-standardise the head BEFORE the loss, score
    /// against the RAW target) on the WIDE σ_y≈19 target and asserts GaussianNll
    /// trips the `>100` divergence threshold within the first few steps, while Crps
    /// (bounded ≈σ) stays finite. That asymmetry — NLL diverges, Crps does not — is
    /// the precise bug the z-space loss removes; the `*_scale_robust` oracles above
    /// prove ALL FOUR converge in z-space, so this guards that the fix is load-bearing.
    #[tokio::test(flavor = "multi_thread")]
    async fn raw_space_gaussian_nll_diverges_on_high_variance_target() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);

        // RAW-space reference dispatch (pre-PR5): head_forward → destandardize →
        // loss against the RAW target — exactly what `regress` used to feed the
        // loss. Each step's loss is run through the SAME guard predicate the
        // production `process_batch_loss` uses (`is_nan() || > 100.0` with a
        // 3-consecutive abort), so `diverged[i]` is true iff the production guard
        // would have RETURNED the divergence error — the RED is the real guard
        // verdict, not just a raw loss-magnitude assertion.
        let mut max_loss = [0.0_f64; 2];
        let mut diverged = [false; 2];
        for (i, loss) in [RegressionLoss::GaussianNll, RegressionLoss::Crps]
            .into_iter()
            .enumerate()
        {
            let config = FineTuneConfig {
                regression_loss: Some(loss),
                ..Default::default()
            };
            let (loop_, varmap) = regression_loop(config, 2, &targets, &device).await;
            let scaler = *loop_.target_scaler.as_ref().unwrap();
            let mut opt = AdamW::new(
                varmap.all_vars(),
                ParamsAdamW {
                    lr: 0.05,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut consecutive = 0u32;
            for _ in 0..10 {
                let z_out = loop_.head_forward(&feats).unwrap();
                let raw_head = scaler
                    .destandardize(&z_out, &loop_.regression_form())
                    .unwrap();
                let batch = TrainingBatch::Regression {
                    input: raw_head,
                    target: targets.clone(),
                };
                let loss_t = loop_.compute_loss(&batch).unwrap();
                let lv = loss_t
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap() as f64;
                max_loss[i] = max_loss[i].max(lv);
                // The production guard predicate, reproduced exactly.
                if lv.is_nan() || lv > 100.0 {
                    consecutive += 1;
                    if consecutive >= 3 {
                        diverged[i] = true;
                    }
                } else {
                    consecutive = 0;
                }
                let grads = loss_t.backward().unwrap();
                opt.step(&grads).unwrap();
            }
        }

        assert!(
            diverged[0] && max_loss[0] > 100.0,
            "raw-space GaussianNll must DIVERGE on a σ_y≈19 target (the bug): the \
             production guard predicate fired ≥3 consecutive (diverged={}), max loss \
             {} > 100 — z-space is what fixes this",
            diverged[0],
            max_loss[0]
        );
        assert!(
            !diverged[1] && max_loss[1] < 100.0,
            "raw-space Crps stays bounded (≈σ) even on σ_y≈19 (diverged={}, max loss \
             {}) — the NLL-diverges-Crps-does-not asymmetry is the bug fingerprint",
            diverged[1],
            max_loss[1]
        );
    }

    /// The Crps oracle's evaluated seed set (post-audit rework): the pinned
    /// 12 PLUS `DEFAULT_FINE_TUNE_SEED` (`jammi_wire::fine_tune`, wire's
    /// fine_tune.rs:362) — the trajectory every caller who does not pass a
    /// seed actually runs; the escape's whole framing is about that default
    /// trajectory. The quantile oracle above is UNCHANGED (still the 12-seed
    /// `PINNED_SEEDS` — an independent audit measured it clean and asked for
    /// it to be kept as-is: same bounds, same 12 seeds, same >=9 bar).
    const CRPS_SEEDS: [u64; 13] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        jammi_wire::fine_tune::DEFAULT_FINE_TUNE_SEED,
    ];

    /// Per-seed measurement for the Crps z-vs-raw preservation oracle. Two
    /// PAIRS of fields, feeding two INDEPENDENT arms of
    /// [`check_crps_aggregate`]:
    /// - `(sigma_ratio, mean_signed_diff)` feed the AGGREGATE arms
    ///   (trimmed-mean, robust to seed-to-seed AdamW noise, sensitive to a
    ///   uniform/systematic serve-time regression).
    /// - `(mean_abs_diff, sigma_abs_diff)` feed the PER-SEED CEILING arm —
    ///   021e48e's original quantities, RE-ADDED (second audit round, block
    ///   1): an aggregate-only checker is blind to SIGN-BALANCED per-seed
    ///   scatter (e.g. half the seeds +10 raw units, half −10 raw units of
    ///   served-mean drift) because a trimmed mean of SIGNED per-seed values
    ///   cancels across seeds even though every individual seed is badly
    ///   wrong — measured on this codebase's OWN mutant (i): its per-seed
    ///   mean diffs are sign-split and the aggregate trims to −0.266 (a PASS)
    ///   even though `mean_abs_diff` is large on every seed. `check_crps_aggregate`
    ///   requires evidence from BOTH kinds of arm to accept a sweep.
    #[derive(Debug, Clone, Copy)]
    struct CrpsSeedStats {
        seed: u64,
        /// σ_z_served / σ_r_served (row 0) — dimensionless; feeds the
        /// aggregate σ-ratio arm. Scoped claim (advisory, second audit
        /// round): a trimmed mean of ratios is exact-linear under a uniform
        /// MULTIPLICATIVE shift of every seed's ratio (the natural failure
        /// model for a scale parameter) — NOT under a uniform ADDITIVE
        /// raw-unit shift to `σ_z_served`, which produces a PER-SEED-VARYING
        /// ratio shift (`delta/σ_r_served`, and `σ_r_served` spans a
        /// measured ~5x range across seeds).
        sigma_ratio: f32,
        /// mean(z_row − r_row) over all rows — SIGNED; feeds the aggregate μ
        /// arm. A trimmed mean of these is exact-linear under a uniform
        /// ADDITIVE raw-unit shift (adding `delta` to every row of every
        /// seed shifts every seed's average, hence the trimmed mean, by
        /// exactly `delta`).
        mean_signed_diff: f32,
        /// max(|z_row − r_row|) over all rows — 021e48e's original per-seed
        /// μ quantity; feeds the per-seed ceiling arm.
        mean_abs_diff: f32,
        /// |σ_z_served − σ_r_served| — 021e48e's original per-seed σ
        /// quantity; feeds the per-seed ceiling arm.
        sigma_abs_diff: f32,
    }

    /// Why `check_crps_aggregate` rejected the sweep. ALL applicable
    /// violations are collected in one call (see [`check_crps_aggregate`]);
    /// only [`Self::NonFinite`] short-circuits, because every other
    /// statistic below is undefined over a non-finite input.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CrpsViolation {
        NonFinite,
        SigmaRatioOutOfRange,
        MeanDrift,
        /// Fewer than `CRPS_PER_SEED_BAR` of the swept seeds individually
        /// clear BOTH per-seed ceilings — the arm that makes SIGN-BALANCED
        /// per-seed scatter visible (see [`CrpsSeedStats`]'s doc).
        PerSeedCeilingViolated,
    }

    /// The `trim_frac`-trimmed mean of `values`: sorts a copy, drops the
    /// top/bottom `round(trim_frac·n)` entries, averages the rest. Exact
    /// scope of the "exact-linear" property (advisory, second audit round):
    /// a trimmed mean is exact-linear under a uniform shift APPLIED TO THE
    /// VALUES BEING AGGREGATED — additive for [`CrpsSeedStats::mean_signed_diff`],
    /// multiplicative for [`CrpsSeedStats::sigma_ratio`] (see each field's
    /// doc) — because such a shift preserves rank order, so the trimmed set
    /// is the SAME before and after. This is what makes a clean
    /// before/after detection-power comparison possible, and (at
    /// `trim_frac=0.15`, 2 of the 13 Crps seeds) discards the two measured
    /// ~15-raw-σ-unit AdamW-noise outlier seeds instead of averaging over
    /// them unweighted.
    fn trimmed_mean(values: &[f32], trim_frac: f32) -> f32 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let trim = ((v.len() as f32) * trim_frac).round() as usize;
        let keep = &v[trim..v.len() - trim];
        keep.iter().sum::<f32>() / keep.len() as f32
    }

    /// σ-axis AGGREGATE arm: `check_crps_aggregate`'s trimmed-mean
    /// sigma-ratio must land in this dimensionless band. Measured
    /// trimmed-mean baseline (correct code): 1.108; per-seed ratios range
    /// 0.66-1.60 (5/13 individual seeds sit below 1.0 — ordinary AdamW
    /// scatter, not a defect; the bound applies to the TRIMMED MEAN, not
    /// individual seeds). `LO=1.0` is an EMPIRICAL anchor, not a
    /// theoretical ideal: the working window is roughly 0.85-1.10 — mutant
    /// (i)'s trimmed-mean ratio (0.852, measured) bounds `LO` from below,
    /// the 1.108 baseline (plus jackknife noise, comparable order to the
    /// μ-axis jackknife below) bounds it from above. MEASURED detection
    /// crossover (full 3-arm checker, real value-level injection): under-
    /// dispersion (the sensitive direction, and the direction K3's actual
    /// bug class moves in) red at -1.8 raw units; over-dispersion (the
    /// opposite direction) red at +4.0 raw units.
    const CRPS_SIGMA_RATIO_LO: f32 = 1.0;
    /// Upper band edge, for the opposite-direction (over-dispersion, e.g. a
    /// doubled multiply) failure mode; kept for two-sided coverage since the
    /// trimmed-mean baseline (1.108) leaves headroom either way.
    const CRPS_SIGMA_RATIO_HI: f32 = 1.3;
    /// μ-axis AGGREGATE arm: `check_crps_aggregate`'s trimmed-mean SIGNED
    /// mean-diff must have `abs() < CRPS_MEAN_AGG_TOL`. Measured
    /// trimmed-mean baseline: 0.765 raw units. MEASURED leave-one-out
    /// jackknife (removing each of the 13 seeds in turn and recomputing the
    /// trimmed mean over the other 12): the largest shift from the
    /// full-sweep value is 0.128 raw units. `0.9` (headroom 0.135 above
    /// baseline) left the bound within ONE jackknife shift of the
    /// unregressed baseline — a coin flip against ordinary per-platform
    /// float/toolchain noise. Widened to a stated margin: baseline + 3×max
    /// jackknife shift ≈ 0.765 + 3·0.128 ≈ 1.15, rounded to `1.2` (headroom
    /// 0.435, >3x the measured jackknife noise). This arm no longer has to
    /// be the sole μ guard: `PerSeedCeilingViolated` below is the arm that
    /// still catches a per-seed-gross violation the widened aggregate bound
    /// tolerates — including the whole SIGN-BALANCED-scatter class this
    /// aggregate structurally cannot see regardless of where the bound
    /// sits (see [`CrpsSeedStats`]'s doc). MEASURED detection crossover
    /// (full 3-arm checker, real value-level injection), reported HONESTLY
    /// after widening for jackknife-robustness (not narrated to look better
    /// than measured): the sensitive direction (a positive/over-fit shift,
    /// no sign-cancellation "dip") is red at +0.5 raw units — this is
    /// arithmetically close to `CRPS_MEAN_AGG_TOL` minus the baseline and
    /// is NOT reported as a standalone virtue, since it is just the
    /// headroom; the worst-case direction (negative, the "dip" a
    /// SIGNED-but-not-scale-invariant statistic centred away from zero
    /// exhibits) is red at -2.0 raw units.
    const CRPS_MEAN_AGG_TOL: f32 = 1.2;
    /// PER-SEED CEILING arm (021e48e's original bound, re-added second audit
    /// round, block 1): a per-seed μ ceiling as a fraction of σ_y.
    const CRPS_PER_SEED_MEAN_CEILING_FRAC: f32 = 0.3;
    /// PER-SEED CEILING arm: a per-seed σ ceiling as a fraction of σ_y.
    const CRPS_PER_SEED_SIGMA_CEILING_FRAC: f32 = 0.5;
    /// PER-SEED CEILING arm: how many of the 13 `CRPS_SEEDS` must
    /// individually clear BOTH ceilings. MEASURED baseline (correct code):
    /// 10/13 pass both ceilings simultaneously (seeds 2 and 8 miss the σ
    /// ceiling — the two known ~15-raw-σ-unit AdamW-noise outliers; seed 10
    /// misses the μ ceiling). `9` keeps one seed of headroom under that
    /// measured 10/13, mirroring 021e48e's own margin philosophy.
    const CRPS_PER_SEED_BAR: usize = 9;

    /// THE Crps-oracle judgment (esc-035, second audit round): collects
    /// EVERY applicable violation from THREE independent arms — no early
    /// return once non-finiteness is ruled out, so hiding one arm's
    /// regression behind another firing first is impossible (measured: with
    /// an earlier single-`Result`, early-return design, setting
    /// `CRPS_MEAN_AGG_TOL` to a vacuous value left ALL FOUR tests in this
    /// module green, because the σ arm's early return hid the fact that the
    /// μ guard had become deletable — the same defect class the first audit
    /// round blocked on the calibration-ratio term, reappearing on this
    /// axis).
    ///
    /// The three arms:
    /// 1. AGGREGATE σ-ratio (trimmed mean in `[CRPS_SIGMA_RATIO_LO,
    ///    CRPS_SIGMA_RATIO_HI]`) — sensitive to a uniform MULTIPLICATIVE σ
    ///    regression (the K3 bug class here, missing/doubled σ_y multiply,
    ///    IS multiplicative), scale-invariant across the measured ~5x
    ///    per-seed σ_r range.
    /// 2. AGGREGATE μ signed diff (trimmed mean, `abs() < CRPS_MEAN_AGG_TOL`)
    ///    — sensitive to a uniform ADDITIVE μ regression.
    /// 3. PER-SEED CEILING count (>= `CRPS_PER_SEED_BAR` of the seeds
    ///    individually clear BOTH `CRPS_PER_SEED_MEAN_CEILING_FRAC·σ_y` and
    ///    `CRPS_PER_SEED_SIGMA_CEILING_FRAC·σ_y`) — the arm that catches
    ///    SIGN-BALANCED per-seed scatter a trimmed mean cancels away (see
    ///    [`CrpsSeedStats`]'s doc; MEASURED: a ±10-raw-unit alternating μ
    ///    scatter and a ×2/÷2 alternating σ scatter both read GREEN through
    ///    arms 1-2 alone, and RED through this arm).
    ///
    /// Non-finite counts as failing: ANY non-finite per-seed measurement
    /// rejects the WHOLE sweep (returns ONLY `NonFinite`) before any other
    /// arm runs — trimming/counting over a non-finite value is undefined,
    /// so a diverged seed can never be laundered away by being trimmed out
    /// or outvoted by finite neighbors.
    fn check_crps_aggregate(stats: &[CrpsSeedStats], sigma_y: f32) -> Vec<CrpsViolation> {
        let any_non_finite = stats.iter().any(|s| {
            !s.sigma_ratio.is_finite()
                || !s.mean_signed_diff.is_finite()
                || !s.mean_abs_diff.is_finite()
                || !s.sigma_abs_diff.is_finite()
        });
        if any_non_finite {
            return vec![CrpsViolation::NonFinite];
        }

        let mut violations = Vec::new();

        let sigma_ratios: Vec<f32> = stats.iter().map(|s| s.sigma_ratio).collect();
        let mean_diffs: Vec<f32> = stats.iter().map(|s| s.mean_signed_diff).collect();
        let tm_sigma_ratio = trimmed_mean(&sigma_ratios, 0.15);
        let tm_mean = trimmed_mean(&mean_diffs, 0.15);
        if !(CRPS_SIGMA_RATIO_LO..=CRPS_SIGMA_RATIO_HI).contains(&tm_sigma_ratio) {
            violations.push(CrpsViolation::SigmaRatioOutOfRange);
        }
        if tm_mean.abs() >= CRPS_MEAN_AGG_TOL {
            violations.push(CrpsViolation::MeanDrift);
        }

        let mean_ceiling = CRPS_PER_SEED_MEAN_CEILING_FRAC * sigma_y;
        let sigma_ceiling = CRPS_PER_SEED_SIGMA_CEILING_FRAC * sigma_y;
        let per_seed_passed = stats
            .iter()
            .filter(|s| s.mean_abs_diff <= mean_ceiling && s.sigma_abs_diff <= sigma_ceiling)
            .count();
        if per_seed_passed < CRPS_PER_SEED_BAR {
            violations.push(CrpsViolation::PerSeedCeilingViolated);
        }

        violations
    }

    /// Train the REAL, unmutated raw-space reference path (the pre-PR5 flow:
    /// head forward -> destandardize -> loss against RAW targets) and return
    /// `(served_r_mean_rows, served_r_sigma0)`. Shared by the real
    /// measurement and every mutant control below, so every mutant's z-path
    /// is compared against the SAME real raw-path training the real oracle
    /// uses — only the z-path differs per mutant.
    async fn measure_raw_reference(
        config: FineTuneConfig,
        targets: &Tensor,
        feats: &Tensor,
        device: &Device,
    ) -> (Vec<f32>, f32) {
        let (loop_r, vm_r) = regression_loop(config, 2, targets, device).await;
        let scaler_r = *loop_r.target_scaler.as_ref().unwrap();
        let mut opt = AdamW::new(
            vm_r.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..1500 {
            let z_out = loop_r.head_forward(feats).unwrap();
            let raw_head = scaler_r
                .destandardize(&z_out, &loop_r.regression_form())
                .unwrap();
            let batch = TrainingBatch::Regression {
                input: raw_head,
                target: targets.clone(),
            };
            let loss = loop_r.compute_loss(&batch).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        let z_head_r = loop_r.head_forward(feats).unwrap();
        let raw_head_r = scaler_r
            .destandardize(&z_head_r, &loop_r.regression_form())
            .unwrap();
        let rows_r = raw_head_r.to_vec2::<f32>().unwrap();
        let served_r_mean: Vec<f32> = rows_r.iter().map(|r| r[0]).collect();
        let served_r_sigma0 =
            super::super::regression_loss::softplus_std_for_test(rows_r[0][1] as f64) as f32;
        (served_r_mean, served_r_sigma0)
    }

    /// Build a [`CrpsSeedStats`] from a z-path serve (mean row-vector + σ row
    /// 0) and a raw-path reference — the one place all four per-seed fields
    /// are computed, shared by the real measurement and every mutant.
    fn crps_seed_stats_from_served(
        seed: u64,
        n: usize,
        served_z_mean: &[f32],
        served_z_sigma0: f32,
        served_r_mean: &[f32],
        served_r_sigma0: f32,
    ) -> CrpsSeedStats {
        let row_diffs: Vec<f32> = served_z_mean
            .iter()
            .zip(served_r_mean.iter())
            .take(n)
            .map(|(z, r)| z - r)
            .collect();
        let mean_signed_diff = row_diffs.iter().sum::<f32>() / row_diffs.len() as f32;
        let mean_abs_diff = row_diffs.iter().fold(0.0f32, |m, &d| m.max(d.abs()));
        let sigma_ratio = served_z_sigma0 / served_r_sigma0;
        let sigma_abs_diff = (served_z_sigma0 - served_r_sigma0).abs();
        CrpsSeedStats {
            seed,
            sigma_ratio,
            mean_signed_diff,
            mean_abs_diff,
            sigma_abs_diff,
        }
    }

    /// Train + serve one seed of the Crps z-vs-raw scenario through the REAL
    /// production dispatch, returning the RAW served vectors (mean row +
    /// σ scalar, both z-path and raw-path reference) rather than the
    /// aggregated [`CrpsSeedStats`]. Shared by [`measure_crps_seed`] and the
    /// two scatter-class regression tests below, so a scatter/injection can
    /// be applied to the SAME real served values every other caller sees —
    /// no post-hoc overwrite of an already-aggregated field, and no
    /// back-solving a raw value out of a ratio.
    async fn measure_crps_seed_raw(
        seed: u64,
        targets: &Tensor,
        feats: &Tensor,
        device: &Device,
    ) -> (Vec<f32>, f32, Vec<f32>, f32) {
        let config = FineTuneConfig {
            regression_loss: Some(RegressionLoss::Crps),
            seed,
            ..Default::default()
        };
        let (loop_z, vm_z) = regression_loop(config.clone(), 2, targets, device).await;
        let z_head = train_through_production_dispatch(&loop_z, &vm_z, feats, targets, 1500);
        let served_z = serve_through_production(&loop_z, &z_head);
        let (served_r_mean, served_r_sigma0) =
            measure_raw_reference(config, targets, feats, device).await;
        (
            served_z[0].clone(),
            served_z[1][0],
            served_r_mean,
            served_r_sigma0,
        )
    }

    /// Train + serve one seed of the Crps z-vs-raw scenario through the REAL
    /// production dispatch (mirrors the pre-rewrite test body; the only free
    /// variable is `config.seed`).
    async fn measure_crps_seed(
        seed: u64,
        targets: &Tensor,
        feats: &Tensor,
        n: usize,
        device: &Device,
    ) -> CrpsSeedStats {
        let (served_z_mean, served_z_sigma0, served_r_mean, served_r_sigma0) =
            measure_crps_seed_raw(seed, targets, feats, device).await;
        crps_seed_stats_from_served(
            seed,
            n,
            &served_z_mean,
            served_z_sigma0,
            &served_r_mean,
            served_r_sigma0,
        )
    }

    /// P10 — the scale-equivariant objectives (Crps, Pinball) share the SAME
    /// population minimizer in z vs raw space: the z loss is the raw loss / σ_y, so
    /// the analytic argmin is identical. The served raw output is therefore
    /// preserved across the two loss spaces — but NOT byte-equal: the production
    /// AdamW is not scale-free (its `eps = 1e-8` is added to `√v̂`, and the
    /// decoupled `weight_decay` shrinks θ by `lr·λ` independent of the loss scale),
    /// so dividing the loss by σ_y ≈ 19 shrinks every gradient by 1/σ_y and the eps
    /// term's relative weight and the moment trajectory shift. The two runs land on
    /// the same minimizer up to that optimizer-perturbation, not to machine epsilon.
    /// (β-NLL is NOT asserted — it is not scale-equivariant, P12, and the raw path
    /// diverges, so there is no raw solution to match.)
    ///
    /// esc-035 audit rework (second round): THREE independent arms —
    /// aggregate trimmed-mean σ-ratio, aggregate trimmed-mean signed μ-diff,
    /// per-seed ceiling count (see [`check_crps_aggregate`]) — over the
    /// [`CRPS_SEEDS`] 13-seed sweep (the pinned 12 + the actual default seed),
    /// not a single-trajectory, per-seed-count-only, or aggregate-only
    /// judgment. Detection-power table (before/after, both axes, both
    /// directions) is in this commit's message.
    #[tokio::test(flavor = "multi_thread")]
    async fn crps_served_output_preserved_within_tolerance_z_vs_raw() {
        let device = Device::Cpu;
        // The σ_y ≈ 19 WIDE target — the realistic scale. Crps is bounded ≈σ, so
        // the RAW-space path also converges cleanly here (it never trips the >100
        // guard), giving a raw solution to compare the z solution against AT the
        // scale where the optimizer perturbation (eps/decay ÷σ_y) actually bites.
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let sigma_y = wide_sigma_y();

        let mut stats = Vec::with_capacity(CRPS_SEEDS.len());
        for seed in CRPS_SEEDS {
            stats.push(measure_crps_seed(seed, &targets, &feats, n, &device).await);
        }
        let violations = check_crps_aggregate(&stats, sigma_y);
        let evaluated_seeds: Vec<u64> = stats.iter().map(|s| s.seed).collect();
        assert!(
            violations.is_empty(),
            "Crps served-output preservation (z vs raw), aggregated over the \
             {}-seed sweep {evaluated_seeds:?}: violations {violations:?}; \
             stats: {stats:?}",
            CRPS_SEEDS.len()
        );
    }

    // ─── esc-035 required-sequence step 3: permanent negative controls ───────
    //
    // The three K3-breaking mutants named by the escape, each asserted to make
    // `check_crps_aggregate` — the SAME checker fn and SAME named constants the
    // real oracle above calls, zero inlined tolerance literals — return `Err`
    // over the SAME `CRPS_SEEDS` sweep.

    /// MUTANT (i): TargetScaler neutralized to identity (μ=0, σ=1) on the
    /// z-path only, paired against the REAL (unmutated) raw-path reference.
    /// MEASURED: `MeanDrift` does NOT fire — the per-seed μ diffs are
    /// SIGN-SPLIT across the sweep (range −5.00…+3.56 raw units), so the
    /// aggregate μ arm's trimmed mean cancels to ~−0.266, a PASS on that arm
    /// alone. This mutant is instead caught by the σ-ratio arm (trimmed-mean
    /// ratio ~0.852, collapsed away from the [1.0, 1.3] band — training
    /// directly on raw-scale targets without ever being z-scored distorts
    /// the trained σ column's scale) AND the per-seed ceiling arm (every
    /// seed's μ diff is individually gross even though sign-balanced in
    /// aggregate): `[SigmaRatioOutOfRange, PerSeedCeilingViolated]`.
    #[tokio::test(flavor = "multi_thread")]
    async fn mutant_scaler_neutralized_rejected_by_aggregate_checker() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let sigma_y = wide_sigma_y();

        let mut stats = Vec::with_capacity(CRPS_SEEDS.len());
        for seed in CRPS_SEEDS {
            let config = FineTuneConfig {
                regression_loss: Some(RegressionLoss::Crps),
                seed,
                ..Default::default()
            };
            // MUTANT z-path: neutralize the scaler the production build just
            // reduced from `targets` — the identity transform, μ=0, σ=1.
            let (mut loop_z, vm_z) = regression_loop(config.clone(), 2, &targets, &device).await;
            loop_z.target_scaler = Some(TargetScaler::from_mean_std(0.0, 1.0));
            let z_head = train_through_production_dispatch(&loop_z, &vm_z, &feats, &targets, 1500);
            let served_z = serve_through_production(&loop_z, &z_head);
            let (served_r_mean, served_r_sigma0) =
                measure_raw_reference(config, &targets, &feats, &device).await;
            stats.push(crps_seed_stats_from_served(
                seed,
                n,
                &served_z[0],
                served_z[1][0],
                &served_r_mean,
                served_r_sigma0,
            ));
        }
        let violations = check_crps_aggregate(&stats, sigma_y);
        // MEASURED: the aggregate μ arm's trimmed-mean SIGNED diff cancels
        // (the per-seed mean diffs are sign-split across the sweep), so this
        // mutant is caught by the σ-ratio arm and the per-seed ceiling arm,
        // NOT the aggregate μ arm — asserting the SPECIFIC variants this
        // mutant fires (not a bare "any violation"), per the second audit
        // round.
        assert!(
            violations.contains(&CrpsViolation::SigmaRatioOutOfRange)
                && violations.contains(&CrpsViolation::PerSeedCeilingViolated),
            "mutant (i) [scaler neutralized] must be REJECTED by \
             check_crps_aggregate over the {}-seed sweep via SigmaRatioOutOfRange \
             AND PerSeedCeilingViolated — a rewrite under which either goes \
             missing is a relaxation hiding the regression. \
             violations: {violations:?}; stats: {stats:?}",
            CRPS_SEEDS.len()
        );
    }

    /// MUTANT (ii): loss-rescaling substitute — train the head directly
    /// against the RAW target (skip `z_score_targets`) and divide the
    /// resulting loss by σ_y, the K3-forbidden "fix" that acts on the loss
    /// instead of the data-space representation (family C: under Adam the
    /// parameter step is ~lr regardless of loss scale, so a loss-rescale
    /// cannot substitute for standardizing the target the head conditions
    /// on). Serving still runs the REAL, unmutated `destandardize`
    /// (production always treats the head output as z-scale), so the
    /// raw-trained head's output is destandardized a SECOND time — the
    /// served mean lands hundreds of raw units from the raw reference.
    /// Rejected by `check_crps_aggregate`'s own mean predicate (`MeanDrift`),
    /// no hand-rolled comparison.
    #[tokio::test(flavor = "multi_thread")]
    async fn mutant_loss_rescale_substitute_rejected_by_aggregate_checker() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let sigma_y = wide_sigma_y();

        let mut stats = Vec::with_capacity(CRPS_SEEDS.len());
        for seed in CRPS_SEEDS {
            let config = FineTuneConfig {
                regression_loss: Some(RegressionLoss::Crps),
                seed,
                ..Default::default()
            };
            let (loop_z, vm_z) = regression_loop(config.clone(), 2, &targets, &device).await;
            let scaler = *loop_z.target_scaler.as_ref().unwrap();
            let mut opt = AdamW::new(
                vm_z.all_vars(),
                ParamsAdamW {
                    lr: 0.05,
                    ..Default::default()
                },
            )
            .unwrap();
            for _ in 0..1500 {
                // MUTANT: score the RAW head output against the RAW target
                // (never z-scored), then rescale the LOSS by σ_y — the
                // forbidden loss-space "fix".
                let head_out = loop_z.head_forward(&feats).unwrap();
                let batch = TrainingBatch::Regression {
                    input: head_out,
                    target: targets.clone(),
                };
                let loss = loop_z.compute_loss(&batch).unwrap();
                let scaled_loss = (loss / scaler.std()).unwrap();
                let grads = scaled_loss.backward().unwrap();
                opt.step(&grads).unwrap();
            }
            // Serve through the REAL, unmutated destandardize — production
            // always treats the head's raw output as z-scale.
            let z_head = loop_z.head_forward(&feats).unwrap();
            let served_z = serve_through_production(&loop_z, &z_head);
            let (served_r_mean, served_r_sigma0) =
                measure_raw_reference(config, &targets, &feats, &device).await;
            stats.push(crps_seed_stats_from_served(
                seed,
                n,
                &served_z[0],
                served_z[1][0],
                &served_r_mean,
                served_r_sigma0,
            ));
        }
        let violations = check_crps_aggregate(&stats, sigma_y);
        // MEASURED: this mutant fires ALL THREE arms — the double-
        // destandardize offset is so large (tm_mean ~643 raw units) it trips
        // `MeanDrift`, the mechanism this mutant is meant to demonstrate
        // (family C: loss-rescale is not data-space standardization); the
        // destandardized head also lands nowhere near the raw reference's σ
        // scale (tm_sigma_ratio ~16.3), tripping `SigmaRatioOutOfRange`; and
        // every individual seed is grossly wrong on both axes, tripping
        // `PerSeedCeilingViolated`. Assert all three explicitly (the SPECIFIC
        // variants measured), not a bare "any violation".
        assert!(
            violations.contains(&CrpsViolation::MeanDrift)
                && violations.contains(&CrpsViolation::SigmaRatioOutOfRange)
                && violations.contains(&CrpsViolation::PerSeedCeilingViolated),
            "mutant (ii) [loss-rescaling substitute] must be REJECTED by ALL \
             THREE of check_crps_aggregate's arms over the {}-seed sweep — a \
             loss-space rescale is not a data-space standardization. \
             violations: {violations:?}; stats: {stats:?}",
            CRPS_SEEDS.len()
        );
    }

    /// MUTANT (iii): served σ built with `gaussian_scaled(1.0)` instead of
    /// `gaussian_scaled(scaler.std())` — the literal defect at
    /// trainer.rs:3214-3217, reproduced here as a hand-built adapter call
    /// (production code itself is untouched). Training is the REAL,
    /// unmutated production dispatch; only the serve-time adapter choice is
    /// corrupted, and only for the z-path (the raw-path reference is real
    /// and unmutated). Non-tautological: `gaussian_scaled(1.0)` collapses the
    /// served σ to ~1 raw unit against a raw-path reference that is
    /// genuinely trajectory-dependent (measured 5.9-29.7 raw units across
    /// seeds) — the real aggregate checker's `sigma_ratio` field, not a
    /// self-cancelling ratio of two numbers both derived from the SAME
    /// forced constant.
    #[tokio::test(flavor = "multi_thread")]
    async fn mutant_served_sigma_gaussian_scaled_one_rejected_by_aggregate_checker() {
        use crate::inference::adapter::{BackendOutput, DistributionAdapter, OutputAdapter};
        use arrow::array::{Array, Float32Array};

        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let sigma_y = wide_sigma_y();

        let mut stats = Vec::with_capacity(CRPS_SEEDS.len());
        for seed in CRPS_SEEDS {
            let config = FineTuneConfig {
                regression_loss: Some(RegressionLoss::Crps),
                seed,
                ..Default::default()
            };
            // REAL, unmutated production training.
            let (loop_z, vm_z) = regression_loop(config.clone(), 2, &targets, &device).await;
            let z_head = train_through_production_dispatch(&loop_z, &vm_z, &feats, &targets, 1500);
            let scaler = loop_z.target_scaler.as_ref().unwrap();
            let raw = scaler
                .destandardize(&z_head, &loop_z.regression_form())
                .unwrap();
            let served_rows = raw.to_vec2::<f32>().unwrap();
            let mean_col: Vec<f32> = served_rows.iter().map(|r| r[0]).collect();
            let width = served_rows.first().map_or(0, Vec::len);
            let flat: Vec<f32> = served_rows.into_iter().flatten().collect();
            let output = BackendOutput {
                float_outputs: vec![flat],
                string_outputs: vec![],
                row_status: vec![true; n],
                row_errors: vec![String::new(); n],
                shapes: vec![(n, width)],
            };
            // MUTANT: gaussian_scaled(1.0) instead of gaussian_scaled(scaler.std()).
            let cols = DistributionAdapter::gaussian_scaled(1.0_f32)
                .adapt(&output, n)
                .unwrap();
            let served_sigma_mutant = cols[1]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0);
            let (served_r_mean, served_r_sigma0) =
                measure_raw_reference(config, &targets, &feats, &device).await;
            stats.push(crps_seed_stats_from_served(
                seed,
                n,
                &mean_col,
                served_sigma_mutant,
                &served_r_mean,
                served_r_sigma0,
            ));
        }
        let violations = check_crps_aggregate(&stats, sigma_y);
        // MEASURED: only the serve-time adapter is corrupted, so the μ-axis
        // AGGREGATE (unaffected) stays clean (`MeanDrift` does NOT fire) —
        // but the collapsed σ makes EVERY individual seed's `sigma_abs_diff`
        // grossly wrong too, so this mutant fires BOTH `SigmaRatioOutOfRange`
        // AND `PerSeedCeilingViolated`. Assert both specifically.
        assert!(
            violations.contains(&CrpsViolation::SigmaRatioOutOfRange)
                && violations.contains(&CrpsViolation::PerSeedCeilingViolated),
            "mutant (iii) [gaussian_scaled(1.0) instead of gaussian_scaled(σ_y)] \
             must be REJECTED by check_crps_aggregate's SigmaRatioOutOfRange \
             AND PerSeedCeilingViolated arms over the {}-seed sweep — a \
             rewrite under which either goes missing is the ~19x \
             under-dispersion regression hiding. \
             violations: {violations:?}; stats: {stats:?}",
            CRPS_SEEDS.len()
        );
    }

    /// PERMANENT REGRESSION (esc-035, second audit round, block 1 — the
    /// "boundedness term" finding): a SIGN-BALANCED per-seed μ scatter that
    /// the two aggregate arms CANNOT see. 6 of the 13 sweep seeds get +10 raw
    /// units, 7 get -10 raw units (the 6/7 split, rather than an even
    /// alternation, compensates for the REAL baseline's own slight positive
    /// skew — MEASURED per-seed baseline `mean_signed_diff` is positive for
    /// 10 of 13 seeds — so the scattered result still lands inside the
    /// aggregate band rather than being pulled outside it by the pre-existing
    /// skew), added directly to every REAL served z-row via
    /// [`measure_crps_seed_raw`] (a real-path serve-time injection, not a
    /// post-hoc overwrite of an already-aggregated field) — a genuine
    /// per-trajectory gross violation on EVERY affected seed, landing so the
    /// trimmed mean of the (now sign-split) per-seed values stays INSIDE
    /// `CRPS_MEAN_AGG_TOL` (measured trimmed mean: -0.435, bound: 1.2).
    /// esc-035 requires K3's standardization to keep the served fit
    /// "BOUNDED across trajectories" — a checker with only two
    /// central-tendency statistics cannot express that, because
    /// sign-balanced scatter always has a valid central tendency near zero.
    /// This is why `PerSeedCeilingViolated` exists: it must fire here, and
    /// the two aggregate arms must NOT (this construction is deliberately
    /// inside their bands — if either now also fires, the construction no
    /// longer isolates the per-seed arm and must be re-derived).
    #[tokio::test(flavor = "multi_thread")]
    async fn mutant_sign_balanced_mean_scatter_rejected_by_per_seed_ceiling() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let sigma_y = wide_sigma_y();

        let mut stats = Vec::with_capacity(CRPS_SEEDS.len());
        for (i, seed) in CRPS_SEEDS.into_iter().enumerate() {
            let (served_z_mean, served_z_sigma0, served_r_mean, served_r_sigma0) =
                measure_crps_seed_raw(seed, &targets, &feats, &device).await;
            // MUTANT: +10 raw units on the first 6 seeds, -10 on the
            // remaining 7 (6/7, not an even alternation — see the doc
            // comment for why), added directly to every REAL served z-row —
            // a sign-balanced per-seed scatter; σ untouched.
            let delta = if i < 6 { 10.0 } else { -10.0 };
            let scattered_z_mean: Vec<f32> = served_z_mean.iter().map(|v| v + delta).collect();
            stats.push(crps_seed_stats_from_served(
                seed,
                n,
                &scattered_z_mean,
                served_z_sigma0,
                &served_r_mean,
                served_r_sigma0,
            ));
        }
        let violations = check_crps_aggregate(&stats, sigma_y);
        assert!(
            violations.contains(&CrpsViolation::PerSeedCeilingViolated),
            "a sign-balanced ±10-raw-unit μ scatter must be REJECTED via \
             PerSeedCeilingViolated — a checker with only aggregate \
             (median/trimmed-mean) statistics cannot see this class of \
             per-trajectory violation, which is exactly why this arm \
             exists. violations: {violations:?}"
        );
        assert!(
            !violations.contains(&CrpsViolation::MeanDrift),
            "this construction is deliberately INSIDE the aggregate μ band \
             (trimmed mean measured -0.435 against bound {CRPS_MEAN_AGG_TOL}) \
             so it isolates the per-seed ceiling arm — if MeanDrift now \
             fires too, re-derive the scatter split/magnitude so the \
             aggregate stays green and this test keeps demonstrating the \
             gap the per-seed arm closes. violations: {violations:?}"
        );
    }

    /// PERMANENT REGRESSION (esc-035, second audit round, block 1): the
    /// σ-axis analogue of the μ scatter above — a per-trajectory σ scale
    /// error injected on 2 of the 13 seeds, landing so the trimmed-mean
    /// ratio stays INSIDE `[CRPS_SIGMA_RATIO_LO, CRPS_SIGMA_RATIO_HI]`. Must
    /// be rejected via `PerSeedCeilingViolated` alone — `SigmaRatioOutOfRange`
    /// must NOT fire, or this construction no longer isolates the per-seed
    /// arm.
    ///
    /// De-pinned from a single dropout-stream trajectory (test-robustness
    /// fix, see this commit's message): the ORIGINAL construction hard-coded
    /// which 2 seeds to scatter (5, 11, "chosen away from the already-
    /// trimmed outlier seeds 2/8") and a fixed ×2/÷2 MULTIPLIER of whatever
    /// `served_z_sigma0` that trajectory happened to produce. Both choices
    /// ride the specific per-seed baseline the OLD host SplitMix64 dropout
    /// stream produced; a stream change (e.g. C7's device-side Philox
    /// dropout) redistributes which seeds are outliers and what their
    /// baseline σ actually is, so a hard-coded seed ID or a multiplier of a
    /// moving baseline can land outside the construction's intended zone.
    ///
    /// The fix derives EVERYTHING from THIS run's MEASURED per-seed baseline
    /// instead of hard-coded constants:
    /// 1. Measure the real (unmutated) per-seed ceiling pass/fail for every
    ///    seed in the sweep.
    /// 2. Pick exactly enough CURRENTLY-PASSING seeds to guarantee the
    ///    passing count drops below `CRPS_PER_SEED_BAR`
    ///    (`max(2, passing_count - (CRPS_PER_SEED_BAR - 1))`), so the
    ///    construction still isolates the per-seed arm even if the measured
    ///    baseline passing count shifts under a different stream — not just
    ///    at today's measured 10/13.
    /// 3. Among the passing candidates, prefer the ones whose sigma_ratio
    ///    sits CLOSEST to the aggregate band's midpoint (generalizes "away
    ///    from the outliers" to whichever seeds are the least-extreme under
    ///    THIS trajectory, not a hard-coded seed ID).
    /// 4. Inject each touched seed's σ RELATIVE TO ITS OWN measured raw
    ///    reference (`served_r_sigma0`), by a FIXED absolute margin
    ///    (`1.5 × sigma_ceiling`, itself derived only from `sigma_y` — a
    ///    property of the fixed `WIDE` targets, not the dropout stream) —
    ///    alternating the sign (+ / −) across touched seeds. This
    ///    guarantees `sigma_abs_diff = 1.5·sigma_ceiling > sigma_ceiling` for
    ///    EVERY touched seed regardless of trajectory (breaches the
    ///    per-seed ceiling by construction), while the AGGREGATE arm stays
    ///    safe by a "breach-one/dilute-many" argument: `check_crps_aggregate`
    ///    averages the (now sign-alternated) touched ratios into the
    ///    trimmed mean over 9 kept seeds, diluting each touched seed's
    ///    individual excess by ~9x — arithmetic that holds for ANY
    ///    trajectory, not a number calibrated to one.
    ///
    /// MEASURED (base, this commit): passing count 10/13 (seeds 2, 8 fail σ;
    /// seed 10 fails μ), so `touched_count = max(2, 10-8) = 2`; the 2 closest
    /// to the band midpoint (1.15) are seeds 11 (`sigma_ratio` 1.1145) and 6
    /// (1.0725). Injecting ±1.5·sigma_ceiling (±14.38 raw units) against
    /// each seed's own `served_r_sigma0` moves the trimmed-mean ratio from
    /// 1.108 (unmutated baseline) to 1.125 — comfortably inside
    /// `[1.0, 1.3]` — while dropping the per-seed-passing count to 8 < 9,
    /// tripping `PerSeedCeilingViolated`. Full pre-change table (including
    /// the superseded seed 5/11 ×2/÷2 numbers) is in this commit's message.
    #[tokio::test(flavor = "multi_thread")]
    async fn mutant_scale_balanced_sigma_scatter_rejected_by_per_seed_ceiling() {
        let device = Device::Cpu;
        let n = WIDE.len();
        let targets = Tensor::from_vec(WIDE.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let sigma_y = wide_sigma_y();
        let mean_ceiling = CRPS_PER_SEED_MEAN_CEILING_FRAC * sigma_y;
        let sigma_ceiling = CRPS_PER_SEED_SIGMA_CEILING_FRAC * sigma_y;

        // Measure the REAL, unmutated per-seed baseline for every swept seed
        // first — the scatter below is derived from THIS run's numbers, not
        // a constant calibrated to one historical trajectory.
        let mut baseline = Vec::with_capacity(CRPS_SEEDS.len());
        for seed in CRPS_SEEDS {
            let (served_z_mean, served_z_sigma0, served_r_mean, served_r_sigma0) =
                measure_crps_seed_raw(seed, &targets, &feats, &device).await;
            let stats = crps_seed_stats_from_served(
                seed,
                n,
                &served_z_mean,
                served_z_sigma0,
                &served_r_mean,
                served_r_sigma0,
            );
            baseline.push((
                seed,
                served_z_mean,
                served_z_sigma0,
                served_r_mean,
                served_r_sigma0,
                stats,
            ));
        }

        // Candidates: seeds that CURRENTLY pass both per-seed ceilings —
        // perturbing an already-failing seed wouldn't demonstrate anything
        // new about the aggregate/per-seed gap.
        let mut passing: Vec<usize> = (0..baseline.len())
            .filter(|&i| {
                let s = &baseline[i].5;
                s.mean_abs_diff <= mean_ceiling && s.sigma_abs_diff <= sigma_ceiling
            })
            .collect();
        let passing_count = passing.len();
        // Enough touched seeds to guarantee the passing count drops below
        // CRPS_PER_SEED_BAR, whatever the measured baseline passing count is
        // (not just today's measured 10/13).
        let touched_count = passing_count.saturating_sub(CRPS_PER_SEED_BAR - 1).max(2);
        assert!(
            passing.len() >= touched_count,
            "need >= {touched_count} currently-passing seeds to construct an \
             isolating scatter; measured passing count {passing_count} is too \
             low — the baseline itself may be regressing. baseline: {baseline:?}",
        );

        // Prefer candidates whose sigma_ratio sits closest to the aggregate
        // band's midpoint — generalizes "away from the outlier seeds" to
        // whichever seeds are least extreme under THIS trajectory.
        let band_mid = (CRPS_SIGMA_RATIO_LO + CRPS_SIGMA_RATIO_HI) / 2.0;
        passing.sort_by(|&a, &b| {
            let ra = baseline[a].5.sigma_ratio;
            let rb = baseline[b].5.sigma_ratio;
            (ra - band_mid)
                .abs()
                .partial_cmp(&(rb - band_mid).abs())
                .unwrap()
        });
        let touched: Vec<usize> = passing[..touched_count].to_vec();

        // Inject each touched seed's σ RELATIVE TO ITS OWN measured raw
        // reference, by a fixed absolute margin derived only from sigma_y —
        // guarantees a per-seed ceiling breach regardless of trajectory (see
        // this fn's doc for the "breach-one/dilute-many" aggregate-safety
        // argument). Alternate sign across touched seeds to keep the
        // aggregate contribution balanced.
        let margin = 1.5 * sigma_ceiling;
        let mut stats = Vec::with_capacity(CRPS_SEEDS.len());
        for (i, (seed, served_z_mean, served_z_sigma0, served_r_mean, served_r_sigma0, _)) in
            baseline.into_iter().enumerate()
        {
            let scattered_z_sigma0 = if let Some(pos) = touched.iter().position(|&t| t == i) {
                if pos % 2 == 0 {
                    served_r_sigma0 + margin
                } else {
                    (served_r_sigma0 - margin).max(1e-3)
                }
            } else {
                served_z_sigma0
            };
            stats.push(crps_seed_stats_from_served(
                seed,
                n,
                &served_z_mean,
                scattered_z_sigma0,
                &served_r_mean,
                served_r_sigma0,
            ));
        }
        let violations = check_crps_aggregate(&stats, sigma_y);
        assert!(
            violations.contains(&CrpsViolation::PerSeedCeilingViolated),
            "a per-trajectory σ scale error on {touched_count} measured-baseline \
             seeds must be REJECTED via PerSeedCeilingViolated — the \
             trimmed-mean ratio arm alone cannot see a minimal-footprint \
             scatter constructed to stay inside its band. touched (indices): \
             {touched:?}; violations: {violations:?}; stats: {stats:?}"
        );
        assert!(
            !violations.contains(&CrpsViolation::SigmaRatioOutOfRange),
            "this construction is derived to be deliberately INSIDE the \
             aggregate σ-ratio band [{CRPS_SIGMA_RATIO_LO}, \
             {CRPS_SIGMA_RATIO_HI}] (a fixed ceiling-relative margin diluted \
             over the trimmed-mean's 9 kept seeds) so it isolates the \
             per-seed ceiling arm — if SigmaRatioOutOfRange now fires too, \
             the dilution argument in this fn's doc no longer holds and the \
             scatter must be re-derived. violations: {violations:?}; stats: \
             {stats:?}"
        );
    }

    /// P9 — degenerate σ_y (constant target): a constant target floors σ_y at
    /// STD_FLOOR, the z-score is finite (every z = 0), the head fits the constant,
    /// and the served σ ≈ the floor (no spread). No NaN anywhere.
    #[tokio::test(flavor = "multi_thread")]
    async fn constant_target_serves_the_constant_with_floored_sigma() {
        let device = Device::Cpu;
        let n = 9;
        let constant = 42.0_f32;
        let targets = Tensor::from_vec(vec![constant; n], (n,), &device).unwrap();
        let config = FineTuneConfig {
            regression_loss: Some(RegressionLoss::Crps),
            ..Default::default()
        };
        let (loop_, varmap) = regression_loop(config, 2, &targets, &device).await;
        let feats = features(n, &device);
        let z_head = train_through_production_dispatch(&loop_, &varmap, &feats, &targets, 500);
        let cols = serve_through_production(&loop_, &z_head);
        for &m in &cols[0] {
            assert!(
                m.is_finite() && (m - constant).abs() < 1.0,
                "constant target: served mean {m} must be the constant {constant}, no NaN"
            );
        }
        for &s in &cols[1] {
            // σ_y = STD_FLOOR, so served σ = σ_y·σ_z ≈ STD_FLOOR·O(1) — tiny, finite.
            assert!(
                s.is_finite() && s > 0.0 && s < 1.0,
                "constant target: served σ {s} must be a finite near-floor value"
            );
        }
    }
}

/// W5-PR0b acceptance — CPU fine-tuning is bit-reproducible **through the real
/// LoRA `forward` path**.
///
/// The headline contract is: a fine-tune on `Device::Cpu` is a pure function of
/// `(seed, source rows, config)` — two runs at the same seed publish a
/// byte-identical `adapter.safetensors`, a different seed publishes a different
/// one. The four nondeterminism sources PR0b fixes — unseeded LoRA Kaiming/
/// Gaussian init (#1/#2), unseeded dropout (#3), and unstable source row order
/// (#6) — each break this.
///
/// Why this module exists and the `tests/it/ft_determinism.rs` integration test
/// does NOT carry the load-bearing coverage: that test feeds the loop
/// *precomputed* `TrainingBatch`es, so the trainer's precomputed branch routes
/// straight to `compute_loss` over the RAW embeddings — `LoraLinear::forward` is
/// never called, so **dropout is never drawn** and **the adapter never trains**
/// (`projection.lora_b` stays all-zeros; the compared bytes are purely the
/// seeded *init* of `lora_a`). That proves #1/#2 only.
///
/// This module instead drives the **production forward dispatch**, the same way
/// the `standardization_contract` oracle above drives `regress` → `compute_loss`:
/// every step runs the projection layer's `forward` (the production
/// `project_frozen_embedding` step — drawing the projection dropout mask),
/// `regress` (the distribution head's `forward` — drawing the distribution
/// dropout mask, then `TargetScaler::destandardize`), and the production
/// `compute_loss` → AdamW step. So the adapter genuinely TRAINS off zero-init and
/// both LoRA layers' seeded dropout is on the executed path. The saved bytes are
/// the production `save_adapter` artifact over the *trained* weights — the exact
/// object the worker publishes.
#[cfg(test)]
mod determinism_through_forward {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::super::data::TrainingBatch;
    use super::super::lora::{build_distribution_head, LoraModel};
    use super::super::regression_loss::TargetScaler;
    use super::super::target::TrainingTarget;
    use super::super::{EarlyStoppingMetric, FineTuneConfig, RegressionLoss};
    use super::{TrainingLoop, TrainingLoopBuilder};
    use crate::fine_tune::adamw::{AdamW, ParamsAdamW};

    const HIDDEN: usize = 8;
    /// A high-offset / low-variance calendar-year target — the same fixture the
    /// standardisation oracle uses, so the head is doing real (de-standardising)
    /// regression work as it trains.
    const YEARS: [f32; 9] = [
        2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
    ];
    /// Steps through the production dispatch. Chosen empirically large enough
    /// that the second-order gradient reaches `projection.lora_b` (which is zero
    /// at step 0, since the distribution head's `lora_b` starts at zero, and only
    /// moves once that has moved off zero) — see the non-zero assertions below.
    const STEPS: usize = 80;

    /// The fine-tune config under test: `lora_dropout > 0` so the seeded-dropout
    /// path is genuinely on the executed forward, β-NLL regression so the head
    /// does de-standardising work, small deterministic loop settings.
    fn determinism_config(seed: u64) -> FineTuneConfig {
        FineTuneConfig {
            seed,
            epochs: 1,
            batch_size: 1,
            validation_fraction: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            lora_dropout: 0.1,
            regression_loss: Some(RegressionLoss::BetaNll { beta: 0.5 }),
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            early_stopping_patience: 10_000,
            learning_rate: 1e-3,
            ..Default::default()
        }
    }

    /// Deterministic O(1) feature matrix `(n, HIDDEN)` — the projected embeddings
    /// the projection head sits on (stands in for a frozen base model's pooled
    /// output, exactly as the standardisation oracle's `features`). Independent of
    /// any seed so the *only* nondeterminism left is the one under test.
    fn features(n: usize, device: &Device) -> Tensor {
        let mut vals = Vec::with_capacity(n * HIDDEN);
        let mut s: u64 = 0x1234_5678;
        for _ in 0..n * HIDDEN {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0;
            vals.push(u * 0.5);
        }
        Tensor::from_vec(vals, (n, HIDDEN), device).unwrap()
    }

    /// Build a real production [`TrainingLoop`] over a regression
    /// [`TrainingTarget::ProjectionHead`] (projection + 2-wide Gaussian head),
    /// seeded at `seed`, with its [`TargetScaler`] reduced from `targets` exactly
    /// as `TrainingLoop::run` does. Goes through the production builder so nothing
    /// about the head/scaler wiring is synthetic.
    async fn regression_loop(
        seed: u64,
        targets: &Tensor,
        device: &Device,
    ) -> (TrainingLoop, VarMap) {
        let config = determinism_config(seed);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_distribution_head(HIDDEN, 2, &config, &varmap, &vb).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "det-model",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: crate::model::ModelTask::Regression,
                base_model_id: None,
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: "det-job",
                base_model_id: "det-model::1",
                training_source: "src",
                loss_type: "regression",
                hyperparams: "{}",
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .unwrap();
        catalog
            .claim_next_training_job("det-worker", std::time::Duration::from_secs(60))
            .await
            .unwrap()
            .expect("queued job is claimable");

        let mut loop_ = TrainingLoopBuilder::new(
            TrainingTarget::ProjectionHead { head },
            varmap.clone(),
            config,
        )
        .device(device.clone())
        .job_id("det-job".into())
        .worker_id("det-worker".into())
        .catalog(catalog)
        .artifact_dir(dir_path)
        .build()
        .unwrap();
        loop_.target_scaler = Some(TargetScaler::from_targets(targets).unwrap());
        (loop_, varmap)
    }

    /// Borrow the projection-head [`LoraModel`] out of a built loop's target.
    fn head_of(loop_: &TrainingLoop) -> &LoraModel {
        match &loop_.target {
            TrainingTarget::ProjectionHead { head } => head,
            _ => unreachable!("regression_loop builds a ProjectionHead target"),
        }
    }

    /// Run one full fine-tune at `seed` through the PRODUCTION forward dispatch and
    /// return the saved `adapter.safetensors` bytes plus the trained weights map.
    ///
    /// Each step is the exact production chain a `db.fine_tune(task=regression)`
    /// runs per batch: projection `forward` (the `project_frozen_embedding` step,
    /// drawing the projection dropout mask) → `head_forward` (distribution head
    /// `forward`, drawing its dropout mask, the raw z-output) → production
    /// `compute_loss` against the z-scored target → backward → AdamW. The adapter
    /// is then written through the production `save_adapter` over the *trained*
    /// weights. The z-score is a pure affine on the target, so the run stays a
    /// pure function of `(seed, rows, config)` and the bytes match across runs.
    async fn run_and_capture(seed: u64) -> (Vec<u8>, std::collections::HashMap<String, Tensor>) {
        let device = Device::Cpu;
        let n = YEARS.len();
        let targets = Tensor::from_vec(YEARS.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);

        let (loop_, varmap) = regression_loop(seed, &targets, &device).await;
        // Z-score the target exactly as `embed_chunk` does — the production loss
        // scores the raw z-head against the z-target.
        let z_target = {
            let scaler = loop_.target_scaler.as_ref().unwrap();
            let z: Vec<f32> = YEARS
                .iter()
                .map(|&y| scaler.standardize_value(y as f64) as f32)
                .collect();
            Tensor::from_vec(z, (n,), &device).unwrap()
        };

        let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW::default()).unwrap();
        for _ in 0..STEPS {
            // PRODUCTION projection forward (== `project_frozen_embedding`): the
            // base-model pooled output `feats` shifted by the projection LoRA. This
            // draws the projection layer's seeded dropout mask.
            let proj = head_of(&loop_).layers[0].1.forward(&feats).unwrap();
            // PRODUCTION distribution forward (`head_forward` reads `head.layers[1]`),
            // the raw z-output. Draws the distribution layer's seeded dropout mask.
            let head_out = loop_.head_forward(&proj).unwrap();
            let batch = TrainingBatch::Regression {
                input: head_out,
                target: z_target.clone(),
            };
            let loss = loop_.compute_loss(&batch).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }

        let weights = loop_.target.named_trainable_weights().unwrap();
        let saved = loop_.target.saved_adapter(
            &loop_.config,
            loop_.target_scaler,
            Some(loop_.regression_form()),
        );
        let dir = tempfile::tempdir().unwrap();
        jammi_lora::save_adapter(dir.path(), &weights, &saved).unwrap();
        let bytes = std::fs::read(dir.path().join("adapter.safetensors")).unwrap();
        (bytes, weights)
    }

    /// L∞ norm of a saved weight tensor, as f32.
    fn max_abs(w: &std::collections::HashMap<String, Tensor>, key: &str) -> f32 {
        w[key]
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// (a) Same seed → byte-identical published adapter, AND the adapter genuinely
    /// trained: both LoRA layers' `lora_a`/`lora_b` are non-zero. The non-zero
    /// `projection.lora_b` is the dead-path guard — a zero `lora_b` would mean the
    /// head's `forward` was never trained (the precomputed-batch regression this
    /// module replaces). Byte-equality here requires BOTH the seeded init AND the
    /// seeded dropout mask to be reproducible: the dropout mask is drawn on the
    /// executed forward every step (see `lora_linear.rs::forward`).
    #[tokio::test(flavor = "multi_thread")]
    async fn same_seed_byte_identical_through_trained_forward() {
        let (a, wa) = run_and_capture(12345).await;
        let (b, _wb) = run_and_capture(12345).await;
        assert_eq!(
            a,
            b,
            "same-seed CPU fine-tunes through the real forward path must publish a \
             byte-identical adapter.safetensors ({} vs {} bytes) — an unseeded \
             init/dropout source remains",
            a.len(),
            b.len()
        );

        // The adapter must have actually trained: a zero `lora_b` means the head's
        // `forward` was never on the optimised path (the dead-path regression).
        for key in [
            "projection.lora_a",
            "projection.lora_b",
            "distribution.lora_b",
        ] {
            let m = max_abs(&wa, key);
            assert!(
                m > 0.0,
                "{key} is all-zero after {STEPS} steps — the LoRA forward/training \
                 path was not exercised (max|·| = {m})"
            );
        }
    }

    /// (b) A different seed → a different published adapter — guards against the
    /// seed being silently ignored (which would make (a) pass vacuously).
    #[tokio::test(flavor = "multi_thread")]
    async fn different_seed_differs_through_trained_forward() {
        let (a, _) = run_and_capture(12345).await;
        let (b, _) = run_and_capture(67890).await;
        assert_ne!(
            a, b,
            "different seeds must publish different adapters — the seed is being ignored"
        );
    }
}

/// W5-PR2 deliverable — the resume invariant, proven byte-exact on `Device::Cpu`.
///
/// A fine-tune that dies at an epoch boundary, resumes from its durable
/// checkpoint, and continues the EXACT trajectory the uninterrupted run would
/// have. The proof is the three-run invariant of the design's §3:
///
///   1. the restored state is BYTE-EQUAL to the reference snapshot at the same
///      boundary (LoRA A/B, AdamW `(m, v)` per param, `step_t`, μ, σ), AND
///   2. the next steps produce weights BYTE-EQUAL to the reference's.
///
/// Assertion (2) is the one that catches a silent moment-reset: weights-only
/// resume passes (1) but fails (2) because zero moments + `step_t = 1`
/// bias-correction diverge immediately. The destructive `weights_only_*` test
/// below stubs exactly that and observes (2) fail, proving (2) is non-vacuous.
///
/// Each run drives the PRODUCTION forward dispatch (`projection.forward` →
/// `regress` → `compute_loss` → `AdamW::step`) with `lora_dropout > 0`, so the
/// seeded dropout mask is genuinely on the executed path; the capture and restore
/// are the trainer's real `capture_resume_bundle` / `restore_from_checkpoint`
/// routines, persisted through a real `file://` `ArtifactStore`. The falsifiers
/// embedded: R1 (≥3 LoRA layers so the optimizer's HashMap order is non-trivially
/// permuted — the moments must be name-keyed, not positional), R3 (dropout > 0,
/// drawn on the forward), R4 (the reference snapshot IS the persisted bundle —
/// same routine, same boundary), R6 (the run is on a multi-thread runtime and
/// asserts bit-exactness — a future candle reduction-order change fails loudly),
/// R7 (the persisted μ/σ is restored, never recomputed).
#[cfg(test)]
mod resume_invariant {
    use std::collections::HashMap;
    use std::sync::Arc;

    use bytes::Bytes;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, VarBuilder, VarMap};

    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    use super::super::data::TrainingBatch;
    use super::super::lora::LoraModel;
    use super::super::regression_loss::TargetScaler;
    use super::super::resume::{load_bundle, RestoredCheckpoint};
    use super::super::target::TrainingTarget;
    use super::super::{EarlyStoppingMetric, FineTuneConfig, RegressionLoss};
    use super::{TrainingLoop, TrainingLoopBuilder};
    use crate::fine_tune::adamw::{AdamW, ParamsAdamW};

    const HIDDEN: usize = 8;
    const YEARS: [f32; 9] = [
        2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
    ];

    /// `lora_dropout > 0` (R3), β-NLL regression so the head does de-standardising
    /// work, deterministic small-loop settings.
    fn resume_config(seed: u64) -> FineTuneConfig {
        FineTuneConfig {
            seed,
            epochs: 1,
            batch_size: 1,
            validation_fraction: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            lora_dropout: 0.1,
            regression_loss: Some(RegressionLoss::BetaNll { beta: 0.5 }),
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            early_stopping_patience: 10_000,
            learning_rate: 1e-3,
            ..Default::default()
        }
    }

    /// Deterministic O(1) feature matrix `(n, HIDDEN)` standing in for a frozen
    /// base model's pooled output — seed-independent so the only nondeterminism is
    /// the one under test.
    fn features(n: usize, device: &Device) -> Tensor {
        let mut vals = Vec::with_capacity(n * HIDDEN);
        let mut s: u64 = 0x1234_5678;
        for _ in 0..n * HIDDEN {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0;
            vals.push(u * 0.5);
        }
        Tensor::from_vec(vals, (n, HIDDEN), device).unwrap()
    }

    /// One seeded LoRA layer (`ZerosB` init + seeded dropout) at `vb.pp(name)`,
    /// registered into `varmap` — the production `build_head_layer` shape.
    fn lora_layer(
        out: usize,
        in_: usize,
        config: &FineTuneConfig,
        varmap: &VarMap,
        vb: &VarBuilder,
        name: &str,
    ) -> jammi_lora::LoraLinear {
        let base = Linear::new(
            Tensor::zeros((out, in_), DType::F32, vb.device()).unwrap(),
            None,
        );
        jammi_lora::LoraLinear::new(
            base,
            config.lora_rank,
            config.lora_alpha,
            config.use_rslora,
            jammi_lora::LoraInitMode::ZerosB,
            Some(config.lora_dropout as f32),
            config.seed,
            varmap,
            &vb.pp(name),
        )
        .unwrap()
    }

    /// A regression `ProjectionHead` with THREE LoRA layers (R1: enough that the
    /// optimizer's `all_vars()` HashMap order is non-trivially permuted, so a
    /// positional moment serialization would load the wrong param's moments). The
    /// `distribution` head stays at index 1 so `TrainingLoop::regress` reads it.
    /// All three layers are exercised and trained by `step_epoch` below.
    async fn build_three_layer_loop(
        seed: u64,
        targets: &Tensor,
        device: &Device,
        store: Arc<ArtifactStore>,
        resume: Option<RestoredCheckpoint>,
        job: &str,
    ) -> (TrainingLoop, VarMap) {
        let config = resume_config(seed);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let projection = lora_layer(HIDDEN, HIDDEN, &config, &varmap, &vb, "projection");
        let distribution = lora_layer(2, HIDDEN, &config, &varmap, &vb, "distribution");
        let aux = lora_layer(HIDDEN, HIDDEN, &config, &varmap, &vb, "aux");
        let head = LoraModel {
            layers: vec![
                ("projection".into(), projection),
                ("distribution".into(), distribution),
                ("aux".into(), aux),
            ],
        };

        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "resume-model",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: crate::model::ModelTask::Regression,
                base_model_id: None,
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: job,
                base_model_id: "resume-model::1",
                training_source: "src",
                loss_type: "regression",
                hyperparams: "{}",
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .unwrap();
        catalog
            .claim_next_training_job("resume-worker", std::time::Duration::from_secs(60))
            .await
            .unwrap()
            .expect("queued job is claimable");

        let mut builder = TrainingLoopBuilder::new(
            TrainingTarget::ProjectionHead { head },
            varmap.clone(),
            config,
        )
        .device(device.clone())
        .job_id(job.into())
        .worker_id("resume-worker".into())
        .catalog(catalog)
        .artifact_dir(dir_path)
        .artifact_store(store);
        if let Some(restored) = resume {
            builder = builder.resume(restored);
        }
        let mut loop_ = builder.build().unwrap();
        loop_.target_scaler = Some(TargetScaler::from_targets(targets).unwrap());
        (loop_, varmap)
    }

    fn head_of(loop_: &TrainingLoop) -> &LoraModel {
        match &loop_.target {
            TrainingTarget::ProjectionHead { head } => head,
            _ => unreachable!("three-layer loop builds a ProjectionHead target"),
        }
    }

    /// One production training step over all three LoRA layers — the exact chain
    /// `TrainingLoop::run`'s production path runs per batch: projection forward,
    /// the aux forward, the distribution forward (`head_forward`, the raw z-output),
    /// production `compute_loss` against the z-scored target, backward, AdamW step.
    /// Dropout masks are drawn on every forward, advancing each layer's seeded
    /// stream.
    fn step_epoch(loop_: &TrainingLoop, opt: &mut AdamW, feats: &Tensor, targets: &Tensor) {
        let head = head_of(loop_);
        let proj = head.layers[0].1.forward(feats).unwrap();
        let aux = head.layers[2].1.forward(&proj).unwrap();
        let head_out = loop_.head_forward(&aux).unwrap();
        // Z-score the target with the resumed scaler (persisted across crash/resume),
        // exactly as `embed_chunk` does in production.
        let z_target = {
            let scaler = loop_.target_scaler.as_ref().unwrap();
            let raw = targets.to_vec1::<f32>().unwrap();
            let z: Vec<f32> = raw
                .iter()
                .map(|&y| scaler.standardize_value(y as f64) as f32)
                .collect();
            Tensor::from_vec(z, (raw.len(),), targets.device()).unwrap()
        };
        let batch = TrainingBatch::Regression {
            input: head_out,
            target: z_target,
        };
        let loss = loop_.compute_loss(&batch).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }

    /// A fresh `file://` artifact store under a kept tempdir — the real durable
    /// resume backend the trainer writes `{job_id}/_resume/` into.
    fn file_store() -> Arc<ArtifactStore> {
        let root_dir = tempfile::tempdir().unwrap().keep();
        let cache = tempfile::tempdir().unwrap().keep();
        let root = StorageUrl::parse(root_dir.to_str().unwrap()).unwrap();
        Arc::new(ArtifactStore::with_root(root, StorageRegistry::new(), cache).unwrap())
    }

    /// Flatten a weights/moments map to a sorted `(key, bytes)` list for
    /// byte-equality assertions independent of HashMap order.
    fn weight_bytes(map: &HashMap<String, Tensor>) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = map
            .iter()
            .map(|(k, t)| {
                let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
                let bytes = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (k.clone(), bytes)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The exact bit-pattern of a tensor's f32 elements.
    fn tensor_bits(t: &Tensor) -> Vec<u8> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// Drive one named-param optimizer for `optim_param_names` from a loop's
    /// varmap, matching the trainer's single-snapshot order.
    fn build_opt(varmap: &VarMap, loop_: &TrainingLoop) -> (AdamW, Vec<String>) {
        let vars = varmap.all_vars();
        let opt = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: 1e-3,
                ..Default::default()
            },
        )
        .unwrap();
        let names = loop_.optimizer_param_names(&vars).unwrap();
        (opt, names)
    }

    /// Persist a resume checkpoint through the real capture routine and the real
    /// store. Mirrors `TrainingLoop::save_resume_checkpoint` exactly — capture via
    /// `capture_resume_bundle`, write via `put_resume_checkpoint` — but `.await`s
    /// the store write instead of `block_on`-ing it, so it is callable from an
    /// async test (the production save runs inside `spawn_blocking`, where
    /// `block_on` is valid; a test thread already drives the runtime).
    #[allow(clippy::too_many_arguments)]
    async fn persist(
        store: &Arc<ArtifactStore>,
        job: &str,
        loop_: &TrainingLoop,
        scratch: &std::path::Path,
        last_completed_epoch: usize,
        global_step: usize,
        opt: &AdamW,
        names: &[String],
    ) {
        let bundle = loop_
            .capture_resume_bundle(scratch, last_completed_epoch, global_step, opt, names)
            .unwrap();
        store.put_resume_checkpoint(job, &bundle).await.unwrap();
    }

    /// The full three-run invariant, multi-thread (R6).
    ///
    /// The "epoch boundary" here is `K` production steps; "the next steps" is `N`
    /// more. Reference: run K steps, persist the durable bundle via the trainer's
    /// `save_resume_checkpoint` (== `S_ref@K`), then run N more → `W_ref`. Crashed:
    /// a second loop runs K steps, persists, and is dropped. Resumed: a third loop
    /// `discover`s the durable bundle, restores via `restore_from_checkpoint`, runs
    /// N steps → `W_resumed`. Then mutate the persisted scaler-source between crash
    /// and resume (R7) and assert `W_resumed` still matches.
    #[tokio::test(flavor = "multi_thread")]
    async fn resume_reproduces_the_exact_trajectory_byte_for_byte() {
        const K: usize = 6;
        const N: usize = 5;
        let device = Device::Cpu;
        let n = YEARS.len();
        let targets = Tensor::from_vec(YEARS.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let store = file_store();

        // ── Reference ──────────────────────────────────────────────────────────
        let (ref_loop, ref_varmap) =
            build_three_layer_loop(42, &targets, &device, Arc::clone(&store), None, "ref-job")
                .await;
        let (mut ref_opt, ref_names) = build_opt(&ref_varmap, &ref_loop);
        for _ in 0..K {
            step_epoch(&ref_loop, &mut ref_opt, &feats, &targets);
        }
        // The reference snapshot at the K boundary IS the durable bundle (R4: same
        // capture routine, same boundary). `global_step K-1 == last completed`. The
        // durable save's `Handle::block_on` is valid only off the async runtime, so
        // the test persists the captured bundle through the store directly (the
        // capture is `capture_resume_bundle`, the exact routine the save uses).
        let scratch = tempfile::tempdir().unwrap();
        persist(
            &store,
            "ref-job",
            &ref_loop,
            scratch.path(),
            K - 1,
            K,
            &ref_opt,
            &ref_names,
        )
        .await;
        let s_ref_at_k = load_bundle(
            store
                .fetch_resume_checkpoint("ref-job")
                .await
                .unwrap()
                .unwrap()
                .dir(),
            &device,
        )
        .unwrap();
        // Continue N steps → the reference forward trajectory.
        for _ in 0..N {
            step_epoch(&ref_loop, &mut ref_opt, &feats, &targets);
        }
        let w_ref: HashMap<String, Tensor> = ref_loop.target.named_trainable_weights().unwrap();

        // ── Crashed ────────────────────────────────────────────────────────────
        let (crash_loop, crash_varmap) =
            build_three_layer_loop(42, &targets, &device, Arc::clone(&store), None, "crash-job")
                .await;
        let (mut crash_opt, crash_names) = build_opt(&crash_varmap, &crash_loop);
        for _ in 0..K {
            step_epoch(&crash_loop, &mut crash_opt, &feats, &targets);
        }
        let crash_scratch = tempfile::tempdir().unwrap();
        persist(
            &store,
            "crash-job",
            &crash_loop,
            crash_scratch.path(),
            K - 1,
            K,
            &crash_opt,
            &crash_names,
        )
        .await;
        let s_crash = load_bundle(
            store
                .fetch_resume_checkpoint("crash-job")
                .await
                .unwrap()
                .unwrap()
                .dir(),
            &device,
        )
        .unwrap();
        drop(crash_loop); // simulate process death

        // ── Assertion (1): restored state BYTE-EQUAL to S_ref@K ──────────────────
        assert_eq!(
            weight_bytes(&s_crash.weights),
            weight_bytes(&s_ref_at_k.weights),
            "restored LoRA A/B must be byte-equal to the reference snapshot"
        );
        assert_eq!(
            s_crash.state.step_t, s_ref_at_k.state.step_t,
            "restored step_t must match"
        );
        assert_eq!(
            s_crash.state.scaler, s_ref_at_k.state.scaler,
            "restored (μ, σ) must match"
        );
        // ≥3 params' moments must each be byte-equal BY NAME (R1): a positional
        // serialization would line up the wrong param under HashMap permutation.
        assert!(
            s_crash.moments.len() >= 3,
            "the head must have ≥3 LoRA params so the optimizer HashMap order is \
             non-trivially permuted (got {})",
            s_crash.moments.len()
        );
        for (name, (m, v)) in &s_ref_at_k.moments {
            let (cm, cv) = s_crash
                .moments
                .get(name)
                .unwrap_or_else(|| panic!("moment '{name}' missing from crash bundle"));
            assert_eq!(tensor_bits(m), tensor_bits(cm), "first moment '{name}'");
            assert_eq!(tensor_bits(v), tensor_bits(cv), "second moment '{name}'");
        }

        // ── R7: mutate the scaler source between crash and resume ────────────────
        // The persisted μ/σ must be authoritative — a recompute over a perturbed
        // source would diverge. We hand the resumed loop a DIFFERENT scaler-source;
        // resume must override it with the persisted (μ, σ) and still match.
        let perturbed = Tensor::from_vec(vec![0.0f32; n], (n,), &device).unwrap();

        // ── Resumed ──────────────────────────────────────────────────────────────
        let (resume_loop, resume_varmap) = build_three_layer_loop(
            42,
            &perturbed,
            &device,
            Arc::clone(&store),
            Some(s_crash),
            "resume-job",
        )
        .await;
        // The loop's restore ran in `build`? No — restore runs inside `run()`; here
        // we drive the forward manually, so apply the same restore the trainer does.
        let (mut resume_opt, resume_names) = build_opt(&resume_varmap, &resume_loop);
        let restored_bundle = load_bundle(
            store
                .fetch_resume_checkpoint("crash-job")
                .await
                .unwrap()
                .unwrap()
                .dir(),
            &device,
        )
        .unwrap();
        let (start_epoch, _gstep) = {
            // Borrow the loop mutably to restore weights/scaler/dropout, and the
            // opt to restore moments — the exact `restore_from_checkpoint` routine.
            let mut rl = resume_loop;
            let se = rl
                .restore_from_checkpoint(restored_bundle, &mut resume_opt, &resume_names)
                .unwrap();
            // The perturbed scaler-source must have been overridden by the
            // persisted (μ, σ) — R7.
            let restored_scaler = rl.target_scaler.unwrap();
            assert!(
                (restored_scaler.mean() - s_ref_at_k.state.scaler.unwrap().0).abs() < 1e-12,
                "resume must load the persisted μ, not recompute it from the \
                 (mutated) source"
            );
            // The restored state is already byte-equal to S_ref@K (weights,
            // moments, step_t, dropout positions, scaler) — the binding that makes
            // the NEXT steps reproduce is that the weights were restored INTO the
            // optimizer's `Var`s, so each post-resume step updates the same tensor
            // the forward reads (see `restore_from_checkpoint`).
            {
                let re_w = rl.target.named_trainable_weights().unwrap();
                assert_eq!(
                    weight_bytes(&re_w),
                    weight_bytes(&s_ref_at_k.weights),
                    "restored weights must be byte-equal to S_ref@K"
                );
                let (_re_m, re_t) =
                    TrainingLoop::capture_moments_by_name(&resume_opt, &resume_names).unwrap();
                assert_eq!(
                    re_t, s_ref_at_k.state.step_t,
                    "restored step_t must match S_ref@K"
                );
            }

            for _ in 0..N {
                step_epoch(&rl, &mut resume_opt, &feats, &targets);
            }
            let w_resumed: HashMap<String, Tensor> = rl.target.named_trainable_weights().unwrap();

            // ── Assertion (2): next-N weights BYTE-EQUAL to the reference ─────────
            assert_eq!(
                weight_bytes(&w_resumed),
                weight_bytes(&w_ref),
                "the resumed run's next-{N}-step weights must be byte-equal to the \
                 uninterrupted run's — a reset moment, lost step_t, recomputed \
                 scaler, or desynced dropout stream would diverge here"
            );
            se
        };
        assert_eq!(start_epoch, K, "resume starts at last_completed + 1");
    }

    /// Non-vacuity of assertion (2): a WEIGHTS-ONLY restore (zero optimizer
    /// moments + `step_t` reset to 0) passes assertion (1) on the weights but
    /// DIVERGES on the next-N steps — exactly the silent moment-reset the contract
    /// must catch. This stubs the broken restore and observes (2) fail, proving the
    /// full test above is not passing trivially.
    #[tokio::test(flavor = "multi_thread")]
    async fn weights_only_restore_diverges_on_next_steps() {
        const K: usize = 6;
        const N: usize = 5;
        let device = Device::Cpu;
        let n = YEARS.len();
        let targets = Tensor::from_vec(YEARS.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let store = file_store();

        let (ref_loop, ref_varmap) =
            build_three_layer_loop(7, &targets, &device, Arc::clone(&store), None, "wo-ref-job")
                .await;
        let (mut ref_opt, ref_names) = build_opt(&ref_varmap, &ref_loop);
        for _ in 0..K {
            step_epoch(&ref_loop, &mut ref_opt, &feats, &targets);
        }
        let scratch = tempfile::tempdir().unwrap();
        persist(
            &store,
            "wo-ref-job",
            &ref_loop,
            scratch.path(),
            K - 1,
            K,
            &ref_opt,
            &ref_names,
        )
        .await;
        let bundle = load_bundle(
            store
                .fetch_resume_checkpoint("wo-ref-job")
                .await
                .unwrap()
                .unwrap()
                .dir(),
            &device,
        )
        .unwrap();
        for _ in 0..N {
            step_epoch(&ref_loop, &mut ref_opt, &feats, &targets);
        }
        let w_ref = ref_loop.target.named_trainable_weights().unwrap();

        // BROKEN resume: restore ONLY the weights, scaler, and dropout — leave the
        // optimizer at zero moments and step_t = 0 (the weights-only checkpoint).
        let (mut wo_loop, wo_varmap) =
            build_three_layer_loop(7, &targets, &device, Arc::clone(&store), None, "wo-job").await;
        wo_loop.target.load_weights(&bundle.weights).unwrap();
        wo_loop
            .target
            .restore_dropout_positions(&bundle.state.dropout_positions)
            .unwrap();
        let (mut wo_opt, _wo_names) = build_opt(&wo_varmap, &wo_loop); // fresh zero moments
        for _ in 0..N {
            step_epoch(&wo_loop, &mut wo_opt, &feats, &targets);
        }
        let w_wo = wo_loop.target.named_trainable_weights().unwrap();

        assert_ne!(
            weight_bytes(&w_wo),
            weight_bytes(&w_ref),
            "a weights-only restore (zero moments + step_t reset) MUST diverge on \
             the next-{N} steps — if it matched, assertion (2) would be vacuous"
        );
    }

    /// R3 (the validation half): a validation pass — `set_training(false)`, a
    /// forward, `set_training(true)`, exactly what `TrainingLoop::run` wraps its
    /// `evaluate` call in — must NOT perturb the dropout stream. If it did, the
    /// masks the next training step draws would desync between a run that
    /// validates and one that resumes, breaking byte-equality.
    ///
    /// Two reference loops run K training steps; one of them interleaves a
    /// validation-mode forward (dropout off) before its next training step. Their
    /// next-step weights must be byte-equal — proving the eval forward drew no
    /// masks and left every layer's stream where the training forwards left it.
    #[tokio::test(flavor = "multi_thread")]
    async fn validation_pass_does_not_perturb_the_dropout_stream() {
        const K: usize = 4;
        let device = Device::Cpu;
        let n = YEARS.len();
        let targets = Tensor::from_vec(YEARS.to_vec(), (n,), &device).unwrap();
        let feats = features(n, &device);
        let store = file_store();

        // Run A: K steps, then one more training step.
        let (a_loop, a_varmap) =
            build_three_layer_loop(99, &targets, &device, Arc::clone(&store), None, "val-a").await;
        let (mut a_opt, _) = build_opt(&a_varmap, &a_loop);
        for _ in 0..K {
            step_epoch(&a_loop, &mut a_opt, &feats, &targets);
        }
        step_epoch(&a_loop, &mut a_opt, &feats, &targets);
        let w_a = a_loop.target.named_trainable_weights().unwrap();

        // Run B: K steps, a VALIDATION-mode forward (dropout off), then the same
        // training step. `set_training` is `&mut`, so take the head out by value
        // through a fresh binding.
        let (mut b_loop, b_varmap) =
            build_three_layer_loop(99, &targets, &device, Arc::clone(&store), None, "val-b").await;
        let (mut b_opt, _) = build_opt(&b_varmap, &b_loop);
        for _ in 0..K {
            step_epoch(&b_loop, &mut b_opt, &feats, &targets);
        }
        // A validation pass: dropout off → no mask draws, the stream is untouched.
        b_loop.target.set_training(false);
        {
            let head = head_of(&b_loop);
            let proj = head.layers[0].1.forward(&feats).unwrap();
            let aux = head.layers[2].1.forward(&proj).unwrap();
            let _ = b_loop.head_forward(&aux).unwrap();
        }
        b_loop.target.set_training(true);
        step_epoch(&b_loop, &mut b_opt, &feats, &targets);
        let w_b = b_loop.target.named_trainable_weights().unwrap();

        assert_eq!(
            weight_bytes(&w_a),
            weight_bytes(&w_b),
            "a validation-mode forward must draw no dropout masks — the dropout \
             stream is a separate, training-only stream that validation cannot \
             perturb, so the post-validation training step is byte-identical"
        );
    }

    /// R5 (the lease gate): a zombie worker — one whose lease was reclaimed, so
    /// its `cancel` flag is set — must NOT regress the shared `{job_id}/_resume/`
    /// checkpoint below the lease-winner's epoch. The trainer gates the durable
    /// save on `!cancel` at the epoch boundary, so a cancelled run writes nothing.
    ///
    /// Winner B persists an epoch-5 bundle. Zombie A then runs `TrainingLoop::run`
    /// with `cancel` pre-set and the SAME `job_id` + store; it bails at the first
    /// epoch-boundary cancel check and writes no durable checkpoint. The next
    /// `discover_resume` still returns the winner's epoch 5 — resume never goes
    /// backwards.
    #[tokio::test(flavor = "multi_thread")]
    async fn zombie_lease_loser_cannot_regress_the_resume_checkpoint() {
        use super::super::data::TrainingDataLoader;
        use std::sync::atomic::AtomicBool;

        let device = Device::Cpu;
        let store = file_store();
        let job = "r5-job";

        // Winner B persists an epoch-5 resume bundle directly through the store
        // (the durable state a healthy attempt would have written).
        let winner_state = super::super::resume::ResumeState {
            schema_version: super::super::resume::RESUME_STATE_SCHEMA_VERSION,
            last_completed_epoch: 5,
            global_step: 60,
            step_t: 60,
            seed: 42,
            scaler: None,
            dropout_positions: HashMap::new(),
        };
        let winner_bundle = vec![
            (
                "resume_state.json".to_string(),
                Bytes::from(serde_json::to_vec(&winner_state).unwrap()),
            ),
            // Minimal valid safetensors so `load_bundle` parses the bundle; the
            // tensor content is irrelevant to what R5 asserts (the epoch counter).
            safetensors_entry("adapter.safetensors", &["w.lora_a", "w.lora_b"], &device),
            safetensors_entry("optimizer.safetensors", &["w.m", "w.v"], &device),
        ];
        store
            .put_resume_checkpoint(job, &winner_bundle)
            .await
            .unwrap();

        // Zombie A: a real run with `cancel` already set, same job_id + store. It
        // must bail before any durable write.
        let cancel = Arc::new(AtomicBool::new(true));
        let config = resume_config(42);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let projection = lora_layer(4, 4, &config, &varmap, &vb, "projection");
        let head = LoraModel {
            layers: vec![("projection".into(), projection)],
        };

        let dir = tempfile::tempdir().unwrap().keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir).await.unwrap());
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "r5-model",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: crate::model::ModelTask::TextEmbedding,
                base_model_id: None,
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: job,
                base_model_id: "r5-model::1",
                training_source: "src",
                loss_type: "cosent",
                hyperparams: "{}",
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .unwrap();
        catalog
            .claim_next_training_job("r5-worker", std::time::Duration::from_secs(60))
            .await
            .unwrap()
            .unwrap();

        let a = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
        let b = Tensor::new(&[[0.0f32, 1.0, 0.0, 0.0]], &device).unwrap();
        let batch = TrainingBatch::Contrastive {
            embeddings_a: a,
            embeddings_b: b,
            scores: Tensor::new(&[1.0f32], &device).unwrap(),
        };
        let loader = TrainingDataLoader::from_precomputed(vec![batch]);

        let mut zombie =
            TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
                .device(device.clone())
                .job_id(job.into())
                .worker_id("r5-worker".into())
                .catalog(catalog)
                .artifact_dir(dir)
                .artifact_store(Arc::clone(&store))
                .cancel(cancel)
                .build()
                .unwrap();

        // The cancelled run bails at the first epoch-boundary check.
        let err = tokio::task::spawn_blocking(move || zombie.run(&loader))
            .await
            .unwrap()
            .unwrap_err();
        assert!(
            err.to_string().contains("training cancelled"),
            "a cancelled run must bail, got: {err}"
        );

        // The durable checkpoint still reports the winner's epoch 5 — the zombie
        // wrote nothing, so resume never regressed.
        let after = load_bundle(
            store
                .fetch_resume_checkpoint(job)
                .await
                .unwrap()
                .unwrap()
                .dir(),
            &device,
        )
        .unwrap();
        assert_eq!(
            after.state.last_completed_epoch, 5,
            "the zombie's stale write must not have regressed the checkpoint below \
             the lease-winner's epoch"
        );
    }

    /// A safetensors bundle entry over the given tensor keys so `load_bundle` can
    /// parse a hand-built winner bundle (R5) whose tensor content is irrelevant.
    fn safetensors_entry(name: &str, keys: &[&str], device: &Device) -> (String, Bytes) {
        let mut map = HashMap::new();
        for key in keys {
            map.insert(
                (*key).to_string(),
                Tensor::zeros((1,), DType::F32, device).unwrap(),
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        candle_core::safetensors::save(&map, &path).unwrap();
        (name.to_string(), Bytes::from(std::fs::read(&path).unwrap()))
    }
}
