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
// `Digest::new`/`Digest::update`/`Digest::finalize` for
// `evaluate_held_out`'s `batch_partition_sha256` (H1, unit 63) — the same
// trait `model/backend/candle.rs`'s content-digest hashing imports.
use sha2::Digest;

use crate::fine_tune::adamw::{AdamW, ParamsAdamW};
use jammi_db::error::{JammiError, Result};

use super::data::{TextChunk, TrainingDataLoader};
use super::optimizer::{accumulate_grads, clip_and_step, DEFAULT_NORM_CHECK_INTERVAL};
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

/// Test-only call counters for `encode_texts`'s `EncoderAdapters`-branch
/// dispatch between [`tokenize_and_bucket`] (train) and
/// [`tokenize_natural_width`] (eval) — adversarial-audit round 2, campaign
/// #443, item 3. Both functions return SELF-CONSISTENT `(rows, cols)` pairs
/// (a caller cannot tell, from `encode_texts`'s pooled `[rows, hidden]`
/// output alone, which one actually ran — bucketing is deliberately
/// output-invariant), so a black-box test cannot observe the routing
/// decision from the return value. Mirrors
/// [`PER_MICRO_BATCH_HOST_READ_COUNT`]'s own role just above: a test cannot
/// observe the internal path taken directly, so this is the structural
/// proxy.
#[cfg(test)]
static BUCKETED_TOKENIZE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static NATURAL_TOKENIZE_CALLS: AtomicU64 = AtomicU64::new(0);

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
    /// The RETAINED per-epoch checkpoints this attempt published (unit 348):
    /// `(epoch_index, artifact_prefix)` in ascending epoch order — already
    /// pruned to `config.keep_last_n_checkpoints` (or every epoch, when
    /// unset) by `TrainingLoop::save_epoch_checkpoint`. Empty when
    /// checkpointing was disabled (`artifact_store` unset on the builder).
    /// The worker's finalize CAS registers one catalog row per entry.
    pub epoch_checkpoints: Vec<(usize, String)>,
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
/// All five fields are borrowed mutably so the function can update batch
/// counts, loss accumulators, and the gradient store in place.
struct EpochState<'a> {
    batch_count: &'a mut usize,
    epoch_loss: &'a mut f64,
    accumulated_grads: &'a mut GradStore,
    /// Whether a micro-batch has been merged into `accumulated_grads` since
    /// the last optimizer step (i.e. whether the epoch-end flush has
    /// anything pending). Deliberately independent of whether
    /// `accumulated_grads` actually contains any entries: a batch whose
    /// backward produces no gradient for any trainable var (e.g. a loss with
    /// no valid contribution on that batch) still occupies a
    /// gradient-accumulation "slot" and must still trigger its window's
    /// optimizer step, exactly as a batch that did produce gradients would —
    /// `GradStore` emptiness is not a proxy for "no micro-batch happened
    /// here". A plain `bool` says exactly that, without introspecting the
    /// store.
    grads_pending: &'a mut bool,
    global_step: &'a mut usize,
}

/// The run's last-optimizer-step horizon plus the one-shot overshoot arm
/// that keeps a resumed run from re-syncing on every step.
///
/// [`Self::is_last_step`] takes the 1-based index of the optimizer step
/// about to run (`global_step + 1`) and is `true` on exactly one step of a
/// well-formed run: `step == horizon`. A resumed run whose horizon shrank
/// below its restored `global_step` (fewer configured epochs, a larger
/// accumulation window, or fewer rows than the run that wrote the
/// checkpoint) never reaches `step == horizon` — every step it takes is past
/// the horizon. Deciding with `>=` would call every one of those steps "the
/// last step" and force [`refuse_nonfinite_norm`]'s device→host read on all
/// of them: the per-step sync [`clip_gradients`] exists to remove, back on
/// every step of the resumed run. So an overshoot is checked ONCE — the
/// first step past the horizon fires the check and disarms — and the modulo
/// cadence covers the rest, exactly as it does for a fresh run.
///
/// Lattice (`step` against `horizon`, with the arm state):
///
/// | cell                          | result                          |
/// |-------------------------------|---------------------------------|
/// | `step < horizon`              | `false` (cadence decides)       |
/// | `step == horizon`             | `true` (the exact last step)    |
/// | `step > horizon`, armed       | `true`, then disarm (one-shot)  |
/// | `step > horizon`, disarmed    | `false` (cadence decides)       |
///
/// Each cell has a run-level oracle in `last_step_horizon_run_oracles`
/// (driven through [`TrainingLoop::run`], counting the reads).
///
/// [`refuse_nonfinite_norm`]: super::optimizer::refuse_nonfinite_norm
/// [`clip_gradients`]: super::optimizer::clip_gradients
struct LastStepHorizon {
    /// The run's actual optimizer-step count for the arm it takes (see the
    /// lattice doc above `total_optimizer_steps` in [`TrainingLoop::run`]).
    horizon: usize,
    /// Whether the one-shot overshoot check is still armed.
    overshoot_armed: bool,
}

impl LastStepHorizon {
    fn new(horizon: usize) -> Self {
        Self {
            horizon,
            overshoot_armed: true,
        }
    }

    /// Whether optimizer step `step` (1-based) must force the non-finite
    /// check as the run's last step — see the type's lattice.
    fn is_last_step(&mut self, step: usize) -> bool {
        if step == self.horizon {
            return true;
        }
        if step > self.horizon && self.overshoot_armed {
            self.overshoot_armed = false;
            return true;
        }
        false
    }
}

/// Immutable per-step context (except for the optimizer and the last-step
/// horizon's one-shot arm, which mutate on a step). Constructed fresh for each call to
/// [`TrainingLoop::process_batch_loss`] and dropped at function return so the
/// caller can keep using `optimizer` directly between iterations.
struct StepContext<'a> {
    trainable_vars: &'a [Var],
    optimizer: &'a mut AdamW,
    checkpoint_dir: &'a Path,
    checkpoint_interval: usize,
    /// The LR schedule's horizon — `compute_lr`'s `total_steps`, the
    /// accumulation-window step count `run` computes before the loop. One
    /// meaning only: where the schedule's decay ends.
    lr_horizon: usize,
    /// The run's last-step horizon (`total_optimizer_steps` in `run`) with
    /// its one-shot overshoot arm. One meaning only: which optimizer step
    /// forces the non-finite check as the run's last. On this arm the two
    /// horizons are one number (`process_batch_loss` asserts it), but they
    /// are named apart because they are NOT one number on the GradCache
    /// arm (`epochs` vs `ceil(batches / grad_accum) * epochs`), and a single
    /// field carrying both meanings is how the GradCache horizon was wrong
    /// once already.
    last_step_horizon: &'a mut LastStepHorizon,
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
    /// This claim's attempt counter, as a string segment (`record.attempts`
    /// in the worker) — the third segment of the attempt-unique publish
    /// prefix per-epoch checkpoints are written under
    /// (`{job_id}/{worker_id}/{attempt}/checkpoints/epoch_{N}/`, unit 348,
    /// K7). Defaults to `"0"` when unset ([`TrainingLoopBuilder::new`]) so a
    /// trainer-internal test that never calls
    /// [`TrainingLoopBuilder::attempt`] still gets a valid (if not
    /// production-meaningful) prefix.
    attempt: String,
    catalog: Arc<Catalog>,
    /// The local directory training scratch (the per-run tempdir holding
    /// checkpoints and the final adapter) is created under. The run owns a
    /// fresh tempdir within it, so two workers training the same `job_id` never
    /// share a training-time path; the worker publishes the final files from
    /// there to the artifact store under a unique per-attempt prefix.
    artifact_dir: PathBuf,
    /// Mirrors `self.target`'s current dropout/training mode, updated ONLY
    /// through [`Self::set_training`] (audit round 63, finding 2). Exists so
    /// [`Self::with_dropout_disabled`] can capture the pre-call state and
    /// restore THAT — rather than hard-coding a restore-to-`true` — because
    /// `TrainingTarget` itself exposes no getter (its training flag lives
    /// distributed across per-layer `LoraLinear`/encoder state).
    ///
    /// `true` at [`TrainingLoopBuilder::build`] is ENFORCED, not assumed
    /// (audit round 63, re-audit finding 1): the prior doc here claimed
    /// "every `TrainingTarget` this crate constructs for training starts in
    /// training mode" and hard-coded the field to `true` at construction —
    /// false for `TrainingTarget::EncoderAdapters`, whose `ModernBert` body
    /// is built with `training: false` (only the injected `LoraLinear`
    /// adapters start `true`), so the target's real state was heterogeneous
    /// and this mirror was a fabricated claim about it. `build` now calls
    /// [`TrainingLoop::set_training`]`(true)` on the freshly assembled loop
    /// before returning it, which recurses through the WHOLE target —
    /// encoder body included — so every layer this loop owns is actually in
    /// training mode and this field is correct BY CONSTRUCTION regardless of
    /// which `TrainingTarget` variant, or how it was itself constructed,
    /// `build` was handed. This also closes the latent bug where an
    /// `EncoderAdapters` production run (`worker::run_fine_tune_blocking`)
    /// trained with its encoder body never switched into training mode (only
    /// its LoRA adapters' dropout gate ever flipped).
    ///
    /// Updated in lockstep with every production `set_training` call this
    /// loop makes after that.
    training_mode: bool,
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
    /// Retained per-epoch checkpoints this attempt has published so far
    /// (unit 348): `(epoch_index, artifact_prefix)` in ascending epoch order.
    /// Appended at every epoch boundary by [`Self::save_epoch_checkpoint`],
    /// which also enforces `config.keep_last_n_checkpoints` by deleting and
    /// dropping the oldest entries once the cap is exceeded — so at any point
    /// this vector holds exactly the RETAINED set, never a stale entry whose
    /// bytes were already reclaimed. Threaded into [`TrainingResult`] at the
    /// end of [`Self::run`] for the worker's finalize to register.
    epoch_checkpoints: Vec<(usize, String)>,
    /// Test seam: runs on the gradients every optimizer step is about to
    /// consume, right after `backward` (and, on the GradCache arm, after the
    /// two-pass `gradcache_backward`), keyed by the 1-based index of that
    /// step. Lets a test poison a chosen step's gradient — the only way to
    /// reach `clip_and_step`'s last-step check through the REAL `run` with a
    /// NaN that the pre-step loss read cannot see. Never compiled into the
    /// production binary.
    #[cfg(test)]
    after_backward: Option<AfterBackwardHook>,
}

/// See [`TrainingLoop::after_backward`]: `(step, grads, trainable_vars)`,
/// where `step` is the 1-based index of the optimizer step about to consume
/// `grads` (`global_step + 1`). `Send` because a `TrainingLoop` is driven
/// from `spawn_blocking`.
#[cfg(test)]
type AfterBackwardHook = Box<dyn FnMut(usize, &mut GradStore, &[Var]) -> Result<()> + Send>;

/// Builder for [`TrainingLoop`].
pub struct TrainingLoopBuilder {
    target: TrainingTarget,
    base_model: Option<Arc<LoadedModel>>,
    varmap: VarMap,
    config: FineTuneConfig,
    job_id: Option<String>,
    worker_id: Option<String>,
    /// See [`TrainingLoop::attempt`]. Defaults to `"0"` — set explicitly
    /// (via [`Self::attempt`]) only by the production worker path.
    attempt: String,
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
            attempt: "0".to_string(),
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

    /// Set this claim's attempt counter — the third segment of the
    /// attempt-unique prefix per-epoch checkpoints publish under (unit 348).
    /// Omit it only for a trainer-internal test (defaults to `"0"`); the
    /// production worker path always sets it to `record.attempts`.
    pub fn attempt(mut self, attempt: String) -> Self {
        self.attempt = attempt;
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
        let mut training_loop = TrainingLoop {
            target: self.target,
            base_model: self.base_model,
            varmap: self.varmap,
            config: self.config,
            job_id,
            worker_id,
            attempt: self.attempt,
            catalog,
            artifact_dir,
            // Placeholder — `set_training(true)` below overwrites this
            // immediately and is the ONLY thing that makes it meaningful.
            training_mode: false,
            divergence_count: 0,
            target_scaler: None,
            device: self.device,
            cancel: self.cancel,
            artifact_store: self.artifact_store,
            resume: self.resume,
            epoch_checkpoints: Vec::new(),
            #[cfg(test)]
            after_backward: None,
        };
        // Enforce, not assume, that a freshly built loop starts in training
        // mode (audit round 63, re-audit finding 1 — see `training_mode`'s
        // own doc). This recurses through the WHOLE target via
        // `TrainingTarget::set_training`, so an `EncoderAdapters` target's
        // encoder body (constructed `training: false` — only its injected
        // LoRA adapters start `true`) is switched into training mode here
        // too, not just the mirror.
        training_loop.set_training(true);
        Ok(training_loop)
    }
}

/// Tokenizes `texts` via `tokenizer`'s own `BatchLongest` padding, then
/// rounds the batch's natural width UP to
/// [`jammi_numerics::bucket_seq_len`]'s bucket ladder and extends every row
/// to that bucketed width — see `crate::fine_tune::batch_bucket`'s module
/// doc for the mechanism/rationale this closes (esc-076,
/// `.jammi/escapes.jsonl`). Returns the bucketed [`BatchEncoding`] alongside
/// the row count and the bucketed column width actually produced, so a
/// caller can build a `[rows, cols]` tensor directly without recomputing
/// either.
///
/// **TRAINING-STEP path only** (adversarial-audit round 2, campaign #443,
/// item 3 — amending esc-076): [`TrainingLoop::encode_texts`]'s
/// `EncoderAdapters` branch calls this ONLY while `self.training_mode` is
/// `true`; see [`tokenize_natural_width`]'s doc for the sibling eval-time
/// path and why bucket-UP padding is wrong there. Bucketing exists to bound
/// the COUNT of distinct tensor shapes a non-caching CUDA allocator sees
/// across the UNBOUNDED sequence of per-training-step batches (esc-076's own
/// mechanism) — an eval pass is not that path.
///
/// Factored out of [`TrainingLoop::encode_texts`]'s `EncoderAdapters` branch
/// (its only caller) — not merely inlined there — so a unit test can drive
/// the PRODUCTION tokenize+bucket step directly and assert its bucketed
/// shape without duplicating the decision: deleting either
/// `pad_rows_to_bucket` call below turns
/// `encode_texts_bucketing_oracle::tokenize_and_bucket_pads_every_row_to_the_bucket_ladder`
/// red (rows stay at their natural, unbucketed width, so `cols` no longer
/// matches every row's actual length).
fn tokenize_and_bucket(
    tokenizer: &crate::model::tokenizer::TokenizerWrapper,
    texts: &[String],
    effective_max: usize,
) -> Result<(crate::model::tokenizer::BatchEncoding, usize, usize)> {
    #[cfg(test)]
    BUCKETED_TOKENIZE_CALLS.fetch_add(1, Ordering::Relaxed);
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let mut encoding = tokenizer.encode_batch(&text_refs, Some(effective_max))?;

    let rows = encoding.input_ids.len();
    let natural_cols = encoding.input_ids.first().map_or(0, |v| v.len());
    // esc-076: round this batch's own (tokenizer `BatchLongest`) natural
    // width UP to a small, fixed bucket ladder — see
    // `crate::fine_tune::batch_bucket`'s module doc for why an UNBOUNDED
    // count of distinct per-step tensor shapes fragments/grows cudarc's
    // non-caching CUDA allocator, and why extending the SAME trailing-zero
    // padding contract `BatchLongest` already relies on (pad id `0`, mask
    // `0`) is output-invariant. Every dtype/objective through this ONE
    // `EncoderAdapters` TRAINING-STEP call site is bucketed uniformly —
    // never an f16-specific knob (but see the eval-time exemption above:
    // this function is only reached from the training-step path today).
    let cols = jammi_numerics::bucket_seq_len(natural_cols, effective_max);
    crate::fine_tune::batch_bucket::pad_rows_to_bucket(&mut encoding.input_ids, cols, 0);
    crate::fine_tune::batch_bucket::pad_rows_to_bucket(&mut encoding.attention_masks, cols, 0);

    Ok((encoding, rows, cols))
}

/// Tokenizes `texts` via `tokenizer`'s own `BatchLongest` padding WITHOUT any
/// further bucket-rounding — every row is exactly the batch's own natural
/// (tokenizer `BatchLongest`) width, matching pre-esc-076 behaviour.
///
/// **EVAL path only** (adversarial-audit round 2, campaign #443, item 3):
/// [`TrainingLoop::encode_texts`]'s `EncoderAdapters` branch calls this
/// while `self.training_mode` is `false` — i.e. inside
/// [`TrainingLoop::with_dropout_disabled`]'s bracket
/// ([`TrainingLoop::evaluate`]/[`TrainingLoop::evaluate_held_out`]).
///
/// **The bound this exemption actually relies on** (r3 finding B4 —
/// superseding an earlier, wrong version of this doc that argued "eval
/// runs infrequently"; that bounds the number of PASSES, not the quantity
/// esc-076 cares about): esc-076's mechanism is the COUNT of DISTINCT
/// tensor shapes a non-caching CUDA allocator (`cudarc`) is ever asked to
/// satisfy — a held-out split with several batches of several different
/// natural widths presents that many distinct shapes in a SINGLE eval
/// pass, regardless of `eval_cadence`; "runs at most a handful of times"
/// says nothing about that count. The real reason eval's contribution is
/// still bounded: the held-out/val partition is DETERMINISTIC — the same
/// rows, in the same batch order, on every pass
/// ([`TrainingLoop::evaluate_held_out`]'s own `example_ids` contract) — so
/// eval re-presents the IDENTICAL sequence of natural widths every time it
/// runs. Its distinct-shape contribution to the allocator is therefore
/// paid EXACTLY ONCE per run (on the first pass; cudarc never returns a
/// reserved block to the OS, so every later pass's widths are already-seen
/// repeats, not new shapes) — never growing per-step or per-epoch the way
/// esc-076's own genuinely UNBOUNDED per-training-step churn did.
///
/// That one-time set's SIZE is caller-dependent, and this doc does not
/// hide that: it is bounded by the held-out split's own natural-width
/// diversity (up to one distinct width per batch in the split, not by the
/// bucket ladder's ~11 rungs), so a sufficiently wide/varied held-out split
/// could still present a nontrivial one-time shape count. This exemption
/// does not newly bound that — it RESTORES the pre-esc-076 baseline for
/// eval (eval encoded at natural width before the training-step bucket
/// ladder existed at all; esc-076's own fix never touched eval's
/// shape-count exposure, only the training step's) rather than introducing
/// a new regression. **esc-076 remains OPEN on this residual axis**: an
/// eval split large/varied enough to itself present many distinct widths
/// in its one-time set is not proven bounded by anything in this module —
/// only that this exemption does not make that pre-existing exposure any
/// worse than it always was.
///
/// [`tokenize_and_bucket`]'s bucket ladder still exists to bound the
/// TRAINING-STEP path's actually-unbounded-across-the-run distinct-shape
/// count (esc-076's own mechanism, `crate::fine_tune::batch_bucket`'s
/// module doc). Rounding an eval batch's real width UP to the run's
/// `max_seq_length` bucket regardless of how short its actual content is
/// (e.g. a genuine 321-token held-out batch padded to the 512 bucket, a
/// measured `~2.5x` softmax-intermediate blow-up: `512² / 321² ≈ 2.5`) paid
/// a real memory cost for a shape-count benefit eval's deterministic,
/// paid-once partition never needed, and measurably OOM'd a `--batch 8
/// --max-seq-length 512` bf16 leg that ran clean at the pre-bucketing
/// baseline (adversarial-audit round 2 dispute; `.jammi/escapes.jsonl`'s
/// esc-076 entry itself never exercised a real eval-time batch at a
/// natural width anywhere near a bucket rung, since its own reporter shape
/// used `max_seq_length: 128`).
fn tokenize_natural_width(
    tokenizer: &crate::model::tokenizer::TokenizerWrapper,
    texts: &[String],
    effective_max: usize,
) -> Result<(crate::model::tokenizer::BatchEncoding, usize, usize)> {
    #[cfg(test)]
    NATURAL_TOKENIZE_CALLS.fetch_add(1, Ordering::Relaxed);
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let encoding = tokenizer.encode_batch(&text_refs, Some(effective_max))?;
    let rows = encoding.input_ids.len();
    let cols = encoding.input_ids.first().map_or(0, |v| v.len());
    Ok((encoding, rows, cols))
}

impl TrainingLoop {
    /// The single scratch subdirectory (under the run's `checkpoint_dir`)
    /// every epoch's checkpoint save reuses — the `_resume_scratch` precedent
    /// [`Self::save_resume_checkpoint`] already follows. Reused, not
    /// per-epoch, so the run's scratch-disk footprint does not grow with
    /// epoch count.
    const EPOCH_CHECKPOINT_SCRATCH: &'static str = "_epoch_checkpoint_scratch";

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
        // `self.config.epochs`, not `total_steps` — with `total_steps`,
        // `global_step + 1` never reaches the horizon on GradCache's actual
        // last step whenever `num_batches > 1`, so the run's real final step
        // would never force `clip_and_step`'s non-finite check the way a run
        // shorter than `DEFAULT_NORM_CHECK_INTERVAL` needs it to (see
        // `optimizer::clip_and_step`'s doc). Computed ONCE,
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
        // for, and how each is handled (each arm has a run-level oracle in
        // `last_step_run_harness` driving `run` with a NaN gradient on that
        // arm's last step):
        //
        //  - **accumulation window** (`process_batch_loss`'s window
        //    boundary): one step per full `grad_accum`-sized window;
        //    `total_steps`'s `div_ceil` counts these.
        //  - **partial last batch** (the epoch-end flush of a trailing window
        //    smaller than `grad_accum`, `batches_per_epoch % grad_accum !=
        //    0`): the same `div_ceil` counts it as one extra step.
        //  - **GradCache**: one step per epoch — `total_optimizer_steps` is
        //    `self.config.epochs` on this arm, not `total_steps`.
        //  - **mined loader** (`trainer.rs`'s hard-negative refresh; see
        //    `mine_hard_negative_loader`'s doc): a row whose mined pool is
        //    entirely excluded is DROPPED, so a mined epoch's row (and
        //    therefore batch/window) count can differ from `train_loader`'s —
        //    what `total_steps` was computed from, once, before the loop.
        //    This desync is NOT resolved here, and why is specific, not just
        //    "known gap": mining is lazy and re-mined only at
        //    `hard_negatives.refresh_every`-epoch boundaries (`mining_eligible
        //    ` + `should_refresh`, above), so the drop count for a REFRESHING
        //    epoch is unknowable until that epoch's own `text_chunks()`/mined
        //    triplets are built — after `total_steps` has already been used to
        //    seed `compute_lr`'s horizon AND `LastStepHorizon`. A correct fix
        //    needs one of: (a) mining every epoch upfront, before the loop,
        //    to know every epoch's row count before the LR schedule is
        //    seeded — defeats the "stale reuse between refreshes" cost
        //    trade-off `mined_loader`'s own doc states as the reason it is
        //    NOT re-mined every epoch; or (b) re-deriving `total_steps` (and
        //    therefore `compute_lr`'s `progress` fraction for every step
        //    already taken) mid-run, the first time a mined epoch's row count
        //    diverges — which turns a fixed run-level LR schedule into one
        //    that can retroactively change shape, a correctness contract this
        //    round's `A1`/`A2`/`B2`/`B3` oracles do not cover and would need
        //    their own sweep to add safely. Both are a materially larger,
        //    separable change than a device-side clip; deferred, not silently
        //    absorbed into "fixed" by this round. The risk this leaves is
        //    bounded to hard-negative-mining runs specifically (`mining_
        //    eligible()` gates it) whose mined pool loses enough rows on a
        //    refresh epoch to shift that epoch's window count — every
        //    non-mining run (the common path, and everything `last_step_run_
        //    harness` and `last_step_horizon_run_oracles` drive) is unaffected.
        //  - **early stopping** (`break` on patience exhaustion): an
        //    early-stopped run's actual last step is whatever `global_step`
        //    reached before the `break`, which `total_optimizer_steps`
        //    (fixed upfront from `self.config.epochs`) does not know ahead of
        //    time — `is_last_step` therefore never fires `true` on an
        //    early-stopped run's TRUE final step. The monitored loss is NOT
        //    the backstop for that: `monitor_loss` is measured PRE-step
        //    (every micro-batch's loss is read before its window's update
        //    lands), so a NaN/Inf produced by the epoch's final update is
        //    invisible to that epoch's average — it would only show up in
        //    the NEXT epoch's forward, as NaN losses tripping
        //    `process_batch_loss`'s three-strikes divergence guard, and an
        //    early-stopped (or final) epoch has no next epoch, while
        //    `checkpoint_best` is written from the CURRENT, post-update
        //    weights. The backstop is `refuse_nonfinite_params` at the epoch
        //    boundary (one host read per epoch, never per step): a
        //    non-finite trainable parameter is a typed refusal before
        //    `checkpoint_best` — the adapter `run` restores before saving
        //    the final one — can ever hold it. The diagnostic gap (the
        //    divergence surfacing at the epoch boundary rather than on the
        //    exact step) remains, and is bounded by one epoch.
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
        //    compared against, so a resumed run's last step is the same
        //    horizon a fresh run's would be. A resumed run whose horizon
        //    SHRANK below its restored `global_step` (fewer configured
        //    epochs, a larger accumulation window, or fewer rows than the
        //    run that wrote the checkpoint) never reaches the horizon at
        //    all: every step it takes is past it. `LastStepHorizon` decides
        //    the exact last step with `==` and checks an overshoot ONCE
        //    (the first step past the horizon), so such a run pays one
        //    extra sync, not one per step — see `LastStepHorizon`'s lattice
        //    and the run-level oracles in `last_step_horizon_run_oracles`.
        let total_optimizer_steps = if Self::wants_gradcache_horizon(
            train_loader.is_precomputed(),
            self.gradcache_eligible(),
        ) {
            self.config.epochs
        } else {
            total_steps
        };
        let mut last_step_horizon = LastStepHorizon::new(total_optimizer_steps);

        // Snapshot the trainable variables ONCE, in a DETERMINISTIC (name-sorted)
        // order — `optimizer::sorted_trainable_vars`, never a raw `VarMap::
        // all_vars()` (its `HashMap` iteration order is stable within a process
        // but randomized ACROSS processes by `HashMap`'s default per-process
        // hasher seed, which would otherwise make the clip's f32 fold order —
        // and therefore its last bits — a function of process-launch randomness
        // rather than `self.config.seed`; see that function's own doc, esc-182).
        // `AdamW`'s optimizer state is positional in the order it was built
        // from, so building the optimizer and `trainable_vars` from one
        // snapshot keeps the gradient accumulation, clipping, and the
        // optimizer's moment vector all aligned to the same (now
        // cross-process-stable) parameter order. The cross-process correlation
        // that makes RESUME safe is still `optim_param_names` below — the
        // moments serialize/restore BY NAME, independent of this order too —
        // this change makes the in-process order itself reproducible on top of
        // that, not a replacement for it.
        let trainable_vars = super::optimizer::sorted_trainable_vars(&self.varmap);

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
        // `(epoch, avg_train_loss)` / `(epoch, avg_val_loss)` rows, one push per
        // epoch actually run — folded into `metrics_json` below as
        // `train_loss_curve` / `val_loss_curve` (issue #441). Mirrors exactly
        // what `tracing::info!("Epoch complete", ...)` below emits for the
        // GPU-capability suite's own `loss_capture` tracing layer to read back
        // (`crates/jammi-ai/tests/gpu_capability/harness.rs::loss_capture`), so
        // the persisted curve and that test harness's captured curve are the
        // SAME numbers, never two independently-computed ones that could drift.
        // `val_loss_curve` only ever grows when this run's `early_stopping_
        // metric` is `ValLoss` (see `avg_val_loss`'s own `None`-sentinel doc
        // below) — an early-stopped run's curves are correctly SHORTER than
        // `self.config.epochs`, since `break` below stops pushing further rows.
        let mut train_loss_curve: Vec<(usize, f64)> = Vec::new();
        let mut val_loss_curve: Vec<(usize, f64)> = Vec::new();
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
            // Accumulated gradients across micro-batches, merged in
            // `accumulate_grads`. Starts (and is reset, at every accumulation
            // window boundary) to `GradStore::default()` rather than `None`
            // — see `EpochState::grads_pending`'s doc for why "is there
            // anything to flush" is tracked by that flag, not by querying
            // this store's contents.
            let mut accumulated_grads = GradStore::default();
            let mut grads_pending = false;
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
                            grads_pending: &mut grads_pending,
                            global_step: &mut global_step,
                        },
                        StepContext {
                            trainable_vars: &trainable_vars,
                            optimizer: &mut optimizer,
                            checkpoint_dir: &checkpoint_dir,
                            checkpoint_interval,
                            lr_horizon: total_steps,
                            last_step_horizon: &mut last_step_horizon,
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
                    &mut last_step_horizon,
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
                            grads_pending: &mut grads_pending,
                            global_step: &mut global_step,
                        },
                        StepContext {
                            trainable_vars: &trainable_vars,
                            optimizer: &mut optimizer,
                            checkpoint_dir: &checkpoint_dir,
                            checkpoint_interval,
                            lr_horizon: total_steps,
                            last_step_horizon: &mut last_step_horizon,
                            batches_per_epoch: train_batches_per_epoch,
                        },
                    )?;
                }
            }

            // Flush any remaining micro-batch gradients that didn't fill a full
            // accumulation window (last partial window of the epoch).
            // `grads_pending` is false here whenever the last window was
            // already flushed at its own boundary (or `process_batch_loss`
            // was never called this epoch, e.g. the GradCache branch above)
            // — see `EpochState::grads_pending`'s doc for why this is a
            // dedicated flag rather than a `GradStore` emptiness check. So
            // this block is never reached on the GradCache arm;
            // `total_optimizer_steps` equals `total_steps` here (see the
            // lattice doc above `total_optimizer_steps`'s definition), used
            // for consistency with the other two call sites rather than
            // because this arm needs the distinction.
            if grads_pending {
                let lr = compute_lr(&self.config, global_step, total_steps);
                optimizer.set_learning_rate(lr);
                // `last_step_horizon` carries the whole run's ACTUAL
                // optimizer-step horizon for the arm this run takes (see the
                // lattice doc above `total_optimizer_steps`), so this names
                // the run's actual final optimizer step, not just this
                // epoch's — the non-finite check must not be skippable by a
                // run shorter than `DEFAULT_NORM_CHECK_INTERVAL` steps (see
                // `clip_and_step`'s doc).
                let is_last_step = last_step_horizon.is_last_step(global_step + 1);
                clip_and_step(
                    &mut optimizer,
                    &trainable_vars,
                    &mut accumulated_grads,
                    self.config.max_grad_norm,
                    DEFAULT_NORM_CHECK_INTERVAL,
                    global_step + 1,
                    is_last_step,
                )?;
                global_step += 1;
                // The flush is an optimizer step like any other, so it sits
                // on the same step-checkpoint cadence as the in-window steps
                // in `process_batch_loss` and the GradCache arm — the
                // trailing step of an epoch whose batch count is not a
                // multiple of `grad_accum` must not be the one step that
                // skips its checkpoint.
                if checkpoint_interval > 0 && global_step.is_multiple_of(checkpoint_interval) {
                    self.save_checkpoint(&checkpoint_dir, global_step)?;
                }
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
                    // Disable dropout for the validation pass — the exact
                    // `set_training(false)` / `evaluate` / `set_training(true)`
                    // sequence this used to spell out inline, now shared with
                    // `evaluate_held_out` (H1, unit 63) via
                    // `with_dropout_disabled` so there is exactly one place
                    // that bracket can get wrong. Pure structural refactor:
                    // same three operations in the same order, so `evaluate`'s
                    // return value — and every pinned value downstream of it
                    // (early stopping, `checkpoint_best`) — is unchanged.
                    Some(self.with_dropout_disabled(|loop_| loop_.evaluate(&val_loader))?)
                }
            };

            // Accumulate this epoch's row into the run-level curves BEFORE the
            // `tracing::info!` below, so both read the exact same `avg_train_loss`
            // / `avg_val_loss` values `loss_capture`'s tracing layer captures
            // from that event.
            train_loss_curve.push((epoch, avg_train_loss));
            if let Some(v) = avg_val_loss {
                val_loss_curve.push((epoch, v));
            }

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

            // A non-finite trainable parameter at the epoch boundary is a
            // typed refusal BEFORE the early-stopping decision can write it
            // as `checkpoint_best` — see the method's doc for why the
            // monitored loss cannot stand in for this check.
            Self::refuse_nonfinite_params(&trainable_vars, epoch)?;

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

                // Per-epoch adapter checkpoint (unit 348) — a full loadable
                // adapter under this attempt's own publish prefix. Same
                // lease gate as the resume checkpoint immediately above: a
                // zombie attempt must not keep publishing (or pruning)
                // checkpoints once its lease is gone, since the worker's
                // finalize CAS will never see them.
                self.save_epoch_checkpoint(&checkpoint_dir, epoch)?;
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
        // `{"epoch": .., "loss": ..}` rows, one per epoch actually run — see
        // `train_loss_curve`/`val_loss_curve`'s own doc above for why these are
        // the SAME numbers `loss_capture`'s tracing layer reads. `val_loss_
        // curve` is folded in only when validation actually ran this attempt
        // (`EarlyStoppingMetric::ValLoss`) — a `TrainLoss`-monitored run never
        // measures a held-out loss, so an empty array there would be a
        // fabricated "measured zero epochs" rather than the honest "not
        // applicable to this run" the field's absence states instead.
        let curve_json = |curve: &[(usize, f64)]| -> serde_json::Value {
            serde_json::Value::Array(
                curve
                    .iter()
                    .map(|(epoch, loss)| serde_json::json!({"epoch": epoch, "loss": loss}))
                    .collect(),
            )
        };
        let mut metrics = serde_json::json!({
            "final_loss": best_val_loss,
            "early_stopping_metric": early_stopping_metric_label,
            "total_steps": global_step,
            "started_at": started_at,
            "completed_at": completed_at,
            "train_loss_curve": curve_json(&train_loss_curve),
        });
        if !val_loss_curve.is_empty() {
            metrics["val_loss_curve"] = curve_json(&val_loss_curve);
        }
        let metrics_json = metrics.to_string();

        Ok(TrainingResult {
            artifact_dir: artifact_tmp,
            final_loss: best_val_loss,
            total_steps: global_step,
            metrics_json,
            epoch_checkpoints: std::mem::take(&mut self.epoch_checkpoints),
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
        // per-batch anchor queries below. Routed through `Self::set_training` (not
        // `self.target.set_training` directly) so `self.training_mode` — the state
        // `with_dropout_disabled` restores from — never drifts from the target's
        // real mode (audit round 63, finding 2).
        self.set_training(false);
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
        self.set_training(true);
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

    /// Whether `total_optimizer_steps` (in `run`) should use GradCache's
    /// per-epoch horizon (`self.config.epochs`) rather than the
    /// accumulation-window arm's `total_steps`. Pulled out of the `if` at its
    /// one call site into its own pure function so the boolean condition
    /// itself is directly unit-testable in isolation — a secondary to the
    /// GradCache arm of `last_step_run_harness`, which drives `run()` with a
    /// real GradCache-eligible, non-precomputed loader and is what reddens
    /// when the selection is wrong (e.g. `total_steps` on this arm).
    fn wants_gradcache_horizon(is_precomputed: bool, gradcache_eligible: bool) -> bool {
        !is_precomputed && gradcache_eligible
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
        last_step_horizon: &mut LastStepHorizon,
        global_step: usize,
    ) -> Result<f64> {
        use super::gradcache::{gradcache_backward, EncodeGroup};

        let (anchors, positives, negatives) = train_loader.in_batch_negative_texts()?;
        let scale = self.mnrl_scale();
        let has_negatives = negatives.is_some();

        // Dropout off for the whole GradCache region so the two encode passes
        // (and the logging re-encode) agree. Toggled while no encode closure
        // borrows `self`, so it does not collide with the immutable borrows
        // below. Concretely, a WHOLE GradCache epoch trains with LoRA
        // dropout OFF — a behavior difference from the per-batch training
        // path that is deliberate: the two-pass gradient equals the
        // single-pass one only when both passes see the same activations,
        // and dropout off is what makes them agree. Routed through
        // `Self::set_training` (audit round 63, finding 2) so `self.training_mode`
        // never drifts from the target's real mode.
        self.set_training(false);

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

        self.set_training(true);
        let (grads, loss_val) = outcome?;
        #[cfg(test)]
        let grads = self.poke_after_backward(global_step + 1, grads, trainable_vars)?;
        let mut grads = grads;

        // GradCache takes exactly ONE optimizer step per EPOCH, not one per
        // accumulation window — the caller builds `last_step_horizon` from
        // `self.config.epochs` on this arm specifically BECAUSE of that (see
        // the lattice doc at its call site in `run`, above the definition of
        // `total_optimizer_steps`). The accumulation-window arm's horizon
        // (`ceil(batches / grad_accum) * epochs`) overcounts GradCache's
        // per-epoch step count whenever an epoch holds more than one
        // memory-bounded chunk, which would make the run's actual last step
        // look like an ordinary one and silently skip its non-finite check
        // on any GradCache run shorter than `DEFAULT_NORM_CHECK_INTERVAL`
        // steps — every multi-epoch GradCache run, at one step per epoch.
        let is_last_step = last_step_horizon.is_last_step(global_step + 1);
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

                let effective_max = self.config.max_seq_length.min(encoder.max_seq_length());
                // esc-076 amendment (adversarial-audit round 2, campaign
                // #443, item 3; r3 finding B4 corrected the bound below —
                // see `tokenize_natural_width`'s own doc for the full
                // argument, never shortened to "eval runs infrequently"
                // again, that bounds passes, not distinct shapes):
                // bucket-UP padding is a TRAINING-STEP-only concern —
                // it bounds the allocator's distinct-shape count against an
                // UNBOUNDED-across-the-run sequence of per-step batches.
                // Eval (`self.training_mode == false`, set by
                // `with_dropout_disabled`'s bracket around
                // `evaluate`/`evaluate_held_out`) instead re-presents the
                // SAME deterministic held-out partition's width sequence on
                // every pass, so its distinct-shape contribution is paid
                // ONCE per run, not per-step — bucketing it up buys no
                // allocator-stability benefit that determinism doesn't
                // already provide, at a real memory cost (the measured OOM
                // `tokenize_natural_width`'s own doc cites). esc-076 stays
                // OPEN on the residual, caller-dependent axis (a held-out
                // split wide/varied enough to itself present many distinct
                // widths in that one-time set) — this exemption restores
                // the pre-esc-076 baseline for eval, it does not newly
                // bound that axis.
                let (encoding, rows, cols) = if self.training_mode {
                    tokenize_and_bucket(tokenizer, texts, effective_max)?
                } else {
                    tokenize_natural_width(tokenizer, texts, effective_max)?
                };

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
    /// **Graph retention.** In production `anchor`/`positive`/`negative`
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
    /// implementation changes to something that isn't a plain `+` — which
    /// also makes that second `detach` untestable by construction, see the
    /// fold's own comment). The
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
            // accumulator (see the doc's "Graph retention" section) —
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
            // Untestable by construction: both operands are already detached
            // leaves, so candle's `BackpropOp::new2` gives their sum no `op`
            // and this `detach()` is observationally the identity — no
            // fixture can make deleting it visible (a mutant that removes it
            // survives by design, not by a coverage gap). Kept as the belt
            // to `batch_mean_f32`'s suspenders: it is what keeps the
            // accumulator a leaf if this fold ever stops being a plain `+`
            // over detached operands.
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

    /// Run the [`Self::after_backward`] test seam (if installed) over the
    /// gradients optimizer step `step` (1-based) is about to consume.
    #[cfg(test)]
    fn poke_after_backward(
        &mut self,
        step: usize,
        mut grads: GradStore,
        trainable_vars: &[Var],
    ) -> Result<GradStore> {
        if let Some(hook) = self.after_backward.as_mut() {
            hook(step, &mut grads, trainable_vars)?;
        }
        Ok(grads)
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
        let new_grads =
            self.poke_after_backward(*epoch.global_step + 1, new_grads, ctx.trainable_vars)?;

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

        // Merge new_grads into the running accumulator. This micro-batch has
        // now occupied its accumulation slot regardless of whether it
        // contributed any actual gradient entries — see
        // `EpochState::grads_pending`'s doc.
        accumulate_grads(epoch.accumulated_grads, new_grads, ctx.trainable_vars)?;
        *epoch.grads_pending = true;

        // Optimizer step every N micro-batches.
        if (*epoch.batch_count).is_multiple_of(self.config.gradient_accumulation_steps) {
            let lr = compute_lr(&self.config, *epoch.global_step, ctx.lr_horizon);
            ctx.optimizer.set_learning_rate(lr);

            // Only the accumulation-window arm reaches this function (the
            // GradCache arm steps in `run_gradcache_epoch`), and on that arm
            // the LR horizon and the last-step horizon are the same count —
            // see `StepContext`'s field docs for why they are still two
            // fields.
            debug_assert_eq!(
                ctx.lr_horizon, ctx.last_step_horizon.horizon,
                "accumulation-window arm: the LR and last-step horizons must agree"
            );
            // See the flush-window call site's doc: this names the run's
            // actual final optimizer step, not just this epoch's.
            let is_last_step = ctx.last_step_horizon.is_last_step(*epoch.global_step + 1);
            clip_and_step(
                ctx.optimizer,
                ctx.trainable_vars,
                epoch.accumulated_grads,
                self.config.max_grad_norm,
                DEFAULT_NORM_CHECK_INTERVAL,
                *epoch.global_step + 1,
                is_last_step,
            )?;
            // Reset to a fresh, empty accumulator for the next window —
            // `mem::take` swaps in `GradStore::default()` and discards the
            // just-consumed values, the same "no Option needed" seed the
            // epoch starts with.
            std::mem::take(epoch.accumulated_grads);
            *epoch.grads_pending = false;

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

    /// Per-example decomposition of [`Self::compute_loss`] (H1, unit 63,
    /// CONTRACT H1): the seam [`Self::evaluate_held_out`] calls this instead
    /// of `compute_loss` to get one loss per row, BEFORE any batch-mean
    /// reduction, rather than `compute_loss`'s single reduced scalar.
    ///
    /// This is a NEW, independent computation path. It calls none of
    /// `compute_loss`'s own code and does not touch `compute_loss`,
    /// `evaluate`, or any of the batch-mean-reducing free functions
    /// (`mnrl_loss` / `triplet_loss` / `cosent_loss` / `angle_loss` /
    /// `cosine_mse_loss`) — every one of those stays byte-for-byte
    /// unchanged (see the module doc's "example-mean is a NEW quantity"
    /// note on [`jammi_wire::fine_tune::HeldOutLoss`]). It reuses only the
    /// pure, stateless building blocks those functions also call
    /// (`l2_normalize_rows`, `cosine_similarity`, `contiguous_matmul`),
    /// which cannot perturb any pinned training value because calling a
    /// pure function a second time from new code changes nothing about its
    /// existing call sites.
    ///
    /// Only the batch kinds with a mathematically well-defined per-row
    /// decomposition are supported:
    /// - `Pairs` (MNRL, always — the only objective this shape trains):
    ///   row-direction (and, symmetrically, column-direction) NLL against
    ///   the diagonal positive. Batch-coupled by construction — a row's
    ///   loss depends on every other row's positive sharing the batch.
    /// - `Triplet`: the MNRL variant (row NLL with the explicit negative
    ///   appended as an extra similarity column, also batch-coupled) when
    ///   `MultipleNegativesRanking` is configured, else the row-independent
    ///   margin loss `max(0, cos(a,n) − cos(a,p) + margin)`.
    /// - `Contrastive` with `CosineMse` (row-independent squared error).
    /// - `Classification` (row-independent cross-entropy).
    ///
    /// `Contrastive` with `CoSent`/`AnglE` is a typed refusal: both fold
    /// EVERY valid pair in the batch into ONE scalar via a pairwise
    /// log-sum-exp ordering ([`pairwise_ordering_loss`]) — there is no row
    /// `i` whose loss is independent of every other row, so inventing a
    /// per-row split would not be a decomposition of the real objective, it
    /// would be a different number scored under a different name. `Ner`
    /// (token-level; this seam has not chosen a per-example convention for
    /// it) and `Regression` (S18's distributional head — a different
    /// objective family, out of this unit's scope) are typed refusals for
    /// the same reason: a fabricated decomposition is worse than an honest
    /// refusal.
    fn compute_loss_per_example(&self, batch: &super::data::TrainingBatch) -> Result<Vec<f64>> {
        match batch {
            super::data::TrainingBatch::Pairs { anchors, positives } => self
                .matryoshka_wrap_per_example(&[anchors, positives], &|dims| {
                    mnrl_loss_per_example(&dims[0], &dims[1], None, self.mnrl_scale(), true)
                }),
            super::data::TrainingBatch::Triplet {
                anchor,
                positive,
                negative,
            } => match self.config.embedding_loss {
                Some(super::EmbeddingLoss::MultipleNegativesRanking { .. }) => self
                    .matryoshka_wrap_per_example(&[anchor, positive, negative], &|dims| {
                        mnrl_loss_per_example(
                            &dims[0],
                            &dims[1],
                            Some(&dims[2]),
                            self.mnrl_scale(),
                            true,
                        )
                    }),
                _ => self.matryoshka_wrap_per_example(&[anchor, positive, negative], &|dims| {
                    self.triplet_loss_per_example(&dims[0], &dims[1], &dims[2])
                }),
            },
            super::data::TrainingBatch::Contrastive {
                embeddings_a,
                embeddings_b,
                scores,
            } => match self.config.embedding_loss {
                Some(super::EmbeddingLoss::CosineMse) => self
                    .matryoshka_wrap_per_example(&[embeddings_a, embeddings_b], &|dims| {
                        cosine_mse_loss_per_example(&dims[0], &dims[1], scores)
                    }),
                Some(super::EmbeddingLoss::CoSent) | None => Err(JammiError::FineTune(
                    "evaluate_held_out: CoSENT is a pairwise-ordering objective over every \
                     valid pair in the batch (one log-sum-exp over ALL pairs) — it has no \
                     per-row decomposition. Choose CosineMse for a per-example held-out \
                     graded-pair eval, or Pairs/MultipleNegativesRanking (or Triplet) for a \
                     batch-coupled/row-independent one."
                        .into(),
                )),
                Some(super::EmbeddingLoss::AnglE) => Err(JammiError::FineTune(
                    "evaluate_held_out: AnglE is a pairwise-ordering objective over every \
                     valid pair in the batch, the same as CoSENT — it has no per-row \
                     decomposition."
                        .into(),
                )),
                Some(super::EmbeddingLoss::MultipleNegativesRanking { .. }) => {
                    Err(JammiError::FineTune(
                        "MultipleNegativesRanking is an in-batch-negative objective over \
                         (anchor, positive) rows; it cannot score a graded (text_a, text_b, \
                         score) batch."
                            .into(),
                    ))
                }
                Some(super::EmbeddingLoss::Triplet { .. }) => Err(JammiError::FineTune(
                    "Triplet loss needs (anchor, positive, negative) rows; it cannot score a \
                     graded (text_a, text_b, score) batch."
                        .into(),
                )),
            },
            super::data::TrainingBatch::Classification { embeddings, labels } => {
                let logits = self.classify(embeddings)?;
                cross_entropy_per_row(&logits, labels)
            }
            super::data::TrainingBatch::Ner { .. } => Err(JammiError::FineTune(
                "evaluate_held_out: NER's natural unit is a token, not a held-out example — \
                 this seam does not define a per-example convention for NER in v1."
                    .into(),
            )),
            super::data::TrainingBatch::Regression { .. } => Err(JammiError::FineTune(
                "evaluate_held_out: Regression (S18) trains a different distributional-head \
                 objective family; a per-example held-out decomposition for it is out of \
                 this unit's scope."
                    .into(),
            )),
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
    /// [`dispatch_contrastive_loss`] — the CoSENT default is the free
    /// [`cosent_loss`]; none of the graded objectives need `self`, they close
    /// purely over the similarity/score tensors.
    fn contrastive_loss(&self, emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
        dispatch_contrastive_loss(
            self.config.embedding_loss,
            emb_a,
            emb_b,
            scores,
            &cosent_loss,
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

    /// Per-example counterpart of [`Self::matryoshka_wrap`] (H1, unit 63):
    /// thin wrapper over the free [`matryoshka_sum_per_example`], used by
    /// [`Self::compute_loss_per_example`] exactly as `matryoshka_wrap` is
    /// used by `compute_loss`.
    fn matryoshka_wrap_per_example(
        &self,
        embeddings: &[&Tensor],
        objective: &dyn Fn(Vec<Tensor>) -> Result<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        matryoshka_sum_per_example(&self.config.matryoshka_dims, embeddings, objective)
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

    /// Token-level cross-entropy loss for NER with `ignore_index=-100`
    /// semantics: positions whose original label is `-100` contribute
    /// **nothing** to the loss value or the gradient, and the mean is taken
    /// over the non-ignored positions only (not `batch * seq_len`). This
    /// matches PyTorch's `torch.nn.CrossEntropyLoss(ignore_index=-100)`
    /// under the default `reduction="mean"`, whose docs specify that the
    /// mean divides by the count of targets that are *not* `ignore_index`
    /// (<https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>).
    ///
    /// Candle's `cross_entropy` has no `ignore_index` argument, so this
    /// gathers the non-ignored rows with [`Tensor::index_select`] *before*
    /// calling it, rather than clamping `-100` to a valid class and masking
    /// the loss value afterwards. That distinction matters for the
    /// gradient, not just the value: because the ignored rows never enter
    /// the cross-entropy computation graph, candle's `index_select`
    /// backward (`grads.or_insert` zero-fills the full-shape gradient, then
    /// `index_add`s only into the selected rows — see
    /// `candle_core::backprop`) leaves the ignored rows' gradient at an
    /// *exact* `0.0`. A "compute-then-multiply-by-zero-mask" formulation
    /// would only get a value of `0.0` for those rows in the forward pass;
    /// the backward pass would still evaluate (and could still overflow to
    /// `inf`/`NaN`) the per-row cross-entropy term before the zero multiply,
    /// which is not exactness, only smallness.
    ///
    /// If every position in the batch is ignored, this returns a `0.0`
    /// scalar loss (no rows to select, hence no gradient contribution),
    /// matching PyTorch's handling of an all-`ignore_index` batch under
    /// `reduction="mean"` (returning `0` rather than propagating a `0/0`
    /// `NaN`).
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

        // Read labels to the host to build the keep-list: -100 marks a
        // position to exclude entirely, per ignore_index=-100 semantics.
        let label_values = flat_labels
            .to_dtype(candle_core::DType::I64)
            .map_err(|e| JammiError::FineTune(format!("NER labels i64: {e}")))?
            .to_vec1::<i64>()
            .map_err(|e| JammiError::FineTune(format!("NER labels to_vec1: {e}")))?;

        let mut keep_indices: Vec<u32> = Vec::with_capacity(label_values.len());
        let mut keep_labels: Vec<u32> = Vec::with_capacity(label_values.len());
        for (idx, &label) in label_values.iter().enumerate() {
            if label != -100 {
                keep_indices.push(idx as u32);
                keep_labels.push(label as u32);
            }
        }

        if keep_indices.is_empty() {
            // Every position ignored: match PyTorch's CrossEntropyLoss
            // (reduction="mean") returning 0.0 rather than a NaN 0/0 mean.
            return Tensor::zeros((), logits.dtype(), logits.device())
                .map_err(|e| JammiError::FineTune(format!("NER all-ignored zero loss: {e}")));
        }

        let n_keep = keep_indices.len();
        let device = flat_logits.device();
        let index_tensor = Tensor::from_vec(keep_indices, n_keep, device)
            .map_err(|e| JammiError::FineTune(format!("NER keep indices: {e}")))?;
        let label_tensor = Tensor::from_vec(keep_labels, n_keep, device)
            .map_err(|e| JammiError::FineTune(format!("NER keep labels: {e}")))?;

        // Gather only the non-ignored rows before computing cross-entropy so
        // the ignored rows never enter the graph (see the doc comment above
        // for why this, not a zero-mask multiply, makes their gradient
        // exactly 0.0).
        let selected_logits = flat_logits
            .index_select(&index_tensor, 0)
            .map_err(|e| JammiError::FineTune(format!("NER index_select logits: {e}")))?;

        candle_nn::loss::cross_entropy(&selected_logits, &label_tensor)
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

    /// Per-example decomposition of [`Self::triplet_loss`] (H1, unit 63):
    /// `max(0, cos(anchor, negative) - cos(anchor, positive) + margin)` per
    /// row, read back to host before the mean `triplet_loss` takes instead.
    /// Row-independent by construction — the margin objective on row `i`
    /// never reads any other row — so this is a genuine decomposition, not
    /// an approximation of one. Reuses the SAME [`cosine_similarity`] calls
    /// `triplet_loss` makes; does not touch `triplet_loss` itself.
    fn triplet_loss_per_example(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negative: &Tensor,
    ) -> Result<Vec<f64>> {
        let margin = match self.config.embedding_loss {
            Some(super::EmbeddingLoss::Triplet { margin }) => margin,
            _ => 0.3,
        };

        let pos_sim = cosine_similarity(anchor, positive)?;
        let neg_sim = cosine_similarity(anchor, negative)?;

        let diff = ((&neg_sim - &pos_sim)
            .map_err(|e| JammiError::FineTune(format!("Triplet per-example diff: {e}")))?
            + margin)
            .map_err(|e| JammiError::FineTune(format!("Triplet per-example margin: {e}")))?;

        let host = diff
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("Triplet per-example dtype: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| JammiError::FineTune(format!("Triplet per-example to_vec1: {e}")))?;

        Ok(host.into_iter().map(|v| v.max(0.0) as f64).collect())
    }

    /// Set `self.target`'s training/dropout mode AND `self.training_mode` in
    /// lockstep — the single place these two are allowed to diverge is
    /// nowhere; every production call that toggles the target's training
    /// flag must go through here (audit round 63, finding 2) so
    /// [`Self::with_dropout_disabled`] can trust `self.training_mode` as an
    /// accurate mirror of the target's actual state at any call site.
    fn set_training(&mut self, training: bool) {
        self.target.set_training(training);
        self.training_mode = training;
    }

    /// Bracket `f` with dropout disabled, mirroring the
    /// `set_training(false)` / call / `set_training(true)` sequence
    /// [`Self::run`] wraps its [`Self::evaluate`] call in (R3). Shared by
    /// `run` and [`Self::evaluate_held_out`] (H1, unit 63) so the held-out
    /// seam cannot be called dropout-hot — it goes through the SAME bracket
    /// the existing validation pass uses, rather than a second copy of it,
    /// so there is exactly one place that can get this wrong.
    ///
    /// Audit round 63, finding 2 (fixed): the pre-fix version propagated `?`
    /// on `f`'s `Err` BEFORE the restore ran, and always restored to `true`
    /// rather than whatever mode the trainer was actually in beforehand. On
    /// the public, typed-refusal-bearing [`Self::evaluate_held_out`] seam,
    /// both halves of that were live bugs: a refusal (e.g. the "not a
    /// multiple of batch_size" or "non-finite loss" checks) left the trainer
    /// permanently eval-mode — every subsequent training step silently
    /// trained with dropout OFF — and calling this seam on a trainer that
    /// was not in training mode to begin with (e.g. an inference-only /
    /// held-out-only handle) would flip it INTO training mode as a side
    /// effect of a read-only evaluation call.
    ///
    /// Fixed per the repo's own restore-on-both-arms idiom (the
    /// `mine_hard_negative_loader` / `run_gradcache_epoch` brackets): `f`'s
    /// result is captured into a local binding — never `?`-propagated
    /// directly — so the restore always runs before the function returns,
    /// on EITHER arm. The restore target is `was_training`, captured from
    /// `self.training_mode` before dropout is disabled, not a hard-coded
    /// `true` — so this bracket is now a strict save/disable/restore, never
    /// an implicit "and also force training on."
    fn with_dropout_disabled<T>(&mut self, f: impl FnOnce(&Self) -> Result<T>) -> Result<T> {
        let was_training = self.training_mode;
        self.set_training(false);
        let result = f(self);
        self.set_training(was_training);
        result
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

    /// The public per-pair held-out evaluation seam (H1, unit 63, CONTRACT
    /// H1). Scores the current weights against a committed held-out split
    /// and returns a [`jammi_wire::fine_tune::HeldOutLoss`]: one loss per
    /// example plus the example-mean, count, tie-fraction, and the batch
    /// partition identity — the quantity C16/H2's paired sign test consumes
    /// (`d_i` = this method's `mean`, paired by seed, at the final epoch).
    ///
    /// ## The example-mean is a NEW quantity, not `evaluate`'s batch-mean
    /// `Self::evaluate` is untouched by this unit, byte for byte (diff the
    /// method above this one). Its private, monitoring-only batch-mean
    /// semantics — early stopping, `checkpoint_best`, every pinned value
    /// that reads it — keep reading exactly what they read today. This
    /// method computes an entirely separate per-example decomposition via
    /// `Self::compute_loss_per_example` and reduces it with its OWN plain
    /// arithmetic mean over `per_example`, independent of `evaluate`'s
    /// mean-of-per-batch-means.
    ///
    /// ## The batch partition IS identity
    /// `example_ids` must be the SAME length as `val_loader`'s row count, in
    /// the EXACT order the loader will yield rows, and that count MUST be a
    /// multiple of `self.config.batch_size` (v2 delta 2) — the committed
    /// held-out fixture is sized this way so every batch is FULL (no ragged
    /// final batch) and, for an MNRL objective, every example sees exactly
    /// `batch_size - 1` in-batch negatives (see
    /// [`jammi_wire::fine_tune::HeldOutLoss::in_batch_negatives_per_example`]
    /// for the non-MNRL objectives, which have none). The partition rule is:
    /// walk `example_ids` (and the
    /// loader's rows) in the given fixed order and group every consecutive
    /// `batch_size` of them into one batch — no shuffling, no re-ordering,
    /// no random split. Two calls over the same `(val_loader, example_ids,
    /// self.config.batch_size)` therefore produce the IDENTICAL partition,
    /// and — since dropout is disabled here and the forward pass is
    /// otherwise deterministic on CPU — the identical `HeldOutLoss`.
    ///
    /// [`jammi_wire::fine_tune::HeldOutLoss::batch_partition_sha256`] is the
    /// SHA-256 hex digest of the partition's canonical form: the JSON
    /// serialization of `Vec<Vec<String>>`, one inner vec per batch holding
    /// that batch's example ids in order — e.g. `[["id-a","id-b"],
    /// ["id-c","id-d"]]` for a 4-example, `batch_size = 2` split. Hashing
    /// the GROUPED (not flat) id list puts the batch BOUNDARIES in the
    /// digest, not just the flat id order, matching the struct doc's "which
    /// examples shared a batch, and in what order."
    ///
    /// ## Dropout bracket
    /// Delegates to `Self::with_dropout_disabled` — the SAME bracket
    /// `run()` wraps its `evaluate()` call in — so calling this seam
    /// mid-training cannot draw a dropout mask or perturb the training RNG
    /// stream (see the `no_rng_perturbation` test in this module).
    ///
    /// ## Typed refusals
    /// - Empty held-out set (`val_loader.is_empty()` or `example_ids` empty).
    /// - `example_ids.len()` not a multiple of `self.config.batch_size`.
    /// - A produced batch whose row count is not exactly
    ///   `self.config.batch_size`, or a batch sequence that consumes fewer
    ///   or more ids than `example_ids` supplies (example-kind mismatch
    ///   between the committed id partition and the batch actually
    ///   produced — e.g. the loader and the id list disagree about how rows
    ///   group). `val_loader.len()` is NOT compared directly against
    ///   `example_ids.len()` up front: for a precomputed loader `len()` is
    ///   the BATCH count, not the row count (see
    ///   [`super::data::TrainingDataLoader::len`]), so the row-level check
    ///   happens per batch, as rows are actually produced.
    /// - A batch kind `Self::compute_loss_per_example` does not support
    ///   (CoSENT/AnglE/NER/Regression — see that method's doc).
    /// - A non-finite per-example loss: refused rather than folded into the
    ///   mean, because a NaN example silently dropped or silently averaged
    ///   in would corrupt every downstream paired-sign-test cell with no
    ///   visible signal.
    pub fn evaluate_held_out(
        &mut self,
        val_loader: &TrainingDataLoader,
        example_ids: &[String],
    ) -> Result<super::HeldOutLoss> {
        self.with_dropout_disabled(|loop_| loop_.evaluate_held_out_inner(val_loader, example_ids))
    }

    /// The read-only body [`Self::evaluate_held_out`] runs inside
    /// [`Self::with_dropout_disabled`]'s bracket. Split out so the bracket
    /// (which needs `&mut self`) and the computation (which only needs
    /// `&self`, exactly like `evaluate`) stay separate — mirroring how
    /// `evaluate` itself takes `&self` and its caller in `run` owns the
    /// surrounding `&mut self` bracket.
    /// The true in-batch-negative count [`Self::compute_loss_per_example`]
    /// scored this batch's rows against — objective-aware (audit round 63,
    /// finding 6: the pre-fix version reported `batch_size - 1`
    /// UNCONDITIONALLY, which is only correct for the MNRL objectives, and
    /// silently mis-described every Triplet-margin / CosineMse /
    /// Classification held-out evaluation as having `batch_size - 1`
    /// negatives it never actually scored against).
    ///
    /// MNRL scores every row against every OTHER row's positive sharing the
    /// batch — `Pairs` (the only objective that shape trains, per
    /// [`Self::compute_loss_per_example`]'s own doc), always; `Triplet` only
    /// when `MultipleNegativesRanking` is configured (mirroring
    /// `compute_loss_per_example`'s own dispatch on `self.config.embedding_loss`
    /// for that batch kind) — `batch_size - 1` negatives either way. Every
    /// other batch kind this seam supports (`Triplet` margin,
    /// `Contrastive`/`CosineMse`, `Classification`) scores each row
    /// independently of every other row in the batch: genuinely ZERO
    /// in-batch negatives, not an approximation of MNRL's count. `Ner` and
    /// `Regression` are typed refusals earlier in
    /// `compute_loss_per_example` and never reach here in a successful call;
    /// they fold to `0` for an exhaustive match rather than a wildcard.
    fn in_batch_negatives_for(
        &self,
        batch: &super::data::TrainingBatch,
        batch_size: usize,
    ) -> usize {
        match batch {
            super::data::TrainingBatch::Pairs { .. } => batch_size.saturating_sub(1),
            super::data::TrainingBatch::Triplet { .. } => match self.config.embedding_loss {
                Some(super::EmbeddingLoss::MultipleNegativesRanking { .. }) => {
                    batch_size.saturating_sub(1)
                }
                _ => 0,
            },
            super::data::TrainingBatch::Contrastive { .. }
            | super::data::TrainingBatch::Classification { .. }
            | super::data::TrainingBatch::Ner { .. }
            | super::data::TrainingBatch::Regression { .. } => 0,
        }
    }

    fn evaluate_held_out_inner(
        &self,
        val_loader: &TrainingDataLoader,
        example_ids: &[String],
    ) -> Result<super::HeldOutLoss> {
        if val_loader.is_empty() || example_ids.is_empty() {
            return Err(JammiError::FineTune(
                "evaluate_held_out: empty held-out set — refusing to fabricate a HeldOutLoss \
                 over zero examples"
                    .into(),
            ));
        }
        let batch_size = self.config.batch_size;
        if batch_size == 0 || !example_ids.len().is_multiple_of(batch_size) {
            return Err(JammiError::FineTune(format!(
                "evaluate_held_out: {} held-out examples is not a multiple of batch_size {} \
                 — the committed held-out fixture must be sized so every batch is full, \
                 fixing every example at the same in-batch-negative count (for an MNRL \
                 objective, batch_size - 1; see HeldOutLoss::in_batch_negatives_per_example)",
                example_ids.len(),
                batch_size
            )));
        }

        let mut per_example: Vec<super::ExampleLoss> = Vec::with_capacity(example_ids.len());
        let mut id_batches: Vec<Vec<String>> = Vec::new();
        let mut offset = 0usize;
        // Objective-aware in-batch-negative count (finding 6), pinned once from
        // the first batch and cross-checked against every subsequent one — a
        // held-out set is scored under a single, homogeneous objective, so a
        // second batch reporting a different count would mean the loader mixed
        // batch kinds mid-split, which this refuses rather than silently
        // reporting whichever count happened to be seen last.
        let mut in_batch_negatives: Option<usize> = None;

        let mut consume = |batch: super::data::TrainingBatch| -> Result<()> {
            let n = batch_row_count(&batch)?;
            if n != batch_size {
                return Err(JammiError::FineTune(format!(
                    "evaluate_held_out: a held-out batch has {n} rows but config.batch_size \
                     is {batch_size} — every held-out batch must be exactly batch_size rows \
                     (kind mismatch between the committed id partition and the batch actually \
                     produced)"
                )));
            }
            if offset + n > example_ids.len() {
                return Err(JammiError::FineTune(format!(
                    "evaluate_held_out: the batch partition needs at least {} example ids \
                     but only {} were supplied — the id list and the loader disagree about \
                     how many rows the held-out split has (kind mismatch)",
                    offset + n,
                    example_ids.len()
                )));
            }
            let ids = &example_ids[offset..offset + n];
            offset += n;
            let losses = self.compute_loss_per_example(&batch)?;
            if losses.len() != n {
                return Err(JammiError::FineTune(format!(
                    "evaluate_held_out: internal: {} per-example losses for a {n}-row batch",
                    losses.len()
                )));
            }
            let this_negatives = self.in_batch_negatives_for(&batch, batch_size);
            match in_batch_negatives {
                None => in_batch_negatives = Some(this_negatives),
                Some(prev) if prev != this_negatives => {
                    return Err(JammiError::FineTune(format!(
                        "evaluate_held_out: internal: in-batch-negative count changed mid \
                         held-out set ({prev} then {this_negatives}) — the held-out split must \
                         be scored under one homogeneous objective/batch kind"
                    )));
                }
                _ => {}
            }
            let mut batch_ids = Vec::with_capacity(n);
            for (id, loss) in ids.iter().zip(losses) {
                if !loss.is_finite() {
                    return Err(JammiError::FineTune(format!(
                        "evaluate_held_out: non-finite loss ({loss}) for held-out example \
                         '{id}' — refusing to fold it into the example-mean"
                    )));
                }
                per_example.push(super::ExampleLoss {
                    example_id: id.clone(),
                    loss,
                });
                batch_ids.push(id.clone());
            }
            id_batches.push(batch_ids);
            Ok(())
        };

        if val_loader.is_precomputed() {
            for batch in val_loader.batches(batch_size)? {
                consume(batch?)?;
            }
        } else {
            for chunk in val_loader.text_chunks(batch_size) {
                let batch = self.encode_chunk(&chunk)?;
                consume(batch)?;
            }
        }

        if offset != example_ids.len() {
            return Err(JammiError::FineTune(format!(
                "evaluate_held_out: internal: consumed {offset} of {} example ids — the \
                 loader produced fewer rows than the id list promised",
                example_ids.len()
            )));
        }

        let count = per_example.len();
        let sum: f64 = per_example.iter().map(|e| e.loss).sum();
        let mean = sum / count as f64;
        // The tie/hinge-floor value is 0.0 for every objective this seam
        // supports (MNRL's saturated in-batch cross-entropy, the margin
        // triplet's `max(0, ·)`, and cosine-MSE/classification's exact
        // residual match) — see `cross_entropy_per_row`'s doc for why MNRL
        // rounds to EXACTLY 0.0 (not merely close to it) on a saturated row.
        let tie_count = per_example.iter().filter(|e| e.loss == 0.0).count();
        let tie_fraction = tie_count as f64 / count as f64;

        let partition_json = serde_json::to_vec(&id_batches).map_err(|e| {
            JammiError::FineTune(format!("evaluate_held_out: partition hash serialize: {e}"))
        })?;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&partition_json);
        let batch_partition_sha256 = hex::encode(hasher.finalize());

        Ok(super::HeldOutLoss {
            per_example,
            mean,
            count,
            tie_fraction,
            batch_partition_sha256,
            // `unwrap_or(0)` is unreachable in practice: the `offset !=
            // example_ids.len()` refusal above already guarantees at least one
            // batch was consumed successfully whenever this line runs, which is
            // exactly when `in_batch_negatives` was set. Kept as a default
            // rather than an `.expect()` so a future refactor of the loop above
            // fails safe (reports "no in-batch negatives" for an empty pass)
            // rather than panicking.
            in_batch_negatives_per_example: in_batch_negatives.unwrap_or(0),
        })
    }

    /// Save a numbered intra-epoch checkpoint. Weights only — the metadata
    /// JSON is written once when the final adapter lands.
    /// Refuse to write `checkpoint_best` over a non-finite trainable
    /// parameter — the epoch-boundary backstop for a divergence the
    /// monitored loss cannot see.
    ///
    /// `monitor_loss` is measured PRE-step: each micro-batch's loss is read
    /// before its window's update lands, so a NaN/Inf produced by an epoch's
    /// final update is invisible to that epoch's average. It would only
    /// surface in the NEXT epoch's forward (as NaN losses tripping the
    /// three-strikes divergence guard in `process_batch_loss`) — and on the
    /// run's final or early-stopped epoch there is no next epoch, while
    /// `checkpoint_best` (restored before the final adapter is saved) is
    /// written from the CURRENT weights, post-update. Without this check a
    /// divergence on such an epoch's last update — off the grad-norm check's
    /// cadence — would be published as the run's result.
    ///
    /// Cost: one device→host read per epoch boundary (a boundary that
    /// already syncs for the epoch's loss and sim stats), never per step.
    /// The sum of every trainable value is folded into one device scalar
    /// (NaN and ±Inf both propagate through `+`; a sum of sane-magnitude
    /// finite weights cannot overflow `f32`), then read once.
    fn refuse_nonfinite_params(trainable_vars: &[Var], epoch: usize) -> Result<()> {
        let mut total: Option<Tensor> = None;
        for var in trainable_vars {
            let t: &Tensor = var;
            let s = t
                .to_dtype(DType::F32)
                .and_then(|t| t.sum_all())
                .map_err(|e| JammiError::FineTune(format!("checkpoint_best param sum: {e}")))?;
            total = Some(match total {
                None => s,
                Some(acc) => (&acc + &s).map_err(|e| {
                    JammiError::FineTune(format!("checkpoint_best param fold: {e}"))
                })?,
            });
        }
        let Some(total) = total else {
            return Ok(());
        };
        let v: f32 = total
            .to_scalar()
            .map_err(|e| JammiError::FineTune(format!("checkpoint_best param read: {e}")))?;
        if !v.is_finite() {
            return Err(JammiError::FineTune(format!(
                "checkpoint_best: non-finite trainable parameter after epoch {epoch} \
                 (parameter sum {v}) — refusing to save or publish it"
            )));
        }
        Ok(())
    }

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

    /// Build a FULL LOADABLE adapter's `(name, bytes)` files at the current
    /// weights — `jammi_lora::save_adapter`'s weights + `SavedAdapter`
    /// metadata output (never the weights-only `save_checkpoint` format) —
    /// via a scratch directory, then read the written files back as bytes for
    /// a `put_artifact` publish. The SAME construction [`Self::run`]'s final
    /// save uses (`named_trainable_weights` + the scaler-gated
    /// `regression_form` + `TrainingTarget::saved_adapter`), so an epoch
    /// checkpoint and the final artifact are loadable through the identical
    /// path — this is what makes a checkpoint row resolvable by
    /// `jammi models describe` and loadable for inference (unit 348,
    /// CONTRACT item 2).
    fn checkpoint_adapter_files(&self, scratch_dir: &Path) -> Result<Vec<(String, bytes::Bytes)>> {
        let weights = self.target.named_trainable_weights()?;
        let regression_form = self.target_scaler.map(|_| self.regression_form());
        let saved = self
            .target
            .saved_adapter(&self.config, self.target_scaler, regression_form);
        jammi_lora::save_adapter(scratch_dir, &weights, &saved)
            .map_err(|e| JammiError::FineTune(format!("Save epoch checkpoint adapter: {e}")))?;

        let mut files = Vec::new();
        for entry in std::fs::read_dir(scratch_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_bytes = std::fs::read(entry.path())?;
            files.push((name, bytes::Bytes::from(file_bytes)));
        }
        Ok(files)
    }

    /// Publish a full loadable adapter checkpoint for the just-completed
    /// `epoch` under the attempt-unique prefix
    /// `{job_id}/{worker_id}/{attempt}/checkpoints/epoch_{N}/` (unit 348,
    /// K7) — the same manifest-last `put_artifact` publish protocol the
    /// worker's own final artifact uses, extended with the `checkpoints/
    /// epoch_{N}` segment rather than a bare job-keyed prefix, so a resumed
    /// attempt (which writes its OWN attempt segment) can never collide with
    /// or overwrite a prior attempt's epoch checkpoints. `N` is the 0-based
    /// loop epoch index — consistent with resume semantics, where a resumed
    /// attempt continues from `last_completed_epoch + 1`.
    ///
    /// DISABLED by default (unit 348 F3): a `None`
    /// `config.keep_last_n_checkpoints` — absent on the wire, and every
    /// pre-unit-348 caller's config — is a no-op BEFORE touching the store
    /// or the artifact directory at all. This is the load-bearing
    /// no-regression property: a job that never opts in writes exactly the
    /// bytes and catalog rows it always did, byte-for-byte. `Some(_)` (any
    /// caller that explicitly sets the field, refused at `0` by
    /// `FineTuneConfig::validate`) enables the whole mechanism below,
    /// including a `None` `artifact_store` still being a no-op, mirroring
    /// [`Self::save_resume_checkpoint`] (a trainer-internal test with no
    /// store configured).
    ///
    /// On success, appends `(epoch, artifact_prefix)` to
    /// [`Self::epoch_checkpoints`] and then enforces
    /// `config.keep_last_n_checkpoints`: once the retained count exceeds the
    /// cap, the OLDEST surviving entry's bytes are best-effort deleted from
    /// the store right here — safe, because no catalog row exists for it yet
    /// (the worker's finalize registers rows only for the trailing retention
    /// window, unit 348 F2) — and dropped from the vector ONLY on a
    /// SUCCESSFUL delete. The delete is `.ok()`-inspected (not `?`),
    /// matching every other GC in this codebase (`TrainingWorker::
    /// publish_and_finalize`'s prefix deletes): a transient storage failure
    /// pruning an old, already-superseded checkpoint must never abort the
    /// run — that would fail a job over a housekeeping op unrelated to
    /// whether training itself succeeded. On a FAILED delete the entry is
    /// deliberately LEFT in the vector (not removed) and the retry loop
    /// BREAKS rather than hot-looping on the same failing delete: the next
    /// epoch boundary's call re-enters this same loop and retries the
    /// identical oldest entry first (FIFO order is unchanged by a failed
    /// attempt). A `tracing::warn!` fires on every failed prune, naming the
    /// job/attempt/epoch, so a persistently broken store is never silent —
    /// and `TrainingWorker::publish_and_finalize`'s winner arm is the
    /// backstop that reclaims a persistently-failed prune's bytes at
    /// termination (unit 348 F2) rather than letting it leak forever if this
    /// attempt's retries never succeed before the run ends.
    ///
    /// Uses [`Self::EPOCH_CHECKPOINT_SCRATCH`], ONE scratch subdirectory
    /// reused across every epoch this attempt saves (the `_resume_scratch`
    /// precedent in [`Self::save_resume_checkpoint`]), not a fresh
    /// per-epoch directory: each call's `jammi_lora::save_adapter` fully
    /// overwrites both files there before the immediate upload reads them
    /// back, so nothing from a prior epoch survives into the next upload,
    /// and the run's scratch disk footprint does not grow with epoch count.
    /// The caller has already confirmed the lease is held (`!cancel`), the
    /// same gate [`Self::save_resume_checkpoint`] runs under.
    fn save_epoch_checkpoint(&mut self, checkpoint_dir: &Path, epoch: usize) -> Result<()> {
        if self.config.keep_last_n_checkpoints.is_none() {
            return Ok(());
        }
        let Some(store) = self.artifact_store.clone() else {
            return Ok(());
        };
        let scratch = checkpoint_dir.join(Self::EPOCH_CHECKPOINT_SCRATCH);
        let files = self.checkpoint_adapter_files(&scratch)?;
        let prefix = tokio::runtime::Handle::current().block_on(store.put_epoch_checkpoint(
            &self.job_id,
            &self.worker_id,
            &self.attempt,
            epoch,
            &files,
        ))?;
        self.epoch_checkpoints
            .push((epoch, prefix.as_str().to_string()));

        if let Some(keep) = self.config.keep_last_n_checkpoints {
            let keep = keep as usize;
            while self.epoch_checkpoints.len() > keep {
                // Retention is FIFO over this attempt's own epoch order: the
                // vector is always epoch-ascending (each save appends), so
                // index 0 is the oldest surviving entry. Peek it (do not pop
                // yet) so a failed delete leaves it exactly where a retry
                // will find it again.
                let (oldest_epoch, _) = self.epoch_checkpoints[0];
                let deleted = tokio::runtime::Handle::current()
                    .block_on(store.delete_epoch_checkpoint(
                        &self.job_id,
                        &self.worker_id,
                        &self.attempt,
                        oldest_epoch,
                    ))
                    .is_ok();
                if !deleted {
                    tracing::warn!(
                        job_id = %self.job_id,
                        attempt = %self.attempt,
                        epoch = oldest_epoch,
                        "epoch-checkpoint retention prune failed; retrying at the next epoch \
                         boundary (the finalize-winner's sweep reclaims it if retries never \
                         succeed before the run ends)"
                    );
                    break;
                }
                self.epoch_checkpoints.remove(0);
            }
        }
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

/// CoSENT loss: the pairwise log-sum-exp ordering ([`pairwise_ordering_loss`])
/// applied to temperature-scaled cosine similarity.
///
/// `pairwise_ordering_loss(scale · cos(a, b), scores)` — cosine similarity is
/// the per-pair signal AnglE's angle magnitude ([`angle_loss`]) stands in for.
///
/// Matches Su et al., 2022, "CoSENT: A more effective sentence embedding
/// scheme" (<https://kexue.fm/archives/8847>) and sentence-transformers'
/// `CoSENTLoss`: `loss = log(1 + Σ_{scores[i] > scores[j]} exp(λ·(cos[j] −
/// cos[i])))` with `λ = 20` (= [`PAIRWISE_SCALE`]).
fn cosent_loss(emb_a: &Tensor, emb_b: &Tensor, scores: &Tensor) -> Result<Tensor> {
    let cos = cosine_similarity(emb_a, emb_b)?;
    let scaled =
        (&cos * PAIRWISE_SCALE).map_err(|e| JammiError::FineTune(format!("CoSENT scale: {e}")))?;
    pairwise_ordering_loss(&scaled, scores)
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
/// `MSE(scale · cos(a, b), scale · score)`. The simplest objective for
/// continuous similarity labels — distinct from CoSENT (pairwise ordering)
/// and MNRL (ranking). Reuses [`PAIRWISE_SCALE`] so the predicted value lives
/// on the same scale as the graded targets the other objectives consume.
///
/// This is, up to the ×400 (`scale²`) this function applies and CoSENT's
/// does not, exactly the value the CoSENT default computed before it was
/// fixed to the real pairwise-ordering objective: the old code scaled by
/// `PAIRWISE_SCALE` then immediately divided by it before squaring, so the
/// scale factors cancelled and it silently ran plain unscaled `MSE(cos(a,
/// b), score)` — this function's value ÷ 400.
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

/// Per-example decomposition of [`cosine_mse_loss`] (H1, unit 63): the same
/// `(scale·cos(a,b) − scale·score)²` residual per row, read back to host
/// before the mean `cosine_mse_loss` takes instead. Row-independent by
/// construction. Reuses the SAME [`cosine_similarity`] call `cosine_mse_loss`
/// makes; does not touch `cosine_mse_loss` itself.
fn cosine_mse_loss_per_example(
    emb_a: &Tensor,
    emb_b: &Tensor,
    scores: &Tensor,
) -> Result<Vec<f64>> {
    let cos = cosine_similarity(emb_a, emb_b)?;
    let pred = (&cos * PAIRWISE_SCALE)
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE per-example scale: {e}")))?;
    let target = (scores * PAIRWISE_SCALE)
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE per-example target scale: {e}")))?;
    let diff = (&pred - &target)
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE per-example diff: {e}")))?;
    let host = diff
        .to_dtype(DType::F32)
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE per-example dtype: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| JammiError::FineTune(format!("cosine-MSE per-example to_vec1: {e}")))?;
    Ok(host.into_iter().map(|v| (v * v) as f64).collect())
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

/// Per-example decomposition of [`mnrl_loss`] (H1, unit 63): builds the
/// IDENTICAL `(n, n [+ hard negatives])` row-direction similarity/logits
/// matrix `mnrl_loss` builds (same [`l2_normalize_rows`] /
/// `contiguous_matmul` calls, same scale, same hard-negative concatenation),
/// but instead of reducing it with `candle_nn::loss::cross_entropy`'s mean,
/// reads it back to host and returns each anchor row's own NLL via
/// [`cross_entropy_per_row`] — one `f64` per row instead of their mean. When
/// `symmetric`, each returned value is `0.5 * (row_i + col_i)`, matching the
/// row+column average `mnrl_loss` itself takes, just per row instead of over
/// the whole batch.
///
/// This is a separate computation from `mnrl_loss` — it does not call it,
/// and `mnrl_loss` is unchanged (see the module doc's "example-mean is a
/// NEW quantity" note). MNRL's per-example loss is BATCH-COUPLED by
/// construction: row `i`'s value depends on every other row's positive
/// sharing the similarity matrix, which is exactly why
/// [`jammi_wire::fine_tune::HeldOutLoss`] carries `batch_partition_sha256`
/// and `in_batch_negatives_per_example` — a different partition of the same
/// example set changes every value this function returns.
fn mnrl_loss_per_example(
    anchor: &Tensor,
    positive: &Tensor,
    hard_negatives: Option<&Tensor>,
    scale: f64,
    symmetric: bool,
) -> Result<Vec<f64>> {
    let n = anchor
        .dim(0)
        .map_err(|e| JammiError::FineTune(format!("mnrl per-example dim: {e}")))?;

    let a_norm = l2_normalize_rows(anchor)?;
    let p_norm = l2_normalize_rows(positive)?;
    let p_t = p_norm
        .t()
        .map_err(|e| JammiError::FineTune(format!("mnrl per-example transpose: {e}")))?;
    let sim = (jammi_encoders::contiguous_matmul(&a_norm, &p_t)
        .map_err(|e| JammiError::FineTune(format!("mnrl per-example matmul: {e}")))?
        * scale)
        .map_err(|e| JammiError::FineTune(format!("mnrl per-example scale: {e}")))?;

    let labels = Tensor::arange(0u32, n as u32, anchor.device())
        .map_err(|e| JammiError::FineTune(format!("mnrl per-example labels: {e}")))?;

    let row_logits = match hard_negatives {
        None => sim.clone(),
        Some(neg) => {
            let neg_norm = l2_normalize_rows(neg)?;
            let neg_t = neg_norm.t().map_err(|e| {
                JammiError::FineTune(format!("mnrl per-example neg transpose: {e}"))
            })?;
            let neg_sim = (jammi_encoders::contiguous_matmul(&a_norm, &neg_t)
                .map_err(|e| JammiError::FineTune(format!("mnrl per-example neg matmul: {e}")))?
                * scale)
                .map_err(|e| JammiError::FineTune(format!("mnrl per-example neg scale: {e}")))?;
            Tensor::cat(&[&sim, &neg_sim], 1)
                .map_err(|e| JammiError::FineTune(format!("mnrl per-example neg cat: {e}")))?
        }
    };

    let row_losses = cross_entropy_per_row(&row_logits, &labels)?;
    if !symmetric {
        return Ok(row_losses);
    }

    let col_logits = sim
        .t()
        .map_err(|e| JammiError::FineTune(format!("mnrl per-example col transpose: {e}")))?;
    let col_losses = cross_entropy_per_row(&col_logits, &labels)?;

    Ok(row_losses
        .iter()
        .zip(col_losses.iter())
        .map(|(r, c)| 0.5 * (r + c))
        .collect())
}

/// Read `logits` `(n, c)` and integer `labels` `(n,)` back to host and
/// compute each row's cross-entropy NLL, `log_sum_exp(row) - row[label]`, in
/// `f32` arithmetic (the tensor's native compute precision) via a
/// numerically stable max-subtract log-sum-exp, folded in the FIXED column
/// order `0..c` (family J determinism — a host reduction over a bounded set
/// of elements folds them in the same order every call, on every platform).
/// Returns one `f64` per row.
///
/// This is the SAME per-row term `candle_nn::loss::cross_entropy`'s mean
/// reduces over, read back individually instead of pre-reduced — it does
/// not call `candle_nn::loss::cross_entropy` and changes no value that
/// function produces. When the target column sits far enough above every
/// other column in a row (a well-separated, near-converged batch), the
/// max-subtracted `sum_exp` rounds to EXACTLY `1.0f32` (every other term is
/// smaller than `f32`'s ~1.19e-7 relative epsilon) and the row's loss rounds
/// to EXACTLY `0.0` — the objective's true floor, not an approximation of
/// it, which is what lets [`TrainingLoop::evaluate_held_out`]'s
/// `tie_fraction` read a genuine `1.0` on a saturated held-out split.
fn cross_entropy_per_row(logits: &Tensor, labels: &Tensor) -> Result<Vec<f64>> {
    let logits_host = logits
        .to_dtype(DType::F32)
        .map_err(|e| JammiError::FineTune(format!("per-row NLL logits dtype: {e}")))?
        .to_vec2::<f32>()
        .map_err(|e| JammiError::FineTune(format!("per-row NLL logits to_vec2: {e}")))?;
    let labels_host = labels
        .to_dtype(DType::U32)
        .map_err(|e| JammiError::FineTune(format!("per-row NLL labels dtype: {e}")))?
        .to_vec1::<u32>()
        .map_err(|e| JammiError::FineTune(format!("per-row NLL labels to_vec1: {e}")))?;
    if logits_host.len() != labels_host.len() {
        return Err(JammiError::FineTune(format!(
            "per-row NLL: {} logit rows but {} labels",
            logits_host.len(),
            labels_host.len()
        )));
    }

    let mut out = Vec::with_capacity(logits_host.len());
    for (row, &label) in logits_host.iter().zip(labels_host.iter()) {
        let label = label as usize;
        let target = *row.get(label).ok_or_else(|| {
            JammiError::FineTune(format!(
                "per-row NLL: label {label} out of range for a {}-wide row",
                row.len()
            ))
        })?;
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &v in row {
            sum_exp += (v - max).exp();
        }
        let logsumexp = max + sum_exp.ln();
        out.push((logsumexp - target) as f64);
    }
    Ok(out)
}

/// Number of rows a [`super::data::TrainingBatch`] carries, regardless of
/// kind — the per-batch row count [`TrainingLoop::evaluate_held_out`] checks
/// against `config.batch_size` and uses to slice its flat `example_ids` list
/// per batch.
fn batch_row_count(batch: &super::data::TrainingBatch) -> Result<usize> {
    let dim0 = |t: &Tensor| {
        t.dim(0)
            .map_err(|e| JammiError::FineTune(format!("held-out batch row count: {e}")))
    };
    match batch {
        super::data::TrainingBatch::Contrastive { embeddings_a, .. } => dim0(embeddings_a),
        super::data::TrainingBatch::Pairs { anchors, .. } => dim0(anchors),
        super::data::TrainingBatch::Triplet { anchor, .. } => dim0(anchor),
        super::data::TrainingBatch::Classification { embeddings, .. } => dim0(embeddings),
        super::data::TrainingBatch::Ner { hidden_states, .. } => dim0(hidden_states),
        super::data::TrainingBatch::Regression { input, .. } => dim0(input),
    }
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

/// Per-example counterpart of [`matryoshka_sum`] (H1, unit 63): narrows every
/// input tensor to each configured prefix width exactly as `matryoshka_sum`
/// does, but sums the per-example `Vec<f64>` `objective` returns elementwise
/// (in the FIXED `dims` order — family J determinism) instead of summing a
/// device `Tensor`. Does not touch `matryoshka_sum`.
fn matryoshka_sum_per_example(
    dims: &[usize],
    embeddings: &[&Tensor],
    objective: &dyn Fn(Vec<Tensor>) -> Result<Vec<f64>>,
) -> Result<Vec<f64>> {
    if dims.is_empty() {
        return objective(embeddings.iter().map(|t| (*t).clone()).collect());
    }

    let full_dim = embeddings
        .first()
        .ok_or_else(|| JammiError::FineTune("matryoshka per-example: no embeddings".into()))?
        .dim(1)
        .map_err(|e| JammiError::FineTune(format!("matryoshka per-example dim: {e}")))?;

    let mut total: Option<Vec<f64>> = None;
    for &dim in dims {
        if dim > full_dim {
            return Err(JammiError::FineTune(format!(
                "matryoshka_dims entry {dim} exceeds the embedding width {full_dim}"
            )));
        }
        let truncated: Vec<Tensor> = embeddings
            .iter()
            .map(|t| {
                t.narrow(1, 0, dim).map_err(|e| {
                    JammiError::FineTune(format!("matryoshka per-example narrow: {e}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let losses = objective(truncated)?;
        total = Some(match total {
            None => losses,
            Some(acc) => {
                if acc.len() != losses.len() {
                    return Err(JammiError::FineTune(format!(
                        "matryoshka per-example: dim {dim} produced {} losses, a prior dim \
                         produced {} — the per-example count must stay fixed across prefixes",
                        losses.len(),
                        acc.len()
                    )));
                }
                acc.iter().zip(losses.iter()).map(|(a, b)| a + b).collect()
            }
        });
    }
    total.ok_or_else(|| {
        JammiError::FineTune("matryoshka_dims was unexpectedly empty (per-example)".into())
    })
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

    /// L2 norm of a gradient tensor as an f64 scalar.
    fn grad_norm(g: &Tensor) -> f64 {
        let sq: f32 = g.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        (sq as f64).sqrt()
    }

    /// Secondary to `last_step_run_harness::gradcache_arm_refuses_a_nonfinite_
    /// gradient_on_the_runs_last_step` (which drives `run()` end-to-end):
    /// `total_optimizer_steps` (in `run`) must use GradCache's per-epoch
    /// horizon ONLY when the loader is NOT precomputed AND the run is
    /// GradCache-eligible — all four truth-table cells pinned directly.
    ///
    /// Mutation tried against this function's body (`!is_precomputed &&
    /// gradcache_eligible` → `is_precomputed && gradcache_eligible`): the
    /// `(false, true)` case flips from `true` to `false` — this test goes
    /// red.
    #[test]
    fn wants_gradcache_horizon_requires_a_non_precomputed_loader() {
        assert!(
            TrainingLoop::wants_gradcache_horizon(false, true),
            "non-precomputed + eligible must want the GradCache horizon"
        );
        assert!(
            !TrainingLoop::wants_gradcache_horizon(true, true),
            "a precomputed loader must never want the GradCache horizon, \
             even when otherwise eligible"
        );
        assert!(
            !TrainingLoop::wants_gradcache_horizon(false, false),
            "not eligible must never want the GradCache horizon"
        );
        assert!(
            !TrainingLoop::wants_gradcache_horizon(true, false),
            "precomputed and not eligible must never want the GradCache horizon"
        );
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

        let cosent = cosent_loss(a_t, &b, &scores).unwrap();
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

    /// Fixed embeddings giving cosine similarity exactly `[0.9, 0.1]`: `a`'s
    /// rows are the unit vector `[1, 0]`, `b`'s rows are `[c, sqrt(1 - c^2)]`
    /// for `c` in `{0.9, 0.1}` — a unit vector by construction, so the dot
    /// product `a · b` is exactly `c` with no normalisation rounding beyond
    /// `c` itself. Shared by the esc-040 CoSENT tests below.
    fn cosent_fixture(device: &Device) -> (Tensor, Tensor) {
        let a = Tensor::new(&[[1.0f32, 0.0], [1.0f32, 0.0]], device).unwrap();
        let b = Tensor::new(
            &[
                [0.9f32, (1.0f32 - 0.81f32).sqrt()],
                [0.1f32, (1.0f32 - 0.01f32).sqrt()],
            ],
            device,
        )
        .unwrap();
        (a, b)
    }

    /// esc-040 finiteness gate: the production CoSENT objective must not
    /// produce NaN/Inf on an ordinary graded batch.
    #[test]
    fn cosent_loss_is_finite() {
        let device = Device::Cpu;
        let (a, b) = cosent_fixture(&device);
        let scores = Tensor::new(&[0.2f32, 0.0f32], &device).unwrap();
        let loss = cosent_loss(&a, &b, &scores)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.is_finite(), "CoSENT loss must be finite, got {loss}");
    }

    /// esc-040: the CoSENT default must be the real pairwise-ordering
    /// objective ([`pairwise_ordering_loss`] over scaled cosine), not plain
    /// MSE on the cosine (the defect: `temperature` was multiplied in and
    /// immediately divided back out, cancelling to `mean((cos - score)^2)`).
    ///
    /// On `cos = [0.9, 0.1]`, `scores = [0.2, 0.0]` the two pairs are already
    /// correctly ordered (the higher-cosine pair also has the higher target
    /// score), so the real CoSENT residual is the tiny
    /// `log(1 + exp(sim[1] - sim[0])) ≈ 1.19e-7` — orders of magnitude below
    /// `cosine_mse_loss`'s `≈ 100` (`= 400 ×` the old buggy value of `0.25`,
    /// see [`cosine_mse_loss`]'s doc). Two-sided: bounded well below the MSE
    /// arm's value AND strictly positive (not a vacuous always-zero stub).
    ///
    /// Also pins CoSENT's scale-invariance in the score: the pairwise mask
    /// reads only the *strict order* of `scores`, so scaling every score by a
    /// positive constant (here: halving) leaves the valid-pair set — and
    /// hence the loss — bit-identical. MSE has no such invariance.
    #[test]
    fn cosent_loss_is_pairwise_ordering_not_mse() {
        let device = Device::Cpu;
        let (a, b) = cosent_fixture(&device);
        let scores = Tensor::new(&[0.2f32, 0.0f32], &device).unwrap();
        let scores_half = Tensor::new(&[0.1f32, 0.0f32], &device).unwrap();

        let cosent = cosent_loss(&a, &b, &scores)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mse = cosine_mse_loss(&a, &b, &scores)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            cosent.is_finite() && mse.is_finite(),
            "both arms must be finite: cosent={cosent}, mse={mse}"
        );
        // Two-sided margin: far below the MSE arm's O(1) value...
        assert!(
            cosent < 1e-4,
            "CoSENT should be ~1.19e-7 on an already-well-ordered pair, got {cosent}"
        );
        // ...but strictly positive — not a vacuous always-zero stub.
        assert!(
            cosent > 0.0,
            "CoSENT must be a real (nonzero) residual, got {cosent}"
        );
        assert!(
            (99.0..101.0).contains(&mse),
            "sanity: cosine_mse_loss should read ~100 on this fixture (400 × the old buggy \
             cosent value of 0.25), got {mse}"
        );

        // Scale-invariance under a positive rescale of the graded scores.
        let cosent_half = cosent_loss(&a, &b, &scores_half)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            cosent, cosent_half,
            "CoSENT must be invariant to a positive rescale of the graded scores"
        );
    }

    /// esc-040 gradient arm: CoSENT must produce a finite, non-vanishing
    /// gradient on a batch containing an inverted pair — the training signal
    /// the loss exists to supply. Reuses [`cosent_fixture`]'s
    /// `cos = [0.9, 0.1]` but swaps the score labels relative to
    /// [`cosent_loss_is_pairwise_ordering_not_mse`]'s well-ordered case: the
    /// higher-cosine pair (index 0) is now graded *below* the lower-cosine
    /// pair (index 1) — a genuine ordering violation, not the near-zero
    /// residual the margin test pins. Asserts fixture non-degeneracy first (a
    /// valid pair exists and the loss reads the violation, not the
    /// no-valid-pairs `log(1) = 0` floor) so a vanishing gradient can only
    /// mean the objective itself is flat, never an empty mask.
    #[test]
    fn cosent_gradient_is_non_vanishing_on_inverted_pair() {
        let device = Device::Cpu;
        let a = Var::from_tensor(&Tensor::new(&[[1.0f32, 0.0], [1.0f32, 0.0]], &device).unwrap())
            .unwrap();
        let b = Tensor::new(
            &[
                [0.9f32, (1.0f32 - 0.81f32).sqrt()],
                [0.1f32, (1.0f32 - 0.01f32).sqrt()],
            ],
            &device,
        )
        .unwrap();
        // Inverted: the more-similar pair (cos 0.9) is graded below the
        // less-similar pair (cos 0.1) — a genuine ordering violation.
        let scores = Tensor::new(&[0.0f32, 0.2f32], &device).unwrap();
        let a_t: &Tensor = &a;

        let loss = cosent_loss(a_t, &b, &scores).unwrap();
        let loss_v = loss.to_scalar::<f32>().unwrap();
        assert!(
            loss_v.is_finite(),
            "CoSENT loss must be finite on an inverted pair, got {loss_v}"
        );
        // Fixture non-degeneracy: a real violation reads as a substantial
        // loss (≈16 here), not the empty-mask `log(1) = 0` floor.
        assert!(
            loss_v > 1.0,
            "fixture non-degeneracy: an inverted pair must produce a substantial loss, got {loss_v}"
        );

        let grad = loss.backward().unwrap();
        let norm = grad_norm(grad.get(a_t).unwrap());
        assert!(
            norm.is_finite(),
            "CoSENT gradient must be finite, got {norm}"
        );
        assert!(
            norm > 1e-6,
            "CoSENT gradient must be non-vanishing on an inverted pair, got {norm}"
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

/// Host-read accounting for the per-micro-batch training path:
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
        per_micro_batch_host_read_count, EpochState, LastStepHorizon, SimStats, StepContext,
        TrainingLoop, TrainingLoopBuilder,
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

    /// Folding a TRACKED per-batch mean (`track_op() == true` whenever
    /// `anchor`/`positive`/`negative` come from a real forward pass, as they
    /// do in production via `encode_chunk`'s LoRA `Var`s) into the epoch
    /// accumulator with no `detach()` retains every micro-batch's forward
    /// subgraph for the whole epoch — the accumulator's
    /// `sorted_nodes().len()` grows by a fixed per-call increment on every
    /// fold. This test builds its triplet batch THROUGH a `Var`
    /// (`w.as_tensor()` has `is_variable() == true`, so every
    /// `cosine_similarity`/`mean_all` op stacked on top of it is tracked) — a
    /// leaf `Tensor::new` fixture, as the two tests above use, is untracked
    /// from the start and CANNOT see this regression.
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

    /// `accumulate_sim_stats` used to do two `to_scalar::<f32>()` calls on
    /// every triplet micro-batch, BEFORE `backward` ever ran. Mutation tried: reinstate a `to_scalar` inside
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
        let mut accumulated_grads = candle_core::backprop::GradStore::default();
        let mut grads_pending = false;
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
                    grads_pending: &mut grads_pending,
                    global_step: &mut global_step,
                },
                StepContext {
                    trainable_vars: &trainable_vars,
                    optimizer: &mut optimizer,
                    checkpoint_dir: checkpoint_dir.path(),
                    checkpoint_interval: 0,
                    lr_horizon: 1,
                    last_step_horizon: &mut LastStepHorizon::new(1),
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

/// esc-052: `ner_loss`'s own doc comment claimed `ignore_index=-100`
/// semantics (positions labelled `-100` excluded from the loss) while its
/// body clamped `-100 -> 0` and ran UNMASKED `cross_entropy` over every
/// `batch*seq_len` position. Latent — nothing in-repo constructs a
/// `TrainingBatch::Ner` (`encode_chunk` refuses `TextChunk::Ner`), but the
/// path is publicly reachable via `TrainingDataLoader::from_precomputed`,
/// which is what these tests drive it through, exactly as a caller who
/// bypasses `encode_chunk` and hand-builds a `TrainingBatch::Ner` batch
/// would.
#[cfg(test)]
mod ner_loss_ignore_index {
    use candle_core::{DType, Device, Tensor, Var};
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;

    use super::super::data::{TextChunk, TrainingBatch, TrainingDataLoader};
    use super::super::lora::build_distribution_head;
    use super::super::target::TrainingTarget;
    use super::super::FineTuneConfig;
    use super::TrainingLoopBuilder;
    use jammi_db::error::JammiError;

    const HIDDEN: usize = 4;

    /// A minimal real [`super::TrainingLoop`], mirroring
    /// `host_read_discipline::minimal_loop`: `ner_loss`/`compute_loss`
    /// never touch the catalog, so an empty, unclaimed-by-anything-else
    /// catalog registered under its own tag is enough.
    async fn minimal_loop(device: &Device) -> super::TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let config = FineTuneConfig::default();
        let head = build_distribution_head(HIDDEN, 2, &config, &varmap, &vb).unwrap();
        let (catalog, dir) = super::test_fixtures::claimed_job("ner-loss-ignore-index").await;
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("ner-loss-ignore-index".into())
            .worker_id("ner-loss-ignore-index-worker".into())
            .catalog(catalog)
            .artifact_dir(dir.path().to_path_buf())
            .build()
            .unwrap()
    }

    /// Build a `(1, 2, 3)` logits `Var` whose position-1 (the ignored row,
    /// label `-100`) is `pos1`, and whose position-0 (label `0`) is fixed
    /// across callers so batches only ever differ at the ignored row.
    fn logits_var(device: &Device, pos1: [f32; 3]) -> Var {
        Var::from_tensor(&Tensor::new(&[[[0.0f32, 0.0, 0.0], pos1]], device).unwrap()).unwrap()
    }

    fn ner_labels(device: &Device) -> Tensor {
        // label 0 at position 0, ignore_index -100 at position 1.
        Tensor::new(&[[0i64, -100i64]], device).unwrap()
    }

    /// Round-trip a `TrainingBatch::Ner` through the PUBLIC
    /// `TrainingDataLoader::from_precomputed` / `batches()` API (rather than
    /// calling `compute_loss` on a hand-held batch directly), so the fix is
    /// verified through the same public surface an out-of-tree caller who
    /// hand-builds a `TrainingBatch::Ner` batch would use.
    fn round_trip_through_precomputed(batch: TrainingBatch) -> TrainingBatch {
        let loader = TrainingDataLoader::from_precomputed(vec![batch]);
        let mut batches = loader.batches(1).unwrap();
        assert_eq!(batches.len(), 1);
        batches.remove(0).unwrap()
    }

    /// THE EVAL (esc-052 triage symptom_spec, verbatim): two `Ner` batches
    /// identical except at the ignored (label `-100`) position must produce
    /// bit-identical losses AND an exactly-zero, finite gradient at that
    /// position's logits.
    ///
    /// RED (pre-fix — verified by reverting the fix hunk in `ner_loss` and
    /// re-running this test): the old body clamped `-100 -> 0` and ran
    /// UNMASKED cross-entropy over all positions, so batch A's
    /// (`pos1 = [0,0,0]`) and batch B's (`pos1 = [0,0,8]`, a confident wrong
    /// prediction against the clamped label `0`) losses differ by *far* more
    /// than 0.1 — the ignored row's huge cross-entropy term leaks straight
    /// into the mean. GREEN (post-fix, asserted below): masking the ignored
    /// row out before `cross_entropy` runs makes the two losses depend only
    /// on the (identical) position-0 row, so they are bit-identical, and
    /// `index_select`'s backward gives the ignored row's gradient an exact
    /// `0.0` (see `ner_loss`'s doc comment for why that is exact, not
    /// merely small).
    #[test]
    fn ignored_position_does_not_move_the_loss_or_the_gradient() {
        let device = Device::Cpu;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let loop_ = rt.block_on(minimal_loop(&device));

        let labels = ner_labels(&device);
        let var_a = logits_var(&device, [0.0, 0.0, 0.0]);
        let var_b = logits_var(&device, [0.0, 0.0, 8.0]);

        let batch_a = round_trip_through_precomputed(TrainingBatch::Ner {
            hidden_states: var_a.as_tensor().clone(),
            labels: labels.clone(),
        });
        let batch_b = round_trip_through_precomputed(TrainingBatch::Ner {
            hidden_states: var_b.as_tensor().clone(),
            labels: labels.clone(),
        });

        let loss_a = loop_.compute_loss(&batch_a).unwrap();
        let loss_b = loop_.compute_loss(&batch_b).unwrap();
        let loss_a_val: f32 = loss_a.to_scalar().unwrap();
        let loss_b_val: f32 = loss_b.to_scalar().unwrap();

        // CONTROL 1 (finite-guard FIRST): a NaN must fail this control
        // outright, before any comparison — never fake either color.
        assert!(
            loss_a_val.is_finite() && loss_b_val.is_finite(),
            "loss_a={loss_a_val} loss_b={loss_b_val} must both be finite before comparing them"
        );

        // GREEN: bit-identical scalars, checked via the raw bit pattern
        // (not merely `abs(a - b) < eps`) — the ignored row must contribute
        // NOTHING, not just something small.
        assert_eq!(
            loss_a_val.to_bits(),
            loss_b_val.to_bits(),
            "loss_a={loss_a_val} loss_b={loss_b_val} must be bit-identical: they differ only at \
             the ignore_index=-100 position, which must not move the loss at all"
        );

        // GREEN: the ignored row's gradient is EXACTLY 0.0 and finite.
        let grads = loss_a.backward().unwrap();
        let grad_a = grads.get(&var_a).expect("logits var must have a gradient");
        let grad_rows = grad_a.to_vec3::<f32>().unwrap();
        let ignored_row = grad_rows[0][1].clone();
        assert!(
            ignored_row.iter().all(|g| g.is_finite()),
            "ignored row's gradient must be finite: {ignored_row:?}"
        );
        assert_eq!(
            ignored_row.as_slice(),
            [0.0f32, 0.0, 0.0],
            "ignored (label=-100) row's gradient must be EXACTLY zero, got {ignored_row:?}"
        );
    }

    /// POSITIVE CONTROL against an over-broad fix: with no `-100` label at
    /// all (`labels = [0, 1]`), the masked path must select every row and
    /// therefore agree with plain unmasked `cross_entropy` to bit-equality
    /// — a fix that (say) always dropped the last position, or scaled the
    /// mean by the wrong denominator, would show up here even though
    /// nothing is actually ignored.
    #[test]
    fn no_ignored_positions_matches_unmasked_cross_entropy_bit_exactly() {
        let device = Device::Cpu;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let loop_ = rt.block_on(minimal_loop(&device));

        let logits = Tensor::new(&[[[1.0f32, 0.2, -0.5], [0.1, 2.0, 0.3]]], &device).unwrap();
        let labels = Tensor::new(&[[0i64, 1i64]], &device).unwrap();
        let batch = round_trip_through_precomputed(TrainingBatch::Ner {
            hidden_states: logits.clone(),
            labels,
        });

        let masked_loss = loop_.compute_loss(&batch).unwrap();
        let masked_val: f32 = masked_loss.to_scalar().unwrap();

        let flat_logits = logits.reshape((2, 3)).unwrap();
        let flat_labels = Tensor::new(&[0u32, 1u32], &device).unwrap();
        let unmasked_loss = candle_nn::loss::cross_entropy(&flat_logits, &flat_labels).unwrap();
        let unmasked_val: f32 = unmasked_loss.to_scalar().unwrap();

        // CONTROL 1 (finite-guard FIRST).
        assert!(
            masked_val.is_finite() && unmasked_val.is_finite(),
            "masked={masked_val} unmasked={unmasked_val} must both be finite before comparing"
        );

        assert_eq!(
            masked_val.to_bits(),
            unmasked_val.to_bits(),
            "with no ignore_index=-100 labels, the masked path (selects every row) must match \
             plain cross_entropy bit-for-bit: masked={masked_val} unmasked={unmasked_val}"
        );
    }

    /// All-ignored batch: PyTorch's `CrossEntropyLoss(reduction="mean")`
    /// returns `0.0` (not a `0/0` NaN) when every target equals
    /// `ignore_index`; `ner_loss` must match that rather than propagating a
    /// NaN into training.
    #[test]
    fn all_ignored_batch_returns_zero_not_nan() {
        let device = Device::Cpu;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let loop_ = rt.block_on(minimal_loop(&device));

        let logits = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &device).unwrap();
        let labels = Tensor::new(&[[-100i64, -100i64]], &device).unwrap();
        let batch = round_trip_through_precomputed(TrainingBatch::Ner {
            hidden_states: logits,
            labels,
        });

        let loss = loop_.compute_loss(&batch).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(
            val.is_finite(),
            "all-ignored loss must be finite, got {val}"
        );
        assert_eq!(val, 0.0, "all-ignored loss must be exactly 0.0, got {val}");
    }

    /// NON-BUG GOLDEN: fixing `ner_loss`'s masking must not be mistaken for
    /// enabling end-to-end NER training. `encode_chunk` still refuses
    /// `TextChunk::Ner` with its typed error — the only in-repo way to
    /// reach `TrainingBatch::Ner` remains the precomputed test seam.
    #[test]
    fn encode_chunk_still_refuses_text_chunk_ner() {
        let device = Device::Cpu;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let loop_ = rt.block_on(minimal_loop(&device));

        let chunk = TextChunk::Ner {
            texts: vec!["Alice works at Acme.".into()],
            entities_json: vec!["[]".into()],
        };
        let err = loop_.encode_chunk(&chunk).err().unwrap();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("NER fine-tuning is not yet available")),
            "encode_chunk must still refuse TextChunk::Ner with its typed error, got {err:?}"
        );
    }
}

/// Last-step lattice row: **GradCache**. `run_gradcache_epoch` has no
/// loss-value divergence check of its own (unlike `process_batch_loss`'s
/// `loss_val.is_nan() || loss_val > 100.0` guard) — the ONLY thing that can
/// ever catch a diverged GradCache step is `clip_and_step`'s cadence-gated
/// `refuse_nonfinite_norm`, so getting `is_last_step` right on this arm is
/// load-bearing in a way the accumulation-window arm (which has the
/// loss-value backstop too) is not.
/// Fixtures shared by the run-level oracles below: the hermetic `tiny_bert`
/// base model and the registered-model + claimed-job catalog state
/// [`TrainingLoop::run`] stamps its start metrics against.
#[cfg(test)]
mod test_fixtures {
    use std::sync::Arc;

    use crate::model::{ModelSource, ModelTask};

    /// Load the hermetic `tiny_bert` cookbook fixture through a real
    /// `InferenceSession`'s model cache — the same resolve+backend-load path
    /// serving uses (see `ModelCache::load_owned_for_test`'s doc on
    /// `session.rs`'s equivalent seam). Real, but tiny and local: no network,
    /// sub-second load, matching every other `tiny_bert`-fixture test in
    /// `tests/it`.
    pub(super) async fn tiny_bert() -> Arc<crate::model::LoadedModel> {
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

    /// A catalog holding a registered model and a training job `tag`
    /// claimed by `{tag}-worker` — what `run`'s lease-guarded
    /// `mark_training_running` needs. Returns the catalog and the tempdir
    /// backing it (also usable as the loop's `artifact_dir`).
    pub(super) async fn claimed_job(
        tag: &str,
    ) -> (Arc<jammi_db::catalog::Catalog>, tempfile::TempDir) {
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
                loss_type: "mnrl",
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
            .expect("queued job is claimable");
        (catalog, dir)
    }
}

#[cfg(test)]
mod gradcache_last_step_oracle {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{ParamsAdamW, VarBuilder, VarMap};

    use super::super::data::TrainingDataLoader;
    use super::super::lora::build_projection_head;
    use super::super::target::TrainingTarget;
    use super::super::{EmbeddingLoss, FineTuneConfig};
    use super::test_fixtures::tiny_bert;
    use super::{LastStepHorizon, TrainingLoopBuilder};
    use crate::fine_tune::adamw::AdamW;

    const HIDDEN: usize = 32; // tiny_bert's hidden width.

    /// Secondary to `last_step_run_harness::gradcache_arm_refuses_a_nonfinite_
    /// gradient_on_the_runs_last_step` (which drives `run()` end-to-end):
    /// `run_gradcache_epoch`'s `is_last_step` must reflect
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
    /// `run_gradcache_epoch` — this test
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

            // Epoch 0: healthy weights, global_step 0 → 1. A horizon of `2`
            // (== `self.config.epochs`) is exactly what `run` builds for
            // this arm (see `total_optimizer_steps`'s doc in `run`).
            let mut horizon = LastStepHorizon::new(2);
            let global_step_after_epoch0 = 0usize;
            loop_
                .run_gradcache_epoch(
                    &loader,
                    &trainable_vars,
                    &mut optimizer,
                    &mut horizon,
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
                    &mut horizon,
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

/// End-to-end last-step harness: every arm of `TrainingLoop::run` drives the
/// REAL entry point with a fixture that reaches that arm, and a gradient
/// poisoned to NaN (through the `after_backward` test seam) on exactly the
/// run's LAST optimizer step. The run is shorter than
/// `DEFAULT_NORM_CHECK_INTERVAL`, so the ONLY thing that can surface that
/// NaN as the grad-norm refusal is the arm's `is_last_step` — the modulo
/// cadence never fires, and step 1 (always checked) is never the poisoned
/// step. Each test asserts the SPECIFIC grad-norm refusal (naming the step),
/// not just "some error": with `is_last_step` forced `false` the NaN trains
/// in silently and the run ends in `refuse_nonfinite_params`'s DIFFERENT
/// refusal at the epoch boundary — so the assertion discriminates the two.
///
/// Per arm, the horizon is that arm's actual loop, not a shared formula:
///  - accumulation window: `ceil(micro_batches / grad_accum) * epochs`, the
///    last step taken INSIDE `process_batch_loss` (micro-batch count a
///    multiple of `grad_accum`);
///  - trailing partial-window flush: same horizon, the last step taken at
///    `run`'s epoch-end flush (micro-batch count NOT a multiple);
///  - plain per-batch (`grad_accum == 1`): `micro_batches * epochs`, the last
///    step inside `process_batch_loss`;
///  - GradCache: `epochs` (one step per epoch, `run_gradcache_epoch`).
///
/// The healthy control per arm (no poison) must complete with
/// `result.total_steps` equal to that horizon — proof the fixture reaches
/// the arm and that the poisoned step IS the run's last.
#[cfg(test)]
mod last_step_run_harness {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::super::data::TrainingDataLoader;
    use super::super::lora::build_projection_head;
    use super::super::resume::{RestoredCheckpoint, ResumeState, RESUME_STATE_SCHEMA_VERSION};
    use super::super::target::TrainingTarget;
    use super::super::{EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig};
    use super::{AfterBackwardHook, TrainingLoopBuilder, TrainingResult};
    use crate::fine_tune::optimizer::DEFAULT_NORM_CHECK_INTERVAL;

    const HIDDEN: usize = 32; // tiny_bert's hidden width.

    pub(super) fn text_config() -> FineTuneConfig {
        FineTuneConfig {
            epochs: 1,
            batch_size: 2,
            validation_fraction: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            lora_rank: 2,
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            early_stopping_patience: 10_000,
            learning_rate: 1e-3,
            ..Default::default()
        }
    }

    pub(super) fn pairs(n: usize) -> TrainingDataLoader {
        TrainingDataLoader::from_pairs(
            (0..n)
                .map(|i| (format!("anchor text {i}"), format!("positive text {i}")))
                .collect(),
        )
    }

    /// A hook that replaces the first trainable var's gradient with NaN on
    /// optimizer step `step` (1-based) and leaves every other step untouched.
    pub(super) fn poison_grad_at(step: usize) -> AfterBackwardHook {
        Box::new(move |s, grads, vars| {
            if s == step {
                let t: &Tensor = &vars[0];
                let nan = Tensor::full(f32::NAN, t.dims(), t.device())
                    .map_err(|e| jammi_db::error::JammiError::FineTune(e.to_string()))?;
                grads.insert(t, nan);
            }
            Ok(())
        })
    }

    /// Build a real text-path loop over `tiny_bert` (registered model +
    /// claimed job, as `run` requires) and run it to completion on `loader`
    /// — on the CALLING thread, so a caller can read the thread-local
    /// `optimizer::thread_sync_read_count` for exactly this run — with an
    /// optional after-backward hook installed and, when `resume_at` is
    /// given, resuming from a hand-built bundle whose `global_step` is that
    /// value (the current weights, zero moments, epoch 0 completed).
    pub(super) fn run_text_loop(
        tag: &str,
        config: FineTuneConfig,
        loader: TrainingDataLoader,
        hook: Option<AfterBackwardHook>,
        resume_at: Option<usize>,
    ) -> jammi_db::error::Result<TrainingResult> {
        // A multi-thread runtime: `run` blocks on the catalog through
        // `Handle::current()`, which on a current-thread runtime cannot
        // drive the IO driver from outside `Runtime::block_on`; with worker
        // threads driving IO, entering the runtime on this thread is enough.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let (mut loop_, _dir) = rt.block_on(async {
            let base_model = super::test_fixtures::tiny_bert().await;
            let (catalog, dir) = super::test_fixtures::claimed_job(tag).await;
            let device = Device::Cpu;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let head = build_projection_head(HIDDEN, &config, &varmap, &vb).unwrap();
            let mut builder = TrainingLoopBuilder::new(
                TrainingTarget::ProjectionHead { head },
                varmap.clone(),
                config.clone(),
            )
            .device(device)
            .job_id(tag.into())
            .worker_id(format!("{tag}-worker"))
            .catalog(catalog)
            .artifact_dir(dir.path().to_path_buf())
            .base_model(base_model);
            if let Some(global_step) = resume_at {
                // A bundle "written by a longer run": copies of the current
                // weights (a `Var` refuses to be set from its own storage),
                // zero moments for every registered Var keyed by name (as
                // `restore_from_checkpoint` looks them up), and a step
                // counter the caller chooses.
                let data = varmap.data().lock().unwrap();
                let mut weights = HashMap::new();
                let mut moments = HashMap::new();
                for (name, var) in data.iter() {
                    let t = var.as_tensor();
                    weights.insert(name.clone(), t.copy().unwrap());
                    moments.insert(
                        name.clone(),
                        (t.zeros_like().unwrap(), t.zeros_like().unwrap()),
                    );
                }
                drop(data);
                builder = builder.resume(RestoredCheckpoint {
                    weights,
                    moments,
                    state: ResumeState {
                        schema_version: RESUME_STATE_SCHEMA_VERSION,
                        last_completed_epoch: 0,
                        global_step,
                        step_t: global_step,
                        seed: config.seed,
                        scaler: None,
                        dropout_positions: HashMap::new(),
                    },
                });
            }
            (builder.build().unwrap(), dir)
        });
        loop_.after_backward = hook;
        let _enter = rt.enter();
        loop_.run(&loader)
    }
    /// The typed grad-norm refusal, discriminated from every other error the
    /// run could end in: it must name the poisoned step.
    fn assert_grad_norm_refusal(err: &jammi_db::error::JammiError, step: usize) {
        let msg = err.to_string();
        assert!(
            msg.contains("non-finite total gradient norm") && msg.contains(&format!("step {step}")),
            "expected the grad-norm refusal naming optimizer step {step}, got: {msg}"
        );
    }

    /// Arm: accumulation window, the last step taken INSIDE
    /// `process_batch_loss` — 8 pairs at `batch_size: 2` = 4 micro-batches,
    /// `grad_accum: 2`, one epoch → 2 optimizer steps, both at window
    /// boundaries (no trailing flush). Poisoned step: 2.
    ///
    /// Mutation: force `is_last_step = false` at `process_batch_loss`'s
    /// `clip_and_step` — RED (the NaN trains in; the run ends in
    /// `refuse_nonfinite_params`'s epoch-boundary refusal, not this one).
    #[test]
    fn accumulation_window_arm_refuses_a_nonfinite_gradient_on_the_runs_last_step() {
        let config = FineTuneConfig {
            gradient_accumulation_steps: 2,
            ..text_config()
        };
        let healthy =
            run_text_loop("last-step-accum-ok", config.clone(), pairs(8), None, None).unwrap();
        assert_eq!(healthy.total_steps, 2, "control: the arm's horizon");
        assert!(healthy.total_steps < DEFAULT_NORM_CHECK_INTERVAL);

        let err = run_text_loop(
            "last-step-accum",
            config,
            pairs(8),
            Some(poison_grad_at(2)),
            None,
        )
        .expect_err("a NaN gradient on the run's last window must be refused");
        assert_grad_norm_refusal(&err, 2);
    }

    /// Arm: the trailing partial-window flush at `run`'s epoch end — 6
    /// pairs = 3 micro-batches, `grad_accum: 2`, one epoch → step 1 at
    /// micro-batch 2 (inside `process_batch_loss`), step 2 = the flush of the
    /// lone trailing micro-batch. Poisoned step: 2.
    ///
    /// Mutation: force `is_last_step = false` at the epoch-end flush's
    /// `clip_and_step` in `run` — RED.
    #[test]
    fn trailing_flush_arm_refuses_a_nonfinite_gradient_on_the_runs_last_step() {
        let config = FineTuneConfig {
            gradient_accumulation_steps: 2,
            ..text_config()
        };
        let healthy =
            run_text_loop("last-step-flush-ok", config.clone(), pairs(6), None, None).unwrap();
        assert_eq!(healthy.total_steps, 2, "control: the arm's horizon");

        let err = run_text_loop(
            "last-step-flush",
            config,
            pairs(6),
            Some(poison_grad_at(2)),
            None,
        )
        .expect_err("a NaN gradient on the run's trailing flush must be refused");
        assert_grad_norm_refusal(&err, 2);
    }

    /// Arm: plain per-batch stepping (`grad_accum: 1`) — 6 pairs = 3
    /// micro-batches, one epoch → 3 steps, every one inside
    /// `process_batch_loss`. Poisoned step: 3 (not step 1, not on the
    /// modulo cadence).
    ///
    /// Mutation: force `is_last_step = false` at `process_batch_loss`'s
    /// `clip_and_step` — RED.
    #[test]
    fn per_batch_arm_refuses_a_nonfinite_gradient_on_the_runs_last_step() {
        let config = text_config();
        let healthy =
            run_text_loop("last-step-plain-ok", config.clone(), pairs(6), None, None).unwrap();
        assert_eq!(healthy.total_steps, 3, "control: the arm's horizon");

        let err = run_text_loop(
            "last-step-plain",
            config,
            pairs(6),
            Some(poison_grad_at(3)),
            None,
        )
        .expect_err("a NaN gradient on the run's last per-batch step must be refused");
        assert_grad_norm_refusal(&err, 3);
    }

    /// Arm: GradCache (non-precomputed loader, `cached` + in-batch-negative
    /// objective) — one optimizer step per EPOCH: 6 pairs chunked at
    /// `batch_size: 2` (3 memory-bounded passes per epoch), `epochs: 2` →
    /// horizon 2. Poisoned step: 2.
    ///
    /// Mutations: force `is_last_step = false` in `run_gradcache_epoch` —
    /// RED; replace `total_optimizer_steps`'s arm selection with the
    /// accumulation-window `total_steps` (`ceil(3 / 1) * 2 == 6`) — RED
    /// (step 2 is then neither the horizon nor past it).
    #[test]
    fn gradcache_arm_refuses_a_nonfinite_gradient_on_the_runs_last_step() {
        let config = FineTuneConfig {
            cached: true,
            embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
            epochs: 2,
            ..text_config()
        };
        let healthy =
            run_text_loop("last-step-gc-ok", config.clone(), pairs(6), None, None).unwrap();
        assert_eq!(healthy.total_steps, 2, "control: one step per epoch");

        let err = run_text_loop(
            "last-step-gc",
            config,
            pairs(6),
            Some(poison_grad_at(2)),
            None,
        )
        .expect_err("a NaN gradient on the run's last GradCache epoch must be refused");
        assert_grad_norm_refusal(&err, 2);
    }

    /// The epoch-boundary backstop: a NaN gradient on a step that is neither
    /// step 1, on the modulo cadence, nor the run's last step (step 2 of 3)
    /// trains in unchecked — `clip_and_step` never reads the norm there, by
    /// design — and the NEXT micro-batch's NaN loss is skipped by the
    /// three-strikes divergence guard, so the epoch ends with a non-finite
    /// adapter that `monitor_loss` (measured pre-step) cannot see.
    /// `refuse_nonfinite_params` must refuse it before `checkpoint_best` is
    /// written.
    ///
    /// Mutation: delete the `refuse_nonfinite_params` call before
    /// `save_checkpoint_tagged(.., "best")` — RED (the run returns `Ok` and
    /// saves a NaN adapter).
    #[test]
    fn checkpoint_best_refuses_a_nonfinite_parameter_the_monitored_loss_cannot_see() {
        let err = run_text_loop(
            "ckpt-best-nan",
            text_config(),
            pairs(6),
            Some(poison_grad_at(2)),
            None,
        )
        .expect_err("a NaN adapter must not be saved as checkpoint_best");
        let msg = err.to_string();
        assert!(
            msg.contains("checkpoint_best") && msg.contains("non-finite trainable parameter"),
            "expected the checkpoint_best refusal, got: {msg}"
        );
    }
}

/// `refuse_nonfinite_params`'s fold over per-`Var` sums must be `+` (mutants
/// sweep finding, P4b R3 finishing round). A finite/non-finite gate cannot
/// distinguish `+`/`-`/`*` when the only corruption on offer is NaN — NaN
/// propagates identically through all three ops — so
/// `checkpoint_best_refuses_a_nonfinite_parameter_the_monitored_loss_cannot_see`
/// above cannot see `+` swapped for `-` or `*`; a scoped `cargo mutants
/// --in-diff` sweep of this round's trainer.rs diff survived both. This
/// oracle instead picks two `Var`s whose sums are `+C`/`-C` for `C` close to
/// `f32::MAX`: the CORRECT `+` fold cancels to `0.0` (finite — a healthy run
/// with two ordinary, if extreme, finite parameters that never touched a
/// NaN must not be refused), while `-` folds to `2C` and `*` to `-C²`, both
/// of which overflow `f32` to infinity.
#[cfg(test)]
mod refuse_nonfinite_params_fold_oracle {
    use candle_core::{Device, Tensor, Var};

    use super::TrainingLoop;

    /// A `Var` whose value-sum is exactly `value` (a single-element tensor,
    /// so `sum_all()` needs no scaling to reach the target).
    fn const_var(value: f32, device: &Device) -> Var {
        Var::from_tensor(&Tensor::new(&[value], device).unwrap()).unwrap()
    }

    /// Mutation: `+` → `-` in the fold — RED (`2C` overflows `f32` to
    /// `+inf`). Mutation: `+` → `*` — RED (`-C²` overflows to `-inf`).
    #[test]
    fn two_var_fold_of_opposite_near_max_sums_stays_finite() {
        let device = Device::Cpu;
        let c = 2.0e38_f32;
        let pos = const_var(c, &device);
        let neg = const_var(-c, &device);
        TrainingLoop::refuse_nonfinite_params(&[pos, neg], 0)
            .expect("c + (-c) == 0.0 must fold to a finite total via +");
    }
}

/// Run-level oracles for `LastStepHorizon`'s lattice, driven through the
/// REAL `TrainingLoop::run` over the `tiny_bert` text path (the precomputed
/// arm computes its loss straight from the given embeddings, touching no
/// LoRA `Var`, so it never has a gradient to clip and never reads a norm —
/// it cannot see any of this). `refuse_nonfinite_norm` invocations are
/// counted through `optimizer::thread_sync_read_count` on the thread the run
/// executes on — the headline "no per-step sync" property measured at run
/// level, not inferred from one call, and immune to every other test's
/// training in the same process.
#[cfg(test)]
mod last_step_horizon_run_oracles {
    use super::super::optimizer::{
        thread_clip_call_count, thread_sync_read_count, DEFAULT_NORM_CHECK_INTERVAL,
    };
    use super::super::FineTuneConfig;
    use super::last_step_run_harness::{pairs, poison_grad_at, run_text_loop, text_config};

    /// Cells `step < horizon` and `step == horizon` on a fresh run: 6 pairs
    /// at `batch_size: 2`, `grad_accum: 1`, one epoch → 3 steps, under the
    /// 50-step cadence. The run reads the norm back EXACTLY twice: step 1
    /// (always) and step 3 (the exact last step); step 2 is neither.
    #[test]
    fn fresh_short_run_reads_the_norm_on_step_one_and_the_last_step_only() {
        let before = thread_sync_read_count();
        let result = run_text_loop("horizon-fresh", text_config(), pairs(6), None, None).unwrap();
        assert_eq!(result.total_steps, 3, "control: the run's horizon");
        assert!(result.total_steps < DEFAULT_NORM_CHECK_INTERVAL);
        assert_eq!(
            thread_sync_read_count() - before,
            2,
            "a fresh 3-step run must read the norm on step 1 and step 3 only"
        );
    }

    /// PR #381 fix-round item 2 (the 246-vs-249 `clip_gradients` call-count
    /// discrepancy between main and the branch, measured over the 12-seed
    /// `regression_surface::untrained_quantile_head_collapses_to_mu_no_
    /// separation` sweep): pins the optimizer-step count for a FIXED `(n_
    /// pairs, batch_size, epochs)` config on BOTH the CPU trainer path
    /// (`result.total_steps`, `trainer.rs`'s own `total_steps`/
    /// `total_optimizer_steps` computation in `run`) AND the clip path
    /// (`thread_clip_call_count`, `optimizer::clip_gradients`'s own
    /// per-invocation counter) — proving trainer.rs's step-count arithmetic
    /// is deterministic and NOT a second, independently-drifting count from
    /// the clip call count, for a config where early stopping cannot
    /// interfere (`early_stopping_metric: TrainLoss`, `early_stopping_
    /// patience: 10_000` — effectively disabled, see `text_config`'s doc).
    ///
    /// 8 pairs at `batch_size: 2` (4 micro-batches/epoch), `grad_accum: 1`,
    /// `epochs: 3` → `total_steps == 4 * 3 == 12`. `clip_and_step` calls
    /// `clip_gradients` exactly once per optimizer step (see `clip_and_
    /// step`'s own doc), so `thread_clip_call_count`'s delta over the run
    /// must equal `result.total_steps` exactly — not merely "close", not
    /// bounded, EQUAL — for every non-GradCache, non-mined, non-early-
    /// stopped run this crate takes (the arms `total_optimizer_steps`'s own
    /// lattice doc in `run` enumerates).
    ///
    /// This is what makes the 246-vs-249 discrepancy on the CHAOTIC 12-seed
    /// sweep (`max_grad_norm` default, `early_stopping_metric: ValLoss`,
    /// `early_stopping_patience: 3`) legible: it is NOT this deterministic
    /// arithmetic drifting between main and the branch (this test would go
    /// RED on either side of that regression) — trainer.rs's step-count
    /// lattice is unchanged in kind between main and the branch and remains
    /// exact here — it is `early_stopping_patience: 3` firing at a
    /// DIFFERENT epoch per seed on main vs. the branch, because the clip
    /// coefficient's `f32`-device-vs-`f64`-host accumulator precision (the
    /// module doc's "Accumulator precision" section, NOT the rounding-count
    /// fix this round makes) perturbs `monitor_loss` enough, over ~20 steps
    /// per seed, to shift a handful of seeds' early-stopping epoch by ±1 —
    /// exactly the "early stopping" arm `total_optimizer_steps`'s own
    /// lattice doc in `run` already documents as NOT reflected in `total_
    /// steps` (an early-stopped run's actual last step is whatever `global_
    /// step` reached before the `break`).
    ///
    /// MEASURED, not reasoned (an env-gated `eprintln!` counting `clip_
    /// gradients` invocations per seed, on both main and this branch,
    /// reverted after the count was captured — never committed; per-seed
    /// `clip_gradients`-call count equals that seed's own `global_step` at
    /// the moment its `patience: 3` window exhausted, confirming the clip
    /// path and the trainer path count the SAME thing per seed too, not
    /// only in aggregate). Seed order `[1,2,3,4,5,6,7,8,9,10,11,42]`:
    ///
    /// | seed | main | branch | Δ (steps) | Δ (epochs, 3 steps/epoch) |
    /// |---|---|---|---|---|
    /// | 1  | 45 | 42 | −3 | −1 |
    /// | 2  | 15 | 15 |  0 |  0 |
    /// | 3  | 21 | 21 |  0 |  0 |
    /// | 4  | 21 | 12 | −9 | −3 |
    /// | 5  | 21 | 39 | +18 | +6 |
    /// | 6  | 15 | 15 |  0 |  0 |
    /// | 7  | 24 | 24 |  0 |  0 |
    /// | 8  | 12 | 12 |  0 |  0 |
    /// | 9  | 15 | 12 | −3 | −1 |
    /// | 10 | 12 | 12 |  0 |  0 |
    /// | 11 | 12 | 12 |  0 |  0 |
    /// | 42 | 33 | 33 |  0 |  0 |
    /// | **sum** | **246** | **249** | **+3** | **+1** |
    ///
    /// FOUR seeds moved (1, 4, 5, 9), not one — two ran FEWER epochs on the
    /// branch (seed 1: −1, seed 4: −3), one ran MORE (seed 5: +6), one ran
    /// fewer again (seed 9: −1). The net `+3` steps (`249 − 246`) is a
    /// CANCELLATION of a −1, a −3, a +6, and a −1, not one seed's uniform
    /// extra epoch — the mechanism (chaotic `monitor_loss` values feeding
    /// an unchanged patience-3 comparison, per the paragraph above) predicts
    /// exactly this kind of two-directional per-seed shift, not a
    /// single-seed one; a prior draft of this doc claimed the latter without
    /// measuring it and was wrong.
    ///
    /// Mutation tried: duplicate `clip_and_step`'s own call to `clip_
    /// gradients` (`let outcome = clip_gradients(...)?; let _x =
    /// clip_gradients(...)?;`) — RED, this fixture's own assertion:
    /// `assertion `left == right` failed: clip_gradients must be invoked
    /// exactly once per optimizer step: 24 clip-path calls vs 12
    /// trainer-path steps for this fixed (n_pairs=8, batch_size=2,
    /// epochs=3) config` (`left: 24, right: 12`). (A previously-drafted
    /// `div_ceil(grad_accum.max(1))` → `/` mutant here was MEASURED to NOT
    /// kill this test — `train_batches_per_epoch == 4` and `grad_accum ==
    /// 1` make `4.div_ceil(1) == 4 / 1 == 4` bit-for-bit for this exact
    /// fixture, so that claim was reasoned, not measured, and false; this
    /// paragraph replaces it with a mutant that was actually run.)
    #[test]
    fn clip_call_count_matches_total_optimizer_steps_for_a_fixed_config() {
        let config = FineTuneConfig {
            epochs: 3,
            ..text_config()
        };
        let clip_calls_before = thread_clip_call_count();
        let result = run_text_loop("clip-count-fixed", config, pairs(8), None, None).unwrap();
        assert_eq!(
            result.total_steps, 12,
            "control: 8 pairs / batch_size 2 = 4 steps/epoch * 3 epochs"
        );
        let clip_calls = thread_clip_call_count() - clip_calls_before;
        assert_eq!(
            clip_calls, result.total_steps as u64,
            "clip_gradients must be invoked exactly once per optimizer step: {clip_calls} \
             clip-path calls vs {} trainer-path steps for this fixed (n_pairs=8, batch_size=2, \
             epochs=3) config",
            result.total_steps
        );
    }

    /// The shrunk-horizon resume fixture: 8 pairs at `batch_size: 2`
    /// (4 steps per epoch), `epochs: 3` → horizon `ceil(4 / 1) * 3 == 12`,
    /// resumed from a checkpoint claiming `global_step == 100` at the end
    /// of epoch 0 — a run whose epoch budget shrank after the crash. The
    /// resumed run takes epochs 1 and 2: 8 steps (101..=108), EVERY one past
    /// the horizon.
    fn shrunk_horizon_config() -> FineTuneConfig {
        FineTuneConfig {
            epochs: 3,
            ..text_config()
        }
    }

    /// Cells `step > horizon` armed → disarmed, over the whole resumed run:
    /// it must read the norm back exactly once (the first overshoot step,
    /// 101, then disarm) — never on all 8.
    ///
    /// Mutation tried: decide `is_last_step` with `step >= horizon` (the
    /// `>=` form) — RED (8 reads, one per step: the sync amplification the
    /// one-shot arm exists to prevent). The contract's run-level bound is
    /// `≤ ceil(N / interval) + 2` reads over `N` steps; the exact count for
    /// this fixture is pinned alongside it.
    #[test]
    fn resumed_run_past_a_shrunk_horizon_checks_the_overshoot_once() {
        let before = thread_sync_read_count();
        let result = run_text_loop(
            "horizon-overshoot",
            shrunk_horizon_config(),
            pairs(8),
            None,
            Some(100),
        )
        .unwrap();
        let n = result.total_steps - 100;
        assert_eq!(n, 8, "control: epochs 1 and 2 of 3, 4 steps each");
        let reads = thread_sync_read_count() - before;
        let bound = (n.div_ceil(DEFAULT_NORM_CHECK_INTERVAL) + 2) as u64;
        assert!(
            reads <= bound,
            "a resumed run past its horizon must not re-sync on every step: {reads} reads over \
             {n} steps, bound {bound}"
        );
        assert_eq!(
            reads, 1,
            "exactly one overshoot check (step 101), then the cadence decides"
        );
    }

    /// Cell `step > horizon`, armed: the one-shot overshoot check is a REAL
    /// check — a NaN gradient on the first step past the horizon (101) is
    /// refused, naming that step.
    ///
    /// Mutation tried: delete the `step > self.horizon && self.overshoot_armed`
    /// arm of `LastStepHorizon::is_last_step` — RED (step 101 is unchecked;
    /// the NaN trains in).
    #[test]
    fn resumed_run_past_a_shrunk_horizon_refuses_a_nan_on_its_first_overshoot_step() {
        let err = run_text_loop(
            "horizon-overshoot-nan",
            shrunk_horizon_config(),
            pairs(8),
            Some(poison_grad_at(101)),
            Some(100),
        )
        .expect_err("the one-shot overshoot check must refuse a NaN on step 101");
        let msg = err.to_string();
        assert!(
            msg.contains("non-finite total gradient norm") && msg.contains("step 101"),
            "expected the grad-norm refusal naming step 101, got: {msg}"
        );
    }

    /// Cell `step > horizon`, disarmed: after the one-shot check, later
    /// overshoot steps are left to the modulo cadence — a NaN on step 102 is
    /// NOT refused by the grad-norm check (that is the per-step sync this
    /// fix refuses to reinstate): the run reads the norm exactly once (step
    /// 101), the NaN trains in, and it is `refuse_nonfinite_params`'s
    /// epoch-boundary backstop — not the grad-norm check — that refuses the
    /// adapter such a run leaves behind.
    #[test]
    fn resumed_run_past_a_shrunk_horizon_leaves_later_overshoot_steps_to_the_cadence() {
        let before = thread_sync_read_count();
        let outcome = run_text_loop(
            "horizon-overshoot-later",
            shrunk_horizon_config(),
            pairs(8),
            Some(poison_grad_at(102)),
            Some(100),
        );
        let err = outcome.expect_err("the NaN adapter is refused at the epoch boundary");
        let msg = err.to_string();
        assert!(
            !msg.contains("gradient norm") && msg.contains("checkpoint_best"),
            "step 102 is past the one-shot check and off the cadence; the epoch-boundary \
             backstop must be what refuses it: {msg}"
        );
        assert_eq!(
            thread_sync_read_count() - before,
            1,
            "the disarmed overshoot must not read the norm again"
        );
    }
}

/// Issue #441 (jammi-ai half): `TrainingResult::metrics_json`'s `train_loss_
/// curve` / `val_loss_curve` arrays must be genuinely retained, not
/// transcribed — measured live against the SAME "Epoch complete" tracing
/// event `crates/jammi-ai/tests/gpu_capability/harness.rs::loss_capture`
/// reads (family F: a headline number is measured-and-asserted, never
/// transcribed). This module's own [`EpochLossLayer`] mirrors that
/// production tracing layer's field names/event match byte-for-byte — the
/// two independently-defined layers agreeing pins that the metrics-JSON
/// curve and what the GPU-capability suite already reads back off `tracing`
/// are provably the SAME numbers, not two sources that merely happen to
/// agree today.
#[cfg(test)]
mod loss_curve_metrics {
    use std::cell::RefCell;
    use std::sync::OnceLock;

    use tracing::field::{Field, Visit};
    use tracing::Event;
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::Layer;

    use super::super::{EarlyStoppingMetric, FineTuneConfig};
    use super::last_step_run_harness::{pairs, run_text_loop, text_config};

    // THREAD-LOCAL, not a shared `Mutex<Vec<_>>`: unlike the `gpu_capability`
    // suite (mandated `--test-threads=1`), this crate's `cargo test` runs
    // fully parallel by default, and every `run_text_loop` call below
    // executes its whole epoch loop (including the "Epoch complete" event
    // itself) synchronously on ITS OWN calling thread (`run()` is a plain
    // sync fn; the one `Handle::block_on` call inside it still polls inline
    // on the calling thread) — so a THREAD-LOCAL buffer is exactly the right
    // granularity to isolate one test's captured curve from every other
    // concurrently-running test that also drives the trainer through this
    // same "Epoch complete" callsite.
    thread_local! {
        static TRAIN_CAPTURE: RefCell<Vec<(u64, f64)>> = const { RefCell::new(Vec::new()) };
        static VAL_CAPTURE: RefCell<Vec<(u64, f64)>> = const { RefCell::new(Vec::new()) };
    }

    /// Captures `(epoch, avg_train_loss)` / `(epoch, avg_val_loss)` off every
    /// "Epoch complete" event — the exact same event, field names, and
    /// `Option<f64>`-skips-when-`None` semantics
    /// `tests/gpu_capability/harness.rs::loss_capture::EpochLossLayer`
    /// depends on, duplicated here (small-duplication-is-fine per this
    /// crate's own convention) so a unit test in THIS binary can prove the
    /// mechanism without depending on the separate `gpu_capability` test
    /// binary's private module. Writes into the THREAD-LOCAL buffers above,
    /// never a shared one.
    struct EpochLossLayer;

    #[derive(Default)]
    struct EpochVisitor {
        epoch: Option<u64>,
        loss: Option<f64>,
        val_loss: Option<f64>,
        is_epoch_event: bool,
    }

    impl Visit for EpochVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" && format!("{value:?}").contains("Epoch complete") {
                self.is_epoch_event = true;
            }
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            if field.name() == "epoch" {
                self.epoch = Some(value);
            }
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            if field.name() == "epoch" {
                self.epoch = Some(value as u64);
            }
        }
        fn record_f64(&mut self, field: &Field, value: f64) {
            if field.name() == "avg_train_loss" {
                self.loss = Some(value);
            } else if field.name() == "avg_val_loss" {
                self.val_loss = Some(value);
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for EpochLossLayer {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut v = EpochVisitor::default();
            event.record(&mut v);
            if v.is_epoch_event {
                if let (Some(e), Some(l)) = (v.epoch, v.loss) {
                    TRAIN_CAPTURE.with(|c| c.borrow_mut().push((e, l)));
                }
                if let (Some(e), Some(l)) = (v.epoch, v.val_loss) {
                    VAL_CAPTURE.with(|c| c.borrow_mut().push((e, l)));
                }
            }
        }
    }

    /// Installs [`EpochLossLayer`] as the process's GLOBAL default tracing
    /// subscriber, exactly once (`OnceLock`, mirroring `tests/gpu_
    /// capability/harness.rs::loss_capture::install`'s own idiom).
    ///
    /// # The TRUE mechanism (round-3 audit re-derivation)
    ///
    /// A prior version of this doc claimed a thread-local (`with_default`
    /// / `set_default`) subscriber could never observe the "Epoch
    /// complete" event because `tracing`'s static max-level hint is
    /// governed by the GLOBAL default dispatcher "never" a scoped one.
    /// That specific claim was wrong: constructing a `Dispatch` — global
    /// OR thread-local — eagerly registers it and reruns interest
    /// rebuilding for every ALREADY-KNOWN callsite, folding in every
    /// currently-live dispatcher (scoped ones included). Proven directly:
    /// `crates/jammi-ai/src/model/backend/candle.rs`'s
    /// `device_tests::default_without_device_falls_back_to_cpu_with_warning`
    /// captures a `tracing::warn!` purely via `tracing::subscriber::
    /// with_default`, with NO global default ever installed anywhere in
    /// that process, and passes when run alone (`cargo test -p jammi-ai
    /// --lib model::backend::candle::device_tests::
    /// default_without_device_falls_back_to_cpu_with_warning --
    /// --test-threads=1 --exact`).
    ///
    /// The GLOBAL install here is still required, but for a DIFFERENT,
    /// narrower reason a scoped guard cannot substitute for: this crate's
    /// `cargo test` runs fully parallel by default (module doc above), and
    /// OTHER, unrelated tests in this SAME lib test binary also call
    /// `run_text_loop` — driving the identical "Epoch complete" callsite —
    /// WITHOUT installing any subscriber of their own. Empirically
    /// reproduced while investigating this: a `set_default`-based scoped
    /// rewrite of this very function passed every time run ALONE or with
    /// `--test-threads=1`, but flaked with an EMPTY `TRAIN_CAPTURE` when run
    /// in the crate's normal parallel mode alongside
    /// `metrics_json_omits_val_loss_curve_when_only_train_loss_is_
    /// monitored` (another test in this module that also drives
    /// `run_text_loop`, uninstrumented). Installing the GLOBAL default —
    /// which `tracing_core::dispatcher::get_default`'s fallback path
    /// consults for every thread that never calls `set_default`/
    /// `with_default` itself — fixes the flake.
    ///
    /// A prior version of this doc went further and named the specific
    /// mechanism: a per-callsite `Interest` cache populated once via a
    /// registration CAS `Once` and never re-evaluated per event. That
    /// mechanism claim is NOT established here and directly conflicts with
    /// the paragraph above it — `tracing-core` rebuilds interest via
    /// `rebuild_interest_cache()` on every new dispatcher registration,
    /// scoped guards included, so "cached permanently" cannot be the whole
    /// story. What IS established, by direct reproduction rather than
    /// by reading `tracing-core`'s source, is only the empirical fact:
    /// with only a thread-local subscriber installed anywhere in the
    /// process, this test flakes under the crate's parallel test mode, and
    /// installing the GLOBAL default fixes it. The precise interest-cache
    /// interleaving responsible for the flake was not isolated. The
    /// THREAD-LOCAL `TRAIN_CAPTURE`/`VAL_CAPTURE` buffers above are what
    /// then keep concurrently-running tests from corrupting each other's
    /// captured curve despite sharing this one global subscriber.
    fn install() {
        static INSTALLED: OnceLock<()> = OnceLock::new();
        INSTALLED.get_or_init(|| {
            let subscriber = tracing_subscriber::registry().with(EpochLossLayer);
            // `.ok()`: if some OTHER global default were already installed
            // (not the case anywhere in this crate today — a genuine
            // surprise, not a race, since `OnceLock` already serializes
            // concurrent callers of THIS fn to one winner), failing softly
            // here is still strictly better than panicking the whole test
            // binary over a tracing-capture convenience harness.
            let _ = tracing::subscriber::set_global_default(subscriber);
        });
    }

    /// 20 pairs, `validation_fraction: 0.34` (`round(20 * 0.34) == 7` held-out
    /// rows, 13 training rows — both non-empty), `early_stopping_metric:
    /// ValLoss` so `avg_val_loss` is actually measured every epoch,
    /// `early_stopping_patience: 10_000` (from `text_config`) so all 3
    /// configured epochs run to completion — the curves' length is therefore
    /// KNOWN, not merely "however many happened to run".
    #[test]
    fn metrics_json_loss_curves_match_the_epoch_complete_tracing_event() {
        install();
        TRAIN_CAPTURE.with(|c| c.borrow_mut().clear());
        VAL_CAPTURE.with(|c| c.borrow_mut().clear());

        let config = FineTuneConfig {
            epochs: 3,
            validation_fraction: 0.34,
            early_stopping_metric: EarlyStoppingMetric::ValLoss,
            ..text_config()
        };
        let result = run_text_loop("loss-curve-metrics", config, pairs(20), None, None).unwrap();

        let metrics: serde_json::Value =
            serde_json::from_str(&result.metrics_json).expect("metrics_json must be valid JSON");
        let train_curve = metrics["train_loss_curve"]
            .as_array()
            .expect("train_loss_curve must be present and an array");
        let val_curve = metrics["val_loss_curve"]
            .as_array()
            .expect("val_loss_curve must be present (this run measures ValLoss every epoch)");

        let captured_train = TRAIN_CAPTURE.with(|c| c.borrow().clone());
        let captured_val = VAL_CAPTURE.with(|c| c.borrow().clone());

        assert_eq!(
            train_curve.len(),
            3,
            "no early stopping fires (patience 10_000): all 3 configured epochs must have a \
             train_loss_curve row, got {train_curve:?}"
        );
        assert_eq!(
            val_curve.len(),
            3,
            "ValLoss is measured every epoch: val_loss_curve must have 3 rows too, got {val_curve:?}"
        );
        assert_eq!(
            train_curve.len(),
            captured_train.len(),
            "metrics_json's train_loss_curve must be exactly as long as the tracing capture: \
             {captured_train:?}"
        );
        assert_eq!(
            val_curve.len(),
            captured_val.len(),
            "metrics_json's val_loss_curve must be exactly as long as the tracing capture: \
             {captured_val:?}"
        );

        for (row, (epoch, loss)) in train_curve.iter().zip(captured_train.iter()) {
            let json_epoch = row["epoch"].as_u64().expect("row.epoch is a u64");
            let json_loss = row["loss"].as_f64().expect("row.loss is an f64");
            assert!(
                json_loss.is_finite(),
                "train_loss_curve row must be finite: {row:?}"
            );
            assert_eq!(json_epoch, *epoch, "epoch must match the tracing capture");
            assert_eq!(
                json_loss, *loss,
                "metrics_json's avg_train_loss must be the SAME f64 the tracing event carried \
                 (bit-identical: both read off the same in-memory `avg_train_loss` local, never \
                 re-derived)"
            );
        }
        for (row, (epoch, loss)) in val_curve.iter().zip(captured_val.iter()) {
            let json_epoch = row["epoch"].as_u64().expect("row.epoch is a u64");
            let json_loss = row["loss"].as_f64().expect("row.loss is an f64");
            assert!(
                json_loss.is_finite(),
                "val_loss_curve row must be finite: {row:?}"
            );
            assert_eq!(json_epoch, *epoch, "epoch must match the tracing capture");
            assert_eq!(
                json_loss, *loss,
                "metrics_json's avg_val_loss must be the SAME f64 the tracing event carried"
            );
        }
    }

    /// The `TrainLoss`-monitored arm: `avg_val_loss` is never measured
    /// (`text_config`'s default `early_stopping_metric`), so `val_loss_curve`
    /// must be ABSENT from `metrics_json` entirely — never an empty array
    /// (family F: a fabricated "measured zero epochs" is a different, false
    /// claim from the honest "not applicable to this run").
    #[test]
    fn metrics_json_omits_val_loss_curve_when_only_train_loss_is_monitored() {
        let config = FineTuneConfig {
            epochs: 2,
            ..text_config()
        };
        let result = run_text_loop("loss-curve-train-only", config, pairs(6), None, None).unwrap();
        let metrics: serde_json::Value =
            serde_json::from_str(&result.metrics_json).expect("metrics_json must be valid JSON");
        let train_curve = metrics["train_loss_curve"]
            .as_array()
            .expect("train_loss_curve must still be present");
        assert_eq!(
            train_curve.len(),
            2,
            "both configured epochs must have a row"
        );
        assert!(
            metrics.get("val_loss_curve").is_none(),
            "a TrainLoss-monitored run must never measure avg_val_loss, so val_loss_curve must \
             be absent, not an empty array: {metrics:?}"
        );
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

/// Unit 348 F2 (round-2 audit): a failed mid-run retention-prune delete must
/// KEEP its entry in `TrainingLoop::epoch_checkpoints` (never drop it just
/// because the delete failed) so the next epoch boundary retries the
/// identical oldest entry, and eventual success (once the failure clears)
/// catches the vector back up to the configured retention window.
///
/// Unix-only, real fault injection via `chmod` — deleting a file requires
/// write permission on its CONTAINING directory (POSIX), so removing write
/// permission from `checkpoints/epoch_0/` makes every delete attempt inside
/// it genuinely fail, the same class of failure a flaky object-store backend
/// would produce, without needing a pluggable `ArtifactStore` fault-injection
/// seam (`ArtifactStore` is a concrete struct wrapping the shared
/// `StorageRegistry`; no such seam is reachable from this crate's tests
/// today, hence pinning this at the unit level with real `file://` I/O
/// rather than attempting a live end-to-end worker/finalize harness for the
/// mid-run retry half specifically — the `Ok(true)` winner-arm reclaim half
/// is exercised separately by `crates/jammi-ai/tests/it/fine_tune.rs`'s
/// `finalize_reclaims_a_persistently_failed_prune_and_warns`, which uses the
/// SAME chmod technique against the real worker/finalize path).
#[cfg(all(test, unix))]
mod epoch_checkpoint_retention_failure {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Arc;

    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    use super::super::lora::build_distribution_head;
    use super::super::target::TrainingTarget;
    use super::super::FineTuneConfig;
    use super::{TrainingLoop, TrainingLoopBuilder};
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    const HIDDEN: usize = 4;

    /// Build the loop synchronously from an already-open `Arc<Catalog>` — the
    /// catalog open is the only genuinely async step, done by the caller
    /// BEFORE this runs, so the whole sequence of `save_epoch_checkpoint`
    /// calls (each internally `Handle::current().block_on(..)`, valid only
    /// off the async runtime — production always calls them from
    /// `spawn_blocking`) can run together inside ONE `spawn_blocking`
    /// closure, matching the real shape rather than fighting Tokio's
    /// "runtime within a runtime" panic.
    fn minimal_loop_with_store(
        device: &Device,
        keep: u32,
        artifact_dir: &std::path::Path,
        store: Arc<ArtifactStore>,
        catalog: Arc<jammi_db::catalog::Catalog>,
    ) -> TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let config = FineTuneConfig {
            keep_last_n_checkpoints: Some(keep),
            ..Default::default()
        };
        let head = build_distribution_head(HIDDEN, 2, &config, &varmap, &vb).unwrap();
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("f2-retry-job".into())
            .worker_id("f2-retry-worker".into())
            .attempt("0".into())
            .catalog(catalog)
            .artifact_dir(artifact_dir.to_path_buf())
            .artifact_store(store)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn failed_prune_stays_tracked_and_catches_up_once_unblocked() {
        let root_dir = tempfile::tempdir().unwrap().keep();
        let cache_dir = tempfile::tempdir().unwrap().keep();
        let root = StorageUrl::parse(root_dir.to_str().unwrap()).unwrap();
        let store =
            Arc::new(ArtifactStore::with_root(root, StorageRegistry::new(), cache_dir).unwrap());
        let artifact_dir = tempfile::tempdir().unwrap().keep();
        let checkpoint_dir = tempfile::tempdir().unwrap().keep();
        // The only async step — done here, before the blocking closure.
        let catalog = Arc::new(
            jammi_db::catalog::Catalog::open(&artifact_dir)
                .await
                .unwrap(),
        );

        let root_dir_for_blocking = root_dir.clone();
        tokio::task::spawn_blocking(move || {
            let device = Device::Cpu;
            let mut loop_ =
                minimal_loop_with_store(&device, 1, &artifact_dir, Arc::clone(&store), catalog);

            // Epoch 0: writes, no pruning yet (len 1 <= keep 1).
            loop_.save_epoch_checkpoint(&checkpoint_dir, 0).unwrap();
            assert_eq!(epoch_indices(&loop_), vec![0]);

            // Block epoch_0's own on-disk directory from further deletions —
            // removing write permission on the directory blocks removing
            // files INSIDE it (POSIX), the real failure mode a flaky store
            // backend would also produce.
            let epoch0_dir = epoch0_local_dir(&root_dir_for_blocking);
            assert!(
                epoch0_dir.join("manifest.json").exists(),
                "epoch_0 must be on disk"
            );
            std::fs::set_permissions(&epoch0_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
            // PROBE the injection before relying on it: root (and
            // mode-ignoring filesystems) can delete through a 0o555
            // directory, in which case the failed-prune premise this test
            // asserts never exists — skip loudly, mirroring the
            // `skip_on_cuda_capable_host` convention in candle.rs's
            // device_tests (the same environment-conditional-test class).
            let probe = epoch0_dir.join(".root_probe");
            if std::fs::write(&probe, b"x").is_ok() {
                let _ = std::fs::remove_file(&probe);
                let _ =
                    std::fs::set_permissions(&epoch0_dir, std::fs::Permissions::from_mode(0o755));
                eprintln!(
                    "epoch_checkpoint_retention_failure: skipping — fault injection \
                     unavailable: process can write despite chmod (root?)"
                );
                return;
            }

            // Epoch 1: writes (len 2 > keep 1), retention tries to prune
            // epoch_0 — FAILS (chmod'd). The entry must stay tracked, not be
            // dropped.
            loop_.save_epoch_checkpoint(&checkpoint_dir, 1).unwrap();
            assert_eq!(
                epoch_indices(&loop_),
                vec![0, 1],
                "a failed prune must keep its entry in the tracked vector, not drop it"
            );
            assert!(
                epoch0_dir.join("manifest.json").exists(),
                "epoch_0's bytes must still be on disk — the delete genuinely failed"
            );

            // Epoch 2: writes (len 3 > keep 1); retention retries epoch_0
            // first (still chmod'd) — STILL fails, still tracked, still no
            // progress.
            loop_.save_epoch_checkpoint(&checkpoint_dir, 2).unwrap();
            assert_eq!(
                epoch_indices(&loop_).len(),
                3,
                "the persistently-failing oldest entry blocks FIFO progress on the newer ones \
                 too (retention always retries the oldest first) — a residual this test pins, \
                 not a bug"
            );

            // Clear the failure (storage "recovers") and drive one more
            // epoch boundary's worth of retries — this time every prune the
            // over-the-cap loop attempts succeeds, catching the vector back
            // up to the configured window.
            std::fs::set_permissions(&epoch0_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
            loop_.save_epoch_checkpoint(&checkpoint_dir, 3).unwrap();
            assert_eq!(
                epoch_indices(&loop_),
                vec![3],
                "once the failure clears, retention catches back up to exactly the retained \
                 window"
            );
            assert!(
                !epoch0_dir.join("manifest.json").exists(),
                "epoch_0's bytes are gone once its retry finally succeeds"
            );
        })
        .await
        .unwrap();
    }

    fn epoch_indices(loop_: &TrainingLoop) -> Vec<usize> {
        loop_.epoch_checkpoints.iter().map(|(e, _)| *e).collect()
    }

    fn epoch0_local_dir(root_dir: &std::path::Path) -> std::path::PathBuf {
        root_dir
            .join("f2-retry-job")
            .join("f2-retry-worker")
            .join("0")
            .join("checkpoints")
            .join("epoch_0")
    }
}

/// H1 (unit 63): CPU-hermetic tests for the public per-pair held-out
/// evaluation seam — [`TrainingLoop::evaluate_held_out`] /
/// [`TrainingLoop::compute_loss_per_example`] and their supporting free
/// functions ([`mnrl_loss_per_example`], [`cross_entropy_per_row`]).
#[cfg(test)]
mod held_out_eval_tests {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::super::adamw::{AdamW, ParamsAdamW};
    use super::super::data::{TrainingBatch, TrainingDataLoader};
    use super::super::lora::{build_classification_head, build_projection_head};
    use super::super::target::TrainingTarget;
    use super::super::FineTuneConfig;
    use super::{TrainingLoop, TrainingLoopBuilder};
    use jammi_db::error::JammiError;

    const HIDDEN: usize = 2;
    const NUM_CLASSES: usize = 3;

    /// A minimal real [`TrainingLoop`] over a `Pairs`/MNRL-shaped
    /// `ProjectionHead` — mirrors `host_read_discipline::minimal_loop`. The
    /// held-out seam's `Pairs` path never touches `self.target`, so a bare,
    /// unclaimed catalog is enough (`evaluate_held_out`/`compute_loss` never
    /// reach the catalog either).
    async fn minimal_pairs_loop(device: &Device, config: FineTuneConfig) -> TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_projection_head(HIDDEN, &config, &varmap, &vb).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("held-out-pairs-job".into())
            .worker_id("held-out-pairs-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap()
    }

    fn ids(labels: &[&str]) -> Vec<String> {
        labels.iter().map(|s| s.to_string()).collect()
    }

    /// A 4-example, `batch_size = 2` `Pairs` held-out set: batch 0's anchor
    /// == positive on orthonormal unit rows (`cos ∈ {1, 0}`, scale 20 ⇒ a
    /// 20-wide logit gap — comfortably past f32's saturation threshold, see
    /// `cross_entropy_per_row`'s doc), so EVERY example in batch 0 saturates
    /// to an exact `0.0` floor. Batch 1 uses a smaller, non-saturating
    /// separation so its losses are strictly positive — a mixed fixture that
    /// exercises both the floor and the general case in one held-out set.
    fn mixed_pairs_fixture(device: &Device) -> (TrainingDataLoader, Vec<String>) {
        let saturated = TrainingBatch::Pairs {
            anchors: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device).unwrap(),
            positives: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device).unwrap(),
        };
        let unsaturated = TrainingBatch::Pairs {
            anchors: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device).unwrap(),
            positives: Tensor::new(&[[0.8f32, 0.6], [0.6, 0.8]], device).unwrap(),
        };
        let loader = TrainingDataLoader::from_precomputed(vec![saturated, unsaturated]);
        (loader, ids(&["ex-0", "ex-1", "ex-2", "ex-3"]))
    }

    /// All four examples on the saturated pattern above — every batch, every
    /// row, `cos = {1, 0}` — so `tie_fraction` must read exactly `1.0`.
    fn fully_saturated_pairs_fixture(device: &Device) -> (TrainingDataLoader, Vec<String>) {
        let batch = |device: &Device| TrainingBatch::Pairs {
            anchors: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device).unwrap(),
            positives: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device).unwrap(),
        };
        let loader = TrainingDataLoader::from_precomputed(vec![batch(device), batch(device)]);
        (loader, ids(&["ex-0", "ex-1", "ex-2", "ex-3"]))
    }

    fn mnrl_config(batch_size: usize) -> FineTuneConfig {
        FineTuneConfig {
            batch_size,
            embedding_loss: Some(super::super::EmbeddingLoss::MultipleNegativesRanking {
                temperature: 20.0,
            }),
            ..Default::default()
        }
    }

    /// Sum-consistency: `sum(per_example.loss) == mean * count`, targeting
    /// the SEAM's example-mean (not `evaluate`'s legacy batch-mean, which
    /// this fixture never calls).
    #[tokio::test(flavor = "multi_thread")]
    async fn sum_of_per_example_equals_mean_times_count() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let (loader, example_ids) = mixed_pairs_fixture(&device);

        let held_out = loop_.evaluate_held_out(&loader, &example_ids).unwrap();

        let sum: f64 = held_out.per_example.iter().map(|e| e.loss).sum();
        assert!(
            (sum - held_out.mean * held_out.count as f64).abs() < 1e-9,
            "sum {sum} must equal mean {} * count {} within f64 tolerance",
            held_out.mean,
            held_out.count
        );
    }

    /// `count == per_example.len()` on the seam's actual output (not just
    /// the wire type's construction-time invariant `jammi-wire` already pins).
    #[tokio::test(flavor = "multi_thread")]
    async fn count_matches_per_example_len() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let (loader, example_ids) = mixed_pairs_fixture(&device);

        let held_out = loop_.evaluate_held_out(&loader, &example_ids).unwrap();

        assert_eq!(held_out.count, held_out.per_example.len());
        assert_eq!(held_out.count, 4);
    }

    /// Determinism: two calls over the identical `(loader, ids)` produce a
    /// bitwise-identical `HeldOutLoss` — every per-example id and loss, the
    /// mean, the tie fraction, the partition hash, and the negatives count.
    #[tokio::test(flavor = "multi_thread")]
    async fn two_calls_are_bitwise_identical() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let (loader, example_ids) = mixed_pairs_fixture(&device);

        let first = loop_.evaluate_held_out(&loader, &example_ids).unwrap();
        let second = loop_.evaluate_held_out(&loader, &example_ids).unwrap();

        assert_eq!(first.count, second.count);
        assert_eq!(first.mean.to_bits(), second.mean.to_bits());
        assert_eq!(first.tie_fraction.to_bits(), second.tie_fraction.to_bits());
        assert_eq!(first.batch_partition_sha256, second.batch_partition_sha256);
        assert_eq!(
            first.in_batch_negatives_per_example,
            second.in_batch_negatives_per_example
        );
        assert_eq!(first.per_example.len(), second.per_example.len());
        for (a, b) in first.per_example.iter().zip(second.per_example.iter()) {
            assert_eq!(a.example_id, b.example_id);
            assert_eq!(a.loss.to_bits(), b.loss.to_bits());
        }
    }

    /// `tie_fraction == 1.0` on a genuinely saturated held-out split — every
    /// row's loss rounds to an EXACT `0.0` in `f32` (see
    /// `cross_entropy_per_row`'s doc), not merely close to it, so the
    /// fraction-at-floor computed from `f64 == 0.0` comparisons reads a
    /// clean `1.0` rather than falling just short of it.
    #[tokio::test(flavor = "multi_thread")]
    async fn tie_fraction_is_one_on_a_saturated_split() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let (loader, example_ids) = fully_saturated_pairs_fixture(&device);

        let held_out = loop_.evaluate_held_out(&loader, &example_ids).unwrap();

        assert_eq!(
            held_out.tie_fraction, 1.0,
            "every example must sit exactly at the saturated floor"
        );
        for example in &held_out.per_example {
            assert_eq!(
                example.loss, 0.0,
                "example '{}' must be an exact 0.0 floor, got {}",
                example.example_id, example.loss
            );
        }
    }

    /// Partition-hash stability: the SAME `(ids, order, batch_size)` always
    /// hashes to the same `batch_partition_sha256`, and a DIFFERENT batch
    /// grouping of the identical id set hashes to a different digest — the
    /// hash is sensitive to batch boundaries, not just the flat id list.
    #[tokio::test(flavor = "multi_thread")]
    async fn partition_hash_is_stable_and_boundary_sensitive() {
        let device = Device::Cpu;

        // Same partition (batch_size = 2, same fixture) called twice.
        let mut loop_2 = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let (loader, example_ids) = mixed_pairs_fixture(&device);
        let a = loop_2.evaluate_held_out(&loader, &example_ids).unwrap();
        let b = loop_2.evaluate_held_out(&loader, &example_ids).unwrap();
        assert_eq!(a.batch_partition_sha256, b.batch_partition_sha256);

        // Same 4 ids, same flat order, but a DIFFERENT batch_size (4 instead
        // of 2) regroups them into one batch instead of two — a different
        // partition over the identical id set.
        let saturated_one_batch = TrainingBatch::Pairs {
            anchors: Tensor::new(
                &[[1.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                &device,
            )
            .unwrap(),
            positives: Tensor::new(
                &[[1.0f32, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]],
                &device,
            )
            .unwrap(),
        };
        let mut loop_4 = minimal_pairs_loop(&device, mnrl_config(4)).await;
        let loader_one_batch = TrainingDataLoader::from_precomputed(vec![saturated_one_batch]);
        let c = loop_4
            .evaluate_held_out(&loader_one_batch, &example_ids)
            .unwrap();
        assert_ne!(
            a.batch_partition_sha256, c.batch_partition_sha256,
            "regrouping the identical id set into a different batch partition must change \
             the partition hash"
        );
    }

    /// Typed refusal: an empty held-out set (empty loader, empty ids) must
    /// not fabricate a `HeldOutLoss` over zero examples.
    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_an_empty_held_out_set() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let loader = TrainingDataLoader::from_precomputed(vec![]);

        let err = loop_.evaluate_held_out(&loader, &[]).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("empty")),
            "expected an 'empty held-out set' refusal, got {err:?}"
        );
    }

    /// Typed refusal: example-kind mismatch — a produced batch's row count
    /// does not match `config.batch_size`, so the committed id partition and
    /// the batch actually produced disagree about how rows group.
    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_a_batch_whose_row_count_mismatches_batch_size() {
        let device = Device::Cpu;
        // batch_size = 2, but the ONE precomputed batch carries all 4 rows
        // (never split into two batch_size-2 batches) — `example_ids.len()
        // == 4` is a multiple of `batch_size`, so this trips the PER-BATCH
        // row-count check specifically, not the leading multiple-of check.
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let batch = TrainingBatch::Pairs {
            anchors: Tensor::new(
                &[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                &device,
            )
            .unwrap(),
            positives: Tensor::new(
                &[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                &device,
            )
            .unwrap(),
        };
        let loader = TrainingDataLoader::from_precomputed(vec![batch]);
        let example_ids = ids(&["ex-0", "ex-1", "ex-2", "ex-3"]);

        let err = loop_.evaluate_held_out(&loader, &example_ids).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("kind mismatch")),
            "expected a batch-size kind-mismatch refusal, got {err:?}"
        );
    }

    /// Typed refusal: a non-finite per-example loss is refused, never folded
    /// into the mean. `positive` row 1 is `NaN`, which poisons that column of
    /// the similarity matrix and so every row's `log_sum_exp` in the batch —
    /// the seam must surface this as a typed error, not a `NaN` `mean`.
    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_a_non_finite_per_example_loss() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let batch = TrainingBatch::Pairs {
            anchors: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap(),
            positives: Tensor::new(&[[1.0f32, 0.0], [f32::NAN, f32::NAN]], &device).unwrap(),
        };
        let loader = TrainingDataLoader::from_precomputed(vec![batch]);
        let example_ids = ids(&["ex-0", "ex-1"]);

        let err = loop_.evaluate_held_out(&loader, &example_ids).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("non-finite")),
            "expected a non-finite-loss refusal, got {err:?}"
        );
    }

    fn weight_bytes(map: &std::collections::HashMap<String, Tensor>) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = map
            .iter()
            .map(|(name, t)| {
                let bits: Vec<u8> = t
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                (name.clone(), bits)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// A minimal real [`TrainingLoop`] over a `Classification`-shaped
    /// `ProjectionHead` with seeded dropout ON (`lora_dropout > 0.0`) — the
    /// no-RNG-perturbation fixture needs a target whose forward path
    /// actually draws a dropout mask when training, unlike the dropout-free
    /// `Pairs` fixture above.
    async fn minimal_classification_loop(
        device: &Device,
        config: FineTuneConfig,
    ) -> (TrainingLoop, VarMap) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_classification_head(HIDDEN, NUM_CLASSES, &config, &varmap, &vb).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        let loop_ = TrainingLoopBuilder::new(
            TrainingTarget::ProjectionHead { head },
            varmap.clone(),
            config,
        )
        .device(device.clone())
        .job_id("held-out-rng-job".into())
        .worker_id("held-out-rng-worker".into())
        .catalog(catalog)
        .artifact_dir(dir_path)
        .build()
        .unwrap();
        (loop_, varmap)
    }

    fn classification_batch(device: &Device) -> TrainingBatch {
        TrainingBatch::Classification {
            embeddings: Tensor::new(&[[1.0f32, 0.2], [0.1, -0.4]], device).unwrap(),
            labels: Tensor::new(&[0u32, 1u32], device).unwrap(),
        }
    }

    /// One real training micro-step: forward `compute_loss` (drawing the
    /// classifier layer's seeded dropout mask when training is on),
    /// backward, `AdamW::step`. Mirrors `resume_invariant::step_epoch`'s
    /// shape.
    fn train_step(loop_: &TrainingLoop, opt: &mut AdamW, device: &Device) {
        let batch = classification_batch(device);
        let loss = loop_.compute_loss(&batch).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }

    /// No-RNG-perturbation (H1, unit 63): a training run with a seam call
    /// interleaved between two of its steps is bitwise identical to the same
    /// run without it. `evaluate_held_out` disables dropout for its own
    /// forward via `with_dropout_disabled` — the SAME bracket `run()` uses
    /// around `evaluate` — so it draws no mask and leaves the seeded dropout
    /// stream exactly where the training forwards left it.
    #[tokio::test(flavor = "multi_thread")]
    async fn no_rng_perturbation_across_an_interleaved_seam_call() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            lora_dropout: 0.3,
            seed: 11,
            batch_size: 2,
            ..Default::default()
        };

        // Run A: 4 plain training steps.
        let (a_loop, a_varmap) = minimal_classification_loop(&device, config.clone()).await;
        let mut a_opt = AdamW::new(
            a_varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..4 {
            train_step(&a_loop, &mut a_opt, &device);
        }
        let w_a = a_loop.target.named_trainable_weights().unwrap();

        // Run B: 2 training steps, an interleaved `evaluate_held_out` call
        // (dropout-hot target, dropout-off forward), then 2 more training steps.
        let (mut b_loop, b_varmap) = minimal_classification_loop(&device, config.clone()).await;
        let mut b_opt = AdamW::new(
            b_varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();
        train_step(&b_loop, &mut b_opt, &device);
        train_step(&b_loop, &mut b_opt, &device);

        let held_out_loader =
            TrainingDataLoader::from_precomputed(vec![classification_batch(&device)]);
        let held_out_ids = ids(&["ho-0", "ho-1"]);
        let held_out = b_loop
            .evaluate_held_out(&held_out_loader, &held_out_ids)
            .unwrap();
        assert_eq!(held_out.count, 2);

        train_step(&b_loop, &mut b_opt, &device);
        train_step(&b_loop, &mut b_opt, &device);
        let w_b = b_loop.target.named_trainable_weights().unwrap();

        assert_eq!(
            weight_bytes(&w_a),
            weight_bytes(&w_b),
            "an interleaved evaluate_held_out call must not perturb the seeded dropout \
             stream — the post-seam training steps must be byte-identical to the same \
             steps without it"
        );
    }

    /// Audit round 63, finding 2 (RED-proof / regression test): a TYPED
    /// REFUSAL from `evaluate_held_out` mid-training must not leave the
    /// trainer stuck in eval mode. Pre-fix, `with_dropout_disabled`
    /// propagated `f`'s `Err` via `?` BEFORE the `set_training(true)` restore
    /// ran, so a refusal here (the leading "empty held-out set" check, which
    /// fires before any forward pass) left `self.target`'s dropout
    /// permanently OFF for the rest of the run — every subsequent training
    /// step would silently train with dropout disabled, with no error
    /// surfaced anywhere.
    ///
    /// Three loops, same seed/config prefix:
    /// - REF: 4 plain training steps, dropout on throughout.
    /// - TEST: 2 training steps, a REFUSED `evaluate_held_out` call, then 2
    ///   more training steps.
    /// - CONTROL: the SAME 2 training steps as TEST (identical trajectory,
    ///   identical dropout-stream position afterward), then dropout is
    ///   forced off DIRECTLY (`control_loop.target.set_training(false)`,
    ///   bypassing the seam) — reproducing exactly what finding 2's bug left
    ///   behind — before its own 2 more training steps.
    ///
    /// `TEST == REF` shows the refused call perturbs neither the RNG stream
    /// nor the training mode. `TEST != CONTROL` shows dropout genuinely still
    /// perturbs the tail two steps: CONTROL's tail is byte-for-byte what
    /// TEST's tail would be if finding 2's bug were still present (same
    /// prefix, same seed, dropout forced off for the tail), so if the fix
    /// regressed, `TEST != CONTROL` would fail because `TEST == CONTROL`.
    #[tokio::test(flavor = "multi_thread")]
    async fn refusal_mid_training_leaves_dropout_enabled() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            lora_dropout: 0.3,
            seed: 11,
            batch_size: 2,
            ..Default::default()
        };
        let opt_params = || ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        };

        // REF: 4 plain training steps, dropout on throughout.
        let (ref_loop, ref_varmap) = minimal_classification_loop(&device, config.clone()).await;
        let mut ref_opt = AdamW::new(ref_varmap.all_vars(), opt_params()).unwrap();
        for _ in 0..4 {
            train_step(&ref_loop, &mut ref_opt, &device);
        }
        let w_ref = ref_loop.target.named_trainable_weights().unwrap();

        // TEST: 2 steps, a refused evaluate_held_out call, 2 more steps.
        let (mut test_loop, test_varmap) =
            minimal_classification_loop(&device, config.clone()).await;
        let mut test_opt = AdamW::new(test_varmap.all_vars(), opt_params()).unwrap();
        train_step(&test_loop, &mut test_opt, &device);
        train_step(&test_loop, &mut test_opt, &device);

        let empty_loader = TrainingDataLoader::from_precomputed(vec![]);
        let err = test_loop.evaluate_held_out(&empty_loader, &[]).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("empty")),
            "expected an 'empty held-out set' refusal, got {err:?}"
        );

        train_step(&test_loop, &mut test_opt, &device);
        train_step(&test_loop, &mut test_opt, &device);
        let w_test = test_loop.target.named_trainable_weights().unwrap();

        // CONTROL: the identical 2-step prefix, then dropout forced off
        // directly (bypassing the seam) — the observable shape of finding
        // 2's bug — before 2 more (now dropout-off) training steps.
        let (mut control_loop, control_varmap) =
            minimal_classification_loop(&device, config.clone()).await;
        let mut control_opt = AdamW::new(control_varmap.all_vars(), opt_params()).unwrap();
        train_step(&control_loop, &mut control_opt, &device);
        train_step(&control_loop, &mut control_opt, &device);
        control_loop.target.set_training(false);
        train_step(&control_loop, &mut control_opt, &device);
        train_step(&control_loop, &mut control_opt, &device);
        let w_control = control_loop.target.named_trainable_weights().unwrap();

        assert_eq!(
            weight_bytes(&w_test),
            weight_bytes(&w_ref),
            "a refused evaluate_held_out call mid-training must not perturb the seeded \
             dropout stream or leave the trainer eval-mode — the post-refusal training \
             steps must be byte-identical to the same steps in a run that never called \
             the seam"
        );
        assert_ne!(
            weight_bytes(&w_test),
            weight_bytes(&w_control),
            "TEST's post-refusal tail must diverge from CONTROL's forced-eval-mode tail \
             — if finding 2's bug were still present, TEST's tail would ALSO train with \
             dropout off (matching CONTROL exactly) and this assertion would fail because \
             TEST == CONTROL"
        );
    }

    /// Audit round 63, finding 2 (regression test): calling the held-out seam
    /// on a trainer that was NOT in training mode to begin with must not flip
    /// it INTO training mode as a side effect. Pre-fix,
    /// `with_dropout_disabled` unconditionally restored `training = true`
    /// regardless of the trainer's actual pre-call state, so a trainer
    /// explicitly placed in eval mode (e.g. an inference-only handle) would
    /// come OUT of a read-only `evaluate_held_out` call silently back in
    /// training mode.
    #[tokio::test(flavor = "multi_thread")]
    async fn seam_on_a_non_training_trainer_leaves_it_non_training() {
        let device = Device::Cpu;
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;

        // Explicitly place the trainer in eval mode BEFORE the seam call —
        // the scenario the pre-fix hard-coded `set_training(true)` restore
        // ignored.
        loop_.set_training(false);
        assert!(
            !loop_.training_mode,
            "test setup: trainer must start non-training"
        );

        let (loader, example_ids) = mixed_pairs_fixture(&device);
        let held_out = loop_.evaluate_held_out(&loader, &example_ids).unwrap();
        assert_eq!(held_out.count, 4);

        assert!(
            !loop_.training_mode,
            "evaluate_held_out must restore the CAPTURED pre-call training mode \
             (false here), not unconditionally force training back on"
        );
    }

    /// Typed refusal: `example_ids.len()` is not a multiple of `batch_size`
    /// (the v2-delta-2 leading guard) — untested before this unit;
    /// `refuses_a_batch_whose_row_count_mismatches_batch_size` above covers
    /// only the DIFFERENT per-batch row-count check further down.
    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_a_held_out_set_not_a_multiple_of_batch_size() {
        let device = Device::Cpu;
        // batch_size = 2, but 3 example ids — 3 is not a multiple of 2.
        let mut loop_ = minimal_pairs_loop(&device, mnrl_config(2)).await;
        let batch = TrainingBatch::Pairs {
            anchors: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device).unwrap(),
            positives: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device).unwrap(),
        };
        let loader = TrainingDataLoader::from_precomputed(vec![batch]);
        let example_ids = ids(&["ex-0", "ex-1", "ex-2"]);

        let err = loop_.evaluate_held_out(&loader, &example_ids).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("is not a multiple of batch_size")),
            "expected a 'not a multiple of batch_size' refusal, got {err:?}"
        );
    }
}

/// Re-audit round 63, re-audit finding 1 (RED-proof / regression tests): the
/// `training_mode` mirror doc claimed "every `TrainingTarget` this crate
/// constructs for training starts in training mode" and hard-coded
/// `training_mode: true` at [`TrainingLoopBuilder::build`] on the strength of
/// that claim. It was false for `TrainingTarget::EncoderAdapters`: its
/// `ModernBert` body is constructed `training: false` (only the injected
/// `LoraLinear` adapters start `true`), so the target's real state was
/// heterogeneous and every restore path (`with_dropout_disabled`, the mining
/// and GradCache brackets) read a fabricated `true`.
///
/// These tests build a REAL `EncoderAdapters` target — the smallest
/// constructible one, the checked-in `tests/fixtures/tiny_modernbert` config +
/// weights also used by `tests/it/encoder_adapters.rs` — and read the
/// encoder's own [`jammi_encoders::ModernBert::is_training`] getter directly,
/// never trusting `TrainingLoop::training_mode` as ground truth (that mirror
/// is exactly the thing under test).
///
/// RED-proof (performed manually against this diff, not committed):
/// hard-coding `training_mode: true` and dropping the `set_training(true)`
/// call from `build` (the exact pre-fix shape) reddens THREE of the four
/// tests below: `build_puts_the_encoder_body_into_training_mode` fails
/// directly (`mb.is_training()` reads `false` right after `build`), and both
/// `with_dropout_disabled_restores_real_encoder_state_on_{ok,err}` fail their
/// own setup assertion (`real_encoder_is_training(&loop_)` before the seam
/// call is already `false`, since the loop never entered training to begin
/// with) — not because `with_dropout_disabled`'s restore logic is wrong (it
/// was already fixed correctly in the prior round), but because it never ran
/// against a target whose real initial state disagreed with the mirror's
/// claim. `refusal_path_leaves_a_non_training_encoder_non_training` stays
/// green even pre-fix: it calls `set_training(false)` explicitly before
/// exercising the refusal path, which forces the real encoder state
/// regardless of what `build` left it at — that test guards a different
/// (already-correct) behaviour, not this finding.
#[cfg(test)]
mod encoder_adapters_training_state_tests {
    use std::path::Path;
    use std::sync::Arc;

    use candle_core::Device;
    use candle_nn::VarMap;
    use jammi_db::error::JammiError;
    use jammi_encoders::{AnyEncoder, ModernBert, ModernBertConfig, Pooling};
    use jammi_lora::{AdapterConfig, LoraBuildConfig, LoraInitMode};

    use super::super::data::TrainingDataLoader;
    use super::super::target::{EncoderAdaptersTarget, TrainingTarget};
    use super::super::FineTuneConfig;
    use super::{TrainingLoop, TrainingLoopBuilder};

    /// The repo-root `tests/fixtures/tiny_modernbert` dir — the same
    /// smallest-constructible ModernBERT config + weights
    /// `tests/it/encoder_adapters.rs` fine-tunes end-to-end. `CARGO_MANIFEST_DIR`
    /// is `crates/jammi-ai`; `tests/fixtures` sits two levels up, at the
    /// workspace root (mirrors `tests/gpu_capability/harness.rs::fixture`).
    pub(super) fn tiny_modernbert_fixture_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures")
            .join("tiny_modernbert")
    }

    /// Build a real `TrainingTarget::EncoderAdapters` over the tiny-ModernBERT
    /// fixture, with LoRA injected into `Wqkv`/`Wo` (ModernBERT's fused
    /// attention linears — the same pair `tests/it/encoder_adapters.rs` uses)
    /// and dropout ON, so a real `LoraLinear` dropout gate is exercised
    /// alongside the encoder body's own `training` flag.
    pub(super) fn build_encoder_adapters_target(
        device: &Device,
        varmap: &VarMap,
    ) -> TrainingTarget {
        let dir = tiny_modernbert_fixture_dir();
        let config_raw = std::fs::read_to_string(dir.join("config.json"))
            .expect("tiny_modernbert fixture config.json must be readable");
        let model_config: ModernBertConfig =
            serde_json::from_str(&config_raw).expect("tiny_modernbert config.json must parse");
        let weights = dir.join("model.safetensors");

        let target_modules = vec!["Wqkv".to_string(), "Wo".to_string()];
        let empty_ranks = std::collections::HashMap::new();
        let lora = LoraBuildConfig {
            target_modules: &target_modules,
            layers_to_transform: &None,
            lora_rank: 4,
            lora_alpha: 8.0,
            use_rslora: false,
            lora_dropout: Some(0.3),
            rank_pattern: &empty_ranks,
            init_mode: LoraInitMode::ZerosB,
            seed: 7,
        };
        let adapter_cfg = AdapterConfig::from_build(
            "modernbert",
            &lora,
            jammi_numerics::ComputePrecision::default(),
        );
        let encoder: ModernBert = ModernBert::builder()
            .pooling(Pooling::Mean)
            .lora(lora)
            .build(&[weights.as_path()], &model_config, device, varmap)
            .expect("tiny_modernbert fixture must build");

        TrainingTarget::EncoderAdapters(Box::new(EncoderAdaptersTarget {
            encoder: AnyEncoder::ModernBert(encoder),
            adapter_cfg,
        }))
    }

    /// A minimal real `TrainingLoop` over the `EncoderAdapters` target above.
    async fn minimal_encoder_adapters_loop(device: &Device) -> TrainingLoop {
        let varmap = VarMap::new();
        let target = build_encoder_adapters_target(device, &varmap);
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        TrainingLoopBuilder::new(target, varmap, FineTuneConfig::default())
            .device(device.clone())
            .job_id("encoder-adapters-training-state-job".into())
            .worker_id("encoder-adapters-training-state-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap()
    }

    /// Read the encoder body's OWN training flag — never `loop_.training_mode`
    /// (the mirror under test) — by reaching through the real `AnyEncoder`.
    fn real_encoder_is_training(loop_: &TrainingLoop) -> bool {
        match &loop_.target {
            TrainingTarget::EncoderAdapters(state) => match &state.encoder {
                AnyEncoder::ModernBert(mb) => mb.is_training(),
                _ => panic!("fixture must build an AnyEncoder::ModernBert"),
            },
            TrainingTarget::ProjectionHead { .. } => {
                panic!("fixture must build an EncoderAdapters target")
            }
        }
    }

    /// (a) RED-PROOF: post-`build`, the encoder BODY (not just its injected
    /// `LoraLinear` adapters) must genuinely be in training mode. Pre-fix,
    /// `build` never called `set_training` on the assembled loop — it just
    /// wrote the literal `training_mode: true` into the mirror — so
    /// `ModernBert`'s own `training: false` at construction stood unchanged
    /// and this assertion reads `false`.
    #[tokio::test(flavor = "multi_thread")]
    async fn build_puts_the_encoder_body_into_training_mode() {
        let device = Device::Cpu;
        let loop_ = minimal_encoder_adapters_loop(&device).await;
        assert!(
            real_encoder_is_training(&loop_),
            "TrainingLoopBuilder::build must switch an EncoderAdapters target's \
             encoder body into training mode, not just its injected LoRA adapters"
        );
        assert!(
            loop_.training_mode,
            "the training_mode mirror must agree with the real encoder state"
        );
    }

    /// (b) `with_dropout_disabled` restores the TRUE pre-call state on the
    /// `Ok` arm, verified against the encoder's own flag (not the mirror).
    #[tokio::test(flavor = "multi_thread")]
    async fn with_dropout_disabled_restores_real_encoder_state_on_ok() {
        let device = Device::Cpu;
        let mut loop_ = minimal_encoder_adapters_loop(&device).await;
        assert!(real_encoder_is_training(&loop_), "must start training");

        let result = loop_.with_dropout_disabled(|_| Ok(42));
        assert_eq!(result.unwrap(), 42);
        assert!(
            real_encoder_is_training(&loop_),
            "with_dropout_disabled must restore the encoder body to its true \
             pre-call training state on the Ok arm"
        );
        assert!(loop_.training_mode, "mirror must agree with real state");
    }

    /// (b) `with_dropout_disabled` restores the TRUE pre-call state on the
    /// `Err` arm too — the restore must run before the error propagates.
    #[tokio::test(flavor = "multi_thread")]
    async fn with_dropout_disabled_restores_real_encoder_state_on_err() {
        let device = Device::Cpu;
        let mut loop_ = minimal_encoder_adapters_loop(&device).await;
        assert!(real_encoder_is_training(&loop_), "must start training");

        let result: Result<(), JammiError> =
            loop_.with_dropout_disabled(|_| Err(JammiError::FineTune("probe refusal".into())));
        assert!(result.is_err());
        assert!(
            real_encoder_is_training(&loop_),
            "with_dropout_disabled must restore the encoder body to its true \
             pre-call training state on the Err arm too, before the error \
             propagates"
        );
        assert!(loop_.training_mode, "mirror must agree with real state");
    }

    /// (c) The non-fabricated analog of the prior round's
    /// `seam_on_a_non_training_trainer_leaves_it_non_training`: on an
    /// `EncoderAdapters` target explicitly placed in eval mode, a REFUSED
    /// `evaluate_held_out` call (the leading "empty held-out set" check, fired
    /// before any forward) must not flip the encoder body back into training
    /// mode as a side effect.
    #[tokio::test(flavor = "multi_thread")]
    async fn refusal_path_leaves_a_non_training_encoder_non_training() {
        let device = Device::Cpu;
        let mut loop_ = minimal_encoder_adapters_loop(&device).await;

        loop_.set_training(false);
        assert!(
            !real_encoder_is_training(&loop_),
            "test setup: encoder body must start non-training"
        );

        let empty_loader = TrainingDataLoader::from_precomputed(vec![]);
        let err = loop_.evaluate_held_out(&empty_loader, &[]).unwrap_err();
        assert!(
            matches!(err, JammiError::FineTune(ref m) if m.contains("empty")),
            "expected an 'empty held-out set' refusal, got {err:?}"
        );

        assert!(
            !real_encoder_is_training(&loop_),
            "a refused evaluate_held_out call must not flip a non-training \
             encoder body INTO training mode as a side effect"
        );
        assert!(!loop_.training_mode, "mirror must agree with real state");
    }
}

/// F4 (adversarial-audit round 2, campaign #443): a production-call-site
/// oracle for esc-076's bucketing fix (`.jammi/escapes.jsonl`).
///
/// Every existing bucketing test lives in
/// `crate::fine_tune::batch_bucket`'s own unit-test module and calls
/// `pad_rows_to_bucket`/`bucket_seq_len` directly — none of them drive
/// [`TrainingLoop::encode_texts`]'s `EncoderAdapters` branch, the ONLY
/// production call site (via this file's own `tokenize_and_bucket`, its
/// sole caller). The prior audit round proved deleting both
/// `pad_rows_to_bucket` calls there left every test in the crate green.
/// This module closes that gap: `tokenize_and_bucket_pads_every_row_to_the_
/// bucket_ladder` drives `tokenize_and_bucket` itself against a real
/// tokenizer and asserts the returned rows are actually padded to the
/// bucket ladder (RED if either `pad_rows_to_bucket` call is deleted), and
/// `encode_texts_output_is_bucket_invariant_at_the_real_call_site` proves
/// that padding does not move the real, production `encode_texts` output
/// versus an independently-built natural-width forward pass.
#[cfg(test)]
mod encode_texts_bucketing_oracle {
    use std::sync::Arc;

    use candle_core::{Device, Tensor};
    use candle_nn::VarMap;
    use serial_test::serial;

    use super::super::target::TrainingTarget;
    use super::super::FineTuneConfig;
    use super::encoder_adapters_training_state_tests::build_encoder_adapters_target;
    use super::TrainingLoopBuilder;
    use crate::model::{LoadedModel, ModelSource, ModelTask};

    // The three tests below all call into `tokenize_and_bucket`/
    // `tokenize_natural_width` (directly, or indirectly via
    // `TrainingLoop::encode_texts`'s `EncoderAdapters` branch), which
    // increment the process-wide `BUCKETED_TOKENIZE_CALLS`/
    // `NATURAL_TOKENIZE_CALLS` test-only counters (c) reads. `cargo test`
    // runs tests in parallel threads within the SAME process, so an
    // unmarked set racing on those counters would be flaky — `#[serial(..)]`
    // under a shared key forces them to run one at a time relative to each
    // other, mirroring this file's own `trainer_host_read_count` precedent.

    /// Real `LoadedModel` (the `base_model` `encode_texts` reads its
    /// tokenizer from) for the SAME `tiny_modernbert` fixture the
    /// `EncoderAdapters` target below is built from — shared vocab/config,
    /// so the tokenizer's emitted token ids are valid inputs to the
    /// encoder's own embedding table. Mirrors
    /// `test_fixtures::tiny_bert`'s real model-cache load path, substituting
    /// `tiny_modernbert_fixture_dir` for the cookbook `tiny_bert` fixture.
    async fn tiny_modernbert_base_model() -> Arc<LoadedModel> {
        let dir = tempfile::tempdir().unwrap();
        let config = jammi_test_utils::test_config(dir.path());
        let session = crate::session::InferenceSession::new(config).await.unwrap();
        let source = ModelSource::Local(
            super::encoder_adapters_training_state_tests::tiny_modernbert_fixture_dir(),
        );
        let guard = session
            .model_cache()
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .unwrap();
        guard.model.clone()
    }

    fn tokenizer_of(base: &LoadedModel) -> &crate::model::tokenizer::TokenizerWrapper {
        match base {
            LoadedModel::Candle(m) => m.tokenizer.as_ref().expect("fixture ships a tokenizer"),
            _ => panic!("tiny_modernbert fixture must load as a Candle model"),
        }
    }

    /// `tiny_modernbert`'s own `max_position_embeddings` (see its committed
    /// `config.json`) — `encode_texts`'s own `effective_max =
    /// self.config.max_seq_length.min(encoder.max_seq_length())` reduces to
    /// this value under `FineTuneConfig::default()`'s `max_seq_length:
    /// 512`.
    const EFFECTIVE_MAX: usize = 128;

    /// Two rows whose tokenizer-emitted natural width (after `[CLS]`/`[SEP]`
    /// and WordPiece per-letter tokens, then the batch's own `BatchLongest`
    /// intra-batch padding) lands strictly between two `bucket_seq_len`
    /// rungs — verified in the test below via the SAME decision
    /// `tokenize_and_bucket` uses, never hand-asserted: `"a b c d e f g h
    /// i"` tokenizes to `[CLS] a b c d e f g h i [SEP]` = 11 tokens, and
    /// `bucket_seq_len(11, 128) == 16` (`> 11`, so this batch genuinely
    /// exercises padding).
    fn ragged_texts() -> Vec<String> {
        vec!["a b c d e f g h i".to_string(), "a".to_string()]
    }

    /// (a) The production oracle: calling `tokenize_and_bucket` — the exact
    /// helper `TrainingLoop::encode_texts`'s `EncoderAdapters` branch calls,
    /// its only caller — against a real tokenizer must return rows extended
    /// to `bucket_seq_len`'s ladder, strictly wider than the batch's own
    /// natural (tokenizer `BatchLongest`) width.
    ///
    /// RED-PROOF: deleting either `pad_rows_to_bucket` call inside
    /// `tokenize_and_bucket` leaves every row at its natural width, so
    /// `row.len() == cols` fails below (`cols` is still computed from
    /// `bucket_seq_len` independently of whether the rows were actually
    /// extended to it — the assertion cannot pass vacuously).
    #[tokio::test(flavor = "multi_thread")]
    #[serial(tokenize_dispatch_calls)]
    async fn tokenize_and_bucket_pads_every_row_to_the_bucket_ladder() {
        let base = tiny_modernbert_base_model().await;
        let tokenizer = tokenizer_of(&base);

        let texts = ragged_texts();
        let (encoding, rows, cols) =
            super::tokenize_and_bucket(tokenizer, &texts, EFFECTIVE_MAX).unwrap();

        assert_eq!(rows, texts.len());

        // Natural width independently recomputed via a SEPARATE, unbucketed
        // `encode_batch` call — never read off `encoding` itself, since
        // that is the very value under test.
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let natural = tokenizer
            .encode_batch(&text_refs, Some(EFFECTIVE_MAX))
            .unwrap();
        let natural_cols = natural.input_ids[0].len();
        assert!(
            cols > natural_cols,
            "fixture texts must produce a non-bucket-aligned natural width \
             ({natural_cols}) so this test actually exercises padding, got bucketed={cols}"
        );
        assert_eq!(
            cols,
            jammi_numerics::bucket_seq_len(natural_cols, EFFECTIVE_MAX),
            "cols must be the bucket_seq_len decision for this batch's own natural width"
        );
        for (i, row) in encoding.input_ids.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "input_ids row {i} must be padded to the bucketed width {cols}, got {}",
                row.len()
            );
        }
        for (i, row) in encoding.attention_masks.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "attention_mask row {i} must be padded to the bucketed width {cols}, got {}",
                row.len()
            );
            // Every position beyond the natural width must be fully masked
            // (K2/K7 correctness — never a wrong answer wearing a fixed
            // shape).
            for &m in &row[natural_cols..] {
                assert_eq!(m, 0, "padded tail of attention_mask row {i} must be masked");
            }
        }
    }

    /// (b) Output-invariance at the REAL call site: [`TrainingLoop::encode_
    /// texts`]'s `EncoderAdapters` branch, driven end-to-end through a real
    /// `TrainingLoop`, must produce the same pooled output as an
    /// independently-built forward pass over the batch's own NATURAL
    /// (unbucketed) width.
    ///
    /// The comparison encoder is a second, independently-loaded instance of
    /// the identical `tiny_modernbert` fixture (`LoraInitMode::ZerosB`
    /// means the injected LoRA branch contributes exactly zero at this
    /// freshly-built, untrained state — regardless of dropout, since `B =
    /// 0` zeroes the whole `B * dropout(A * x)` term — so two independently
    /// loaded encoders over the same frozen base weights agree bit-for-bit
    /// on a pure base forward pass); this isolates the bucketing effect
    /// from any risk of the comparison silently reusing `encode_texts`'s
    /// own internals.
    ///
    /// Tolerance: `1e-4` absolute per lane, derived from this fixture's
    /// `hidden_size=32` F32 accumulation error over an attention softmax
    /// reduction whose extra (fully-masked) columns still enter the
    /// max/sum reduction before their `exp()` drives them to ~0 — looser
    /// than `batch_bucket.rs`'s own `1e-5` (a hand-built fixture with a
    /// narrower padded tail, 8 vs this fixture's 16) to account for the
    /// wider padded tail here, and orders of magnitude tighter than any
    /// real bug (a wrong mask, wrong pad id, or a divisor that counted
    /// padding) would produce, which collapses agreement completely.
    #[tokio::test(flavor = "multi_thread")]
    #[serial(tokenize_dispatch_calls)]
    async fn encode_texts_output_is_bucket_invariant_at_the_real_call_site() {
        let device = Device::Cpu;
        let base_model = tiny_modernbert_base_model().await;
        let texts = ragged_texts();

        // The REAL, bucketed path: exactly what a training step calls.
        let varmap = VarMap::new();
        let target = build_encoder_adapters_target(&device, &varmap);
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        let loop_ = TrainingLoopBuilder::new(target, varmap, FineTuneConfig::default())
            .device(device.clone())
            .base_model(base_model.clone())
            .job_id("encode-texts-bucketing-oracle-job".into())
            .worker_id("encode-texts-bucketing-oracle-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap();
        let bucketed_out = loop_
            .encode_texts(&texts)
            .expect("bucketed encode_texts must succeed")
            .to_vec2::<f32>()
            .unwrap();

        // The natural (unbucketed) comparison.
        let varmap2 = VarMap::new();
        let target2 = build_encoder_adapters_target(&device, &varmap2);
        let encoder2 = match &target2 {
            TrainingTarget::EncoderAdapters(state) => &state.encoder,
            TrainingTarget::ProjectionHead { .. } => {
                unreachable!("build_encoder_adapters_target always builds EncoderAdapters")
            }
        };
        let tokenizer = tokenizer_of(&base_model);
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let natural = tokenizer
            .encode_batch(&text_refs, Some(EFFECTIVE_MAX))
            .unwrap();
        let rows = natural.input_ids.len();
        let cols = natural.input_ids[0].len();
        let to_tensor = |data: Vec<Vec<u32>>| -> Tensor {
            Tensor::from_vec(
                data.into_iter().flatten().collect::<Vec<u32>>(),
                (rows, cols),
                &device,
            )
            .unwrap()
        };
        let input_ids = to_tensor(natural.input_ids);
        let attention_mask = to_tensor(natural.attention_masks);
        let natural_out = encoder2
            .forward(&input_ids, &attention_mask)
            .expect("natural-width forward must succeed")
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(bucketed_out.len(), natural_out.len());
        const TOLERANCE: f32 = 1e-4;
        for (row_i, (b_row, n_row)) in bucketed_out.iter().zip(&natural_out).enumerate() {
            assert_eq!(b_row.len(), n_row.len());
            for (lane, (b, n)) in b_row.iter().zip(n_row).enumerate() {
                assert!(
                    (b - n).abs() <= TOLERANCE,
                    "row {row_i} lane {lane}: bucketed={b} natural={n} differ by {} > \
                     tolerance {TOLERANCE} -- bucketing at the real `encode_texts` call site \
                     must not move the pooled output",
                    (b - n).abs()
                );
            }
        }
    }

    /// (c) esc-076 eval amendment (adversarial-audit round 2, campaign
    /// #443, item 3): `encode_texts`'s `EncoderAdapters` branch must
    /// dispatch to [`super::tokenize_and_bucket`] while `self.training_mode
    /// == true` and to [`super::tokenize_natural_width`] while it is
    /// `false` — proven via the process-wide call counters
    /// ([`super::BUCKETED_TOKENIZE_CALLS`]/[`super::NATURAL_TOKENIZE_CALLS`])
    /// since both functions are, by design, output-invariant (bucketing a
    /// batch does not change its pooled result — test (b) above), so a
    /// black-box comparison of `encode_texts`'s RETURN VALUE cannot tell
    /// which path actually ran.
    ///
    /// RED-PROOF: hard-coding `encode_texts`'s `EncoderAdapters` branch to
    /// always call `tokenize_and_bucket` (dropping the `if
    /// self.training_mode` dispatch this fix adds) turns this test red at
    /// its eval-mode counter assertions — `NATURAL_TOKENIZE_CALLS` would
    /// stay at its pre-call snapshot while `BUCKETED_TOKENIZE_CALLS` moves
    /// instead.
    #[tokio::test(flavor = "multi_thread")]
    #[serial(tokenize_dispatch_calls)]
    async fn encode_texts_dispatches_on_training_mode_between_bucketed_and_natural_tokenize() {
        use std::sync::atomic::Ordering;

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let target = build_encoder_adapters_target(&device, &varmap);
        let base_model = tiny_modernbert_base_model().await;
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        let mut loop_ = TrainingLoopBuilder::new(target, varmap, FineTuneConfig::default())
            .device(device.clone())
            .base_model(base_model.clone())
            .job_id("encode-texts-eval-width-oracle-job".into())
            .worker_id("encode-texts-eval-width-oracle-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap();
        assert!(
            loop_.training_mode,
            "TrainingLoopBuilder::build must leave the loop in training mode"
        );

        let texts = ragged_texts();

        // Train mode (the loop's own post-`build` state): must dispatch to
        // `tokenize_and_bucket`, never `tokenize_natural_width`.
        let bucketed_before = super::BUCKETED_TOKENIZE_CALLS.load(Ordering::Relaxed);
        let natural_before = super::NATURAL_TOKENIZE_CALLS.load(Ordering::Relaxed);
        loop_
            .encode_texts(&texts)
            .expect("train-mode encode_texts must succeed");
        assert_eq!(
            super::BUCKETED_TOKENIZE_CALLS.load(Ordering::Relaxed),
            bucketed_before + 1,
            "train-mode encode_texts must call tokenize_and_bucket exactly once"
        );
        assert_eq!(
            super::NATURAL_TOKENIZE_CALLS.load(Ordering::Relaxed),
            natural_before,
            "train-mode encode_texts must NOT call tokenize_natural_width"
        );

        // Flip to eval mode via the SAME production seam
        // `evaluate`/`evaluate_held_out` use (`with_dropout_disabled` ->
        // `set_training(false)`), never a raw field write.
        loop_.set_training(false);
        let bucketed_before = super::BUCKETED_TOKENIZE_CALLS.load(Ordering::Relaxed);
        let natural_before = super::NATURAL_TOKENIZE_CALLS.load(Ordering::Relaxed);
        loop_
            .encode_texts(&texts)
            .expect("eval-mode encode_texts must succeed");
        assert_eq!(
            super::NATURAL_TOKENIZE_CALLS.load(Ordering::Relaxed),
            natural_before + 1,
            "eval-mode encode_texts must call tokenize_natural_width exactly once"
        );
        assert_eq!(
            super::BUCKETED_TOKENIZE_CALLS.load(Ordering::Relaxed),
            bucketed_before,
            "eval-mode encode_texts must NOT call tokenize_and_bucket (esc-076 eval amendment \
             — bucketing an eval batch up to the run's max_seq_length bucket measurably OOM'd \
             a shape that ran clean pre-bucketing; see tokenize_natural_width's own doc)"
        );

        // Sanity: this fixture's texts really do produce a bucket/natural
        // gap, so the counters above are distinguishing a REAL difference,
        // not two paths that happen to coincide for this input.
        let tokenizer = tokenizer_of(&base_model);
        let (_, _, bucketed_cols) =
            super::tokenize_and_bucket(tokenizer, &texts, EFFECTIVE_MAX).unwrap();
        let (_, _, natural_cols) =
            super::tokenize_natural_width(tokenizer, &texts, EFFECTIVE_MAX).unwrap();
        assert!(
            bucketed_cols > natural_cols,
            "fixture must produce a real bucket ({bucketed_cols}) vs natural ({natural_cols}) \
             gap for this test to be meaningful"
        );
    }

    /// (d) r3 finding B4: pins argument (a) of `tokenize_natural_width`'s
    /// own doc — eval's distinct-shape contribution to the allocator is
    /// paid ONCE per run because the held-out/val partition presents the
    /// IDENTICAL sequence of natural widths on every pass, never a
    /// reshuffled or re-ordered one. `tokenize_natural_width` itself is a
    /// pure function of its input texts (tokenization has no randomness),
    /// so this cannot catch a bug INSIDE it — what it CAN catch is a caller
    /// that fed a re-ordered/re-partitioned split across passes (which
    /// would invalidate the "paid once" argument the doc above relies on):
    /// encodes the SAME ordered, multi-batch split — mirroring
    /// `evaluate_held_out`'s own fixed `example_ids` partition — batch by
    /// batch, across two independent "passes", and asserts the per-batch
    /// natural-width SEQUENCE is identical.
    #[tokio::test(flavor = "multi_thread")]
    #[serial(tokenize_dispatch_calls)]
    async fn eval_tokenize_natural_width_repeats_the_same_width_sequence_across_passes_over_the_same_split(
    ) {
        let base_model = tiny_modernbert_base_model().await;
        let tokenizer = tokenizer_of(&base_model);

        // A 3-batch split with deliberately different natural widths per
        // batch (mirroring how a real held-out set groups rows of varying
        // length) — a width-SEQUENCE comparison across passes is only
        // meaningful if the sequence itself has more than one distinct
        // value.
        let split: Vec<Vec<String>> = vec![
            vec!["a".to_string(), "a b".to_string()],
            ragged_texts(),
            vec!["a b c d e f g h i j k l m n o p".to_string()],
        ];

        let widths_for_one_pass = |split: &[Vec<String>]| -> Vec<usize> {
            split
                .iter()
                .map(|batch| {
                    let (_, _, cols) =
                        super::tokenize_natural_width(tokenizer, batch, EFFECTIVE_MAX).unwrap();
                    cols
                })
                .collect()
        };

        let pass_1 = widths_for_one_pass(&split);
        let pass_2 = widths_for_one_pass(&split);
        assert_eq!(
            pass_1, pass_2,
            "the SAME ordered split must produce the IDENTICAL per-batch natural-width \
             sequence across repeated passes -- this determinism is what \
             tokenize_natural_width's own doc relies on to argue eval's distinct-shape \
             contribution is paid once per run, never once per pass"
        );
        // Sanity: the split's own widths actually vary, so pass_1 == pass_2
        // is not a vacuous single-element-sequence agreement.
        assert!(
            pass_1
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1,
            "split fixture must present more than one distinct natural width for this test \
             to be meaningful, got {pass_1:?}"
        );
    }
}

/// Finding 7 (audit round 63): a per-objective decomposition ORACLE for every
/// objective [`TrainingLoop::compute_loss_per_example`] supports.
///
/// The pre-existing oracle
/// (`held_out_eval_tests::sum_of_per_example_equals_mean_times_count`) only
/// checks the per-example seam against ITSELF — `sum(per_example) == mean *
/// count` — which is trivially true BY CONSTRUCTION of how `evaluate_held_out`
/// computes `mean` (`sum / count`, from the very same `per_example` array). It
/// would pass unchanged even if `compute_loss_per_example` scored every row
/// under a completely different (or wrong) formula, as long as it summed and
/// averaged internally consistently — it cannot catch a per-example function
/// that has drifted from the real, production objective.
///
/// These tests instead compare `mean(compute_loss_per_example(batch))` — an
/// INDEPENDENT computation — against `compute_loss(batch)`, the batch-mean
/// tensor path every real training step and `Trainer::evaluate` actually
/// takes. This bites: it is only true if the per-example decomposition
/// genuinely reduces to the same objective the tensor path computes, for
/// every objective this seam supports (MNRL over `Pairs`, MNRL over a
/// `Triplet` batch, Triplet-margin, Classification, Contrastive/CosineMse —
/// the same five arms [`TrainingLoop::compute_loss_per_example`]'s own doc
/// enumerates as supported; `CoSENT`/`AnglE`/`Ner`/`Regression` are typed
/// refusals there, not decompositions, so they have no oracle here).
///
/// RED-proof (performed manually against this diff, not committed): stashing
/// a one-line perturbation into `mnrl_loss_per_example` (adding a constant to
/// each returned row) reddened `mnrl_pairs_decomposition_matches_compute_loss`
/// and `mnrl_triplet_consumed_decomposition_matches_compute_loss` while every
/// other oracle in this module stayed green — confirming the oracle actually
/// exercises the code path it claims to, not a vacuous pass.
#[cfg(test)]
mod decomposition_oracle_tests {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::super::data::TrainingBatch;
    use super::super::lora::{build_classification_head, build_projection_head};
    use super::super::target::TrainingTarget;
    use super::super::{EmbeddingLoss, FineTuneConfig};
    use super::{TrainingLoop, TrainingLoopBuilder};

    const HIDDEN: usize = 2;
    const NUM_CLASSES: usize = 3;

    /// Relative tolerance for comparing the tensor-path `compute_loss` (an f32
    /// device computation, cast to f32 then read back as f64) against
    /// `mean(compute_loss_per_example(...))` (an f64 host reduction over
    /// per-row f32 device reads). Both paths run the SAME arithmetic
    /// (`cosine_similarity` / `contiguous_matmul` / softmax / log-sum-exp) on
    /// f32 tensors, so they can only diverge in floating-point SUMMATION
    /// ORDER — one batched on-device `mean_all()` vs. an f64 host sum over
    /// per-row f32 reads, not in the underlying math. f32 carries ~7
    /// significant decimal digits (mantissa epsilon ≈1.2e-7); a handful of
    /// unfused ops (matmul, exp, log, division) each contribute a few ULPs of
    /// rounding, so `1e-5` relative is roughly two orders of magnitude of
    /// headroom above the expected accumulated f32 error on these small
    /// (2–3 row) fixtures — tight enough that a genuine per-example defect
    /// (which perturbs the VALUE, not merely its last few bits) still trips
    /// it, while legitimate summation-order noise does not flake it.
    const REL_TOL: f64 = 1e-5;

    /// A minimal real [`TrainingLoop`] over a `ProjectionHead` target, for the
    /// four objectives whose loss functions never touch `self.target` (they
    /// operate directly on the batch's already-embedded tensors): MNRL/Pairs,
    /// MNRL/Triplet, Triplet-margin, Contrastive/CosineMse.
    async fn minimal_loop(device: &Device, config: FineTuneConfig) -> TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_projection_head(HIDDEN, &config, &varmap, &vb).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("decomp-oracle-job".into())
            .worker_id("decomp-oracle-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap()
    }

    /// A minimal real [`TrainingLoop`] over a classification-shaped
    /// `ProjectionHead` (projection + classifier layers) — Classification's
    /// loss DOES route through `self.target` (`Self::classify`).
    async fn minimal_classification_loop(device: &Device, config: FineTuneConfig) -> TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_classification_head(HIDDEN, NUM_CLASSES, &config, &varmap, &vb).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("decomp-oracle-cls-job".into())
            .worker_id("decomp-oracle-cls-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap()
    }

    /// Read a scalar device loss back to host as `f64`, exactly how
    /// `Trainer::evaluate` and the training step read `compute_loss`'s output.
    fn loss_scalar(loss: &Tensor) -> f64 {
        loss.to_dtype(DType::F32)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    }

    /// The oracle itself: `mean(compute_loss_per_example(batch))` must match
    /// `compute_loss(batch)` within [`REL_TOL`]. `label` is folded into the
    /// panic message so a failing objective is identifiable at a glance.
    fn assert_decomposition_matches(loop_: &TrainingLoop, batch: &TrainingBatch, label: &str) {
        let batch_loss = loss_scalar(&loop_.compute_loss(batch).unwrap());
        let per_example = loop_.compute_loss_per_example(batch).unwrap();
        assert!(
            !per_example.is_empty(),
            "{label}: empty per-example decomposition"
        );
        let mean: f64 = per_example.iter().sum::<f64>() / per_example.len() as f64;
        let tol = REL_TOL * batch_loss.abs().max(1.0);
        assert!(
            (mean - batch_loss).abs() <= tol,
            "{label}: mean(per_example) = {mean} must match compute_loss = {batch_loss} \
             within relative tolerance {REL_TOL} (abs tol {tol}); per_example = {per_example:?}"
        );
    }

    /// Non-degenerate 3-row `(anchor, positive)` fixture: varied cosine
    /// similarities (not uniformly saturated to 0 or 1), so the oracle
    /// exercises the general (non-floor) case.
    fn mnrl_pairs_batch(device: &Device) -> TrainingBatch {
        TrainingBatch::Pairs {
            anchors: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [0.6, 0.8]], device).unwrap(),
            positives: Tensor::new(&[[0.8f32, 0.6], [0.6, 0.8], [1.0, 0.0]], device).unwrap(),
        }
    }

    /// Objective 1: MNRL over a `Pairs` batch — "the only objective this
    /// shape trains" per [`TrainingLoop::compute_loss_per_example`]'s doc.
    #[tokio::test(flavor = "multi_thread")]
    async fn mnrl_pairs_decomposition_matches_compute_loss() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
            ..Default::default()
        };
        let loop_ = minimal_loop(&device, config).await;
        assert_decomposition_matches(&loop_, &mnrl_pairs_batch(&device), "MNRL/Pairs");
    }

    /// Non-degenerate 3-row `(anchor, positive, negative)` fixture shared by
    /// the MNRL-triplet-consumed and Triplet-margin objectives.
    fn triplet_batch(device: &Device) -> TrainingBatch {
        TrainingBatch::Triplet {
            anchor: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [0.6, 0.8]], device).unwrap(),
            positive: Tensor::new(&[[0.8f32, 0.6], [0.6, 0.8], [1.0, 0.0]], device).unwrap(),
            negative: Tensor::new(&[[0.0f32, 1.0], [1.0, 0.0], [0.8, 0.6]], device).unwrap(),
        }
    }

    /// Objective 2: MNRL over a `Triplet` batch — consumed as MNRL's row NLL
    /// with the explicit negative appended as an extra similarity column, when
    /// `MultipleNegativesRanking` is configured (`compute_loss`'s `Triplet`
    /// arm's `Some(MultipleNegativesRanking { .. })` branch).
    #[tokio::test(flavor = "multi_thread")]
    async fn mnrl_triplet_consumed_decomposition_matches_compute_loss() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
            ..Default::default()
        };
        let loop_ = minimal_loop(&device, config).await;
        assert_decomposition_matches(&loop_, &triplet_batch(&device), "MNRL/Triplet-consumed");
    }

    /// Objective 3: Triplet margin (`max(0, cos(a,n) - cos(a,p) + margin)`) —
    /// `compute_loss`'s `Triplet` arm's row-independent `_` (non-MNRL) branch.
    #[tokio::test(flavor = "multi_thread")]
    async fn triplet_margin_decomposition_matches_compute_loss() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            embedding_loss: Some(EmbeddingLoss::Triplet { margin: 0.3 }),
            ..Default::default()
        };
        let loop_ = minimal_loop(&device, config).await;
        assert_decomposition_matches(&loop_, &triplet_batch(&device), "Triplet margin");
    }

    /// Objective 4: Classification cross-entropy. `lora_dropout: 0.0` is
    /// load-bearing here, not incidental: `compute_loss` and
    /// `compute_loss_per_example` each independently call
    /// `Self::classify(embeddings)`, which draws a FRESH seeded dropout mask
    /// per call when dropout is on — with dropout on, the two calls would
    /// score two DIFFERENT forward passes and the oracle would be comparing
    /// unlike quantities, not testing the decomposition. Zero dropout keeps
    /// both calls' forward pass identical.
    #[tokio::test(flavor = "multi_thread")]
    async fn classification_decomposition_matches_compute_loss() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            lora_dropout: 0.0,
            ..Default::default()
        };
        let loop_ = minimal_classification_loop(&device, config).await;
        let batch = TrainingBatch::Classification {
            embeddings: Tensor::new(&[[1.0f32, 0.2], [0.1, -0.4], [-0.3, 0.9]], &device).unwrap(),
            labels: Tensor::new(&[0u32, 1u32, 2u32], &device).unwrap(),
        };
        assert_decomposition_matches(&loop_, &batch, "Classification");
    }

    /// Objective 5: Contrastive/CosineMse — `scale·cos(a,b)` regressed onto a
    /// graded target score, a row-independent squared residual.
    #[tokio::test(flavor = "multi_thread")]
    async fn cosine_mse_decomposition_matches_compute_loss() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            embedding_loss: Some(EmbeddingLoss::CosineMse),
            ..Default::default()
        };
        let loop_ = minimal_loop(&device, config).await;
        let batch = TrainingBatch::Contrastive {
            embeddings_a: Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [0.6, 0.8]], &device).unwrap(),
            embeddings_b: Tensor::new(&[[0.8f32, 0.6], [0.6, 0.8], [1.0, 0.0]], &device).unwrap(),
            scores: Tensor::new(&[0.5f32, 0.9, -0.2], &device).unwrap(),
        };
        assert_decomposition_matches(&loop_, &batch, "Contrastive/CosineMse");
    }

    /// Hidden width for the matryoshka oracle only: two non-degenerate prefix
    /// dims (`[4, 2]`) need an embedding wider than [`HIDDEN`] (2) to narrow
    /// into.
    const MATRYOSHKA_HIDDEN: usize = 4;

    /// A minimal real [`TrainingLoop`] over a `ProjectionHead` target sized
    /// for the matryoshka oracle ([`MATRYOSHKA_HIDDEN`]-wide embeddings)
    /// rather than [`minimal_loop`]'s fixed [`HIDDEN`].
    async fn minimal_matryoshka_loop(device: &Device, config: FineTuneConfig) -> TrainingLoop {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head = build_projection_head(MATRYOSHKA_HIDDEN, &config, &varmap, &vb).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.keep();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(&dir_path).await.unwrap());
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id("decomp-oracle-matryoshka-job".into())
            .worker_id("decomp-oracle-matryoshka-worker".into())
            .catalog(catalog)
            .artifact_dir(dir_path)
            .build()
            .unwrap()
    }

    /// Objective 6 (re-audit round-2 advisory): MNRL over a `Pairs` batch with
    /// `matryoshka_dims = [4, 2]` — TWO prefix dims, so the wrap genuinely
    /// sums more than one term on both the `compute_loss` (tensor,
    /// [`TrainingLoop::matryoshka_wrap`]) and `compute_loss_per_example`
    /// (host `Vec<f64>`, [`TrainingLoop::matryoshka_wrap_per_example`]) sides.
    ///
    /// The five oracles above all run with `matryoshka_dims` empty (the
    /// wrapper's `dims.is_empty()` short-circuit, which calls the objective
    /// exactly once on the full embedding and neither narrows nor sums) —
    /// none of them exercises the actual narrow-then-sum loop in either
    /// wrapper. This oracle is the one place `mean(compute_loss_per_example)
    /// ≈ compute_loss` is checked WITH that loop live on both sides, so a
    /// per-dim narrow or an accumulation-order defect in
    /// `matryoshka_sum_per_example` that happened to still match
    /// `matryoshka_sum`'s SUM (e.g. narrowing the wrong dim count, or folding
    /// dims in the wrong order under two dims where order cannot yet be
    /// distinguished) has a real chance of being caught here — where the
    /// prior five oracles could not have caught it at all, since they never
    /// entered the loop body.
    #[tokio::test(flavor = "multi_thread")]
    async fn matryoshka_mnrl_pairs_decomposition_matches_compute_loss() {
        let device = Device::Cpu;
        let config = FineTuneConfig {
            embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
            matryoshka_dims: vec![4, 2],
            ..Default::default()
        };
        let loop_ = minimal_matryoshka_loop(&device, config).await;
        let batch = TrainingBatch::Pairs {
            anchors: Tensor::new(
                &[
                    [1.0f32, 0.0, 0.2, 0.1],
                    [0.0, 1.0, -0.3, 0.4],
                    [0.6, 0.8, 0.1, -0.2],
                ],
                &device,
            )
            .unwrap(),
            positives: Tensor::new(
                &[
                    [0.8f32, 0.6, 0.05, 0.15],
                    [0.6, 0.8, -0.1, 0.05],
                    [1.0, 0.0, 0.2, -0.1],
                ],
                &device,
            )
            .unwrap(),
        };
        assert_decomposition_matches(&loop_, &batch, "Matryoshka MNRL/Pairs (dims=[4, 2])");
    }
}
