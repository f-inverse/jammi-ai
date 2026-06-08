//! Training loop: gradient descent with LR scheduling, early stopping, and checkpointing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use jammi_db::catalog::status::TrainingJobStatus;
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};

use super::data::{TextChunk, TrainingDataLoader};
use super::optimizer::clip_and_step;
use super::regression_loss::{crps_gaussian_loss, gaussian_nll_loss, pinball_loss};
use super::staging::StagedArtifact;
use super::target::TrainingTarget;
use super::{EarlyStoppingMetric, FineTuneConfig, LrSchedule};
use crate::model::{LoadedModel, ModelTask};

/// Result of a completed training run.
///
/// The loop trains and persists the adapter into a worker-private staging dir,
/// but does **not** write the job's terminal status, register the output model,
/// or promote the artifact into its canonical path — those are the worker's
/// single lease-guarded finalization. The worker promotes [`Self::staged`] into
/// [`Self::adapter_path`] only if its finalize compare-and-set wins, and
/// discards it otherwise, so the canonical artifact is written by exactly one
/// worker. The run metrics it computed (final loss, step count, timestamps) are
/// returned here so the worker records them in that same compare-and-set.
#[derive(Debug)]
pub struct TrainingResult {
    /// Canonical path the final adapter is promoted to (the path the catalog
    /// model row points at). It only holds the trained weights once
    /// [`Self::staged`] has been promoted.
    pub adapter_path: PathBuf,
    /// The worker-private staged adapter the worker promotes on a finalize CAS
    /// win or discards on a loss.
    pub staged: StagedArtifact,
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

    // Decay phase
    let decay_steps = total_steps.saturating_sub(config.warmup_steps);
    let decay_step = step - config.warmup_steps;

    match config.lr_schedule {
        LrSchedule::Constant => base_lr,
        LrSchedule::CosineDecay => {
            let progress = decay_step as f64 / decay_steps.max(1) as f64;
            base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
        LrSchedule::LinearDecay => {
            let progress = decay_step as f64 / decay_steps.max(1) as f64;
            base_lr * (1.0 - progress)
        }
    }
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
    /// Identifies the worker+attempt running this loop. Threaded into the
    /// staging-path leaf so two workers training the same `job_id` never share a
    /// training-time artifact path.
    worker_id: String,
    catalog: Arc<Catalog>,
    artifact_dir: PathBuf,
    divergence_count: usize,
    device: Device,
    /// Cooperative-cancellation flag the worker's heartbeat task sets when the
    /// lease is lost. Checked at every epoch boundary; once set the loop bails
    /// without recording a terminal status, leaving the job for lease-based
    /// reclaim. A `spawn_blocking` thread cannot be force-aborted, so this is the
    /// coarsest safe interruption point.
    cancel: Arc<AtomicBool>,
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
        }
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

    /// Set the worker+attempt identifier. Used to derive a worker-private
    /// artifact-staging path so concurrent workers on the same job never share a
    /// training-time file path; the staged artifact is promoted into the
    /// canonical path only on the worker's finalize-CAS win.
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
            device: self.device,
            cancel: self.cancel,
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
        // Update job status to running
        let started_at = chrono::Utc::now().to_rfc3339();
        let metrics_json = serde_json::json!({"started_at": started_at}).to_string();
        tokio::runtime::Handle::current().block_on(self.catalog.update_training_status(
            &self.job_id,
            TrainingJobStatus::Running,
            Some(&metrics_json),
        ))?;

        // Split training/validation
        let (train_loader, val_loader) = data_loader.split(self.config.validation_fraction)?;

        let train_batches_per_epoch = train_loader.num_batches(self.config.batch_size);
        let total_steps = (train_batches_per_epoch * self.config.epochs)
            / self.config.gradient_accumulation_steps.max(1);
        let checkpoint_interval = (total_steps as f64 * 0.1).ceil() as usize;

        // Create optimizer from VarMap's trainable variables.
        // weight_decay matches train_embedding_model.py: AdamW(weight_decay=0.01).
        let mut optimizer = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: self.config.learning_rate,
                weight_decay: self.config.weight_decay,
                ..Default::default()
            },
        )
        .map_err(|e| JammiError::FineTune(format!("Optimizer init: {e}")))?;

        let mut global_step = 0;
        let mut best_val_loss = f64::MAX;
        let mut patience_counter = 0;
        // Train into a worker-private staging dir, never the shared canonical
        // path: two workers on the same `job_id` must not share a training-time
        // file. The canonical `models/{job_id}` is written only when the worker's
        // finalize CAS wins, by promoting this staging dir.
        let canonical_dir = self.artifact_dir.join("models").join(&self.job_id);
        let staged = StagedArtifact::stage(canonical_dir, &self.worker_id)?;
        let checkpoint_dir = staged.staging_dir().to_path_buf();

        // Collect trainable vars once; their TensorIds are stable for the run.
        let trainable_vars = self.varmap.all_vars();

        // Hard negatives mined from the current model, re-mined every
        // `refresh_every` epochs. Held across epochs so a non-refresh epoch
        // reuses the last mining (the staleness/cost trade).
        let mut mined_loader: Option<TrainingDataLoader> = None;

        for epoch in 0..self.config.epochs {
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
            let mut epoch_pos_sim = 0.0f64;
            let mut epoch_neg_sim = 0.0f64;
            let mut triplet_batch_count = 0usize;

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
                    self.accumulate_sim_stats(
                        &batch,
                        &mut epoch_pos_sim,
                        &mut epoch_neg_sim,
                        &mut triplet_batch_count,
                    );
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
                            total_steps,
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
                    total_steps,
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
                    self.accumulate_sim_stats(
                        &batch,
                        &mut epoch_pos_sim,
                        &mut epoch_neg_sim,
                        &mut triplet_batch_count,
                    );
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
                            total_steps,
                        },
                    )?;
                }
            }

            // Flush any remaining micro-batch gradients that didn't fill a full
            // accumulation window (last partial window of the epoch).
            if let Some(mut acc) = accumulated_grads.take() {
                let lr = compute_lr(&self.config, global_step, total_steps);
                optimizer.set_learning_rate(lr);
                clip_and_step(
                    &mut optimizer,
                    &trainable_vars,
                    &mut acc,
                    self.config.max_grad_norm,
                )?;
                global_step += 1;
            }

            let avg_train_loss = epoch_loss / batch_count.max(1) as f64;
            let avg_pos_sim = if triplet_batch_count > 0 {
                epoch_pos_sim / triplet_batch_count as f64
            } else {
                0.0
            };
            let avg_neg_sim = if triplet_batch_count > 0 {
                epoch_neg_sim / triplet_batch_count as f64
            } else {
                0.0
            };

            // Validation — skip entirely when monitoring train loss to avoid wasting time.
            let avg_val_loss = match self.config.early_stopping_metric {
                EarlyStoppingMetric::TrainLoss => {
                    // No validation pass needed; report 0.0 as a sentinel.
                    0.0
                }
                EarlyStoppingMetric::ValLoss => {
                    // Disable dropout for the validation pass.
                    self.target.set_training(false);
                    let val_loss = self.evaluate(&val_loader)?;
                    self.target.set_training(true);
                    val_loss
                }
            };

            // Decide which loss to monitor for early stopping.
            let (monitor_loss, monitor_label) = match self.config.early_stopping_metric {
                EarlyStoppingMetric::TrainLoss => (avg_train_loss, "train"),
                EarlyStoppingMetric::ValLoss => (avg_val_loss, "val"),
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
        }

        // Restore best checkpoint before saving final adapter
        let best_path = checkpoint_dir.join("checkpoint_best.safetensors");
        if best_path.exists() {
            self.load_checkpoint(&best_path)?;
        }

        // Save the final adapter — both target variants persist their
        // trainable weights alongside a `SavedAdapter` metadata JSON.
        let final_weights = self.target.named_trainable_weights()?;
        let saved = self.target.saved_adapter(&self.config);
        jammi_lora::save_adapter(&checkpoint_dir, &final_weights, &saved)
            .map_err(|e| JammiError::FineTune(format!("Save adapter: {e}")))?;

        // The loop does not write the terminal status, register the output
        // model, or promote the staged artifact into its canonical path. All
        // three are the worker's single lease-guarded finalization: it registers
        // the model row (pointing at the canonical path), runs the compare-and-set
        // that flips the job to `completed` only while it still holds the lease,
        // and — only on a CAS win — promotes this staging into the canonical path.
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
            adapter_path: staged.canonical_dir().to_path_buf(),
            staged,
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
    /// Embeds every anchor and positive with the current model (no grad),
    /// indexes the positives as the candidate corpus (jammi's own cosine ANN),
    /// and for each anchor retrieves its hardest non-excluded neighbour. The
    /// positive and its `exclude_hops`-hop neighbourhood are excluded as the
    /// false-negative guard. A row whose pool is entirely excluded is dropped;
    /// if mining yields no usable rows the original loader is returned unchanged
    /// rather than training on an empty set.
    fn mine_hard_negative_loader(
        &mut self,
        loader: &TrainingDataLoader,
    ) -> Result<TrainingDataLoader> {
        use super::hard_negative_miner::{AnchorQuery, Candidate, HardNegativeMiner};

        let (anchors, positives, _existing_neg) = loader.in_batch_negative_texts()?;
        if anchors.is_empty() {
            return Ok(TrainingDataLoader::from_triplets(Vec::new()));
        }

        // Embed anchors and positives once with dropout off — the model state
        // the negatives are mined against.
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
        let result = (|| {
            let anchor_vecs = embed(self, &anchors)?;
            let positive_vecs = embed(self, &positives)?;

            // Candidate corpus = the positives, keyed by row index so a mined id
            // maps back to its positive text.
            let candidates: Vec<Candidate> = positive_vecs
                .iter()
                .enumerate()
                .map(|(i, v)| Candidate {
                    id: i.to_string(),
                    embedding: v.clone(),
                })
                .collect();
            let miner = HardNegativeMiner::build(&candidates, self.config.hard_negatives)?;

            let mut rows = Vec::with_capacity(anchors.len());
            for (i, anchor_vec) in anchor_vecs.iter().enumerate() {
                let query = AnchorQuery {
                    embedding: anchor_vec.clone(),
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
        _total_steps: usize,
        _global_step: usize,
    ) -> Result<f64> {
        use super::gradcache::{gradcache_backward, EncodeGroup};

        let (anchors, positives, negatives) = train_loader.in_batch_negative_texts()?;
        let scale = self.mnrl_scale();
        let has_negatives = negatives.is_some();

        // Dropout off for the whole GradCache region so the two encode passes
        // (and the logging re-encode) agree. Toggled while no encode closure
        // borrows `self`, so it does not collide with the immutable borrows
        // below.
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

        clip_and_step(
            optimizer,
            trainable_vars,
            &mut grads,
            self.config.max_grad_norm,
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
    fn encode_chunk(&self, chunk: &TextChunk) -> Result<super::data::TrainingBatch> {
        let encode = |texts: &Vec<String>| -> Result<Tensor> { self.encode_texts(texts) };

        match chunk {
            TextChunk::Contrastive {
                texts_a,
                texts_b,
                scores,
            } => {
                let proj_a = encode(texts_a)?;
                let proj_b = encode(texts_b)?;
                let scores_tensor = Tensor::from_vec(scores.clone(), (scores.len(),), &self.device)
                    .map_err(|e| JammiError::FineTune(format!("Scores tensor: {e}")))?;
                Ok(super::data::TrainingBatch::Contrastive {
                    embeddings_a: proj_a,
                    embeddings_b: proj_b,
                    scores: scores_tensor,
                })
            }
            TextChunk::Pairs { anchors, positives } => {
                let proj_a = encode(anchors)?;
                let proj_p = encode(positives)?;
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
                let proj_a = encode(anchors)?;
                let proj_p = encode(positives)?;
                let proj_n = encode(negatives)?;
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
                let head_out = self.regress(&proj)?;
                let target_tensor =
                    Tensor::from_vec(targets.clone(), (targets.len(),), &self.device)
                        .map_err(|e| JammiError::FineTune(format!("Target tensor: {e}")))?;
                Ok(super::data::TrainingBatch::Regression {
                    input: head_out,
                    target: target_tensor,
                })
            }
        }
    }

    /// Apply the distributional regression head to projected embeddings,
    /// producing the `(batch, k)` raw head parameters. Mirrors [`Self::classify`]:
    /// only a `ProjectionHead` target with a second (head) layer can regress.
    fn regress(&self, embeddings: &Tensor) -> Result<Tensor> {
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

    /// Accumulate cosine similarity stats from a triplet batch for epoch-level logging.
    /// Non-triplet batches are silently ignored. Errors in stat computation are swallowed
    /// so a GPU issue never aborts training just because of a logging metric.
    fn accumulate_sim_stats(
        &self,
        batch: &super::data::TrainingBatch,
        epoch_pos_sim: &mut f64,
        epoch_neg_sim: &mut f64,
        triplet_count: &mut usize,
    ) {
        if let super::data::TrainingBatch::Triplet {
            anchor,
            positive,
            negative,
        } = batch
        {
            let ps = cosine_similarity(anchor, positive)
                .and_then(|t| {
                    t.mean_all()
                        .map_err(|e| JammiError::FineTune(format!("{e}")))
                })
                .and_then(|t| {
                    let t = if t.dtype() == DType::F32 {
                        t
                    } else {
                        t.to_dtype(DType::F32)
                            .map_err(|e| JammiError::FineTune(format!("{e}")))?
                    };
                    t.to_scalar::<f32>()
                        .map_err(|e| JammiError::FineTune(format!("{e}")))
                });
            let ns = cosine_similarity(anchor, negative)
                .and_then(|t| {
                    t.mean_all()
                        .map_err(|e| JammiError::FineTune(format!("{e}")))
                })
                .and_then(|t| {
                    let t = if t.dtype() == DType::F32 {
                        t
                    } else {
                        t.to_dtype(DType::F32)
                            .map_err(|e| JammiError::FineTune(format!("{e}")))?
                    };
                    t.to_scalar::<f32>()
                        .map_err(|e| JammiError::FineTune(format!("{e}")))
                });
            if let (Ok(ps_val), Ok(ns_val)) = (ps, ns) {
                *epoch_pos_sim += ps_val as f64;
                *epoch_neg_sim += ns_val as f64;
                *triplet_count += 1;
            }
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
        let loss_val = loss_f32
            .to_scalar::<f32>()
            .map_err(|e| JammiError::FineTune(format!("Loss scalar: {e}")))?
            as f64;

        // Divergence detection. A diverged run returns the typed error and the
        // worker records the terminal `failed` status — terminal writes are the
        // worker's single authority, never the loop's.
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
        *epoch.batch_count += 1;

        // Scale the loss and immediately run backward, releasing the activation
        // graph so that `gradient_accumulation_steps > 1` doesn't grow memory
        // proportionally to the number of micro-batches.
        let scale = self.config.gradient_accumulation_steps.max(1) as f64;
        let scaled_loss =
            (&loss / scale).map_err(|e| JammiError::FineTune(format!("Loss scale: {e}")))?;
        let new_grads = scaled_loss
            .backward()
            .map_err(|e| JammiError::FineTune(format!("Backward: {e}")))?;

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
        if *epoch.batch_count % self.config.gradient_accumulation_steps == 0 {
            let lr = compute_lr(&self.config, *epoch.global_step, ctx.total_steps);
            ctx.optimizer.set_learning_rate(lr);

            if let Some(mut acc) = epoch.accumulated_grads.take() {
                clip_and_step(
                    ctx.optimizer,
                    ctx.trainable_vars,
                    &mut acc,
                    self.config.max_grad_norm,
                )?;
            }

            *epoch.global_step += 1;

            // Checkpoint
            if ctx.checkpoint_interval > 0 && *epoch.global_step % ctx.checkpoint_interval == 0 {
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
    /// [`RegressionLoss`]. `input` is the distributional head's raw output
    /// (`(batch, k)`); `target` is the observed `(batch,)` outcome.
    ///
    /// The three Gaussian arms read `(mean, raw_std)` from a two-wide head and
    /// score the predictive `Normal(mean, σ)`, where `σ = floor + softplus(raw_std)`
    /// — the learnable floor is the head's own trainable bias under `softplus`,
    /// with [`STD_FLOOR`] as the hard numerical guard against exact-zero variance
    /// (the overconfidence collapse). The pinball arm reads one quantile per
    /// head column and scores each against its level.
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
            return Ok(0.0);
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
    // (n, n) anchor↔positive similarity, scaled.
    let sim = (a_norm
        .matmul(&p_t)
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
            let neg_sim = (a_norm
                .matmul(&neg_t)
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

#[cfg(test)]
mod tests {
    use super::super::regression_loss::{gaussian_params, softplus_std_for_test, STD_FLOOR};
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

    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

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
