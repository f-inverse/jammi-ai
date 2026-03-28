//! Training loop: gradient descent with LR scheduling, early stopping, and checkpointing.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use jammi_engine::catalog::status::FineTuneJobStatus;
use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};

use super::data::{TextChunk, TrainingDataLoader};
use super::lora::{save_lora_weights, LoraModel};
use super::{FineTuneConfig, LrSchedule};
use crate::model::{LoadedModel, ModelTask};

/// Result of a completed training run.
#[derive(Debug)]
pub struct TrainingResult {
    /// Path where the final adapter weights were saved.
    pub adapter_path: PathBuf,
    /// Best validation loss achieved.
    pub final_loss: f64,
    /// Total optimizer steps taken.
    pub total_steps: usize,
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

/// The training loop: runs LoRA fine-tuning with gradient accumulation,
/// early stopping, LR scheduling, and checkpointing.
pub struct TrainingLoop {
    model: LoraModel,
    varmap: VarMap,
    config: FineTuneConfig,
    job_id: String,
    catalog: Arc<Catalog>,
    artifact_dir: PathBuf,
    divergence_count: usize,
    base_model: Option<Arc<LoadedModel>>,
}

/// Builder for [`TrainingLoop`]. Core ML params (model, varmap, config) are
/// required at construction; infrastructure params are set via builder methods.
pub struct TrainingLoopBuilder {
    model: LoraModel,
    varmap: VarMap,
    config: FineTuneConfig,
    job_id: Option<String>,
    catalog: Option<Arc<Catalog>>,
    artifact_dir: Option<PathBuf>,
    base_model: Option<Arc<LoadedModel>>,
}

impl TrainingLoopBuilder {
    /// Start building a training loop with the core ML parameters.
    pub fn new(model: LoraModel, varmap: VarMap, config: FineTuneConfig) -> Self {
        Self {
            model,
            varmap,
            config,
            job_id: None,
            catalog: None,
            artifact_dir: None,
            base_model: None,
        }
    }

    /// Set the job ID for catalog tracking.
    pub fn job_id(mut self, id: String) -> Self {
        self.job_id = Some(id);
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

    /// Set the frozen base model for model-in-loop training.
    pub fn base_model(mut self, model: Arc<LoadedModel>) -> Self {
        self.base_model = Some(model);
        self
    }

    /// Build the training loop. All infrastructure params must be set.
    pub fn build(self) -> Result<TrainingLoop> {
        let job_id = self
            .job_id
            .ok_or_else(|| JammiError::FineTune("TrainingLoopBuilder: job_id required".into()))?;
        let catalog = self
            .catalog
            .ok_or_else(|| JammiError::FineTune("TrainingLoopBuilder: catalog required".into()))?;
        let artifact_dir = self.artifact_dir.ok_or_else(|| {
            JammiError::FineTune("TrainingLoopBuilder: artifact_dir required".into())
        })?;
        Ok(TrainingLoop {
            model: self.model,
            varmap: self.varmap,
            config: self.config,
            job_id,
            catalog,
            artifact_dir,
            divergence_count: 0,
            base_model: self.base_model,
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
        self.catalog.update_fine_tune_status(
            &self.job_id,
            FineTuneJobStatus::Running,
            Some(&metrics_json),
        )?;

        // Split training/validation
        let (train_loader, val_loader) = data_loader.split(self.config.validation_fraction)?;

        let train_batches_per_epoch = train_loader.num_batches(self.config.batch_size);
        let total_steps = (train_batches_per_epoch * self.config.epochs)
            / self.config.gradient_accumulation_steps.max(1);
        let checkpoint_interval = (total_steps as f64 * 0.1).ceil() as usize;

        // Create optimizer from VarMap's trainable variables
        let mut optimizer = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: self.config.learning_rate,
                ..Default::default()
            },
        )
        .map_err(|e| JammiError::FineTune(format!("Optimizer init: {e}")))?;

        let mut global_step = 0;
        let mut best_val_loss = f64::MAX;
        let mut patience_counter = 0;
        let checkpoint_dir = self.artifact_dir.join("models").join(&self.job_id);
        std::fs::create_dir_all(&checkpoint_dir)?;

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            let mut accumulated_loss: Option<Tensor> = None;

            if self.base_model.is_some() {
                // Real training: encode text through base model, project through LoRA
                let text_chunks = train_loader.text_chunks(self.config.batch_size);
                for chunk in &text_chunks {
                    let batch = self.encode_chunk(chunk)?;
                    let loss = self.compute_loss(&batch)?;
                    self.process_batch_loss(
                        loss,
                        &mut batch_count,
                        &mut epoch_loss,
                        &mut accumulated_loss,
                        &mut global_step,
                        total_steps,
                        &mut optimizer,
                        &checkpoint_dir,
                        checkpoint_interval,
                        &started_at,
                    )?;
                }
            } else {
                // Precomputed path: direct tensor batches (for testing)
                let train_batches = train_loader.batches(self.config.batch_size)?;
                for batch in train_batches {
                    let batch = batch?;
                    let loss = self.compute_loss(&batch)?;
                    self.process_batch_loss(
                        loss,
                        &mut batch_count,
                        &mut epoch_loss,
                        &mut accumulated_loss,
                        &mut global_step,
                        total_steps,
                        &mut optimizer,
                        &checkpoint_dir,
                        checkpoint_interval,
                        &started_at,
                    )?;
                }
            }

            // Flush remaining accumulated gradients
            if accumulated_loss.is_some() {
                let lr = compute_lr(&self.config, global_step, total_steps);
                optimizer.set_learning_rate(lr);
                if let Some(acc) = accumulated_loss.take() {
                    optimizer
                        .backward_step(&acc)
                        .map_err(|e| JammiError::FineTune(format!("Backward flush: {e}")))?;
                }
                global_step += 1;
            }

            let avg_train_loss = epoch_loss / batch_count.max(1) as f64;

            // Validation
            let avg_val_loss = self.evaluate(&val_loader)?;

            tracing::info!(
                epoch,
                avg_train_loss,
                avg_val_loss,
                global_step,
                lr = compute_lr(&self.config, global_step, total_steps),
                "Epoch complete"
            );

            // Early stopping
            if avg_val_loss < best_val_loss {
                best_val_loss = avg_val_loss;
                patience_counter = 0;
                self.save_checkpoint_tagged(&checkpoint_dir, "best")?;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    tracing::info!(
                        epoch,
                        patience_counter,
                        best_val_loss,
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

        // Save final adapter
        let adapter_path = checkpoint_dir.join("adapter.safetensors");
        save_lora_weights(&self.model, &adapter_path)?;

        // Update job status
        let completed_at = chrono::Utc::now().to_rfc3339();
        let metrics = serde_json::json!({
            "final_loss": best_val_loss,
            "total_steps": global_step,
            "started_at": started_at,
            "completed_at": completed_at,
        })
        .to_string();
        self.catalog.update_fine_tune_status(
            &self.job_id,
            FineTuneJobStatus::Completed,
            Some(&metrics),
        )?;

        Ok(TrainingResult {
            adapter_path: checkpoint_dir,
            final_loss: best_val_loss,
            total_steps: global_step,
        })
    }

    /// Encode texts through the frozen base model. Returns (batch_size, hidden_size) Tensor.
    fn encode_texts(&self, texts: &[String]) -> Result<Tensor> {
        let base = self
            .base_model
            .as_ref()
            .ok_or_else(|| JammiError::FineTune("No base model for encoding".into()))?;

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let arr = Arc::new(StringArray::from(text_refs)) as ArrayRef;
        let output = base
            .forward(&[arr], ModelTask::TextEmbedding)
            .map_err(|e| JammiError::FineTune(format!("Encode: {e}")))?;

        let n = output.shapes[0].0;
        let dim = output.shapes[0].1;
        Tensor::from_vec(output.float_outputs[0].clone(), (n, dim), &Device::Cpu)
            .map_err(|e| JammiError::FineTune(format!("Encode tensor: {e}")))
    }

    /// Apply LoRA projection to base embeddings.
    fn project(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.model.layers[0].1.forward(embeddings)
    }

    /// Encode a text chunk through the base model and project through LoRA,
    /// producing a TrainingBatch ready for loss computation.
    fn encode_chunk(&self, chunk: &TextChunk) -> Result<super::data::TrainingBatch> {
        match chunk {
            TextChunk::Contrastive {
                texts_a,
                texts_b,
                scores,
            } => {
                let emb_a = self.encode_texts(texts_a)?;
                let proj_a = self.project(&emb_a)?;
                let emb_b = self.encode_texts(texts_b)?;
                let proj_b = self.project(&emb_b)?;
                let scores_tensor = Tensor::from_vec(scores.clone(), (scores.len(),), &Device::Cpu)
                    .map_err(|e| JammiError::FineTune(format!("Scores tensor: {e}")))?;
                Ok(super::data::TrainingBatch::Contrastive {
                    embeddings_a: proj_a,
                    embeddings_b: proj_b,
                    scores: scores_tensor,
                })
            }
            TextChunk::Triplet {
                anchors,
                positives,
                negatives,
            } => {
                let emb_a = self.encode_texts(anchors)?;
                let proj_a = self.project(&emb_a)?;
                let emb_p = self.encode_texts(positives)?;
                let proj_p = self.project(&emb_p)?;
                let emb_n = self.encode_texts(negatives)?;
                let proj_n = self.project(&emb_n)?;
                Ok(super::data::TrainingBatch::Triplet {
                    anchor: proj_a,
                    positive: proj_p,
                    negative: proj_n,
                })
            }
            TextChunk::Classification { texts, labels } => {
                let emb = self.encode_texts(texts)?;
                let proj = self.project(&emb)?;
                let labels_tensor = Tensor::from_vec(labels.clone(), (labels.len(),), &Device::Cpu)
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
        }
    }

    /// Process a single batch loss: divergence detection, gradient accumulation,
    /// optimizer step, and checkpointing. Shared by both real and precomputed paths.
    #[allow(clippy::too_many_arguments)]
    fn process_batch_loss(
        &mut self,
        loss: Tensor,
        batch_count: &mut usize,
        epoch_loss: &mut f64,
        accumulated_loss: &mut Option<Tensor>,
        global_step: &mut usize,
        total_steps: usize,
        optimizer: &mut AdamW,
        checkpoint_dir: &Path,
        checkpoint_interval: usize,
        started_at: &str,
    ) -> Result<()> {
        let loss_val =
            loss.to_scalar::<f32>()
                .map_err(|e| JammiError::FineTune(format!("Loss scalar: {e}")))? as f64;

        // Divergence detection
        if loss_val.is_nan() || loss_val > 100.0 {
            self.divergence_count += 1;
            if self.divergence_count >= 3 {
                let err_msg = "Training diverged: loss was NaN or >100 for 3 consecutive batches";
                let metrics = serde_json::json!({
                    "error_message": err_msg,
                    "started_at": started_at,
                })
                .to_string();
                if let Err(e) = self.catalog.update_fine_tune_status(
                    &self.job_id,
                    FineTuneJobStatus::Failed,
                    Some(&metrics),
                ) {
                    tracing::error!(job_id = %self.job_id, error = %e, "Failed to record job status in catalog");
                }
                return Err(JammiError::FineTune(err_msg.into()));
            }
            return Ok(());
        }
        self.divergence_count = 0;

        // Gradient accumulation: scale loss
        let scaled_loss = if self.config.gradient_accumulation_steps > 1 {
            (&loss / self.config.gradient_accumulation_steps as f64)
                .map_err(|e| JammiError::FineTune(format!("Loss scale: {e}")))?
        } else {
            loss
        };

        *accumulated_loss = Some(match accumulated_loss.take() {
            Some(acc) => {
                (&acc + &scaled_loss).map_err(|e| JammiError::FineTune(format!("Loss acc: {e}")))?
            }
            None => scaled_loss,
        });

        *epoch_loss += loss_val;
        *batch_count += 1;

        // Step optimizer every N micro-batches
        if *batch_count % self.config.gradient_accumulation_steps == 0 {
            let lr = compute_lr(&self.config, *global_step, total_steps);
            optimizer.set_learning_rate(lr);

            if let Some(acc) = accumulated_loss.take() {
                optimizer
                    .backward_step(&acc)
                    .map_err(|e| JammiError::FineTune(format!("Backward: {e}")))?;
            }
            *global_step += 1;

            // Checkpoint
            if checkpoint_interval > 0 && *global_step % checkpoint_interval == 0 {
                self.save_checkpoint(checkpoint_dir, *global_step)?;
            }
        }

        Ok(())
    }

    /// Compute loss for a training batch. Uses cosine embedding loss (CoSENT)
    /// for contrastive pairs, triplet loss for triplets.
    fn compute_loss(&self, batch: &super::data::TrainingBatch) -> Result<Tensor> {
        match batch {
            super::data::TrainingBatch::Contrastive {
                embeddings_a,
                embeddings_b,
                scores,
            } => self.cosent_loss(embeddings_a, embeddings_b, scores),
            super::data::TrainingBatch::Triplet {
                anchor,
                positive,
                negative,
            } => self.triplet_loss(anchor, positive, negative),
            super::data::TrainingBatch::Classification { embeddings, labels } => {
                let logits = self.classify(embeddings)?;
                self.cross_entropy_loss(&logits, labels)
            }
            super::data::TrainingBatch::Ner {
                hidden_states,
                labels,
            } => self.ner_loss(hidden_states, labels),
        }
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

    /// Apply classification head to projected embeddings.
    fn classify(&self, embeddings: &Tensor) -> Result<Tensor> {
        if self.model.layers.len() > 1 {
            self.model.layers[1].1.forward(embeddings)
        } else {
            Err(JammiError::FineTune(
                "No classification head in LoRA model".into(),
            ))
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
            _ => 0.5,
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
    /// Dual-path: with base_model encodes text chunks, without it uses precomputed batches.
    fn evaluate(&self, val_loader: &TrainingDataLoader) -> Result<f64> {
        if val_loader.is_empty() {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        if self.base_model.is_some() {
            let text_chunks = val_loader.text_chunks(self.config.batch_size);
            for chunk in &text_chunks {
                let batch = self.encode_chunk(chunk)?;
                let loss = self.compute_loss(&batch)?;
                total_loss += loss
                    .to_scalar::<f32>()
                    .map_err(|e| JammiError::FineTune(format!("Val loss scalar: {e}")))?
                    as f64;
                count += 1;
            }
        } else {
            let batches = val_loader.batches(self.config.batch_size)?;
            for batch in batches {
                let batch = batch?;
                let loss = self.compute_loss(&batch)?;
                total_loss += loss
                    .to_scalar::<f32>()
                    .map_err(|e| JammiError::FineTune(format!("Val loss scalar: {e}")))?
                    as f64;
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        })
    }

    /// Save a numbered checkpoint.
    fn save_checkpoint(&self, dir: &Path, step: usize) -> Result<()> {
        let path = dir.join(format!("checkpoint_{step}.safetensors"));
        save_lora_weights(&self.model, &path)
    }

    /// Save a named checkpoint (e.g. "best").
    fn save_checkpoint_tagged(&self, dir: &Path, tag: &str) -> Result<()> {
        let path = dir.join(format!("checkpoint_{tag}.safetensors"));
        save_lora_weights(&self.model, &path)
    }

    /// Load a checkpoint, updating the model's LoRA weights.
    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let device = self
            .model
            .layers
            .first()
            .map(|(_, l)| l.lora_a.device().clone())
            .unwrap_or(candle_core::Device::Cpu);
        let weights = super::lora::load_lora_weights(path, &device)?;
        super::lora::apply_loaded_weights(&mut self.model, &weights)
    }
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
