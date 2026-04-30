//! Fine-tuning: LoRA adapter training on user data.
//!
//! This module provides LoRA-based fine-tuning for embedding and classification
//! models. Training data is read through DataFusion, so any registered source
//! (Parquet, CSV, Postgres) works as long as it has the right schema.

pub mod data;
pub mod deep_lora;
pub mod job;
pub mod lora;
pub mod trainer;

use std::collections::HashMap;

use candle_core::DType;
use serde::{Deserialize, Serialize};

/// Supported fine-tuning methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FineTuneMethod {
    /// Low-Rank Adaptation — trains small adapter matrices alongside frozen base weights.
    Lora,
}

impl std::fmt::Display for FineTuneMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lora => write!(f, "lora"),
        }
    }
}

impl std::str::FromStr for FineTuneMethod {
    type Err = jammi_engine::error::JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "lora" => Ok(Self::Lora),
            other => Err(jammi_engine::error::JammiError::FineTune(format!(
                "Unknown fine-tuning method '{other}'. Supported: lora"
            ))),
        }
    }
}

/// Loss function for embedding fine-tuning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingLoss {
    /// CoSENT: sorts pairs by score, applies cross-entropy on cosine similarity ordering.
    CoSent,
    /// Triplet loss: `max(0, cos(a,neg) - cos(a,pos) + margin)`.
    Triplet { margin: f64 },
    /// InfoNCE with in-batch negatives. `τ` is the temperature.
    MultipleNegativesRanking { temperature: f64 },
}

impl Default for EmbeddingLoss {
    fn default() -> Self {
        Self::CoSent
    }
}

/// Loss function for classification fine-tuning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationLoss {
    /// Standard cross-entropy loss.
    CrossEntropy,
}

impl Default for ClassificationLoss {
    fn default() -> Self {
        Self::CrossEntropy
    }
}

/// Which loss signal to monitor for early stopping.
///
/// `ValLoss` (default) — monitors held-out validation loss; requires
/// `validation_fraction > 0`.  Matches `train_embedding_model.py --val-file` behaviour.
///
/// `TrainLoss` — monitors average training loss each epoch; the full
/// dataset is used for training (set `validation_fraction = 0.0`).  Matches
/// `train_embedding_model.py` without `--val-file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyStoppingMetric {
    /// Monitor held-out validation loss (default).
    ValLoss,
    /// Monitor epoch-average training loss — no validation split needed.
    TrainLoss,
}

impl Default for EarlyStoppingMetric {
    fn default() -> Self {
        Self::ValLoss
    }
}

/// Learning rate schedule applied after warmup.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LrSchedule {
    /// Fixed learning rate throughout.
    Constant,
    /// Cosine annealing from base LR to 0 (default).
    CosineDecay,
    /// Linear ramp from base LR to 0.
    LinearDecay,
}

impl Default for LrSchedule {
    fn default() -> Self {
        Self::CosineDecay
    }
}

/// Dtype for the frozen backbone weights loaded during deep LoRA fine-tuning.
///
/// `BF16` halves backbone memory (~450 MB → ~225 MB for base, multi-GB for
/// large) with negligible impact on training dynamics because the frozen
/// backbone weights are never updated.  The trainable LoRA A/B matrices always
/// remain in F32 regardless of this setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackboneDtype {
    /// Full precision (default; maximally compatible). 
    F32,
    /// BFloat16 — recommended for CUDA/Metal; cuts backbone VRAM by ~half.
    BF16,
    /// Half-precision float — compatible with most CUDA devices.
    F16,
}

impl Default for BackboneDtype {
    fn default() -> Self {
        Self::F32
    }
}

impl From<BackboneDtype> for DType {
    fn from(d: BackboneDtype) -> Self {
        match d {
            BackboneDtype::F32 => DType::F32,
            BackboneDtype::BF16 => DType::BF16,
            BackboneDtype::F16 => DType::F16,
        }
    }
}

/// Configuration for a fine-tuning job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuneConfig {
    /// LoRA rank (number of low-rank dimensions). Default: 8.
    pub lora_rank: usize,
    /// LoRA scaling factor. Default: 16.0.
    pub lora_alpha: f64,
    /// LoRA dropout probability applied in the LoRA path during training. Default: 0.05.
    pub lora_dropout: f64,
    /// Base learning rate. Default: 2e-4.
    pub learning_rate: f64,
    /// Number of training epochs. Default: 3.
    pub epochs: usize,
    /// Micro-batch size. Default: 8.
    pub batch_size: usize,
    /// Maximum sequence length for tokenization. Default: 512.
    pub max_seq_length: usize,
    /// Loss function for embedding fine-tuning. Auto-selected from data format if None.
    pub embedding_loss: Option<EmbeddingLoss>,
    /// Loss function for classification fine-tuning. Auto-selected if None.
    pub classification_loss: Option<ClassificationLoss>,
    /// Gradient accumulation steps. Effective batch = batch_size × this. Default: 1.
    pub gradient_accumulation_steps: usize,
    /// Fraction of data held out for validation. Default: 0.1.
    pub validation_fraction: f64,
    /// Epochs without improvement before stopping. Default: 3.
    pub early_stopping_patience: usize,
    /// Steps of linear warmup from 0 to base LR. Default: 100.
    pub warmup_steps: usize,
    /// Decay schedule after warmup. Default: CosineDecay.
    pub lr_schedule: LrSchedule,
    /// Which loss to monitor for early stopping.
    /// Default: `ValLoss` (held-out split).
    /// Set to `TrainLoss` when `validation_fraction = 0.0` to replicate
    /// `train_embedding_model.py` without `--val-file`.
    #[serde(default)]
    pub early_stopping_metric: EarlyStoppingMetric,

    // ── PEFT-style deep LoRA fields ────────────────────────────────────────
    /// Layer name suffixes that receive LoRA adapters (PEFT `target_modules`).
    ///
    /// Empty = projection-only (current default behaviour).
    /// `["all-linear"]` = every linear layer.
    /// Model-specific examples: `["query", "value"]` for BERT/RoBERTa;
    /// `["q_lin", "v_lin"]` for DistilBERT; `["Wqkv"]` for ModernBERT.
    #[serde(default)]
    pub target_modules: Vec<String>,

    /// Only apply LoRA to these 0-based encoder layer indices.
    /// `None` (default) = all layers.
    #[serde(default)]
    pub layers_to_transform: Option<Vec<usize>>,

    /// Use rank-stabilized scaling: `alpha / sqrt(rank)` instead of `alpha / rank`.
    #[serde(default)]
    pub use_rslora: bool,

    /// Per-module rank overrides keyed by module-name substring.
    /// E.g. `{"query": 16, "value": 4}` overrides the global `lora_rank` for
    /// matching modules. An empty map uses `lora_rank` everywhere.
    #[serde(default)]
    pub rank_pattern: HashMap<String, usize>,

    /// Initialization strategy for the LoRA A (and optionally B) matrix.
    #[serde(default)]
    pub init_lora_weights: lora::LoraInitMode,

    /// Dtype for the frozen backbone weights. `BF16` cuts backbone VRAM by ~half.
    /// LoRA A/B matrices are always kept in F32 for numerical stability.
    /// Default: `F32` for backward compatibility.
    #[serde(default)]
    pub backbone_dtype: BackboneDtype,

    /// AdamW weight decay (L2 regularization coefficient). Default: 0.01.
    /// Matches `train_embedding_model.py` which uses `AdamW(weight_decay=0.01)`.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Maximum global L2 norm for gradient clipping. `0.0` disables clipping.
    /// Default: 1.0. Matches `train_embedding_model.py` which uses
    /// `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`.
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
}

fn default_weight_decay() -> f64 { 0.01 }
fn default_max_grad_norm() -> f64 { 1.0 }

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            lora_rank: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.05,
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 8,
            max_seq_length: 512,
            embedding_loss: None,
            classification_loss: None,
            gradient_accumulation_steps: 1,
            validation_fraction: 0.1,
            early_stopping_patience: 3,
            warmup_steps: 100,
            lr_schedule: LrSchedule::CosineDecay,
            early_stopping_metric: EarlyStoppingMetric::ValLoss,
            target_modules: Vec::new(),
            layers_to_transform: None,
            use_rslora: false,
            rank_pattern: HashMap::new(),
            init_lora_weights: lora::LoraInitMode::ZerosB,
            backbone_dtype: BackboneDtype::F32,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

impl FineTuneConfig {
    /// Validate all fields. Returns an error describing the first invalid field.
    pub fn validate(&self) -> jammi_engine::error::Result<()> {
        use jammi_engine::error::JammiError;

        if self.lora_rank == 0 {
            return Err(JammiError::FineTune("lora_rank must be > 0".into()));
        }
        if self.lora_alpha <= 0.0 {
            return Err(JammiError::FineTune("lora_alpha must be > 0".into()));
        }
        if !(0.0..1.0).contains(&self.lora_dropout) {
            return Err(JammiError::FineTune(
                "lora_dropout must be in [0.0, 1.0)".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(JammiError::FineTune("learning_rate must be > 0".into()));
        }
        if self.epochs == 0 {
            return Err(JammiError::FineTune("epochs must be > 0".into()));
        }
        if self.batch_size == 0 {
            return Err(JammiError::FineTune("batch_size must be > 0".into()));
        }
        if self.gradient_accumulation_steps == 0 {
            return Err(JammiError::FineTune(
                "gradient_accumulation_steps must be > 0".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.validation_fraction) {
            return Err(JammiError::FineTune(
                "validation_fraction must be in [0.0, 1.0)".into(),
            ));
        }
        if self.early_stopping_patience == 0 {
            return Err(JammiError::FineTune(
                "early_stopping_patience must be > 0".into(),
            ));
        }
        Ok(())
    }
}
