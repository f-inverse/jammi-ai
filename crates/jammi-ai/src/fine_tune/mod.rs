//! Fine-tuning: LoRA adapter training on user data.
//!
//! This module provides LoRA-based fine-tuning for embedding and classification
//! models. Training data is read through DataFusion, so any registered source
//! (Parquet, CSV, Postgres) works as long as it has the right schema.

// The candle-backed training engine (data loading, the LoRA model, the trainer
// loop, the job handle). Gated behind the default-on `local` feature; the
// config vocabulary below (`FineTuneConfig`, `FineTuneMethod`, the loss / schedule
// enums) stays transport-neutral so the `wire` surface and `RemoteSession` can
// encode a fine-tune request without the engine.
#[cfg(feature = "local")]
pub mod classifier;
#[cfg(feature = "local")]
pub mod data;
#[cfg(feature = "local")]
pub mod gradcache;
#[cfg(feature = "local")]
pub mod graph_sampler;
#[cfg(feature = "local")]
pub mod hard_negative_miner;
#[cfg(feature = "local")]
pub mod job;
#[cfg(feature = "local")]
pub mod lora;
#[cfg(feature = "local")]
pub mod target;
#[cfg(feature = "local")]
pub mod trainer;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// The LoRA init/backbone-dtype knobs are part of `FineTuneConfig`'s public
// shape, so re-export them here: a consumer constructing a config through the
// SDK boundary reaches every field's type from this module, without depending
// on `jammi-lora` directly.
pub use jammi_lora::{BackboneDtype, LoraInitMode};

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
    type Err = jammi_db::error::JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "lora" => Ok(Self::Lora),
            other => Err(jammi_db::error::JammiError::FineTune(format!(
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
    /// Multiple-Negatives-Ranking (InfoNCE / NT-Xent): for a batch of
    /// `(anchor, positive)` rows, every other row's positive is an in-batch
    /// negative. The scaled cosine-similarity matrix `S = normalize(A) ·
    /// normalize(P)ᵀ · temperature` is scored against its diagonal with a
    /// symmetric (row + column) cross-entropy. A `Triplet` batch supplies
    /// explicit hard negatives that are appended as extra similarity columns.
    /// `temperature` is the similarity scale; `20.0` is the standard default.
    MultipleNegativesRanking { temperature: f64 },
    /// AnglE: optimises an angle difference in complex space, escaping the
    /// vanishing-gradient saturation zones of cosine objectives near ±1.
    /// Splits each embedding into real/imaginary halves and applies the same
    /// pairwise log-sum-exp ordering as CoSENT over the angle magnitude.
    /// CoSENT's successor for STS quality.
    AnglE,
    /// cosine-MSE: regress scaled cosine similarity onto a graded target score
    /// with mean-squared error. The simplest objective for continuous
    /// similarity labels; prefer it over CoSENT/MNRL when labels are graded
    /// scores rather than pairs or rankings.
    CosineMse,
}

impl Default for EmbeddingLoss {
    fn default() -> Self {
        Self::CoSent
    }
}

/// Proper-scoring objective for a distributional regression head (S18).
///
/// Three of the four arms train the **parametric Gaussian** head — the head
/// emits `(mean, raw_std)` per row and the loss reads a positive `σ` from
/// `raw_std` via `floor + softplus(raw_std)` (a *learnable* floor, the
/// [`RegressionHead::Gaussian`] `std_floor`). The fourth trains the **quantile**
/// head (one output per level) with the pinball loss.
///
/// Every arm is a **proper score**: minimising it rewards a calibrated
/// *distribution*, not merely an accurate mean. (Plain MSE on the mean is *not*
/// proper for a distribution and is offered only as a secondary diagnostic, not
/// a training objective.) The default is [`Self::BetaNll`] — Seitzer's
/// variance-weighted NLL, which avoids the variance-collapse / mean-starvation
/// pathology of the naive joint `μ,σ²` NLL ([Seitzer et al. 2022]; [Nix &
/// Weigend 1994]); [`Self::Crps`] (closed-form Gaussian CRPS) is the other
/// collapse-resistant choice.
///
/// A parametric Gaussian head models **aleatoric** (irreducible data) noise
/// only. It does *not* know what it has not seen: off-distribution it can be
/// confidently wrong. Epistemic uncertainty is NP4 (amortized posterior) or S17
/// (distribution-free conformal) — pick along that spectrum; do not read this
/// head's `σ` as epistemic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionLoss {
    /// Gaussian negative log-likelihood, `½(log σ² + (y−μ)²/σ²)` (+const). The
    /// classic heteroscedastic mean-variance objective ([Nix & Weigend 1994]).
    /// Strictly proper, but the joint `μ,σ²` gradient down-weights high-error
    /// points by inflating their variance, starving the mean's gradient
    /// (variance collapse / overconfidence). Provided for completeness and as
    /// the pathology baseline; prefer `BetaNll` or `Crps`.
    GaussianNll,
    /// β-NLL ([Seitzer et al. 2022]): the per-row Gaussian NLL weighted by a
    /// stop-gradient `σ^{2β}`, which restores the mean's gradient on
    /// high-variance rows and removes the collapse. `beta ∈ [0, 1]`; `0`
    /// recovers plain NLL, `1` recovers (up to a constant) the MSE-on-the-mean
    /// gradient. The default `0.5` is Seitzer's recommended setting. This is the
    /// default regression objective.
    BetaNll {
        /// Variance-weighting exponent. `0.5` is the recommended default.
        beta: f64,
    },
    /// Closed-form Gaussian continuous ranked probability score (CRPS), from
    /// [`jammi_numerics::calibration::crps_gaussian`] — the same primitive R2
    /// headlines as a metric. Strictly proper and, unlike NLL, bounded in the
    /// outcome's units and far more stable under joint `μ,σ²` training. The
    /// recommended collapse-resistant alternative to `BetaNll`.
    Crps,
    /// Pinball / quantile loss ([Koenker & Bassett 1978]) for the quantile head.
    /// Each predicted quantile is trained to its level by the asymmetric
    /// absolute deviation `max(q·(y−ŷ), (q−1)·(y−ŷ))`, summed over levels. A
    /// non-crossing penalty discourages quantile crossing during training; the
    /// serving adapter additionally sorts post-hoc.
    Pinball,
}

impl Default for RegressionLoss {
    fn default() -> Self {
        // β-NLL is the collapse-resistant default; β=0.5 is Seitzer's setting.
        Self::BetaNll { beta: 0.5 }
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

/// Hard-negative mining via jammi's own ANN index.
///
/// When `mine` is set, the trainer periodically embeds the training corpus,
/// builds a cosine index over it, and retrieves the top-`k` nearest neighbours
/// of each anchor as hard negatives — near-misses the current model ranks too
/// highly. The anchor's own positive and the positive's `k`-hop neighbourhood
/// are excluded from the candidate pool, because a true-but-unlabelled positive
/// retrieved as a "negative" would supply a false-negative gradient.
///
/// Mined negatives go stale as the model moves, so re-mining every step is
/// wasteful; `refresh_every` re-mines once per that many epochs (ANCE's
/// asynchronous-index-refresh trade: fresher negatives cost more index builds).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HardNegativeConfig {
    /// Mine hard negatives from the model's own retrieval index. Default `false`.
    pub mine: bool,
    /// Number of hard negatives to retrieve per anchor. Default `1`.
    pub k: usize,
    /// Hops of the positive's neighbourhood to exclude from the negative pool,
    /// guarding against false negatives on near-duplicate corpora. Default `1`.
    pub exclude_hops: usize,
    /// Re-mine once every this many epochs. `1` re-mines every epoch. Default `1`.
    pub refresh_every: usize,
}

impl Default for HardNegativeConfig {
    fn default() -> Self {
        Self {
            mine: false,
            k: 1,
            exclude_hops: 1,
            refresh_every: 1,
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
    /// Proper-scoring objective for a distributional regression head (S18).
    /// `None` selects the collapse-resistant default ([`RegressionLoss::default`],
    /// β-NLL with β=0.5). A `Pinball` choice trains the quantile head over
    /// [`Self::quantile_levels`]; the other arms train the parametric Gaussian
    /// head.
    #[serde(default)]
    pub regression_loss: Option<RegressionLoss>,
    /// Quantile levels for a pinball-trained regression head, ascending in
    /// `(0, 1)` (e.g. `[0.05, 0.5, 0.95]`). Ignored by the Gaussian objectives.
    /// Empty (default) is valid only for the parametric arms; the pinball arm
    /// requires at least one level.
    #[serde(default)]
    pub quantile_levels: Vec<f64>,
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

    // ── Encoder-adapters fields (LoRA injected inside the encoder) ─────────
    /// Layer name suffixes that receive LoRA adapters (PEFT `target_modules`).
    ///
    /// Empty = train a projection head on top of the frozen base model.
    /// Non-empty = inject LoRA into the encoder's internal linears at the
    /// listed sites and train those.
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
    pub init_lora_weights: jammi_lora::LoraInitMode,

    /// Dtype for the frozen backbone weights. `BF16` cuts backbone VRAM by ~half.
    /// LoRA A/B matrices are always kept in F32 for numerical stability.
    /// Default: `F32` for backward compatibility.
    #[serde(default)]
    pub backbone_dtype: jammi_lora::BackboneDtype,

    /// AdamW weight decay (L2 regularization coefficient). Default: 0.01.
    /// Matches `train_embedding_model.py` which uses `AdamW(weight_decay=0.01)`.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Maximum global L2 norm for gradient clipping. `0.0` disables clipping.
    /// Default: 1.0. Matches `train_embedding_model.py` which uses
    /// `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`.
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,

    /// GradCache: compute the in-batch-negative loss in two passes so the
    /// effective negative pool is the whole batch without holding every
    /// representation's activation graph at once. A no-grad pass embeds all
    /// rows and caches each representation's loss-gradient; a second pass
    /// re-embeds chunk by chunk with grad and backpropagates through the
    /// cached gradient, freeing each chunk's graph before the next. The
    /// optimiser sees the same gradient as a single-pass run (a tolerance test
    /// pins this), but peak memory scales with the chunk, not the batch.
    /// Distinct from `gradient_accumulation_steps`, which does *not* enlarge
    /// the in-batch negative pool. Only applies to the in-batch-negative
    /// objective (`MultipleNegativesRanking`). Default: `false`.
    #[serde(default)]
    pub cached: bool,

    /// Hard-negative mining configuration. With `mine = true` the trainer
    /// mines hard negatives from its own ANN index (see [`HardNegativeConfig`]).
    /// Default: mining off.
    #[serde(default)]
    pub hard_negatives: HardNegativeConfig,

    /// Matryoshka representation dimensions. When non-empty, the embedding
    /// objective is evaluated at each listed prefix dimension and the losses
    /// summed, so the leading coordinates of the embedding carry the most
    /// information and a consumer can truncate the served vector to any listed
    /// dimension with graceful quality decay. Importance-ordering is *created*
    /// by training with this on, so truncation at serve time is only valid for
    /// a model trained with these dims. Empty (default) trains the full
    /// dimension only. Each entry must be `> 0` and `<=` the embedding width.
    #[serde(default)]
    pub matryoshka_dims: Vec<usize>,
}

fn default_weight_decay() -> f64 {
    0.01
}
fn default_max_grad_norm() -> f64 {
    1.0
}

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
            regression_loss: None,
            quantile_levels: Vec::new(),
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
            init_lora_weights: jammi_lora::LoraInitMode::ZerosB,
            backbone_dtype: jammi_lora::BackboneDtype::F32,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            cached: false,
            hard_negatives: HardNegativeConfig::default(),
            matryoshka_dims: Vec::new(),
        }
    }
}

impl FineTuneConfig {
    /// Validate all fields. Returns an error describing the first invalid field.
    pub fn validate(&self) -> jammi_db::error::Result<()> {
        use jammi_db::error::JammiError;

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
        if self.hard_negatives.mine {
            if self.hard_negatives.k == 0 {
                return Err(JammiError::FineTune(
                    "hard_negatives.k must be > 0 when mining is enabled".into(),
                ));
            }
            if self.hard_negatives.refresh_every == 0 {
                return Err(JammiError::FineTune(
                    "hard_negatives.refresh_every must be > 0 when mining is enabled".into(),
                ));
            }
        }
        if self.matryoshka_dims.contains(&0) {
            return Err(JammiError::FineTune(
                "matryoshka_dims entries must all be > 0".into(),
            ));
        }
        if let Some(RegressionLoss::BetaNll { beta }) = self.regression_loss {
            if !(0.0..=1.0).contains(&beta) {
                return Err(JammiError::FineTune(
                    "regression_loss BetaNll beta must be in [0.0, 1.0]".into(),
                ));
            }
        }
        if matches!(self.regression_loss, Some(RegressionLoss::Pinball)) {
            if self.quantile_levels.is_empty() {
                return Err(JammiError::FineTune(
                    "Pinball regression loss requires at least one quantile level".into(),
                ));
            }
            if self
                .quantile_levels
                .iter()
                .any(|&q| !(0.0..1.0).contains(&q) || q <= 0.0)
            {
                return Err(JammiError::FineTune(
                    "quantile_levels must lie strictly in (0, 1)".into(),
                ));
            }
            if self.quantile_levels.windows(2).any(|w| w[1] <= w[0]) {
                return Err(JammiError::FineTune(
                    "quantile_levels must be strictly ascending".into(),
                ));
            }
        }
        Ok(())
    }
}
