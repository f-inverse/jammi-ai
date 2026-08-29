//! Fine-tune request vocabulary: the transport-neutral config + method types.
//!
//! These are the knobs a fine-tune request carries — `FineTuneConfig`, the loss
//! and schedule enums, and the `FineTuneMethod` selector. They hold no engine
//! state, so they live on the wire substrate: the embedded engine reads them to
//! build a training run, the gRPC converters encode/decode them, and a
//! data-plane client builds a request from them without the candle stack.
//!
//! [`HeldOutLoss`] / [`ExampleLoss`] are the companion RESULT vocabulary — the
//! public per-pair held-out evaluation seam (unit 63, CONTRACT H1). They are
//! plain `serde` types, not proto-backed, mirroring how `jammi_wire::eval`'s
//! report shapes (`EmbeddingEvalReport`, `PerQueryRecord`, …) cross the public
//! API surface today: those results carry no `.proto` message of their own —
//! they travel over the wire as an opaque JSON payload the caller decodes —
//! rather than declaring a new RPC, so a plain struct is the shape a consumer
//! actually deserializes. `HeldOutLoss` follows that precedent rather than the
//! request-side `FineTuneConfig` pattern above (which IS proto-backed because
//! it fills a structured field of the `StartTraining` request message). This
//! module carries only the types; computing a `HeldOutLoss` from a trained
//! model's held-out split is `jammi_ai::Trainer::evaluate_held_out` (H1,
//! ai-core domain) — no computation lives here, and this commit adds no new
//! RPC or wire endpoint.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// The LoRA init knob and the compute-precision vocabulary are part of
// `FineTuneConfig`'s public shape, so re-export them here: a consumer
// constructing a config through the SDK boundary reaches every field's type
// from this module, without depending on `jammi-lora` / `jammi-numerics`
// directly. `ComputePrecision` lives in `jammi-numerics` (not `jammi-lora`) —
// it is the same "what dtype does this backbone run at" concept the
// inference engine's compute-precision knob uses, so both share one type.
pub use jammi_lora::LoraInitMode;
pub use jammi_numerics::ComputePrecision;

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingLoss {
    /// CoSENT: sorts pairs by score, applies cross-entropy on cosine similarity ordering.
    #[default]
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

/// Proper-scoring objective for a distributional regression head (S18).
///
/// Three of the four arms train the **parametric Gaussian** head — the head
/// emits `(mean, raw_std)` per row and the loss reads a positive `σ` from
/// `raw_std` via `floor + softplus(raw_std)` (a *learnable* floor, the
/// `RegressionHead::Gaussian` `std_floor`). The fourth trains the **quantile**
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationLoss {
    /// Standard cross-entropy loss.
    #[default]
    CrossEntropy,
}

/// Which loss signal to monitor for early stopping.
///
/// `ValLoss` (default) — monitors held-out validation loss; requires
/// `validation_fraction > 0`.  Matches `train_embedding_model.py --val-file` behaviour.
///
/// `TrainLoss` — monitors average training loss each epoch; the full
/// dataset is used for training (set `validation_fraction = 0.0`).  Matches
/// `train_embedding_model.py` without `--val-file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EarlyStoppingMetric {
    /// Monitor held-out validation loss (default).
    #[default]
    ValLoss,
    /// Monitor epoch-average training loss — no validation split needed.
    TrainLoss,
}

/// Learning rate schedule applied after warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LrSchedule {
    /// Fixed learning rate throughout.
    Constant,
    /// Cosine annealing from base LR to 0 (default).
    #[default]
    CosineDecay,
    /// Linear ramp from base LR to 0.
    LinearDecay,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// Default: `F32`.
    #[serde(default)]
    pub backbone_dtype: jammi_numerics::ComputePrecision,

    /// AdamW weight decay (L2 regularization coefficient). Default: 0.01.
    /// Matches `train_embedding_model.py` which uses `AdamW(weight_decay=0.01)`.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Maximum global L2 norm for gradient clipping. `0.0` disables clipping.
    /// Default: 1.0. Matches `train_embedding_model.py` which uses
    /// `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`. Must be
    /// finite: refused at deserialization (`finite_max_grad_norm`, the
    /// `deserialize_with` hook) and by [`Self::validate`].
    #[serde(
        default = "default_max_grad_norm",
        deserialize_with = "finite_max_grad_norm"
    )]
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

    /// Random seed for the fine-tune. The LoRA A/B initialisation and the
    /// dropout mask are a pure function of this seed and each parameter's name,
    /// so on `Device::Cpu` two runs with the same `(seed, source rows, config)`
    /// produce byte-identical adapter weights — run-to-run and across processes.
    /// Fixed per job (never drawn from entropy at construction); the default
    /// [`DEFAULT_FINE_TUNE_SEED`] is used when a caller does not specify one.
    #[serde(default = "default_fine_tune_seed")]
    pub seed: u64,
}

/// The fixed default fine-tune seed. A constant (not entropy) so a job that
/// omits `seed` is still reproducible.
pub const DEFAULT_FINE_TUNE_SEED: u64 = 42;

fn default_fine_tune_seed() -> u64 {
    DEFAULT_FINE_TUNE_SEED
}

fn default_weight_decay() -> f64 {
    0.01
}
fn default_max_grad_norm() -> f64 {
    1.0
}

/// Refuse a non-finite `max_grad_norm` at the deserialization edge, before
/// the value can reach a training loop. `NaN` compares `false` against
/// everything, so it would fall through the trainer's `max_norm <= 0.0`
/// "clipping disabled" guard and scale every gradient by a NaN coefficient
/// (the clip's own boundary refuses it too — this is the config edge, one
/// layer earlier, so a request carrying it is refused before any model is
/// loaded). `serde_json` already refuses the only JSON spelling of a
/// non-finite number (an over-range literal such as `1e999` is "number out
/// of range"); this guard makes the refusal format-independent for any
/// other `Deserializer` — and [`FineTuneConfig::validate`] covers a config
/// built in-process.
fn finite_max_grad_norm<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = f64::deserialize(deserializer)?;
    if !v.is_finite() {
        return Err(serde::de::Error::custom(format!(
            "max_grad_norm must be finite, got {v}"
        )));
    }
    Ok(v)
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
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            cached: false,
            hard_negatives: HardNegativeConfig::default(),
            matryoshka_dims: Vec::new(),
            seed: DEFAULT_FINE_TUNE_SEED,
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
        // Domain-validity at the edge: a NaN `max_grad_norm` compares `false`
        // against everything, so it would pass a `<= 0.0` "disabled" check
        // downstream and scale every gradient by NaN; ±inf is the same class
        // of non-finite tuning parameter. `0.0` (disable) and any finite
        // value stay legal.
        if !self.max_grad_norm.is_finite() {
            return Err(JammiError::FineTune(format!(
                "max_grad_norm must be finite, got {}",
                self.max_grad_norm
            )));
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
        // Monitoring a metric that will never be measured. Refused rather than
        // coerced to TrainLoss: the caller asked for two things that cannot both
        // hold, and silently picking one for them means the run they get is not
        // the run they configured. This catches the explicit zero; a fraction
        // that rounds to zero rows depends on the dataset size, which this
        // config cannot see, and is refused at the split instead.
        if self.validation_fraction == 0.0
            && self.early_stopping_metric == EarlyStoppingMetric::ValLoss
        {
            return Err(JammiError::FineTune(
                "early_stopping_metric=val_loss requires validation_fraction > 0; \
                 set early_stopping_metric=train_loss to train on the whole \
                 dataset, or raise validation_fraction to hold out a split"
                    .into(),
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

// ─── Held-out evaluation seam (H1, unit 63) ────────────────────────────────
//
// `evaluate_held_out` (`jammi-ai`, ai-core domain) scores a trained model
// against a committed held-out split and returns a `HeldOutLoss`. The
// numbers here are the seam's public RESULT vocabulary; no computation lives
// in this crate, and landing these types adds no new RPC or wire endpoint.

/// One held-out example's stable id and the model's loss on it.
///
/// `example_id` is the STABLE id from the committed fixture (a string triple
/// id, `cookbook/fixtures/finetune_heldout/heldout_ids.txt` — H3), never a
/// row index: index-based ids would silently repoint at a different example
/// if the fixture's row order ever changed, which a stable string id cannot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExampleLoss {
    /// Stable id of the held-out example, from the committed fixture's id list.
    pub example_id: String,
    /// The model's loss on this example.
    pub loss: f64,
}

/// Result of one `evaluate_held_out` call: per-example losses over a
/// committed held-out split, plus the aggregate and partition metadata a
/// paired significance test (C16/H2) consumes.
///
/// **`mean` is the EXAMPLE-mean** — the plain arithmetic mean of
/// [`Self::per_example`]'s losses — and is a NEW quantity distinct from
/// `Trainer::evaluate`'s existing (private, monitoring-only) BATCH-mean: the
/// mean of per-BATCH losses, which weights a short final batch the same as a
/// full one. That legacy batch-mean semantics stays UNTOUCHED by this seam
/// (early stopping, `checkpoint_best`, and every pinned value that reads it
/// keep reading exactly what they read today) — `evaluate_held_out`
/// DELEGATES to the same forward pass but additionally captures each row's
/// individual loss before it is reduced, and this struct's `mean` is computed
/// over THAT per-example array, never re-derived from the batch-mean.
///
/// **Batch partition is part of the measurement's identity.** The
/// Multiple-Negatives-Ranking objective's per-example loss is BATCH-COUPLED:
/// for a batch of `(anchor, positive)` rows, every other row's positive in
/// the SAME batch is scored as this row's in-batch negative, so a given
/// example's loss depends on which other examples share its batch, not on
/// the example alone. Re-partitioning the identical example set into
/// different batches (a different `batch_size`, a different row order, a
/// different shard) changes every `per_example` value even though nothing
/// about the model or the examples changed. Because the batch partition is
/// therefore a property of *(model, partition)* and not a property of the
/// example set alone, [`Self::batch_partition_sha256`] and
/// [`Self::in_batch_negatives_per_example`] are recorded ON THIS STRUCT
/// (v2 delta 9) rather than left to be inferred from the held-out split
/// alone — a `HeldOutLoss` from a re-partitioned run is NOT directly
/// comparable to one from this run even over the identical example ids, and
/// the two hashes make that non-comparability checkable rather than silent.
/// This is also why the held-out split is sized to a MULTIPLE of
/// `batch_size` via an explicit committed id list (v2 delta 2) rather than
/// `validation_fraction` rounding: it fixes every example at the same
/// in-batch-negative count for objectives that have one, so
/// `in_batch_negatives_per_example` is one number, not a per-example
/// distribution with a ragged final batch.
///
/// **`in_batch_negatives_per_example` is objective-aware** (audit round 63,
/// finding 6): `batch_size - 1` for the MNRL objectives (`Pairs`, always;
/// `Triplet` when `MultipleNegativesRanking` is configured), which score each
/// row against every OTHER row's positive sharing the batch. It is `0` for
/// every other objective this seam supports (`Triplet` margin,
/// `Contrastive`/`CosineMse`, `Classification`) — genuinely zero, not a
/// placeholder: those objectives score each row independently of every other
/// row in the batch, so there is no in-batch-negative pool to size. See the
/// field's own doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeldOutLoss {
    /// Per-example losses, in the held-out split's fixed (committed) id order.
    pub per_example: Vec<ExampleLoss>,
    /// The EXAMPLE-mean of [`Self::per_example`]'s losses — `sum(loss) /
    /// count`, NOT `Trainer::evaluate`'s private batch-mean. See the struct
    /// doc for why these are two distinct quantities.
    pub mean: f64,
    /// `per_example.len()`. Redundant with `per_example` BY CONSTRUCTION —
    /// kept (rather than dropped) because the contract lists it as a field a
    /// consumer reads directly (e.g. to size a sign-test cell) without first
    /// counting `per_example`, mirroring `CalibrationAggregate::n` beside
    /// `CalibrationEvalReport::per_record` in `jammi_wire::eval::report`. Any
    /// constructor of this type MUST set `count == per_example.len()`; the
    /// invariant is pinned by this module's tests, not enforced by a private
    /// field (this crate's report types keep every field public, matching
    /// [`FineTuneConfig`] and the `jammi_wire::eval::report` shapes).
    pub count: usize,
    /// Fraction of [`Self::per_example`] whose loss sits exactly at the
    /// objective's hinge floor / tie value — the loss a perfectly-ranked (or
    /// exactly-tied) row saturates to and cannot improve past. A
    /// hinge-shaped objective (e.g. `Triplet`, or an MNRL batch with a
    /// saturated in-batch ranking) can drive many rows to this floor well
    /// before the model has converged elsewhere, so a high `tie_fraction`
    /// flags that per-example loss is losing resolution on those rows — the
    /// per-pair statistic (C16/H2) still pairs by example, but a saturated
    /// tie fraction is a caveat on how much signal that pairing carries.
    /// `1.0` when every held-out example is at the floor (the saturated-hinge
    /// case).
    pub tie_fraction: f64,
    /// SHA-256 (hex) of the batch partition this result was scored under —
    /// which examples shared a batch, and in what order. A property of
    /// *(model, partition)*, not of the example set alone; see the struct
    /// doc. Two `HeldOutLoss` values are only directly comparable per-example
    /// when this hash (and [`Self::in_batch_negatives_per_example`]) match.
    pub batch_partition_sha256: String,
    /// The number of in-batch negatives every held-out example was scored
    /// against — **objective-aware** (audit round 63, finding 6; the field
    /// used to be `batch_size - 1` unconditionally, which was only correct
    /// for the MNRL objectives and silently mis-described every other one):
    ///
    /// - MNRL objectives (`Pairs`, always; `Triplet` when
    ///   `MultipleNegativesRanking` is configured): `batch_size - 1`, because
    ///   the held-out split is sized to a multiple of `batch_size` (v2 delta
    ///   2), so no batch is short and every example has the same
    ///   negative-pool size.
    /// - Every other objective this seam supports (`Triplet` margin,
    ///   `Contrastive`/`CosineMse`, `Classification`): `0`. These score each
    ///   row independently of every other row in the batch — `0` is their
    ///   TRUE in-batch-negative count, not a sentinel or a fallback.
    ///
    /// A single scalar (not a per-example distribution) precisely because
    /// the held-out split is scored under one homogeneous objective/batch
    /// kind throughout, so this count cannot vary example-to-example within
    /// one `HeldOutLoss`.
    pub in_batch_negatives_per_example: usize,
}

#[cfg(test)]
mod held_out_loss_tests {
    use super::*;

    fn example(id: &str, loss: f64) -> ExampleLoss {
        ExampleLoss {
            example_id: id.to_string(),
            loss,
        }
    }

    fn sample_held_out_loss() -> HeldOutLoss {
        let per_example = vec![
            example("arxiv:0001.0001v1#0", 0.10),
            example("arxiv:0001.0002v1#0", 0.20),
            example("arxiv:0001.0003v1#0", 0.30),
            example("arxiv:0001.0004v1#0", 0.10),
        ];
        let count = per_example.len();
        let mean = per_example.iter().map(|e| e.loss).sum::<f64>() / count as f64;
        HeldOutLoss {
            per_example,
            mean,
            count,
            tie_fraction: 0.5,
            batch_partition_sha256: "a".repeat(64),
            in_batch_negatives_per_example: 7,
        }
    }

    /// Construction + serde round-trip: the shape a consumer actually
    /// deserializes off the wire matches what was constructed, field for
    /// field — the same guarantee `FineTuneConfig`'s round-trip tests pin for
    /// the request side.
    #[test]
    fn serde_round_trips() {
        let original = sample_held_out_loss();
        let json = serde_json::to_string(&original).expect("HeldOutLoss serializes");
        let decoded: HeldOutLoss = serde_json::from_str(&json).expect("HeldOutLoss deserializes");
        assert_eq!(decoded, original);

        let example_json =
            serde_json::to_string(&original.per_example[0]).expect("ExampleLoss serializes");
        let decoded_example: ExampleLoss =
            serde_json::from_str(&example_json).expect("ExampleLoss deserializes");
        assert_eq!(decoded_example, original.per_example[0]);
    }

    /// `count` is redundant with `per_example.len()` by construction (the
    /// field doc's stated invariant) — pinned here so a future edit that lets
    /// the two drift apart is caught. This is the invariant note the field
    /// doc promises, made checkable.
    #[test]
    fn count_matches_per_example_len_by_construction() {
        let held_out = sample_held_out_loss();
        assert_eq!(held_out.count, held_out.per_example.len());
    }

    /// `mean` is the EXAMPLE-mean over `per_example` — a plain arithmetic
    /// mean of the per-example losses, independent of any batch grouping.
    /// This is the "example-mean, not the legacy batch-mean" distinction the
    /// struct doc documents, pinned as a computable check on the type's own
    /// shape (the seam's actual computation is `jammi_ai`'s to test).
    #[test]
    fn mean_is_the_example_mean() {
        let held_out = sample_held_out_loss();
        let expected: f64 =
            held_out.per_example.iter().map(|e| e.loss).sum::<f64>() / held_out.count as f64;
        assert!(
            (held_out.mean - expected).abs() < 1e-12,
            "mean {} must equal the plain per-example mean {expected}",
            held_out.mean
        );
    }

    /// `tie_fraction == 1.0` is a legal, representable value — the saturated
    /// case where every held-out example sits at the hinge floor. Pinned as a
    /// positive control on the field's documented range, mirroring how
    /// `FineTuneConfig`'s boundary tests pin legal edge values.
    #[test]
    fn tie_fraction_one_is_representable() {
        let mut held_out = sample_held_out_loss();
        held_out.tie_fraction = 1.0;
        let json = serde_json::to_string(&held_out).unwrap();
        let decoded: HeldOutLoss = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.tie_fraction, 1.0);
    }

    /// `batch_partition_sha256` and `in_batch_negatives_per_example` live ON
    /// `HeldOutLoss` (v2 delta 9) and survive the wire round-trip alongside
    /// the per-example data — the property-of-(model, partition) fields are
    /// not dropped or defaulted away.
    #[test]
    fn partition_identity_fields_round_trip() {
        let held_out = sample_held_out_loss();
        let json = serde_json::to_string(&held_out).unwrap();
        let decoded: HeldOutLoss = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.batch_partition_sha256,
            held_out.batch_partition_sha256
        );
        assert_eq!(
            decoded.in_batch_negatives_per_example,
            held_out.in_batch_negatives_per_example
        );
    }

    /// Two `HeldOutLoss` values over the identical example set but a
    /// DIFFERENT batch partition carry different `batch_partition_sha256` —
    /// the type does not collapse "same examples" into "same measurement".
    /// This is a shape-level sanity check (construction only); the actual
    /// per-example divergence under re-partitioning is MNRL's batch-coupling
    /// behaviour, `jammi_ai`'s to test.
    #[test]
    fn differing_partitions_carry_differing_partition_hashes() {
        let a = sample_held_out_loss();
        let mut b = a.clone();
        b.batch_partition_sha256 = "b".repeat(64);
        assert_ne!(a.batch_partition_sha256, b.batch_partition_sha256);
        assert_eq!(
            a.per_example, b.per_example,
            "same example ids/losses in this construction check"
        );
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    /// RED for #347: a run cannot monitor a metric it will never measure.
    ///
    /// `validation_fraction = 0.0` holds out nothing, so under the DEFAULT
    /// `early_stopping_metric = ValLoss` the trainer monitored a validation loss
    /// that was never computed. `evaluate` returned a `0.0` sentinel for the
    /// empty split, so epoch 0 won `0.0 < f64::MAX` and wrote `checkpoint_best`,
    /// every later epoch failed `0.0 < 0.0` and burned patience, the loop broke
    /// at `patience + 1` epochs, and the epoch-0 adapter was published as the
    /// run's result with a reported `final_loss` of 0.0. A silently untrained
    /// model reported as perfect.
    ///
    /// Refused, not coerced. The issue proposes auto-switching the metric to
    /// `TrainLoss`; that silently gives the caller a different run than the one
    /// they configured, and the two settings are equally plausible as the
    /// intended one.
    #[test]
    fn zero_validation_fraction_with_val_loss_is_refused() {
        let cfg = FineTuneConfig {
            validation_fraction: 0.0,
            early_stopping_metric: EarlyStoppingMetric::ValLoss,
            ..Default::default()
        };
        let err = cfg
            .validate()
            .expect_err("val_loss over an empty split must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("validation_fraction"),
            "the error must name the field: {msg}"
        );
        assert!(
            msg.contains("train_loss"),
            "the error must name the usable alternative: {msg}"
        );
    }

    /// Positive control: the refusal is narrow. Each setting is legal on its own
    /// and only the combination is refused, so this cannot pass by rejecting
    /// everything.
    #[test]
    fn the_zero_split_refusal_is_narrow() {
        FineTuneConfig {
            validation_fraction: 0.0,
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            ..Default::default()
        }
        .validate()
        .expect("no split + train_loss is the documented way to train on everything");

        FineTuneConfig {
            validation_fraction: 0.2,
            early_stopping_metric: EarlyStoppingMetric::ValLoss,
            ..Default::default()
        }
        .validate()
        .expect("a real split + val_loss is the default shape");

        FineTuneConfig::default()
            .validate()
            .expect("the shipped default must remain valid");
    }

    /// A non-finite `max_grad_norm` is refused at BOTH edges — `validate`
    /// (a config built in-process) and deserialization (the
    /// `deserialize_with` guard, driven through a value deserializer since
    /// JSON has no spelling for NaN/inf; an over-range JSON literal is
    /// refused by `serde_json` itself, pinned too). Mutation tried: delete
    /// the `is_finite` guard in `finite_max_grad_norm` — the deserializer
    /// half goes red; delete the `validate` clause — the `validate` half
    /// goes red.
    #[test]
    fn nonfinite_max_grad_norm_is_refused_at_the_config_edge() {
        use serde::de::IntoDeserializer;

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let cfg = FineTuneConfig {
                max_grad_norm: bad,
                ..Default::default()
            };
            let err = cfg
                .validate()
                .expect_err("a non-finite max_grad_norm must be refused by validate");
            assert!(
                err.to_string().contains("max_grad_norm"),
                "the error must name the field: {err}"
            );

            let de: serde::de::value::F64Deserializer<serde::de::value::Error> =
                bad.into_deserializer();
            let err = finite_max_grad_norm(de)
                .expect_err("a non-finite max_grad_norm must be refused at deserialization");
            assert!(
                err.to_string().contains("max_grad_norm"),
                "the error must name the field: {err}"
            );
        }

        let over_range = default_json_with_max_grad_norm("1e999");
        assert!(
            serde_json::from_str::<FineTuneConfig>(&over_range).is_err(),
            "an over-range JSON literal must not deserialize"
        );
    }

    /// The serialized default config with its `max_grad_norm` value replaced
    /// by the literal `value` — a complete request body, not a bare field.
    fn default_json_with_max_grad_norm(value: &str) -> String {
        let json = serde_json::to_string(&FineTuneConfig::default()).unwrap();
        assert!(
            json.contains(r#""max_grad_norm":1.0"#),
            "default shape: {json}"
        );
        json.replace(
            r#""max_grad_norm":1.0"#,
            &format!(r#""max_grad_norm":{value}"#),
        )
    }

    /// Positive control: the refusal is narrow — the shipped default, an
    /// explicit disable (`0.0`), and an ordinary finite value all still
    /// deserialize and validate.
    #[test]
    fn finite_max_grad_norm_still_round_trips() {
        for good in ["1.0", "0.0", "2.5"] {
            let cfg: FineTuneConfig =
                serde_json::from_str(&default_json_with_max_grad_norm(good)).unwrap();
            cfg.validate().unwrap();
            assert!(cfg.max_grad_norm.is_finite());
            assert_eq!(cfg.max_grad_norm, good.parse::<f64>().unwrap());
        }
    }
}
