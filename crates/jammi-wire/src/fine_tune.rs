//! Fine-tune request vocabulary: the transport-neutral config + method types.
//!
//! These are the knobs a fine-tune request carries — `FineTuneConfig`, the loss
//! and schedule enums, and the `FineTuneMethod` selector. They hold no engine
//! state, so they live on the wire substrate: the embedded engine reads them to
//! build a training run, the gRPC converters encode/decode them, and a
//! data-plane client builds a request from them without the candle stack.

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

/// The display name of `loss` if it is a pairwise-ordering embedding
/// objective — CoSENT, AnglE, or the `None` default (which resolves to
/// CoSENT) — or `None` for every non-ordering arm (`CosineMse` regresses
/// onto the score directly and has no per-batch pair requirement;
/// `MultipleNegativesRanking`/`Triplet` are not graded-pair objectives at
/// all). Used by [`FineTuneConfig::validate`]'s batch-size check. The engine
/// crate's data-loader validation and per-batch degenerate-batch diagnostic
/// answer the same "is this an ordering objective" question independently (a
/// duplicated ~3-line match, not a cross-crate call) — this type has no
/// downstream dependents to share it with.
fn ordering_objective_name(loss: Option<EmbeddingLoss>) -> Option<&'static str> {
    match loss {
        None | Some(EmbeddingLoss::CoSent) => Some("CoSENT"),
        Some(EmbeddingLoss::AnglE) => Some("AnglE"),
        Some(EmbeddingLoss::CosineMse)
        | Some(EmbeddingLoss::MultipleNegativesRanking { .. })
        | Some(EmbeddingLoss::Triplet { .. }) => None,
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
        // An ORDERING objective (CoSENT, AnglE — and the `None` default, which
        // resolves to CoSENT) scores `scores[i] < scores[j]` pairs *within* a
        // batch: with `batch_size < 2` no such pair can ever exist, so every
        // batch trains at the loss's `log(1) = 0` floor with zero gradient —
        // the run "converges" instantly and silently, with nothing learned.
        // Refused at the input edge, naming the objective and the minimum,
        // rather than left to surface as an unexplained zero loss mid-run.
        if let Some(name) = ordering_objective_name(self.embedding_loss) {
            if self.batch_size < 2 {
                return Err(JammiError::FineTune(format!(
                    "{name} is a pairwise-ordering objective: it needs at least 2 rows per \
                     batch to form a score-ordered pair, got batch_size={}. Raise batch_size \
                     to at least 2, or choose a non-ordering embedding_loss \
                     (e.g. CosineMse) for batch_size=1.",
                    self.batch_size
                )));
            }
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

    // ── B3(a): an ordering objective needs batch_size ≥ 2 ─────────────────

    /// `batch_size = 1` under the DEFAULT (`embedding_loss: None`, which
    /// resolves to CoSENT) objective is refused: with one row per batch there
    /// is never a `scores[i] < scores[j]` pair to form, so every batch trains
    /// at CoSENT's `log(1) = 0` floor with zero gradient — the run
    /// "converges" instantly and silently. The error must name the objective
    /// and the minimum batch size.
    #[test]
    fn batch_size_one_with_default_ordering_objective_is_refused() {
        let cfg = FineTuneConfig {
            batch_size: 1,
            embedding_loss: None,
            ..Default::default()
        };
        let err = cfg
            .validate()
            .expect_err("batch_size=1 under the ordering default must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("CoSENT"),
            "the error must name the objective: {msg}"
        );
        assert!(
            msg.contains('2'),
            "the error must name the minimum batch size: {msg}"
        );
    }

    /// Same refusal, explicit `Some(AnglE)` — named as AnglE, not CoSENT.
    #[test]
    fn batch_size_one_with_angle_objective_is_refused() {
        let cfg = FineTuneConfig {
            batch_size: 1,
            embedding_loss: Some(EmbeddingLoss::AnglE),
            ..Default::default()
        };
        let err = cfg
            .validate()
            .expect_err("batch_size=1 under AnglE must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("AnglE"),
            "the error must name the objective: {msg}"
        );
    }

    /// Positive control: the ordering-objective refusal is narrow. `batch_size
    /// = 1` is legal under a non-ordering objective (`CosineMse` has no
    /// per-batch pair requirement — it regresses onto the score directly),
    /// and `batch_size = 2` is legal under every ordering objective — only
    /// the `(ordering objective, batch_size < 2)` combination is refused.
    #[test]
    fn ordering_batch_size_refusal_is_narrow() {
        FineTuneConfig {
            batch_size: 1,
            embedding_loss: Some(EmbeddingLoss::CosineMse),
            ..Default::default()
        }
        .validate()
        .expect("batch_size=1 is legal for a non-ordering objective");

        for loss in [
            None,
            Some(EmbeddingLoss::CoSent),
            Some(EmbeddingLoss::AnglE),
        ] {
            FineTuneConfig {
                batch_size: 2,
                embedding_loss: loss,
                ..Default::default()
            }
            .validate()
            .unwrap_or_else(|e| {
                panic!("batch_size=2 must satisfy every ordering objective ({loss:?}): {e}")
            });
        }
    }
}
