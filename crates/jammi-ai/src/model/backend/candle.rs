use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::ComputeDevice;
use jammi_encoders::{
    Bert, BertConfig, DistilBert, DistilBertConfig, ModernBert, ModernBertConfig, Pooling,
};

use jammi_encoders::{
    ClipText, ClipTextConfig, HtsatAudio, HtsatAudioConfig, OpenClipVisionConfig,
    OpenClipVisionTransformer,
};

use super::open_clip_text::OpenClipTextForward;
use super::{DeviceConfig, ModelBackend};
use crate::fine_tune::classifier::SeqClassifier;
use crate::inference::adapter::BackendOutput;
use crate::inference::{
    arrow_to_audio, arrow_to_images, arrow_to_texts, audio_preprocess, image_preprocess,
};
use crate::model::tokenizer::{BatchEncoding, TokenizerWrapper};
use crate::model::{LoadedModel, ModelDimensions, ModelTask, ResolvedModel, TokenizerSource};

/// Candle backend — loads safetensors models via candle.
pub struct CandleBackend;

/// Text architectures produce hidden states from tokenized input.
///
/// `forward_hidden` returns `[batch, seq, hidden]` per-token hidden states
/// for classification / NER paths. `forward_pooled` returns the final
/// `[batch, output_dim]` pooled-and-L2-normalized embedding used by the
/// embedding path; the three BERT-family wrappers override it to pool with
/// the strategy their model's `1_Pooling/config.json` declares (mean
/// fallback when the file is absent — see `pooling_from_config`), while the
/// OpenCLIP text tower overrides it to expose its pre-pooled projected
/// output directly. The trait-default implementation (mean-pool +
/// L2-normalize over `forward_hidden`) is the base for any future wrapper
/// that doesn't need model-declared pooling.
pub(crate) trait CandleTextForward: Send + Sync {
    /// The longest token sequence this encoder accepts (`max_position_embeddings`
    /// for BERT-family towers, `context_length` for the OpenCLIP text tower).
    /// The text path truncates tokenization to this bound so an over-long input
    /// yields a truncated embedding rather than a hard forward failure — the
    /// limit is the model's own, never a fixed constant that silently drops a
    /// CLIP row (context 77) at a BERT-shaped 512.
    fn max_sequence_length(&self) -> usize;

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor>;

    /// Pooled and L2-normalized `[batch, output_dim]` embedding. Default
    /// implementation mean-pools the masked output of `forward_hidden` and
    /// L2-normalizes it; encoders that need a different or model-declared
    /// strategy (the BERT family) or whose `forward_hidden` is already
    /// pooled (e.g. OpenCLIP text) override this directly.
    fn forward_pooled(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, attention_mask, encoding, device)?;
        let pooled = mean_pool(&hidden, attention_mask)?;
        l2_normalize(&pooled)
    }
}

/// Vision architectures produce embeddings from pixel tensors.
/// Preprocessing config (mean, std, image_size) is model-driven.
pub(crate) trait CandleVisionForward: Send + Sync {
    fn forward_image(&self, pixel_values: &Tensor) -> Result<Tensor>;
    fn image_size(&self) -> usize;
    fn preprocess_mean(&self) -> [f32; 3];
    fn preprocess_std(&self) -> [f32; 3];
}

impl CandleVisionForward for OpenClipVisionTransformer {
    fn forward_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.forward(pixel_values)
            .map_err(|e| JammiError::Inference(format!("Vision forward pass failed: {e}")))
    }
    fn image_size(&self) -> usize {
        self.image_size()
    }
    fn preprocess_mean(&self) -> [f32; 3] {
        self.preprocess_mean()
    }
    fn preprocess_std(&self) -> [f32; 3] {
        self.preprocess_std()
    }
}

/// Audio architectures produce embeddings from a 4-channel CLAP fusion
/// spectrogram. The bytes-to-spectrogram front-end geometry (sample rate, FFT
/// size, hop, mel band) is owned by the feature-extractor `ClapFrontendConfig`
/// read off `preprocessor_config.json`, not the tower; the tower reports only
/// `num_mel_bins`, which the front-end's mel-filter count must match.
pub(crate) trait CandleAudioForward: Send + Sync {
    /// Pooled, L2-normalized `[batch, projection_dim]` embedding for a
    /// `[batch, 4, time, num_mel_bins]` CLAP fusion spectrogram batch. `is_longer`
    /// gates the per-sample fusion path in the patch embedding.
    fn forward_audio(&self, input_features: &Tensor, is_longer: &[bool]) -> Result<Tensor>;
    /// Mel bins the input fusion spectrogram must carry.
    fn num_mel_bins(&self) -> usize;
}

impl CandleAudioForward for HtsatAudio {
    fn forward_audio(&self, input_features: &Tensor, is_longer: &[bool]) -> Result<Tensor> {
        self.forward(input_features, is_longer)
            .map_err(|e| JammiError::Inference(format!("Audio forward pass failed: {e}")))
    }
    fn num_mel_bins(&self) -> usize {
        self.num_mel_bins()
    }
}

/// Pool+L2-normalize `hidden` per `strategy`, via the single shared
/// implementation in `jammi_encoders` — DRY across every BERT-family
/// embedding wrapper's `forward_pooled` override.
fn pool_via(hidden: &Tensor, attention_mask: &Tensor, strategy: Pooling) -> Result<Tensor> {
    jammi_encoders::pool_and_normalize(hidden, attention_mask, strategy)
        .map_err(|e| JammiError::Inference(format!("pooling failed: {e}")))
}

/// BERT-family forward pass (bert, roberta, camembert, xlm-roberta). Carries
/// the pooling strategy the model's `1_Pooling/config.json` declares (mean
/// fallback if the file is absent) and pools+normalizes with it.
struct BertForward {
    model: Bert,
    pooling: Pooling,
}

impl CandleTextForward for BertForward {
    fn max_sequence_length(&self) -> usize {
        self.model.max_seq_length()
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.model
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("BERT forward pass failed: {e}")))
    }

    fn forward_pooled(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, attention_mask, encoding, device)?;
        pool_via(&hidden, attention_mask, self.pooling)
    }
}

/// ModernBERT forward pass (rotary embeddings, GeGLU, no token_type_ids).
/// Carries the pooling strategy the model's `1_Pooling/config.json` declares
/// (mean fallback if the file is absent) and pools+normalizes with it.
struct ModernBertForward {
    model: ModernBert,
    pooling: Pooling,
}

impl CandleTextForward for ModernBertForward {
    fn max_sequence_length(&self) -> usize {
        self.model.max_seq_length()
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.model
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("ModernBERT forward pass failed: {e}")))
    }

    fn forward_pooled(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, attention_mask, encoding, device)?;
        pool_via(&hidden, attention_mask, self.pooling)
    }
}

/// DistilBERT forward pass (no token_type_ids, different architecture from
/// BERT). Carries the pooling strategy the model's `1_Pooling/config.json`
/// declares (mean fallback if the file is absent) and pools+normalizes with
/// it.
struct DistilBertForward {
    model: DistilBert,
    pooling: Pooling,
}

impl CandleTextForward for DistilBertForward {
    fn max_sequence_length(&self) -> usize {
        self.model.max_seq_length()
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.model
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("DistilBERT forward pass failed: {e}")))
    }

    fn forward_pooled(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, attention_mask, encoding, device)?;
        pool_via(&hidden, attention_mask, self.pooling)
    }
}

/// DistilBERT sequence classification: encoder → CLS → pre_classifier → ReLU → classifier → softmax.
struct DistilBertClassificationForward {
    distilbert: DistilBert,
    pre_classifier: candle_nn::Linear,
    classifier: candle_nn::Linear,
}

impl CandleTextForward for DistilBertClassificationForward {
    fn max_sequence_length(&self) -> usize {
        self.distilbert.max_seq_length()
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        let hidden = self
            .distilbert
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| {
                JammiError::Inference(format!("DistilBERT classification forward failed: {e}"))
            })?;
        let cls = hidden
            .i((.., 0, ..))
            .map_err(|e| JammiError::Inference(format!("CLS pooling failed: {e}")))?
            .contiguous()
            .map_err(|e| JammiError::Inference(format!("CLS contiguous failed: {e}")))?;
        let pre = cls
            .apply(&self.pre_classifier)
            .map_err(|e| JammiError::Inference(format!("Pre-classifier failed: {e}")))?
            .relu()
            .map_err(|e| JammiError::Inference(format!("ReLU failed: {e}")))?;
        let logits = pre
            .apply(&self.classifier)
            .map_err(|e| JammiError::Inference(format!("Classifier forward failed: {e}")))?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(|e| JammiError::Inference(format!("Softmax failed: {e}")))
    }
}

/// ModernBERT sequence classification forward pass.
/// Returns softmaxed logits of shape (batch, num_classes).
struct ModernBertClassificationForward(SeqClassifier);

impl CandleTextForward for ModernBertClassificationForward {
    fn max_sequence_length(&self) -> usize {
        self.0.max_seq_length()
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        let logits = self.0.forward(input_ids, attention_mask).map_err(|e| {
            JammiError::Inference(format!("ModernBERT classification forward failed: {e}"))
        })?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(|e| JammiError::Inference(format!("Softmax failed: {e}")))
    }
}

/// BERT-family sequence classification forward pass.
/// Applies CLS pooling + linear classifier + softmax on top of Bert.
struct BertClassificationForward {
    bert: Bert,
    classifier: candle_nn::Linear,
}

impl CandleTextForward for BertClassificationForward {
    fn max_sequence_length(&self) -> usize {
        self.bert.max_seq_length()
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        let hidden = self
            .bert
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| {
                JammiError::Inference(format!("BERT classification forward failed: {e}"))
            })?;
        // CLS pooling: take first token
        let cls = hidden
            .i((.., 0, ..))
            .map_err(|e| JammiError::Inference(format!("CLS pooling failed: {e}")))?
            .contiguous()
            .map_err(|e| JammiError::Inference(format!("CLS contiguous failed: {e}")))?;
        let logits = cls
            .apply(&self.classifier)
            .map_err(|e| JammiError::Inference(format!("Classifier forward failed: {e}")))?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(|e| JammiError::Inference(format!("Softmax failed: {e}")))
    }
}

/// A candle-loaded model ready for inference.
pub struct CandleModel {
    /// Architecture dimensions for memory estimation and output sizing.
    pub dimensions: ModelDimensions,
    /// Text architecture forward pass (BERT, ModernBERT, DistilBERT).
    text: Option<Box<dyn CandleTextForward>>,
    /// Vision architecture forward pass (OpenCLIP ViT).
    vision: Option<Box<dyn CandleVisionForward>>,
    /// Audio architecture forward pass (HTSAT-Swin CLAP audio tower).
    audio: Option<Box<dyn CandleAudioForward>>,
    /// CLAP fusion front-end geometry, read off `preprocessor_config.json`.
    /// `Some` exactly when `audio` is — the audio path turns raw bytes into the
    /// tower's 4-channel fusion spectrogram through it.
    audio_frontend: Option<audio_preprocess::ClapFrontendConfig>,
    /// Tokenizer for text-to-token conversion, if available.
    pub tokenizer: Option<TokenizerWrapper>,
    /// Device the model weights reside on (CPU, CUDA, or Metal).
    pub device: Device,
    /// Projection head applied after pooling on models fine-tuned via the
    /// `ProjectionHead` training target. `None` for base models and for
    /// models fine-tuned via the `EncoderAdapters` target (those carry
    /// their LoRA inside the encoder, not on top of it).
    pub projection_head: Option<jammi_lora::LoraLinear>,
    /// The regression `distribution` head layer (`hidden → output_dim`), reloaded
    /// for a `ModelTask::Regression` fine-tune. This is the layer the trainer's
    /// `regress` applies to the pooled embedding (NOT `projection_head`), so
    /// serving applies it to reproduce the trained `(mean, raw_std)` / quantile
    /// output. `None` for every non-regression head.
    distribution_head: Option<jammi_lora::LoraLinear>,
    /// De-standardising affine for a regression head, reloaded from the adapter
    /// config. The regression head learns z-space parameters and emits a raw,
    /// outcome-unit distribution after this affine; serving applies the same
    /// affine the trainer did, so the served mean/quantiles carry the target
    /// offset. `None` for every non-regression head.
    regression_scaler: Option<crate::fine_tune::regression_loss::TargetScaler>,
    /// Predictive distribution form of a reloaded regression head: `Gaussian`
    /// (de-standardise the mean column only) or `Quantile` (de-standardise every
    /// column). This is the authoritative gaussian-vs-quantile signal persisted
    /// with the head; serving dispatches the de-standardisation on it rather than
    /// on head width, so a 2-level quantile head (also width 2) is de-standardised
    /// as a quantile head. `Some` exactly when `regression_scaler` is.
    regression_form: Option<crate::inference::adapter::DistributionForm>,
    /// Label index → label string mapping for classification/NER models.
    id2label: Option<HashMap<u32, String>>,
    /// Token-level classifier for NER models (applied per token, no pooling).
    ner_classifier: Option<candle_nn::Linear>,
    /// The effective inference compute precision this model loaded at — the
    /// resolved per-model override or the global default, unless a saved
    /// fine-tune adapter's own persisted `backbone_dtype` won instead (see
    /// `effective_precision` in `CandleBackend::load`). Output-affecting, so
    /// the materialization contract folds it into `ModelIdentity`.
    pub(crate) compute_precision: jammi_numerics::ComputePrecision,
}

/// Mean-pool the `[batch, seq, hidden]` tensor along seq using
/// `attention_mask` to zero out padding positions.
pub(crate) fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask
        .unsqueeze(2)
        .map_err(|e| JammiError::Inference(e.to_string()))?
        .to_dtype(hidden.dtype())
        .map_err(|e| JammiError::Inference(e.to_string()))?;
    let masked = hidden
        .broadcast_mul(&mask)
        .map_err(|e| JammiError::Inference(e.to_string()))?;
    let sum = masked
        .sum(1)
        .map_err(|e| JammiError::Inference(e.to_string()))?;
    let count = mask
        .sum(1)
        .map_err(|e| JammiError::Inference(e.to_string()))?;
    sum.broadcast_div(&count)
        .map_err(|e| JammiError::Inference(e.to_string()))
}

/// L2-normalize each row of a `[batch, dim]` tensor.
pub(crate) fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    let norm = tensor
        .sqr()
        .map_err(|e| JammiError::Inference(e.to_string()))?
        .sum_keepdim(1)
        .map_err(|e| JammiError::Inference(e.to_string()))?
        .sqrt()
        .map_err(|e| JammiError::Inference(e.to_string()))?
        .clamp(1e-12, f64::MAX)
        .map_err(|e| JammiError::Inference(e.to_string()))?;
    tensor
        .broadcast_div(&norm)
        .map_err(|e| JammiError::Inference(e.to_string()))
}

/// Map a parsed `1_Pooling/config.json` to the engine's [`Pooling`] strategy,
/// failing loudly rather than silently mean-pooling a model whose declared
/// mode the engine cannot represent.
///
/// - `cfg == None` (the file is genuinely absent) → `Mean`, the historical
///   sentence-transformers default for bare BERT repos — observable via
///   `tracing::info!`, never silent.
/// - a present file must declare at least one recognized `pooling_mode_*`
///   flag as `true`, all recognized true flags must map to the *same*
///   [`Pooling`] strategy, and no true flag may be one the enum cannot
///   represent (e.g. `pooling_mode_lasttoken`) — any violation is a hard
///   error, because silently choosing `Mean` for an unrepresentable or
///   ambiguous declaration is exactly the confident-wrong-embedding bug this
///   mechanism exists to kill.
fn pooling_from_config(cfg: Option<&serde_json::Value>, model_id: &str) -> Result<Pooling> {
    let cfg = match cfg {
        None => {
            tracing::info!(
                model_id,
                "no 1_Pooling/config.json found; defaulting to mean pooling \
                 (the sentence-transformers default for bare BERT repos)"
            );
            return Ok(Pooling::Mean);
        }
        Some(cfg) => cfg,
    };

    let obj = cfg.as_object().ok_or_else(|| JammiError::Model {
        model_id: model_id.to_string(),
        message: "1_Pooling/config.json is not a JSON object".into(),
    })?;

    // `pooling_mode_mean_sqrt_len_tokens` divides the token sum by sqrt(len)
    // rather than len; both are positive scalar multiples of the same token
    // sum, and `pool_and_normalize` mandatorily L2-normalizes its output, so
    // the two are byte-identical post-normalization. This maps to `Mean`
    // deliberately — it is an exact equivalence, not an approximation.
    const RECOGNIZED: &[(&str, Pooling)] = &[
        ("pooling_mode_cls_token", Pooling::Cls),
        ("pooling_mode_mean_tokens", Pooling::Mean),
        ("pooling_mode_max_tokens", Pooling::Max),
        ("pooling_mode_weightedmean_tokens", Pooling::WeightedMean),
        ("pooling_mode_mean_sqrt_len_tokens", Pooling::Mean),
    ];

    let true_flags: Vec<&str> = obj
        .iter()
        .filter(|(k, v)| k.starts_with("pooling_mode_") && v.as_bool() == Some(true))
        .map(|(k, _)| k.as_str())
        .collect();

    if true_flags.is_empty() {
        return Err(JammiError::Model {
            model_id: model_id.to_string(),
            message: "1_Pooling/config.json is present but declares no true pooling_mode_* flag"
                .into(),
        });
    }

    let unsupported: Vec<&str> = true_flags
        .iter()
        .copied()
        .filter(|k| !RECOGNIZED.iter().any(|(name, _)| name == k))
        .collect();
    if !unsupported.is_empty() {
        return Err(JammiError::Model {
            model_id: model_id.to_string(),
            message: format!(
                "unsupported pooling mode(s) {unsupported:?} declared in 1_Pooling/config.json \
                 — the engine cannot serve this encoder correctly"
            ),
        });
    }

    let mut distinct: Vec<Pooling> = Vec::new();
    for flag in &true_flags {
        let strategy = RECOGNIZED
            .iter()
            .find(|(name, _)| name == flag)
            .map(|(_, p)| *p)
            .expect("flag already checked against RECOGNIZED above");
        if !distinct.contains(&strategy) {
            distinct.push(strategy);
        }
    }

    if distinct.len() > 1 {
        return Err(JammiError::Model {
            model_id: model_id.to_string(),
            message: format!(
                "1_Pooling/config.json declares {} distinct pooling strategies from \
                 {true_flags:?} — sentence-transformers concatenates enabled modes (output \
                 dim = n·hidden), which this engine's single-mode path cannot represent",
                distinct.len()
            ),
        });
    }

    Ok(distinct[0])
}

impl CandleModel {
    /// L2-normalize each vector in a [batch, hidden_size] tensor.
    pub(crate) fn l2_normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        l2_normalize(tensor)
    }

    /// Convert token ID vectors into a candle Tensor on this model's device.
    pub(crate) fn tokens_to_tensor(&self, vecs: &[Vec<u32>]) -> Result<Tensor> {
        tokens_to_tensor(vecs, &self.device)
    }

    /// The persisted predictive-distribution form of a reloaded regression head,
    /// or `None` for a non-regression model (or a regression head saved without a
    /// form). Serving reads this to select the `Infer` output adapter
    /// (`gaussian()` vs `quantile(levels)`) — the authoritative signal the head
    /// was trained for, never a head-width guess.
    pub(crate) fn regression_form(&self) -> Option<&crate::inference::adapter::DistributionForm> {
        self.regression_form.as_ref()
    }

    /// The persisted scaler's σ_y (target standard deviation), or `None` for a
    /// non-regression / no-scaler head. Serving reads this to scale a Gaussian
    /// head's served σ from z-units (σ_z, what the z-space loss trains) back to
    /// raw units (`σ_y·σ_z`) — the σ-axis de-standardise that mirrors the mean
    /// affine the backend already applies. The mean/quantile columns carry σ_y in
    /// their affine at the backend; only the post-softplus σ needs σ_y here.
    pub(crate) fn regression_std_scale(&self) -> Option<f32> {
        self.regression_scaler.as_ref().map(|s| s.std() as f32)
    }

    /// TEST-ONLY non-vacuity seam: zero the trained regression distribution
    /// head's LoRA `B` factor, collapsing the head to its zero-initialised base
    /// (`zeros(output_dim, hidden_size)`). A head in this state emits exactly the
    /// scaler offset `μ_y` for every input through the de-standardising affine
    /// (`μ_y + σ_y·0`), so a served prediction no longer tracks the input — the
    /// exact behaviour an *untrained* head exhibits.
    ///
    /// This is the in-process equivalent of an auditor destructively zeroing
    /// `distribution.lora_b` on disk: it lets the regression-surface tests prove
    /// their group-separation assertion FAILS when the head carries no learned
    /// signal, locking the tests against a future regression that drops the
    /// trained head on serve (the original Break 5). Production never calls it; it
    /// only mutates a per-test owned model, never a cached/shared one.
    #[doc(hidden)]
    pub fn zero_distribution_head_for_test(&mut self) {
        if let Some(head) = self.distribution_head.as_mut() {
            head.lora_b = head
                .lora_b
                .zeros_like()
                .expect("zeros_like on a loaded LoRA B tensor is infallible");
        }
    }

    /// Access the text forward pass, returning an error if this model has no
    /// text tower loaded — e.g. a CLAP checkpoint loaded audio-only.
    fn text_forward(&self) -> Result<&dyn CandleTextForward> {
        self.text
            .as_deref()
            .ok_or_else(|| JammiError::Inference("No text model loaded for this task".into()))
    }

    /// Run forward pass dispatching by task.
    pub fn forward(
        &self,
        content: &[arrow::array::ArrayRef],
        task: ModelTask,
    ) -> Result<BackendOutput> {
        match task {
            ModelTask::TextEmbedding => self.forward_embedding(content),
            ModelTask::ImageEmbedding => self.forward_image_embedding(content),
            ModelTask::AudioEmbedding => self.forward_audio_embedding(content),
            ModelTask::Classification => self.forward_classification(content),
            ModelTask::Ner => self.forward_ner(content),
            ModelTask::Regression => self.forward_regression(content),
        }
    }

    /// Forward a regression model: pool the encoder output and apply the
    /// fine-tuned distributional projection head, emitting the raw
    /// `(mean, raw_std)` Gaussian parameters per row. The
    /// [`DistributionAdapter`](crate::inference::adapter::DistributionAdapter)
    /// maps `raw_std` through `softplus + floor` into the served `predicted_std`,
    /// so the backend head stays in the unconstrained space the proper-scoring
    /// objective trained it in.
    ///
    /// A regression head is a fine-tuned projection head over the frozen
    /// encoder — the same serving shape classification fine-tunes use — so a
    /// base checkpoint with no such head cannot serve this task. That is a typed
    /// capability error, not a silent wrong output.
    fn forward_regression(&self, content: &[arrow::array::ArrayRef]) -> Result<BackendOutput> {
        // The regression `distribution` head — the `hidden → output_dim` layer
        // the trainer's `regress` applies to the pooled embedding. It is the
        // authoritative regression head; `projection_head` (a hidden→hidden layer)
        // is the trained projection that regression does NOT apply on the forward
        // path, so serving must use `distribution_head` to match training.
        let head = self.distribution_head.as_ref().ok_or_else(|| {
            JammiError::Inference(
                "Regression inference needs a fine-tuned distributional head; \
                 this model carries none. Fine-tune a ModelTask::Regression head first."
                    .into(),
            )
        })?;

        let texts = arrow_to_texts(content)?;
        let num_rows = texts.len();
        if num_rows == 0 {
            return Ok(BackendOutput {
                float_outputs: vec![vec![]],
                string_outputs: vec![],
                row_status: vec![],
                row_errors: vec![],
                shapes: vec![(0, 2)],
            });
        }

        let mut row_status = vec![true; num_rows];
        let mut row_errors = vec![String::new(); num_rows];
        let mut valid_indices = Vec::new();
        let mut valid_texts = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                row_status[i] = false;
                row_errors[i] = "Empty or null text input".into();
            } else {
                valid_indices.push(i);
                valid_texts.push(text.as_str());
            }
        }

        // An all-empty batch still produces a well-formed Gaussian-width head so
        // the adapter sees the expected shape; the (failed) rows are nulled.
        if valid_texts.is_empty() {
            return Ok(BackendOutput {
                float_outputs: vec![vec![0.0; num_rows * 2]],
                string_outputs: vec![],
                row_status,
                row_errors,
                shapes: vec![(num_rows, 2)],
            });
        }

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            JammiError::Inference("No tokenizer loaded for regression model".into())
        })?;
        let encoding = tokenizer.encode_batch(
            &valid_texts,
            Some(self.text_forward()?.max_sequence_length()),
        )?;
        let input_ids = self.tokens_to_tensor(&encoding.input_ids)?;
        let attention_mask = self.tokens_to_tensor(&encoding.attention_masks)?;

        // Pool through the frozen encoder, then apply the distributional head —
        // the same pooled-embedding → projection-head shape the embedding and
        // classification fine-tunes use. The head emits the raw distribution
        // parameters; the `DistributionAdapter` maps `raw_std` to a positive
        // served std.
        let pooled = self.text_forward()?.forward_pooled(
            &input_ids,
            &attention_mask,
            &encoding,
            &self.device,
        )?;
        let params = head
            .forward(&pooled)
            .map_err(|e| JammiError::Inference(format!("Regression head forward: {e}")))?;
        let params = if params.dtype() == DType::F32 {
            params
        } else {
            params
                .to_dtype(DType::F32)
                .map_err(|e| JammiError::Inference(format!("Regression head dtype cast: {e}")))?
        };
        // De-standardise the raw head output with the affine the trainer carried
        // on the head: the mean column (Gaussian) or every quantile column maps
        // `μ_y + σ_y·z`, so the served distribution carries the target offset. A
        // base/no-scaler head leaves the output untouched. The σ column stays raw
        // here — the loss trained it in z-space (σ_z), so the `DistributionAdapter`
        // turns it into a positive served std via softplus AND scales it by σ_y
        // (`σ_y·softplus(raw)`), the σ-axis half of the de-standardise contract.
        //
        // The gaussian-vs-quantile choice dispatches on the persisted
        // `DistributionForm`, the authoritative signal the head was trained for —
        // never on head width, which cannot tell a 2-level quantile head from a
        // Gaussian one (both are width 2). The scaler and the form are persisted
        // together, so a scaler without a form is an inconsistent saved head, not
        // a case to paper over with a width guess.
        let params = if let Some(scaler) = self.regression_scaler.as_ref() {
            let form = self.regression_form.as_ref().ok_or_else(|| {
                JammiError::Inference(
                    "regression head carries a de-standardising scaler but no distribution form \
                     (the persisted head is inconsistent)"
                        .into(),
                )
            })?;
            scaler
                .destandardize(&params, form)
                .map_err(|e| JammiError::Inference(format!("Regression de-standardise: {e}")))?
        } else {
            params
        };
        let rows = params
            .to_vec2::<f32>()
            .map_err(|e| JammiError::Inference(format!("Regression head to vec: {e}")))?;

        // The head output width is its number of distribution parameters (2 for
        // the Gaussian form, one per level for the quantile form). Derive it
        // from the tensor so the backend never hard-codes a head shape.
        let head_width = rows.first().map_or(2, Vec::len);
        let mut flat = vec![0.0_f32; num_rows * head_width];
        for (batch_idx, &orig_idx) in valid_indices.iter().enumerate() {
            let row = &rows[batch_idx];
            flat[orig_idx * head_width..orig_idx * head_width + head_width].copy_from_slice(row);
        }

        Ok(BackendOutput {
            float_outputs: vec![flat],
            string_outputs: vec![],
            row_status,
            row_errors,
            shapes: vec![(num_rows, head_width)],
        })
    }

    fn forward_embedding(&self, content: &[arrow::array::ArrayRef]) -> Result<BackendOutput> {
        let texts = arrow_to_texts(content)?;
        let num_rows = texts.len();

        if num_rows == 0 {
            return Ok(BackendOutput {
                float_outputs: vec![vec![]],
                string_outputs: vec![],
                row_status: vec![],
                row_errors: vec![],
                shapes: vec![(0, self.dimensions.hidden_size)],
            });
        }

        // Track per-row status for null/empty text handling
        let mut row_status = vec![true; num_rows];
        let mut row_errors = vec![String::new(); num_rows];

        // Filter out empty texts, track which rows are valid
        let mut valid_indices = Vec::new();
        let mut valid_texts = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                row_status[i] = false;
                row_errors[i] = "Empty or null text input".into();
            } else {
                valid_indices.push(i);
                valid_texts.push(text.as_str());
            }
        }

        // Initialize output with zeros (failed rows stay zero, then get nulled by adapter)
        let hidden_size = self.dimensions.hidden_size;
        let mut all_embeddings = vec![0.0_f32; num_rows * hidden_size];

        if !valid_texts.is_empty() {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                JammiError::Inference("No tokenizer loaded for embedding model".into())
            })?;
            let encoding = tokenizer.encode_batch(
                &valid_texts,
                Some(self.text_forward()?.max_sequence_length()),
            )?;

            let input_ids = self.tokens_to_tensor(&encoding.input_ids)?;
            let attention_mask = self.tokens_to_tensor(&encoding.attention_masks)?;

            // Each encoder controls its own pooling: BERT-family pools with
            // the strategy the model declares in `1_Pooling/config.json`
            // (mean fallback when the file is absent), OpenCLIP text returns
            // its pre-pooled projected output. The result is already
            // L2-normalized.
            let normalized = self.text_forward()?.forward_pooled(
                &input_ids,
                &attention_mask,
                &encoding,
                &self.device,
            )?;

            // Apply the trained projection head if one was loaded.
            let final_output = if let Some(ref head) = self.projection_head {
                head.forward(&normalized)
                    .map_err(|e| JammiError::Inference(format!("Projection head: {e}")))?
            } else {
                normalized
            };

            let final_output_f32 = if final_output.dtype() == DType::F32 {
                final_output
            } else {
                final_output
                    .to_dtype(DType::F32)
                    .map_err(|e| JammiError::Inference(format!("Embedding dtype cast: {e}")))?
            };
            let embeddings = final_output_f32
                .to_vec2::<f32>()
                .map_err(|e| JammiError::Inference(format!("Tensor to vec failed: {e}")))?;

            // Place valid embeddings into the correct positions
            for (emb_idx, &orig_idx) in valid_indices.iter().enumerate() {
                let start = orig_idx * hidden_size;
                all_embeddings[start..start + hidden_size].copy_from_slice(&embeddings[emb_idx]);
            }
        }

        Ok(BackendOutput {
            float_outputs: vec![all_embeddings],
            string_outputs: vec![],
            row_status,
            row_errors,
            shapes: vec![(num_rows, hidden_size)],
        })
    }

    fn forward_image_embedding(&self, content: &[arrow::array::ArrayRef]) -> Result<BackendOutput> {
        let vision = self.vision.as_deref().ok_or_else(|| {
            JammiError::Inference("No vision model loaded for image embedding".into())
        })?;

        let images = arrow_to_images(content)?;
        let num_rows = images.len();

        if num_rows == 0 {
            return Ok(BackendOutput {
                float_outputs: vec![vec![]],
                string_outputs: vec![],
                row_status: vec![],
                row_errors: vec![],
                shapes: vec![(0, self.dimensions.hidden_size)],
            });
        }

        let mut row_status = vec![true; num_rows];
        let mut row_errors = vec![String::new(); num_rows];
        let mut valid_indices = Vec::new();
        let mut valid_images = Vec::new();

        for (i, img) in images.iter().enumerate() {
            match img {
                Some(im) => {
                    valid_indices.push(i);
                    valid_images.push(im.clone());
                }
                None => {
                    row_status[i] = false;
                    row_errors[i] = "Null or missing image input".into();
                }
            }
        }

        let hidden_size = self.dimensions.hidden_size;
        let mut all_embeddings = vec![0.0_f32; num_rows * hidden_size];

        if !valid_images.is_empty() {
            let target_size = vision.image_size() as u32;
            let mean = vision.preprocess_mean();
            let std = vision.preprocess_std();
            let pixel_values = image_preprocess::preprocess_image_batch(
                &valid_images,
                target_size,
                &mean,
                &std,
                &self.device,
            )?;

            let output = vision.forward_image(&pixel_values)?;

            let normalized = self.l2_normalize(&output)?;

            let normalized_f32 = if normalized.dtype() == DType::F32 {
                normalized
            } else {
                normalized.to_dtype(DType::F32).map_err(|e| {
                    JammiError::Inference(format!("Image embedding dtype cast: {e}"))
                })?
            };
            let embeddings = normalized_f32
                .to_vec2::<f32>()
                .map_err(|e| JammiError::Inference(format!("Tensor to vec failed: {e}")))?;

            for (emb_idx, &orig_idx) in valid_indices.iter().enumerate() {
                let start = orig_idx * hidden_size;
                all_embeddings[start..start + hidden_size].copy_from_slice(&embeddings[emb_idx]);
            }
        }

        Ok(BackendOutput {
            float_outputs: vec![all_embeddings],
            string_outputs: vec![],
            row_status,
            row_errors,
            shapes: vec![(num_rows, hidden_size)],
        })
    }

    fn forward_audio_embedding(&self, content: &[arrow::array::ArrayRef]) -> Result<BackendOutput> {
        let audio = self.audio.as_deref().ok_or_else(|| {
            JammiError::Inference("No audio model loaded for audio embedding".into())
        })?;
        let frontend = self.audio_frontend.as_ref().ok_or_else(|| {
            JammiError::Inference("No audio feature-extractor config loaded".into())
        })?;

        let clips = arrow_to_audio(content)?;
        let num_rows = clips.len();

        if num_rows == 0 {
            return Ok(BackendOutput {
                float_outputs: vec![vec![]],
                string_outputs: vec![],
                row_status: vec![],
                row_errors: vec![],
                shapes: vec![(0, self.dimensions.hidden_size)],
            });
        }

        let mut row_status = vec![true; num_rows];
        let mut row_errors = vec![String::new(); num_rows];
        let mut valid_indices = Vec::new();
        let mut valid_clips = Vec::new();

        for (i, clip) in clips.into_iter().enumerate() {
            match clip {
                Some(c) => {
                    valid_indices.push(i);
                    valid_clips.push(c);
                }
                None => {
                    row_status[i] = false;
                    row_errors[i] = "Null or missing audio input".into();
                }
            }
        }

        let hidden_size = self.dimensions.hidden_size;
        let mut all_embeddings = vec![0.0_f32; num_rows * hidden_size];

        if !valid_clips.is_empty() {
            // The front-end's mel-filter count must match the tower's input
            // contract; a mismatch is a misconfigured preprocessor_config.json.
            if frontend.n_mels != audio.num_mel_bins() {
                return Err(JammiError::Inference(format!(
                    "Audio feature-extractor feature_size ({}) does not match the tower's \
                     num_mel_bins ({})",
                    frontend.n_mels,
                    audio.num_mel_bins()
                )));
            }

            // Decode → resample → CLAP fusion front-end → [B, 4, time, n_mels]
            // plus the `is_longer` flags. The front-end emits all-true
            // (deterministic always-fusion) so every clip runs the AFF path,
            // reproducing HF's canonical get_audio_features embedding; the tower
            // gates fusion per sample, so it still honors a false flag if passed.
            let (input_features, is_longer) =
                audio_preprocess::preprocess_clap_fusion(&valid_clips, frontend, &self.device)?;

            // The CLAP audio tower emits L2-normalized embeddings directly
            // (like the text tower), so no further normalization is applied —
            // unlike the vision tower whose raw output is normalized here.
            let normalized = audio.forward_audio(&input_features, &is_longer)?;

            // Apply the trained projection head if one was loaded. The head is
            // a post-pool transform on the shared-latent embedding, so an audio
            // fine-tune trained as a projection head shifts audio embeddings
            // exactly as a text fine-tune shifts text embeddings.
            let projected = if let Some(ref head) = self.projection_head {
                head.forward(&normalized)
                    .map_err(|e| JammiError::Inference(format!("Projection head: {e}")))?
            } else {
                normalized
            };

            let normalized_f32 = if projected.dtype() == DType::F32 {
                projected
            } else {
                projected.to_dtype(DType::F32).map_err(|e| {
                    JammiError::Inference(format!("Audio embedding dtype cast: {e}"))
                })?
            };
            let embeddings = normalized_f32
                .to_vec2::<f32>()
                .map_err(|e| JammiError::Inference(format!("Tensor to vec failed: {e}")))?;

            for (emb_idx, &orig_idx) in valid_indices.iter().enumerate() {
                let start = orig_idx * hidden_size;
                all_embeddings[start..start + hidden_size].copy_from_slice(&embeddings[emb_idx]);
            }
        }

        Ok(BackendOutput {
            float_outputs: vec![all_embeddings],
            string_outputs: vec![],
            row_status,
            row_errors,
            shapes: vec![(num_rows, hidden_size)],
        })
    }

    fn forward_classification(&self, content: &[arrow::array::ArrayRef]) -> Result<BackendOutput> {
        let id2label = self.id2label.as_ref().ok_or_else(|| {
            JammiError::Inference("No id2label mapping for classification model".into())
        })?;

        let texts = arrow_to_texts(content)?;
        let num_rows = texts.len();

        if num_rows == 0 {
            return Ok(BackendOutput {
                float_outputs: vec![vec![]],
                string_outputs: vec![vec![], vec![]],
                row_status: vec![],
                row_errors: vec![],
                shapes: vec![(0, 0)],
            });
        }

        let mut row_status = vec![true; num_rows];
        let mut row_errors = vec![String::new(); num_rows];
        let mut valid_indices = Vec::new();
        let mut valid_texts = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                row_status[i] = false;
                row_errors[i] = "Empty or null text input".into();
            } else {
                valid_indices.push(i);
                valid_texts.push(text.as_str());
            }
        }

        // Initialize outputs for all rows (failed rows stay empty/zero)
        let mut all_confidences = vec![0.0_f32; num_rows];
        let mut all_labels = vec![String::new(); num_rows];
        let mut all_scores_json = vec![String::new(); num_rows];

        if !valid_texts.is_empty() {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                JammiError::Inference("No tokenizer loaded for classification model".into())
            })?;
            let encoding = tokenizer.encode_batch(
                &valid_texts,
                Some(self.text_forward()?.max_sequence_length()),
            )?;

            let input_ids = self.tokens_to_tensor(&encoding.input_ids)?;
            let attention_mask = self.tokens_to_tensor(&encoding.attention_masks)?;

            // Forward pass returns (batch, num_classes) with softmax applied
            let logits = self.text_forward()?.forward_hidden(
                &input_ids,
                &attention_mask,
                &encoding,
                &self.device,
            )?;

            let logits_f32 = if logits.dtype() == DType::F32 {
                logits
            } else {
                logits
                    .to_dtype(DType::F32)
                    .map_err(|e| JammiError::Inference(format!("Logits dtype cast: {e}")))?
            };
            let probs = logits_f32
                .to_vec2::<f32>()
                .map_err(|e| JammiError::Inference(format!("Logits to vec failed: {e}")))?;

            for (batch_idx, &orig_idx) in valid_indices.iter().enumerate() {
                let row_probs = &probs[batch_idx];

                // Argmax → label, max → confidence
                let (max_idx, &max_val) = row_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));

                let label = id2label
                    .get(&(max_idx as u32))
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{max_idx}"));

                // Build JSON of all scores
                let scores_map: serde_json::Map<String, serde_json::Value> = id2label
                    .iter()
                    .map(|(&idx, name)| {
                        let score = row_probs.get(idx as usize).copied().unwrap_or(0.0);
                        (name.clone(), serde_json::Value::from(score))
                    })
                    .collect();
                let scores_json = serde_json::Value::Object(scores_map).to_string();

                all_confidences[orig_idx] = max_val;
                all_labels[orig_idx] = label;
                all_scores_json[orig_idx] = scores_json;
            }
        }

        Ok(BackendOutput {
            float_outputs: vec![all_confidences],
            string_outputs: vec![all_labels, all_scores_json],
            row_status,
            row_errors,
            shapes: vec![(num_rows, 0)],
        })
    }

    fn forward_ner(&self, content: &[arrow::array::ArrayRef]) -> Result<BackendOutput> {
        let id2label = self
            .id2label
            .as_ref()
            .ok_or_else(|| JammiError::Inference("No id2label mapping for NER model".into()))?;
        let ner_classifier = self.ner_classifier.as_ref().ok_or_else(|| {
            JammiError::Inference("No token classifier loaded for NER model".into())
        })?;

        let texts = arrow_to_texts(content)?;
        let num_rows = texts.len();

        if num_rows == 0 {
            return Ok(BackendOutput {
                float_outputs: vec![],
                string_outputs: vec![vec![]],
                row_status: vec![],
                row_errors: vec![],
                shapes: vec![(0, 0)],
            });
        }

        let mut row_status = vec![true; num_rows];
        let mut row_errors = vec![String::new(); num_rows];
        let mut valid_indices = Vec::new();
        let mut valid_texts = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                row_status[i] = false;
                row_errors[i] = "Empty or null text input".into();
            } else {
                valid_indices.push(i);
                valid_texts.push(text.as_str());
            }
        }

        let mut all_entities_json = vec![String::new(); num_rows];

        if !valid_texts.is_empty() {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| JammiError::Inference("No tokenizer loaded for NER model".into()))?;
            let encoding = tokenizer.encode_batch(
                &valid_texts,
                Some(self.text_forward()?.max_sequence_length()),
            )?;

            let input_ids = self.tokens_to_tensor(&encoding.input_ids)?;
            let attention_mask = self.tokens_to_tensor(&encoding.attention_masks)?;

            // Encoder returns (batch, seq_len, hidden)
            let hidden_states = self.text_forward()?.forward_hidden(
                &input_ids,
                &attention_mask,
                &encoding,
                &self.device,
            )?;

            // Apply token classifier: (batch, seq_len, hidden) → (batch, seq_len, num_labels)
            let logits = hidden_states.apply(ner_classifier).map_err(|e| {
                JammiError::Inference(format!("NER classifier forward failed: {e}"))
            })?;

            let logits_f32 = if logits.dtype() == DType::F32 {
                logits
            } else {
                logits
                    .to_dtype(DType::F32)
                    .map_err(|e| JammiError::Inference(format!("NER logits dtype cast: {e}")))?
            };
            let logits_vec = logits_f32
                .to_vec3::<f32>()
                .map_err(|e| JammiError::Inference(format!("NER logits to vec failed: {e}")))?;

            for (batch_idx, &orig_idx) in valid_indices.iter().enumerate() {
                let token_logits = &logits_vec[batch_idx];
                let offsets = &encoding.offsets[batch_idx];
                let mask = &encoding.attention_masks[batch_idx];

                match jammi_numerics::ner::decode_bio_spans(
                    token_logits,
                    offsets,
                    mask,
                    id2label,
                    &texts[orig_idx],
                ) {
                    Ok(entities) => {
                        all_entities_json[orig_idx] =
                            serde_json::to_string(&entities).unwrap_or_else(|_| "[]".to_string());
                    }
                    // `decode_bio_spans`' only error today is a non-finite
                    // (NaN/±inf) token logit — the model diverged or the
                    // checkpoint is corrupt, never a per-row INPUT fault (the
                    // row's text is already known-valid: empty/null text was
                    // filtered before the forward ran, above). The `_status`
                    // channel's contract (see `InferenceRunner::run_chunks` in
                    // runner.rs) reserves per-row `_status = error` for a bad
                    // INPUT, never "the model is broken" — so this is a
                    // batch-level typed failure, propagated exactly like any
                    // other systemic non-OOM forward failure, naming the
                    // offending row.
                    Err(e) => {
                        return Err(JammiError::Inference(format!(
                            "NER row {orig_idx}: non-finite logits — model diverged or \
                             checkpoint corrupt ({e})"
                        )));
                    }
                }
            }
        }

        Ok(BackendOutput {
            float_outputs: vec![],
            string_outputs: vec![all_entities_json],
            row_status,
            row_errors,
            shapes: vec![(num_rows, 0)],
        })
    }
}

impl ModelBackend for CandleBackend {
    fn load(&self, resolved: &ResolvedModel, device_config: &DeviceConfig) -> Result<LoadedModel> {
        let device = select_device(device_config)?;

        let model_type = resolved
            .model_config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("bert");

        // Per-model `compute_precision` in `config.json` wins over the global
        // `DeviceConfig` default; both default to `F32`. Read the same
        // best-effort way `id2label` is read below: a malformed/unknown value
        // is honestly "not declared", never a hard error over an optional
        // field.
        let per_model_precision: Option<jammi_numerics::ComputePrecision> = resolved
            .model_config
            .get("compute_precision")
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        let compute_precision = per_model_precision.unwrap_or(device_config.compute_precision);
        let compute_dtype = match compute_precision {
            jammi_numerics::ComputePrecision::F32 | jammi_numerics::ComputePrecision::F16 => {
                jammi_encoders::compute_precision_to_dtype(compute_precision)
            }
            jammi_numerics::ComputePrecision::BF16 => {
                // bf16 is a GPU-tier precision: its tensor-core kernels are an
                // Ampere (sm_80) innovation. Device admission (`select_device`,
                // via `check_compute_cap_floor`) already rejects any CUDA
                // device below that architecture floor before this ever runs,
                // so a `Device::Cuda(_)` reaching this match is unconditionally
                // bf16-capable — this arm's only remaining job is CUDA vs
                // non-CUDA, which is precision-vs-backend, not architecture.
                #[cfg(feature = "cuda")]
                {
                    match &device {
                        Device::Cuda(_) => {
                            jammi_encoders::compute_precision_to_dtype(compute_precision)
                        }
                        _ => {
                            return Err(JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: "bf16 inference requires a CUDA device with compute \
                                          capability >= 8.0 (Ampere+); the resolved device is \
                                          non-CUDA. Use f16 or f32."
                                    .into(),
                            });
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: "bf16 inference requires a CUDA device with compute capability \
                                  >= 8.0 (Ampere+); this build has no CUDA support. Use f16 or \
                                  f32."
                            .into(),
                    });
                }
            }
        };

        // The root `VarBuilder` loads every weight at `compute_dtype` — the
        // encoder backbone AND every head built from it (classifier,
        // projection, CLAP/OpenCLIP towers) — because a mismatched backbone ×
        // head matmul dtype errors in candle. The one exception is the
        // fine-tune adapter path below, which loads the frozen backbone at its
        // own *persisted* `backbone_dtype` (a training-time choice, independent
        // of this inference-time knob) when a saved adapter is present.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&resolved.weights_paths, compute_dtype, &device)
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to load safetensors: {e}"),
                })?
        };

        // Parse id2label from config if present (classification models)
        let id2label: Option<HashMap<u32, String>> = resolved
            .model_config
            .get("id2label")
            .and_then(|v| v.as_object())
            .map(|map| {
                map.iter()
                    .filter_map(|(k, v)| {
                        let idx: u32 = k.parse().ok()?;
                        let label = v.as_str()?.to_string();
                        Some((idx, label))
                    })
                    .collect()
            });

        let is_classification = resolved.task == ModelTask::Classification && id2label.is_some();
        let is_ner = resolved.task == ModelTask::Ner && id2label.is_some();
        // HF-CLAP audio checkpoints (`ClapAudioModelWithProjection`) declare
        // `model_type == "clap_audio_model"` at the top level (flat
        // `ClapAudioConfig`) or under a nested `audio_config` (top-level
        // `ClapConfig`), and/or list `ClapModel`/`ClapAudioModelWithProjection`
        // in `architectures`. OpenCLIP vision checkpoints carry `model_cfg`.
        // The two are disjoint, so the audio branch is checked first.
        let is_clap = is_hf_clap_config(&resolved.model_config);
        let is_open_clip = !is_clap && resolved.model_config.get("model_cfg").is_some();

        // Normalize DistilBERT config fields to standard BERT names.
        // DistilBERT uses dim/n_heads/n_layers/hidden_dim instead of
        // hidden_size/num_attention_heads/num_hidden_layers/intermediate_size.
        let model_config = if model_type == "distilbert" {
            normalize_distilbert_config(&resolved.model_config)
        } else {
            resolved.model_config.clone()
        };

        // Read the saved adapter, if any. Both flavours of `SavedAdapter`
        // share the same on-disk layout (`adapter.safetensors` plus
        // `adapter_config.json` with the `adapter_type` discriminator); the
        // variant is the type-level switch that decides whether to wire
        // LoRA inside the encoder or leave it as an external projection
        // head applied post-pool.
        let saved_adapter: Option<(crate::fine_tune::target::SavedAdapter, std::path::PathBuf)> =
            resolved.adapter_path.as_ref().and_then(|p| {
                let cfg_path = p.join("adapter_config.json");
                let weights_path = p.join("adapter.safetensors");
                if !cfg_path.exists() || !weights_path.exists() {
                    return None;
                }
                let cfg_str = std::fs::read_to_string(&cfg_path).ok()?;
                let saved: crate::fine_tune::target::SavedAdapter =
                    serde_json::from_str(&cfg_str).ok()?;
                Some((saved, weights_path))
            });

        let encoder_adapter = saved_adapter.as_ref().and_then(|(saved, weights)| {
            if let crate::fine_tune::target::SavedAdapter::EncoderAdapters(cfg) = saved {
                Some(((**cfg).clone(), weights.as_path()))
            } else {
                None
            }
        });
        let encoder_owned = encoder_adapter.as_ref().map(|(cfg, _)| {
            (
                cfg.target_modules.clone(),
                cfg.layers_to_transform.clone(),
                cfg.rank_pattern.clone(),
            )
        });
        let lora_build = match (&encoder_adapter, &encoder_owned) {
            (Some((cfg, _)), Some((targets, layers, pattern))) => jammi_lora::LoraBuildConfig {
                target_modules: targets,
                layers_to_transform: layers,
                lora_rank: cfg.lora_rank,
                lora_alpha: cfg.lora_alpha,
                use_rslora: cfg.use_rslora,
                lora_dropout: None,
                rank_pattern: pattern,
                init_mode: jammi_lora::LoraInitMode::ZerosB,
                // Inference path: the seeded init is immediately overwritten by
                // the loaded adapter weights, so the seed is never observed.
                seed: 0,
            },
            _ => jammi_lora::LoraBuildConfig::frozen(),
        };
        let encoder_adapter_file: Option<&std::path::Path> =
            encoder_adapter.as_ref().map(|(_, p)| *p);
        // A fine-tune adapter's backbone loads at its own *persisted*
        // `backbone_dtype` (a training-time choice); with no adapter, the
        // backbone follows the resolved inference `compute_precision` the root
        // `vb` above just loaded at — so an unadapted model's backbone and its
        // heads always agree on dtype. This is also the precision the
        // materialization contract folds into `ModelIdentity`, so it is
        // computed once here (as `ComputePrecision`) and carried through to
        // `CandleModel::compute_precision`, never re-derived.
        let effective_precision: jammi_numerics::ComputePrecision = encoder_adapter
            .as_ref()
            .map(|(cfg, _)| cfg.backbone_dtype)
            .unwrap_or(compute_precision);
        let encoder_backbone_dtype =
            jammi_encoders::compute_precision_to_dtype(effective_precision);
        let weights_paths_ref: Vec<&std::path::Path> =
            resolved.weights_paths.iter().map(|p| p.as_path()).collect();
        let dummy_varmap = VarMap::new();

        // The pooling strategy the text-embedding path uses for the three
        // BERT-family wrappers, selected once from the model's declared
        // `1_Pooling/config.json` (mean fallback if the file is absent). Read
        // before the model-type match so every embedding wrapper below shares
        // the identical resolved strategy.
        let pooling = pooling_from_config(resolved.pooling_config.as_ref(), &resolved.model_id.0)?;

        // Branch: cross-modal model selection.
        //   - HF-CLAP (`clap_audio_model`): a single HTSAT-Swin audio tower
        //     producing shared-latent embeddings; routed in `forward()` by
        //     `ModelTask::AudioEmbedding`.
        //   - OpenCLIP (`model_cfg.vision_cfg`): both vision and text towers in
        //     one checkpoint, routed by `ModelTask::{Image,Text}Embedding`.
        //   - otherwise: text-only (BERT family).
        #[allow(clippy::type_complexity)]
        let (text, vision, audio): (
            Option<Box<dyn CandleTextForward>>,
            Option<Box<dyn CandleVisionForward>>,
            Option<Box<dyn CandleAudioForward>>,
        ) = if is_clap {
            let audio_config = HtsatAudioConfig::from_hf_clap_config(&resolved.model_config)
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to parse CLAP audio config: {e}"),
                })?;
            // HF-CLAP safetensors keys are rooted at `audio_model.audio_encoder.*`
            // and `audio_projection.*`, so the tower loads from the root VarBuilder.
            let audio_inner =
                HtsatAudio::load(vb.clone(), &audio_config, &device).map_err(|e| {
                    JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to construct HTSAT-Swin CLAP audio tower: {e}"),
                    }
                })?;
            (
                None,
                None,
                Some(Box::new(audio_inner) as Box<dyn CandleAudioForward>),
            )
        } else if is_open_clip {
            let vision_config = OpenClipVisionConfig::from_open_clip_config(&resolved.model_config)
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to parse OpenCLIP vision config: {e}"),
                })?;
            let vision_inner = OpenClipVisionTransformer::load(vb.pp("visual"), &vision_config)
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to construct OpenCLIP ViT: {e}"),
                })?;

            let text_config = ClipTextConfig::from_open_clip_config(&resolved.model_config)
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to parse OpenCLIP text config: {e}"),
                })?;
            let text_inner =
                ClipText::load(vb.clone(), &text_config).map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to construct OpenCLIP text tower: {e}"),
                })?;

            (
                Some(Box::new(OpenClipTextForward(text_inner)) as Box<dyn CandleTextForward>),
                Some(Box::new(vision_inner) as Box<dyn CandleVisionForward>),
                None,
            )
        } else {
            let text_inner: Box<dyn CandleTextForward> = match model_type {
                "distilbert" if is_classification => {
                    let db_config: DistilBertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse DistilBERT config: {e}"),
                        })?;
                    let distilbert = DistilBert::builder()
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &db_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct DistilBERT model: {e}"),
                        })?;
                    let num_classes = id2label.as_ref().map_or(2, |m| m.len());
                    let hidden_size = db_config.hidden_size;
                    let pre_classifier =
                        candle_nn::linear(hidden_size, hidden_size, vb.pp("pre_classifier"))
                            .map_err(|e| JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to load DistilBERT pre_classifier: {e}"),
                            })?;
                    let classifier =
                        candle_nn::linear(hidden_size, num_classes, vb.pp("classifier")).map_err(
                            |e| JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to load DistilBERT classifier head: {e}"),
                            },
                        )?;
                    Box::new(DistilBertClassificationForward {
                        distilbert,
                        pre_classifier,
                        classifier,
                    })
                }
                "distilbert" => {
                    let db_config: DistilBertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse DistilBERT config: {e}"),
                        })?;
                    let model = DistilBert::builder()
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &db_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct DistilBERT model: {e}"),
                        })?;
                    Box::new(DistilBertForward { model, pooling })
                }
                "bert" | "roberta" | "camembert" | "xlm-roberta" if is_classification => {
                    let bert_config: BertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse BERT config: {e}"),
                        })?;
                    let bert = Bert::builder()
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &bert_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct BERT model: {e}"),
                        })?;
                    let num_classes = id2label.as_ref().map_or(2, |m| m.len());
                    let hidden_size = bert_config.hidden_size;
                    let classifier =
                        candle_nn::linear(hidden_size, num_classes, vb.pp("classifier")).map_err(
                            |e| JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to load BERT classifier head: {e}"),
                            },
                        )?;
                    Box::new(BertClassificationForward { bert, classifier })
                }
                "bert" | "roberta" | "camembert" | "xlm-roberta" => {
                    let bert_config: BertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse BERT config: {e}"),
                        })?;
                    let bert = Bert::builder()
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &bert_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct BERT model: {e}"),
                        })?;
                    Box::new(BertForward {
                        model: bert,
                        pooling,
                    })
                }
                "modernbert" if is_classification => {
                    let mb_config: ModernBertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse ModernBERT config: {e}"),
                        })?;
                    let backbone = ModernBert::builder()
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &mb_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct ModernBERT model: {e}"),
                        })?;
                    let num_classes = id2label.as_ref().map_or(2, |m| m.len());
                    let classifier = SeqClassifier::new(backbone, num_classes, vb.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to load ModernBERT classifier head: {e}"),
                        })?;
                    Box::new(ModernBertClassificationForward(classifier))
                }
                "modernbert" => {
                    let mb_config: ModernBertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse ModernBERT config: {e}"),
                        })?;
                    let model = ModernBert::builder()
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &mb_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct ModernBERT model: {e}"),
                        })?;
                    Box::new(ModernBertForward { model, pooling })
                }
                unsupported => {
                    return Err(JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!(
                            "Unsupported model architecture '{unsupported}'. Supported: \
                                 bert, roberta, distilbert, camembert, xlm-roberta, modernbert"
                        ),
                    });
                }
            };
            (Some(text_inner), None, None)
        };

        let tokenizer = resolved
            .tokenizer
            .as_ref()
            .map(|src| match src {
                TokenizerSource::HuggingFaceJson(p) => TokenizerWrapper::from_file(p),
                TokenizerSource::OpenClipBpe(p) => TokenizerWrapper::from_open_clip_bpe(p),
            })
            .transpose()?;

        let dimensions =
            ModelDimensions::from_config(&model_config).ok_or_else(|| JammiError::Model {
                model_id: resolved.model_id.0.clone(),
                message: "Could not parse model dimensions from config".into(),
            })?;

        // Load the post-pool projection head, if the saved adapter is one.
        // Encoder-adapters are installed inside `text` above via the encoder
        // builder's `.lora(...)` + `.adapter(Some(...))` calls.
        let (projection_head, distribution_head, regression_scaler, regression_form) =
            match saved_adapter.as_ref() {
                Some((
                    crate::fine_tune::target::SavedAdapter::ProjectionHead(cfg),
                    weights_path,
                )) => {
                    (
                        load_projection_head(
                            weights_path,
                            cfg.lora_alpha,
                            &device,
                            &dimensions,
                            &resolved.model_id.0,
                        )?,
                        // The `distribution` layer is present only for a
                        // regression head; `load_distribution_head` returns `None`
                        // for embedding/classification/NER projection heads.
                        load_distribution_head(
                            weights_path,
                            cfg.lora_alpha,
                            &device,
                            &dimensions,
                            &resolved.model_id.0,
                        )?,
                        cfg.target_scaler,
                        cfg.regression_form.clone(),
                    )
                }
                _ => (None, None, None, None),
            };

        // Load NER token classifier if this is a NER model
        let ner_classifier = if is_ner {
            let num_labels = id2label.as_ref().map_or(3, |m| m.len());
            let hidden_size = dimensions.hidden_size;
            // NER models use a VarBuilder scoped to the same safetensors, at
            // the same dtype as the backbone whose hidden states this
            // classifier applies to (`encoder_backbone_dtype`) — otherwise a
            // non-F32 backbone × F32 classifier matmul errors.
            let ner_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &resolved.weights_paths,
                    encoder_backbone_dtype,
                    &device,
                )
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to reload safetensors for NER classifier: {e}"),
                })?
            };
            Some(
                candle_nn::linear(hidden_size, num_labels, ner_vb.pp("classifier")).map_err(
                    |e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to load NER classifier head: {e}"),
                    },
                )?,
            )
        } else {
            None
        };

        // Audio models need the CLAP fusion front-end geometry from
        // `preprocessor_config.json`; an audio tower without it is unusable.
        let audio_frontend = if audio.is_some() {
            let prep = resolved
                .preprocessor_config
                .as_ref()
                .ok_or_else(|| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: "CLAP audio model is missing preprocessor_config.json \
                              (the feature-extractor geometry the front-end is driven by)"
                        .into(),
                })?;
            Some(
                clap_frontend_from_preprocessor(prep).map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Invalid CLAP preprocessor_config.json: {e}"),
                })?,
            )
        } else {
            None
        };

        Ok(LoadedModel::Candle(Box::new(CandleModel {
            dimensions,
            text,
            vision,
            audio,
            audio_frontend,
            tokenizer,
            device,
            projection_head,
            distribution_head,
            regression_scaler,
            regression_form,
            id2label,
            ner_classifier,
            compute_precision: effective_precision,
        })))
    }

    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize {
        resolved
            .weights_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum()
    }
}

/// Detect an HF-CLAP audio checkpoint (`ClapAudioModelWithProjection` lineage)
/// from its config: `model_type == "clap_audio_model"` at the top level (flat
/// `ClapAudioConfig`) or under a nested `audio_config` (top-level `ClapConfig`),
/// or `architectures` listing `ClapModel` / `ClapAudioModelWithProjection`.
fn is_hf_clap_config(config: &serde_json::Value) -> bool {
    let model_type_is_clap = |v: &serde_json::Value| {
        v.get("model_type").and_then(|m| m.as_str()) == Some("clap_audio_model")
    };
    if model_type_is_clap(config) {
        return true;
    }
    if config.get("audio_config").is_some_and(model_type_is_clap) {
        return true;
    }
    config
        .get("architectures")
        .and_then(|a| a.as_array())
        .is_some_and(|arch| {
            arch.iter().any(|a| {
                matches!(
                    a.as_str(),
                    Some("ClapModel") | Some("ClapAudioModelWithProjection")
                )
            })
        })
}

/// Build the CLAP fusion front-end geometry from a HuggingFace
/// `preprocessor_config.json` (`ClapFeatureExtractor` arguments). Every numeric
/// the bytes-to-spectrogram transform needs is read from the config — nothing
/// is hardcoded.
fn clap_frontend_from_preprocessor(
    prep: &serde_json::Value,
) -> Result<audio_preprocess::ClapFrontendConfig> {
    let u = |key: &str| -> Result<u64> {
        prep.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| JammiError::Inference(format!("missing integer field '{key}'")))
    };
    let f = |key: &str| -> Result<f64> {
        prep.get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| JammiError::Inference(format!("missing numeric field '{key}'")))
    };
    Ok(audio_preprocess::ClapFrontendConfig {
        n_mels: u("feature_size")? as usize,
        sample_rate: u("sampling_rate")? as u32,
        fft_window_size: u("fft_window_size")? as usize,
        hop_length: u("hop_length")? as usize,
        frequency_min: f("frequency_min")?,
        frequency_max: f("frequency_max")?,
        max_length_s: u("max_length_s")? as u32,
    })
}

/// Convert token ID vectors into a candle Tensor on the given device.
fn tokens_to_tensor(vecs: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let rows = vecs.len();
    let cols = vecs.first().map_or(0, |v| v.len());
    let flat: Vec<u32> = vecs.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (rows, cols), device).map_err(|e| JammiError::Inference(e.to_string()))
}

/// Load the projection head from `adapter_file` using the alpha recorded in
/// the adapter's saved config. Returns `Some(LoraLinear)` keyed at
/// `projection.lora_a` / `projection.lora_b`, or `None` if the projection
/// keys are absent (the adapter was a classifier/NER head with no embedding
/// projection — that case does not produce a post-pool projection).
fn load_projection_head(
    adapter_file: &std::path::Path,
    lora_alpha: f64,
    device: &Device,
    dimensions: &crate::model::ModelDimensions,
    model_id: &str,
) -> jammi_db::error::Result<Option<jammi_lora::LoraLinear>> {
    let adapter_weights =
        candle_core::safetensors::load(adapter_file, device).map_err(|e| JammiError::Model {
            model_id: model_id.to_string(),
            message: format!("Load adapter: {e}"),
        })?;

    let hidden_size = dimensions.hidden_size;
    let identity = Tensor::eye(hidden_size, DType::F32, device).map_err(|e| JammiError::Model {
        model_id: model_id.to_string(),
        message: format!("Identity weight: {e}"),
    })?;
    let base_linear = candle_nn::Linear::new(identity, None);

    let lora_a = match adapter_weights.get("projection.lora_a") {
        Some(t) => t.clone(),
        None => return Ok(None),
    };
    let lora_b = match adapter_weights.get("projection.lora_b") {
        Some(t) => t.clone(),
        None => return Ok(None),
    };

    Ok(Some(jammi_lora::LoraLinear::from_loaded(
        base_linear,
        lora_a,
        lora_b,
        lora_alpha,
    )))
}

/// Reload the regression `distribution` head layer — the `hidden → output_dim`
/// LoRA layer that maps the pooled encoder embedding to the raw distribution
/// parameters (`(mean, raw_std)` for Gaussian, one per level for quantile). This
/// is the layer the trainer's `regress` applies to the pooled embedding (it uses
/// `head.layers[1]`, the distribution layer — NOT the `projection` layer), so
/// serving must apply the same one to reproduce the trained output shape.
///
/// Its zeros base spans `output_dim → hidden_size`; `output_dim` is recovered
/// from the persisted `distribution.lora_b` row count (B is `output_dim × rank`),
/// so the served head width matches the trained head without re-deriving it from
/// the form. Returns `None` when the adapter carries no `distribution` layer
/// (i.e. it is not a regression head).
fn load_distribution_head(
    adapter_file: &std::path::Path,
    lora_alpha: f64,
    device: &Device,
    dimensions: &crate::model::ModelDimensions,
    model_id: &str,
) -> jammi_db::error::Result<Option<jammi_lora::LoraLinear>> {
    let adapter_weights =
        candle_core::safetensors::load(adapter_file, device).map_err(|e| JammiError::Model {
            model_id: model_id.to_string(),
            message: format!("Load adapter: {e}"),
        })?;

    let lora_a = match adapter_weights.get("distribution.lora_a") {
        Some(t) => t.clone(),
        None => return Ok(None),
    };
    let lora_b = match adapter_weights.get("distribution.lora_b") {
        Some(t) => t.clone(),
        None => return Ok(None),
    };

    // `output_dim` = rows of B (`output_dim × rank`). The base is the trained
    // head's `zeros(output_dim, hidden_size)`; the learned signal lives entirely
    // in the LoRA A/B factors, exactly as at train time.
    let output_dim = lora_b
        .dims2()
        .map_err(|e| JammiError::Model {
            model_id: model_id.to_string(),
            message: format!("distribution.lora_b shape: {e}"),
        })?
        .0;
    let hidden_size = dimensions.hidden_size;
    let base = Tensor::zeros((output_dim, hidden_size), DType::F32, device).map_err(|e| {
        JammiError::Model {
            model_id: model_id.to_string(),
            message: format!("distribution head base: {e}"),
        }
    })?;
    let base_linear = candle_nn::Linear::new(base, None);

    Ok(Some(jammi_lora::LoraLinear::from_loaded(
        base_linear,
        lora_a,
        lora_b,
        lora_alpha,
    )))
}

/// Normalize DistilBERT config fields to standard BERT names.
///
/// DistilBERT uses different field names and omits some fields that
/// candle's `BertConfig` requires. This maps them to BERT equivalents.
fn normalize_distilbert_config(config: &serde_json::Value) -> serde_json::Value {
    let mut normalized = config.clone();
    if let Some(obj) = normalized.as_object_mut() {
        // Field renames: DistilBERT → BERT
        let mappings: &[(&str, &str)] = &[
            ("dim", "hidden_size"),
            ("n_heads", "num_attention_heads"),
            ("n_layers", "num_hidden_layers"),
            ("hidden_dim", "intermediate_size"),
            ("dropout", "hidden_dropout_prob"),
            ("attention_dropout", "attention_probs_dropout_prob"),
        ];
        for &(src, dst) in mappings {
            if let Some(val) = obj.get(src).cloned() {
                obj.entry(dst).or_insert(val);
            }
        }
        // activation → hidden_act (string value)
        if let Some(val) = obj.get("activation").cloned() {
            obj.entry("hidden_act").or_insert(val);
        }
        // Defaults for fields DistilBERT doesn't have but BertConfig requires
        obj.entry("type_vocab_size")
            .or_insert(serde_json::Value::from(2));
        obj.entry("layer_norm_eps")
            .or_insert(serde_json::json!(1e-12));
    }
    normalized
}

/// Resolve the inference device from configuration.
///
/// - A negative `gpu_device` selects CPU unconditionally.
/// - When a GPU is requested and present (the build's accelerator feature is
///   compiled in and the device initializes), the accelerator device is used.
///   A CUDA device additionally clears two floors before admission — the
///   driver ([`check_driver_floor`]) and the architecture
///   ([`check_compute_cap_floor`]) — so every precision this build supports
///   on CUDA is admissible on any device this function returns.
/// - When a GPU is requested but unavailable — no accelerator feature in this
///   build, or the device fails to initialize — the outcome depends on
///   `require_gpu`: by default it degrades to CPU with a loud warning; if
///   `require_gpu` is set it returns a [`JammiError::Gpu`] so the server fails
///   fast rather than silently serving on CPU.
///
/// This relies on `Device::new_cuda` / `Device::new_metal` returning `Err` when
/// no device is present; it never papers over a panic.
pub(crate) fn select_device(config: &DeviceConfig) -> Result<Device> {
    if config.gpu_device < 0 {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(config.gpu_device as usize) {
            // Fail fast on a driver too old to JIT this build's PTX, rather than
            // letting the first model load surface a raw
            // `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` from deep in candle.
            check_driver_floor(cuda_driver_cuda_version()?)?;
            // Fail fast on an architecture this build's PTX cannot run on at
            // all — independent of the driver-JIT floor above. Device
            // admission owns this: once a device clears it, every precision
            // this build supports on CUDA is admissible on that device.
            let (major, minor) = cuda_compute_capability(&dev)?;
            check_compute_cap_floor(major, minor)?;
            return Ok(dev);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(config.gpu_device as usize) {
            return Ok(dev);
        }
    }
    gpu_unavailable(config.gpu_device, config.require_gpu)
}

/// Minimum CUDA version the deployment driver must support, in
/// `cuDriverGetVersion` integer form (`major * 1000 + minor * 10`).
///
/// This build's CUDA kernels ship as **PTX only** (candle emits `build_ptx()`,
/// no native SASS), so the deployment driver JIT-compiles them at model load.
/// The PTX is produced by the pinned **CUDA 12.6** toolkit (PTX ISA 8.5), and a
/// driver can only JIT PTX whose ISA is ≤ its own CUDA version — so the driver
/// must itself support CUDA 12.6 (`12060`) or newer. On Linux that is NVIDIA
/// driver **r560+** (≥ 560.28.03). An older driver rejects the PTX at load with
/// `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` / `CUDA_ERROR_INVALID_PTX` — a cause
/// independent of an arch mismatch. Keep this in lockstep with the toolkit
/// version in `.docker/ci-cuda.Dockerfile` and the driver floor documented in
/// `docs/guide/src/deploy-server.md`.
#[cfg(any(feature = "cuda", test))]
const MIN_DRIVER_CUDA_VERSION: i32 = 12_060;

/// Reject a driver whose supported CUDA version is below [`MIN_DRIVER_CUDA_VERSION`].
///
/// `reported` is `cuDriverGetVersion`'s integer form. Pure so the boundary is
/// unit-tested without a GPU; the CUDA-only [`cuda_driver_cuda_version`] feeds it
/// the live value.
#[cfg(any(feature = "cuda", test))]
fn check_driver_floor(reported: i32) -> Result<()> {
    if reported < MIN_DRIVER_CUDA_VERSION {
        return Err(JammiError::Gpu(format!(
            "NVIDIA driver too old to run this GPU build: the driver supports up to CUDA \
             {rep_major}.{rep_minor}, but the build ships CUDA {min_major}.{min_minor} PTX that \
             the driver must JIT-compile at model load. Upgrade to driver r560 or later \
             (≥ 560.28.03 on Linux); otherwise model load fails with \
             CUDA_ERROR_UNSUPPORTED_PTX_VERSION. (cuDriverGetVersion reported {reported}.)",
            rep_major = reported / 1000,
            rep_minor = (reported % 1000) / 10,
            min_major = MIN_DRIVER_CUDA_VERSION / 1000,
            min_minor = (MIN_DRIVER_CUDA_VERSION % 1000) / 10,
        )));
    }
    Ok(())
}

/// The CUDA version the installed driver supports (`cuDriverGetVersion`), in
/// `major * 1000 + minor * 10` form. Callable once a CUDA device has
/// initialized — the driver is loaded by then.
#[cfg(feature = "cuda")]
fn cuda_driver_cuda_version() -> Result<i32> {
    use candle_core::cuda::cudarc::driver::sys;
    let mut version: core::ffi::c_int = 0;
    // SAFETY: `cuDriverGetVersion` writes a single `int` through the pointer and
    // reads nothing else; the driver is present because `Device::new_cuda` just
    // succeeded. cudarc is linked (`dynamic-linking`), so the symbol resolves.
    let status = unsafe { sys::cuDriverGetVersion(&mut version) };
    if status != sys::CUresult::CUDA_SUCCESS {
        return Err(JammiError::Gpu(format!(
            "cuDriverGetVersion failed ({status:?}); cannot verify the NVIDIA driver meets the \
             CUDA {}.{} PTX floor this build requires",
            MIN_DRIVER_CUDA_VERSION / 1000,
            (MIN_DRIVER_CUDA_VERSION % 1000) / 10,
        )));
    }
    Ok(version)
}

/// Minimum CUDA compute capability (architecture) this build's PTX targets,
/// as `(major, minor)`.
///
/// candle compiles its CUDA kernels to single-architecture PTX at
/// `compute_80` (`CUDA_COMPUTE_CAP` in `.docker/ci-cuda.Dockerfile`): sm_80
/// (Ampere) is the floor because that PTX JIT-forward-runs on every
/// supported datacenter GPU — A100 (8.0), A10/A6000 (8.6), L40S (8.9), H100
/// (9.0) — while Turing (sm_75, e.g. T4) and older cannot load this build's
/// PTX at all. This is an architecture floor, independent of
/// [`MIN_DRIVER_CUDA_VERSION`] above (a stale-driver failure): a device below
/// this floor cannot run the build's kernels no matter how new its driver
/// is. Keep this in lockstep with `CUDA_COMPUTE_CAP` in
/// `.docker/ci-cuda.Dockerfile`.
#[cfg(any(feature = "cuda", test))]
const MIN_CUDA_COMPUTE_CAP: (i32, i32) = (8, 0);

/// Reject a device whose compute capability is below [`MIN_CUDA_COMPUTE_CAP`].
///
/// Pure so the boundary is unit-tested without a GPU; the CUDA-only
/// [`cuda_compute_capability`] feeds it the live value.
#[cfg(any(feature = "cuda", test))]
fn check_compute_cap_floor(major: i32, minor: i32) -> Result<()> {
    if (major, minor) < MIN_CUDA_COMPUTE_CAP {
        return Err(JammiError::Gpu(format!(
            "GPU architecture too old to run this GPU build: device is sm_{major}{minor}, but \
             this build's CUDA kernels are compiled for sm_80+ (Ampere or newer — e.g. A100, \
             A10/A6000, L40S, H100). Turing (sm_75, e.g. T4) and older GPUs are unsupported; no \
             driver upgrade can fix an architecture mismatch."
        )));
    }
    Ok(())
}

/// The compute capability (`major`, `minor`) of an initialized CUDA device.
#[cfg(feature = "cuda")]
fn cuda_compute_capability(dev: &Device) -> Result<(i32, i32)> {
    let cuda = dev.as_cuda_device().map_err(|e| {
        JammiError::Gpu(format!(
            "expected a CUDA device to query compute capability: {e}"
        ))
    })?;
    cuda.cuda_stream()
        .context()
        .compute_capability()
        .map_err(|e| JammiError::Gpu(format!("could not query CUDA compute capability: {e}")))
}

/// The [`ComputeDevice`] the engine *effectively* runs on for `config` — the
/// device-identity the materialization contract folds into its definition hash,
/// so a CPU run and a CUDA run of the same model hash differently. Resolved
/// through the same [`select_device`] logic the loader uses (including the
/// CPU-fallback when a requested GPU is unavailable), so the recorded device is
/// the one that actually produced the floats, never merely the one requested.
pub(crate) fn effective_compute_device(config: &DeviceConfig) -> ComputeDevice {
    match select_device(config) {
        Ok(Device::Cpu) => ComputeDevice::Cpu,
        #[cfg(feature = "cuda")]
        Ok(Device::Cuda(_)) => ComputeDevice::Cuda {
            ordinal: config.gpu_device.max(0) as u32,
        },
        #[cfg(feature = "metal")]
        Ok(Device::Metal(_)) => ComputeDevice::Metal {
            ordinal: config.gpu_device.max(0) as u32,
        },
        // An accelerator variant compiled out, or a `require_gpu` error: the
        // engine runs on CPU in every such case (a hard `require_gpu` failure
        // surfaces at load, before any materialization reaches here).
        _ => ComputeDevice::Cpu,
    }
}

/// Decide what to do when a GPU was requested but could not be acquired.
fn gpu_unavailable(gpu_device: i32, require_gpu: bool) -> Result<Device> {
    if require_gpu {
        return Err(JammiError::Gpu(format!(
            "GPU required (gpu.device={gpu_device}, require_gpu=true) but no usable GPU was found"
        )));
    }
    // `cfg!` is a compile-time constant here, so the message distinguishes two
    // genuinely different situations that the old text conflated: a GPU build
    // whose driver/runtime wasn't reachable, versus a CPU-only build that has no
    // accelerator backend compiled in at all (where the request could never
    // succeed and no loader-path fix applies).
    if cfg!(feature = "cuda") {
        tracing::warn!(
            gpu_device,
            "GPU device {gpu_device} requested but no CUDA device could be initialized; \
             running on CPU. Check the NVIDIA driver and that the CUDA runtime libraries are \
             on the dynamic loader path (LD_LIBRARY_PATH / ldconfig)."
        );
    } else if cfg!(feature = "metal") {
        tracing::warn!(
            gpu_device,
            "GPU device {gpu_device} requested but no Metal device could be initialized; \
             running on CPU."
        );
    } else {
        tracing::warn!(
            gpu_device,
            "GPU device {gpu_device} requested but this build has no GPU support compiled in \
             (CPU-only build); running on CPU. Use the CUDA-enabled server build for GPU \
             inference, or set gpu.device=-1 to select CPU explicitly and silence this warning."
        );
    }
    Ok(Device::Cpu)
}

#[cfg(test)]
mod device_tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use tracing::field::{Field, Visit};
    use tracing::{Event, Level, Metadata, Subscriber};

    use super::*;

    /// Minimal subscriber that records whether a warn-level event carrying the
    /// CPU-fallback message was emitted, so the "loud" fallback can be asserted
    /// without depending on `tracing-subscriber`.
    #[derive(Clone, Default)]
    struct WarnCapture {
        saw_fallback_warning: Arc<AtomicBool>,
    }

    impl Subscriber for WarnCapture {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() == Level::WARN
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, event: &Event<'_>) {
            struct MsgVisitor {
                hit: bool,
            }
            impl Visit for MsgVisitor {
                fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                    if field.name() == "message" && format!("{value:?}").contains("running on CPU")
                    {
                        self.hit = true;
                    }
                }
            }
            let mut visitor = MsgVisitor { hit: false };
            event.record(&mut visitor);
            if visitor.hit {
                self.saw_fallback_warning.store(true, Ordering::SeqCst);
            }
        }

        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[test]
    fn require_gpu_without_device_fails_fast() {
        let config = DeviceConfig {
            gpu_device: 0,
            memory_fraction: 0.9,
            require_gpu: true,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };
        // On a host with no usable GPU (and on the default non-accelerator
        // build), selection must surface a typed GPU error rather than serving
        // on CPU.
        match select_device(&config) {
            Err(JammiError::Gpu(msg)) => {
                assert!(msg.contains("GPU required"), "unexpected message: {msg}");
            }
            other => panic!("expected JammiError::Gpu, got {other:?}"),
        }
    }

    #[test]
    fn default_without_device_falls_back_to_cpu_with_warning() {
        let capture = WarnCapture::default();
        let flag = Arc::clone(&capture.saw_fallback_warning);

        let device = tracing::subscriber::with_default(capture, || {
            let config = DeviceConfig {
                gpu_device: 0,
                memory_fraction: 0.9,
                require_gpu: false,
                compute_precision: jammi_numerics::ComputePrecision::F32,
            };
            select_device(&config).expect("default fallback must not error")
        });

        assert!(
            matches!(device, Device::Cpu),
            "expected CPU fallback, got {device:?}"
        );
        assert!(
            flag.load(Ordering::SeqCst),
            "expected a loud warn-level CPU-fallback log"
        );
    }

    /// The PTX-ISA driver floor (#304): a driver below the CUDA 12.6 line the
    /// build's PTX requires is rejected with a typed, actionable error; the
    /// 12.6 line and anything newer pass.
    #[test]
    fn driver_floor_rejects_below_and_accepts_at_or_above() {
        // 12.4 (r550) — the reporter's driver in #277: below the floor, rejected.
        match check_driver_floor(12_040) {
            Err(JammiError::Gpu(msg)) => {
                assert!(msg.contains("r560"), "message must name the fix: {msg}");
                assert!(
                    msg.contains("12.4"),
                    "message must report the driver's CUDA version: {msg}"
                );
            }
            other => panic!("expected JammiError::Gpu for an old driver, got {other:?}"),
        }
        // Exactly at the floor (12.6) and newer (12.8) both pass.
        assert!(check_driver_floor(MIN_DRIVER_CUDA_VERSION).is_ok());
        assert!(check_driver_floor(12_080).is_ok());
    }

    /// The architecture floor (#306): a device below sm_80 is rejected with a
    /// typed, actionable error naming its real architecture and the Ampere+
    /// requirement; sm_80 and anything newer pass.
    #[test]
    fn compute_cap_floor_rejects_below_and_accepts_at_or_above() {
        // Turing (sm_75, e.g. T4): below the floor, rejected.
        match check_compute_cap_floor(7, 5) {
            Err(JammiError::Gpu(msg)) => {
                assert!(msg.contains("sm_75"), "message must name the device: {msg}");
                assert!(
                    msg.contains("sm_80") && msg.contains("Ampere"),
                    "message must state the Ampere/sm_80+ requirement: {msg}"
                );
                assert!(
                    msg.contains("Turing"),
                    "message must name Turing as unsupported: {msg}"
                );
            }
            other => panic!("expected JammiError::Gpu for a sub-Ampere device, got {other:?}"),
        }
        // Even older (Maxwell, sm_50) is also rejected.
        assert!(check_compute_cap_floor(5, 0).is_err());
        // Exactly at the floor (8.0) and newer (8.6, 9.0) all pass.
        assert!(check_compute_cap_floor(8, 0).is_ok());
        assert!(check_compute_cap_floor(8, 6).is_ok());
        assert!(check_compute_cap_floor(9, 0).is_ok());
    }

    /// Ties the build's CUDA architecture floor to every `ComputePrecision`'s
    /// hardware requirement (#306 follow-up). Device admission
    /// (`check_compute_cap_floor` above) is what lets the bf16 gate in
    /// `CandleBackend::load` admit a `Device::Cuda(_)` unconditionally rather
    /// than re-querying its capability — that shortcut is only sound while
    /// `MIN_CUDA_COMPUTE_CAP` is at least as strict as every precision's
    /// declared floor. If the build floor were ever lowered below a
    /// precision's requirement (e.g. an sm_75 build variant), that precision
    /// would silently admit on hardware that cannot run it. The inner match
    /// has no wildcard arm, so a newly added `ComputePrecision` variant fails
    /// to compile here until its requirement (if any) is checked.
    #[test]
    fn build_floor_covers_every_cuda_precision_requirement() {
        for precision in [
            jammi_numerics::ComputePrecision::F32,
            jammi_numerics::ComputePrecision::F16,
            jammi_numerics::ComputePrecision::BF16,
        ] {
            let precision = match precision {
                jammi_numerics::ComputePrecision::F32 => precision,
                jammi_numerics::ComputePrecision::F16 => precision,
                jammi_numerics::ComputePrecision::BF16 => precision,
            };
            if let Some(req) = precision.min_cuda_capability() {
                assert!(
                    MIN_CUDA_COMPUTE_CAP >= req,
                    "{precision:?} requires CUDA compute capability {req:?}, but this build's \
                     device-admission floor (MIN_CUDA_COMPUTE_CAP) is only \
                     {MIN_CUDA_COMPUTE_CAP:?} — a device admitted by `check_compute_cap_floor` \
                     would not actually support {precision:?}, and the bf16 gate in \
                     `CandleBackend::load` no longer re-checks capability per device"
                );
            }
        }
    }
}

#[cfg(test)]
mod pooling_from_config_tests {
    use super::*;

    #[test]
    fn absent_file_falls_back_to_mean() {
        assert_eq!(
            pooling_from_config(None, "test-model").unwrap(),
            Pooling::Mean
        );
    }

    #[test]
    fn cls_flag_selects_cls() {
        let cfg = serde_json::json!({
            "pooling_mode_cls_token": true,
            "pooling_mode_mean_tokens": false,
        });
        assert_eq!(
            pooling_from_config(Some(&cfg), "test-model").unwrap(),
            Pooling::Cls
        );
    }

    #[test]
    fn mean_sqrt_len_alone_maps_to_mean() {
        let cfg = serde_json::json!({
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_sqrt_len_tokens": true,
        });
        assert_eq!(
            pooling_from_config(Some(&cfg), "test-model").unwrap(),
            Pooling::Mean
        );
    }

    #[test]
    fn lasttoken_flag_is_a_hard_error() {
        let cfg = serde_json::json!({
            "pooling_mode_cls_token": false,
            "pooling_mode_lasttoken": true,
        });
        match pooling_from_config(Some(&cfg), "test-model") {
            Err(JammiError::Model { model_id, message }) => {
                assert_eq!(model_id, "test-model");
                assert!(
                    message.contains("pooling_mode_lasttoken"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected a hard error, got {other:?}"),
        }
    }

    #[test]
    fn cls_and_mean_both_true_is_a_hard_error() {
        let cfg = serde_json::json!({
            "pooling_mode_cls_token": true,
            "pooling_mode_mean_tokens": true,
        });
        assert!(
            pooling_from_config(Some(&cfg), "test-model").is_err(),
            "an ambiguous multi-mode declaration must fail loudly, not silently pick one"
        );
    }

    #[test]
    fn present_file_with_no_true_flag_is_a_hard_error() {
        let cfg = serde_json::json!({
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": false,
        });
        assert!(pooling_from_config(Some(&cfg), "test-model").is_err());
    }

    #[test]
    fn wrong_shape_json_is_a_hard_error() {
        // Syntactically valid JSON of the wrong shape (an array, not an
        // object) — caught by `as_object()`. This is a different code path
        // from a syntactically unparseable file (see the integration test
        // covering an actually-corrupt `1_Pooling/config.json` at the
        // resolver, since `pooling_from_config` only ever sees already-parsed
        // `serde_json::Value`s and can never observe a parse failure itself).
        let cfg = serde_json::json!(["not", "an", "object"]);
        assert!(pooling_from_config(Some(&cfg), "test-model").is_err());
    }
}

#[cfg(test)]
mod ner_nonfinite_logit_tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, StringArray};
    use candle_nn::Linear;

    use super::*;

    /// A `CandleTextForward` stub whose `forward_hidden` returns a hand-set
    /// `(batch, seq, hidden)` tensor, with `nan_row`'s hidden states forced to
    /// NaN — bypassing a real encoder to reproduce, deterministically, the
    /// shape `forward_ner` receives from a genuinely diverged model (a NaN
    /// hidden state feeds through the token classifier's linear layer to a
    /// non-finite logit row, since `NaN * weight + bias` is NaN).
    struct FixedHiddenForward {
        max_seq_len: usize,
        hidden_size: usize,
        nan_row: usize,
    }

    impl CandleTextForward for FixedHiddenForward {
        fn max_sequence_length(&self) -> usize {
            self.max_seq_len
        }

        fn forward_hidden(
            &self,
            input_ids: &Tensor,
            _attention_mask: &Tensor,
            _encoding: &BatchEncoding,
            device: &Device,
        ) -> Result<Tensor> {
            let (batch, seq) = input_ids
                .dims2()
                .map_err(|e| JammiError::Inference(e.to_string()))?;
            let mut data = vec![0.1_f32; batch * seq * self.hidden_size];
            if self.nan_row < batch {
                let row_start = self.nan_row * seq * self.hidden_size;
                for v in &mut data[row_start..row_start + seq * self.hidden_size] {
                    *v = f32::NAN;
                }
            }
            Tensor::from_vec(data, (batch, seq, self.hidden_size), device)
                .map_err(|e| JammiError::Inference(e.to_string()))
        }
    }

    /// A minimal NER-shaped `CandleModel`: a real tiny_bert tokenizer, `text`
    /// wired to `FixedHiddenForward` so the encoder's hidden states are
    /// hand-set (`nan_row`'s corrupted), and a tiny real `ner_classifier`
    /// linear layer — no real weights or forward compute needed beyond that
    /// to exercise `forward_ner`'s non-finite-logit handling.
    fn model_with_hidden_states(nan_row: usize) -> CandleModel {
        const HIDDEN: usize = 4;
        let tokenizer_path = jammi_test_utils::cookbook_fixture("tiny_bert").join("tokenizer.json");
        let tokenizer = TokenizerWrapper::from_file(&tokenizer_path).unwrap();
        let mut id2label = HashMap::new();
        id2label.insert(0u32, "O".to_string());
        id2label.insert(1u32, "B-PER".to_string());

        let device = Device::Cpu;
        let weight = Tensor::zeros((2, HIDDEN), DType::F32, &device).unwrap();
        let bias = Tensor::zeros((2,), DType::F32, &device).unwrap();
        let ner_classifier = Linear::new(weight, Some(bias));

        CandleModel {
            dimensions: ModelDimensions {
                hidden_size: HIDDEN,
                num_layers: 1,
                num_attention_heads: 1,
                intermediate_size: HIDDEN,
            },
            text: Some(Box::new(FixedHiddenForward {
                max_seq_len: 128,
                hidden_size: HIDDEN,
                nan_row,
            })),
            vision: None,
            audio: None,
            audio_frontend: None,
            tokenizer: Some(tokenizer),
            device,
            projection_head: None,
            distribution_head: None,
            regression_scaler: None,
            regression_form: None,
            id2label: Some(id2label),
            ner_classifier: Some(ner_classifier),
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn two_row_content() -> Vec<ArrayRef> {
        vec![Arc::new(StringArray::from(vec!["fine row", "diverged row"])) as ArrayRef]
    }

    /// RED-first (the class-sweep audit finding): a diverged model's row (row
    /// 1's hidden states are NaN, so its token logits are non-finite) must
    /// refuse the WHOLE call with a typed `JammiError::Inference` naming the
    /// offending row, through the public `CandleModel::forward` NER entry —
    /// never route a non-finite MODEL output onto the per-row `_status`
    /// channel, whose contract (`InferenceRunner::run_chunks` in runner.rs)
    /// reserves `_status = error` for a bad row INPUT, never "the model is
    /// broken". Reverting the candle.rs fix (keeping this test) shows this
    /// assertion fails: the old code silently reports row 1 as an ordinary
    /// per-row NER decode failure (`_status = error`, `entities = "[]"`)
    /// instead of refusing the batch — an operator reading that row would be
    /// told to look at their input text, not their model.
    #[test]
    fn nan_hidden_state_row_is_refused_with_a_typed_error_naming_the_row() {
        let model = model_with_hidden_states(1);
        let content = two_row_content();

        match model.forward(&content, ModelTask::Ner) {
            Err(JammiError::Inference(msg)) => {
                assert!(
                    msg.contains("row 1"),
                    "error must name the offending row index (1): {msg}"
                );
                assert!(
                    msg.to_lowercase().contains("non-finite")
                        || msg.to_lowercase().contains("diverg"),
                    "error must explain the cause is a diverged model, not bad input: {msg}"
                );
            }
            Err(other) => panic!(
                "expected JammiError::Inference for a non-finite NER logit row, got a \
                 different JammiError variant: {other:?}"
            ),
            Ok(_) => panic!(
                "expected a typed JammiError::Inference refusal for a non-finite NER logit \
                 row, but the call succeeded"
            ),
        }
    }

    /// Positive control: the SAME two-row batch shape with all-finite hidden
    /// states (no NaN row) succeeds — proves the refusal above triggers on
    /// the non-finite hidden state specifically, not on some incidental
    /// shape/setup difference in the test harness.
    #[test]
    fn all_finite_hidden_states_decode_successfully() {
        // `nan_row = 2` is out of range for a 2-row batch (valid indices 0
        // and 1), so `FixedHiddenForward` never corrupts either row.
        let model = model_with_hidden_states(2);
        let content = two_row_content();

        let output = model
            .forward(&content, ModelTask::Ner)
            .expect("an all-finite batch must decode successfully");
        assert!(
            output.row_status.iter().all(|&ok| ok),
            "expected every row to decode successfully, got row_status {:?}",
            output.row_status
        );
    }
}
