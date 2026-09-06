use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::{ComputeDevice, ModelContentDigest};
use jammi_encoders::{
    Bert, BertConfig, DistilBert, DistilBertConfig, ModernBert, ModernBertConfig, Pooling,
};
use sha2::Digest;

use jammi_encoders::{
    ClipText, ClipTextConfig, HtsatAudio, HtsatAudioConfig, OpenClipVisionConfig,
    OpenClipVisionTransformer,
};

use super::gguf::{self, GgufArchitecture};
use super::open_clip_text::OpenClipTextForward;
use super::{DeviceConfig, ModelBackend};
use crate::fine_tune::classifier::SeqClassifier;
use crate::inference::adapter::BackendOutput;
use crate::inference::{
    arrow_to_audio, arrow_to_images, arrow_to_texts, audio_preprocess, image_preprocess,
};
use crate::model::arch::EncoderFamily;
use crate::model::tokenizer::{BatchEncoding, TokenizerWrapper};
use crate::model::{
    LoadedModel, ModelDimensions, ModelTask, ResolvedModel, TokenizerSource, WeightsFormat,
};

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
    ///
    /// **Pairing rule with [`Self::resolved_pooling`] (audit round 62
    /// advisory A1, folded round 4)**: this default performs REAL mean
    /// pooling — a caller that reads [`Self::resolved_pooling`]'s default
    /// (`None`) after calling this default would wrongly conclude no
    /// pooling occurred. If you override ONE of these two methods, override
    /// BOTH, so `resolved_pooling()` always reports the SAME strategy this
    /// method actually applies.
    ///
    /// **A2 correction (audit round 62, adversarial round 6, R5-F2)**: this
    /// doc previously claimed the three classification wrappers below
    /// (`*ClassificationForward`) were safe inheriting BOTH trait defaults
    /// together because `LoadedModel::forward` "never routes
    /// `ModelTask::TextEmbedding` to a classification-loaded model" — that
    /// was an ASSUMED invariant, not an ENFORCED one, and it does not hold:
    /// [`super::super::cache::ModelCache`] keys its warm cache purely on
    /// [`crate::model::ModelId`] (no [`ModelTask`] on a `CacheEntry`), and neither
    /// `ModelCache::get_or_load`'s warm-hit path nor `EmbeddingPipeline`
    /// compares the requesting task against the task the entry was
    /// originally loaded for. A model loaded once for `Classification` and
    /// then requested again for `TextEmbedding` against the SAME id
    /// genuinely reaches this trait's default `forward_pooled` over a
    /// classification wrapper's softmax-logit `forward_hidden` output —
    /// mean-pooling class probabilities into a shape/width that has nothing
    /// to do with an embedding. The three classification wrappers now
    /// override BOTH this method (a typed refusal —
    /// `classification_pooling_refusal`) and [`Self::resolved_pooling`]
    /// (`classification_resolved_pooling`, honestly `None`), so the
    /// mismatch is refused AT THE SURFACE the moment it is reached, rather
    /// than assumed unreachable by a routing invariant nothing enforced.
    /// The cache's id-only key is a separate, known, pre-existing design
    /// point (whether to rekey on task too) — this refusal closes the
    /// SAFETY hole without touching that key.
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

    /// The pooling strategy [`Self::forward_pooled`] ACTUALLY applies, if
    /// pooling is a meaningful concept for this encoder at all — `None` for
    /// a wrapper whose `forward_pooled` bypasses pooling entirely (e.g. the
    /// OpenCLIP text tower, already-pooled-and-projected) or that doesn't
    /// pool at all (a classification head). The three BERT-family wrappers
    /// (`BertForward`/`ModernBertForward`/`DistilBertForward`) override this
    /// to report the SAME `pooling: Pooling` field `forward_pooled` reads —
    /// never a second, independently-resolved value (unit-62 F-5': the
    /// accessor a caller reads must be wired to the value that actually
    /// determined the served output, the same discipline
    /// `LoadedModel::compute_precision` already follows). Trait-default
    /// `None` so a future wrapper doesn't silently claim a pooling strategy
    /// it doesn't have.
    ///
    /// **Pairing rule with [`Self::forward_pooled`] (audit round 62 advisory
    /// A1, folded round 4)**: `forward_pooled`'s own default is NOT a no-op
    /// — it mean-pools for real — so this method's `None` default is
    /// truthful ONLY for a wrapper that never actually reaches
    /// `forward_pooled`'s default at runtime. A wrapper that inherits
    /// `forward_pooled`'s default AND is reachable via
    /// `ModelTask::TextEmbedding` must override THIS method to return
    /// `Some(Pooling::Mean)` — inheriting both defaults together for a
    /// wrapper that pooling actually applies to is the exact silent
    /// mismatch this note exists to prevent. The three classification
    /// wrappers instead override BOTH methods to refuse the mismatch
    /// outright rather than inherit either default — see
    /// [`Self::forward_pooled`]'s A2 correction (R5-F2) for why the
    /// underlying routing invariant this pairing rule warned about is not
    /// merely theoretical.
    fn resolved_pooling(&self) -> Option<Pooling> {
        None
    }

    /// Whether [`Self::forward_hidden`] returns classification logits
    /// (`[batch, num_classes]`, softmax already applied — see the three
    /// `*ClassificationForward` wrappers below) rather than raw per-token
    /// hidden states (`[batch, seq_len, hidden_size]`). Trait-default
    /// `false`: every text-embedding/NER wrapper's `forward_hidden` is the
    /// raw-hidden-states shape.
    ///
    /// **R7-F2 (audit round 62, adversarial round 8 advisory fold)**: the
    /// mirror of R5-F2's `forward_pooled` refusal, in the OTHER mismatch
    /// direction. [`super::super::cache::ModelCache`]'s warm cache is keyed
    /// purely on [`crate::model::ModelId`] (no [`ModelTask`]), so a model
    /// loaded once for `TextEmbedding` or `Ner` and then requested again for
    /// `Classification` against the SAME id reaches this same warm entry's
    /// `forward_hidden` — but many BERT-family checkpoints carry an
    /// `id2label` map in `config.json` (often the architecture's default
    /// 2-label declaration) even when loaded for a non-classification task,
    /// so `CandleModel::forward_classification`'s `id2label.is_some()` guard
    /// alone does not catch this mismatch. Without this signal,
    /// `forward_classification` reaches `to_vec2::<f32>()` over a 3-D
    /// `[batch, seq_len, hidden_size]` tensor it expects to be the 2-D
    /// `[batch, num_classes]` softmax output, and dies with an opaque candle
    /// rank error instead of a legible typed refusal — see
    /// `classification_kind_mismatch_refusal`.
    fn is_classification_head(&self) -> bool {
        false
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

/// The typed refusal every classification wrapper's [`CandleTextForward::forward_pooled`]
/// override returns (audit round 62, adversarial round 6, R5-F2). `forward_pooled`
/// is [`ModelTask::TextEmbedding`]'s sole entry point — pooling is not a
/// meaningful operation over a classification head's softmax-logit
/// `forward_hidden` output, so this refuses rather than silently running
/// `CandleTextForward::forward_pooled`'s trait-default mean-pool over it
/// (which either shape-errors, or — worse — mean-pools softmax
/// probabilities into a confident-wrong "embedding" of the WRONG width).
///
/// This closes the gap the pre-fix A1 doc's routing invariant assumed away:
/// [`super::super::cache::ModelCache`] keys its cache purely on
/// [`crate::model::ModelId`], carrying no [`ModelTask`] on a warm entry — a caller requesting
/// `TextEmbedding` against a model id that was actually loaded (by an
/// earlier caller, or a different task on the same process) for
/// `Classification` reaches this same warm entry and this same
/// `forward_pooled` call. The mismatch must be refused HERE, at the one
/// seam every such call passes through, rather than assumed unreachable by
/// a doc comment nothing enforces.
fn classification_pooling_refusal() -> JammiError {
    JammiError::Inference(
        "model was loaded for classification; text-embedding pooling is not defined for it".into(),
    )
}

/// The [`CandleTextForward::resolved_pooling`] every classification
/// wrapper overrides to alongside [`classification_pooling_refusal`]
/// (audit round 62, adversarial round 6, R5-F2): `None` is honest here —
/// pooling is not merely "unresolved," it is refused outright by
/// `forward_pooled` above, so there is no strategy to report. Overriding
/// this explicitly (rather than relying on the trait's `None` default)
/// keeps the two methods' pairing visible at each classification wrapper's
/// own `impl` block, matching the discipline the A1 pairing-rule doc
/// requires of every wrapper that overrides one of this pair.
fn classification_resolved_pooling() -> Option<Pooling> {
    None
}

/// The typed refusal `CandleModel::forward_classification` returns when the
/// warm text wrapper it is dispatched to is NOT one of the three
/// `*ClassificationForward` wrappers (`CandleTextForward::is_classification_head`
/// is `false`) — audit round 62, adversarial round 8 advisory fold (R7-F2),
/// the mirror of [`classification_pooling_refusal`] in the OTHER direction.
/// Without this, a `Classification` call against a warm entry that was
/// actually loaded for `TextEmbedding`/`Ner` (reachable whenever that
/// checkpoint's `config.json` happens to carry an `id2label` map, which
/// `forward_classification`'s own presence guard alone does not catch —
/// see `CandleTextForward::is_classification_head`'s doc) would instead
/// reach `to_vec2::<f32>()` over a raw `[batch, seq_len, hidden_size]`
/// hidden-states tensor and die with an opaque candle rank error.
fn classification_kind_mismatch_refusal() -> JammiError {
    JammiError::Inference(
        "model was not loaded for classification; this warm entry's text forward pass does not \
         produce classification logits"
            .into(),
    )
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

    fn resolved_pooling(&self) -> Option<Pooling> {
        Some(self.pooling)
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

    fn resolved_pooling(&self) -> Option<Pooling> {
        Some(self.pooling)
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

    fn resolved_pooling(&self) -> Option<Pooling> {
        Some(self.pooling)
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

    fn forward_pooled(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        Err(classification_pooling_refusal())
    }

    fn resolved_pooling(&self) -> Option<Pooling> {
        classification_resolved_pooling()
    }

    fn is_classification_head(&self) -> bool {
        true
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

    fn forward_pooled(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        Err(classification_pooling_refusal())
    }

    fn resolved_pooling(&self) -> Option<Pooling> {
        classification_resolved_pooling()
    }

    fn is_classification_head(&self) -> bool {
        true
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

    fn forward_pooled(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        Err(classification_pooling_refusal())
    }

    fn resolved_pooling(&self) -> Option<Pooling> {
        classification_resolved_pooling()
    }

    fn is_classification_head(&self) -> bool {
        true
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
    /// The model's content digest (esc-057, K7): a SHA-256 fold of the
    /// resolved model directory's `config.json` / `1_Pooling/config.json` /
    /// tokenizer / weights bytes, computed once here by
    /// [`compute_model_content_digest`] and carried through unchanged. See
    /// that function for the exact input set and ordering. Output-affecting
    /// (two directories that share one `model_id` but differ in any of those
    /// bytes must never collide on one `DefinitionHash`), so the
    /// materialization contract folds it into `ModelIdentity.content_digest`.
    pub(crate) content_digest: ModelContentDigest,
    /// The load-time `stat`-only staleness fingerprint (esc-058) over the
    /// same input set `content_digest` was hashed from, computed once here
    /// by [`compute_model_fingerprint`] and re-probed on every warm
    /// `ModelCache::get_or_load` hit via [`ModelFingerprint::probe`]. See
    /// that type's doc for the exact guarantee (a tripwire, not a
    /// cryptographic one) and [`compute_model_fingerprint`] for the input
    /// set.
    pub(crate) fingerprint: ModelFingerprint,
    /// The GGUF/k-quant weight-storage format this model's backbone loaded
    /// from — `Some` (the MODAL quantized dtype among the backbone's
    /// matmul-site tensors) for a `model.gguf` load, `None` for every
    /// safetensors/ONNX load (issue #351). See
    /// [`super::super::LoadedModel::quantization`]'s doc for the
    /// output-affecting rationale.
    pub(crate) quantization: Option<jammi_numerics::WeightQuantization>,
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

/// Streaming SHA-256 (hex-encoded) of `path`'s raw bytes, plus the byte
/// length — a bounded-size buffer, never `std::fs::read`'s whole-file `Vec`:
/// a real checkpoint's `model.safetensors` can be several GB, and loading it
/// entirely into memory just to hash it would roughly double this call's
/// peak RSS for no reason — the hasher only ever needs the current chunk.
///
/// This is the SOLE hashing routine [`compute_model_content_digest`] calls
/// for every input file (config, `1_Pooling/config.json`, tokenizer,
/// weights) — never a second, independently-drifting implementation.
/// `jammi-bench` carries an function-identical streaming pattern
/// (`crates/jammi-bench/src/finetune_step.rs::sha256_and_len`) for its own
/// checkpoint-identity hashing, but `jammi-bench` depends on `jammi-ai` (not
/// the reverse), so that implementation cannot be imported here — this is an
/// independent copy of the same technique, not a divergent one.
fn sha256_and_len(path: &std::path::Path) -> Result<(String, u64)> {
    use std::io::Read;

    let file = std::fs::File::open(path).map_err(|e| JammiError::Model {
        model_id: path.display().to_string(),
        message: format!("failed to open {path:?} for content-digest hashing: {e}"),
    })?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = sha2::Sha256::new();
    // 64 KiB: large enough to amortize per-read syscall overhead, small
    // enough that this function's own peak RSS contribution stays negligible
    // regardless of the file's total size.
    let mut buf = [0u8; 65536];
    let mut total_len: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| JammiError::Model {
            model_id: path.display().to_string(),
            message: format!("failed to read {path:?} for content-digest hashing: {e}"),
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total_len += n as u64;
    }
    Ok((hex::encode(hasher.finalize()), total_len))
}

/// Compute a resolved model's content digest (esc-057, K7): the
/// [`ModelIdentity`](jammi_db::store::manifest::ModelIdentity) determinant
/// that closes the defect where `model_id` alone did not change when the
/// referenced directory's `1_Pooling/config.json`, tokenizer, or weights
/// bytes changed. Called ONCE per model load, from
/// [`compute_model_identity_facets`] — see that function's doc for the
/// ordering invariant (audit round 62, F-4''): this MUST run AFTER
/// [`compute_model_fingerprint`], never before, and never independently
/// from a production call site. A future edit that reorders the two calls
/// (or adds a second call site to either) breaks the invariant silently
/// unless it also edits `compute_model_identity_facets`'s doc — review
/// should treat any diff touching that ordering as load-bearing.
///
/// **Input set** — every file whose bytes are output-affecting relative to
/// the bare `model_id` string: `resolved.config_path` (`config.json` or
/// `open_clip_config.json`), `1_Pooling/config.json` when
/// `resolved.pooling_config` is `Some` (mirrors exactly the presence check
/// `pooling_from_config` itself gates on, so the digest never hashes a file
/// the loader did not actually read), `preprocessor_config.json` when
/// `resolved.preprocessor_config` is `Some` (audit round 62, F-2: the CLAP
/// audio tower reads its whole feature-extractor geometry from it — the
/// identical reconstruction pattern as `1_Pooling/config.json`),
/// `resolved.tokenizer`'s file when present, every path in
/// `resolved.weights_paths`, and — when `resolved.adapter_path` is `Some`
/// — the fine-tune adapter's own `adapter_config.json` /
/// `adapter.safetensors` pair (audit round 62, F-1), gated by the IDENTICAL
/// presence check [`CandleBackend::load`] itself applies before reading them
/// (`adapter_config.json` and `adapter.safetensors` both exist under
/// `adapter_path`) — never hashing a file the loader did not actually read,
/// and never silently omitting one it did.
///
/// **Ordering** — entries are sorted by their RELATIVE path, lexicographically
/// by UTF-8 bytes. NOT filesystem/readdir order (unspecified across platforms
/// and mtimes) and NOT construction order — so the digest is reproducible
/// across hosts and process runs regardless of how the resolver assembled
/// `resolved.weights_paths`. Every entry is expressed relative to ONE of two
/// fixed anchor directories: the resolved model directory
/// (`resolved.config_path`'s parent) for config/pooling/preprocessor/
/// tokenizer/weights, or `resolved.adapter_path` (prefixed `adapter/`, so it
/// can never collide with a model-dir-anchored relpath of the same name) for
/// the adapter pair — never an absolute, host-specific path. A candidate that
/// cannot be expressed relative to its own anchor (every path this engine
/// resolves today is always a child of one of the two anchors, so this is
/// unreachable in production) is a typed refusal rather than a silent
/// absolute-path fallback: folding an absolute path into a digest documented
/// to be reproducible across hosts would make that claim false the moment two
/// hosts mounted the SAME model content under different paths.
///
/// **Combination** — each sorted entry is hashed once via [`sha256_and_len`]
/// (the same streaming routine for every input, never duplicated), then
/// folds its relative-path string, its own hex digest, and its byte length
/// into one outer SHA-256 as a canonical `"{relpath}\0{hex}\0{len}\n"`
/// record. The NUL/newline framing makes the combined digest unambiguous — a
/// path cannot silently absorb bytes from an adjacent record.
///
/// **Errors are typed refusals (K2)** — an IO failure while hashing (a file
/// vanishing between resolve and load, a permission error, a truncated read)
/// propagates as a `JammiError`, never silently collapsing into
/// [`ModelContentDigest::Unavailable`]. `Unavailable` is reserved for the
/// external-producer import path (`pipeline::import`), which has no local
/// model directory to hash at all — a categorically different case from "the
/// loader tried to hash local files and failed."
fn content_digest_entries(resolved: &ResolvedModel) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut gated: Vec<(String, std::path::PathBuf)> = all_candidate_paths(resolved)?
        .into_iter()
        .flat_map(|slot| slot.arms.into_iter())
        .filter(|arm| arm.gated)
        .map(|arm| (arm.rel, arm.path))
        .collect();
    gated.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(gated)
}

/// One arm within a [`DigestSlot`] — a single candidate filename the
/// resolver's own preference chain considers for that slot, anchored the
/// same way every other candidate is (see [`all_candidate_paths`]'s
/// rel/anchor machinery). `gated` mirrors the pre-reshape `DigestCandidate`'s
/// per-file flag: whether the resolver actually SELECTED this specific arm
/// for the CURRENT load — i.e. whether [`content_digest_entries`] hashes it.
/// At most one arm is EVER gated within any given [`DigestSlot`] — every
/// slot models mutually SUBSTITUTABLE alternates only (the config/tokenizer
/// slots' resolver chain picks exactly one filename; the weights slot's
/// three named arms — `model.safetensors` / `open_clip_model.safetensors` /
/// `model.onnx` — are likewise alternates, never a set the loader needs
/// jointly). A sharded HF download's shard files are each their OWN
/// single-arm, always-gated, `absence_tolerated: false` slot (audit round
/// 62, adversarial round 12) — never additional arms folded into the
/// weights slot above, because a shard is CONJUNCTIVELY required (every
/// shard must be present for `VarBuilder::from_mmaped_safetensors` to
/// succeed) rather than a disjunctive alternate a cold resolve could pick
/// one of. See [`all_candidate_paths`]'s weights-slot construction for the
/// full reasoning and the primary-file edge this closes.
struct SlotArm {
    rel: String,
    path: std::path::PathBuf,
    gated: bool,
}

/// A resolver preference-chain SLOT (audit round 62, adversarial round 10 —
/// "the terminal class closure", exhaustively enumerated over
/// `all_candidate_paths` and all four resolver paths): every arm the
/// resolver's OWN chain considers for one logical file, grouped together —
/// replacing the earlier flat `DigestCandidate` list, whose per-file
/// `optional: bool` could only describe ONE file in isolation. Two blind
/// spots this closes, both real for the config slot (`config.json` /
/// `open_clip_config.json`) and the weights slot (`model.safetensors` /
/// `open_clip_model.safetensors` / `model.onnx`):
///
/// 1. **An unselected arm's appearance was untracked.** The pre-reshape flat
///    list fingerprinted only the ONE arm the resolver actually selected for
///    THIS load — so a NEW file appearing in a currently-unselected arm
///    (`open_clip_config.json` appearing next to an existing `config.json`;
///    `model.onnx` appearing next to `model.safetensors`, which ALSO flips
///    the backend a COLD resolve would pick — `resolve_local` prefers ORT
///    the instant `model.onnx` exists) never touched any tracked
///    `(len, mtime)`, and `probe` reported fresh forever even though a cold
///    reload might now pick a different arm, or a different backend
///    entirely. (The tokenizer slot already closed its own version of this
///    hole in round 8, via ad-hoc dual candidates — folded into this same
///    slot shape here for uniformity, see [`all_candidate_paths`].)
/// 2. **A flat `optional: bool` cannot express "required unless an
///    alternate exists".** The selected arm's OWN deletion, with an
///    alternate arm already present on disk, is not a hard failure — a cold
///    resolve would simply pick the alternate — but the old binary
///    optional/required flag could only ever mark the whole candidate
///    always-stale or always-refuse, never "refuse only once every arm is
///    gone."
///
/// [`ModelFingerprint::probe`] tracks EVERY arm of every slot (present ->
/// `(len, mtime)`, absent -> a marker) and treats ANY arm's snapshot
/// changing — in EITHER direction — as staleness for that slot, refusing
/// (arm c) only when the slot as a whole cannot satisfy a cold resolve; see
/// [`Self::absence_tolerated`] and [`ModelFingerprint::probe`]'s own doc for
/// the full per-slot lattice.
struct DigestSlot {
    arms: Vec<SlotArm>,
    /// `true` when the loader accepts EVERY arm of this slot being absent —
    /// today, only the tokenizer: every resolver path already re-derives
    /// `None` on total absence and `CandleBackend::load`'s `.transpose()?`
    /// accepts it, so [`ModelFingerprint::probe`] never refuses for this
    /// slot, only ever reports stale. `false` for a slot the loader has NO
    /// fallback for once every arm is gone (config, weights, the adapter
    /// pair, and — per-class — `preprocessor_config.json` on an HF-CLAP
    /// audio model, via [`preprocessor_config_is_required`]): `probe`
    /// refuses (arm c) ONLY in that all-arms-absent case, never merely
    /// because the arm THIS load happened to select vanished while a
    /// still-present alternate arm would let a cold resolve succeed.
    absence_tolerated: bool,
}

/// Enumerate the FULL candidate SLOT set both [`content_digest_entries`]
/// (hashes the bytes of each slot's `gated` arm(s) only) and
/// [`compute_model_fingerprint`] (esc-058 F-4b, `stat`s every arm of every
/// slot, including absent ones, so a later APPEARANCE on ANY arm is
/// detectable) walk — the SAME set, computed by this ONE function, so the
/// two can never drift from each other. See [`compute_model_content_digest`]
/// for exactly which files this is, why, and the ordering/anchoring
/// guarantee; see [`DigestSlot`]'s doc for why candidates are grouped into
/// slots-with-alternates rather than a flat per-file list.
fn all_candidate_paths(resolved: &ResolvedModel) -> Result<Vec<DigestSlot>> {
    struct RawArm {
        path: std::path::PathBuf,
        anchor: std::path::PathBuf,
        gated: bool,
    }
    struct RawSlot {
        arms: Vec<RawArm>,
        absence_tolerated: bool,
    }

    let model_dir = resolved
        .config_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let mut slots: Vec<RawSlot> = Vec::new();

    // Config slot (audit round 62, adversarial round 10): the resolver's OWN
    // `config.json` / `open_clip_config.json` preference chain
    // (`try_catalog_lookup`, `resolve_local`, `resolve_hf_hub` — every path
    // checks `config.json` first, `open_clip_config.json` second;
    // resolver.rs:135-137/223-225/327). `resolved.config_path` names
    // whichever arm was actually selected for THIS load; the OTHER arm is
    // still a tracked candidate so its appearance is detectable and a cold
    // resolve preferring it is never silently masked. Required unless an
    // alternate exists: a reload can only succeed while AT LEAST ONE of the
    // two is present.
    slots.push(RawSlot {
        arms: crate::model::arch::CONFIG_CANDIDATE_NAMES
            .into_iter()
            .map(|name| {
                let path = model_dir.join(name);
                RawArm {
                    gated: path == resolved.config_path,
                    path,
                    anchor: model_dir.clone(),
                }
            })
            .collect(),
        absence_tolerated: false,
    });

    // `1_Pooling/config.json`: gated for the digest by whether the resolver
    // actually read one — the identical presence test `pooling_from_config`
    // gates its mean-fallback on — but always a CANDIDATE for the
    // fingerprint (F-4b: appearance must be detectable). Absence is always
    // tolerated (F-4'): a load-time-present, now-deleted config file here
    // means a reload legitimately succeeds via the mean-pooling fallback.
    slots.push(RawSlot {
        arms: vec![RawArm {
            path: model_dir.join("1_Pooling/config.json"),
            anchor: model_dir.clone(),
            gated: resolved.pooling_config.is_some(),
        }],
        absence_tolerated: true,
    });

    // `preprocessor_config.json` (F-2): same CANDIDATE pattern as
    // `1_Pooling/config.json` above, but its absence-tolerance is PER-CLASS,
    // not fixed (audit round 62, F-B) — see
    // [`preprocessor_config_is_required`]'s doc.
    slots.push(RawSlot {
        arms: vec![RawArm {
            path: model_dir.join("preprocessor_config.json"),
            anchor: model_dir.clone(),
            gated: resolved.preprocessor_config.is_some(),
        }],
        absence_tolerated: !preprocessor_config_is_required(resolved),
    });

    // The tokenizer slot (audit round 62, adversarial round 6, R5-F1; round
    // 8, R7-F1; migrated into this uniform [`DigestSlot`] shape in round
    // 10): NOT required, despite the pre-R5-F1 comment's claim that "the
    // loader has no fallback" — every resolver path (`discover_local_tokenizer`
    // locally, resolver.rs:504-514; the HF Hub `tokenizer.json` /
    // `bpe_simple_vocab_16e6.txt.gz` fallback chain remotely, resolver.rs:379-387)
    // already re-derives `tokenizer: None` when NEITHER file is present
    // instead of erroring, and `CandleBackend::load`'s `.transpose()?` over
    // that `Option` accepts `None`: the reload succeeds with
    // `self.tokenizer == None`, matching cold-process semantics exactly. A
    // text-encoding call fails LATER with that path's own typed error ("No
    // tokenizer loaded for ... model") at USE time, not at load time; a CLAP
    // audio model's `forward_audio_embedding` never reads `self.tokenizer`
    // at all, so it serves unaffected. `absence_tolerated: true` — the ONLY
    // slot for which this holds unconditionally, regardless of how many
    // arms are gone.
    //
    // R7-F1: both filenames the resolver's preference chain considers —
    // `tokenizer.json` (checked first) and `bpe_simple_vocab_16e6.txt.gz`
    // (the OpenCLIP fallback) — are UNCONDITIONAL arms of this ONE slot,
    // anchored under `model_dir`, exactly mirroring the `1_Pooling`/
    // `preprocessor` pattern above. `gated` is true for whichever arm
    // `resolved.tokenizer` actually names (using its own path, so the
    // anchor-mismatch refusal below still fires if that path is ever outside
    // `model_dir`) and the digest keeps hashing only the file the loader
    // actually read; the OTHER arm — and BOTH arms when `resolved.tokenizer`
    // is `None` — carries an absent-marker path (`model_dir.join(name)`),
    // `gated: false`, so `compute_model_fingerprint` still records its
    // `None` snapshot and a later appearance of either file trips `probe`.
    let tokenizer_json_default = model_dir.join("tokenizer.json");
    let tokenizer_bpe_default = model_dir.join("bpe_simple_vocab_16e6.txt.gz");
    let (json_path, json_gated, bpe_path, bpe_gated) = match &resolved.tokenizer {
        Some(TokenizerSource::HuggingFaceJson(p)) => {
            (p.clone(), true, tokenizer_bpe_default, false)
        }
        Some(TokenizerSource::OpenClipBpe(p)) => (tokenizer_json_default, false, p.clone(), true),
        None => (tokenizer_json_default, false, tokenizer_bpe_default, false),
    };
    slots.push(RawSlot {
        arms: vec![
            RawArm {
                path: json_path,
                anchor: model_dir.clone(),
                gated: json_gated,
            },
            RawArm {
                path: bpe_path,
                anchor: model_dir.clone(),
                gated: bpe_gated,
            },
        ],
        absence_tolerated: true,
    });

    // Weights ALTERNATES slot (audit round 62, adversarial round 10; reshaped
    // in round 12 — F-1): the resolver's OWN `model.safetensors` /
    // `open_clip_model.safetensors` / `model.onnx` preference chain
    // (`try_catalog_lookup`, resolver.rs:160-162; `resolve_local`'s
    // `has_onnx`/`has_safetensors` backend auto-selection,
    // resolver.rs:240/251-255/261-263/273-274; `download_safetensors`'s two
    // single-name tries, resolver.rs:426-431). Every known filename is a
    // tracked arm regardless of which one THIS load actually selected —
    // `model.onnx` APPEARING next to a currently-loaded `model.safetensors`
    // is exactly the case that flips the backend a cold resolve would choose
    // (`resolve_local` prefers ORT the instant `has_onnx` is true), so its
    // appearance must be just as detectable as its own disappearance. This
    // slot models ONLY these three mutually SUBSTITUTABLE named alternates —
    // a cold resolve picks exactly one of them — never a sharded download's
    // extra files (see the per-shard slots pushed separately, immediately
    // below, and F-1's fix for why folding them in here was wrong).
    //
    // **Primary-file edge (round 12)**: `download_safetensors` returns as
    // soon as EITHER single-name `repo.get` succeeds (resolver.rs:426-431),
    // WITHOUT ever enumerating shards — so whenever `resolved.weights_paths`
    // has a single entry matching one of the three names below, that is the
    // gated arm of THIS slot, exactly as before. The shard-enumeration
    // branch (resolver.rs:432-442) is reached only once BOTH single-name
    // tries fail, and HF's sharded convention names shards
    // `model-NNNNN-of-MMMMM.safetensors` — never literally `model.safetensors`
    // — so in that state `resolved.weights_paths` names NO known primary at
    // all: every arm of this slot is honestly `gated: false` (and, on disk,
    // absent), and every entry becomes its own per-shard slot below instead.
    // Both states are handled uniformly by the same per-entry name check
    // (never keyed on the entry's ORDER within `weights_paths`), so a
    // hypothetical future mixed shape (a named primary alongside extra
    // shard-like entries) would still be classified honestly rather than by
    // accident.
    //
    // **Honest residual (round 12) — this slot is `backend_hint`-blind.**
    // `DigestSlot`/`probe` decide arm (b) "stale, a cold resolve would
    // succeed via the alternate" purely from which named files exist on
    // disk — they never see the `backend_hint` the ORIGINAL `get_or_load`
    // call was made with. `resolve_local` (resolver.rs:251-270), when
    // `backend_hint == Some(Candle)`, pins `backend` to `Candle`
    // UNCONDITIONALLY and never falls back to `model.onnx` even if it
    // exists (`resolve_local`'s ORT auto-pick, resolver.rs:251-255, applies
    // ONLY when `backend_hint` is `None`) — so `model.safetensors` deleted
    // while an ungated `model.onnx` sits alongside it, under a Candle-pinned
    // load, is a case where THIS slot still reports arm (b) `Ok(false)`
    // stale (an arm — `model.onnx` — is present now), even though a cold
    // `resolve_local` call carrying that SAME `backend_hint` would hit the
    // typed refusal at resolver.rs:266-270 ("No safetensors weights found
    // for Candle backend"), not succeed via a different backend. The probe
    // therefore does not itself refuse here, contrary to arm (c)'s contract
    // doc above ("no arm can satisfy a cold resolve" -> `Err`) — this is
    // arm (b) by the slot's own on-disk-only view, whether or not the
    // RECORDED hint would actually doom the reload.
    //
    // This is deliberately left undetected rather than plumbing
    // `backend_hint` into `ResolvedModel`/`ModelFingerprint` (a
    // shared-declaration change out of this fix's scope): the reported
    // "stale" verdict is not a silent wrong-answer — `ModelCache::get_or_load`
    // (cache.rs) evicts on `Ok(false)` and falls through to `do_load`, which
    // calls `self.resolver.resolve(source, task, backend_hint)` with the
    // SAME `backend_hint` this `get_or_load` invocation was itself called
    // with (the hint is a parameter of the whole retry loop, not something
    // `probe` re-derives) — so the reload immediately hits the identical
    // typed refusal a direct probe-time `Err` would have produced, one hop
    // later, surfaced to the SAME caller as the SAME typed `JammiError`.
    // The only externally observable difference from a hypothetical
    // hint-aware `Err` here is that the stale `CacheEntry` is evicted before
    // the reload's refusal — strictly conservative (never serves the
    // now-incomplete entry again) and not a correctness gap. A caller that
    // varies `backend_hint` across calls for the SAME model id (not done by
    // any call site in this codebase today) could observe a DIFFERENT
    // backend's reload attempt than the one that built this fingerprint —
    // that cross-hint interaction predates this fix and is a property of
    // `get_or_load`'s per-call `backend_hint` parameter, not of this slot.
    // `model.gguf` (issue #351): a fourth mutually-substitutable named arm,
    // the SAME "appearance flips the backend/format a cold resolve would
    // pick" precedent `model.onnx` already established — a `model.gguf`
    // appearing alongside an existing `model.safetensors` is exactly the
    // shape the resolver's own FROZEN precedence (safetensors wins) makes
    // invisible to a cold resolve, so it must be just as tracked here.
    // Sourced from `model::arch` (issue #421 D7) so the tracked-candidate
    // list and every resolution chain in the crate name the same files. The
    // ORDER and CONTENT are unchanged, so the emitted digests are unchanged
    // (pinned by `tests/it/content_digest.rs`).
    let weight_arms: Vec<RawArm> = crate::model::arch::WEIGHTS_CANDIDATE_NAMES
        .into_iter()
        .map(|name| {
            let path = model_dir.join(name);
            let gated = resolved.weights_paths.iter().any(|w| w == &path);
            RawArm {
                gated,
                path,
                anchor: model_dir.clone(),
            }
        })
        .collect();
    slots.push(RawSlot {
        arms: weight_arms,
        absence_tolerated: false,
    });

    // Per-shard weights slots (audit round 62, adversarial round 12 — F-1,
    // the fix for the round-10 reshape's own bug): a sharded HF download
    // (`download_safetensors`'s shard fallback, resolver.rs:432-442) names
    // files OUTSIDE the fixed three-name set entirely, and — unlike the
    // three named arms above, which are mutually substitutable — every
    // shard is CONJUNCTIVELY required: `VarBuilder::from_mmaped_safetensors`
    // (candle.rs:2382) needs ALL of them, and a cold resolve always
    // re-fetches the SAME shard set (resolver.rs:432-442 collects every
    // `.safetensors` sibling deterministically), so any ONE shard's own
    // loss makes a cold resolve fail regardless of the other shards' — or
    // the three named arms' — state. Folding a shard into the alternates
    // slot above (the pre-round-12 shape) modeled it as just another
    // DISJUNCTIVE arm: deleting one shard while a sibling shard (or a named
    // arm) happened to still exist made `probe` report merely STALE
    // (`Ok(false)`) instead of the typed refusal a cold resolve of the SAME
    // now-incomplete shard set would actually hit. Each shard therefore gets
    // its OWN single-arm, always-gated, `absence_tolerated: false` slot —
    // losing it reaches the typed refusal (arm c) exactly the way losing the
    // sole `model.safetensors` does, never the alternates slot's
    // (b)-stale/(c)-refuse decision, which is keyed on the WRONG set of
    // arms for a conjunctive requirement.
    for weights in &resolved.weights_paths {
        if crate::model::arch::WEIGHTS_CANDIDATE_NAMES
            .iter()
            .all(|name| weights != &model_dir.join(name))
        {
            slots.push(RawSlot {
                arms: vec![RawArm {
                    path: weights.clone(),
                    anchor: model_dir.clone(),
                    gated: true,
                }],
                absence_tolerated: false,
            });
        }
    }

    // The fine-tune adapter's own pair (F-1): two independent required
    // files that are never alternates of each other, so each is its own
    // degenerate single-arm slot — anchored at `adapter_path` itself, NOT
    // the base model's directory, which the adapter's files never live
    // under (the resolver fetches them from the artifact store into their
    // own directory) — gated by the SAME presence check `CandleBackend::load`
    // applies (its `saved_adapter` read block, below) before reading them,
    // so this candidate set can never drift from what the loader actually
    // reads. `absence_tolerated: false` — once `adapter_path` is `Some`,
    // the loader has no unadapted-fallback and refuses outright on either
    // file's absence (see
    // `candle_backend_load_refuses_some_adapter_path_missing_files` below),
    // so a reload after either file vanishes fails identically.
    if let Some(adapter_dir) = &resolved.adapter_path {
        let cfg_path = adapter_dir.join("adapter_config.json");
        let weights_path = adapter_dir.join("adapter.safetensors");
        let gated = cfg_path.exists() && weights_path.exists();
        slots.push(RawSlot {
            arms: vec![RawArm {
                path: cfg_path,
                anchor: adapter_dir.clone(),
                gated,
            }],
            absence_tolerated: false,
        });
        slots.push(RawSlot {
            arms: vec![RawArm {
                path: weights_path,
                anchor: adapter_dir.clone(),
                gated,
            }],
            absence_tolerated: false,
        });
    }

    let mut result: Vec<DigestSlot> = Vec::with_capacity(slots.len());
    for slot in slots {
        let mut arms = Vec::with_capacity(slot.arms.len());
        for arm in slot.arms {
            let rel = arm
                .path
                .strip_prefix(&arm.anchor)
                .map_err(|_| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!(
                        "content-digest enumeration: {:?} is not nested under its expected \
                         anchor directory {:?} — every path this engine resolves today is \
                         always a child of its own anchor (the model directory, or the \
                         adapter directory), so this is a genuinely unexpected resolved-model \
                         shape; refusing rather than silently folding an ABSOLUTE, \
                         host-specific path into a digest documented to be reproducible \
                         across hosts",
                        arm.path, arm.anchor
                    ),
                })?;
            let rel_string = if arm.anchor == model_dir {
                rel.to_string_lossy().into_owned()
            } else {
                format!("adapter/{}", rel.to_string_lossy())
            };
            arms.push(SlotArm {
                rel: rel_string,
                path: arm.path,
                gated: arm.gated,
            });
        }
        result.push(DigestSlot {
            arms,
            absence_tolerated: slot.absence_tolerated,
        });
    }
    Ok(result)
}

fn compute_model_content_digest(resolved: &ResolvedModel) -> Result<ModelContentDigest> {
    let entries = content_digest_entries(resolved)?;

    let mut combined = sha2::Sha256::new();
    for (rel, path) in &entries {
        let (hex_digest, len) = sha256_and_len(path)?;
        combined.update(rel.as_bytes());
        combined.update([0u8]);
        combined.update(hex_digest.as_bytes());
        combined.update([0u8]);
        combined.update(len.to_string().as_bytes());
        combined.update([b'\n']);
    }

    Ok(ModelContentDigest::Sha256(hex::encode(combined.finalize())))
}

/// A `stat`-only on-disk staleness fingerprint (esc-058) of the SAME input
/// file set [`compute_model_content_digest`] hashes: `(relpath, len, mtime)`
/// per file, captured once at load time by [`compute_model_fingerprint`] and
/// carried on [`CandleModel::fingerprint`]. [`ModelCache::get_or_load`]'s
/// warm fast path calls [`ModelFingerprint::probe`] before serving the cached
/// `Arc<LoadedModel>` — `stat`, never a re-hash — so an in-place model-dir
/// mutation (e.g. rewriting `1_Pooling/config.json`) between two calls in the
/// same warm process is detected and forces a fresh load instead of silently
/// replaying pre-mutation weights while attesting the pre-mutation digest.
///
/// **Honest residual**: `(len, mtime)` is a staleness TRIPWIRE, not a
/// cryptographic guarantee. A content swap that lands on the exact same byte
/// length at the exact same modification time (a crafted rewrite, or a
/// same-second overwrite on a filesystem with coarse mtime resolution) is
/// invisible to this probe and will keep serving the stale in-memory model.
/// The [`ModelContentDigest`] recomputed on every actual reload remains the
/// sole authoritative attestation of what bytes were hashed; this type only
/// decides WHEN a reload is triggered, and does not strengthen the digest's
/// own guarantee.
///
/// `(relpath, absolute_path, snapshot)` for one arm within a
/// [`FingerprintSlot`]. `snapshot` is `Some((len, mtime))` when the arm
/// existed at load time, or `None` when it did not (see [`ModelFingerprint`]'s
/// doc, F-4b).
type FingerprintArm = (
    String,
    std::path::PathBuf,
    Option<(u64, std::time::SystemTime)>,
);

/// One fingerprinted SLOT — every arm of one [`DigestSlot`], captured at
/// load time, plus that slot's [`DigestSlot::absence_tolerated`] flag
/// carried through unchanged. See [`DigestSlot`]'s doc for why the
/// fingerprint is grouped this way (audit round 62, adversarial round 10)
/// rather than as a flat per-file list with an isolated `optional: bool`.
#[derive(Debug, Clone)]
struct FingerprintSlot {
    arms: Vec<FingerprintArm>,
    absence_tolerated: bool,
}

/// **Contract (unit-62 design pressure-test, PINNED).** This fingerprint
/// enforces the NARROW staleness contract esc-058 specified: detect
/// in-place mutation — content change, deletion, or appearance — of the
/// FILES the resolver selected (and their preference-chain alternates)
/// under the resolve inputs recorded at load (`source`, `task`,
/// `backend_hint`, catalog state). It deliberately does NOT re-verify
/// non-file resolve inputs:
///
/// - catalog `artifact_path`/`backend` rewrites — a retrained fine-tuned
///   model whose adapter dir is content-addressed and immutable will probe
///   fresh until process restart: `fetch_artifact` never touches bytes this
///   type is already watching, and the catalog ROW pointing at a NEW dir is
///   invisible to this type entirely (esc-057's mutable-pointer defect
///   recurring one layer up);
/// - catalog-vs-local precedence (`ModelResolver::try_catalog_lookup`'s
///   catalog-first ordering vs. a shadowed `ModelSource::Local` fallthrough
///   that could resolve differently on a cold path);
/// - task/backend_hint cache keying — cache entries are keyed by `ModelId`
///   alone, narrower than the `(source, task, backend_hint)` resolve key;
/// - HF `refs/<rev>` revision moves — the mutable pointer hf-hub resolves
///   through (`refs/<rev> -> snapshots/<sha>`) sits outside this type's
///   file-set anchor entirely and is structurally inexpressible as an arm;
///   the snapshot blobs themselves ARE immutable, so this subsystem is
///   honestly a no-op for HF sources today;
/// - remote sibling listings — a shard set derived from a remote glob
///   listing, not a finite local arm list this type could enumerate.
///
/// Those classes are unit 65's scope (`docs/plans/65-resolve-witness`), not
/// this type's — see that plan for the resolver-emitted-witness direction
/// that would widen the contract. `ModelCache::get_or_load`'s own doc
/// carries a one-line summary of this same boundary.
///
/// [`ModelCache::get_or_load`]: super::super::cache::ModelCache::get_or_load
#[derive(Debug, Clone)]
pub(crate) struct ModelFingerprint {
    /// One entry per [`DigestSlot`], in [`all_candidate_paths`]'s push
    /// order. A fixed *present-only* snapshot cannot detect a later-appearing
    /// output-affecting file (F-4b) because appearance never touches any
    /// `(len, mtime)` already being tracked — recording the ABSENCE
    /// explicitly closes that blind spot, for EVERY arm of every slot, not
    /// just the arm a given load happened to select (audit round 62,
    /// adversarial round 10). Empty for a synthetic (non-disk-backed)
    /// fixture, in which case [`ModelFingerprint::probe`] is vacuously
    /// fresh.
    slots: Vec<FingerprintSlot>,
}

impl ModelFingerprint {
    /// A fingerprint with no files to check — `probe` always reports fresh.
    /// Used only by in-process test fixtures that synthesize a `CandleModel`
    /// with no backing model directory; every real load goes through
    /// [`compute_model_fingerprint`] instead.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self { slots: vec![] }
    }

    /// Re-`stat` every arm of every candidate SLOT and compare against the
    /// load-time snapshot, evaluated one slot at a time.
    ///
    /// - `Ok(true)` — every slot's every arm's current on-disk state matches
    ///   load time (present arms have an unchanged `(len, mtime)`; absent
    ///   arms are still absent): serve the cached model.
    /// - `Ok(false)` — at least one slot has at least one arm diverged
    ///   (a present arm's `len`/`mtime` changed, an absent-at-load arm now
    ///   EXISTS — F-4b, on ANY arm, not only the selected one — or a
    ///   present-at-load arm is now `NotFound`), AND the slot as a whole can
    ///   still satisfy a cold resolve (some arm is present now, whether or
    ///   not it is the one THIS load selected, or the slot tolerates total
    ///   absence): the caller must evict and reload, never serve.
    /// - `Err` — a slot whose divergence leaves EVERY arm absent, on a slot
    ///   that does not tolerate that (config, weights, the adapter pair, or
    ///   a per-class-required preprocessor slot) — no arm, selected or
    ///   alternate, can satisfy a cold resolve, so a reload would fail
    ///   identically — or ANY arm is unreadable for a reason OTHER than
    ///   `NotFound` (e.g. a permission error, for which slot tolerance
    ///   carries no meaning): a typed refusal (K2), never silently treated
    ///   as fresh (would replay stale weights) or as stale (would mask a
    ///   real IO failure as an ordinary reload, which would then hit the
    ///   identical error inside `compute_model_content_digest` anyway —
    ///   surfacing it here is strictly more informative, naming the probe
    ///   as the cause).
    ///
    /// **The per-slot lattice (audit round 62, adversarial round 10 —
    /// reshaped from F-4''s flat, per-candidate `optional: bool`, which
    /// could only express one file in isolation)** — evaluated per SLOT,
    /// over "did any arm change" × "is any arm present now":
    ///
    /// (a) no arm of this slot changed: unchanged — continue to the next
    ///     slot.
    /// (b) some arm changed, AND at least one arm of this slot is present
    ///     now — whether or not it is the SAME arm the resolver originally
    ///     selected (e.g. `model.safetensors` deleted while
    ///     `open_clip_model.safetensors` still exists; `1_Pooling/config.json`
    ///     deleted from a live model directory; the tokenizer deleted from
    ///     ANY live model directory — audit round 62, adversarial round 6,
    ///     R5-F1): the slot can still satisfy a cold resolve, either via the
    ///     surviving alternate arm or via a documented per-class fallback
    ///     (mean pooling / no preprocessor geometry / `tokenizer: None`,
    ///     with any text-encoding call refused at USE time by its own typed
    ///     error instead of at load time — see `all_candidate_paths`'
    ///     tokenizer slot doc) — a fresh reload legitimately SUCCEEDS, with
    ///     a NEW digest reflecting whichever arm/fallback a cold resolve
    ///     now takes. Reported as STALE (`Ok(false)`), never `Err`: the
    ///     pre-round-10 per-file model collapsed the alternate-exists case
    ///     into the SAME arm (c) `Err` as total absence, permanently
    ///     wedging `ModelCache::get_or_load` on a model a cold reload would
    ///     serve just fine via the alternate.
    /// (c) some arm changed, AND every arm of this slot is now absent, on a
    ///     slot that does NOT tolerate total absence (`absence_tolerated ==
    ///     false` — config, weights, the adapter pair, and — per-class, see
    ///     [`preprocessor_config_is_required`] — `preprocessor_config.json`
    ///     on an HF-CLAP audio model): no arm can satisfy a cold resolve, so
    ///     this remains the typed refusal (`Err`), unchanged from before the
    ///     round-10 reshape. The tokenizer slot's `absence_tolerated == true`
    ///     means it NEVER reaches this arm, for any number of vanished arms.
    ///
    /// **Evict-on-`Err` (unit-62 design pressure-test, item 3 — wedge
    /// elimination).** `ModelCache::get_or_load`'s `Err` arm now evicts this
    /// `CacheEntry` before returning, exactly like its `Ok(false)` arm does.
    /// Under the narrow staleness contract (see [`ModelFingerprint`]'s own
    /// doc), the honest behavior here is cold-equivalence, not a silent
    /// recovery: the entry is gone, so the NEXT `get_or_load` call takes a
    /// full cold resolve + reload instead of re-probing this identical dead
    /// entry. If the cause that produced this `Err` is still present, that
    /// cold reload fails too — with the LOADER's own typed error (a
    /// different message than this probe's), the identical observable
    /// outcome (refusal), never a wedge into a different failure mode. If
    /// the cause was transient (e.g. the catalog-vs-local precedence class
    /// unit 65 scopes), the cold reload succeeds and the system self-heals
    /// instead of staying wedged on a dead in-memory entry forever.
    ///
    /// **Honest residual (round 12) — arm (b) is `backend_hint`-blind for the
    /// weights slot.** "Some arm of this slot present now" is an on-disk-only
    /// check; it does not know whether the ORIGINAL `get_or_load` call's
    /// `backend_hint` would make a cold resolve reject that surviving arm
    /// anyway (e.g. `model.safetensors` deleted while an ungated
    /// `model.onnx` survives, under a Candle-pinned `backend_hint`:
    /// `resolve_local` never auto-falls-back to ORT when the hint is `Some`,
    /// resolver.rs:251-270). Such a load reports arm (b) `Ok(false)` here
    /// even though a cold resolve carrying that SAME hint would in fact hit
    /// arm (c)'s typed refusal. This is NOT a silent wrong answer:
    /// `ModelCache::get_or_load` evicts on `Ok(false)` and immediately
    /// re-resolves with the SAME `backend_hint` this call was made with, so
    /// the caller gets the identical typed `JammiError` one hop later
    /// instead of directly from `probe`. See [`all_candidate_paths`]'s
    /// weights-alternates-slot doc for the full accounting of why this is
    /// left undetected rather than plumbing `backend_hint` into this type.
    ///
    /// **Slot evaluation is per-slot-local and first-change-wins (unit-62
    /// design pressure-test, item 4b — stated, not changed; widening this is
    /// unit 65's witness work).** The loop below walks `self.slots` in
    /// [`all_candidate_paths`]'s push order and returns as soon as ONE slot
    /// reports arm (b) or arm (c) — it never scans the remaining slots to
    /// see whether a LATER slot would also have changed, and never
    /// aggregates across slots. So whether a caller of this probe observes
    /// a stale-reload (arm b) or a typed refusal (arm c) on a given call is
    /// PUSH-ORDER dependent whenever more than one slot has actually
    /// diverged since load: if the weights slot (pushed first) has an arm
    /// (b) alternate-exists divergence and, say, the required config slot
    /// (pushed after it) independently has an arm (c) total-absence
    /// divergence, this call reports `Ok(false)` and never even inspects
    /// the config slot — the caller reloads, and only discovers the
    /// config-slot refusal on THAT reload's own resolve. A different push
    /// order (config first) would report the `Err` directly, on this same
    /// call. Both are typed, neither is silently wrong — but the SPECIFIC
    /// arm a caller sees for a multi-slot-divergence input is an artifact of
    /// `all_candidate_paths`'s enumeration order, not a property of the
    /// underlying divergence itself.
    pub(crate) fn probe(&self) -> Result<bool> {
        for slot in &self.slots {
            let mut changed = false;
            let mut any_arm_present_now = false;
            let mut vanished_arm_error: Option<JammiError> = None;

            for (rel, path, snapshot) in &slot.arms {
                match std::fs::metadata(path) {
                    Ok(meta) => {
                        any_arm_present_now = true;
                        let current_mtime = meta.modified().map_err(|e| JammiError::Model {
                            model_id: path.display().to_string(),
                            message: format!(
                                "esc-058 staleness probe: {path:?} has no mtime available on \
                                 this platform/filesystem: {e}"
                            ),
                        })?;
                        match snapshot {
                            Some((len, mtime)) => {
                                if meta.len() != *len || current_mtime != *mtime {
                                    changed = true;
                                }
                            }
                            // Absent at load time, present now: an
                            // output-affecting file APPEARED (F-4b) — even
                            // one this load did NOT select (audit round 62,
                            // adversarial round 10: a cold resolve may pick
                            // it instead, and for the weights slot may pick
                            // a different BACKEND entirely).
                            None => changed = true,
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound && snapshot.is_none() => {
                        // Still absent, exactly as at load time — fine.
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // This arm existed at load time and is now gone.
                        // Whether the SLOT as a whole is arm (b) or arm (c)
                        // depends on the OTHER arms — decided once the whole
                        // slot has been scanned, below.
                        changed = true;
                        if vanished_arm_error.is_none() {
                            vanished_arm_error = Some(JammiError::Model {
                                model_id: path.display().to_string(),
                                message: format!(
                                    "esc-058 staleness probe: {path:?} (fingerprinted as \
                                     {rel:?} at load time) is no longer readable: {e}"
                                ),
                            });
                        }
                    }
                    // Any arm unreadable for a reason OTHER than NotFound
                    // (e.g. a permission error): slot tolerance carries no
                    // meaning here — the failure is not "this file doesn't
                    // exist", it is "this file couldn't be inspected". A
                    // typed refusal (K2), immediately.
                    Err(e) => {
                        return Err(JammiError::Model {
                            model_id: path.display().to_string(),
                            message: format!(
                                "esc-058 staleness probe: {path:?} (fingerprinted as {rel:?} \
                                 at load time) is no longer readable: {e}"
                            ),
                        });
                    }
                }
            }

            if !changed {
                continue;
            }
            if !slot.absence_tolerated && !any_arm_present_now {
                return Err(vanished_arm_error.expect(
                    "a required slot (`absence_tolerated == false`) with every arm absent \
                     and `changed == true` must have had at least one arm transition from \
                     present-at-load to absent-now",
                ));
            }
            return Ok(false);
        }
        Ok(true)
    }
}

/// Compute the load-time staleness fingerprint (esc-058) over the FULL
/// candidate slot set [`all_candidate_paths`] returns
/// ([`content_digest_entries`]'s present-only input set is a filtered VIEW of
/// this same enumeration, so the two can never drift). `stat`s every arm of
/// every slot (never reads its bytes, unlike the digest); an arm absent at
/// load time is recorded as such (F-4b) rather than omitted. Errors are
/// typed refusals (K2) for anything other than "does not exist", the
/// identical stance [`compute_model_content_digest`] takes.
///
/// **Ordering invariant (audit round 62, F-4'')**: called ONCE per model
/// load, from [`compute_model_identity_facets`], and MUST run BEFORE
/// [`compute_model_content_digest`] there — never after, and never from any
/// other production call site. See that function's doc for why the order
/// matters (a mutation landing between the two calls must be caught by a
/// re-hash, never silently stamped as "fresh" against a stale digest
/// forever).
fn compute_model_fingerprint(resolved: &ResolvedModel) -> Result<ModelFingerprint> {
    let raw_slots = all_candidate_paths(resolved)?;
    let mut slots = Vec::with_capacity(raw_slots.len());
    for slot in raw_slots {
        let mut arms = Vec::with_capacity(slot.arms.len());
        for arm in slot.arms {
            let SlotArm {
                rel,
                path,
                gated: _gated,
            } = arm;
            match std::fs::metadata(&path) {
                Ok(meta) => {
                    let mtime = meta.modified().map_err(|e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!(
                            "{path:?} has no mtime available for esc-058 load-time \
                             fingerprinting: {e}"
                        ),
                    })?;
                    arms.push((rel, path, Some((meta.len(), mtime))));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    arms.push((rel, path, None));
                }
                Err(e) => {
                    return Err(JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!(
                            "failed to stat {path:?} for esc-058 load-time fingerprinting: {e}"
                        ),
                    });
                }
            }
        }
        slots.push(FingerprintSlot {
            arms,
            absence_tolerated: slot.absence_tolerated,
        });
    }
    Ok(ModelFingerprint { slots })
}

/// Compute BOTH per-load staleness facets — [`ModelFingerprint`] (stat) and
/// [`ModelContentDigest`] (hash) — in the ONLY safe order: fingerprint
/// FIRST, digest SECOND. This is the sole caller of either function; every
/// other call site (the `digest_fingerprint_audit62_tests` / `content_digest`
/// unit tests below) calls the two directly and independently, which is
/// exactly what makes those tests order-agnostic — production code must
/// route through here.
///
/// **Ordering invariant (audit round 62, F-4'')**: stamping the fingerprint
/// AFTER hashing the digest (the pre-fix order) leaves a window, between
/// the hash read and the stat, in which a concurrent in-place mutation of
/// the model directory stamps a POST-mutation `(len, mtime)` fingerprint
/// against a PRE-mutation digest. The warm-path probe
/// ([`ModelFingerprint::probe`]) then reports "fresh" FOREVER — its stat
/// matches the fingerprint it was given, even though that fingerprint was
/// never the one the digest was actually hashed from — so the process keeps
/// serving the POST-mutation weights under the STALE, PRE-mutation digest
/// folded into `ModelIdentity`. Stat-then-hash instead: a mutation landing
/// in the (now much narrower, and in the opposite direction) window between
/// the two calls is caught by the immediate re-hash the fingerprint no
/// longer precedes — worst case, one extra reload converges on the correct
/// digest; the fingerprint can never be silently wrong forever. Composing
/// both calls in this ONE function (rather than two independent call sites
/// in [`CandleBackend::load`]) pins the order STRUCTURALLY: a future edit
/// cannot silently swap the two calls back without editing this function's
/// two-line body, which review can see and require justification for. Do
/// not call [`compute_model_fingerprint`] and [`compute_model_content_digest`]
/// separately from any production path — only from this function, or from a
/// unit test that is deliberately probing one facet in isolation.
fn compute_model_identity_facets(
    resolved: &ResolvedModel,
) -> Result<(ModelFingerprint, ModelContentDigest)> {
    let fingerprint = compute_model_fingerprint(resolved)?;
    let content_digest = compute_model_content_digest(resolved)?;
    Ok((fingerprint, content_digest))
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

    /// The CLAP fusion front-end geometry this model was loaded with (sample
    /// rate, FFT size, hop, mel band), read off the checkpoint's own
    /// `preprocessor_config.json` at load time. `None` for every non-audio
    /// model.
    ///
    /// Exposed (issue #421 D8) so an encoder-adapters AUDIO fine-tune runs
    /// the byte→spectrogram front end through the SAME geometry the serving
    /// path uses. The trainer forwards the LoRA-wrapped tower directly rather
    /// than through `forward_audio_embedding`, so without this accessor it
    /// would have to construct a front-end config of its own — and a
    /// front-end that differs from serving's by one mel filter trains an
    /// embedding that serving can never reproduce.
    pub(crate) fn audio_frontend(&self) -> Option<&audio_preprocess::ClapFrontendConfig> {
        self.audio_frontend.as_ref()
    }

    /// The pooling strategy the text-embedding forward path ACTUALLY resolved
    /// to (unit-62 F-5'): delegates to [`CandleTextForward::resolved_pooling`]
    /// on the loaded text wrapper — the SAME `Pooling` value `forward_pooled`
    /// applies, never a re-derivation from `resolved.pooling_config` (which
    /// would drift the moment `pooling_from_config`'s own resolution logic
    /// changed without this accessor changing in lockstep). `None` for a
    /// model with no text wrapper at all (CLAP audio) or whose text wrapper
    /// doesn't pool (OpenCLIP text, DistilBERT classification).
    pub(crate) fn resolved_pooling(&self) -> Option<Pooling> {
        self.text.as_ref().and_then(|t| t.resolved_pooling())
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

        // R7-F2 (audit round 62, adversarial round 8 advisory fold): the
        // `id2label` presence check above does NOT prove this warm entry was
        // actually loaded for `Classification` — many BERT-family
        // checkpoints carry an `id2label` map in `config.json` even when
        // loaded for `TextEmbedding`/`Ner` (see
        // `CandleTextForward::is_classification_head`'s doc), and
        // `ModelCache`'s id-only warm-cache key makes that mismatch
        // reachable at runtime. A cheap, typed kind-mismatch refusal here —
        // BEFORE tokenizing anything — mirrors R5-F2's `forward_pooled`
        // refusal in the other direction, so this call fails legibly instead
        // of reaching `to_vec2::<f32>()` over the wrong-rank hidden-states
        // tensor and dying with an opaque candle rank error.
        if !self.text_forward()?.is_classification_head() {
            return Err(classification_kind_mismatch_refusal());
        }

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

        // Computed ONCE per model load, before the (potentially expensive)
        // weight loading below, so a hashing/stat IO failure (K2: a typed
        // refusal, never a silent `Unavailable`) surfaces fast rather than
        // after mmapping several GB of safetensors. Routed through the ONE
        // composed function — never the two independently — so the
        // fingerprint-before-digest order (audit round 62, F-4'') is pinned
        // structurally. See `compute_model_identity_facets`'s doc for the
        // exact input set, ordering invariant, and why it matters.
        let (fingerprint, content_digest) = compute_model_identity_facets(resolved)?;

        // The raw `model_type` SPELLING the text arm below dispatches on and
        // every refusal message names, read through the ONE shared reader
        // (`crate::model::arch::config_model_type`) rather than this site's own
        // `unwrap_or("bert")`. Behaviour here is unchanged for every string a
        // config can carry; what the shared reader buys is that the
        // absent-key default and `EncoderFamily::from_config`'s answer for the
        // same file are now the same rule, so a `config.json` without a
        // `model_type` can no longer serve as BERT here while the fine-tune
        // worker refuses it (issue #421 D7).
        let model_type = crate::model::arch::config_model_type(&resolved.model_config);

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

        // HF-CLAP audio checkpoints (`ClapAudioModelWithProjection`) declare
        // `model_type == "clap_audio_model"` at the top level (flat
        // `ClapAudioConfig`) or under a nested `audio_config` (top-level
        // `ClapConfig`), and/or list `ClapModel`/`ClapAudioModelWithProjection`
        // in `architectures`. OpenCLIP vision checkpoints carry `model_cfg`.
        // The two are disjoint, so the audio branch is checked first.
        // The ONE architecture predicate (issue #421 D7): the same
        // `EncoderFamily` the fine-tune worker dispatches on and the adapter
        // load-seam below validates against, so a checkpoint can never be
        // "CLAP" to one of them and "OpenCLIP" to the other.
        let base_family = EncoderFamily::from_config(&resolved.model_config);
        let is_clap = base_family == Some(EncoderFamily::ClapAudio);
        let is_open_clip = base_family == Some(EncoderFamily::OpenClip);

        // Normalize DistilBERT config fields to standard BERT names — the
        // SAME normalization authority `gguf::gguf_num_layers` (below, and
        // through it `estimate_gguf_residency`/the fine-tune GGUF arm)
        // routes through, so a DistilBERT config.json (whose only geometry
        // fields are its own `dim`/`n_heads`/`n_layers`/`hidden_dim` names)
        // can never diverge between this ordinary encoder-config build and
        // any GGUF consumer (issue #351 wave 5 audit).
        let model_config = gguf::normalize_model_config(model_type, &resolved.model_config);

        // GGUF weight-storage format (issue #351): the resolver already
        // classified this at resolve time (`ResolvedModel.weights_format`) —
        // never re-derived by extension-sniffing here. `Ort` never resolves
        // a GGUF path (`ModelResolver`'s local/HF arms only ever look for
        // `model.onnx`), so `(Ort, Gguf)` cannot reach this Candle backend
        // at all.
        let is_gguf = resolved.weights_format == WeightsFormat::Gguf;
        if is_gguf && (is_clap || is_open_clip) {
            return Err(JammiError::Model {
                model_id: resolved.model_id.0.clone(),
                message: format!(
                    "quantized serving not supported for this architecture (model_type \
                     '{model_type}') — GGUF loading is threaded only through the \
                     BERT-family/DistilBERT/ModernBERT text towers"
                ),
            });
        }
        let gguf_arch = if is_gguf {
            Some(
                GgufArchitecture::from_model_type(model_type).ok_or_else(|| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!(
                        "quantized serving not supported for this architecture (model_type \
                         '{model_type}')"
                    ),
                })?,
            )
        } else {
            None
        };
        // Everything the GGUF load path needs, built ONCE here: a
        // per-matmul-site `FrozenBase` map plus a synthesized in-memory
        // safetensors file carrying every OTHER tensor densified to
        // `compute_dtype` (embeddings, norms, classifier/NER heads — see
        // `gguf::load_gguf_backbone`'s own doc). `None` for a non-GGUF load
        // — every downstream site below stays byte-identical to today.
        let gguf_backbone = match gguf_arch {
            Some(arch) => {
                let num_layers = gguf::gguf_num_layers(model_type, &resolved.model_config)
                    .ok_or_else(|| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: "GGUF load requires num_hidden_layers (or num_layers) in \
                                  config.json"
                            .into(),
                    })?;
                Some(gguf::load_gguf_backbone(
                    &resolved.weights_paths[0],
                    arch,
                    num_layers,
                    compute_dtype,
                    &device,
                    &resolved.model_id.0,
                )?)
            }
            None => None,
        };
        // MODAL quantized dtype among the backbone's matmul-site tensors —
        // the value `ModelIdentity.quantization` reports (issue #351, pin
        // Δ2). Honest residual: a GGUF file whose matmul-site tensors are
        // ALL stored densely (F32/F16/BF16, no genuine k-quant tensor at
        // all — a pathological, self-defeating "GGUF" checkpoint) reports
        // `None` here rather than fabricating a quantized format that was
        // never actually used.
        let gguf_quantization = gguf_backbone.as_ref().and_then(|b| b.modal_quantization);
        // The `FrozenWeightLookup`-shaped closure every text-tower builder
        // below consults via `.weight_source(..)` — `None` for a non-GGUF
        // load (every builder call site skips `.weight_source(..)`
        // entirely, byte-identical to today).
        let gguf_lookup = gguf_backbone.as_ref().map(|b| b.lookup());

        // The root `VarBuilder` loads every weight at `compute_dtype` — the
        // encoder backbone AND every head built from it (classifier,
        // projection, CLAP/OpenCLIP towers) — because a mismatched backbone ×
        // head matmul dtype errors in candle. The one exception is the
        // fine-tune adapter path below, which loads the frozen backbone at its
        // own *persisted* `backbone_dtype` (a training-time choice, independent
        // of this inference-time knob) when a saved adapter is present.
        //
        // For a GGUF load, this reads the SYNTHESIZED densified safetensors
        // file (`gguf_backbone`'s own doc), not `resolved.weights_paths`
        // (a `model.gguf` file candle's own safetensors reader cannot parse
        // at all) — every construction site below that never consults
        // `gguf_lookup` (embeddings, LayerNorms, classifier/NER heads) reads
        // this exactly the way it reads a real safetensors checkpoint.
        let vb_weights_paths: Vec<std::path::PathBuf> = match &gguf_backbone {
            Some(b) => vec![b.densified_path.clone()],
            None => resolved.weights_paths.clone(),
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vb_weights_paths, compute_dtype, &device).map_err(
                |e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to load safetensors: {e}"),
                },
            )?
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

        // Read the saved adapter, if any. Both flavours of `SavedAdapter`
        // share the same on-disk layout (`adapter.safetensors` plus
        // `adapter_config.json` with the `adapter_type` discriminator); the
        // variant is the type-level switch that decides whether to wire
        // LoRA inside the encoder or leave it as an external projection
        // head applied post-pool.
        //
        // `resolved.adapter_path` is `Some` only via the fine-tuned-model
        // catalog-lookup path (`ModelResolver::try_catalog_lookup`), which
        // sets it exactly when a fine-tuned model record's `artifact_path`
        // was fetched from the artifact store into a local directory — i.e.
        // the resolver has already asserted "this model IS fine-tuned and
        // its adapter bundle lives here". A missing `adapter_config.json` /
        // `adapter.safetensors` under that directory therefore signals a
        // genuinely broken artifact (a partial fetch, corruption, an
        // artifact-store/catalog inconsistency) — not "no adapter". The
        // pre-fix code (`.and_then` + `.ok()?`) silently collapsed BOTH a
        // missing-file condition AND a read/parse failure into "serve the
        // unadapted base model", which would drop the fine-tuning entirely
        // with no signal to the caller (K2/K7: an output-affecting file that
        // is expected to be present must fail loudly when it is not, never
        // silently degrade to a different, unrequested model). Audit round
        // 62, F-1's ruling: both conditions are now typed refusals.
        let saved_adapter: Option<(crate::fine_tune::target::SavedAdapter, std::path::PathBuf)> =
            match resolved.adapter_path.as_ref() {
                None => None,
                Some(p) => {
                    let cfg_path = p.join("adapter_config.json");
                    let weights_path = p.join("adapter.safetensors");
                    if !cfg_path.exists() || !weights_path.exists() {
                        return Err(JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!(
                                "resolved adapter_path {p:?} is missing adapter_config.json \
                                 and/or adapter.safetensors — a fine-tuned model's adapter \
                                 directory must carry both files; refusing to silently fall \
                                 back to serving the unadapted base model, which would drop \
                                 the fine-tuning with no signal"
                            ),
                        });
                    }
                    let cfg_str =
                        std::fs::read_to_string(&cfg_path).map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("failed to read {cfg_path:?}: {e}"),
                        })?;
                    let saved: crate::fine_tune::target::SavedAdapter =
                        serde_json::from_str(&cfg_str).map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("failed to parse {cfg_path:?}: {e}"),
                        })?;
                    Some((saved, weights_path))
                }
            };

        let encoder_adapter = saved_adapter.as_ref().and_then(|(saved, weights)| {
            if let crate::fine_tune::target::SavedAdapter::EncoderAdapters(cfg) = saved {
                Some(((**cfg).clone(), weights.as_path()))
            } else {
                None
            }
        });
        // Adapter IDENTITY validation (issue #421 D9), at the one seam where
        // the saved adapter is read.
        //
        // Two questions, both answered FAMILY-to-FAMILY rather than
        // string-to-string:
        //
        // 1. Does this adapter belong to this base at all? A `clap_audio_model`
        //    adapter installed on an OpenCLIP checkpoint, or an `open_clip`
        //    adapter on a BERT one, would previously have been resolved and
        //    then silently DISCARDED — the fine-tuning dropped with no signal,
        //    serving the unadapted base under the fine-tuned model's id.
        //    `EncoderFamily::from_adapter_model_type` maps every legacy/alias
        //    text id onto `Bert` on purpose (see its doc), so this refusal
        //    fires only on genuine cross-family mismatches and never on an
        //    already-shipped BERT-family adapter.
        //
        // 2. Does the base actually HAVE the tower the adapter names? A
        //    `vision` adapter on a single-tower text checkpoint has nowhere to
        //    install. `tower: None` (every adapter written before the field
        //    existed) is accepted everywhere; the multi-tower arms below
        //    additionally require it to be present, because "which of two
        //    towers" has no defensible default.
        if let Some((cfg, _)) = &encoder_adapter {
            // `None` means the architecture itself is one this backend has no
            // loader for; the branch below refuses by name, which is strictly
            // more informative than anything this check could add.
            if let Some(base) = base_family {
                let adapter_family = EncoderFamily::from_adapter_model_type(&cfg.model_type);
                if adapter_family != base {
                    return Err(JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!(
                            "saved adapter targets architecture family {adapter_family:?} \
                             (adapter_config.json model_type '{}') but the base model is \
                             {base:?} (model_type '{model_type}'); refusing to load an \
                             adapter trained on a different architecture",
                            cfg.model_type
                        ),
                    });
                }
                if !base.has_tower(cfg.tower) {
                    return Err(JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!(
                            "saved adapter names tower {:?}, which a {base:?} base model does \
                             not have (its towers: {})",
                            cfg.tower,
                            base.towers()
                        ),
                    });
                }
            }
        }
        // Which tower of a multi-tower checkpoint the adapter installs on.
        // `None` here means either "no adapter at all" or "an adapter that
        // did not say"; the multi-tower arms below distinguish the two.
        let adapter_tower: Option<jammi_lora::Tower> =
            encoder_adapter.as_ref().and_then(|(cfg, _)| cfg.tower);
        let has_encoder_adapter = encoder_adapter.is_some();

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
        // For a GGUF load this points at the SAME synthesized densified
        // file `vb` above reads (`vb_weights_paths`'s own doc) — every
        // `*Builder::build` call below constructs its OWN `frozen_vb` from
        // this path independently of `vb`, so it must resolve to a real
        // safetensors file too, never `resolved.weights_paths`'s raw
        // `model.gguf`.
        let weights_paths_ref: Vec<&std::path::Path> =
            vb_weights_paths.iter().map(|p| p.as_path()).collect();
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
            // and `audio_projection.*`, so the tower loads from the root
            // VarBuilder (the builder scopes the same way).
            //
            // With a saved adapter, the tower is built through the SAME
            // builder the BERT-family arms use — LoRA sites wrapped, A/B
            // tensors read from `adapter.safetensors`, backbone materialised
            // at the adapter's OWN persisted `backbone_dtype` (a fine-tuned
            // model's backbone precision is its adapter's). With NO adapter
            // the historical `load(vb, ..)` call is kept verbatim, so an
            // unadapted base model's served bytes are unchanged (K4).
            let audio_inner = if has_encoder_adapter {
                HtsatAudio::builder()
                    .lora(lora_build)
                    .backbone_dtype(encoder_backbone_dtype)
                    .adapter(encoder_adapter_file)
                    .build(&weights_paths_ref, &audio_config, &device, &dummy_varmap)
                    .map_err(|e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!(
                            "Failed to construct LoRA-adapted HTSAT-Swin CLAP audio tower: {e}"
                        ),
                    })?
            } else {
                HtsatAudio::load(vb.clone(), &audio_config, &device).map_err(|e| {
                    JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to construct HTSAT-Swin CLAP audio tower: {e}"),
                    }
                })?
            };
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
            let text_config = ClipTextConfig::from_open_clip_config(&resolved.model_config)
                .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to parse OpenCLIP text config: {e}"),
                })?;

            // An OpenCLIP checkpoint carries TWO independently adaptable
            // towers under one architecture id, so `adapter_cfg.tower`
            // decides which one receives the LoRA; the sibling is built
            // frozen. Both are built at the SAME `encoder_backbone_dtype` —
            // one model, one identity, one precision, the rule the
            // BERT-family arms already follow — because that precision is
            // folded into `ModelIdentity` and a checkpoint cannot honestly
            // report two.
            //
            // With NO adapter both towers keep today's root-`vb`-at-
            // `compute_dtype` construction verbatim, so an unadapted
            // OpenCLIP base's served bytes are unchanged (K4).
            let (vision_inner, text_inner) = if has_encoder_adapter {
                let build_vision =
                    |lora: jammi_lora::LoraBuildConfig<'_>, adapter: Option<&std::path::Path>| {
                        OpenClipVisionTransformer::builder()
                            .lora(lora)
                            .backbone_dtype(encoder_backbone_dtype)
                            .adapter(adapter)
                            .build(&weights_paths_ref, &vision_config, &device, &dummy_varmap)
                            .map_err(|e| JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to construct OpenCLIP ViT: {e}"),
                            })
                    };
                let build_text =
                    |lora: jammi_lora::LoraBuildConfig<'_>, adapter: Option<&std::path::Path>| {
                        ClipText::builder()
                            .lora(lora)
                            .backbone_dtype(encoder_backbone_dtype)
                            .adapter(adapter)
                            .build(&weights_paths_ref, &text_config, &device, &dummy_varmap)
                            .map_err(|e| JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to construct OpenCLIP text tower: {e}"),
                            })
                    };
                match adapter_tower {
                    Some(jammi_lora::Tower::Vision) => (
                        build_vision(lora_build, encoder_adapter_file)?,
                        build_text(jammi_lora::LoraBuildConfig::frozen(), None)?,
                    ),
                    Some(jammi_lora::Tower::Text) => (
                        build_vision(jammi_lora::LoraBuildConfig::frozen(), None)?,
                        build_text(lora_build, encoder_adapter_file)?,
                    ),
                    // `Some(Audio)` was already refused by the tower check at
                    // the adapter seam; `None` is an adapter that names no
                    // tower on a checkpoint that HAS two — there is no
                    // defensible default, and guessing would install the
                    // weights on the wrong tower.
                    other => {
                        return Err(JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!(
                                "saved adapter for an OpenCLIP checkpoint must name which \
                                 tower it installs on (\"text\" or \"vision\") in \
                                 adapter_config.json; found {other:?}"
                            ),
                        })
                    }
                }
            } else {
                (
                    OpenClipVisionTransformer::load(vb.pp("visual"), &vision_config).map_err(
                        |e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct OpenCLIP ViT: {e}"),
                        },
                    )?,
                    ClipText::load(vb.clone(), &text_config).map_err(|e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to construct OpenCLIP text tower: {e}"),
                    })?,
                )
            };

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
                    let mut distilbert_builder = DistilBert::builder()
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file);
                    if let Some(lookup) = &gguf_lookup {
                        distilbert_builder = distilbert_builder.weight_source(lookup);
                    }
                    let distilbert = distilbert_builder
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
                    let mut distilbert_builder = DistilBert::builder()
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file);
                    if let Some(lookup) = &gguf_lookup {
                        distilbert_builder = distilbert_builder.weight_source(lookup);
                    }
                    let model = distilbert_builder
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
                    let mut bert_builder = Bert::builder()
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file);
                    if let Some(lookup) = &gguf_lookup {
                        bert_builder = bert_builder.weight_source(lookup);
                    }
                    let bert = bert_builder
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
                    let mut bert_builder = Bert::builder()
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file);
                    if let Some(lookup) = &gguf_lookup {
                        bert_builder = bert_builder.weight_source(lookup);
                    }
                    let bert = bert_builder
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
                    let mut modernbert_builder = ModernBert::builder()
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file);
                    if let Some(lookup) = &gguf_lookup {
                        modernbert_builder = modernbert_builder.weight_source(lookup);
                    }
                    let backbone = modernbert_builder
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
                    let mut modernbert_builder = ModernBert::builder()
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file);
                    if let Some(lookup) = &gguf_lookup {
                        modernbert_builder = modernbert_builder.weight_source(lookup);
                    }
                    let model = modernbert_builder
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
                            cfg.use_rslora,
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
                            cfg.use_rslora,
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
                    &vb_weights_paths,
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
            content_digest,
            fingerprint,
            quantization: gguf_quantization,
        })))
    }

    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize {
        // GGUF (issue #351, pin V5) and safetensors (issue #431): the
        // resolver already computed a conservative, header-parsed residency
        // figure at resolve time (`gguf::estimate_gguf_residency` /
        // `safetensors_residency::estimate_safetensors_residency`) — reuse
        // it verbatim rather than re-deriving anything from `weights_paths`
        // here. For GGUF this is because `weights_paths` is a single-file
        // byte size wildly unrepresentative of resident memory; for
        // safetensors it is because the plain on-disk file-byte sum is
        // dtype-blind — it under-reports true residency whenever the
        // on-disk dtype is narrower than the `compute_dtype`
        // `VarBuilder::from_mmaped_safetensors` (below) actually
        // materializes every weight at (issue #431). The ONNX arm below
        // (the only `WeightsFormat` a Candle-backend resolve never
        // produces, but kept as an exhaustive-match-safe fallback) stays
        // the plain file-byte sum, byte-identical to every prior release.
        match resolved.weights_format {
            WeightsFormat::Gguf | WeightsFormat::Safetensors => resolved.estimated_memory,
            WeightsFormat::Onnx => resolved
                .weights_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() as usize)
                .sum(),
        }
    }
}

/// Detect an HF-CLAP audio checkpoint (`ClapAudioModelWithProjection`
/// lineage) from its config.
///
/// The RULES moved to [`EncoderFamily::from_config`] (issue #421 D7) — this
/// is the one-line shim the digest-slot predicate below still reads through,
/// so there is exactly one CLAP-detection body in the crate.
fn is_hf_clap_config(config: &serde_json::Value) -> bool {
    EncoderFamily::from_config(config) == Some(EncoderFamily::ClapAudio)
}

/// Whether `preprocessor_config.json` is a REQUIRED digest candidate for
/// `resolved`, or an OPTIONAL one with a correctness-preserving fallback
/// (audit round 62, F-B).
///
/// `all_candidate_paths` previously marked this slot `optional: true`
/// unconditionally, but [`CandleBackend::load`]'s audio branch
/// (`audio_frontend`'s construction, gated on `audio.is_some()`) hard-refuses
/// an HF-CLAP audio model with no `preprocessor_config.json` — there is no
/// fallback on that path. A deleted `preprocessor_config.json` on a live
/// CLAP model therefore mis-labels [`ModelFingerprint::probe`]'s arm (c)
/// ("required candidate vanished — typed refusal, no fallback exists") as
/// arm (b) ("optional candidate vanished — stale, a reload legitimately
/// succeeds via the fallback"): the probe reports `Ok(false)` (stale), the
/// cache evicts and reloads, and the reload hits `CandleBackend::load`'s
/// hard error instead of the honest typed refusal `probe` should have
/// produced directly.
///
/// The fix: derive `optional` from the candidate's OWN loader predicate —
/// the resolved model's CLASS, using the identical structural signal
/// [`CandleBackend::load`] itself branches the audio path on
/// ([`is_hf_clap_config`] on `resolved.model_config`), not a hardcoded
/// model-name match. Every non-CLAP-audio class (BERT-family text towers,
/// OpenCLIP, classification/NER heads) keeps the pre-existing
/// mean/absent-geometry fallback and stays optional.
fn preprocessor_config_is_required(resolved: &ResolvedModel) -> bool {
    is_hf_clap_config(&resolved.model_config)
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

/// Load the projection head from `adapter_file` using the alpha (and
/// rSLoRA flag) recorded in the adapter's saved config — `use_rslora` must
/// be the persisted training-time choice
/// ([`crate::fine_tune::target::ProjectionHeadConfig::use_rslora`]), never a
/// fixed default: rSLoRA scales by `alpha / sqrt(rank)` instead of `alpha /
/// rank`, so serving a rSLoRA-trained adapter at the vanilla scaling
/// silently shrinks its contribution by `1/sqrt(rank)` (esc-041). Returns
/// `Some(LoraLinear)` keyed at `projection.lora_a` / `projection.lora_b`, or
/// `None` if the projection keys are absent (the adapter was a
/// classifier/NER head with no embedding projection — that case does not
/// produce a post-pool projection).
fn load_projection_head(
    adapter_file: &std::path::Path,
    lora_alpha: f64,
    use_rslora: bool,
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

    let lora =
        jammi_lora::LoraLinear::from_loaded(base_linear, lora_a, lora_b, lora_alpha, use_rslora)
            .map_err(|e| JammiError::Model {
                model_id: model_id.to_string(),
                message: format!("projection LoRA scaling: {e}"),
            })?;
    Ok(Some(lora))
}

/// Reload the regression `distribution` head layer — the `hidden → output_dim`
/// LoRA layer that maps the pooled encoder embedding to the raw distribution
/// parameters (`(mean, raw_std)` for Gaussian, one per level for quantile). This
/// is the layer the trainer's `regress` applies to the pooled embedding (it uses
/// `head.layers[1]`, the distribution layer — NOT the `projection` layer), so
/// serving must apply the same one to reproduce the trained output shape.
///
/// `use_rslora` must be the persisted training-time choice — see
/// [`load_projection_head`]'s doc for why a fixed default silently mis-scales
/// a rSLoRA-trained adapter (esc-041).
///
/// Its zeros base spans `output_dim → hidden_size`; `output_dim` is recovered
/// from the persisted `distribution.lora_b` row count (B is `output_dim × rank`),
/// so the served head width matches the trained head without re-deriving it from
/// the form. Returns `None` when the adapter carries no `distribution` layer
/// (i.e. it is not a regression head).
fn load_distribution_head(
    adapter_file: &std::path::Path,
    lora_alpha: f64,
    use_rslora: bool,
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

    let lora =
        jammi_lora::LoraLinear::from_loaded(base_linear, lora_a, lora_b, lora_alpha, use_rslora)
            .map_err(|e| JammiError::Model {
                model_id: model_id.to_string(),
                message: format!("distribution LoRA scaling: {e}"),
            })?;
    Ok(Some(lora))
}

/// A panic payload folded to a human-readable string. `std::panic::
/// catch_unwind`'s error type is `Box<dyn std::any::Any + Send>`, which
/// carries no `Display`/`Debug` of its own; `panic!("{msg}")`/`panic!(msg)`
/// (the two shapes `std`'s own panic machinery ever constructs) box either a
/// `&'static str` or an owned `String`, so those two downcasts cover every
/// REAL panic payload this process can produce. Mirrors
/// `crates/jammi-kernels/tests/metal_parity.rs`'s helper of the same name.
///
/// Gated like [`check_driver_floor`]: without an accelerator feature (or
/// `test`, which exercises this via [`acquire_accelerator_device`]) nothing
/// in the lib calls this, so it would otherwise be dead code in a CPU-only
/// build.
#[cfg(any(feature = "cuda", feature = "metal", test))]
fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Acquire an accelerator device through `ctor`, folding BOTH a returned
/// `Err` and a caught PANIC into the same `None` ("unavailable") outcome.
/// [`select_device`] treats every `None` from this function identically,
/// regardless of which failure shape produced it: it moves on to the next
/// backend, or — once none are left — calls [`gpu_unavailable`], which
/// degrades to CPU with a loud warning by default or returns a typed
/// [`JammiError::Gpu`] under `require_gpu`.
///
/// This exists because `Device::new_metal` does not always keep its `Result`
/// contract: on at least one real GH `macos-14` runner (and, per this fix's
/// own investigation, any macOS-14 host — a supported shipped configuration)
/// it PANICS instead of returning `Err`, inside an `objc2` class lookup in
/// candle-metal-kernels' `residency_set.rs` for `MTLResidencySetDescriptor`
/// — a class that only exists on macOS 15+. A bare
/// `if let Ok(dev) = Device::new_metal(...)` cannot model that failure mode:
/// the unwind would escape `select_device` entirely, skipping BOTH of its
/// documented degrade arms.
///
/// Wrapping the call in `catch_unwind` is sound at this exact call site: the
/// probe owns no lock and mutates no shared/static state before it can fail
/// (`ctor` is a plain constructor call, not a critical section), so
/// unwinding out of it leaves nothing poisoned to clean up — unlike
/// catching a panic across a held mutex guard or a half-mutated `static`.
/// Mirrors the test-side mechanism in
/// `crates/jammi-kernels/tests/metal_parity.rs`'s `metal_device_or_skip`
/// (added at 29e8b569), which found this exact panic on a real `macos-14`
/// runner.
///
/// `ctor` need not be `UnwindSafe` itself — `Device::new_cuda` /
/// `Device::new_metal` capture nothing and trivially are, but requiring the
/// bound would leak into every call site including test-injected closures.
/// `AssertUnwindSafe` is sound here for the same no-shared-mutable-state
/// reason documented above.
///
/// Gated like [`check_driver_floor`]: without an accelerator feature (or
/// `test`) nothing in the lib calls this, so it would otherwise be dead
/// code in a CPU-only build.
#[cfg(any(feature = "cuda", feature = "metal", test))]
fn acquire_accelerator_device(
    backend: &str,
    ordinal: usize,
    ctor: impl FnOnce(usize) -> std::result::Result<Device, candle_core::Error>,
) -> Option<Device> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ctor(ordinal))) {
        Ok(Ok(dev)) => Some(dev),
        Ok(Err(_)) => None,
        Err(payload) => {
            let panic_message = panic_payload_to_string(payload.as_ref());
            tracing::warn!(
                backend,
                ordinal,
                panic = %panic_message,
                "{backend} device {ordinal} acquisition PANICKED instead of returning Err \
                 (known mechanism on some hosts: an objc2 MTLResidencySetDescriptor class \
                 lookup inside candle-metal-kernels that only resolves on macOS 15+); treating \
                 the device as unavailable rather than letting the panic escape"
            );
            None
        }
    }
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
///   build, the device fails to initialize, or acquisition PANICS (see
///   [`acquire_accelerator_device`]) — the outcome depends on `require_gpu`:
///   by default it degrades to CPU with a loud warning; if `require_gpu` is
///   set it returns a [`JammiError::Gpu`] so the server fails fast rather
///   than silently serving on CPU.
///
/// Device acquisition goes through [`acquire_accelerator_device`], which
/// wraps `Device::new_cuda` / `Device::new_metal` in `catch_unwind`: a
/// PANICKING acquisition (measured on real `macos-14` hosts — see that
/// function's doc) is folded into the SAME unavailable outcome as a returned
/// `Err`, so both of the arms above are reachable no matter which failure
/// shape the underlying constructor produces. It never papers over a panic
/// by swallowing it silently — every caught panic is logged with its
/// payload before falling through to the documented degrade behavior.
pub(crate) fn select_device(config: &DeviceConfig) -> Result<Device> {
    if config.gpu_device < 0 {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(dev) =
            acquire_accelerator_device("cuda", config.gpu_device as usize, Device::new_cuda)
        {
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
        if let Some(dev) =
            acquire_accelerator_device("metal", config.gpu_device as usize, Device::new_metal)
        {
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
///
/// It calls [`select_device`] directly rather than re-deriving device
/// identity from `config`, so it inherits [`select_device`]'s panic-safe
/// acquisition ([`acquire_accelerator_device`]) for free: a host on which
/// `Device::new_metal`/`Device::new_cuda` panics resolves here the same way
/// it resolves in `select_device` — CPU (via the `_` arm below), never a
/// propagated panic — with no separate handling needed at this call site.
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

/// esc-041: `load_projection_head` must apply the *persisted* rSLoRA choice,
/// not a fixed default. Every quantity here is measured through the
/// production `load_projection_head` — writing a real adapter.safetensors to
/// a tempdir, loading it, and reading the actual forward-pass magnitude —
/// never asserted against a hand-derived formula alone.
#[cfg(test)]
mod projection_head_rslora_tests {
    use super::*;
    use candle_core::safetensors;
    use std::collections::HashMap;

    /// Build a one-`projection`-layer adapter file with `lora_a`/`lora_b`
    /// filled with ones (so the LoRA delta on an all-ones input is a known,
    /// nonzero constant — `hidden * rank`), load it through the production
    /// `load_projection_head` with the given `use_rslora`, and return the
    /// scaling factor `load_projection_head` actually applied (back-derived
    /// from the base-identity forward-pass delta, not asserted directly).
    fn served_scaling(rank: usize, alpha: f64, use_rslora: bool) -> f64 {
        let device = Device::Cpu;
        let hidden = 4usize;
        let lora_a = Tensor::ones((rank, hidden), DType::F32, &device).unwrap();
        let lora_b = Tensor::ones((hidden, rank), DType::F32, &device).unwrap();
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("projection.lora_a".to_string(), lora_a);
        weights.insert("projection.lora_b".to_string(), lora_b);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");
        safetensors::save(&weights, &path).unwrap();

        let dims = crate::model::ModelDimensions {
            hidden_size: hidden,
            num_layers: 1,
            num_attention_heads: 1,
            intermediate_size: hidden,
        };
        let lora = load_projection_head(&path, alpha, use_rslora, &device, &dims, "esc-041-test")
            .unwrap()
            .expect("adapter carries a projection layer");
        let x = Tensor::ones((1, hidden), DType::F32, &device).unwrap();
        let out = lora.forward(&x).unwrap();
        let out_v: Vec<Vec<f32>> = out.to_vec2().unwrap();
        // base is identity, so `out - x = scaling * (hidden * rank)` (see
        // this fn's doc for the ones-filled A/B derivation).
        let delta = out_v[0][0] - 1.0;
        (delta / (hidden as f32 * rank as f32)) as f64
    }

    /// esc-041 (a) anti-vacuity, (b) finiteness, (c) two-sided ratio: at both
    /// r = 16 and r = 4, serving with `use_rslora = false` (what
    /// `load_projection_head` did unconditionally before this fix) applies
    /// only `1/sqrt(rank)` of the true rSLoRA-trained scaling — `1/4` at
    /// r = 16, `1/2` at r = 4 — while serving with the *persisted*
    /// `use_rslora = true` matches the trained scaling exactly (ratio
    /// `1.0`). The two served values are also required to differ (anti-
    /// vacuity: the parameter has a real, non-trivial effect), and both to
    /// be finite.
    #[test]
    fn served_scaling_ratio_matches_persisted_rslora_flag() {
        let alpha = 8.0;
        for (rank, expected_naive_ratio) in [(16usize, 0.25), (4usize, 0.5)] {
            let true_scaling = alpha / (rank as f64).sqrt();

            let naive = served_scaling(rank, alpha, false);
            let fixed = served_scaling(rank, alpha, true);

            assert!(naive.is_finite(), "naive-path scaling must be finite");
            assert!(fixed.is_finite(), "fixed-path scaling must be finite");
            assert!(
                (naive - fixed).abs() > 1e-9,
                "use_rslora must have a non-vacuous effect: naive={naive}, fixed={fixed}"
            );

            let naive_ratio = naive / true_scaling;
            let fixed_ratio = fixed / true_scaling;
            assert!(
                (naive_ratio - expected_naive_ratio).abs() < 1e-9,
                "rank={rank}: serving without the persisted rSLoRA flag should apply \
                 1/sqrt(rank) = {expected_naive_ratio} of the trained scaling, got ratio \
                 {naive_ratio}"
            );
            assert!(
                (fixed_ratio - 1.0).abs() < 1e-9,
                "rank={rank}: serving with the persisted rSLoRA flag must match the trained \
                 scaling exactly (ratio 1.0), got {fixed_ratio}"
            );
        }
    }
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

    /// An acquisition "constructor" that always reports the device as
    /// unavailable — via a plain `Err`, never a panic. Used below to reach
    /// `select_device`'s NO-USABLE-GPU arm deterministically, through the
    /// same [`acquire_accelerator_device`] seam production uses, instead of
    /// depending on the host actually lacking a GPU.
    ///
    /// Before this seam existed, `require_gpu_without_device_fails_fast` and
    /// `default_without_device_falls_back_to_cpu_with_warning` called
    /// `select_device` directly and skipped on a CUDA-capable host (the
    /// no-usable-GPU arm is unreachable there — selection correctly serves
    /// the GPU). That skip only covered CUDA: on a real-Metal host (e.g.
    /// this crate's own `--features metal,local` CI lane and any macOS dev
    /// machine with a physical GPU) `select_device` likewise legitimately
    /// acquires the real Metal device, and the old tests failed outright
    /// ("expected CPU fallback, got Metal(..)") rather than skipping. Both
    /// tests now inject `unavailable_ctor` through
    /// `acquire_accelerator_device` — the exact function `select_device`'s
    /// `cuda`/`metal` branches call — so they assert the SAME
    /// `None` → [`gpu_unavailable`] fallback logic `select_device` runs,
    /// without depending on what accelerator (if any) the test host has.
    fn unavailable_ctor(_ordinal: usize) -> std::result::Result<Device, candle_core::Error> {
        Err(candle_core::Error::Msg(
            "no device (test seam: unavailable_ctor)".to_string(),
        ))
    }

    /// An acquisition "constructor" that always PANICS — the failure mode
    /// `Device::new_metal` was measured to hit on at least one real
    /// `macos-14` runner (`acquire_accelerator_device`'s doc). Used to prove
    /// the panic-catching arm is reachable: without `catch_unwind` in
    /// `acquire_accelerator_device`, calling this would abort the test
    /// process instead of yielding `None`.
    fn panicking_ctor(_ordinal: usize) -> std::result::Result<Device, candle_core::Error> {
        panic!("simulated Device::new_metal panic (objc2 MTLResidencySetDescriptor lookup)");
    }

    #[test]
    fn require_gpu_without_device_fails_fast() {
        let acquired = acquire_accelerator_device("test", 0, unavailable_ctor);
        assert!(
            acquired.is_none(),
            "an Err-returning ctor must fold to None"
        );
        // On a host with no usable GPU (and on the default non-accelerator
        // build), selection must surface a typed GPU error rather than serving
        // on CPU.
        match gpu_unavailable(0, true) {
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
            let acquired = acquire_accelerator_device("test", 0, unavailable_ctor);
            assert!(
                acquired.is_none(),
                "an Err-returning ctor must fold to None"
            );
            gpu_unavailable(0, false)
        })
        .expect("default fallback must not error");

        assert!(
            matches!(device, Device::Cpu),
            "expected CPU fallback, got {device:?}"
        );
        assert!(
            flag.load(Ordering::SeqCst),
            "expected a loud warn-level CPU-fallback log"
        );
    }

    /// REACHABLE-proof (round-2 audit BLOCK, item 4): a PANICKING acquisition
    /// — the exact shape `Device::new_metal` was measured to produce on a
    /// real `macos-14` runner — must fold to `None`, not propagate the
    /// panic and abort the test process. This is the direct evidence that
    /// `acquire_accelerator_device`'s `catch_unwind` wrapper actually runs
    /// on the panicking path, independent of any particular host's
    /// hardware.
    #[test]
    fn acquire_accelerator_device_catches_panic_and_returns_none() {
        let acquired = acquire_accelerator_device("metal", 0, panicking_ctor);
        assert!(
            acquired.is_none(),
            "a caught panic must fold to None, exactly like an Err ctor — never propagate"
        );
    }

    /// REACHABLE-proof, degrade arm: composes the SAME two calls
    /// `select_device`'s `metal`/`cuda` branches make — [`acquire_accelerator_device`]
    /// then, on `None`, [`gpu_unavailable`] — with a PANICKING ctor, and
    /// proves the result is the documented default degrade-to-CPU-with-
    /// warning outcome, not a crashed test process.
    #[test]
    fn acquisition_panic_degrades_to_cpu_with_warning() {
        let capture = WarnCapture::default();
        let flag = Arc::clone(&capture.saw_fallback_warning);

        let device = tracing::subscriber::with_default(capture, || {
            let acquired = acquire_accelerator_device("metal", 0, panicking_ctor);
            assert!(acquired.is_none(), "a caught panic must fold to None");
            gpu_unavailable(0, false)
        })
        .expect("default fallback must not error even when acquisition panicked");

        assert!(
            matches!(device, Device::Cpu),
            "expected CPU fallback after a caught panic, got {device:?}"
        );
        assert!(
            flag.load(Ordering::SeqCst),
            "expected a loud warn-level CPU-fallback log after a caught panic"
        );
    }

    /// REACHABLE-proof, `require_gpu` arm: the same composition as above,
    /// but with `require_gpu = true` — proves a panicking acquisition still
    /// surfaces the typed [`JammiError::Gpu`] refusal rather than either
    /// crashing the process or silently serving CPU.
    #[test]
    fn acquisition_panic_with_require_gpu_fails_typed() {
        let acquired = acquire_accelerator_device("metal", 0, panicking_ctor);
        assert!(acquired.is_none(), "a caught panic must fold to None");
        match gpu_unavailable(0, true) {
            Err(JammiError::Gpu(msg)) => {
                assert!(msg.contains("GPU required"), "unexpected message: {msg}");
            }
            other => panic!(
                "expected JammiError::Gpu after a caught panic under require_gpu, got {other:?}"
            ),
        }
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
    ///
    /// **Pairing-discipline exemption (audit round 62, adversarial round 10
    /// advisory fold)**: this stub overrides neither `forward_pooled` nor
    /// `resolved_pooling`, so it inherits BOTH trait defaults together — the
    /// exact combination [`CandleTextForward::forward_pooled`]'s own pairing
    /// rule (audit round 62 advisory A1) warns never to inherit silently for
    /// a wrapper reachable via `ModelTask::TextEmbedding`, since the default
    /// `forward_pooled` performs REAL mean pooling while the default
    /// `resolved_pooling` dishonestly reports `None`. This module below only
    /// ever drives this stub through `ModelTask::Ner`
    /// (`model.forward(&content, ModelTask::Ner)`, both tests) — `forward_pooled`
    /// is never called, and the mismatch this stub's inherited defaults would
    /// otherwise create is unreachable dead weight, not a silent exception to
    /// the pairing rule.
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
            // No real model directory backs this synthetic fixture, so there
            // is nothing to hash — an arbitrary fixed placeholder is fine
            // here (no test in this module asserts on the digest value).
            content_digest: ModelContentDigest::Sha256("test-fixture-digest".into()),
            // No real model directory backs this fixture either, so there is
            // nothing to fingerprint — `empty()` probes vacuously fresh.
            fingerprint: ModelFingerprint::empty(),
            quantization: None,
        }
    }

    fn two_row_content() -> Vec<ArrayRef> {
        vec![Arc::new(StringArray::from(vec!["fine row", "diverged row"])) as ArrayRef]
    }

    /// A diverged model's row (row 1's hidden states are NaN, so its token
    /// logits are non-finite) must refuse the WHOLE call with a typed
    /// `JammiError::Inference` naming the offending row, through the public
    /// `CandleModel::forward` NER entry — never route a non-finite MODEL
    /// output onto the per-row `_status` channel, whose contract
    /// (`InferenceRunner::run_chunks` in runner.rs) reserves `_status =
    /// error` for a bad row INPUT, never "the model is broken". The prior
    /// `Err` arm here set `row_status[orig_idx] = false` and left that row's
    /// `all_entities_json` entry as an empty string (never `entities =
    /// "[]"` — only the `Ok` arm writes that), which `NerAdapter` (via
    /// `nullify_strings`) turns into a SQL NULL for the row regardless of
    /// the string's contents — the same per-row-input-fault treatment a
    /// genuinely malformed row's text gets, not a signal that the model
    /// itself is broken. The invariant this test pins: a model-side
    /// non-finite row is a batch-level typed failure, never a per-row input
    /// error.
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

/// Audit round 62 (F-1, F-4b, and the strip-prefix-refusal advisory):
/// unit-level proofs for mechanisms a full `ModelCache` /
/// `InferenceSession` round-trip cannot reach directly.
///
/// The adapter-appearance (F-4b) and missing-adapter-files (F-1 ruling)
/// scenarios specifically CANNOT be exercised through the public
/// `ModelCache` today: `ModelResolver` only ever sets `adapter_path: Some`
/// via the fine-tuned-model catalog-lookup path (a full fine-tune
/// round-trip), and — after this round's F-1 fix — `CandleBackend::load`
/// now hard-errors the instant `adapter_path` is `Some` with either file
/// missing, so no cached/served `CandleModel` can ever exist in an
/// "adapter_path Some, files absent" state to probe appearance against.
/// Testing `compute_model_fingerprint` / `ModelFingerprint::probe` directly
/// proves the mechanism is correct and general (the honest, function-level
/// proof) rather than forcing an end-to-end scenario the production wiring
/// no longer permits.
#[cfg(test)]
mod digest_fingerprint_audit62_tests {
    use super::*;
    use std::collections::HashMap;

    /// Copy the hermetic `tiny_bert` fixture's config/weights/tokenizer into
    /// `dst` (no `1_Pooling/`, no `preprocessor_config.json`) and return a
    /// `ResolvedModel` pointing at it, with `adapter_path` set to whatever
    /// the caller passes. Hand-built rather than routed through
    /// `ModelResolver` — every `ResolvedModel` field is `pub`, and this is
    /// the same "resolved, not yet loaded" contract `ModelResolver::resolve`
    /// itself fulfills; the resolver's OWN local-source path never produces
    /// `adapter_path: Some` (only the fine-tuned-model catalog-lookup path
    /// does), so constructing one directly is the only way to unit-test the
    /// adapter-anchored candidates in isolation.
    fn tiny_bert_resolved(
        dst: &std::path::Path,
        adapter_path: Option<std::path::PathBuf>,
    ) -> ResolvedModel {
        std::fs::create_dir_all(dst).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(name), dst.join(name)).unwrap();
        }
        let model_config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dst.join("config.json")).unwrap()).unwrap();
        ResolvedModel {
            model_id: crate::model::ModelId(format!("local:{}", dst.display())),
            backend: crate::model::BackendType::Candle,
            weights_format: crate::model::WeightsFormat::Safetensors,
            task: ModelTask::TextEmbedding,
            config_path: dst.join("config.json"),
            weights_paths: vec![dst.join("model.safetensors")],
            tokenizer: Some(TokenizerSource::HuggingFaceJson(dst.join("tokenizer.json"))),
            model_config,
            preprocessor_config: None,
            pooling_config: None,
            base_model_id: None,
            adapter_path,
            estimated_memory: 0,
        }
    }

    /// Write a valid `ProjectionHead`-flavoured `adapter_config.json` +
    /// `adapter.safetensors` pair at `dir`. The weights carry a single
    /// `marker` tensor — neither `projection.lora_a`/`lora_b` nor
    /// `distribution.lora_a`/`lora_b`, so `load_projection_head` /
    /// `load_distribution_head` both return `Ok(None)` and
    /// `CandleBackend::load` succeeds as an ordinary (adapter-inert) load —
    /// the adapter's FILES are still digested/fingerprinted, which is all
    /// F-1/F-4b need; a `ProjectionHead` adapter with no matching keys
    /// needs no `target_modules`/LoRA-shape knowledge of `tiny_bert` at all.
    fn write_projection_adapter(dir: &std::path::Path, marker_value: f32) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg_json = serde_json::json!({
            "adapter_type": "projection_head",
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "use_rslora": false,
            "head_layers": []
        });
        std::fs::write(
            dir.join("adapter_config.json"),
            serde_json::to_string(&cfg_json).unwrap(),
        )
        .unwrap();

        let device = Device::Cpu;
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert(
            "marker".to_string(),
            Tensor::new(&[marker_value], &device).unwrap(),
        );
        candle_core::safetensors::save(&weights, dir.join("adapter.safetensors")).unwrap();
    }

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    /// F-1 core: mutating `adapter.safetensors` bytes in place, under a
    /// constant `resolved.model_id` / `adapter_path`, must change
    /// `compute_model_content_digest`'s output — the peer of
    /// `content_digest.rs`'s weights/tokenizer/pooling mutation tests, for
    /// the adapter pair specifically. RED before F-1: `content_digest_entries`
    /// never enumerated the adapter files at all, so this mutation was
    /// invisible to the digest.
    #[test]
    fn adapter_weights_byte_mutation_changes_content_digest() {
        let model_tmp = tempfile::tempdir().unwrap();
        let adapter_tmp = tempfile::tempdir().unwrap();
        write_projection_adapter(adapter_tmp.path(), 1.0);
        let resolved = tiny_bert_resolved(
            &model_tmp.path().join("model"),
            Some(adapter_tmp.path().to_path_buf()),
        );

        let digest_before = compute_model_content_digest(&resolved).unwrap();

        let weights_path = adapter_tmp.path().join("adapter.safetensors");
        let mut bytes = std::fs::read(&weights_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&weights_path, &bytes).unwrap();

        let digest_after = compute_model_content_digest(&resolved).unwrap();

        assert_ne!(
            digest_before, digest_after,
            "mutating adapter.safetensors bytes in place, under a constant \
             adapter_path, must change the content digest (F-1)"
        );
    }

    /// F-1 peer: the same mutation on `adapter_config.json` itself.
    #[test]
    fn adapter_config_byte_mutation_changes_content_digest() {
        let model_tmp = tempfile::tempdir().unwrap();
        let adapter_tmp = tempfile::tempdir().unwrap();
        write_projection_adapter(adapter_tmp.path(), 1.0);
        let resolved = tiny_bert_resolved(
            &model_tmp.path().join("model"),
            Some(adapter_tmp.path().to_path_buf()),
        );

        let digest_before = compute_model_content_digest(&resolved).unwrap();

        let cfg_path = adapter_tmp.path().join("adapter_config.json");
        let mut bytes = std::fs::read(&cfg_path).unwrap();
        bytes.push(b'\n');
        std::fs::write(&cfg_path, &bytes).unwrap();

        let digest_after = compute_model_content_digest(&resolved).unwrap();

        assert_ne!(
            digest_before, digest_after,
            "mutating adapter_config.json bytes in place must change the content \
             digest (F-1)"
        );
    }

    /// Non-vacuous control: an `adapter_path` that is `Some` but whose two
    /// files do not (yet) exist is NOT an error for the DIGEST — the digest
    /// mirrors the loader's own presence gate and simply omits the adapter
    /// pair — even though (per the ruling below) `CandleBackend::load`
    /// itself refuses to load such a model. The two concerns are decoupled
    /// on purpose: enumerating candidates never fails just because a
    /// candidate happens to be absent.
    #[test]
    fn missing_adapter_files_does_not_fail_digest_computation() {
        let model_tmp = tempfile::tempdir().unwrap();
        let adapter_tmp = tempfile::tempdir().unwrap(); // no adapter files written
        let resolved = tiny_bert_resolved(
            &model_tmp.path().join("model"),
            Some(adapter_tmp.path().to_path_buf()),
        );

        let digest = compute_model_content_digest(&resolved);
        assert!(
            digest.is_ok(),
            "an adapter_path with no adapter files yet must not fail digest \
             computation: {digest:?}"
        );
    }

    /// F-1 ruling: `resolved.adapter_path` is `Some` only via the
    /// fine-tuned-model catalog-lookup path, which asserts "this model IS
    /// fine-tuned, its adapter lives here" — so a missing
    /// `adapter_config.json` / `adapter.safetensors` under that directory is
    /// a typed refusal (K2/K7), never a silent fall-back to the unadapted
    /// base model. Both files missing, and each file missing individually,
    /// must all refuse.
    #[test]
    fn candle_backend_load_refuses_some_adapter_path_missing_files() {
        let cases: [&str; 3] = ["neither", "config_only", "weights_only"];
        for case in cases {
            let model_tmp = tempfile::tempdir().unwrap();
            let adapter_tmp = tempfile::tempdir().unwrap();
            match case {
                "config_only" => {
                    write_projection_adapter(adapter_tmp.path(), 1.0);
                    std::fs::remove_file(adapter_tmp.path().join("adapter.safetensors")).unwrap();
                }
                "weights_only" => {
                    write_projection_adapter(adapter_tmp.path(), 1.0);
                    std::fs::remove_file(adapter_tmp.path().join("adapter_config.json")).unwrap();
                }
                _ => {}
            }
            let resolved = tiny_bert_resolved(
                &model_tmp.path().join("model"),
                Some(adapter_tmp.path().to_path_buf()),
            );
            let backend = CandleBackend;
            let result = backend.load(&resolved, &device_config());
            assert!(
                result.is_err(),
                "case {case:?}: adapter_path Some with a missing adapter file must \
                 refuse to load, never silently serve the unadapted base model"
            );
        }
    }

    /// Positive control for the refusal above: BOTH adapter files present
    /// loads successfully — proves the refusal fires on absence
    /// specifically, not on some incidental fixture/setup issue.
    #[test]
    fn candle_backend_load_succeeds_when_both_adapter_files_present() {
        let model_tmp = tempfile::tempdir().unwrap();
        let adapter_tmp = tempfile::tempdir().unwrap();
        write_projection_adapter(adapter_tmp.path(), 1.0);
        let resolved = tiny_bert_resolved(
            &model_tmp.path().join("model"),
            Some(adapter_tmp.path().to_path_buf()),
        );
        let backend = CandleBackend;
        let result = backend.load(&resolved, &device_config());
        assert!(
            result.is_ok(),
            "both adapter files present must load successfully: {:?}",
            result.err()
        );
    }

    /// F-4b: an adapter pair that did not exist when
    /// `compute_model_fingerprint` ran, but exists by the time `probe` is
    /// called, must flip the probe to stale (`Ok(false)`) — the mechanism a
    /// production warm-hit reload depends on to detect a newly-appearing
    /// output-affecting file. Contrasted with a no-op probe (nothing
    /// changed) staying fresh, so this isn't vacuously always-stale.
    #[test]
    fn adapter_files_appearing_after_fingerprint_trips_the_probe() {
        let model_tmp = tempfile::tempdir().unwrap();
        let adapter_tmp = tempfile::tempdir().unwrap(); // absent at fingerprint time
        let resolved = tiny_bert_resolved(
            &model_tmp.path().join("model"),
            Some(adapter_tmp.path().to_path_buf()),
        );

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "nothing changed yet: probe must report fresh"
        );

        write_projection_adapter(adapter_tmp.path(), 1.0);

        assert!(
            !fingerprint.probe().unwrap(),
            "an adapter pair that appeared after the fingerprint was captured must \
             trip the probe to stale (F-4b)"
        );
    }

    /// F-4b peer: `1_Pooling/config.json` appearing after the fingerprint
    /// was captured also trips the probe — proven at the same unit level as
    /// the adapter case above (the end-to-end `ModelCache` peer of this test
    /// lives in `cache_staleness.rs`).
    #[test]
    fn pooling_config_appearing_after_fingerprint_trips_the_probe() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);
        assert!(resolved.pooling_config.is_none());

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(fingerprint.probe().unwrap());

        let pooling_dir = dst.join("1_Pooling");
        std::fs::create_dir_all(&pooling_dir).unwrap();
        std::fs::write(
            pooling_dir.join("config.json"),
            r#"{"pooling_mode_cls_token": true}"#,
        )
        .unwrap();

        assert!(
            !fingerprint.probe().unwrap(),
            "1_Pooling/config.json appearing after the fingerprint was captured \
             must trip the probe to stale (F-4b)"
        );
    }

    /// F-4b peer: `preprocessor_config.json` appearing after the fingerprint
    /// was captured also trips the probe.
    #[test]
    fn preprocessor_config_appearing_after_fingerprint_trips_the_probe() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);
        assert!(resolved.preprocessor_config.is_none());

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(fingerprint.probe().unwrap());

        std::fs::write(
            dst.join("preprocessor_config.json"),
            r#"{"sample_rate": 48000}"#,
        )
        .unwrap();

        assert!(
            !fingerprint.probe().unwrap(),
            "preprocessor_config.json appearing after the fingerprint was captured \
             must trip the probe to stale (F-4b)"
        );
    }

    /// Advisory: a candidate path that cannot be expressed relative to its
    /// own anchor directory (every path this engine resolves today is
    /// always a child of one) is a typed refusal, never a silent fold of an
    /// absolute, host-specific path into a digest documented to be
    /// reproducible across hosts. Simulated with a tokenizer path deliberately
    /// OUTSIDE the model directory — a shape `ModelResolver` never produces
    /// today, proving the refusal is reachable and correctly wired even
    /// though production cannot trigger it.
    #[test]
    fn candidate_outside_its_anchor_is_a_typed_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let outside_tmp = tempfile::tempdir().unwrap();
        let mut resolved = tiny_bert_resolved(&model_tmp.path().join("model"), None);
        // A tokenizer file that exists, but lives OUTSIDE the model
        // directory (`config_path`'s parent) — the unreachable-in-production
        // shape the refusal exists to reject.
        let outside_tokenizer = outside_tmp.path().join("tokenizer.json");
        std::fs::copy(
            jammi_test_utils::cookbook_fixture("tiny_bert").join("tokenizer.json"),
            &outside_tokenizer,
        )
        .unwrap();
        resolved.tokenizer = Some(TokenizerSource::HuggingFaceJson(outside_tokenizer));

        let result = compute_model_content_digest(&resolved);
        assert!(
            result.is_err(),
            "a candidate outside its anchor directory must be a typed refusal, \
             never a silent absolute-path fallback: {result:?}"
        );
    }

    /// Positive control for the refusal above: the SAME resolved model with
    /// its tokenizer back under the model directory succeeds — proves the
    /// refusal fires on the anchor mismatch specifically.
    #[test]
    fn candidate_inside_its_anchor_succeeds() {
        let model_tmp = tempfile::tempdir().unwrap();
        let resolved = tiny_bert_resolved(&model_tmp.path().join("model"), None);
        let result = compute_model_content_digest(&resolved);
        assert!(result.is_ok(), "expected success, got {result:?}");
    }

    // ── F-4' (audit round 62, adversarial round 3): the optional/required \
    //    lattice at the `ModelFingerprint::probe` level ──

    /// F-4' core, unit level (the peer of `cache_staleness.rs`'s integration
    /// test): an OPTIONAL candidate (`1_Pooling/config.json`) present at
    /// fingerprint-capture time and `NotFound` at probe time must report
    /// STALE (`Ok(false)`), never `Err`. RED pre-F-4': `probe()` collapsed
    /// this into `Err` regardless of `optional`.
    #[test]
    fn probe_optional_candidate_deleted_after_capture_is_stale_not_a_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let mut resolved = tiny_bert_resolved(&dst, None);
        std::fs::create_dir_all(dst.join("1_Pooling")).unwrap();
        let pooling_json = serde_json::json!({"pooling_mode_cls_token": true});
        std::fs::write(
            dst.join("1_Pooling/config.json"),
            serde_json::to_string(&pooling_json).unwrap(),
        )
        .unwrap();
        resolved.pooling_config = Some(pooling_json);

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        std::fs::remove_file(dst.join("1_Pooling/config.json")).unwrap();

        let result = fingerprint.probe();
        assert!(
            matches!(result, Ok(false)),
            "an OPTIONAL candidate deleted after the fingerprint was captured \
             must report STALE (Ok(false)) — a reload legitimately succeeds \
             via the mean-pooling fallback — never Err (F-4'), got {result:?}"
        );
    }

    /// F-4' control, unit level: a REQUIRED candidate (`model.safetensors`)
    /// present at capture time and `NotFound` at probe time must remain the
    /// typed refusal (`Err`) — unchanged from before F-4', since no fallback
    /// exists and a reload would fail identically.
    #[test]
    fn probe_required_candidate_deleted_after_capture_stays_a_typed_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        std::fs::remove_file(dst.join("model.safetensors")).unwrap();

        let result = fingerprint.probe();
        assert!(
            result.is_err(),
            "a REQUIRED candidate (model.safetensors) deleted after the \
             fingerprint was captured must remain a typed refusal (F-4') — \
             a reload would fail identically — got {result:?}"
        );
    }

    // ── F-B (audit round 62, adversarial round 4): `preprocessor_config.json` \
    //    is optional PER MODEL CLASS, not per slot ──

    /// [`preprocessor_config_is_required`] unit level: a text-tower class
    /// (a plain BERT `config.json`, the identical shape `tiny_bert_resolved`
    /// produces) has no `model_type`/`architectures` signal that
    /// [`is_hf_clap_config`] recognises, so the slot stays optional — the
    /// pre-existing mean/absent-geometry fallback still applies.
    #[test]
    fn preprocessor_config_is_required_false_for_a_text_tower_class() {
        let model_tmp = tempfile::tempdir().unwrap();
        let resolved = tiny_bert_resolved(&model_tmp.path().join("model"), None);
        assert!(
            !preprocessor_config_is_required(&resolved),
            "a BERT-family text-tower model_config must classify as \
             NOT requiring preprocessor_config.json"
        );
    }

    /// [`preprocessor_config_is_required`] unit level, the other class: an
    /// HF-CLAP audio `model_config` (`model_type == \"clap_audio_model\"`,
    /// the SAME structural signal [`CandleBackend::load`]'s own `is_clap`
    /// branches the audio path on) has no fallback for a missing
    /// `preprocessor_config.json` — the slot must classify as required.
    #[test]
    fn preprocessor_config_is_required_true_for_a_clap_audio_class() {
        let model_tmp = tempfile::tempdir().unwrap();
        let mut resolved = tiny_bert_resolved(&model_tmp.path().join("model"), None);
        resolved.model_config = serde_json::json!({ "model_type": "clap_audio_model" });
        assert!(
            preprocessor_config_is_required(&resolved),
            "an HF-CLAP audio model_config must classify as REQUIRING \
             preprocessor_config.json — CandleBackend::load's audio branch \
             hard-refuses without it, with no fallback"
        );

        // The nested-`audio_config` shape (top-level `ClapConfig`) is the
        // SAME class by the SAME signal `is_hf_clap_config` recognises.
        resolved.model_config = serde_json::json!({
            "audio_config": { "model_type": "clap_audio_model" }
        });
        assert!(
            preprocessor_config_is_required(&resolved),
            "the nested audio_config CLAP shape must also classify as \
             requiring preprocessor_config.json"
        );
    }

    /// End-to-end (F-B, block): reproduces the auditor's exact scenario at
    /// the [`ModelFingerprint::probe`] level, not just the classifier
    /// function in isolation. A CLAP-shaped resolved model has a
    /// `preprocessor_config.json` present at fingerprint-capture time; it is
    /// then deleted from the live model directory (as if a caller pruned
    /// stale artifacts, or the file was overwritten by an in-place update
    /// that failed partway).
    ///
    /// RED pre-fix: `preprocessor_config.json`'s `optional` was fixed
    /// `true` unconditionally, so this candidate always fell into arm (b)
    /// — `probe()` returns `Ok(false)` (stale, "a reload will succeed via
    /// the fallback"). `ModelCache::get_or_load` would evict and reload,
    /// and the reload hits `CandleBackend::load`'s hard error ("CLAP audio
    /// model is missing preprocessor_config.json") instead of the honest
    /// typed refusal `probe` should have surfaced directly.
    ///
    /// GREEN post-fix: `preprocessor_config_is_required` classifies this
    /// resolved model as CLAP audio, so the candidate is `optional: false`
    /// — arm (c) applies and `probe()` returns `Err` directly, matching
    /// what a reload would do anyway, without first pretending the model is
    /// merely stale.
    #[test]
    fn probe_clap_preprocessor_config_deleted_after_capture_is_a_typed_refusal_not_stale() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let mut resolved = tiny_bert_resolved(&dst, None);
        resolved.model_config = serde_json::json!({ "model_type": "clap_audio_model" });
        let preprocessor_json = serde_json::json!({
            "sampling_rate": 48000,
            "hop_length": 480,
            "max_length_s": 10,
        });
        std::fs::write(
            dst.join("preprocessor_config.json"),
            serde_json::to_string(&preprocessor_json).unwrap(),
        )
        .unwrap();
        resolved.preprocessor_config = Some(preprocessor_json);

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        std::fs::remove_file(dst.join("preprocessor_config.json")).unwrap();

        let result = fingerprint.probe();
        assert!(
            result.is_err(),
            "preprocessor_config.json deleted from a live CLAP model must \
             be a typed refusal (F-B): CandleBackend::load has no fallback \
             for a CLAP audio tower missing this file, so a reload would \
             fail identically — reporting STALE (Ok(false)) here would \
             mis-route the caller into an eviction+reload that cannot \
             succeed. Got {result:?}"
        );
    }

    // ── R5-F1 (audit round 62, adversarial round 6): the tokenizer \
    //    candidate is unconditionally OPTIONAL, not over-required ──

    /// R5-F1 (block): reproduces the auditor's exact scenario. `tokenizer.json`
    /// is present at fingerprint-capture time (`tiny_bert_resolved` always
    /// ships one), then deleted from the live model directory — as if a
    /// caller pruned stale artifacts.
    ///
    /// RED pre-fix: the tokenizer candidate's `optional` was fixed `false`
    /// unconditionally ("the loader has no fallback"), which is false for
    /// this slot — every resolver path re-derives `tokenizer: None` on
    /// absence instead of erroring, and `CandleBackend::load`'s
    /// `.transpose()?` accepts `None` outright. `probe()` nonetheless fell
    /// into arm (c) and returned `Err`, and because `ModelCache::get_or_load`
    /// retains the `CacheEntry` on an `Err` probe (never evicts it), every
    /// SUBSEQUENT `get_or_load` on the same id would re-probe the same
    /// vanished file and re-`Err` — permanently wedging a model a cold
    /// process would serve just fine.
    ///
    /// GREEN post-fix: the tokenizer candidate classifies as `optional:
    /// true` — arm (b) applies, `probe()` reports `Ok(false)` (stale), and
    /// the caller evicts + reloads instead of wedging forever.
    #[test]
    fn probe_tokenizer_deleted_after_capture_is_stale_not_a_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        std::fs::remove_file(dst.join("tokenizer.json")).unwrap();

        let result = fingerprint.probe();
        assert!(
            matches!(result, Ok(false)),
            "tokenizer.json deleted from a live model directory must report \
             STALE (Ok(false)) — every resolver path re-derives \
             `tokenizer: None` on absence and `CandleBackend::load` accepts \
             it, so a reload legitimately succeeds mirroring cold-process \
             semantics (R5-F1) — never Err, which would permanently wedge \
             `ModelCache::get_or_load` on this id. Got {result:?}"
        );
    }

    /// R5-F1 peer: two consecutive `probe()` calls after the SAME deletion
    /// must both report stale (never wedge on the second call either) — the
    /// exact "permanently wedges" failure mode the auditor named, expressed
    /// at the fingerprint level (the fingerprint itself is immutable once
    /// captured, so "two consecutive calls" is "the same fingerprint probed
    /// twice"; `ModelCache::get_or_load`'s own two-consecutive-calls
    /// behavior is covered end-to-end in `cache.rs`).
    #[test]
    fn probe_tokenizer_deleted_never_wedges_on_repeated_probes() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        std::fs::remove_file(dst.join("tokenizer.json")).unwrap();

        for attempt in 0..2 {
            let result = fingerprint.probe();
            assert!(
                matches!(result, Ok(false)),
                "repeated probe #{attempt} against the same deleted-tokenizer \
                 fingerprint must keep reporting stale, never wedge into a \
                 permanent Err — got {result:?}"
            );
        }
    }

    /// R5-F1 (digest side): once `tokenizer.json` is gone and the model is
    /// reloaded (simulated here by re-resolving with `tokenizer: None`, the
    /// same shape `discover_local_tokenizer` returns for a tokenizer-less
    /// directory), the content digest must no longer include a tokenizer
    /// record — `content_digest_entries` gates each candidate on the
    /// resolver's own presence signal, so the tokenizer entry leaves the
    /// fold entirely rather than hashing a vanished path.
    #[test]
    fn content_digest_drops_tokenizer_entry_once_resolver_reports_it_absent() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved_with_tokenizer = tiny_bert_resolved(&dst, None);
        let digest_with_tokenizer = compute_model_content_digest(&resolved_with_tokenizer).unwrap();

        std::fs::remove_file(dst.join("tokenizer.json")).unwrap();
        let mut resolved_without_tokenizer = resolved_with_tokenizer;
        resolved_without_tokenizer.tokenizer = None;
        let digest_without_tokenizer =
            compute_model_content_digest(&resolved_without_tokenizer).unwrap();

        assert_ne!(
            digest_with_tokenizer, digest_without_tokenizer,
            "the content digest must change once the resolver stops \
             reporting a tokenizer candidate (R5-F1) — the tokenizer record \
             must leave the fold, not silently keep hashing a path that no \
             longer exists"
        );
    }

    // ── F-4'' (audit round 62, adversarial round 3): fingerprint-before- \
    //    digest ordering ──
    //
    // A literal concurrent-mutation test racing `compute_model_identity_facets`'s
    // own two internal calls is not deterministically constructible: that
    // function is synchronous with no `.await` point to pause at, and is
    // deliberately NOT parameterised with a `#[cfg(test)]` instrumentation
    // seam (adding one would itself be a second, harder-to-review place the
    // order could drift from). Instead, the two tests below reproduce the
    // auditor's exact defect and its fix directly on the two primitives
    // `compute_model_identity_facets` composes, called in each order — the
    // observable CONSEQUENCE of the ordering, which is what actually matters
    // — while `compute_model_identity_facets`'s doc comment pins WHICH order
    // production takes, structurally (see that function for why a future
    // reorder cannot happen silently).

    /// The pre-fix (broken) order: hash first, stat second. A mutation
    /// racing in the window between the two calls makes the fingerprint
    /// attest the POST-mutation state while the digest attests the
    /// PRE-mutation bytes — so `probe()` reports fresh FOREVER (nothing
    /// changes after the fingerprint was captured) despite the digest being
    /// stale. This is the F-4'' defect itself, reproduced directly.
    #[test]
    fn hash_before_stat_would_mask_a_racing_mutation_forever() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);

        // Pre-fix order: hash FIRST.
        let digest_before = compute_model_content_digest(&resolved).unwrap();

        // A mutation races in the window between the hash and the stat.
        let weights_path = dst.join("model.safetensors");
        let mut bytes = std::fs::read(&weights_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&weights_path, &bytes).unwrap();

        // Fingerprint stamped AFTER the mutation: its (len, mtime) matches
        // the POST-mutation file exactly.
        let fingerprint_after = compute_model_fingerprint(&resolved).unwrap();
        let digest_after_mutation = compute_model_content_digest(&resolved).unwrap();
        assert_ne!(
            digest_before, digest_after_mutation,
            "the mutation must actually change the digest — otherwise this \
             test cannot distinguish stale from fresh"
        );

        // THE defect: probe reports fresh forever (nothing changes on disk
        // after this point), even though a process holding `digest_before`
        // would keep attesting a digest that no longer matches the bytes on
        // disk — the "confident wrong number" F-4'' names.
        assert!(
            fingerprint_after.probe().unwrap(),
            "hash-before-stat: the fingerprint captured AFTER the mutation \
             reports fresh — the exact defect a fingerprint-before-digest \
             ordering closes"
        );
    }

    /// The fixed order — what `compute_model_identity_facets` actually does:
    /// stat first, hash second. The identical mutation, raced into the
    /// identical window, is now caught by the VERY NEXT probe: the
    /// fingerprint captured BEFORE the mutation correctly reports stale,
    /// converging on a reload rather than silently attesting a digest the
    /// bytes no longer match.
    #[test]
    fn stat_before_hash_converges_after_a_racing_mutation() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);

        // Fixed order: stat FIRST.
        let fingerprint_before = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint_before.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        let weights_path = dst.join("model.safetensors");
        let mut bytes = std::fs::read(&weights_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&weights_path, &bytes).unwrap();

        let _digest_after = compute_model_content_digest(&resolved).unwrap();

        assert!(
            !fingerprint_before.probe().unwrap(),
            "stat-before-hash: a mutation landing after the fingerprint was \
             captured (but before the digest was hashed) must be caught by \
             the NEXT probe — this order can never leave a fingerprint \
             permanently attesting a stale digest (F-4'')"
        );
    }

    /// Pins the composed function's OUTPUT correctness — its digest matches
    /// an independent manual `compute_model_content_digest` call, and its
    /// fingerprint reports fresh — and its return shape,
    /// `(ModelFingerprint, ModelContentDigest)`.
    ///
    /// **What this test does NOT prove (audit round 62 advisory A1, folded
    /// round 4)**: over an UNMUTATED directory (the only case exercised
    /// here), fingerprint-then-digest and digest-then-fingerprint produce
    /// byte-identical results — there is no racing mutation in this test to
    /// make the two internal calls' ORDER observable, so this assertion
    /// stays green even if `compute_model_identity_facets`'s two-line body
    /// silently swapped `compute_model_fingerprint` and
    /// `compute_model_content_digest`. This test is NOT what pins that
    /// order. Making the order itself observable would require racing a
    /// real mutation into the window between the two internal calls, which
    /// needs a `#[cfg(test)]` pause seam INSIDE
    /// `compute_model_identity_facets` — deliberately not added (see that
    /// function's doc: a second instrumentation seam is itself a second,
    /// harder-to-review place the order could drift from). The actual
    /// guarantees against a silent reorder are: (1) STRUCTURAL —
    /// `compute_model_identity_facets` is the sole production call site
    /// (this file's only other callers of the two primitives are tests),
    /// its doc comment calls out the ordering invariant explicitly, and a
    /// diff touching its two-line body is exactly the size a reviewer can
    /// actually read and hold to that doc; and (2) BEHAVIORAL —
    /// `stat_before_hash_converges_after_a_racing_mutation` (above) proves
    /// WHY stat-first is the safe order, using the two primitives called
    /// manually with a real racing mutation in between, even though it does
    /// not call the composed function itself.
    #[test]
    fn compute_model_identity_facets_matches_manual_stat_then_hash_calls() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_bert_resolved(&dst, None);

        let (fingerprint, digest) = compute_model_identity_facets(&resolved).unwrap();
        let manual_digest = compute_model_content_digest(&resolved).unwrap();
        assert_eq!(
            digest, manual_digest,
            "compute_model_identity_facets's digest must match a manual, \
             independent compute_model_content_digest call over the same \
             (unmutated) directory"
        );
        assert!(
            fingerprint.probe().unwrap(),
            "compute_model_identity_facets's fingerprint must report fresh \
             over the same (unmutated) directory it was captured from"
        );
    }

    // ── Round 10 (audit round 62, adversarial round 10 — "the terminal \
    //    class closure"): the config and weights slots gain the SAME \
    //    appearance / deletion-with-alternate / all-arms-gone lattice the \
    //    tokenizer slot already had since round 8, via the `DigestSlot` \
    //    reshape ──

    /// Build a `ResolvedModel` whose CONFIG SLOT is selected via
    /// `selected_name` (`"config.json"` or `"open_clip_config.json"`) —
    /// writing ONLY that one file initially, so these tests can add or
    /// remove the alternate arm independently. Reuses `tiny_bert`'s
    /// weights/tokenizer files unmodified; only the config slot is under
    /// test here.
    fn resolved_with_config_arm(dst: &std::path::Path, selected_name: &str) -> ResolvedModel {
        std::fs::create_dir_all(dst).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for name in ["model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(name), dst.join(name)).unwrap();
        }
        let config_path = dst.join(selected_name);
        std::fs::copy(fixture.join("config.json"), &config_path).unwrap();
        let model_config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path).unwrap()).unwrap();
        ResolvedModel {
            model_id: crate::model::ModelId(format!("local:{}", dst.display())),
            backend: crate::model::BackendType::Candle,
            weights_format: crate::model::WeightsFormat::Safetensors,
            task: ModelTask::TextEmbedding,
            config_path,
            weights_paths: vec![dst.join("model.safetensors")],
            tokenizer: Some(TokenizerSource::HuggingFaceJson(dst.join("tokenizer.json"))),
            model_config,
            preprocessor_config: None,
            pooling_config: None,
            base_model_id: None,
            adapter_path: None,
            estimated_memory: 0,
        }
    }

    /// (a) appearance: load with ONLY the alternate arm
    /// (`open_clip_config.json`) present; `config.json` — the arm the
    /// resolver's chain checks FIRST — then appears. RED pre-round-10: the
    /// config slot only ever fingerprinted the ONE arm the resolver
    /// selected for this load, so `config.json` appearing was invisible to
    /// `probe`, which reported fresh forever even though a cold resolve
    /// would now prefer it over `open_clip_config.json`.
    #[test]
    fn config_slot_alternate_arm_appearing_trips_the_probe() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_config_arm(&dst, "open_clip_config.json");

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        // config.json — the UNSELECTED arm, never read by this load — APPEARS.
        std::fs::copy(
            jammi_test_utils::cookbook_fixture("tiny_bert").join("config.json"),
            dst.join("config.json"),
        )
        .unwrap();

        assert!(
            !fingerprint.probe().unwrap(),
            "config.json appearing alongside a load selected via its alternate arm \
             (open_clip_config.json) must trip the probe to stale — a cold resolve \
             checks config.json FIRST and would now pick it instead (audit round 62, \
             adversarial round 10)"
        );
    }

    /// (b) deletion-with-alternate: BOTH arms exist on disk; the SELECTED
    /// arm (`config.json`) is deleted while the alternate
    /// (`open_clip_config.json`) survives. A cold resolve would succeed via
    /// the alternate, so this must be STALE, never a refusal.
    #[test]
    fn config_slot_selected_arm_deleted_with_alternate_present_is_stale_not_a_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_config_arm(&dst, "config.json");
        // The alternate arm ALSO exists on disk — untracked by the
        // pre-round-10 fingerprint, since only the selected arm was ever a
        // candidate.
        std::fs::copy(
            jammi_test_utils::cookbook_fixture("tiny_bert").join("config.json"),
            dst.join("open_clip_config.json"),
        )
        .unwrap();

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(fingerprint.probe().unwrap());

        std::fs::remove_file(dst.join("config.json")).unwrap();

        let result = fingerprint.probe();
        assert!(
            matches!(result, Ok(false)),
            "deleting the SELECTED config.json while open_clip_config.json (an \
             alternate arm) still exists must report STALE, never a refusal — a \
             cold resolve would succeed via the alternate (audit round 62, \
             adversarial round 10). Got {result:?}"
        );
    }

    /// (c) all-arms-gone on a required slot: NO alternate exists at all —
    /// unchanged from before the round-10 reshape, since a cold resolve
    /// cannot succeed either.
    #[test]
    fn config_slot_all_arms_gone_stays_a_typed_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_config_arm(&dst, "config.json");
        // No open_clip_config.json exists at all.

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        std::fs::remove_file(dst.join("config.json")).unwrap();

        let result = fingerprint.probe();
        assert!(
            result.is_err(),
            "deleting config.json with NO alternate (open_clip_config.json) present \
             must remain a typed refusal — no arm of the config slot can satisfy a \
             cold resolve — unchanged from before the round-10 reshape. Got {result:?}"
        );
    }

    /// Build a `ResolvedModel` whose WEIGHTS SLOT is selected via
    /// `selected_name` (one of the three well-known weight filenames) —
    /// writing ONLY that one file initially. `backend` is set to match
    /// (`Ort` for `"model.onnx"`, `Candle` otherwise), mirroring what
    /// `resolve_local`'s own `has_onnx` branch would have picked, even
    /// though this fixture is hand-built rather than routed through the
    /// resolver.
    fn resolved_with_weights_arm(dst: &std::path::Path, selected_name: &str) -> ResolvedModel {
        std::fs::create_dir_all(dst).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for name in ["config.json", "tokenizer.json"] {
            std::fs::copy(fixture.join(name), dst.join(name)).unwrap();
        }
        let weights_path = dst.join(selected_name);
        std::fs::copy(fixture.join("model.safetensors"), &weights_path).unwrap();
        let model_config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dst.join("config.json")).unwrap()).unwrap();
        let backend = if selected_name == "model.onnx" {
            crate::model::BackendType::Ort
        } else {
            crate::model::BackendType::Candle
        };
        let weights_format = if selected_name == "model.onnx" {
            crate::model::WeightsFormat::Onnx
        } else if selected_name == "model.gguf" {
            crate::model::WeightsFormat::Gguf
        } else {
            crate::model::WeightsFormat::Safetensors
        };
        ResolvedModel {
            model_id: crate::model::ModelId(format!("local:{}", dst.display())),
            backend,
            weights_format,
            task: ModelTask::TextEmbedding,
            config_path: dst.join("config.json"),
            weights_paths: vec![weights_path],
            tokenizer: Some(TokenizerSource::HuggingFaceJson(dst.join("tokenizer.json"))),
            model_config,
            preprocessor_config: None,
            pooling_config: None,
            base_model_id: None,
            adapter_path: None,
            estimated_memory: 0,
        }
    }

    /// (a) appearance: load with ONLY `model.safetensors` selected;
    /// `model.onnx` — an arm this load did NOT select — then appears. RED
    /// pre-round-10: the weights slot only ever fingerprinted the resolved
    /// `weights_paths` entries themselves, never the OTHER well-known
    /// filenames, so `model.onnx` appearing was invisible to `probe` —
    /// silently masking the fact that a cold resolve (`resolve_local`'s
    /// `has_onnx` branch) would now pick the ORT backend instead. The
    /// backend flip itself is a COLD-side property, verified independently
    /// at the resolver level by `models.rs`'s
    /// `resolve_local_prefers_onnx_once_it_appears_alongside_existing_safetensors`
    /// — this test asserts what the warm probe can honestly assert:
    /// staleness was detected.
    #[test]
    fn weights_slot_alternate_arm_appearing_trips_the_probe() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_weights_arm(&dst, "model.safetensors");

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        // model.onnx — the UNSELECTED arm — APPEARS.
        std::fs::write(dst.join("model.onnx"), b"fake-onnx-bytes").unwrap();

        assert!(
            !fingerprint.probe().unwrap(),
            "model.onnx appearing alongside a load resolved via model.safetensors must \
             trip the probe to stale (audit round 62, adversarial round 10) — it is not \
             a candle-side-irrelevant file: resolve_local prefers ORT the instant \
             model.onnx exists"
        );
    }

    /// (b) deletion-with-alternate: BOTH `model.safetensors` (selected) and
    /// `open_clip_model.safetensors` (alternate, untracked pre-round-10)
    /// exist; the selected arm is deleted. A cold resolve's own
    /// standard/open_clip fallback chain would succeed via the alternate, so
    /// this must be STALE, never a refusal.
    #[test]
    fn weights_slot_selected_arm_deleted_with_alternate_present_is_stale_not_a_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_weights_arm(&dst, "model.safetensors");
        std::fs::copy(
            dst.join("model.safetensors"),
            dst.join("open_clip_model.safetensors"),
        )
        .unwrap();

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(fingerprint.probe().unwrap());

        std::fs::remove_file(dst.join("model.safetensors")).unwrap();

        let result = fingerprint.probe();
        assert!(
            matches!(result, Ok(false)),
            "deleting the SELECTED model.safetensors while open_clip_model.safetensors \
             (an alternate arm) still exists must report STALE, never a refusal — a cold \
             resolve would succeed via the alternate (audit round 62, adversarial round \
             10). Got {result:?}"
        );
    }

    /// (c) all-arms-gone on a required slot: NO alternate exists at all —
    /// unchanged from before the round-10 reshape (the peer of
    /// `probe_required_candidate_deleted_after_capture_stays_a_typed_refusal`
    /// above, expressed explicitly against the round-10 slot model with its
    /// own dedicated fixture).
    #[test]
    fn weights_slot_all_arms_gone_stays_a_typed_refusal() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_weights_arm(&dst, "model.safetensors");
        // No open_clip_model.safetensors or model.onnx exists at all.

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        std::fs::remove_file(dst.join("model.safetensors")).unwrap();

        let result = fingerprint.probe();
        assert!(
            result.is_err(),
            "deleting model.safetensors with NO alternate weights file present must \
             remain a typed refusal — no arm of the weights slot can satisfy a cold \
             resolve — unchanged from before the round-10 reshape. Got {result:?}"
        );
    }

    /// Build a `ResolvedModel` whose weights slot has a PRIMARY arm
    /// (`model.safetensors`, matching one of the three known names — the
    /// "primary-file edge" from `all_candidate_paths`' weights-alternates
    /// doc) PLUS `extra_shard_count` additional CONJUNCTIVELY-required
    /// shard-named files (`model-NNNNN-of-MMMMM.safetensors`, mirroring HF's
    /// own sharded naming, resolver.rs:432-442) — every one of them present
    /// in `weights_paths`, i.e. "all gated". Each shard's bytes are distinct
    /// (`marker` byte written first) so a per-shard mutation test can target
    /// exactly one shard unambiguously. Real `download_safetensors` never
    /// actually produces a named-primary-plus-shards mix (it returns a
    /// single named file OR the full shard set, never both — see the
    /// primary-file-edge doc) — this fixture exercises the code path
    /// generically regardless, since nothing in `all_candidate_paths`'
    /// per-entry classification depends on that never happening.
    fn resolved_with_primary_and_shards(
        dst: &std::path::Path,
        extra_shard_count: usize,
    ) -> ResolvedModel {
        std::fs::create_dir_all(dst).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::copy(fixture.join(name), dst.join(name)).unwrap();
        }
        let mut weights_paths = vec![dst.join("model.safetensors")];
        for i in 0..extra_shard_count {
            let shard_name = format!("model-{:05}-of-{:05}.safetensors", i + 1, extra_shard_count);
            let shard_path = dst.join(&shard_name);
            std::fs::write(&shard_path, [i as u8, 0xAB, 0xCD, 0xEF]).unwrap();
            weights_paths.push(shard_path);
        }
        let model_config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dst.join("config.json")).unwrap()).unwrap();
        ResolvedModel {
            model_id: crate::model::ModelId(format!("local:{}", dst.display())),
            backend: crate::model::BackendType::Candle,
            weights_format: crate::model::WeightsFormat::Safetensors,
            task: ModelTask::TextEmbedding,
            config_path: dst.join("config.json"),
            weights_paths,
            tokenizer: Some(TokenizerSource::HuggingFaceJson(dst.join("tokenizer.json"))),
            model_config,
            preprocessor_config: None,
            pooling_config: None,
            base_model_id: None,
            adapter_path: None,
            estimated_memory: 0,
        }
    }

    /// (a) F-1 core, round 12: deleting ONE shard of a multi-shard weights
    /// download must be a typed REFUSAL, never merely stale — a shard is
    /// CONJUNCTIVELY required (`VarBuilder::from_mmaped_safetensors` needs
    /// ALL of them), so no sibling shard's continued presence can rescue a
    /// cold resolve of the now-incomplete set.
    ///
    /// RED pre-fix (round-10 folded-slot shape, reproduced via the stash
    /// methodology: reverting `all_candidate_paths`' weights-slot
    /// construction to append every shard as an extra ALWAYS-GATED arm of
    /// the SAME disjunctive alternates slot): the deleted shard was just
    /// one more arm of that slot, and the slot's OTHER arms (the still-intact
    /// second shard, or the primary) satisfy `any_arm_present_now`, so
    /// `probe` fell into arm (b) and wrongly reported `Ok(false)` (stale)
    /// instead of the typed refusal a cold resolve of the now-incomplete
    /// shard set would actually hit. GREEN post-fix: each shard is its own
    /// single-arm, `absence_tolerated: false` slot, so losing it is arm (c)
    /// unconditionally.
    #[test]
    fn shard_slot_one_shard_deleted_is_a_typed_refusal_never_stale() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_primary_and_shards(&dst, 2);

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        assert!(
            fingerprint.probe().unwrap(),
            "sanity: fresh immediately after capture"
        );

        // Delete exactly one shard (the second extra shard); the primary
        // and the OTHER shard remain untouched.
        std::fs::remove_file(dst.join("model-00002-of-00002.safetensors")).unwrap();

        let result = fingerprint.probe();
        assert!(
            result.is_err(),
            "deleting one shard of a multi-shard weights download must be a typed \
             refusal (audit round 62, adversarial round 12, F-1) — a shard is \
             CONJUNCTIVELY required, so the primary file and the OTHER shard still \
             being present must NOT rescue this into merely stale. Got {result:?}"
        );
    }

    /// (b) slot independence: the weights ALTERNATES slot's own arm-(c)
    /// contract (all three named arms gone -> typed refusal) must still hold
    /// even when this fixture ALSO has healthy, untouched per-shard slots —
    /// proving the round-12 reshape did not accidentally let a healthy shard
    /// slot mask a genuine alternates-slot refusal (each slot is evaluated,
    /// and can independently refuse, on its own).
    #[test]
    fn alternates_slot_all_arms_gone_is_a_typed_refusal_even_with_shards_intact() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_primary_and_shards(&dst, 2);
        // No open_clip_model.safetensors or model.onnx exists at all, so
        // deleting the primary leaves EVERY arm of the alternates slot gone.

        let fingerprint = compute_model_fingerprint(&resolved).unwrap();
        std::fs::remove_file(dst.join("model.safetensors")).unwrap();
        // The two shard files are left completely untouched.

        let result = fingerprint.probe();
        assert!(
            result.is_err(),
            "deleting the sole named primary, with no open_clip_model.safetensors or \
             model.onnx alternate present, must remain a typed refusal for the \
             ALTERNATES slot regardless of the per-shard slots being entirely \
             untouched and healthy — each slot refuses independently. Got {result:?}"
        );
    }

    /// (d) the shard-append reshape must still digest-gate every shard: a
    /// byte mutation to any ONE shard file must change
    /// `compute_model_content_digest`'s output, under a constant
    /// `resolved.weights_paths` — the multi-shard peer of the existing
    /// single-file `weights_slot_*` digest coverage (the adapter/config/
    /// tokenizer/pooling mutation family in this module), extended to the
    /// per-shard-slot shape (round 12).
    #[test]
    fn shard_byte_mutation_changes_content_digest() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = resolved_with_primary_and_shards(&dst, 2);

        let digest_before = compute_model_content_digest(&resolved).unwrap();

        let shard_path = dst.join("model-00001-of-00002.safetensors");
        let mut bytes = std::fs::read(&shard_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&shard_path, &bytes).unwrap();

        let digest_after = compute_model_content_digest(&resolved).unwrap();

        assert_ne!(
            digest_before, digest_after,
            "mutating ONE shard's bytes in place, under a constant `weights_paths`, must \
             change the content digest — every shard remains its own always-gated slot \
             after the round-12 reshape, exactly like the pre-reshape flat per-file model"
        );
    }
}

// ── R5-F2 (audit round 62, adversarial round 6): a classification-loaded \
//    model must refuse a TextEmbedding request, not silently mean-pool \
//    softmax logits ──

#[cfg(test)]
mod r5_f2_classification_pooling_tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, StringArray};

    use super::*;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn two_row_content() -> Vec<ArrayRef> {
        vec![Arc::new(StringArray::from(vec!["fine row", "another row"])) as ArrayRef]
    }

    /// A `ResolvedModel` for the `tiny_modernbert_classifier` cookbook
    /// fixture (`model_type: "modernbert"`, `id2label` present — the exact
    /// shape `CandleBackend::load`'s `is_classification` gate requires),
    /// resolved for `ModelTask::Classification` — the SAME construction a
    /// real `ModelResolver::resolve` + `ModelCache::get_or_load` call would
    /// produce for a caller that loaded this model to classify.
    fn tiny_modernbert_classifier_resolved(dst: &std::path::Path) -> ResolvedModel {
        std::fs::create_dir_all(dst).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_modernbert_classifier");
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(name), dst.join(name)).unwrap();
        }
        let model_config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dst.join("config.json")).unwrap()).unwrap();
        ResolvedModel {
            model_id: crate::model::ModelId(format!("local:{}", dst.display())),
            backend: crate::model::BackendType::Candle,
            weights_format: crate::model::WeightsFormat::Safetensors,
            task: ModelTask::Classification,
            config_path: dst.join("config.json"),
            weights_paths: vec![dst.join("model.safetensors")],
            tokenizer: Some(TokenizerSource::HuggingFaceJson(dst.join("tokenizer.json"))),
            model_config,
            preprocessor_config: None,
            pooling_config: None,
            base_model_id: None,
            adapter_path: None,
            estimated_memory: 0,
        }
    }

    /// R5-F2 (block): reproduces the auditor's exact mismatch at the
    /// closest constructible seam — `CandleBackend::load` a genuine
    /// classification-shaped checkpoint (the SAME construction a
    /// `ModelCache` warm entry loaded for `ModelTask::Classification` would
    /// hold), then request `ModelTask::TextEmbedding` against that SAME
    /// `LoadedModel` — exactly what a second caller reaches through
    /// `ModelCache::get_or_load` today, since the cache keys purely on
    /// `ModelId` and never compares the originally-loaded task against a
    /// new request's task.
    ///
    /// RED pre-fix: none of the three classification wrappers overrode
    /// `forward_pooled`, so `LoadedModel::forward(.., TextEmbedding)` →
    /// `CandleModel::forward_embedding` → `self.text_forward()?.forward_pooled(..)`
    /// silently fell through to `CandleTextForward`'s trait-default —
    /// REAL mean-pooling over the classification wrapper's softmax-logit
    /// `forward_hidden` output (a `[batch, num_classes]` probability
    /// distribution, not a per-token hidden state sequence) — producing
    /// either a shape error deep inside candle (opaque, not a typed
    /// refusal) or, worse, a confident-wrong tensor of the wrong width.
    ///
    /// GREEN post-fix: the classification wrapper's `forward_pooled`
    /// override refuses immediately with a named, typed
    /// `JammiError::Inference`, never reaching `mean_pool`/`l2_normalize`
    /// at all.
    #[test]
    fn classification_loaded_model_refuses_text_embedding_request() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_modernbert_classifier_resolved(&dst);

        let backend = CandleBackend;
        let loaded = backend
            .load(&resolved, &device_config())
            .expect("a genuinely classification-shaped checkpoint must load successfully");

        let content = two_row_content();
        let result = loaded.forward(&content, ModelTask::TextEmbedding);

        match result {
            Ok(_) => panic!(
                "a classification-loaded model must REFUSE a TextEmbedding \
                 request with a typed error (R5-F2) — never silently produce \
                 a tensor by falling through to forward_pooled's mean-pool \
                 default over softmax-logit output"
            ),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("classification") && msg.contains("pooling"),
                    "expected the classification-pooling-mismatch typed \
                     refusal (from `classification_pooling_refusal`), got: {msg}"
                );
            }
        }
    }

    /// Control (F family: non-vacuous negative control): the IDENTICAL
    /// classification-loaded model must still serve its OWN task
    /// (`Classification`) successfully — the fix must refuse the MISMATCH
    /// specifically, not classification serving in general.
    #[test]
    fn classification_loaded_model_still_serves_classification_request() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_modernbert_classifier_resolved(&dst);

        let backend = CandleBackend;
        let loaded = backend
            .load(&resolved, &device_config())
            .expect("a genuinely classification-shaped checkpoint must load successfully");

        let content = two_row_content();
        let result = loaded.forward(&content, ModelTask::Classification);
        match result {
            Ok(_) => {}
            Err(e) => panic!(
                "the classification wrapper's OWN task must still serve \
                 successfully — the fix refuses the TextEmbedding MISMATCH, \
                 not classification itself. Got Err({e})"
            ),
        }
    }

    /// A `ResolvedModel` for the SAME `tiny_modernbert_classifier` cookbook
    /// fixture as above (its `config.json` carries `id2label` — a real
    /// checkpoint's default classification-head declaration), but resolved
    /// for `ModelTask::TextEmbedding` instead of `Classification` — the
    /// SAME construction a caller who loads this checkpoint to embed, not
    /// classify, would produce. `CandleBackend::load`'s `is_classification`
    /// gate (`resolved.task == ModelTask::Classification && id2label.is_some()`)
    /// is `false` here, so a plain `ModernBertForward` (not
    /// `ModernBertClassificationForward`) is built — but `CandleModel::id2label`
    /// is still populated, since that parse reads `config.json` unconditionally,
    /// independent of `is_classification`.
    fn tiny_modernbert_classifier_resolved_for_embedding(dst: &std::path::Path) -> ResolvedModel {
        let mut resolved = tiny_modernbert_classifier_resolved(dst);
        resolved.task = ModelTask::TextEmbedding;
        resolved
    }

    /// R7-F2 (audit round 62, adversarial round 8 advisory fold): the mirror
    /// of `classification_loaded_model_refuses_text_embedding_request`
    /// above, in the OTHER mismatch direction — an embedding-loaded warm
    /// entry (`ModelTask::TextEmbedding`, `id2label` present in
    /// `config.json` regardless) requested for `ModelTask::Classification`.
    ///
    /// RED pre-fix: `forward_classification`'s only guard was
    /// `self.id2label.is_some()`, which this scenario satisfies — so the
    /// call proceeded to `self.text_forward()?.forward_hidden(..)` (the
    /// PLAIN `ModernBertForward`'s raw per-token hidden states,
    /// `[batch, seq_len, hidden_size]`) and then `to_vec2::<f32>()` over
    /// that 3-D tensor, dying with an opaque candle rank error instead of a
    /// legible typed refusal.
    ///
    /// GREEN post-fix: `is_classification_head()` is `false` for the plain
    /// `ModernBertForward` wrapper, so `forward_classification` refuses
    /// immediately with `classification_kind_mismatch_refusal`, never
    /// reaching `to_vec2` at all.
    #[test]
    fn embedding_loaded_model_with_id2label_refuses_classification_request() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_modernbert_classifier_resolved_for_embedding(&dst);

        let backend = CandleBackend;
        let loaded = backend
            .load(&resolved, &device_config())
            .expect("an id2label-bearing checkpoint loaded for TextEmbedding must still succeed");

        let content = two_row_content();
        let result = loaded.forward(&content, ModelTask::Classification);

        match result {
            Ok(_) => panic!(
                "an embedding-loaded model must REFUSE a Classification \
                 request with a typed error (R7-F2) — never reach \
                 to_vec2::<f32> over the raw hidden-states tensor and fail \
                 with an opaque candle rank error"
            ),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("not loaded for classification"),
                    "expected the classification-kind-mismatch typed refusal \
                     (from `classification_kind_mismatch_refusal`), got: {msg}"
                );
            }
        }
    }

    /// Control (F family: non-vacuous negative control) for R7-F2: the
    /// IDENTICAL embedding-loaded, `id2label`-bearing model must still serve
    /// its OWN task (`TextEmbedding`) successfully — the fix refuses the
    /// MISMATCH specifically, not embedding serving in general.
    #[test]
    fn embedding_loaded_model_with_id2label_still_serves_text_embedding_request() {
        let model_tmp = tempfile::tempdir().unwrap();
        let dst = model_tmp.path().join("model");
        let resolved = tiny_modernbert_classifier_resolved_for_embedding(&dst);

        let backend = CandleBackend;
        let loaded = backend
            .load(&resolved, &device_config())
            .expect("an id2label-bearing checkpoint loaded for TextEmbedding must still succeed");

        let content = two_row_content();
        let result = loaded.forward(&content, ModelTask::TextEmbedding);
        match result {
            Ok(_) => {}
            Err(e) => panic!(
                "the embedding wrapper's OWN task must still serve \
                 successfully — the fix refuses the Classification \
                 MISMATCH, not text-embedding itself. Got Err({e})"
            ),
        }
    }
}
