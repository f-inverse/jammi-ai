//! Closed-enum dispatch over the encoder families used by `jammi-ai`, across
//! all three modalities.
//!
//! The three BERT-family encoders ([`Bert`], [`DistilBert`], [`ModernBert`])
//! share a uniform contract: a `[batch, seq, hidden]` `forward_hidden` plus
//! LoRA-aware training-mode hooks. [`ClipText`], [`OpenClipVisionTransformer`]
//! and [`HtsatAudio`] are the three cross-modal towers; each produces a
//! pooled `[batch, output_dim]` shared-latent embedding and — since this
//! commit — carries real LoRA sites, so every training hook on this enum is
//! live for all six variants.
//!
//! # One input vocabulary, one refusal shape
//!
//! A text tower consumes token ids, a vision tower pixels, an audio tower a
//! fusion spectrogram: three genuinely different input types that a single
//! `forward(input_ids, mask)` signature cannot express. [`EncoderInput`]
//! names all three, [`AnyEncoder::forward_input`] dispatches on it, and a
//! mismatch between the input's [`Modality`] and the encoder's own is a
//! TYPED refusal naming both — never a silently reinterpreted tensor.
//! [`AnyEncoder::forward`] stays as the text convenience and is exactly
//! `forward_input(EncoderInput::Text { .. })`.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use crate::bert::Bert;
use crate::clip_text::ClipText;
use crate::distilbert::DistilBert;
use crate::error::EncoderError;
use crate::htsat_audio::HtsatAudio;
use crate::modernbert::ModernBert;
use crate::open_clip_vision::OpenClipVisionTransformer;

/// The number of channels a CLAP fusion spectrogram carries, fixed by the
/// HTSAT patch embedding's own split: one GLOBAL mel channel plus three
/// LOCAL ones (`HtsatAudioEncoder::forward_front` → `HtsatPatchEmbed::
/// forward`'s `x.narrow(1, 0, 1)` / `x.narrow(1, 1, 3)`). Not a config
/// field — the tower's structure.
const CLAP_FUSION_CHANNELS: usize = 4;

/// Time frames in [`AnyEncoder::probe_input`]'s audio batch.
///
/// The front half resamples the time axis to a FIXED width
/// (`spec_size * freq_ratio`) before anything else looks at it, so every
/// downstream geometry — the `reshape_mel2img` fold, the patch grid, the
/// Swin windows — is independent of the input frame count. The only
/// constraint left is the bicubic resample itself, whose Keys kernel is a
/// 4-tap window (`TimeInterp::cubic_coefficients` returns exactly four
/// weights): 4 is therefore the smallest input length at which each tap can
/// address a distinct input frame rather than collapsing onto clamped edge
/// replicas. Smaller inputs still evaluate — the taps clamp — but they
/// exercise a degenerate corner of the resample rather than the geometry a
/// probe is meant to check.
const AUDIO_PROBE_TIME_FRAMES: usize = 4;

/// Tokens in [`AnyEncoder::probe_input`]'s text batch — long enough that
/// attention is not a 1×1 degenerate case, short enough to stay under every
/// tower's context length.
const TEXT_PROBE_TOKENS: usize = 4;

/// Which kind of input an encoder consumes (and an [`EncoderInput`] carries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// Token ids plus an attention mask.
    Text,
    /// Preprocessed pixel values.
    Image,
    /// A CLAP fusion spectrogram plus its per-clip `is_longer` flags.
    Audio,
}

impl Modality {
    /// Lower-case name used in refusal messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }
}

/// A borrowed batch for exactly one modality — the input vocabulary
/// [`AnyEncoder::forward_input`] dispatches on.
pub enum EncoderInput<'a> {
    /// `input_ids` `[batch, seq]` (integer dtype) and `attention_mask`
    /// `[batch, seq]`.
    Text {
        /// `[batch, seq]` token ids.
        input_ids: &'a Tensor,
        /// `[batch, seq]` attention mask.
        attention_mask: &'a Tensor,
    },
    /// `pixel_values` `[batch, 3, image_size, image_size]`, already
    /// normalized with the tower's own mean/std.
    Image {
        /// `[batch, 3, image_size, image_size]` preprocessed pixels.
        pixel_values: &'a Tensor,
    },
    /// A CLAP fusion spectrogram `[batch, 4, time, num_mel_bins]` with one
    /// `is_longer` flag per clip.
    Audio {
        /// `[batch, 4, time, num_mel_bins]` fusion spectrogram.
        input_features: &'a Tensor,
        /// One flag per batch row: did the source clip exceed the fixed
        /// fusion window?
        is_longer: &'a [bool],
    },
}

impl EncoderInput<'_> {
    /// Which modality this batch is.
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text { .. } => Modality::Text,
            Self::Image { .. } => Modality::Image,
            Self::Audio { .. } => Modality::Audio,
        }
    }
}

/// The owning twin of [`EncoderInput`], for a caller that must MATERIALISE a
/// batch and hand it around (e.g. [`AnyEncoder::probe_input`], which
/// constructs one). Borrow it back with [`Self::as_input`].
pub enum OwnedEncoderInput {
    /// See [`EncoderInput::Text`].
    Text {
        /// `[batch, seq]` token ids.
        input_ids: Tensor,
        /// `[batch, seq]` attention mask.
        attention_mask: Tensor,
    },
    /// See [`EncoderInput::Image`].
    Image {
        /// `[batch, 3, image_size, image_size]` preprocessed pixels.
        pixel_values: Tensor,
    },
    /// See [`EncoderInput::Audio`].
    Audio {
        /// `[batch, 4, time, num_mel_bins]` fusion spectrogram.
        input_features: Tensor,
        /// One flag per batch row.
        is_longer: Vec<bool>,
    },
}

impl OwnedEncoderInput {
    /// Borrow this batch as an [`EncoderInput`].
    pub fn as_input(&self) -> EncoderInput<'_> {
        match self {
            Self::Text {
                input_ids,
                attention_mask,
            } => EncoderInput::Text {
                input_ids,
                attention_mask,
            },
            Self::Image { pixel_values } => EncoderInput::Image { pixel_values },
            Self::Audio {
                input_features,
                is_longer,
            } => EncoderInput::Audio {
                input_features,
                is_longer,
            },
        }
    }

    /// Which modality this batch is.
    pub fn modality(&self) -> Modality {
        self.as_input().modality()
    }
}

/// Family-erased encoder for callers that need to hand around any of the
/// supported encoder types without trait-object overhead.
pub enum AnyEncoder {
    /// BERT / RoBERTa / CamemBERT / XLM-RoBERTa.
    Bert(Bert),
    /// DistilBERT.
    DistilBert(DistilBert),
    /// ModernBERT.
    ModernBert(ModernBert),
    /// OpenCLIP text tower. Produces shared-latent `[batch, embed_dim]`
    /// outputs from `forward`; it has no per-token hidden-state output.
    ClipText(ClipText),
    /// OpenCLIP vision tower. Consumes pixels; produces shared-latent
    /// `[batch, embed_dim]` outputs.
    OpenClipVision(OpenClipVisionTransformer),
    /// HTSAT-Swin CLAP audio tower. Consumes a 4-channel fusion
    /// spectrogram; produces shared-latent `[batch, projection_dim]`
    /// outputs.
    ///
    /// BOXED (`clippy::large_enum_variant`): an [`HtsatAudio`] is roughly
    /// four times the next-largest variant, and an unboxed payload would
    /// make EVERY `AnyEncoder` — including a small BERT one — carry that
    /// footprint. The same reasoning `jammi-ai`'s own `TrainingTarget`
    /// applies to its `EncoderAdapters` payload.
    Htsat(Box<HtsatAudio>),
}

impl AnyEncoder {
    /// Which modality this encoder consumes.
    pub fn modality(&self) -> Modality {
        match self {
            Self::Bert(_) | Self::DistilBert(_) | Self::ModernBert(_) | Self::ClipText(_) => {
                Modality::Text
            }
            Self::OpenClipVision(_) => Modality::Image,
            Self::Htsat(_) => Modality::Audio,
        }
    }

    /// Pooled `[batch, output_dim]` embedding for a batch of this
    /// encoder's OWN modality.
    ///
    /// A modality mismatch is a typed [`EncoderError::Config`] naming BOTH
    /// the encoder's modality and the input's — the one place a caller that
    /// wired the wrong batch to the wrong tower finds out, instead of a
    /// tensor being reinterpreted into a confidently-wrong embedding (a
    /// `[batch, 4, time, mel]` spectrogram is not shape-compatible with a
    /// text tower today, but a `[batch, seq]` integer tensor and a
    /// `[batch, 3, h, w]` pixel tensor are both "just tensors" to a
    /// downstream matmul).
    ///
    /// # Input dtype is NOT part of the contract
    ///
    /// A MEDIA batch (image or audio) is accepted in any floating dtype,
    /// on a backbone of any dtype: each tower casts at its own forward
    /// edge (`OpenClipVisionTransformer::forward`,
    /// `HtsatAudioEncoder::forward_front`), so the F32 batch every
    /// production front end emits works against an F16 or BF16 backbone
    /// without the caller knowing [`Self::dtype`]. A caller that already
    /// holds a batch in the backbone dtype pays nothing for this: casting
    /// to the dtype a tensor already has is a clone, so pre-cast and
    /// not-pre-cast produce bit-identical outputs. TEXT input carries no
    /// floating dtype at all — ids and mask are integer tensors.
    ///
    /// SHAPE, by contrast, IS part of the contract and is not repaired
    /// here: a wrong image side or mel-bin count is still a refusal from
    /// the tower's own geometry check.
    pub fn forward_input(&self, input: &EncoderInput<'_>) -> Result<Tensor, EncoderError> {
        // Each arm narrows on the INPUT, then asks `self` once whether it
        // can serve that modality; a `None`/non-matching `self` falls
        // through to the one typed refusal below. No arm asserts a
        // narrowing the compiler cannot see, so a seventh `AnyEncoder`
        // variant is a compile error in `Self::text_forward` (a
        // non-exhaustive match) rather than a runtime panic on some input
        // shape nobody tested.
        match input {
            EncoderInput::Text {
                input_ids,
                attention_mask,
            } => {
                if let Some(out) = self.text_forward(input_ids, attention_mask) {
                    return out;
                }
            }
            EncoderInput::Image { pixel_values } => {
                if let Self::OpenClipVision(e) = self {
                    return e.forward(pixel_values);
                }
            }
            EncoderInput::Audio {
                input_features,
                is_longer,
            } => {
                if let Self::Htsat(e) = self {
                    return e.forward(input_features, is_longer);
                }
            }
        }
        Err(EncoderError::Config(format!(
            "encoder modality mismatch: this encoder consumes {} input, but a {} batch was \
             supplied",
            self.modality().name(),
            input.modality().name()
        )))
    }

    /// Token-ids forward for the four TEXT variants, `None` for the two
    /// media ones — the single exhaustive match over `AnyEncoder` that
    /// [`Self::forward_input`]'s text arm delegates to. Returning
    /// `Option<Result<..>>` keeps "this variant does not consume text"
    /// (structural, `None`) distinct from "it does, and the forward
    /// failed" (`Some(Err)`).
    fn text_forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Option<Result<Tensor, EncoderError>> {
        match self {
            Self::Bert(e) => Some(e.forward(input_ids, attention_mask)),
            Self::DistilBert(e) => Some(e.forward(input_ids, attention_mask)),
            Self::ModernBert(e) => Some(e.forward(input_ids, attention_mask)),
            Self::ClipText(e) => Some(e.forward(input_ids, attention_mask)),
            Self::OpenClipVision(_) | Self::Htsat(_) => None,
        }
    }

    /// Pooled `[batch, output_dim]` embedding from token ids — the TEXT
    /// convenience, exactly `forward_input(EncoderInput::Text { .. })`
    /// (asserted by `tests::forward_equals_forward_input_text`), so a media
    /// variant refuses here with the same typed message.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        self.forward_input(&EncoderInput::Text {
            input_ids,
            attention_mask: mask,
        })
    }

    /// Per-token `[batch, seq, hidden]` hidden states. Only the BERT-family
    /// variants expose this; the three cross-modal towers produce a pooled
    /// projected output with no peer hidden-state output, and refuse.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        match self {
            Self::Bert(e) => e.forward_hidden(input_ids, mask),
            Self::DistilBert(e) => e.forward_hidden(input_ids, mask),
            Self::ModernBert(e) => e.forward_hidden(input_ids, mask),
            Self::ClipText(_) => Err(EncoderError::Config(
                "ClipText does not expose forward_hidden; use forward for pooled CLIP embeddings"
                    .into(),
            )),
            Self::OpenClipVision(_) | Self::Htsat(_) => Err(EncoderError::Config(format!(
                "forward_hidden is a token-sequence operation and this encoder consumes {} \
                 input; use forward_input for its pooled embedding",
                self.modality().name()
            ))),
        }
    }

    /// Maximum input sequence length. For BERT-family variants this is
    /// `max_position_embeddings`; for [`Self::ClipText`] the fixed OpenCLIP
    /// `context_length` (typically 77).
    ///
    /// A media variant has NO token-sequence capacity at all, so this is a
    /// typed refusal there rather than a number. It returns `Result` for
    /// exactly that reason: the previous infallible signature had no honest
    /// value to give a vision or audio tower, and every plausible filler
    /// (`0`, `usize::MAX`, the tower's patch count) would flow straight into
    /// a caller's `min()` as a confidently wrong sequence bound.
    pub fn max_seq_length(&self) -> Result<usize, EncoderError> {
        match self {
            Self::Bert(e) => Ok(e.max_seq_length()),
            Self::DistilBert(e) => Ok(e.max_seq_length()),
            Self::ModernBert(e) => Ok(e.max_seq_length()),
            Self::ClipText(e) => Ok(e.context_length()),
            Self::OpenClipVision(_) | Self::Htsat(_) => Err(EncoderError::Config(format!(
                "max_seq_length is a token-sequence property and this encoder consumes {} \
                 input; it has no sequence capacity",
                self.modality().name()
            ))),
        }
    }

    /// Output dimensionality of [`Self::forward_input`]: `hidden_size` for
    /// the BERT family, the shared CLIP latent `embed_dim` for the two
    /// OpenCLIP towers, the shared CLAP latent `projection_dim` for HTSAT.
    /// Always defined — every variant produces a pooled embedding.
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Bert(e) => e.hidden_size(),
            Self::DistilBert(e) => e.hidden_size(),
            Self::ModernBert(e) => e.hidden_size(),
            Self::ClipText(e) => e.embed_dim(),
            Self::OpenClipVision(e) => e.embed_dim(),
            Self::Htsat(e) => e.projection_dim(),
        }
    }

    /// Dtype this encoder's FROZEN BACKBONE weights are materialised at.
    ///
    /// Every variant derives it from a real weight tensor it holds (the
    /// word/token-embedding table for the four text variants, the
    /// patch-embedding kernel for vision and audio), so it is the dtype an
    /// input actually meets rather than a builder setting somebody
    /// remembered to record. Always defined — there is no encoder without
    /// weights.
    ///
    /// This is what makes [`Self::probe_input`]'s "every field is derived
    /// from the encoder" claim true for the FLOATING batches: a probe built
    /// at this dtype meets the backbone with no conversion at all.
    ///
    /// It is a REPORT, not a precondition on [`Self::forward_input`]. The
    /// raw candle ops in a media tower's front end do refuse a mismatched
    /// input (candle's `conv2d` on an F32 batch against an F16 kernel is a
    /// `dtype mismatch in conv2d`; HTSAT's leading `BatchNorm` is a `dtype
    /// mismatch in sub`), which is precisely why each tower casts at its
    /// own forward edge instead — the one place that knows this value. A
    /// caller may hand [`Self::forward_input`] an F32 media batch on ANY
    /// backbone dtype; see that method's own "Input dtype is NOT part of
    /// the contract" section.
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bert(e) => e.dtype(),
            Self::DistilBert(e) => e.dtype(),
            Self::ModernBert(e) => e.dtype(),
            Self::ClipText(e) => e.dtype(),
            Self::OpenClipVision(e) => e.dtype(),
            Self::Htsat(e) => e.dtype(),
        }
    }

    /// Square input side an image batch must carry — vision variants only.
    pub fn image_size(&self) -> Result<usize, EncoderError> {
        match self {
            Self::OpenClipVision(e) => Ok(e.image_size()),
            other => Err(other.not_a_media_accessor("image_size", Modality::Image)),
        }
    }

    /// Per-channel preprocessing mean — vision variants only.
    pub fn preprocess_mean(&self) -> Result<[f32; 3], EncoderError> {
        match self {
            Self::OpenClipVision(e) => Ok(e.preprocess_mean()),
            other => Err(other.not_a_media_accessor("preprocess_mean", Modality::Image)),
        }
    }

    /// Per-channel preprocessing standard deviation — vision variants only.
    pub fn preprocess_std(&self) -> Result<[f32; 3], EncoderError> {
        match self {
            Self::OpenClipVision(e) => Ok(e.preprocess_std()),
            other => Err(other.not_a_media_accessor("preprocess_std", Modality::Image)),
        }
    }

    /// Mel bins an audio batch's spectrogram must carry — audio variants
    /// only.
    pub fn num_mel_bins(&self) -> Result<usize, EncoderError> {
        match self {
            Self::Htsat(e) => Ok(e.num_mel_bins()),
            other => Err(other.not_a_media_accessor("num_mel_bins", Modality::Audio)),
        }
    }

    /// The one refusal message shape the four modality-specific accessors
    /// above share.
    fn not_a_media_accessor(&self, accessor: &str, needs: Modality) -> EncoderError {
        EncoderError::Config(format!(
            "{accessor} is a preprocessing property of {} encoders and this encoder consumes \
             {} input",
            needs.name(),
            self.modality().name()
        ))
    }

    /// The smallest VALID batch for this variant's own input geometry, ready
    /// to feed [`Self::forward_input`].
    ///
    /// Exists so a caller that must DRIVE one forward pass to observe
    /// something about it (a kernel-dispatch/acceleration probe, a warm-up)
    /// can do so on any variant without knowing that variant's geometry.
    /// Every field is derived from the encoder itself — the image side from
    /// the vision config, the mel-bin count from the audio config — rather
    /// than from a caller-side table that could fall out of step with a
    /// checkpoint.
    ///
    /// The batches are content-free (zeros) but shape-VALID: a probe is
    /// about which code path runs, not about what the numbers mean. Audio
    /// uses `is_longer = [true]`, the branch that exercises the AFF fusion
    /// path as well as the plain patch-conv (the `false` branch computes a
    /// strict subset of it).
    ///
    /// The DTYPE is derived too, via [`Self::dtype`], so the probe reaches
    /// the backbone with no conversion in the way — a probe is about which
    /// code path runs, and a cast the production path would not perform is
    /// one more thing between the caller and that path. (It is no longer
    /// REQUIRED: [`Self::forward_input`] accepts an F32 media batch on any
    /// backbone dtype. Derived is still the honest default.) The text batch
    /// is integer (`U32` ids and mask) and carries no floating dtype at
    /// all.
    pub fn probe_input(&self, device: &Device) -> Result<OwnedEncoderInput, EncoderError> {
        Ok(match self {
            Self::Bert(_) | Self::DistilBert(_) | Self::ModernBert(_) | Self::ClipText(_) => {
                OwnedEncoderInput::Text {
                    input_ids: Tensor::zeros((1, TEXT_PROBE_TOKENS), DType::U32, device)?,
                    attention_mask: Tensor::ones((1, TEXT_PROBE_TOKENS), DType::U32, device)?,
                }
            }
            Self::OpenClipVision(e) => {
                let side = e.image_size();
                OwnedEncoderInput::Image {
                    pixel_values: Tensor::zeros((1, 3, side, side), e.dtype(), device)?,
                }
            }
            Self::Htsat(e) => OwnedEncoderInput::Audio {
                input_features: Tensor::zeros(
                    (
                        1,
                        CLAP_FUSION_CHANNELS,
                        AUDIO_PROBE_TIME_FRAMES,
                        e.num_mel_bins(),
                    ),
                    e.dtype(),
                    device,
                )?,
                is_longer: vec![true],
            },
        })
    }

    /// The LoRA SELECTOR NAMES this variant's builder can wrap — the
    /// vocabulary a caller's `target_modules` is matched against.
    ///
    /// Each family names its sites in its own checkpoint's terms, and the
    /// vocabularies genuinely differ — BERT's dotted `attention.self.query`
    /// / … / `output.dense`, DistilBERT's `q_lin` / `k_lin` / `v_lin` /
    /// `out_lin` / `lin1` / `lin2`, ModernBERT's fused `Wqkv` / `Wo` / `Wi`
    /// / `mlp.Wo`, the two OpenCLIP towers' shared `in_proj` / `out_proj` /
    /// `c_fc` / `c_proj`, and HTSAT's nine. Each list is a `pub(crate) const
    /// LORA_SITE_NAMES` in the owning module, built from the very constants
    /// its sites are wrapped under wherever those exist.
    ///
    /// This exists so a caller can tell a name that selects NOTHING from a
    /// name that selects something BEFORE building: a `target_modules` list
    /// of plausible-but-wrong strings produces a tower with zero trainable
    /// parameters and no error at all, which then trains happily and
    /// updates nothing.
    ///
    /// It is the EXACT-MATCH vocabulary, not the set of accepted strings:
    /// `jammi_lora::should_apply_lora` also accepts `all-linear` and any
    /// SUFFIX of one of these names, which is why the short `["query",
    /// "value"]` form works against a BERT checkpoint whose sites are
    /// offered as `attention.self.query` / `attention.self.value`. Do not
    /// infer the short form by trimming: BERT's adapter-key leaves
    /// (`intermediate_dense`, `output_dense`) are NOT selectors — the
    /// selector is `intermediate.dense` / `output.dense`, and that
    /// distinction is what this accessor exists to stop a caller
    /// re-deriving by hand.
    ///
    /// Never empty for any variant: every variant carries real LoRA sites.
    /// Asserted per variant on a real fixture — every name here selects at
    /// least one site, and their union is exactly what `all-linear` selects
    /// — by `tests/it/lora_site_names.rs`.
    pub fn lora_site_names(&self) -> &'static [&'static str] {
        match self {
            Self::Bert(_) => crate::bert::LORA_SITE_NAMES,
            Self::DistilBert(_) => crate::distilbert::LORA_SITE_NAMES,
            Self::ModernBert(_) => crate::modernbert::LORA_SITE_NAMES,
            // ONE list for both OpenCLIP towers: they load the same
            // `crate::open_clip_block::ResidualAttentionBlock`, so their
            // site names are the same four by construction, not by
            // coincidence.
            Self::ClipText(_) | Self::OpenClipVision(_) => crate::open_clip_block::LORA_SITE_NAMES,
            Self::Htsat(_) => crate::htsat_audio::LORA_SITE_NAMES,
        }
    }

    /// Trainable tensors across every LoRA-wrapped site. Empty for a fully
    /// frozen backbone.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::Bert(e) => e.trainable_params(),
            Self::DistilBert(e) => e.trainable_params(),
            Self::ModernBert(e) => e.trainable_params(),
            Self::ClipText(e) => e.trainable_params(),
            Self::OpenClipVision(e) => e.trainable_params(),
            Self::Htsat(e) => e.trainable_params(),
        }
    }

    /// Named LoRA A/B tensors, ready for safetensors serialisation. Each
    /// family uses its own checkpoint-shaped key layout — see the owning
    /// tower's `named_trainable_weights`.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        match self {
            Self::Bert(e) => e.named_trainable_weights(),
            Self::DistilBert(e) => e.named_trainable_weights(),
            Self::ModernBert(e) => e.named_trainable_weights(),
            Self::ClipText(e) => e.named_trainable_weights(),
            Self::OpenClipVision(e) => e.named_trainable_weights(),
            Self::Htsat(e) => e.named_trainable_weights(),
        }
    }

    /// Switch every LoRA site, LayerNorm and attention softmax into / out of
    /// training mode. Real on all six variants.
    pub fn set_training(&mut self, training: bool) {
        match self {
            Self::Bert(e) => e.set_training(training),
            Self::DistilBert(e) => e.set_training(training),
            Self::ModernBert(e) => e.set_training(training),
            Self::ClipText(e) => e.set_training(training),
            Self::OpenClipVision(e) => e.set_training(training),
            Self::Htsat(e) => e.set_training(training),
        }
    }

    /// Restore LoRA A/B tensors from a [`Self::named_trainable_weights`]-shaped
    /// map.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        match self {
            Self::Bert(e) => e.load_weights(weights),
            Self::DistilBert(e) => e.load_weights(weights),
            Self::ModernBert(e) => e.load_weights(weights),
            Self::ClipText(e) => e.load_weights(weights),
            Self::OpenClipVision(e) => e.load_weights(weights),
            Self::Htsat(e) => e.load_weights(weights),
        }
    }

    /// Per-site dropout-stream positions for every LoRA-wrapped linear, keyed by
    /// the same site names [`Self::named_trainable_weights`] uses — the resume
    /// state for the adapter's dropout so a resumed run replays each stream to its
    /// epoch-boundary position. Empty for a backbone with no installed adapters or
    /// no dropout.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        match self {
            Self::Bert(e) => e.dropout_positions(),
            Self::DistilBert(e) => e.dropout_positions(),
            Self::ModernBert(e) => e.dropout_positions(),
            Self::ClipText(e) => e.dropout_positions(),
            Self::OpenClipVision(e) => e.dropout_positions(),
            Self::Htsat(e) => e.dropout_positions(),
        }
    }

    /// Restore each LoRA site's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. Missing keys are no-ops.
    pub fn restore_dropout_positions(
        &self,
        positions: &HashMap<String, u64>,
    ) -> Result<(), EncoderError> {
        match self {
            Self::Bert(e) => e.restore_dropout_positions(positions),
            Self::DistilBert(e) => e.restore_dropout_positions(positions),
            Self::ModernBert(e) => e.restore_dropout_positions(positions),
            Self::ClipText(e) => e.restore_dropout_positions(positions),
            Self::OpenClipVision(e) => e.restore_dropout_positions(positions),
            Self::Htsat(e) => e.restore_dropout_positions(positions),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip_text::{ClipText, ClipTextConfig};
    use crate::test_support::{assert_finite_nonzero, deterministic_fill_varmap, find_var};
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;

    /// A `VarMap`-backed vision tower with a tiny geometry — enough to
    /// exercise every dispatch/refusal arm without a checkpoint.
    fn tiny_vision(device: &Device) -> OpenClipVisionTransformer {
        let cfg = crate::open_clip_vision::OpenClipVisionConfig {
            width: 32,
            layers: 1,
            heads: 4,
            mlp_ratio: 4.0,
            image_size: 8,
            patch_size: 4,
            embed_dim: 16,
            global_average_pool: true,
            preprocess_mean: [0.5, 0.5, 0.5],
            preprocess_std: [0.25, 0.25, 0.25],
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        OpenClipVisionTransformer::load(vb.pp("visual"), &cfg).unwrap()
    }

    /// A `VarMap`-backed audio tower sized by the committed tiny CLAP
    /// config. Constructed only — these tests exercise dispatch and
    /// refusals, which happen before any forward.
    fn tiny_audio(device: &Device) -> HtsatAudio {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/htsat_clap_tiny/config.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let cfg = crate::htsat_audio::HtsatAudioConfig::from_hf_clap_config(&json).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        HtsatAudio::load(vb, &cfg, device).unwrap()
    }

    fn tiny_config() -> ClipTextConfig {
        ClipTextConfig {
            context_length: 16,
            vocab_size: 64,
            width: 16,
            layers: 2,
            heads: 2,
            embed_dim: 8,
        }
    }

    fn fixed_batch(cfg: &ClipTextConfig, device: &Device) -> (Tensor, Tensor) {
        let ids: Vec<u32> = vec![
            1,
            2,
            3,
            4,
            (cfg.vocab_size - 1) as u32, // EOT at index 4
            5,
            6,
            (cfg.vocab_size - 1) as u32,
            0,
            0, // EOT at index 2, padded
        ];
        let input_ids = Tensor::from_vec(ids, (2, 5), device).unwrap();
        let mask = Tensor::ones((2, 5), DType::U32, device).unwrap();
        (input_ids, mask)
    }

    fn nonuniform_loss(out: &Tensor, channels: usize, device: &Device) -> Tensor {
        let weights: Vec<f32> = (0..channels).map(|i| 1.0 + i as f32 * 0.37).collect();
        let weights = Tensor::from_vec(weights, channels, device).unwrap();
        out.broadcast_mul(&weights).unwrap().sum_all().unwrap()
    }

    fn slice_grad_norm(grad: &Tensor, row_start: usize, width: usize) -> f32 {
        grad.narrow(0, row_start, width)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    }

    /// Deletion-catching oracle for [`AnyEncoder::set_training`]'s
    /// `Self::ClipText(e) => e.set_training(training)` arm — the ONLY
    /// production entry point that reaches `ClipText`'s training flag
    /// through this enum (`AnyEncoder` is what `jammi-ai` actually holds).
    /// If that line regressed to `Self::ClipText(_) => {}` (a silent no-op,
    /// exactly matching this enum's other frozen-tower arms like
    /// `load_weights`), `ClipText` would stay stuck on its default
    /// `training=false` regardless of what the caller asked for.
    ///
    /// `training=true` through `AnyEncoder`: a forward+backward through
    /// `AnyEncoder::forward` (which for `ClipText` IS the full public
    /// `ClipText::forward`, `ln_final` and all — unlike `ClipText`'s own
    /// `#[cfg(test)]`-only `block0_backward` helper, which deliberately
    /// stops short of `ln_final` to isolate the softmax site — see that
    /// helper's doc) reaches layer-0's Q/K slices of `in_proj_weight` with
    /// a finite, nonzero gradient. `training=false` (the default, no
    /// `set_training` call): `in_proj_weight` gets NO gradient entry AT
    /// ALL, not merely zero Q/K rows — `ln_final`'s own `BackpropOp::none()`
    /// truncates backward before it reaches ANY block, matching `ClipText`'s
    /// own full-forward eval oracle
    /// (`training_false_full_forward_grads_are_none_before_ln_final`), NOT
    /// its block-level one (which bypasses `ln_final` and therefore still
    /// sees a partial, V-slice-only gradient — a materially different
    /// fixture from what `AnyEncoder::forward` actually runs).
    ///
    /// RED-verified: reverting `Self::ClipText(e) => e.set_training(training)`
    /// to `Self::ClipText(_) => {}` flips the training=true half of this
    /// test (`in_proj_weight` comes back with NO gradient entry at all,
    /// same as the eval half, since the tower never actually leaves eval
    /// mode) while the training=false half stays green (both are already
    /// `training=false` in that case).
    #[test]
    fn any_encoder_set_training_reaches_clip_text_q_k_gradient() {
        let device = Device::Cpu;
        let cfg = tiny_config();

        // training = true.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let clip = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        let mut any = AnyEncoder::ClipText(clip);
        any.set_training(true);

        // `AnyEncoder::ClipText`'s forward reaches `crate::layer_norm`'s
        // process-wide `LN_DISPATCH_COUNTERS` (ln_1/ln_2/ln_final, all
        // biased) at `training=true` — this test never reads that counter
        // itself, but the forward below still bumps it, so it must take
        // the SAME lock the asserting tests in `clip_text.rs`/`layer_norm.rs`
        // hold, or its bump can land inside one of those tests' own
        // before/after window under parallel `cargo test` (see
        // `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s doc).
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let out = any.forward(&input_ids, &mask).unwrap();
        let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
        let grads = loss.backward().unwrap();
        let in_proj_weight = find_var(&varmap, "resblocks.0.attn.in_proj_weight");
        let grad = grads
            .get(in_proj_weight.as_tensor())
            .expect("in_proj_weight must have a gradient under training=true");
        let width = cfg.width;
        let q_norm = slice_grad_norm(grad, 0, width);
        let k_norm = slice_grad_norm(grad, width, width);
        assert_finite_nonzero(q_norm, "Q slice (via AnyEncoder::set_training(true))");
        assert_finite_nonzero(k_norm, "K slice (via AnyEncoder::set_training(true))");

        // training = false (the default; no set_training call at all).
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let clip = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        let any = AnyEncoder::ClipText(clip);

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let out = any.forward(&input_ids, &mask).unwrap();
        let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
        let grads = loss.backward().unwrap();
        let in_proj_weight = find_var(&varmap, "resblocks.0.attn.in_proj_weight");
        assert!(
            grads.get(in_proj_weight.as_tensor()).is_none(),
            "in_proj_weight grad must be None under eval (default, no set_training call) \
             through AnyEncoder::forward's full public forward — ln_final truncates backward \
             before it reaches any block, not merely zeroing Q/K rows"
        );
    }

    /// A4: EVERY wrong (encoder, input) modality pairing is a typed refusal
    /// naming BOTH modalities — not a shape error from deep inside a matmul,
    /// and never a silently reinterpreted tensor. Walks the full 3x3 grid so
    /// no arm is covered by accident.
    #[test]
    fn forward_input_refuses_every_mismatched_modality_pairing() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let text = AnyEncoder::ClipText(ClipText::load(vb, &cfg).unwrap());
        let vision = AnyEncoder::OpenClipVision(tiny_vision(&device));
        let audio = AnyEncoder::Htsat(Box::new(tiny_audio(&device)));

        let ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let mask = Tensor::ones((1, 4), DType::U32, &device).unwrap();
        let pixels = Tensor::zeros((1, 3, 8, 8), DType::F32, &device).unwrap();
        let feats = Tensor::zeros((1, 4, 4, 32), DType::F32, &device).unwrap();
        let longer = [true];

        let inputs = [
            (
                Modality::Text,
                EncoderInput::Text {
                    input_ids: &ids,
                    attention_mask: &mask,
                },
            ),
            (
                Modality::Image,
                EncoderInput::Image {
                    pixel_values: &pixels,
                },
            ),
            (
                Modality::Audio,
                EncoderInput::Audio {
                    input_features: &feats,
                    is_longer: &longer,
                },
            ),
        ];

        let mut mismatches = 0usize;
        for encoder in [&text, &vision, &audio] {
            for (modality, input) in &inputs {
                if *modality == encoder.modality() {
                    continue;
                }
                mismatches += 1;
                let err = encoder.forward_input(input).unwrap_err();
                let EncoderError::Config(message) = &err else {
                    panic!("expected a typed Config refusal, got {err:?}");
                };
                assert!(
                    message.contains(encoder.modality().name())
                        && message.contains(modality.name()),
                    "the refusal must name BOTH the encoder's modality ({}) and the input's \
                     ({}), got: {message}",
                    encoder.modality().name(),
                    modality.name()
                );
            }
        }
        assert_eq!(
            mismatches, 6,
            "the 3x3 grid has exactly six mismatched cells"
        );
    }

    /// A4: the two token-sequence-only accessors refuse on a media variant
    /// rather than inventing a number. `max_seq_length` in particular has no
    /// honest value for a vision or audio tower — see its own doc.
    #[test]
    fn forward_hidden_and_max_seq_length_refuse_on_media_variants() {
        let device = Device::Cpu;
        let ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let mask = Tensor::ones((1, 4), DType::U32, &device).unwrap();

        for encoder in [
            AnyEncoder::OpenClipVision(tiny_vision(&device)),
            AnyEncoder::Htsat(Box::new(tiny_audio(&device))),
        ] {
            let name = encoder.modality().name();
            let err = encoder.forward_hidden(&ids, &mask).unwrap_err();
            assert!(
                matches!(&err, EncoderError::Config(m) if m.contains(name)),
                "{name}: forward_hidden must refuse naming the modality, got {err:?}"
            );
            let err = encoder.max_seq_length().unwrap_err();
            assert!(
                matches!(&err, EncoderError::Config(m) if m.contains(name)),
                "{name}: max_seq_length must refuse naming the modality, got {err:?}"
            );
            // But the pooled output dim IS defined for every variant.
            assert!(encoder.hidden_size() > 0);
        }
    }

    /// The modality-specific preprocessing accessors are reachable on their
    /// OWN variant and a typed refusal elsewhere — a caller that must
    /// preprocess for a tower reads the geometry off the tower, and a
    /// caller that asks the wrong tower is told so.
    #[test]
    fn media_preprocessing_accessors_answer_on_their_own_variant_and_refuse_elsewhere() {
        let device = Device::Cpu;
        let vision = AnyEncoder::OpenClipVision(tiny_vision(&device));
        let audio = AnyEncoder::Htsat(Box::new(tiny_audio(&device)));

        assert_eq!(vision.image_size().unwrap(), 8);
        assert_eq!(vision.preprocess_mean().unwrap(), [0.5, 0.5, 0.5]);
        assert_eq!(vision.preprocess_std().unwrap(), [0.25, 0.25, 0.25]);
        assert!(vision.num_mel_bins().is_err());

        assert_eq!(audio.num_mel_bins().unwrap(), 32);
        assert!(audio.image_size().is_err());
        assert!(audio.preprocess_mean().is_err());
        assert!(audio.preprocess_std().is_err());
    }

    /// `forward(input_ids, mask)` is EXACTLY `forward_input(Text { .. })` —
    /// the text convenience is a spelling, not a second code path that
    /// could drift from the dispatching one.
    #[test]
    fn forward_equals_forward_input_text_bit_for_bit() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = AnyEncoder::ClipText(ClipText::load(vb, &cfg).unwrap());
        deterministic_fill_varmap(&varmap, &device);

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let via_forward = encoder.forward(&input_ids, &mask).unwrap();
        let via_input = encoder
            .forward_input(&EncoderInput::Text {
                input_ids: &input_ids,
                attention_mask: &mask,
            })
            .unwrap();
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        assert_eq!(bits(&via_forward), bits(&via_input));
    }

    /// `probe_input` yields a batch for the encoder's OWN modality, and one
    /// its own `forward_input` accepts — asserted here for the text
    /// variants (the media legs, which need real checkpoints to forward,
    /// live in `tests/tower_lora.rs`'s A7 oracle).
    #[test]
    fn probe_input_is_accepted_by_its_own_encoder() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = AnyEncoder::ClipText(ClipText::load(vb, &cfg).unwrap());
        deterministic_fill_varmap(&varmap, &device);

        let probe = encoder.probe_input(&device).unwrap();
        assert_eq!(probe.modality(), Modality::Text);
        let out = encoder.forward_input(&probe.as_input()).unwrap();
        assert_eq!(out.dims(), &[1, cfg.embed_dim]);
    }
}
