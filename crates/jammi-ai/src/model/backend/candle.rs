use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use jammi_db::error::{JammiError, Result};
use jammi_encoders::{
    Bert, BertConfig, DistilBert, DistilBertConfig, ModernBert, ModernBertConfig, Pooling,
};

use jammi_encoders::{ClipText, ClipTextConfig, HtsatAudio, HtsatAudioConfig};

use super::open_clip_text::OpenClipTextForward;
use super::open_clip_vit::{OpenClipVisionConfig, OpenClipVisionTransformer};
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
/// embedding path; BERT-family encoders fall through to the provided
/// default (mean-pool + L2-normalize over `forward_hidden`), while the
/// OpenCLIP text tower overrides it to expose its pre-pooled projected
/// output directly.
pub(crate) trait CandleTextForward: Send + Sync {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor>;

    /// Pooled and L2-normalized `[batch, output_dim]` embedding. Default
    /// implementation mean-pools the masked output of `forward_hidden` and
    /// L2-normalizes it; encoders whose `forward_hidden` is already pooled
    /// (e.g. OpenCLIP text) override this directly.
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

/// BERT-family forward pass (bert, roberta, camembert, xlm-roberta).
struct BertForward(Bert);

impl CandleTextForward for BertForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.0
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("BERT forward pass failed: {e}")))
    }
}

/// ModernBERT forward pass (rotary embeddings, GeGLU, no token_type_ids).
struct ModernBertForward(ModernBert);

impl CandleTextForward for ModernBertForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.0
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("ModernBERT forward pass failed: {e}")))
    }
}

/// DistilBERT forward pass (no token_type_ids, different architecture from BERT).
struct DistilBertForward(DistilBert);

impl CandleTextForward for DistilBertForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.0
            .forward_hidden(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("DistilBERT forward pass failed: {e}")))
    }
}

/// DistilBERT sequence classification: encoder → CLS → pre_classifier → ReLU → classifier → softmax.
struct DistilBertClassificationForward {
    distilbert: DistilBert,
    pre_classifier: candle_nn::Linear,
    classifier: candle_nn::Linear,
}

impl CandleTextForward for DistilBertClassificationForward {
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
    /// Label index → label string mapping for classification/NER models.
    id2label: Option<HashMap<u32, String>>,
    /// Token-level classifier for NER models (applied per token, no pooling).
    ner_classifier: Option<candle_nn::Linear>,
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

impl CandleModel {
    /// L2-normalize each vector in a [batch, hidden_size] tensor.
    pub(crate) fn l2_normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        l2_normalize(tensor)
    }

    /// Convert token ID vectors into a candle Tensor on this model's device.
    pub(crate) fn tokens_to_tensor(&self, vecs: &[Vec<u32>]) -> Result<Tensor> {
        tokens_to_tensor(vecs, &self.device)
    }

    /// Access the text forward pass, returning an error if this is a vision-only model.
    fn text_forward(&self) -> Result<&dyn CandleTextForward> {
        self.text.as_deref().ok_or_else(|| {
            JammiError::Inference("Cannot run text task on a vision-only model".into())
        })
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
        }
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
            let encoding = tokenizer.encode_batch(&valid_texts, Some(512))?;

            let input_ids = self.tokens_to_tensor(&encoding.input_ids)?;
            let attention_mask = self.tokens_to_tensor(&encoding.attention_masks)?;

            // Each encoder controls its own pooling: BERT-family uses the
            // default (mean-pool over `forward_hidden` masked by attention),
            // OpenCLIP text returns its pre-pooled projected output. The
            // result is already L2-normalized.
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
            // plus the per-clip `is_longer` flags that gate the patch-embed
            // fusion per sample (a short clip uses the global patch-conv alone).
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
            let encoding = tokenizer.encode_batch(&valid_texts, Some(512))?;

            let input_ids = self.tokens_to_tensor(&encoding.input_ids)?;
            let attention_mask = self.tokens_to_tensor(&encoding.attention_masks)?;

            // Forward pass returns (batch, num_classes) with softmax applied
            let logits = self.text_forward()?.forward_hidden(
                &input_ids,
                &attention_mask,
                &encoding,
                &self.device,
            )?;

            let probs = logits
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
            let encoding = tokenizer.encode_batch(&valid_texts, Some(512))?;

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

            let logits_vec = logits
                .to_vec3::<f32>()
                .map_err(|e| JammiError::Inference(format!("NER logits to vec failed: {e}")))?;

            for (batch_idx, &orig_idx) in valid_indices.iter().enumerate() {
                let token_logits = &logits_vec[batch_idx];
                let offsets = &encoding.offsets[batch_idx];
                let mask = &encoding.attention_masks[batch_idx];

                let entities = jammi_numerics::ner::decode_bio_spans(
                    token_logits,
                    offsets,
                    mask,
                    id2label,
                    &texts[orig_idx],
                );

                all_entities_json[orig_idx] =
                    serde_json::to_string(&entities).unwrap_or_else(|_| "[]".to_string());
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
        let device = select_device(device_config);

        let model_type = resolved
            .model_config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("bert");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&resolved.weights_paths, DType::F32, &device)
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
            },
            _ => jammi_lora::LoraBuildConfig::frozen(),
        };
        let encoder_adapter_file: Option<&std::path::Path> =
            encoder_adapter.as_ref().map(|(_, p)| *p);
        let encoder_backbone_dtype = encoder_adapter
            .as_ref()
            .map(|(cfg, _)| candle_core::DType::from(cfg.backbone_dtype))
            .unwrap_or(DType::F32);
        let weights_paths_ref: Vec<&std::path::Path> =
            resolved.weights_paths.iter().map(|p| p.as_path()).collect();
        let dummy_varmap = VarMap::new();

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
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &db_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct DistilBERT model: {e}"),
                        })?;
                    Box::new(DistilBertForward(model))
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
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &bert_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct BERT model: {e}"),
                        })?;
                    Box::new(BertForward(bert))
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
                        .pooling(Pooling::Mean)
                        .lora(lora_build)
                        .backbone_dtype(encoder_backbone_dtype)
                        .adapter(encoder_adapter_file)
                        .build(&weights_paths_ref, &mb_config, &device, &dummy_varmap)
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct ModernBERT model: {e}"),
                        })?;
                    Box::new(ModernBertForward(model))
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
        let projection_head = match saved_adapter.as_ref() {
            Some((crate::fine_tune::target::SavedAdapter::ProjectionHead(cfg), weights_path)) => {
                load_projection_head(
                    weights_path,
                    cfg.lora_alpha,
                    &device,
                    &dimensions,
                    &resolved.model_id.0,
                )?
            }
            _ => None,
        };

        // Load NER token classifier if this is a NER model
        let ner_classifier = if is_ner {
            let num_labels = id2label.as_ref().map_or(3, |m| m.len());
            let hidden_size = dimensions.hidden_size;
            // NER models use a VarBuilder scoped to the same safetensors
            let ner_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&resolved.weights_paths, DType::F32, &device)
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
            id2label,
            ner_classifier,
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

pub(crate) fn select_device(config: &DeviceConfig) -> Device {
    if config.gpu_device < 0 {
        return Device::Cpu;
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(config.gpu_device as usize) {
            return dev;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(config.gpu_device as usize) {
            return dev;
        }
    }
    Device::Cpu
}
