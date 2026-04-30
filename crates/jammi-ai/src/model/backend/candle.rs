use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::distilbert::{Config as DistilBertConfig, DistilBertModel};
use candle_transformers::models::modernbert::{
    Config as ModernBertConfig, ModernBert, ModernBertForSequenceClassification,
};
use jammi_engine::error::{JammiError, Result};

use super::open_clip_vit::{OpenClipVisionConfig, OpenClipVisionTransformer};
use super::{DeviceConfig, ModelBackend};
use crate::inference::adapter::BackendOutput;
use crate::inference::{arrow_to_images, arrow_to_texts, image_preprocess};
use crate::model::tokenizer::{BatchEncoding, TokenizerWrapper};
use crate::model::{LoadedModel, ModelDimensions, ModelTask, ResolvedModel};

/// Candle backend — loads safetensors models via candle.
pub struct CandleBackend;

/// Text architectures produce hidden states from tokenized input.
trait CandleTextForward: Send + Sync {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor>;
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

/// BERT-family forward pass (bert, roberta, distilbert, camembert, xlm-roberta).
struct BertForward(BertModel);

impl CandleTextForward for BertForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor> {
        let token_type_ids = tokens_to_tensor(&encoding.type_ids, device)?;
        self.0
            .forward(input_ids, &token_type_ids, Some(attention_mask))
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
            .forward(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("ModernBERT forward pass failed: {e}")))
    }
}

/// DistilBERT forward pass (no token_type_ids, different architecture from BERT).
struct DistilBertForward(DistilBertModel);

impl CandleTextForward for DistilBertForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.0
            .forward(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("DistilBERT forward pass failed: {e}")))
    }
}

/// DistilBERT sequence classification: encoder → CLS → pre_classifier → ReLU → classifier → softmax.
struct DistilBertClassificationForward {
    distilbert: DistilBertModel,
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
            .forward(input_ids, attention_mask)
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
struct ModernBertClassificationForward(ModernBertForSequenceClassification);

impl CandleTextForward for ModernBertClassificationForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.0.forward(input_ids, attention_mask).map_err(|e| {
            JammiError::Inference(format!("ModernBERT classification forward failed: {e}"))
        })
    }
}

/// BERT-family sequence classification forward pass.
/// Applies CLS pooling + linear classifier + softmax on top of BertModel.
struct BertClassificationForward {
    bert: BertModel,
    classifier: candle_nn::Linear,
}

impl CandleTextForward for BertClassificationForward {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor> {
        let token_type_ids = tokens_to_tensor(&encoding.type_ids, device)?;
        let hidden = self
            .bert
            .forward(input_ids, &token_type_ids, Some(attention_mask))
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
    /// Tokenizer for text-to-token conversion, if available.
    pub tokenizer: Option<TokenizerWrapper>,
    /// Device the model weights reside on (CPU, CUDA, or Metal).
    pub device: Device,
    /// Optional LoRA projection applied after pooling (for fine-tuned models).
    pub lora_projection: Option<crate::fine_tune::lora::LoraLinear>,
    /// Label index → label string mapping for classification/NER models.
    id2label: Option<HashMap<u32, String>>,
    /// Token-level classifier for NER models (applied per token, no pooling).
    ner_classifier: Option<candle_nn::Linear>,
    /// Optional deep LoRA encoder. When set, overrides the standard text forward
    /// + mean-pool + L2-norm + optional projection path for text embedding tasks.
    deep_lora_inference: Option<Box<dyn crate::fine_tune::deep_lora::DeepLoraEncoder>>,
}

impl CandleModel {
    /// Mean-pool the last hidden state using the attention mask.
    pub(crate) fn mean_pool(&self, hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
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

    /// L2-normalize each vector in a [batch, hidden_size] tensor.
    pub(crate) fn l2_normalize(&self, tensor: &Tensor) -> Result<Tensor> {
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

            let final_output = if let Some(ref deep) = self.deep_lora_inference {
                // Deep LoRA path: encoder handles pooling + L2 norm internally.
                deep.forward(&input_ids, &attention_mask)
                    .map_err(|e| JammiError::Inference(format!("Deep LoRA forward: {e}")))?
            } else {
                let output = self.text_forward()?.forward_hidden(
                    &input_ids,
                    &attention_mask,
                    &encoding,
                    &self.device,
                )?;

                let pooled = self.mean_pool(&output, &attention_mask)?;
                let normalized = self.l2_normalize(&pooled)?;

                // Apply projection-only LoRA if present
                if let Some(ref lora) = self.lora_projection {
                    lora.forward(&normalized)
                        .map_err(|e| JammiError::Inference(format!("LoRA projection: {e}")))?
                } else {
                    normalized
                }
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

                let entities = crate::inference::ner_decode::decode_bio_spans(
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
        let is_open_clip = resolved.model_config.get("model_cfg").is_some();

        // Normalize DistilBERT config fields to standard BERT names.
        // DistilBERT uses dim/n_heads/n_layers/hidden_dim instead of
        // hidden_size/num_attention_heads/num_hidden_layers/intermediate_size.
        let model_config = if model_type == "distilbert" {
            normalize_distilbert_config(&resolved.model_config)
        } else {
            resolved.model_config.clone()
        };

        // Branch: vision model (OpenCLIP) vs text model (BERT family)
        #[allow(clippy::type_complexity)]
        let (text, vision): (
            Option<Box<dyn CandleTextForward>>,
            Option<Box<dyn CandleVisionForward>>,
        ) = if is_open_clip {
            let clip_config = OpenClipVisionConfig::from_open_clip_config(&resolved.model_config)
                .map_err(|e| JammiError::Model {
                model_id: resolved.model_id.0.clone(),
                message: format!("Failed to parse OpenCLIP config: {e}"),
            })?;
            let model =
                OpenClipVisionTransformer::load(vb.pp("visual"), &clip_config).map_err(|e| {
                    JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to construct OpenCLIP ViT: {e}"),
                    }
                })?;
            (None, Some(Box::new(model) as Box<dyn CandleVisionForward>))
        } else {
            let text_inner: Box<dyn CandleTextForward> = match model_type {
                "distilbert" if is_classification => {
                    let db_config: DistilBertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse DistilBERT config: {e}"),
                        })?;
                    let distilbert =
                        DistilBertModel::load(vb.clone(), &db_config).map_err(|e| {
                            JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to construct DistilBERT model: {e}"),
                            }
                        })?;
                    let num_classes = id2label.as_ref().map_or(2, |m| m.len());
                    let hidden_size = db_config.dim;
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
                    let model =
                        DistilBertModel::load(vb, &db_config).map_err(|e| JammiError::Model {
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
                    let bert = BertModel::load(vb.clone(), &bert_config).map_err(|e| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to construct BERT model: {e}"),
                        }
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
                    let bert =
                        BertModel::load(vb, &bert_config).map_err(|e| JammiError::Model {
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
                    let model =
                        ModernBertForSequenceClassification::load(vb, &mb_config).map_err(|e| {
                            JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Failed to construct ModernBERT classifier: {e}"),
                            }
                        })?;
                    Box::new(ModernBertClassificationForward(model))
                }
                "modernbert" => {
                    let mb_config: ModernBertConfig = serde_json::from_value(model_config.clone())
                        .map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse ModernBERT config: {e}"),
                        })?;
                    let model =
                        ModernBert::load(vb, &mb_config).map_err(|e| JammiError::Model {
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
            (Some(text_inner), None)
        };

        let tokenizer = resolved
            .tokenizer_path
            .as_ref()
            .map(|p| TokenizerWrapper::from_file(p))
            .transpose()?;

        let dimensions =
            ModelDimensions::from_config(&model_config).ok_or_else(|| JammiError::Model {
                model_id: resolved.model_id.0.clone(),
                message: "Could not parse model dimensions from config".into(),
            })?;

        // Load LoRA adapter if present (fine-tuned models)
        //
        // Two paths:
        //   1. deep_lora — adapter_config.json has "adapter_type": "deep_lora"
        //      → rebuild the encoder with frozen base weights + loaded LoRA A/B.
        //   2. projection-only (legacy) — adapter.safetensors holds only the
        //      external projection layer's A/B matrices.
        let (lora_projection, deep_lora_inference) =
            if let Some(ref adapter_path) = resolved.adapter_path {
                let adapter_config_file = adapter_path.join("adapter_config.json");
                let adapter_file = adapter_path.join("adapter.safetensors");

                if adapter_config_file.exists() && adapter_file.exists() {
                    // Read and parse adapter_config.json
                    let cfg_str = std::fs::read_to_string(&adapter_config_file).map_err(|e| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Read adapter_config.json: {e}"),
                        }
                    })?;
                    let adapter_cfg: crate::fine_tune::deep_lora::DeepLoraAdapterConfig =
                        serde_json::from_str(&cfg_str).map_err(|e| JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Parse adapter_config.json: {e}"),
                        })?;

                    if adapter_cfg.adapter_type == "deep_lora" {
                        // Rebuild the deep LoRA encoder for inference.
                        // Pass adapter_file so LoRA A/B are loaded from the saved weights.
                        let dummy_varmap = candle_nn::VarMap::new();
                        let weights_paths_ref: Vec<&std::path::Path> =
                            resolved.weights_paths.iter().map(|p| p.as_path()).collect();

                        let mut encoder: Box<dyn crate::fine_tune::deep_lora::DeepLoraEncoder> =
                            match adapter_cfg.model_type.as_str() {
                                "distilbert" => Box::new(
                                    crate::fine_tune::deep_lora::distilbert::build(
                                        &weights_paths_ref,
                                        &model_config,
                                        &adapter_cfg.target_modules,
                                        &adapter_cfg.layers_to_transform,
                                        adapter_cfg.lora_rank,
                                        adapter_cfg.lora_alpha,
                                        adapter_cfg.use_rslora,
                                        None,
                                        &adapter_cfg.rank_pattern,
                                        crate::fine_tune::lora::LoraInitMode::ZerosB,
                                        &device,
                                        &dummy_varmap,
                                        Some(adapter_file.as_path()),
                                        adapter_cfg.backbone_dtype.into(),
                                    )
                                    .map_err(|e| {
                                        JammiError::Model {
                                            model_id: resolved.model_id.0.clone(),
                                            message: format!("Deep LoRA DistilBERT: {e}"),
                                        }
                                    })?,
                                ),
                                "modernbert" => Box::new(
                                    crate::fine_tune::deep_lora::modernbert::build(
                                        &weights_paths_ref,
                                        &model_config,
                                        &adapter_cfg.target_modules,
                                        &adapter_cfg.layers_to_transform,
                                        adapter_cfg.lora_rank,
                                        adapter_cfg.lora_alpha,
                                        adapter_cfg.use_rslora,
                                        None,
                                        &adapter_cfg.rank_pattern,
                                        crate::fine_tune::lora::LoraInitMode::ZerosB,
                                        &device,
                                        &dummy_varmap,
                                        Some(adapter_file.as_path()),
                                        adapter_cfg.backbone_dtype.into(),
                                    )
                                    .map_err(|e| {
                                        JammiError::Model {
                                            model_id: resolved.model_id.0.clone(),
                                            message: format!("Deep LoRA ModernBERT: {e}"),
                                        }
                                    })?,
                                ),
                                _ => Box::new(
                                    crate::fine_tune::deep_lora::bert::build(
                                        &weights_paths_ref,
                                        &model_config,
                                        &adapter_cfg.target_modules,
                                        &adapter_cfg.layers_to_transform,
                                        adapter_cfg.lora_rank,
                                        adapter_cfg.lora_alpha,
                                        adapter_cfg.use_rslora,
                                        None,
                                        &adapter_cfg.rank_pattern,
                                        crate::fine_tune::lora::LoraInitMode::ZerosB,
                                        &device,
                                        &dummy_varmap,
                                        Some(adapter_file.as_path()),
                                        adapter_cfg.backbone_dtype.into(),
                                    )
                                    .map_err(|e| {
                                        JammiError::Model {
                                            model_id: resolved.model_id.0.clone(),
                                            message: format!("Deep LoRA BERT: {e}"),
                                        }
                                    })?,
                                ),
                            };
                        // Loaded for inference: disable dropout in all LoRA layers.
                        encoder.set_training(false);
                        (None, Some(encoder))
                    } else {
                        // Legacy projection-only adapter
                        (
                            load_projection_adapter(
                                &adapter_file,
                                &device,
                                &dimensions,
                                &resolved.model_id.0,
                            )?,
                            None,
                        )
                    }
                } else if adapter_file.exists() {
                    // Legacy projection-only (no adapter_config.json)
                    (
                        load_projection_adapter(
                            &adapter_file,
                            &device,
                            &dimensions,
                            &resolved.model_id.0,
                        )?,
                        None,
                    )
                } else {
                    (None, None)
                }
            } else {
                (None, None)
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

        Ok(LoadedModel::Candle(Box::new(CandleModel {
            dimensions,
            text,
            vision,
            tokenizer,
            device,
            lora_projection,
            id2label,
            ner_classifier,
            deep_lora_inference,
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

/// Convert token ID vectors into a candle Tensor on the given device.
fn tokens_to_tensor(vecs: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let rows = vecs.len();
    let cols = vecs.first().map_or(0, |v| v.len());
    let flat: Vec<u32> = vecs.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (rows, cols), device).map_err(|e| JammiError::Inference(e.to_string()))
}

/// Load a legacy projection-only LoRA adapter from a safetensors file.
///
/// Returns the single `projection` LoraLinear, or `None` if the file cannot
/// be read or the expected keys are missing.
fn load_projection_adapter(
    adapter_file: &std::path::Path,
    device: &Device,
    dimensions: &crate::model::ModelDimensions,
    model_id: &str,
) -> jammi_engine::error::Result<Option<crate::fine_tune::lora::LoraLinear>> {
    let adapter_weights =
        crate::fine_tune::lora::load_lora_weights(adapter_file, device).map_err(|e| {
            JammiError::Model {
                model_id: model_id.to_string(),
                message: format!("Load adapter: {e}"),
            }
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

    Ok(Some(crate::fine_tune::lora::LoraLinear::from_loaded(
        base_linear,
        lora_a,
        lora_b,
        16.0, // default alpha — matches FineTuneConfig::default()
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
