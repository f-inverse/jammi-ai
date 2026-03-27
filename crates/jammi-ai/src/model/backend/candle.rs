use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::modernbert::{
    Config as ModernBertConfig, ModernBert, ModernBertForSequenceClassification,
};
use jammi_engine::error::{JammiError, Result};

use super::{DeviceConfig, ModelBackend};
use crate::inference::adapter::BackendOutput;
use crate::inference::arrow_to_texts;
use crate::model::tokenizer::{BatchEncoding, TokenizerWrapper};
use crate::model::{LoadedModel, ModelDimensions, ModelTask, ResolvedModel};

/// Candle backend — loads safetensors models via candle.
pub struct CandleBackend;

/// Each architecture implements this to produce hidden states from tokenized input.
trait CandleForward: Send + Sync {
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoding: &BatchEncoding,
        device: &Device,
    ) -> Result<Tensor>;
}

/// BERT-family forward pass (bert, roberta, distilbert, camembert, xlm-roberta).
struct BertForward(BertModel);

impl CandleForward for BertForward {
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

impl CandleForward for ModernBertForward {
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

/// ModernBERT sequence classification forward pass.
/// Returns softmaxed logits of shape (batch, num_classes).
struct ModernBertClassificationForward(ModernBertForSequenceClassification);

impl CandleForward for ModernBertClassificationForward {
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

impl CandleForward for BertClassificationForward {
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
    /// Architecture-specific forward pass.
    inner: Box<dyn CandleForward>,
    /// Tokenizer for text-to-token conversion, if available.
    pub tokenizer: Option<TokenizerWrapper>,
    /// Device the model weights reside on (CPU, CUDA, or Metal).
    pub device: Device,
    /// Optional LoRA projection applied after pooling (for fine-tuned models).
    pub lora_projection: Option<crate::fine_tune::lora::LoraLinear>,
    /// Label index → label string mapping for classification models.
    id2label: Option<HashMap<u32, String>>,
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

    /// Run forward pass dispatching by task.
    pub fn forward(
        &self,
        content: &[arrow::array::ArrayRef],
        task: ModelTask,
    ) -> Result<BackendOutput> {
        match task {
            ModelTask::Embedding => self.forward_embedding(content),
            ModelTask::Classification => self.forward_classification(content),
            other => Err(JammiError::Inference(format!(
                "Candle forward pass not implemented for task {other:?}."
            ))),
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

            let output =
                self.inner
                    .forward_hidden(&input_ids, &attention_mask, &encoding, &self.device)?;

            let pooled = self.mean_pool(&output, &attention_mask)?;
            let normalized = self.l2_normalize(&pooled)?;

            // Apply LoRA projection if this is a fine-tuned model
            let final_output = if let Some(ref lora) = self.lora_projection {
                lora.forward(&normalized)
                    .map_err(|e| JammiError::Inference(format!("LoRA projection: {e}")))?
            } else {
                normalized
            };

            let embeddings = final_output
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
            let logits =
                self.inner
                    .forward_hidden(&input_ids, &attention_mask, &encoding, &self.device)?;

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

        let inner: Box<dyn CandleForward> = match model_type {
            "bert" | "roberta" | "distilbert" | "camembert" | "xlm-roberta"
                if is_classification =>
            {
                let bert_config: BertConfig = serde_json::from_value(resolved.model_config.clone())
                    .map_err(|e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to parse BERT config: {e}"),
                    })?;
                let bert =
                    BertModel::load(vb.clone(), &bert_config).map_err(|e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to construct BERT model: {e}"),
                    })?;
                let num_classes = id2label.as_ref().map_or(2, |m| m.len());
                let hidden_size = bert_config.hidden_size;
                let classifier = candle_nn::linear(hidden_size, num_classes, vb.pp("classifier"))
                    .map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to load BERT classifier head: {e}"),
                })?;
                Box::new(BertClassificationForward { bert, classifier })
            }
            "bert" | "roberta" | "distilbert" | "camembert" | "xlm-roberta" => {
                let bert_config: BertConfig = serde_json::from_value(resolved.model_config.clone())
                    .map_err(|e| JammiError::Model {
                        model_id: resolved.model_id.0.clone(),
                        message: format!("Failed to parse BERT config: {e}"),
                    })?;
                let bert = BertModel::load(vb, &bert_config).map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to construct BERT model: {e}"),
                })?;
                Box::new(BertForward(bert))
            }
            "modernbert" if is_classification => {
                let mb_config: ModernBertConfig =
                    serde_json::from_value(resolved.model_config.clone()).map_err(|e| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse ModernBERT config: {e}"),
                        }
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
                let mb_config: ModernBertConfig =
                    serde_json::from_value(resolved.model_config.clone()).map_err(|e| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Failed to parse ModernBERT config: {e}"),
                        }
                    })?;
                let model = ModernBert::load(vb, &mb_config).map_err(|e| JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!("Failed to construct ModernBERT model: {e}"),
                })?;
                Box::new(ModernBertForward(model))
            }
            unsupported => {
                return Err(JammiError::Model {
                    model_id: resolved.model_id.0.clone(),
                    message: format!(
                        "Unsupported model architecture '{unsupported}'. \
                         Supported: bert, roberta, distilbert, camembert, xlm-roberta, modernbert"
                    ),
                });
            }
        };

        let tokenizer = resolved
            .tokenizer_path
            .as_ref()
            .map(|p| TokenizerWrapper::from_file(p))
            .transpose()?;

        let dimensions = ModelDimensions::from_config(&resolved.model_config).ok_or_else(|| {
            JammiError::Model {
                model_id: resolved.model_id.0.clone(),
                message: "Could not parse model dimensions from config".into(),
            }
        })?;

        // Load LoRA adapter if present (fine-tuned models)
        let lora_projection =
            if let Some(ref adapter_path) = resolved.adapter_path {
                let adapter_file = adapter_path.join("adapter.safetensors");
                if adapter_file.exists() {
                    let adapter_weights =
                        crate::fine_tune::lora::load_lora_weights(&adapter_file, &device).map_err(
                            |e| JammiError::Model {
                                model_id: resolved.model_id.0.clone(),
                                message: format!("Load adapter: {e}"),
                            },
                        )?;

                    let hidden_size = dimensions.hidden_size;
                    let identity = Tensor::eye(hidden_size, DType::F32, &device).map_err(|e| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: format!("Identity weight: {e}"),
                        }
                    })?;
                    let base_linear = candle_nn::Linear::new(identity, None);

                    let lora_a = adapter_weights.get("projection.lora_a").ok_or_else(|| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: "Adapter missing projection.lora_a".into(),
                        }
                    })?;
                    let lora_b = adapter_weights.get("projection.lora_b").ok_or_else(|| {
                        JammiError::Model {
                            model_id: resolved.model_id.0.clone(),
                            message: "Adapter missing projection.lora_b".into(),
                        }
                    })?;

                    Some(crate::fine_tune::lora::LoraLinear::from_loaded(
                        base_linear,
                        lora_a.clone(),
                        lora_b.clone(),
                        16.0, // default alpha — matches FineTuneConfig::default()
                    ))
                } else {
                    None
                }
            } else {
                None
            };

        Ok(LoadedModel::Candle(Box::new(CandleModel {
            dimensions,
            inner,
            tokenizer,
            device,
            lora_projection,
            id2label,
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

fn select_device(config: &DeviceConfig) -> Device {
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
