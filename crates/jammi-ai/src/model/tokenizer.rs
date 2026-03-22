use std::path::Path;

use jammi_engine::error::JammiError;

type Result<T> = std::result::Result<T, JammiError>;

/// Wraps the HuggingFace tokenizer with Jammi's batching conventions.
pub struct TokenizerWrapper {
    inner: tokenizers::Tokenizer,
}

impl TokenizerWrapper {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(path).map_err(|e| JammiError::Model {
                model_id: String::new(),
                message: format!("Failed to load tokenizer: {e}"),
            })?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        Ok(Self { inner: tokenizer })
    }

    /// Encode a batch of texts with optional truncation.
    pub fn encode_batch(&self, texts: &[&str], max_length: Option<usize>) -> Result<BatchEncoding> {
        if let Some(max_len) = max_length {
            let mut tokenizer = self.inner.clone();
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: max_len,
                    ..Default::default()
                }))
                .map_err(|e| JammiError::Inference(e.to_string()))?;
            Self::do_encode(&tokenizer, texts)
        } else {
            Self::do_encode(&self.inner, texts)
        }
    }

    fn do_encode(tokenizer: &tokenizers::Tokenizer, texts: &[&str]) -> Result<BatchEncoding> {
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| JammiError::Inference(e.to_string()))?;

        Ok(BatchEncoding {
            input_ids: encodings.iter().map(|e| e.get_ids().to_vec()).collect(),
            attention_masks: encodings
                .iter()
                .map(|e| e.get_attention_mask().to_vec())
                .collect(),
            type_ids: encodings
                .iter()
                .map(|e| e.get_type_ids().to_vec())
                .collect(),
            seq_len: encodings.first().map_or(0, |e| e.len()),
        })
    }
}

pub struct BatchEncoding {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_masks: Vec<Vec<u32>>,
    pub type_ids: Vec<Vec<u32>>,
    pub seq_len: usize,
}
