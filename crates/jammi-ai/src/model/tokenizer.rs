use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, PoisonError, RwLock};

use jammi_db::error::JammiError;

use super::clip_bpe::load_open_clip_bpe;

type Result<T> = std::result::Result<T, JammiError>;

/// Wraps the HuggingFace `tokenizers` crate with Jammi's batching conventions.
///
/// Two source layouts are supported transparently:
///   * HuggingFace `tokenizer.json` (any architecture) via [`Self::from_file`].
///   * OpenCLIP's native `bpe_simple_vocab_16e6.txt.gz` via
///     [`Self::from_open_clip_bpe`] — used when an OpenCLIP repo ships the
///     legacy vocab file instead of a converted `tokenizer.json`.
///
/// Truncation is a property of the `tokenizers::Tokenizer` value, not of a
/// call, so one tokenizer is held per truncation length this wrapper has
/// been asked for, built on first use and shared by every later call. A
/// `tokenizers` model's word cache lives in that value (a BPE clone starts
/// with an empty one), so holding the value is also what lets the cache warm
/// across batches.
pub struct TokenizerWrapper {
    /// Untruncated, batch-longest padding.
    inner: Arc<tokenizers::Tokenizer>,
    /// `inner` at each truncation length asked for so far.
    truncated: RwLock<HashMap<usize, Arc<tokenizers::Tokenizer>>>,
}

impl TokenizerWrapper {
    /// Load a tokenizer from a `tokenizer.json` file with batch-longest padding.
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
        Ok(Self::from_tokenizer(tokenizer))
    }

    /// Load an OpenCLIP-native BPE tokenizer from
    /// `bpe_simple_vocab_16e6.txt.gz`. Produces a tokenizer that wraps every
    /// sequence in `<|startoftext|> ... <|endoftext|>` so the EOT-pool path
    /// in the OpenCLIP text tower finds the EOT marker at the correct index.
    pub fn from_open_clip_bpe(path: &Path) -> Result<Self> {
        Ok(Self::from_tokenizer(load_open_clip_bpe(path)?))
    }

    /// Wrap an already-built tokenizer as it is: its padding and every other
    /// setting are the caller's.
    pub fn from_tokenizer(inner: tokenizers::Tokenizer) -> Self {
        Self {
            inner: Arc::new(inner),
            truncated: RwLock::new(HashMap::new()),
        }
    }

    /// The tokenizer truncating at `max_length`, built once per length.
    fn truncated_at(&self, max_length: usize) -> Result<Arc<tokenizers::Tokenizer>> {
        if let Some(tokenizer) = self
            .truncated
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&max_length)
        {
            return Ok(Arc::clone(tokenizer));
        }
        let mut tokenizer = (*self.inner).clone();
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| JammiError::Inference(e.to_string()))?;
        let tokenizer = Arc::new(tokenizer);
        // Two callers racing on the first use each build one; the first to
        // insert wins and both encode identically, since truncation is the
        // only difference between the two values.
        Ok(Arc::clone(
            self.truncated
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(max_length)
                .or_insert(tokenizer),
        ))
    }

    /// Encode a batch of texts with optional truncation.
    pub fn encode_batch(&self, texts: &[&str], max_length: Option<usize>) -> Result<BatchEncoding> {
        let tokenizer = match max_length {
            Some(max_length) => self.truncated_at(max_length)?,
            None => Arc::clone(&self.inner),
        };
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| JammiError::Inference(e.to_string()))?;
        Ok(BatchEncoding {
            input_ids: encodings.iter().map(|e| e.get_ids().to_vec()).collect(),
            attention_masks: encodings
                .iter()
                .map(|e| e.get_attention_mask().to_vec())
                .collect(),
            offsets: encodings.iter().map(|e| e.get_offsets().to_vec()).collect(),
            seq_len: encodings.first().map_or(0, |e| e.len()),
        })
    }
}

/// Tokenized batch output with padding applied.
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    /// Token IDs for each input text, padded to equal length.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention masks (1 = real token, 0 = padding).
    pub attention_masks: Vec<Vec<u32>>,
    /// Character byte offsets per token: `(start, end)`. Special tokens have `(0, 0)`.
    pub offsets: Vec<Vec<(usize, usize)>>,
    /// Padded sequence length (columns in the batch).
    pub seq_len: usize,
}

impl BatchEncoding {
    /// The real (unpadded) token count of each row: its attention-mask sum.
    pub fn row_lengths(&self) -> Vec<u32> {
        self.attention_masks
            .iter()
            .map(|mask| mask.iter().sum())
            .collect()
    }

    /// Extend every row to `width` columns with the pad id (`0`, the value
    /// `tokenizers`' own `BatchLongest` padding uses) and a fully-masked
    /// attention entry (`0`), so the extra positions contribute nothing to
    /// any pooled output and the batch runs at a width of the caller's
    /// choosing — a [`jammi_numerics::ShapeLadder`] rung. A `width` at or
    /// below the current one changes nothing: padding never truncates.
    pub fn pad_to(&mut self, width: usize) {
        if width <= self.seq_len {
            return;
        }
        for rows in [&mut self.input_ids, &mut self.attention_masks] {
            for row in rows.iter_mut() {
                row.resize(width, 0);
            }
        }
        for row in &mut self.offsets {
            row.resize(width, (0, 0));
        }
        self.seq_len = width;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_rows() -> BatchEncoding {
        BatchEncoding {
            input_ids: vec![vec![7, 8, 0], vec![4, 5, 6]],
            attention_masks: vec![vec![1, 1, 0], vec![1, 1, 1]],
            offsets: vec![vec![(0, 1), (2, 3), (0, 0)], vec![(0, 1), (2, 3), (4, 5)]],
            seq_len: 3,
        }
    }

    #[test]
    fn row_lengths_are_the_unpadded_token_counts() {
        assert_eq!(two_rows().row_lengths(), vec![2, 3]);
    }

    /// Padding extends every row with the pad id and a masked entry, never
    /// truncates, and leaves the real lengths untouched.
    #[test]
    fn pad_to_extends_every_row_and_never_truncates() {
        let mut encoding = two_rows();
        encoding.pad_to(8);
        assert_eq!(encoding.seq_len, 8);
        assert_eq!(encoding.input_ids[0], vec![7, 8, 0, 0, 0, 0, 0, 0]);
        assert_eq!(encoding.attention_masks[1], vec![1, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(encoding.offsets[1].len(), 8);
        assert_eq!(encoding.row_lengths(), vec![2, 3]);
        encoding.pad_to(4);
        assert_eq!(encoding.seq_len, 8);
        assert_eq!(encoding.input_ids[0].len(), 8);
    }
}
