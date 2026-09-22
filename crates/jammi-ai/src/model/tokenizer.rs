use std::path::Path;

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
pub struct TokenizerWrapper {
    inner: tokenizers::Tokenizer,
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
        Ok(Self { inner: tokenizer })
    }

    /// Load an OpenCLIP-native BPE tokenizer from
    /// `bpe_simple_vocab_16e6.txt.gz`. Produces a tokenizer that wraps every
    /// sequence in `<|startoftext|> ... <|endoftext|>` so the EOT-pool path
    /// in the OpenCLIP text tower finds the EOT marker at the correct index.
    pub fn from_open_clip_bpe(path: &Path) -> Result<Self> {
        let inner = load_open_clip_bpe(path)?;
        Ok(Self { inner })
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
    /// Token type IDs for segment disambiguation.
    pub type_ids: Vec<Vec<u32>>,
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
        for rows in [
            &mut self.input_ids,
            &mut self.attention_masks,
            &mut self.type_ids,
        ] {
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
            type_ids: vec![vec![0; 3]; 2],
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
        assert_eq!(encoding.type_ids[0].len(), 8);
        assert_eq!(encoding.offsets[1].len(), 8);
        assert_eq!(encoding.row_lengths(), vec![2, 3]);
        encoding.pad_to(4);
        assert_eq!(encoding.seq_len, 8);
        assert_eq!(encoding.input_ids[0].len(), 8);
    }
}
