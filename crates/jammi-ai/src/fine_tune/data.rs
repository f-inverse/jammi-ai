//! Training data loader: reads from DataFusion, produces training batches.
//!
//! Two modes:
//! - **Encode-in-loop** (`from_contrastive` / `from_triplets` /
//!   `from_audio_triplets` / `from_rows`): stores raw inputs (text strings or
//!   encoded audio clips). Use `text_chunks()` to get batches for
//!   model-in-loop training (encode through the base model, project through
//!   LoRA). Text and audio chunks differ only in how the base model turns one
//!   example into an embedding — the loss, head, and optimizer are shared.
//! - **Precomputed** (`from_precomputed`): stores pre-built tensor batches.
//!   `batches()` returns them as-is. Used in tests.

use candle_core::Tensor;
use jammi_db::error::{JammiError, Result};

/// A training batch — either contrastive pairs or triplets.
#[derive(Clone)]
pub enum TrainingBatch {
    /// Contrastive: two embedding tensors plus target similarity scores.
    Contrastive {
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        scores: Tensor,
    },
    /// Triplet: anchor, positive, and negative embeddings.
    Triplet {
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    },
    /// Classification: embeddings + integer class labels.
    Classification {
        embeddings: Tensor,
        labels: Tensor, // shape (batch_size,) u32
    },
    /// NER: hidden states for all tokens + per-token labels.
    Ner {
        hidden_states: Tensor, // (batch, seq_len, hidden)
        labels: Tensor,        // (batch, seq_len) as i64, -100 for ignored tokens
    },
}

/// Format of training data, detected from column names.
#[derive(Debug, Clone, Copy)]
pub enum TrainingFormat {
    /// `text_a, text_b, score` — contrastive pairs with scores.
    Contrastive,
    /// `anchor, positive, negative` — text triplet format.
    Triplet,
    /// `anchor, positive, negative` — audio triplet format. The three
    /// columns carry encoded audio clips (WAV/FLAC/MP3/Ogg bytes), not text.
    /// What makes a clip a "positive" (augmentation-similar or
    /// co-occurring-complementary) is the caller's data, not the trainer's
    /// concern — the loss only minimizes the triplet objective over whatever
    /// clips the caller paired.
    AudioTriplet,
    /// Classification with label-to-index mapping.
    Classification { num_classes: usize },
    /// NER with BIO tag mapping.
    Ner { num_labels: usize },
}

/// A chunk of text data for one training batch. The training loop encodes
/// these through the base model before computing loss.
pub enum TextChunk {
    Contrastive {
        texts_a: Vec<String>,
        texts_b: Vec<String>,
        scores: Vec<f32>,
    },
    Triplet {
        anchors: Vec<String>,
        positives: Vec<String>,
        negatives: Vec<String>,
    },
    /// One batch of audio triplets. Each clip is encoded audio bytes the base
    /// audio model decodes itself; the training loop runs them through the
    /// frozen audio encoder, then the LoRA projection head, exactly as the
    /// text path does for [`TextChunk::Triplet`].
    AudioTriplet {
        anchors: Vec<Vec<u8>>,
        positives: Vec<Vec<u8>>,
        negatives: Vec<Vec<u8>>,
    },
    Classification {
        texts: Vec<String>,
        labels: Vec<u32>,
    },
    Ner {
        texts: Vec<String>,
        /// Per-text entity spans as JSON strings (same format as inference output).
        entities_json: Vec<String>,
    },
}

/// Internal storage: either text rows (from source) or precomputed batches (for tests).
enum LoaderData {
    TextRows(Vec<TrainingRow>),
    Precomputed(Vec<TrainingBatch>),
}

/// Loads training data and produces batches of tensors.
///
/// Two construction modes:
/// - `from_contrastive` / `from_triplets` / `from_rows`: stores text.
///   Use `text_chunks()` for model-in-loop training.
/// - `from_precomputed`: stores pre-built batches, `batches()` returns them as-is.
///   Used in tests to exercise divergence detection, early stopping, etc.
pub struct TrainingDataLoader {
    format: TrainingFormat,
    data: LoaderData,
}

/// Text data for one training example.
#[derive(Clone)]
enum TrainingRow {
    Contrastive {
        text_a: String,
        text_b: String,
        score: f32,
    },
    Triplet {
        anchor: String,
        positive: String,
        negative: String,
    },
    AudioTriplet {
        anchor: Vec<u8>,
        positive: Vec<u8>,
        negative: Vec<u8>,
    },
    Classification {
        text: String,
        label: u32,
    },
    Ner {
        text: String,
        /// JSON-serialized entity spans.
        entities_json: String,
    },
}

impl TrainingDataLoader {
    /// Create a loader from contrastive pair rows.
    pub fn from_contrastive(rows: Vec<(String, String, f32)>) -> Self {
        Self {
            format: TrainingFormat::Contrastive,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, b, s)| TrainingRow::Contrastive {
                        text_a: a,
                        text_b: b,
                        score: s,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from classification rows (text + integer label).
    pub fn from_classification(rows: Vec<(String, u32)>, num_classes: usize) -> Self {
        Self {
            format: TrainingFormat::Classification { num_classes },
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(text, label)| TrainingRow::Classification { text, label })
                    .collect(),
            ),
        }
    }

    /// Create a loader from NER rows (text + JSON entity spans).
    pub fn from_ner(rows: Vec<(String, String)>, num_labels: usize) -> Self {
        Self {
            format: TrainingFormat::Ner { num_labels },
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(text, entities_json)| TrainingRow::Ner {
                        text,
                        entities_json,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from triplet rows.
    pub fn from_triplets(rows: Vec<(String, String, String)>) -> Self {
        Self {
            format: TrainingFormat::Triplet,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p, n)| TrainingRow::Triplet {
                        anchor: a,
                        positive: p,
                        negative: n,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from audio triplet rows. Each element is
    /// `(anchor_bytes, positive_bytes, negative_bytes)` where every field is
    /// one encoded audio clip. The trainer encodes these through the frozen
    /// audio base model and the LoRA projection head; the contrastive
    /// objective is identical to the text triplet path.
    pub fn from_audio_triplets(rows: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>) -> Self {
        Self {
            format: TrainingFormat::AudioTriplet,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p, n)| TrainingRow::AudioTriplet {
                        anchor: a,
                        positive: p,
                        negative: n,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader with N synthetic rows (for testing validation split logic).
    pub fn from_rows(count: usize) -> Self {
        Self {
            format: TrainingFormat::Contrastive,
            data: LoaderData::TextRows(
                (0..count)
                    .map(|i| TrainingRow::Contrastive {
                        text_a: format!("text_a_{i}"),
                        text_b: format!("text_b_{i}"),
                        score: 0.5,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from pre-built batches. `batches()` returns clones of
    /// these directly instead of generating placeholder tensors. Used in tests
    /// to supply crafted tensors that trigger specific training behaviors
    /// (divergence, early stopping, etc.).
    pub fn from_precomputed(batches: Vec<TrainingBatch>) -> Self {
        Self {
            format: TrainingFormat::Contrastive,
            data: LoaderData::Precomputed(batches),
        }
    }

    /// Total number of data points (rows for text, batches for precomputed).
    pub fn len(&self) -> usize {
        match &self.data {
            LoaderData::TextRows(rows) => rows.len(),
            LoaderData::Precomputed(batches) => batches.len(),
        }
    }

    /// Whether the loader has no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of batches at the given batch size.
    pub fn num_batches(&self, batch_size: usize) -> usize {
        match &self.data {
            LoaderData::TextRows(rows) => {
                if batch_size == 0 || rows.is_empty() {
                    0
                } else {
                    rows.len().div_ceil(batch_size)
                }
            }
            LoaderData::Precomputed(batches) => batches.len(),
        }
    }

    /// Deterministic split: last `fraction` of data goes to validation.
    pub fn split(&self, fraction: f64) -> Result<(TrainingDataLoader, TrainingDataLoader)> {
        match &self.data {
            LoaderData::TextRows(rows) => {
                let val_count = (rows.len() as f64 * fraction).round() as usize;
                let train_count = rows.len() - val_count;
                Ok((
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::TextRows(rows[..train_count].to_vec()),
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::TextRows(rows[train_count..].to_vec()),
                    },
                ))
            }
            LoaderData::Precomputed(batches) => {
                let val_count = (batches.len() as f64 * fraction).round() as usize;
                let train_count = batches.len() - val_count;
                Ok((
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Precomputed(batches[..train_count].to_vec()),
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Precomputed(batches[train_count..].to_vec()),
                    },
                ))
            }
        }
    }

    /// Indices of rows in this loader (for testing split logic).
    pub fn indices(&self) -> impl Iterator<Item = usize> {
        0..self.len()
    }

    /// Produce training batches (precomputed mode only).
    ///
    /// For text-based loaders: returns an error — use `text_chunks()` instead.
    /// For precomputed loaders: returns the pre-built batches.
    pub fn batches(&self, _batch_size: usize) -> Result<Vec<Result<TrainingBatch>>> {
        match &self.data {
            LoaderData::TextRows(_) => Err(JammiError::FineTune(
                "Text-based loaders require model-in-loop encoding. Use text_chunks() instead."
                    .into(),
            )),
            LoaderData::Precomputed(batches) => Ok(batches.iter().map(|b| Ok(b.clone())).collect()),
        }
    }

    /// Produce text chunks for model-in-loop training. Each chunk is one
    /// batch of text data to be encoded through the base model.
    /// Only works for text-based loaders (from_contrastive/from_triplets/from_rows).
    /// Returns empty for precomputed loaders.
    pub fn text_chunks(&self, batch_size: usize) -> Vec<TextChunk> {
        match &self.data {
            LoaderData::TextRows(rows) => rows
                .chunks(batch_size)
                .map(|chunk| match self.format {
                    TrainingFormat::Contrastive => TextChunk::Contrastive {
                        texts_a: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Contrastive { text_a, .. } => text_a.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        texts_b: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Contrastive { text_b, .. } => text_b.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        scores: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Contrastive { score, .. } => *score,
                                _ => 0.0,
                            })
                            .collect(),
                    },
                    TrainingFormat::Triplet => TextChunk::Triplet {
                        anchors: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Triplet { anchor, .. } => anchor.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        positives: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Triplet { positive, .. } => positive.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        negatives: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Triplet { negative, .. } => negative.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                    },
                    TrainingFormat::AudioTriplet => TextChunk::AudioTriplet {
                        anchors: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::AudioTriplet { anchor, .. } => anchor.clone(),
                                _ => Vec::new(),
                            })
                            .collect(),
                        positives: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::AudioTriplet { positive, .. } => positive.clone(),
                                _ => Vec::new(),
                            })
                            .collect(),
                        negatives: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::AudioTriplet { negative, .. } => negative.clone(),
                                _ => Vec::new(),
                            })
                            .collect(),
                    },
                    TrainingFormat::Classification { .. } => TextChunk::Classification {
                        texts: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Classification { text, .. } => text.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        labels: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Classification { label, .. } => *label,
                                _ => 0,
                            })
                            .collect(),
                    },
                    TrainingFormat::Ner { .. } => TextChunk::Ner {
                        texts: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Ner { text, .. } => text.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        entities_json: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Ner { entities_json, .. } => entities_json.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                    },
                })
                .collect(),
            LoaderData::Precomputed(_) => Vec::new(),
        }
    }

    /// The detected training data format.
    pub fn format(&self) -> TrainingFormat {
        self.format
    }

    /// Whether this loader was constructed from pre-built tensor batches
    /// (typically a test fixture) rather than text rows that must be
    /// encoded through a model.
    pub fn is_precomputed(&self) -> bool {
        matches!(self.data, LoaderData::Precomputed(_))
    }
}
