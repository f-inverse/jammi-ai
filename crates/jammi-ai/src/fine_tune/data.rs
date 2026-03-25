//! Training data loader: reads from DataFusion, produces training batches.
//!
//! Two modes:
//! - **Text-based** (`from_contrastive` / `from_triplets` / `from_rows`):
//!   stores raw text. Use `text_chunks()` to get batches of text for
//!   model-in-loop training (encode through base model, project through LoRA).
//! - **Precomputed** (`from_precomputed`): stores pre-built tensor batches.
//!   `batches()` returns them as-is. Used in tests.

use candle_core::Tensor;
use jammi_engine::error::{JammiError, Result};

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
}

/// Format of training data, detected from column names.
#[derive(Debug, Clone, Copy)]
pub enum TrainingFormat {
    /// `text_a, text_b, score` — contrastive pairs with scores.
    Contrastive,
    /// `anchor, positive, negative` — triplet format.
    Triplet,
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
                })
                .collect(),
            LoaderData::Precomputed(_) => Vec::new(),
        }
    }

    /// The detected training data format.
    pub fn format(&self) -> TrainingFormat {
        self.format
    }
}
