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
    /// Pairs: anchor and positive embeddings only. In-batch negatives are the
    /// other rows' positives, so no explicit negative column is carried — the
    /// `MultipleNegativesRanking` objective scores each anchor against every
    /// positive in the batch.
    Pairs { anchors: Tensor, positives: Tensor },
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
    /// Regression (S18): the distributional head's raw output plus the observed
    /// continuous targets. `input` is `(batch, k)` — the unconstrained head
    /// parameters (`k = 2` `(mean, raw_std)` for the Gaussian objectives,
    /// `k = levels` for the pinball objective); `target` is `(batch,)` the
    /// observed `y`. The proper-scoring loss reads a positive `σ` from
    /// `raw_std` via `floor + softplus`, so the head trains in the
    /// unconstrained space.
    Regression { input: Tensor, target: Tensor },
}

/// Format of training data, detected from column names.
#[derive(Debug, Clone, Copy)]
pub enum TrainingFormat {
    /// `text_a, text_b, score` — contrastive pairs with scores.
    Contrastive,
    /// `anchor, positive` — contrastive pairs with no explicit negatives. The
    /// `MultipleNegativesRanking` objective draws negatives from the rest of
    /// the batch.
    Pairs,
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
    /// Regression (S18): `text, target` rows — one input text and one observed
    /// continuous outcome. The trainer encodes the text through the frozen base
    /// model + the distributional projection head, then scores the head's
    /// parameters against the target with the configured proper-scoring
    /// objective.
    Regression,
    /// Graph-supervised (S11): the rows were sampled from a graph (node text +
    /// edge table) by biased random walks into `(anchor, positive,
    /// [hard_negative])` pairs. It carries **no new loss** — it is a
    /// data-loading shape that drives the existing in-batch-negative
    /// (`Pairs`/MNRL) or `Triplet` path, selected by `has_negatives`:
    /// `false` → `Pairs` (in-batch negatives only), `true` → `Triplet` (the
    /// sampler mined structure-aware hard negatives). The variant is retained
    /// for provenance — a consumer can see the supervision came from the graph —
    /// while every downstream step reuses the Pairs/Triplet machinery.
    Graph { has_negatives: bool },
}

/// The concrete batch/loss shape a [`TrainingFormat`] resolves to once the
/// provenance-carrying `Graph` variant is mapped onto its in-batch-negative
/// shape. There is no `Graph` here by construction — a graph loader trains as
/// `Pairs` or `Triplet`, so the chunk/loss dispatch matches on this exhaustive
/// set without a phantom arm.
#[derive(Debug, Clone, Copy)]
enum UnderlyingFormat {
    Contrastive,
    Pairs,
    Triplet,
    AudioTriplet,
    Classification,
    Ner,
    Regression,
}

impl TrainingFormat {
    /// The concrete shape a format trains as: a graph with mined hard negatives
    /// is a `Triplet`, one without is `Pairs`; every other format is itself.
    /// This is the single place that maps the provenance-carrying `Graph`
    /// variant onto the loss/batch machinery, so `text_chunks` /
    /// `in_batch_negative_texts` stay DRY.
    fn underlying(self) -> UnderlyingFormat {
        match self {
            TrainingFormat::Contrastive => UnderlyingFormat::Contrastive,
            TrainingFormat::Pairs => UnderlyingFormat::Pairs,
            TrainingFormat::Triplet => UnderlyingFormat::Triplet,
            TrainingFormat::AudioTriplet => UnderlyingFormat::AudioTriplet,
            TrainingFormat::Classification { .. } => UnderlyingFormat::Classification,
            TrainingFormat::Ner { .. } => UnderlyingFormat::Ner,
            TrainingFormat::Regression => UnderlyingFormat::Regression,
            TrainingFormat::Graph {
                has_negatives: true,
            } => UnderlyingFormat::Triplet,
            TrainingFormat::Graph {
                has_negatives: false,
            } => UnderlyingFormat::Pairs,
        }
    }
}

/// A chunk of text data for one training batch. The training loop encodes
/// these through the base model before computing loss.
pub enum TextChunk {
    Contrastive {
        texts_a: Vec<String>,
        texts_b: Vec<String>,
        scores: Vec<f32>,
    },
    Pairs {
        anchors: Vec<String>,
        positives: Vec<String>,
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
    /// One batch of regression rows: input texts and their observed continuous
    /// targets. The training loop encodes the texts through the base model + the
    /// distributional head, then scores the head output against `targets`.
    Regression {
        texts: Vec<String>,
        targets: Vec<f32>,
    },
}

/// The flattened in-batch-negative view of a text loader: `(anchors,
/// positives, optional explicit negatives)`. Consumed by GradCache and
/// hard-negative mining, which treat the dataset as one in-batch-negative batch.
pub type InBatchNegativeTexts = (Vec<String>, Vec<String>, Option<Vec<String>>);

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
    Pairs {
        anchor: String,
        positive: String,
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
    Regression {
        text: String,
        target: f32,
    },
}

/// Whether `loss` is a pairwise-ordering embedding objective — CoSENT,
/// AnglE, or the `None` default (which resolves to CoSENT) — the arms whose
/// loss reads only the *strict order* of the graded `score` column and is
/// therefore degenerate on a dataset (or a batch — see
/// [`super::trainer`]'s per-batch diagnostic) that carries fewer than 2
/// distinct score levels. `CosineMse` regresses onto the score directly (no
/// ordering requirement); `MultipleNegativesRanking`/`Triplet` are not
/// graded-pair objectives at all.
///
/// **A1 (PR-C round 4):** this is the conjunction of two INDEPENDENT facts —
/// [`super::EmbeddingLoss::reads_graded_scores`] ("does this objective's
/// batch carry a `score` column to order at all") and
/// [`super::EmbeddingLoss::min_batch_size`] `>= 2` ("does this objective
/// need a second row for ITS OWN reason") — composed explicitly, never one
/// derived from the other. The previous implementation used
/// `min_batch_size() >= 2` alone as a proxy for "reads graded scores", which
/// happens to answer correctly for `{CoSent, AnglE}` (an ordering pair does
/// need a second row) but WRONGLY calls `MultipleNegativesRanking` an
/// ordering objective too — MNRL's `min_batch_size() == 2` comes from
/// needing a second row to draw an in-batch NEGATIVE from, an unrelated
/// mechanism that reads no `score` column whatsoever. `reads_graded_scores`
/// closes that gap: MNRL now reads `false` for the right reason (no scores),
/// not merely because it happens to be moot in the current call path (see
/// `TrainingDataLoader::from_pairs`'s doc for why an MNRL-dispatched `Pairs`
/// batch never reaches here in practice).
fn is_ordering_embedding_objective(loss: Option<super::EmbeddingLoss>) -> bool {
    let loss = loss.unwrap_or_default();
    loss.reads_graded_scores() && loss.min_batch_size() >= 2
}

impl TrainingDataLoader {
    /// Create a loader from contrastive pair rows.
    ///
    /// Under an ORDERING embedding objective (CoSENT, AnglE — and the `None`
    /// default, which resolves to CoSENT), a dataset whose graded `score`
    /// column carries fewer than 2 distinct FINITE values is refused: the
    /// pairwise ordering loss reads only `scores[i] < scores[j]` pairs, so a
    /// single-level dataset has none — every batch would train at the
    /// objective's `log(1) = 0` floor with zero gradient, "converging"
    /// instantly and silently. A non-ordering objective (`CosineMse`
    /// regresses onto the score directly) has no such requirement, so
    /// `embedding_loss` gates whether this check applies at all — this is a
    /// data-shape refusal, not a batch_size one (see
    /// `FineTuneConfig::validate`'s companion `batch_size < 2` refusal for
    /// the complementary edge).
    ///
    /// Non-finite (NaN/±inf) scores are filtered out before the
    /// distinct-level count is taken: a NaN forms no valid
    /// `scores[i] < scores[j]` pair with anything (IEEE-754 comparison
    /// against NaN is always `false`), so it supplies no ordering signal and
    /// must not be allowed to inflate the apparent level count. In the
    /// production path `build_training_data_loader` already refuses a null
    /// or non-finite `score` cell at the Arrow-column edge (see
    /// `extract_numeric_column` in `worker.rs`), so a NaN never reaches
    /// here; this filter is defense in depth for a caller that constructs a
    /// loader directly.
    ///
    /// This refusal is a deliberate jammi divergence from
    /// `sentence_transformers`'s `CoSENTLoss`
    /// (`sentence_transformers/losses/CoSENTLoss.py`), which accepts a
    /// single-level dataset and trains it silently at the same `log(1) = 0`
    /// floor; jammi refuses at construction instead of shipping a run that
    /// "converges" on zero gradient.
    pub fn from_contrastive(
        rows: Vec<(String, String, f32)>,
        embedding_loss: Option<super::EmbeddingLoss>,
    ) -> Result<Self> {
        if is_ordering_embedding_objective(embedding_loss) {
            // `dedup_by` only merges ADJACENT equal runs, so sort first: equal
            // scores collapse to one entry regardless of input row order.
            // Non-finite scores are dropped first (see this fn's doc) so a
            // NaN can never masquerade as a distinct ordering level.
            let mut sorted: Vec<f32> = rows
                .iter()
                .map(|(_, _, s)| *s)
                .filter(|s| s.is_finite())
                .collect();
            sorted.sort_by(f32::total_cmp);
            sorted.dedup_by(|a, b| a == b);
            if sorted.len() < 2 {
                return Err(JammiError::FineTune(format!(
                    "CoSENT/AnglE need \u{2265} 2 distinct finite score levels; use CosineMse \
                     for a single-level dataset (got {} distinct finite level(s) across {} \
                     row(s))",
                    sorted.len(),
                    rows.len()
                )));
            }
        }
        Ok(Self {
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
        })
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

    /// Create a loader from regression rows (input text + observed continuous
    /// target). The trainer encodes each text through the base model and the
    /// distributional projection head, then scores the head's parameters against
    /// the target with the configured proper-scoring objective (S18).
    pub fn from_regression(rows: Vec<(String, f32)>) -> Self {
        Self {
            format: TrainingFormat::Regression,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(text, target)| TrainingRow::Regression { text, target })
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

    /// Create a loader from contrastive pair rows `(anchor, positive)`. The
    /// `MultipleNegativesRanking` objective draws negatives from the rest of
    /// each batch, so no explicit negative column is needed — and, unlike
    /// [`Self::from_contrastive`]'s `embedding_loss`-gated check, this is
    /// UNCONDITIONAL: a `Pairs` source is always trained by
    /// `MultipleNegativesRanking`'s in-batch-negative objective regardless of
    /// which `embedding_loss` the caller's config names (`compute_loss`
    /// dispatches every `TrainingBatch::Pairs` to `mnrl_loss`, never to
    /// `dispatch_contrastive_loss`). `FineTuneConfig::validate`'s own
    /// `batch_size` refusal cannot see this — it only sees the *configured*
    /// objective (e.g. `CosineMse`), never that the underlying source is
    /// `Pairs`-shaped — so a `CosineMse` config with `batch_size = 1` sails
    /// past that check and would otherwise land here and train MNRL at a
    /// single row: no other row's positive to draw an in-batch negative
    /// from, so the softmax is over one class and the loss sits at its
    /// `log(1) = 0` floor with zero gradient. `batch_size` is therefore
    /// refused HERE, at the data edge, naming the objective the source will
    /// actually train, not the one the caller configured.
    ///
    /// **`rows.len() < 2` is ALSO refused here (B3), independently of
    /// `batch_size`.** `batch_size` guards the config knob, not the data: a
    /// 0- or 1-row dataset passes any `batch_size >= min_batch` check, since
    /// chunking too few rows still just produces one too-small batch. This
    /// refusal is the data-volume half of the same fact `batch_size <
    /// min_batch` is the config-volume half of; see `TrainingLoop::run`'s
    /// post-split re-check for why this construction-time refusal alone is
    /// not sufficient (the train/validation split can shrink the train side
    /// below this floor even when the pre-split dataset cleared it).
    pub fn from_pairs(rows: Vec<(String, String)>, batch_size: usize) -> Result<Self> {
        let min_batch =
            super::EmbeddingLoss::MultipleNegativesRanking { temperature: 0.0 }.min_batch_size();
        // B3: `batch_size` guards the CONFIG knob, not the DATA. A 0- or
        // 1-row dataset sails past the `batch_size < min_batch` check below
        // regardless of how large `batch_size` is configured — chunking 1
        // row at any batch size still produces exactly one 1-row batch, with
        // no OTHER row's positive to draw an in-batch negative from, so
        // every batch of the run trains at MNRL's `log(1) = 0` floor with
        // zero gradient. Refused here, at the data edge, naming the row
        // count the caller can actually act on (add rows), distinct from
        // the `batch_size` remedy the check below names.
        if rows.len() < min_batch {
            return Err(JammiError::FineTune(format!(
                "an (anchor, positive) Pairs source needs at least {min_batch} rows total: with \
                 fewer than {min_batch} rows there is no OTHER row's positive to draw an \
                 in-batch negative from under MultipleNegativesRanking, so every batch trains at \
                 its log(1) = 0 floor with zero gradient regardless of batch_size. Got \
                 {} row(s). Add more rows.",
                rows.len()
            )));
        }
        if batch_size < min_batch {
            return Err(JammiError::FineTune(format!(
                "an (anchor, positive) Pairs source is always trained with \
                 MultipleNegativesRanking's in-batch-negative objective, regardless of the \
                 configured embedding_loss: it needs at least {min_batch} rows per batch to \
                 draw an in-batch negative from another row's positive \
                 (sentence_transformers/losses/MultipleNegativesRankingLoss.py) — with fewer \
                 than {min_batch} rows the softmax is over a single class and the loss sits at \
                 its log(1) = 0 floor with zero gradient. Got batch_size={batch_size}. Raise \
                 batch_size to at least {min_batch}."
            )));
        }
        Ok(Self {
            format: TrainingFormat::Pairs,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p)| TrainingRow::Pairs {
                        anchor: a,
                        positive: p,
                    })
                    .collect(),
            ),
        })
    }

    /// Create a loader from a graph (S11): sample the node-text + edge-table
    /// graph into `(anchor, positive, [hard_negative])` text rows by biased
    /// random walks, then store them as the underlying `Pairs` (no mined
    /// negatives) or `Triplet` (structure-mined hard negatives) rows. The loader
    /// reports [`TrainingFormat::Graph`] for provenance, but the rows feed the
    /// existing MNRL/Triplet path unchanged — S11 adds **no new loss**.
    ///
    /// The sampler enforces the text-bearing precondition (every edge endpoint
    /// must resolve to a [`super::graph_sampler::TextNode`]) and the collapse /
    /// false-negative guards, so any violation surfaces here as a typed error.
    ///
    /// **Circularity caveat:** if the edges are S9-similarity edges the
    /// supervision largely re-learns the base metric; genuine gain comes from
    /// declared / external edges (see [`super::graph_sampler`]).
    pub fn from_graph(sampler: &super::graph_sampler::GraphSampler) -> Result<Self> {
        let pairs = sampler.sample()?;
        // The whole dataset shares one shape: if any pair carries mined hard
        // negatives the format is Triplet, otherwise Pairs. The sampler emits a
        // uniform shape (hard_negatives is a single config knob), so the first
        // pair determines it; an empty dataset is already a sampler error.
        let has_negatives = pairs.first().is_some_and(|p| !p.hard_negatives.is_empty());

        let rows = pairs
            .into_iter()
            .map(|p| {
                if has_negatives {
                    // Use the first mined negative as the explicit triplet
                    // negative; the rest still contribute via in-batch negatives
                    // (MNRL appends the explicit one as an extra column).
                    let negative = p.hard_negatives.into_iter().next().ok_or_else(|| {
                        JammiError::FineTune(
                            "graph pair declared hard negatives but supplied none".into(),
                        )
                    })?;
                    Ok(TrainingRow::Triplet {
                        anchor: p.anchor,
                        positive: p.positive,
                        negative,
                    })
                } else {
                    Ok(TrainingRow::Pairs {
                        anchor: p.anchor,
                        positive: p.positive,
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            format: TrainingFormat::Graph { has_negatives },
            data: LoaderData::TextRows(rows),
        })
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
                // A `Graph` loader stores `Pairs`/`Triplet` rows, so it encodes
                // through its underlying shape — the provenance variant carries
                // no chunk shape of its own.
                .map(|chunk| match self.format.underlying() {
                    UnderlyingFormat::Contrastive => TextChunk::Contrastive {
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
                    UnderlyingFormat::Pairs => TextChunk::Pairs {
                        anchors: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Pairs { anchor, .. } => anchor.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        positives: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Pairs { positive, .. } => positive.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                    },
                    UnderlyingFormat::Triplet => TextChunk::Triplet {
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
                    UnderlyingFormat::AudioTriplet => TextChunk::AudioTriplet {
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
                    UnderlyingFormat::Classification => TextChunk::Classification {
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
                    UnderlyingFormat::Ner => TextChunk::Ner {
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
                    UnderlyingFormat::Regression => TextChunk::Regression {
                        texts: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Regression { text, .. } => text.clone(),
                                _ => String::new(),
                            })
                            .collect(),
                        targets: chunk
                            .iter()
                            .map(|r| match r {
                                TrainingRow::Regression { target, .. } => *target,
                                _ => 0.0,
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

    /// Every regression target in this loader, in row order — the whole-dataset
    /// view the trainer reduces into a fixed target scaler once before the
    /// loop. `None` for any non-regression loader (no targets to standardise) and
    /// for the precomputed test path (which supplies head/target tensors
    /// directly, not text rows).
    pub fn regression_targets(&self) -> Option<Vec<f32>> {
        if !matches!(self.format, TrainingFormat::Regression) {
            return None;
        }
        match &self.data {
            LoaderData::TextRows(rows) => Some(
                rows.iter()
                    .filter_map(|row| match row {
                        TrainingRow::Regression { target, .. } => Some(*target),
                        _ => None,
                    })
                    .collect(),
            ),
            LoaderData::Precomputed(_) => None,
        }
    }

    /// Flatten the in-batch-negative text rows into `(anchors, positives,
    /// negatives)`, the whole-dataset view GradCache and hard-negative mining
    /// consume. `negatives` is `Some` for a `Triplet` loader (explicit hard
    /// negatives) and `None` for a `Pairs` loader. Returns an error for any
    /// other format — only in-batch-negative training has this shape.
    pub fn in_batch_negative_texts(&self) -> Result<InBatchNegativeTexts> {
        let rows = match &self.data {
            LoaderData::TextRows(rows) => rows,
            LoaderData::Precomputed(_) => {
                return Err(JammiError::FineTune(
                    "GradCache requires text rows, not precomputed batches".into(),
                ))
            }
        };
        // A `Graph` loader is itself an in-batch-negative loader — it stores
        // `Pairs`/`Triplet` rows — so it resolves through `underlying()` and
        // flows into mining / GradCache exactly like a hand-built pair set.
        match self.format.underlying() {
            UnderlyingFormat::Pairs => {
                let mut anchors = Vec::with_capacity(rows.len());
                let mut positives = Vec::with_capacity(rows.len());
                for row in rows {
                    if let TrainingRow::Pairs { anchor, positive } = row {
                        anchors.push(anchor.clone());
                        positives.push(positive.clone());
                    }
                }
                Ok((anchors, positives, None))
            }
            UnderlyingFormat::Triplet => {
                let mut anchors = Vec::with_capacity(rows.len());
                let mut positives = Vec::with_capacity(rows.len());
                let mut negatives = Vec::with_capacity(rows.len());
                for row in rows {
                    if let TrainingRow::Triplet {
                        anchor,
                        positive,
                        negative,
                    } = row
                    {
                        anchors.push(anchor.clone());
                        positives.push(positive.clone());
                        negatives.push(negative.clone());
                    }
                }
                Ok((anchors, positives, Some(negatives)))
            }
            other => Err(JammiError::FineTune(format!(
                "GradCache applies only to in-batch-negative formats (pairs/triplet), not {other:?}"
            ))),
        }
    }

    /// Whether this loader was constructed from pre-built tensor batches
    /// (typically a test fixture) rather than text rows that must be
    /// encoded through a model.
    pub fn is_precomputed(&self) -> bool {
        matches!(self.data, LoaderData::Precomputed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A regression loader carries `TrainingFormat::Regression` and chunks its
    /// rows into `TextChunk::Regression { texts, targets }` — the shape the
    /// trainer encodes through the distributional head. Pins the S18 data path
    /// from constructor to chunk.
    #[test]
    fn regression_loader_chunks_into_regression_text_chunks() {
        let loader = TrainingDataLoader::from_regression(vec![
            ("cheap".into(), 0.1),
            ("mid".into(), 0.5),
            ("dear".into(), 0.9),
        ]);
        assert!(matches!(loader.format(), TrainingFormat::Regression));
        assert_eq!(loader.len(), 3);

        let chunks = loader.text_chunks(2);
        assert_eq!(chunks.len(), 2, "3 rows at batch 2 → two chunks");
        match &chunks[0] {
            TextChunk::Regression { texts, targets } => {
                assert_eq!(texts, &["cheap".to_string(), "mid".to_string()]);
                assert_eq!(targets, &[0.1, 0.5]);
            }
            _ => panic!("regression loader must yield a Regression chunk"),
        }
        match &chunks[1] {
            TextChunk::Regression { texts, targets } => {
                assert_eq!(texts, &["dear".to_string()]);
                assert_eq!(targets, &[0.9]);
            }
            _ => panic!("regression loader must yield a Regression chunk"),
        }
    }

    /// The validation split preserves the regression format on both halves.
    #[test]
    fn regression_split_keeps_format() {
        let loader = TrainingDataLoader::from_regression(
            (0..10).map(|i| (format!("r{i}"), i as f32)).collect(),
        );
        let (train, val) = loader.split(0.2).unwrap();
        assert!(matches!(train.format(), TrainingFormat::Regression));
        assert!(matches!(val.format(), TrainingFormat::Regression));
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    // ── B3(b): an ordering objective needs ≥ 2 distinct score levels ─────────

    /// A single-level contrastive dataset (every row scored identically) under
    /// the DEFAULT (`None`, which resolves to CoSENT) objective is refused:
    /// CoSENT's pairwise ordering loss reads only `scores[i] < scores[j]`
    /// pairs, and a single-level dataset has none.
    #[test]
    fn single_level_scores_under_default_ordering_objective_is_refused() {
        let rows = vec![
            ("a1".to_string(), "b1".to_string(), 0.5f32),
            ("a2".to_string(), "b2".to_string(), 0.5f32),
            ("a3".to_string(), "b3".to_string(), 0.5f32),
        ];
        // `TrainingDataLoader` does not derive `Debug` (it can hold a
        // `Tensor`-bearing `TrainingBatch`), so `expect_err`/`unwrap_err`
        // (which require `T: Debug`) do not apply — match instead.
        let err = match TrainingDataLoader::from_contrastive(rows, None) {
            Err(e) => e,
            Ok(_) => panic!("a single-level dataset under the ordering default must be refused"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("CoSENT") && msg.contains("2"),
            "the error must name the objective and the minimum distinct level count: {msg}"
        );
    }

    /// Same refusal under explicit `Some(AnglE)`.
    #[test]
    fn single_level_scores_under_angle_is_refused() {
        let rows = vec![
            ("a1".to_string(), "b1".to_string(), 1.0f32),
            ("a2".to_string(), "b2".to_string(), 1.0f32),
        ];
        let err = match TrainingDataLoader::from_contrastive(
            rows,
            Some(super::super::EmbeddingLoss::AnglE),
        ) {
            Err(e) => e,
            Ok(_) => panic!("a single-level dataset under AnglE must be refused"),
        };
        assert!(err.to_string().contains("AnglE") || err.to_string().contains("CoSENT"));
    }

    /// Positive control: the refusal is narrow. A single-level dataset is
    /// legal under a non-ordering objective (`CosineMse`), and a
    /// multi-level dataset is legal under every ordering objective — only the
    /// `(ordering objective, < 2 distinct levels)` combination is refused.
    /// Row order is irrelevant (the distinct-level count, not adjacency,
    /// gates the refusal).
    #[test]
    fn ordering_objective_score_diversity_refusal_is_narrow() {
        let single_level = vec![
            ("a1".to_string(), "b1".to_string(), 0.5f32),
            ("a2".to_string(), "b2".to_string(), 0.5f32),
        ];
        TrainingDataLoader::from_contrastive(
            single_level,
            Some(super::super::EmbeddingLoss::CosineMse),
        )
        .expect("a single-level dataset is legal for a non-ordering objective");

        // Two distinct levels, interleaved with duplicates, in NON-adjacent
        // order — the sort-then-dedup check must still find 2 distinct levels.
        let multi_level = vec![
            ("a1".to_string(), "b1".to_string(), 0.9f32),
            ("a2".to_string(), "b2".to_string(), 0.1f32),
            ("a3".to_string(), "b3".to_string(), 0.9f32),
            ("a4".to_string(), "b4".to_string(), 0.1f32),
        ];
        for loss in [
            None,
            Some(super::super::EmbeddingLoss::CoSent),
            Some(super::super::EmbeddingLoss::AnglE),
        ] {
            TrainingDataLoader::from_contrastive(multi_level.clone(), loss).unwrap_or_else(|e| {
                panic!("a 2-level dataset must satisfy every ordering objective ({loss:?}): {e}")
            });
        }
    }

    // ── defense in depth: NaN never inflates the distinct-level count ────────
    //
    // The production path (`build_training_data_loader` in `worker.rs`)
    // already refuses a null/non-finite `score` cell before it ever reaches
    // `from_contrastive`. These pin `from_contrastive` itself, for a caller
    // that constructs a loader directly, bypassing that Arrow-column edge.

    /// An all-NaN score column must be refused exactly like an all-single-value
    /// one: a plain `dedup_by(|a, b| a == b)` would never collapse NaN
    /// (`NaN == NaN` is `false`), so each NaN would count as its own
    /// "distinct" level and let 3 all-NaN rows sail past the `>= 2` check.
    /// Filtering non-finite scores first (this fn's doc) collapses them to
    /// zero finite levels instead.
    #[test]
    fn all_nan_scores_are_refused_not_treated_as_distinct_levels() {
        let rows = vec![
            ("a1".to_string(), "b1".to_string(), f32::NAN),
            ("a2".to_string(), "b2".to_string(), f32::NAN),
            ("a3".to_string(), "b3".to_string(), f32::NAN),
        ];
        let err = match TrainingDataLoader::from_contrastive(rows, None) {
            Err(e) => e,
            Ok(_) => panic!("an all-NaN score column must be refused, not admitted as diverse"),
        };
        assert!(
            err.to_string().contains("finite"),
            "the refusal must name the finite-level requirement: {err}"
        );
    }

    /// `{0.5, 0.5, NaN}`: only ONE finite level (`0.5`, appearing twice) once
    /// the NaN is filtered out — still refused, not "2 distinct values"
    /// (`0.5` and `NaN`) as a NaN-naive dedup would have counted it.
    #[test]
    fn mixed_real_and_nan_scores_count_only_finite_levels() {
        let rows = vec![
            ("a1".to_string(), "b1".to_string(), 0.5f32),
            ("a2".to_string(), "b2".to_string(), 0.5f32),
            ("a3".to_string(), "b3".to_string(), f32::NAN),
        ];
        let err = match TrainingDataLoader::from_contrastive(rows, None) {
            Err(e) => e,
            Ok(_) => {
                panic!("one finite level plus a NaN must still be refused (only 1 finite level)")
            }
        };
        assert!(err.to_string().contains("finite"));
    }

    // ── F1: a `Pairs` source is unconditionally MNRL-trained ─────────────────

    /// `from_pairs` refuses `batch_size < 2` regardless of the caller's
    /// configured `embedding_loss` — an (anchor, positive) source is always
    /// trained by `MultipleNegativesRanking`'s in-batch-negative objective,
    /// which needs another row's positive to draw a negative from.
    #[test]
    fn from_pairs_refuses_batch_size_below_mnrl_minimum() {
        let rows = vec![
            ("a1".to_string(), "p1".to_string()),
            ("a2".to_string(), "p2".to_string()),
        ];
        let err = match TrainingDataLoader::from_pairs(rows, 1) {
            Err(e) => e,
            Ok(_) => panic!("batch_size=1 must be refused for a Pairs source"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("MultipleNegativesRanking") && msg.contains("2"),
            "the error must name the actual objective the source trains and the minimum: {msg}"
        );
    }

    /// Positive control: `batch_size >= 2` is accepted.
    #[test]
    fn from_pairs_accepts_batch_size_at_or_above_minimum() {
        let rows = vec![
            ("a1".to_string(), "p1".to_string()),
            ("a2".to_string(), "p2".to_string()),
        ];
        for batch_size in [2usize, 8, 32] {
            TrainingDataLoader::from_pairs(rows.clone(), batch_size)
                .unwrap_or_else(|e| panic!("batch_size={batch_size} must be accepted: {e}"));
        }
    }

    /// B3 (PR-C round 4): a 0-row `Pairs` source is refused regardless of how
    /// large `batch_size` is configured — `batch_size` alone guards the
    /// config knob, never the data. Before this refusal, `from_pairs(vec![],
    /// 8)` returned `Ok`, and the run would only discover the problem when
    /// every batch (there are none) trained at MNRL's zero-gradient floor —
    /// silently, since an empty epoch never even logs a degenerate-batch
    /// warning.
    #[test]
    fn from_pairs_refuses_zero_rows_regardless_of_batch_size() {
        let err = match TrainingDataLoader::from_pairs(Vec::new(), 8) {
            Err(e) => e,
            Ok(_) => panic!("a 0-row Pairs source must be refused even at batch_size=8"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("at least 2 rows") && msg.contains("Got 0 row"),
            "the refusal must name the row-count floor and the actual count: {msg}"
        );
    }

    /// B3 (PR-C round 4): the 1-row sibling of the test above — the
    /// narrowest case where `batch_size` alone could never catch the defect
    /// (a single row chunks into exactly one 1-row batch at ANY
    /// `batch_size >= 1`, so `batch_size < min_batch` never fires).
    #[test]
    fn from_pairs_refuses_one_row_regardless_of_batch_size() {
        let rows = vec![("a1".to_string(), "p1".to_string())];
        let err = match TrainingDataLoader::from_pairs(rows, 32) {
            Err(e) => e,
            Ok(_) => panic!("a 1-row Pairs source must be refused even at batch_size=32"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("at least 2 rows") && msg.contains("Got 1 row"),
            "the refusal must name the row-count floor and the actual count: {msg}"
        );
    }

    // ── A4: GradCache's whole-dataset-as-one-batch degeneracy primitive ──────

    /// The primitive `TrainingLoop::run`'s GradCache arm reads to detect a
    /// degenerate whole-dataset "batch" (A4, PR-C round 4): a 1-row `Pairs`
    /// loader's `in_batch_negative_texts()` returns exactly 1 anchor and NO
    /// explicit negative column — the same `(rows < min_rows,
    /// negatives.is_none())` pair the GradCache arm gates its diagnostic
    /// counter on.
    ///
    /// Bypasses `from_pairs`'s own construction-time refusal by building the
    /// private struct literal directly (same module) — that refusal (B3) is
    /// exactly why the trainer's GradCache diagnostic is unreachable through
    /// any PUBLIC construction path today (see the `A4` comment at the
    /// GradCache call site in `trainer.rs`). This test pins the underlying
    /// primitive's behavior directly, independent of whichever guards
    /// happen to keep the wired-in trainer.rs check from firing.
    #[test]
    fn in_batch_negative_texts_flags_a_single_row_pairs_loader_as_negative_free() {
        let loader = TrainingDataLoader {
            format: TrainingFormat::Pairs,
            data: LoaderData::TextRows(vec![TrainingRow::Pairs {
                anchor: "a1".to_string(),
                positive: "p1".to_string(),
            }]),
        };
        let (anchors, positives, negatives) = loader.in_batch_negative_texts().unwrap();
        assert_eq!(anchors.len(), 1);
        assert_eq!(positives.len(), 1);
        assert!(
            negatives.is_none(),
            "a Pairs loader draws its negatives in-batch, so in_batch_negative_texts must \
             report NO explicit negative column — the fact the GradCache diagnostic gates on"
        );
    }

    /// The `Triplet` sibling: a 1-row `Triplet` loader carries an EXPLICIT
    /// negative per row, so it must never be flagged degenerate by the same
    /// rule — the GradCache arm's `gc_negatives.is_none()` gate exists
    /// precisely to exclude this case.
    #[test]
    fn in_batch_negative_texts_never_flags_a_triplet_loader_as_negative_free() {
        let loader = TrainingDataLoader {
            format: TrainingFormat::Triplet,
            data: LoaderData::TextRows(vec![TrainingRow::Triplet {
                anchor: "a1".to_string(),
                positive: "p1".to_string(),
                negative: "n1".to_string(),
            }]),
        };
        let (anchors, _positives, negatives) = loader.in_batch_negative_texts().unwrap();
        assert_eq!(anchors.len(), 1);
        assert!(
            negatives.is_some(),
            "a Triplet loader carries an explicit negative per row and must never be reported \
             as negative-free, even at 1 row"
        );
    }
}
