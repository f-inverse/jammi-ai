//! Training data loader: reads from DataFusion, produces training batches.
//!
//! Two modes:
//! - **Encode-in-loop** (`from_contrastive` / `from_triplets` /
//!   `from_media_triplets` / `from_rows`): stores raw inputs (text strings or
//!   encoded media blobs — audio clips or images). Use `text_chunks()` to get
//!   batches for model-in-loop training (encode through the base model,
//!   project through LoRA). Text and media chunks differ only in how the base
//!   model turns one example into an embedding — the loss, head, and
//!   optimizer are shared.
//! - **Precomputed** (`from_precomputed`): stores pre-built tensor batches.
//!   `batches()` returns them as-is. Used in tests.

use candle_core::Tensor;
use jammi_db::error::{JammiError, Result};

/// The train/validation split boundary over a row COUNT alone (#500 U2c
/// §10): the last `round(total * fraction)` rows go to validation, so the
/// train prefix is `[0, split_index(total, fraction))`. The ONE place this
/// arithmetic is spelled — [`TrainingDataLoader::split`] (over an already
/// in-memory row/batch count) and [`super::source::StreamedSet`]'s
/// `train_count` (over a catalog row count, no scan) both call this, so a
/// resident loader and a streamed source over the SAME `(total, fraction)`
/// can never disagree about where the boundary falls. Pinned for every
/// `total ∈ 0..=1000` and every fraction the config admits by
/// `source::split_index_matches_the_resident_split_boundary` in this
/// crate's test suite.
pub(crate) fn split_index(total: usize, fraction: f64) -> usize {
    let val_count = (total as f64 * fraction).round() as usize;
    total - val_count
}

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
    /// `anchor, positive, negative` — MEDIA triplet format. The three
    /// columns carry encoded binary blobs, not text: audio clips
    /// (WAV/FLAC/MP3/Ogg bytes) or images (PNG/JPEG/… bytes).
    ///
    /// The MODALITY is not carried by this variant and is not sniffed from
    /// the bytes: it is the job's own [`crate::model::ModelTask`]
    /// (`audio_embedding` / `image_embedding`), which the loader's caller
    /// already supplies and which the trainer dispatches its front end on.
    /// An encoded WAV and an encoded PNG are both `Vec<u8>`; the column
    /// shape genuinely cannot distinguish them, so the declared task is the
    /// authority rather than a byte-header guess.
    ///
    /// What makes a blob a "positive" (augmentation-similar or
    /// co-occurring-complementary) is the caller's data, not the trainer's
    /// concern — the loss only minimizes the triplet objective over whatever
    /// items the caller paired.
    MediaTriplet,
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
    MediaTriplet,
    Classification,
    Ner,
    Regression,
}

/// Every canonical training-format tag, in declaration order — the CLOSED set
/// [`TrainingFormat::format_tag`] maps onto.
///
/// The set is closed by the round-trip test below rather than by convention:
/// the tag of every variant must appear here, the entries must be pairwise
/// distinct, and [`TrainingFormat::from_format_tag`] must recover a value with
/// the same tag for each. A tag that changes is a *different* training set, so
/// a rename is a breaking change to every recorded definition hash, not a
/// cosmetic edit.
pub const TRAINING_FORMAT_TAGS: &[&str] = &[
    "contrastive",
    "pairs",
    "triplet",
    "media_triplet",
    "classification",
    "ner",
    "regression",
    "graph_pairs",
    "graph_triplet",
];

impl TrainingFormat {
    /// The canonical string tag a materialised training set records this format
    /// under — the ONE mapping from a [`TrainingFormat`] to the `format` field
    /// of
    /// [`ProducingDescriptor::TrainingSet`](jammi_db::store::manifest::ProducingDescriptor::TrainingSet).
    ///
    /// `jammi-db` depends on no jammi crate but `jammi-numerics`, so the
    /// descriptor folds a string rather than this enum, and the completeness
    /// burden lands here: **a format distinction this mapping does not spell is
    /// two different training sets colliding on one definition hash.** The match
    /// is exhaustive with no `_` arm, so a new variant cannot reach the
    /// descriptor without being given a tag.
    ///
    /// The two parameterised variants deliberately drop their parameter.
    /// `Classification`'s `num_classes` and `Ner`'s `num_labels` are *functions
    /// of the rows* — the count of distinct labels the loader observed — not
    /// independent choices a caller makes, so they cannot distinguish two
    /// training sets built from the same source, columns and task, and folding
    /// them would require reading the rows before naming the table that holds
    /// them. `Graph`'s `has_negatives` is not in that class: it changes the
    /// projected column set (a mined negative is a third column) and therefore
    /// the committed bytes, so it takes two distinct tags.
    pub fn format_tag(self) -> &'static str {
        match self {
            TrainingFormat::Contrastive => "contrastive",
            TrainingFormat::Pairs => "pairs",
            TrainingFormat::Triplet => "triplet",
            TrainingFormat::MediaTriplet => "media_triplet",
            TrainingFormat::Classification { .. } => "classification",
            TrainingFormat::Ner { .. } => "ner",
            TrainingFormat::Regression => "regression",
            TrainingFormat::Graph {
                has_negatives: false,
            } => "graph_pairs",
            TrainingFormat::Graph {
                has_negatives: true,
            } => "graph_triplet",
        }
    }

    /// The inverse of [`Self::format_tag`] over [`TRAINING_FORMAT_TAGS`]: the
    /// representative format a recorded tag names, with the two data-derived
    /// parameters at zero (the tag never carried them — see
    /// [`Self::format_tag`] — so no value of theirs is recoverable and zero is
    /// the neutral stand-in, never a claim about the rows).
    ///
    /// `None` for a tag this build does not know, which is how a table
    /// committed by a newer build is refused rather than read as some
    /// near-miss format.
    pub fn from_format_tag(tag: &str) -> Option<Self> {
        match tag {
            "contrastive" => Some(TrainingFormat::Contrastive),
            "pairs" => Some(TrainingFormat::Pairs),
            "triplet" => Some(TrainingFormat::Triplet),
            "media_triplet" => Some(TrainingFormat::MediaTriplet),
            "classification" => Some(TrainingFormat::Classification { num_classes: 0 }),
            "ner" => Some(TrainingFormat::Ner { num_labels: 0 }),
            "regression" => Some(TrainingFormat::Regression),
            "graph_pairs" => Some(TrainingFormat::Graph {
                has_negatives: false,
            }),
            "graph_triplet" => Some(TrainingFormat::Graph {
                has_negatives: true,
            }),
            _ => None,
        }
    }
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
            TrainingFormat::MediaTriplet => UnderlyingFormat::MediaTriplet,
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
#[derive(Debug, PartialEq)]
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
    /// One batch of MEDIA triplets. Each element is one encoded blob —
    /// audio clip or image, per the job's task — which the training loop
    /// runs through that modality's front end and then the tower (LoRA
    /// inside it, for an encoder-adapters run) or the frozen tower plus the
    /// LoRA projection head, exactly as the text path does for
    /// [`TextChunk::Triplet`].
    MediaTriplet {
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

impl TextChunk {
    /// The number of rows this chunk carries — every field of a well-formed
    /// chunk is the same length, so any one of them reports it. A zero here
    /// is a valid, well-formed state (a zero-row rank at the trailing global
    /// batch, DESIGN.md §2, K2), not a malformed chunk: every producer of a
    /// [`TextChunk`] (`TrainingDataLoader::rows_to_text_chunk`) builds it by
    /// `.map().collect()` over a row slice that can itself be empty.
    pub fn row_count(&self) -> usize {
        match self {
            TextChunk::Contrastive { texts_a, .. } => texts_a.len(),
            TextChunk::Pairs { anchors, .. } => anchors.len(),
            TextChunk::Triplet { anchors, .. } => anchors.len(),
            TextChunk::MediaTriplet { anchors, .. } => anchors.len(),
            TextChunk::Classification { texts, .. } => texts.len(),
            TextChunk::Ner { texts, .. } => texts.len(),
            TextChunk::Regression { texts, .. } => texts.len(),
        }
    }
}

/// The flattened in-batch-negative view of a text loader: `(anchors,
/// positives, optional explicit negatives)`. Consumed by GradCache and
/// hard-negative mining, which treat the dataset as one in-batch-negative batch.
pub type InBatchNegativeTexts = (Vec<String>, Vec<String>, Option<Vec<String>>);

/// Internal storage: text rows already resident (from source, or the tests-only
/// synthetic constructors), or precomputed batches (tests only).
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
    /// The eager collected batches' pool reservation, held for this loader's
    /// own lifetime (#500 U2c c3c, P-R) — `None` for every loader that never
    /// went through a pool-accounted collect (every `from_*` constructor
    /// below builds one this way; `training_set::read_back_with_reservation`'s
    /// caller attaches the real one via [`Self::with_reservation`]).
    /// `MemoryReservation`'s own `Drop` frees its held bytes back to the pool
    /// the instant the LAST reservation over it goes out of scope — no
    /// manual `Drop` impl is needed on this type for the same reason
    /// [`super::stream::OwnedChunk`] needs none.
    reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
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
    MediaTriplet {
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
            reservation: None,
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
            reservation: None,
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
            reservation: None,
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
            reservation: None,
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
            reservation: None,
        }
    }

    /// Create a loader from contrastive pair rows `(anchor, positive)`. The
    /// `MultipleNegativesRanking` objective draws negatives from the rest of
    /// each batch, so no explicit negative column is needed.
    pub fn from_pairs(rows: Vec<(String, String)>) -> Self {
        Self {
            format: TrainingFormat::Pairs,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p)| TrainingRow::Pairs {
                        anchor: a,
                        positive: p,
                    })
                    .collect(),
            ),
            reservation: None,
        }
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
            reservation: None,
        })
    }

    /// Create a loader from MEDIA triplet rows. Each element is
    /// `(anchor_bytes, positive_bytes, negative_bytes)` where every field is
    /// one encoded blob — an audio clip or an image, per the job's task (see
    /// [`TrainingFormat::MediaTriplet`] for why the task, not the bytes, is
    /// the modality authority). The trainer encodes these through that
    /// modality's front end and the base model; the contrastive objective is
    /// identical to the text triplet path.
    pub fn from_media_triplets(rows: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>) -> Self {
        Self {
            format: TrainingFormat::MediaTriplet,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p, n)| TrainingRow::MediaTriplet {
                        anchor: a,
                        positive: p,
                        negative: n,
                    })
                    .collect(),
            ),
            reservation: None,
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
            reservation: None,
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
            reservation: None,
        }
    }

    /// Attach the eager collected read's pool reservation to this loader,
    /// moving ownership in: the bytes it reserved release only when this
    /// loader (or whichever half of a [`Self::split`] carries it) drops
    /// (#500 U2c c3c, P-R). `pub(crate)`: only `worker.rs`'s Resident
    /// construction site calls this — every `from_*` constructor above
    /// stays reservation-free by design.
    pub(crate) fn with_reservation(
        mut self,
        reservation: datafusion::execution::memory_pool::MemoryReservation,
    ) -> Self {
        self.reservation = Some(reservation);
        self
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
    ///
    /// The train/validation boundary itself is `split_index` — the SAME
    /// arithmetic a [`super::source::StreamedSet`] uses to derive its own
    /// `train_count` from a row COUNT alone (#500 U2c §10), so a resident
    /// loader's split and a streamed source's window never disagree about
    /// where the boundary falls for the same `(total, fraction)`.
    ///
    /// **The eager reservation moves to the TRAIN half** (#500 U2c c3c,
    /// P-R): `self`'s ENTIRE currently-held reservation is carved, by
    /// `MemoryReservation::split`, into a fresh reservation the returned
    /// train loader owns — so the pool accounting follows the loader that
    /// actually stays resident through every epoch, never the transient
    /// pre-split original or the validation half this run reads only
    /// occasionally.
    ///
    /// **Consumes `self` (#500 U2c closing round, A2/P-B4) and moves rows,
    /// never clones them.** `self`'s row `Vec` is truncated in place via
    /// `Vec::split_off` — the validation half's rows are MOVED out (no
    /// `TrainingRow` is ever cloned by this call; an earlier `&self`
    /// revision cloned BOTH halves via `.to_vec()` since it could not move
    /// out of a shared reference). Taking `self` by value also closes A2:
    /// `MemoryReservation::split` drains atomically, so `self`'s reservation
    /// would sit at size zero after a first call, and a second `split` on
    /// the SAME loader would previously hand the new "train" half a
    /// reservation carrying zero bytes silently — a correct-looking loader
    /// whose pool accounting had already gone stale. A second `split` on a
    /// moved loader is now a COMPILE error (the moved-value diagnostic
    /// [`Self::split`]'s own doctest below pins) rather than a silent
    /// runtime one.
    ///
    /// ```compile_fail,E0382
    /// use jammi_ai::fine_tune::data::TrainingDataLoader;
    ///
    /// let loader = TrainingDataLoader::from_rows(4);
    /// let (train, _val) = loader.split(0.25);
    /// // `loader` was moved into the call above; a second `split` on it
    /// // cannot compile — the exact shape A2 found reachable at runtime
    /// // when `split` took `&self`.
    /// let (_train2, _val2) = loader.split(0.25);
    /// # let _ = train;
    /// ```
    pub fn split(self, fraction: f64) -> (TrainingDataLoader, TrainingDataLoader) {
        let train_reservation = self.reservation.as_ref().map(|r| r.split(r.size()));
        match self.data {
            LoaderData::TextRows(mut rows) => {
                let train_count = split_index(rows.len(), fraction);
                let val_rows = rows.split_off(train_count);
                (
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::TextRows(rows),
                        reservation: train_reservation,
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::TextRows(val_rows),
                        reservation: None,
                    },
                )
            }
            LoaderData::Precomputed(mut batches) => {
                let train_count = split_index(batches.len(), fraction);
                let val_batches = batches.split_off(train_count);
                (
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Precomputed(batches),
                        reservation: train_reservation,
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Precomputed(val_batches),
                        reservation: None,
                    },
                )
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
                .map(|chunk| self.rows_to_text_chunk(chunk))
                .collect(),
            LoaderData::Precomputed(_) => Vec::new(),
        }
    }

    /// One [`TextChunk`] from a slice of this loader's own [`TrainingRow`]s —
    /// the single per-chunk row→chunk converter [`Self::text_chunks`] (every
    /// row, `batch_size` at a time) and [`Self::text_chunk_for_rank`] (one
    /// [`super::partition::PartitionSpec`]-selected slice at a time) both
    /// drive, so the two never risk decoding a chunk differently. `chunk` may
    /// be EMPTY — a zero-row rank (DESIGN.md §2) still produces a well-formed
    /// [`TextChunk`] with empty inner vectors, never a panic: every arm here
    /// is a plain `.map().collect()` over `chunk`, which is total on an empty
    /// slice.
    ///
    /// A `Graph` loader stores `Pairs`/`Triplet` rows, so it encodes through
    /// its underlying shape — the provenance variant carries no chunk shape
    /// of its own.
    fn rows_to_text_chunk(&self, chunk: &[TrainingRow]) -> TextChunk {
        match self.format.underlying() {
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
            UnderlyingFormat::MediaTriplet => TextChunk::MediaTriplet {
                anchors: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::MediaTriplet { anchor, .. } => anchor.clone(),
                        _ => Vec::new(),
                    })
                    .collect(),
                positives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::MediaTriplet { positive, .. } => positive.clone(),
                        _ => Vec::new(),
                    })
                    .collect(),
                negatives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::MediaTriplet { negative, .. } => negative.clone(),
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
        }
    }

    /// The [`TextChunk`] rank `spec.rank` of `spec.world` holds for global
    /// step `step`, over THIS loader's own row count as the train prefix
    /// (DESIGN.md §2, partition rule v1 — [`super::partition::PartitionSpec::
    /// rows_for_step`] computes the slice; this method decodes it through the
    /// SAME `Self::rows_to_text_chunk` converter [`Self::text_chunks`]
    /// uses). A rank whose slice is empty (a zero-row rank at the trailing
    /// global batch, K2) yields a well-formed [`TextChunk`] with empty inner
    /// vectors, never an out-of-bounds panic — `rows_for_step` never returns
    /// a range past `rows.len()`.
    ///
    /// Text-rows-backed loaders only (`Precomputed` has no row-level
    /// partition).
    pub fn text_chunk_for_rank(
        &self,
        spec: &super::partition::PartitionSpec,
        step: usize,
    ) -> Result<TextChunk> {
        match &self.data {
            LoaderData::TextRows(rows) => {
                let range = spec.rows_for_step(rows.len(), step);
                Ok(self.rows_to_text_chunk(&rows[range]))
            }
            LoaderData::Precomputed(_) => Err(JammiError::FineTune(
                "a precomputed loader has no row-level partition".into(),
            )),
        }
    }

    /// The detected training data format.
    pub fn format(&self) -> TrainingFormat {
        self.format
    }

    /// Every regression target in this loader, in row order — the whole-dataset
    /// view the trainer reduces into a fixed target scaler once before the
    /// loop (K3). `None` for any non-regression loader (no targets to
    /// standardise) and for the precomputed test path (which supplies
    /// head/target tensors directly, not text rows).
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
        let rows: &[TrainingRow] = match &self.data {
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

    /// A `Precomputed` loader has no row-level partition, so
    /// `text_chunk_for_rank` refuses it outright
    /// rather than attempting to slice pre-built tensor batches by row. Dies
    /// if the refusal is removed (the `LoaderData::Precomputed(_) => Err(...)`
    /// arm at `text_chunk_for_rank`'s match) — the call would then need to
    /// fall through to some OTHER arm, which does not compile for an empty
    /// `Vec<TrainingBatch>` fed in here without inventing a bogus row slice.
    #[test]
    fn precomputed_loader_refuses_text_chunk_for_rank() {
        use super::super::partition::{PartitionRule, PartitionSpec};
        let loader = TrainingDataLoader::from_precomputed(Vec::new());
        let spec = PartitionSpec::for_test(0, 1, 4, PartitionRule::BlockByGlobalBatch);
        match loader.text_chunk_for_rank(&spec, 0) {
            Err(e) => assert!(
                e.to_string().contains("no row-level partition"),
                "expected the typed 'no row-level partition' refusal, got: {e}"
            ),
            Ok(_) => panic!("a Precomputed loader must refuse text_chunk_for_rank"),
        }
    }

    /// Every [`TrainingFormat`] value the tag mapping must cover, in
    /// [`TRAINING_FORMAT_TAGS`] order. Hand-written, and ANCHORED to the
    /// compiler by [`enumeration_index`] below — a new variant fails to
    /// compile there until it is added here too, so the round-trip tests can
    /// never range over a stale subset.
    fn every_training_format() -> Vec<TrainingFormat> {
        vec![
            TrainingFormat::Contrastive,
            TrainingFormat::Pairs,
            TrainingFormat::Triplet,
            TrainingFormat::MediaTriplet,
            TrainingFormat::Classification { num_classes: 7 },
            TrainingFormat::Ner { num_labels: 5 },
            TrainingFormat::Regression,
            TrainingFormat::Graph {
                has_negatives: false,
            },
            TrainingFormat::Graph {
                has_negatives: true,
            },
        ]
    }

    /// The compiler anchor for [`every_training_format`]: an exhaustive match
    /// with no `_` arm mapping each variant to its position in that list.
    fn enumeration_index(format: TrainingFormat) -> usize {
        match format {
            TrainingFormat::Contrastive => 0,
            TrainingFormat::Pairs => 1,
            TrainingFormat::Triplet => 2,
            TrainingFormat::MediaTriplet => 3,
            TrainingFormat::Classification { .. } => 4,
            TrainingFormat::Ner { .. } => 5,
            TrainingFormat::Regression => 6,
            TrainingFormat::Graph {
                has_negatives: false,
            } => 7,
            TrainingFormat::Graph {
                has_negatives: true,
            } => 8,
        }
    }

    #[test]
    fn the_format_enumeration_is_the_whole_enum() {
        let all = every_training_format();
        for (i, format) in all.iter().enumerate() {
            assert_eq!(
                enumeration_index(*format),
                i,
                "{format:?} is out of position in every_training_format()"
            );
        }
        // Every index the anchor can return is occupied, so the list has no
        // hole a variant could hide in.
        let mut occupied: Vec<usize> = all.iter().map(|f| enumeration_index(*f)).collect();
        occupied.sort_unstable();
        assert_eq!(occupied, (0..all.len()).collect::<Vec<_>>());
    }

    /// The tag mapping is a bijection between the variants and
    /// [`TRAINING_FORMAT_TAGS`], and it round-trips through
    /// [`TrainingFormat::from_format_tag`].
    ///
    /// Injectivity is the load-bearing half: two variants sharing a tag are two
    /// different training sets colliding on one definition hash, which is
    /// exactly the failure the string-tag indirection risks.
    #[test]
    fn format_tags_are_a_bijection_and_round_trip() {
        let all = every_training_format();

        let tags: Vec<&str> = all.iter().map(|f| f.format_tag()).collect();
        assert_eq!(
            tags, TRAINING_FORMAT_TAGS,
            "the declared tag set and the variants' own tags must agree, in order"
        );

        let mut distinct = tags.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            tags.len(),
            "two formats share a tag, so two different training sets would              collide on one definition hash: {tags:?}"
        );

        for format in &all {
            let tag = format.format_tag();
            let recovered = TrainingFormat::from_format_tag(tag)
                .unwrap_or_else(|| panic!("tag {tag:?} is not decodable"));
            assert_eq!(recovered.format_tag(), tag);
        }
        for tag in TRAINING_FORMAT_TAGS {
            let recovered = TrainingFormat::from_format_tag(tag)
                .unwrap_or_else(|| panic!("declared tag {tag:?} is not decodable"));
            assert_eq!(&recovered.format_tag(), tag);
        }
    }

    /// The two data-derived parameters are dropped by design, and the one
    /// byte-affecting parameter is not.
    #[test]
    fn the_tag_drops_row_derived_parameters_and_keeps_the_shape_one() {
        assert_eq!(
            TrainingFormat::Classification { num_classes: 3 }.format_tag(),
            TrainingFormat::Classification { num_classes: 900 }.format_tag(),
            "num_classes is a function of the rows, not of the table's identity"
        );
        assert_eq!(
            TrainingFormat::Ner { num_labels: 3 }.format_tag(),
            TrainingFormat::Ner { num_labels: 900 }.format_tag(),
            "num_labels is a function of the rows, not of the table's identity"
        );
        assert_ne!(
            TrainingFormat::Graph {
                has_negatives: false
            }
            .format_tag(),
            TrainingFormat::Graph {
                has_negatives: true
            }
            .format_tag(),
            "a mined negative is a third projected column, so it is a different table"
        );
    }

    #[test]
    fn an_unknown_tag_is_refused_rather_than_approximated() {
        assert!(TrainingFormat::from_format_tag("graph").is_none());
        assert!(TrainingFormat::from_format_tag("Contrastive").is_none());
        assert!(TrainingFormat::from_format_tag("").is_none());
    }

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
        let (train, val) = loader.split(0.2);
        assert!(matches!(train.format(), TrainingFormat::Regression));
        assert!(matches!(val.format(), TrainingFormat::Regression));
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    /// Acceptance (b): for W ∈ {1, 2, 4}, the MULTISET of rows over all ranks
    /// at each global step equals the W=1 batch at that step, on a
    /// `train_count` (7) that is NOT a multiple of `W·B` for any tested W —
    /// asserting a zero-row rank actually occurs for W=2 and W=4 (DESIGN.md
    /// §2; PRESSURE round-2 design finding 6). RED at base: neither
    /// `PartitionSpec` nor `text_chunk_for_rank` exist there.
    ///
    /// Per-determinant table (reported alongside this test in the
    /// eval-verdict): the fixture is chosen so W=2 hits its zero-row rank at
    /// the LAST step (rank 1) and W=4 hits it at the FIRST step (rank 3) —
    /// two different positions in the epoch, not the same one twice.
    #[test]
    fn partition_rule_multiset_matches_the_w1_batch_with_zero_row_ranks() {
        use super::super::partition::{PartitionRule, PartitionSpec};

        let train_count = 7usize;
        let per_rank_batch = 3usize;
        let loader = TrainingDataLoader::from_pairs(
            (0..train_count)
                .map(|i| (format!("a{i}"), format!("p{i}")))
                .collect(),
        );

        fn anchors_of(chunk: &TextChunk) -> Vec<String> {
            match chunk {
                TextChunk::Pairs { anchors, .. } => anchors.clone(),
                other => panic!("expected a Pairs chunk, got a different TextChunk arm: {other:?}"),
            }
        }

        for &world in &[1usize, 2, 4] {
            let w1_ref = PartitionSpec::for_test(
                0,
                1,
                per_rank_batch * world,
                PartitionRule::BlockByGlobalBatch,
            );
            let mut zero_row_seen = false;
            let mut step = 0usize;
            loop {
                let w1_range = w1_ref.rows_for_step(train_count, step);
                if w1_range.is_empty() {
                    break;
                }
                let w1_chunk = loader.text_chunk_for_rank(&w1_ref, step).unwrap();
                let expected = anchors_of(&w1_chunk);

                let mut union = Vec::new();
                for rank in 0..world {
                    let spec = PartitionSpec::for_test(
                        rank,
                        world,
                        per_rank_batch,
                        PartitionRule::BlockByGlobalBatch,
                    );
                    let chunk = loader.text_chunk_for_rank(&spec, step).unwrap();
                    let rank_anchors = anchors_of(&chunk);
                    if rank_anchors.is_empty() {
                        zero_row_seen = true;
                    }
                    union.extend(rank_anchors);
                }
                assert_eq!(
                    union, expected,
                    "world={world} step={step}: the union over ranks must equal the W=1 batch"
                );
                step += 1;
            }
            if world > 1 {
                assert!(
                    zero_row_seen,
                    "world={world} on train_count={train_count}, batch={per_rank_batch} must hit \
                     a zero-row rank on this fixture, and none did"
                );
            }
        }
    }

    /// R-A for (b): shrinking a rank's batch to a size the fixture cannot
    /// possibly fill for a real chunk still returns a well-formed (empty)
    /// chunk rather than panicking, and a step past every rank's data is
    /// empty for every rank — the zero-row state is total, not a
    /// coincidence of the one fixture above.
    #[test]
    fn partition_rule_a_step_past_the_train_prefix_is_zero_rows_for_every_rank() {
        use super::super::partition::{PartitionRule, PartitionSpec};
        let train_count = 5usize;
        let loader = TrainingDataLoader::from_pairs(
            (0..train_count)
                .map(|i| (format!("a{i}"), format!("p{i}")))
                .collect(),
        );
        for world in [1usize, 2, 4] {
            for rank in 0..world {
                let spec =
                    PartitionSpec::for_test(rank, world, 3, PartitionRule::BlockByGlobalBatch);
                // Step 10 is far past any row this 5-row fixture could ever
                // reach at batch 3 for any tested world.
                let chunk = loader.text_chunk_for_rank(&spec, 10).unwrap();
                match &chunk {
                    TextChunk::Pairs { anchors, positives } => {
                        assert!(anchors.is_empty() && positives.is_empty());
                    }
                    _ => panic!("expected a Pairs chunk"),
                }
            }
        }
    }
}
